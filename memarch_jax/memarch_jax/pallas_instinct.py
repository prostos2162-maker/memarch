"""Злите ядро кроку Instinct з гомеостазом (norm="hs") для TPU — Pallas, два проходи по пам'яті замість ~4–5.

Математика — рівно як exp_opt._instinct_step_impl для kind "hs" (2D, (R, C)) і "hs3" (експерти (E, in, out)), з
bf16-вагою без fp32-копії (BF16_ALL) і стохастичним округленням запису:

    mf  = m·β + g·(1−β)                                   (fp32)
    u1  = mf / (rms_A(mf) + ε)                            A — «перша» вісь (hs: по стовпцях рядка; hs3: по входах)
    u2  = u1 / (rms_B(u1) + ε)                            B — друга вісь (hs: по рядках; hs3: по виходах)
    u3  = u2 / (rms_A(u2) + ε)
    m  ← SR(mf);   w ← SR(w·(1 − lr·wd) − u3·lr·eff)

Чому два проходи — мінімум: rms_B потребує ВСІХ рядків, але суму квадратів u1 по осі B можна накопичувати вже в
першому проході (рядок цілком у плитці → rms_A відома одразу). Другий прохід ділить на обидві норми, перераховує
rms_A(u2) всередині плитки й пише m, w. Байти: прохід А читає g, m (4 Б/вагу), прохід Б читає g, m, w і пише m, w
(10 Б/вагу) — 14 Б/вагу проти ~10 у теоретичному ідеалі.

Стохастичне округлення — лічильникове: випадкові 16 біт = хеш(насіння кроку, номер елемента) (lowbias32), без
окремого генератора й без зайвих байтів пам'яті; у середньому оновлення зберігається, як і в exp_opt._sr_bf16.

Перевірка без TPU: interpret=True (той самий код ядра на процесорі) — hyperop/check_pallas_instinct.py.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

try:
    from jax.experimental.pallas import tpu as pltpu
except Exception:                                   # на ПК без TPU-розширень — лише режим інтерпретації
    pltpu = None

EPS = 1e-8
import os as _os
_ALIASES = {} if _os.environ.get("PALLAS_NO_ALIAS") == "1" else {3: 0, 4: 1}
_ALIASES_B = {} if _ALIASES == {} else {4: 0, 5: 1}     # прохід Б: (sc, cinv, rinv, g, m, w) — m і w на місці


def _hash_u32(x):
    """lowbias32: дешевий якісний хеш 32-бітного лічильника (лише цілі множення, зсуви, xor — VPU)."""
    x = x ^ (x >> 16)
    x = x * jnp.uint32(0x7FEB352D)
    x = x ^ (x >> 15)
    x = x * jnp.uint32(0x846CA68B)
    x = x ^ (x >> 16)
    return x


def _sr_to_bf16(x, idx_u32, seed_u32):
    """fp32 → bf16 зі стохастичним округленням: +випадкові молодші 16 біт до подання fp32, далі відкинути їх."""
    r = _hash_u32(idx_u32 ^ (seed_u32 * jnp.uint32(0x9E3779B9))) & jnp.uint32(0xFFFF)
    return _sr_with(x, r)


def _sr_with(x, r16):
    xi = jax.lax.bitcast_convert_type(x, jnp.uint32)
    xi = (xi + r16) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(xi, jnp.float32).astype(jnp.bfloat16)


def _sr_pair(m32, w32, idx_u32, seed_u32):
    """Обидва записи (інерція й вага) з ОДНОГО хешу: молодші 16 біт — для m, старші — для w (у lowbias32 вони
    незалежні) — удвічі менше цілочисельної роботи на VPU, ніж два хеші."""
    h = _hash_u32(idx_u32 ^ (seed_u32 * jnp.uint32(0x9E3779B9)))
    return _sr_with(m32, h & jnp.uint32(0xFFFF)), _sr_with(w32, h >> 16)


def _rms(x, axis):
    return jnp.sqrt(jnp.mean(x * x, axis=axis, keepdims=True))


# ------------------------------------------------------------------------------------------------- 2D (hs)
def _pass_a_rows(sc_ref, g_ref, m_ref, colsq_ref, rinv_ref):
    """Прохід А для (R, C), плитка — BR цілих рядків: rms рядка (віддається й проходу Б — не рахувати вдруге) і
    накопичення Σ_рядків u1² по стовпцях."""
    beta = sc_ref[0]

    @pl.when(pl.program_id(0) == 0)
    def _init():
        colsq_ref[...] = jnp.zeros_like(colsq_ref)

    mf = m_ref[...].astype(jnp.float32) * beta + g_ref[...].astype(jnp.float32) * (1 - beta)
    r1 = 1.0 / (_rms(mf, 1) + EPS)
    rinv_ref[...] = r1
    u1 = mf * r1
    colsq_ref[...] += jnp.sum(u1 * u1, axis=0, keepdims=True)


def _pass_b_rows(sc_ref, cinv_ref, rinv_ref, g_ref, m_ref, w_ref, m_out, w_out, *, BR, C):
    beta, lr_eff, wdmul = sc_ref[0], sc_ref[1], sc_ref[2]
    seed = jax.lax.bitcast_convert_type(sc_ref[3], jnp.uint32)
    mf = m_ref[...].astype(jnp.float32) * beta + g_ref[...].astype(jnp.float32) * (1 - beta)
    u1 = mf * rinv_ref[...]                         # 1/rms рядка — з проходу А
    u2 = u1 * cinv_ref[...]
    u3 = u2 / (_rms(u2, 1) + EPS)
    i = pl.program_id(0)
    row = jax.lax.broadcasted_iota(jnp.uint32, (BR, C), 0) + (i * BR).astype(jnp.uint32)
    col = jax.lax.broadcasted_iota(jnp.uint32, (BR, C), 1)
    idx = row * jnp.uint32(C) + col
    w_new = w_ref[...].astype(jnp.float32) * wdmul - u3 * lr_eff
    m_out[...], w_out[...] = _sr_pair(mf, w_new, idx, seed)


# бюджет на одну fp32-плитку: у ядрі ~6 таких проміжних (mf, u1, u2, u3, нова вага) плюс подвійна буферизація
# входів/виходів — 1 МБ на плитку тримає все в межах VMEM з запасом (на [2560, 5120] плитка 256 рядків — VMEM OOM)
TILE_BYTES = int(_os.environ.get("PALLAS_TILE_BYTES", 1 << 20))
VMEM_LIMIT = int(_os.environ.get("PALLAS_VMEM_LIMIT", 32 << 20))


def _pick_block(n, pref, mult=8):
    """Найбільший дільник n, не більший за pref і кратний mult (якщо можна) — плитки без доповнень."""
    for b in range(min(pref, n), 0, -1):
        if n % b == 0 and (b % mult == 0 or b == n):
            return b
    return n


def instinct_hs_2d(g, m, w, scalars, *, interpret=False, block_rows=256):
    """Крок для 2D-ваги (R, C). scalars: f32[4] = (β, lr·eff, 1−lr·wd, насіння як біти f32). → (m_new, w_new)."""
    R, C = g.shape
    BR = _pick_block(R, min(block_rows, max(8, TILE_BYTES // (4 * C))))
    nb = R // BR
    smem = dict(memory_space=pltpu.SMEM) if (pltpu is not None and not interpret) else {}
    sc_spec = pl.BlockSpec(**smem) if smem else pl.BlockSpec((4,), lambda i: (0,))
    row_spec = pl.BlockSpec((BR, C), lambda i: (i, 0))
    colsq, rinv = pl.pallas_call(
        _pass_a_rows,
        out_shape=(jax.ShapeDtypeStruct((1, C), jnp.float32), jax.ShapeDtypeStruct((R, 1), jnp.float32)),
        grid=(nb,),
        in_specs=[sc_spec, row_spec, row_spec],
        out_specs=(pl.BlockSpec((1, C), lambda i: (0, 0)), pl.BlockSpec((BR, 1), lambda i: (i, 0))),
        interpret=interpret,
        **_params(("arbitrary",), interpret),
    )(scalars, g, m)
    cinv = 1.0 / (jnp.sqrt(colsq / R) + EPS)
    m_new, w_new = pl.pallas_call(
        functools.partial(_pass_b_rows, BR=BR, C=C),
        out_shape=(jax.ShapeDtypeStruct((R, C), jnp.bfloat16), jax.ShapeDtypeStruct((R, C), jnp.bfloat16)),
        grid=(nb,),
        in_specs=[sc_spec, pl.BlockSpec((1, C), lambda i: (0, 0)), pl.BlockSpec((BR, 1), lambda i: (i, 0)),
                  row_spec, row_spec, row_spec],
        out_specs=(row_spec, row_spec),
        input_output_aliases=_ALIASES_B,              # m, w — на місці (PALLAS_NO_ALIAS=1 — для перевірки)
        interpret=interpret,
        **_params(("parallel",), interpret),
    )(scalars, cinv, rinv, g, m, w)
    return m_new, w_new


# ------------------------------------------------------------------------------------------------- 3D (hs3)
def _pass_a_cols(sc_ref, g_ref, m_ref, rowsq_ref, cinv1_ref):
    """Прохід А для експертів (E, K, N) — плитка (1, K, BN): rms по входах K (вісь 1, ціла в плитці) і накопичення
    Σ_виходів u1² для кожного входу (друга вісь нормування hs3 — вихідна, N)."""
    beta = sc_ref[0]

    @pl.when(pl.program_id(1) == 0)
    def _init():
        rowsq_ref[...] = jnp.zeros_like(rowsq_ref)

    mf = m_ref[...].astype(jnp.float32) * beta + g_ref[...].astype(jnp.float32) * (1 - beta)
    r1 = 1.0 / (_rms(mf, 1) + EPS)                  # (1, 1, BN): 1/rms по входах для кожного виходу — і проходу Б
    cinv1_ref[...] = r1
    u1 = mf * r1
    rowsq_ref[...] += jnp.sum(u1 * u1, axis=2, keepdims=True)


def _pass_b_cols(sc_ref, rinv_ref, cinv1_ref, g_ref, m_ref, w_ref, m_out, w_out, *, K, N, BN):
    beta, lr_eff, wdmul = sc_ref[0], sc_ref[1], sc_ref[2]
    seed = jax.lax.bitcast_convert_type(sc_ref[3], jnp.uint32)
    mf = m_ref[...].astype(jnp.float32) * beta + g_ref[...].astype(jnp.float32) * (1 - beta)
    u1 = mf * cinv1_ref[...]                        # з проходу А
    u2 = u1 * rinv_ref[...]
    u3 = u2 / (_rms(u2, 1) + EPS)
    e, j = pl.program_id(0), pl.program_id(1)
    kk = jax.lax.broadcasted_iota(jnp.uint32, (1, K, BN), 1)
    nn = jax.lax.broadcasted_iota(jnp.uint32, (1, K, BN), 2) + (j * BN).astype(jnp.uint32)
    idx = (e.astype(jnp.uint32) * jnp.uint32(K) + kk) * jnp.uint32(N) + nn
    w_new = w_ref[...].astype(jnp.float32) * wdmul - u3 * lr_eff
    m_out[...], w_out[...] = _sr_pair(mf, w_new, idx, seed)


def instinct_hs3(g, m, w, scalars, *, interpret=False, block_cols=512):
    """Крок для ваг експертів (E, K, N) з kind "hs3" (нормування: K → N → K). → (m_new, w_new)."""
    E, K, N = g.shape
    BN = _pick_block(N, min(block_cols, max(128, TILE_BYTES // (4 * K))), mult=128)   # остання вісь — кратна 128
    nb = N // BN
    smem = dict(memory_space=pltpu.SMEM) if (pltpu is not None and not interpret) else {}
    sc_spec = pl.BlockSpec(**smem) if smem else pl.BlockSpec((4,), lambda e, j: (0,))
    tile = pl.BlockSpec((1, K, BN), lambda e, j: (e, 0, j))
    rowsq, cinv1 = pl.pallas_call(
        _pass_a_cols,
        out_shape=(jax.ShapeDtypeStruct((E, K, 1), jnp.float32), jax.ShapeDtypeStruct((E, 1, N), jnp.float32)),
        grid=(E, nb),
        in_specs=[sc_spec, tile, tile],
        out_specs=(pl.BlockSpec((1, K, 1), lambda e, j: (e, 0, 0)), pl.BlockSpec((1, 1, BN), lambda e, j: (e, 0, j))),
        interpret=interpret,
        **_params(("parallel", "arbitrary"), interpret),
    )(scalars, g, m)
    rinv = 1.0 / (jnp.sqrt(rowsq / N) + EPS)
    m_new, w_new = pl.pallas_call(
        functools.partial(_pass_b_cols, K=K, N=N, BN=BN),
        out_shape=(jax.ShapeDtypeStruct((E, K, N), jnp.bfloat16), jax.ShapeDtypeStruct((E, K, N), jnp.bfloat16)),
        grid=(E, nb),
        in_specs=[sc_spec, pl.BlockSpec((1, K, 1), lambda e, j: (e, 0, 0)), pl.BlockSpec((1, 1, BN), lambda e, j: (e, 0, j)),
                  tile, tile, tile],
        out_specs=(tile, tile),
        input_output_aliases=_ALIASES_B,
        interpret=interpret,
        **_params(("parallel", "parallel"), interpret),
    )(scalars, rinv, cinv1, g, m, w)
    return m_new, w_new


# ------------------------------------------------------------------------------------------------- vocab (таблиці слів)
def _vocab_rows(sc_ref, g_ref, m_ref, w_ref, m_out, w_out, *, BR, C):
    """kind "vocab": нормування лише по рядку → один прохід. Рядок без градієнта (слово не трапилось у кроці) не
    рухається оновленням, але інерція згасає й wd діє — як exp_opt._instinct_step_impl."""
    beta, lr_eff, wdmul = sc_ref[0], sc_ref[1], sc_ref[2]
    seed = jax.lax.bitcast_convert_type(sc_ref[3], jnp.uint32)
    g = g_ref[...].astype(jnp.float32)
    mf = m_ref[...].astype(jnp.float32) * beta + g * (1 - beta)
    hit = jnp.sum(jnp.abs(g), axis=1, keepdims=True) > 0
    upd = jnp.where(hit, mf / (_rms(mf, 1) + EPS), 0.0)
    i = pl.program_id(0)
    row = jax.lax.broadcasted_iota(jnp.uint32, (BR, C), 0) + (i * BR).astype(jnp.uint32)
    col = jax.lax.broadcasted_iota(jnp.uint32, (BR, C), 1)
    idx = row * jnp.uint32(C) + col
    w_new = w_ref[...].astype(jnp.float32) * wdmul - upd * lr_eff
    m_out[...], w_out[...] = _sr_pair(mf, w_new, idx, seed)


def instinct_vocab(g, m, w, scalars, *, interpret=False, block_rows=256):
    """Крок для таблиці слів (V, C), kind "vocab": один прохід. → (m_new, w_new)."""
    R, C = g.shape
    BR = _pick_block(R, min(block_rows, max(8, TILE_BYTES // (4 * C))))
    smem = dict(memory_space=pltpu.SMEM) if (pltpu is not None and not interpret) else {}
    sc_spec = pl.BlockSpec(**smem) if smem else pl.BlockSpec((4,), lambda i: (0,))
    row_spec = pl.BlockSpec((BR, C), lambda i: (i, 0))
    return pl.pallas_call(
        functools.partial(_vocab_rows, BR=BR, C=C),
        out_shape=(jax.ShapeDtypeStruct((R, C), jnp.bfloat16), jax.ShapeDtypeStruct((R, C), jnp.bfloat16)),
        grid=(R // BR,),
        in_specs=[sc_spec, row_spec, row_spec, row_spec],
        out_specs=(row_spec, row_spec),
        input_output_aliases={2: 0, 3: 1} if _ALIASES else {},
        interpret=interpret,
        **_params(("parallel",), interpret),
    )(scalars, g, m, w)


def _params(sem, interpret):
    if interpret or pltpu is None:
        return {}
    return dict(compiler_params=pltpu.CompilerParams(dimension_semantics=sem, vmem_limit_bytes=VMEM_LIMIT))


# ------------------------------------------------------------------------------------------------- еталон
def reference(g, m, w, scalars, kind, sr_bits=None):
    """Та сама математика звичайним jnp (як exp_opt._instinct_step_impl), округлення — до найближчого (sr_bits=None)."""
    beta, lr_eff, wdmul = scalars[0], scalars[1], scalars[2]
    mf = m.astype(jnp.float32) * beta + g.astype(jnp.float32) * (1 - beta)
    if kind == "vocab":
        hit = jnp.sum(jnp.abs(g.astype(jnp.float32)), axis=1, keepdims=True) > 0
        upd = jnp.where(hit, mf / (_rms(mf, 1) + EPS), 0.0)
        return mf, w.astype(jnp.float32) * wdmul - upd * lr_eff
    a, b = (1, 0) if kind == "hs" else (1, 2)
    u1 = mf / (_rms(mf, a) + EPS)
    u2 = u1 / (_rms(u1, b) + EPS)
    u3 = u2 / (_rms(u2, a) + EPS)
    w_new = w.astype(jnp.float32) * wdmul - u3 * lr_eff
    return mf, w_new                                   # fp32 до округлення — для порівняння


# ------------------------------------------------------------------------------------------------- міст у PyTorch
_OPS = {}


def torch_op(kind):
    """Операція PyTorch (torch.tpu.pallas.jax_op) для kind "hs" (2D) або "hs3" (експерти): (g, m, w, scalars) →
    (m_new, w_new), усе bf16, scalars f32[4]. Лише на TPU (torch_tpu); під torch.compile — як звичайна операція."""
    if kind in _OPS:
        return _OPS[kind]
    import torch  # noqa: F401
    import torch_tpu  # noqa: F401
    from torch.tpu import pallas as tpp
    if kind == "hs":
        def f(g: jax.Array, m: jax.Array, w: jax.Array, sc: jax.Array) -> tuple[jax.Array, jax.Array]:
            return instinct_hs_2d(g, m, w, sc)
    elif kind == "vocab":
        def f(g: jax.Array, m: jax.Array, w: jax.Array, sc: jax.Array) -> tuple[jax.Array, jax.Array]:
            return instinct_vocab(g, m, w, sc)
    else:
        def f(g: jax.Array, m: jax.Array, w: jax.Array, sc: jax.Array) -> tuple[jax.Array, jax.Array]:
            return instinct_hs3(g, m, w, sc)
    f.__name__ = f"instinct_{kind}"
    if TORCH_DONATE:
        # PALLAS_TORCH_DONATE=1: m (арг. 1) і w (арг. 2) віддаються мосту (donate_argnums) — XLA може писати виходи в
        # їхні буфери без захисних копій; що тоді з самими torch-тензорами m, w — перевіряє check_pallas_torch
        _OPS[kind] = tpp.jax_op(f"memarch::instinct_{kind}_d", f, donate_argnums=(1, 2))
    elif TORCH_ALIAS:
        # PALLAS_TORCH_ALIAS=1: m (арг. 1) і w (арг. 2) — вихідні буфери самого ядра (без захисних копій моста);
        # чи міст тоді змінює вхідні тензори на місці — перевіряє hyperop/check_pallas_torch.py на TPU
        _OPS[kind] = tpp.jax_op(f"memarch::instinct_{kind}_a", f, input_output_aliases={1: 0, 2: 1})
    else:
        _OPS[kind] = tpp.jax_op(f"memarch::instinct_{kind}", f)
    return _OPS[kind]


TORCH_ALIAS = _os.environ.get("PALLAS_TORCH_ALIAS") == "1"
TORCH_DONATE = _os.environ.get("PALLAS_TORCH_DONATE") == "1"
# PALLAS_TORCH_INPLACE=1 — лише якщо check_pallas_torch довів, що з donate міст змінює m, w на місці (тоді copy_ зайвий)
TORCH_INPLACE = _os.environ.get("PALLAS_TORCH_INPLACE") == "1"
