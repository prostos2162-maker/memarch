"""Злите ядро кроку Instinct з гомеостазом ЗА ОДИН ПРОХІД (OPT_HS_LAG=1) для TPU — Pallas.

Математика — рівно як гілка `kind in ("hs", "hs3") and cs is not None` у exp_opt._instinct_step_impl:

    mf  = m·β + g·(1−β)                                       (fp32)
    u1  = mf / (rms_ro(mf) + ε)                               ro — вісь 1 (hs: стовпці рядка; hs3: входи K)
    now = mean_co(u1²)                                        co — hs: вісь 0 → (1, C); hs3: вісь 2 → (E, K, 1)
    col = where(cs > 0, cs, now);   cs ← now
    u2  = u1 / (sqrt(col) + ε)
    u3  = u2 / (rms_ro(u2) + ε)
    m  ← SR(mf);   w ← SR(w·(1 − lr·wd) − u3·lr·eff)

Чому один прохід: норма другої осі (col) — з ПОПЕРЕДНЬОГО кроку (cs), а нова (now) лише накопичується попутно з того
самого u1 і віддається як cs_new. Байти: читає g, m, w (6 Б/вагу), пише m, w (4 Б/вагу) — 10 Б/вагу проти 14 у
двопрохідному pallas_instinct (прохід А ще раз читає g, m).

Контракт першого кроку / нулів у cs: де cs == 0 (≤ 0), торч бере свіжу норму now — її в один прохід не знати.
Обгортка сама (на пристрої, без синхронізації з хостом) перевіряє min(cs) > 0 і через lax.cond у тій самій
jit-функції: так — одразу один прохід з cs; ні — спершу прохід лише статистики (читає g, m, рахує now), далі
col = where(cs > 0, cs, now) і той самий однопрохідний крок з col. Це ТОЧНО як торч навіть для часткових нулів
(напр. стовпці, що досі не мали градієнта). Викликач нічого не перевіряє й не передає прапорця.

Стохастичне округлення — те саме лічильникове, що в pallas_instinct (_sr_pair: один хеш, молодші 16 біт — m,
старші — w; номер = плоский номер елемента; насіння — біти scalars[3]) — бітово ті самі значення, що й у
двопрохідному ядрі за однакових mf, w_new.

Перевірка без TPU: interpret=True — hyperop/check_hs1.py.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from . import pallas_instinct as pi
from .pallas_instinct import EPS, _rms, _sr_pair, _pick_block, _params, pltpu

_ALIASES = {} if pi._ALIASES == {} else {3: 0, 4: 1}     # (sc, cinv, g, m, w) — m і w на місці


def _smem_spec(interpret, ndim_grid):
    if pltpu is not None and not interpret:
        return pl.BlockSpec(memory_space=pltpu.SMEM)
    return pl.BlockSpec((4,), (lambda i: (0,)) if ndim_grid == 1 else (lambda e, j: (0,)))


# ------------------------------------------------------------------------------------------------- 2D (hs)
def _hs1_rows(sc_ref, cinv_ref, g_ref, m_ref, w_ref, m_out, w_out, colsq_ref, *, BR, C):
    """Плитка — BR цілих рядків: rms рядка, ділення на НОРМУ СТОВПЦЯ З МИНУЛОГО КРОКУ (cinv), rms рядка u2, запис
    m, w — і попутно Σ_рядків u1² по стовпцях (нова норма на наступний крок)."""
    beta, lr_eff, wdmul = sc_ref[0], sc_ref[1], sc_ref[2]
    seed = jax.lax.bitcast_convert_type(sc_ref[3], jnp.uint32)
    i = pl.program_id(0)

    @pl.when(i == 0)
    def _init():
        colsq_ref[...] = jnp.zeros_like(colsq_ref)

    mf = m_ref[...].astype(jnp.float32) * beta + g_ref[...].astype(jnp.float32) * (1 - beta)
    u1 = mf * (1.0 / (_rms(mf, 1) + EPS))
    colsq_ref[...] += jnp.sum(u1 * u1, axis=0, keepdims=True)
    u2 = u1 * cinv_ref[...]
    u3 = u2 / (_rms(u2, 1) + EPS)
    row = jax.lax.broadcasted_iota(jnp.uint32, (BR, C), 0) + (i * BR).astype(jnp.uint32)
    col = jax.lax.broadcasted_iota(jnp.uint32, (BR, C), 1)
    idx = row * jnp.uint32(C) + col
    w_new = w_ref[...].astype(jnp.float32) * wdmul - u3 * lr_eff
    m_out[...], w_out[...] = _sr_pair(mf, w_new, idx, seed)


def _stat_rows(sc_ref, g_ref, m_ref, colsq_ref):
    """Лише статистика (запасний шлях, коли в cs є нулі): Σ_рядків u1² по стовпцях."""
    beta = sc_ref[0]

    @pl.when(pl.program_id(0) == 0)
    def _init():
        colsq_ref[...] = jnp.zeros_like(colsq_ref)

    mf = m_ref[...].astype(jnp.float32) * beta + g_ref[...].astype(jnp.float32) * (1 - beta)
    u1 = mf * (1.0 / (_rms(mf, 1) + EPS))
    colsq_ref[...] += jnp.sum(u1 * u1, axis=0, keepdims=True)


def _block_rows(R, C, block_rows):
    return _pick_block(R, min(block_rows, max(8, pi.TILE_BYTES // (4 * C))))


def _now_2d(g, m, scalars, interpret, block_rows):
    R, C = g.shape
    BR = _block_rows(R, C, block_rows)
    row_spec = pl.BlockSpec((BR, C), lambda i: (i, 0))
    colsq = pl.pallas_call(
        _stat_rows,
        out_shape=jax.ShapeDtypeStruct((1, C), jnp.float32),
        grid=(R // BR,),
        in_specs=[_smem_spec(interpret, 1), row_spec, row_spec],
        out_specs=pl.BlockSpec((1, C), lambda i: (0, 0)),
        interpret=interpret,
        **_params(("arbitrary",), interpret),
    )(scalars, g, m)
    return colsq / R


def _step_2d(g, m, w, col, scalars, interpret, block_rows):
    R, C = g.shape
    BR = _block_rows(R, C, block_rows)
    row_spec = pl.BlockSpec((BR, C), lambda i: (i, 0))
    cinv = 1.0 / (jnp.sqrt(col) + EPS)
    m_new, w_new, colsq = pl.pallas_call(
        functools.partial(_hs1_rows, BR=BR, C=C),
        out_shape=(jax.ShapeDtypeStruct((R, C), jnp.bfloat16), jax.ShapeDtypeStruct((R, C), jnp.bfloat16),
                   jax.ShapeDtypeStruct((1, C), jnp.float32)),
        grid=(R // BR,),
        in_specs=[_smem_spec(interpret, 1), pl.BlockSpec((1, C), lambda i: (0, 0)), row_spec, row_spec, row_spec],
        out_specs=(row_spec, row_spec, pl.BlockSpec((1, C), lambda i: (0, 0))),
        input_output_aliases=_ALIASES,
        interpret=interpret,
        # "arbitrary": вихід colsq — накопичувач через усі плитки (на v6e одне ядро TC — втрат немає)
        **_params(("arbitrary",), interpret),
    )(scalars, cinv, g, m, w)
    return m_new, w_new, colsq / R


def instinct_hs1_2d(g, m, w, cs, scalars, *, interpret=False, block_rows=256):
    """Крок OPT_HS_LAG для 2D-ваги (R, C). cs: f32 (1, C) — середні u1² по рядках з минулого кроку (0 — ще немає).
    scalars: f32[4] = (β, lr·eff, 1−lr·wd, насіння як біти f32). → (m_new, w_new, cs_new), m/w bf16, cs_new f32."""
    cs = cs.astype(jnp.float32)
    col = jax.lax.cond(jnp.min(cs) > 0, lambda: cs,
                       lambda: jnp.where(cs > 0, cs, _now_2d(g, m, scalars, interpret, block_rows)))
    return _step_2d(g, m, w, col, scalars, interpret, block_rows)


# ------------------------------------------------------------------------------------------------- 3D (hs3)
def _hs1_cols(sc_ref, cinv_ref, g_ref, m_ref, w_ref, m_out, w_out, rowsq_ref, *, K, N, BN):
    """Експерти (E, K, N), плитка (1, K, BN): rms по входах K (ціла вісь у плитці), ділення на норму входу з минулого
    кроку (cinv: (1, K, 1)), rms по K для u2, запис m, w; попутно Σ_виходів u1² для кожного входу."""
    beta, lr_eff, wdmul = sc_ref[0], sc_ref[1], sc_ref[2]
    seed = jax.lax.bitcast_convert_type(sc_ref[3], jnp.uint32)
    e, j = pl.program_id(0), pl.program_id(1)

    @pl.when(j == 0)
    def _init():
        rowsq_ref[...] = jnp.zeros_like(rowsq_ref)

    mf = m_ref[...].astype(jnp.float32) * beta + g_ref[...].astype(jnp.float32) * (1 - beta)
    u1 = mf * (1.0 / (_rms(mf, 1) + EPS))
    sq = u1 * u1
    if N % BN:                                            # крайня неповна плитка (N не кратне BN): хвіст не рахуємо
        nn0 = jax.lax.broadcasted_iota(jnp.int32, (1, K, BN), 2) + j * BN
        sq = jnp.where(nn0 < N, sq, 0.0)
    rowsq_ref[...] += jnp.sum(sq, axis=2, keepdims=True)
    u2 = u1 * cinv_ref[...]
    u3 = u2 / (_rms(u2, 1) + EPS)
    kk = jax.lax.broadcasted_iota(jnp.uint32, (1, K, BN), 1)
    nn = jax.lax.broadcasted_iota(jnp.uint32, (1, K, BN), 2) + (j * BN).astype(jnp.uint32)
    idx = (e.astype(jnp.uint32) * jnp.uint32(K) + kk) * jnp.uint32(N) + nn
    w_new = w_ref[...].astype(jnp.float32) * wdmul - u3 * lr_eff
    m_out[...], w_out[...] = _sr_pair(mf, w_new, idx, seed)


def _stat_cols(sc_ref, g_ref, m_ref, rowsq_ref, *, K, N, BN):
    beta = sc_ref[0]

    @pl.when(pl.program_id(1) == 0)
    def _init():
        rowsq_ref[...] = jnp.zeros_like(rowsq_ref)

    mf = m_ref[...].astype(jnp.float32) * beta + g_ref[...].astype(jnp.float32) * (1 - beta)
    u1 = mf * (1.0 / (_rms(mf, 1) + EPS))
    sq = u1 * u1
    if N % BN:                                            # крайня неповна плитка: хвіст — сміття, у суму не бере участі
        nn = jax.lax.broadcasted_iota(jnp.int32, (1, K, BN), 2) + pl.program_id(1) * BN
        sq = jnp.where(nn < N, sq, 0.0)
    rowsq_ref[...] += jnp.sum(sq, axis=2, keepdims=True)


def _block_cols(K, N, block_cols):
    pref = min(block_cols, max(128, pi.TILE_BYTES // (4 * K)))
    bn = _pick_block(N, pref, mult=128)
    if bn == N and N > pref:                              # дільника, кратного 128, нема (N=5119 при MOE_K=3): без цього
        bn = max(128, pref // 128 * 128)                  # плитка = вся вісь N (50 МБ ×2 буфери) → переповнення VMEM;
    return bn                                             # тепер плитки 128 з неповною крайньою (grid = cdiv)


def _vmem_3d(K, BN):
    """Оцінка VMEM однопрохідного hs3 з доповненням смуг: блок (1, K, 1) f32 у VMEM займає (K, 128) — 512·K Б;
    cinv (вхід) і rowsq (вихід) — по 2 буфери. Плитки bf16 (g, m, w, m_out, w_out) × 2 буфери, ~8 тимчасових fp32
    (mf, u1, u2, u3, w_new, idx, хеш, ...). Для (4, 3840, 2560), BN=128: 7.9 + 9.8 + 15.7 ≈ 33 МБ — більше за 32 МБ."""
    return 4 * K * 128 * 4 + 5 * 2 * K * BN * 2 + 8 * K * BN * 4


def _params_vmem(sem, interpret, est):
    """Як pi._params, але межа VMEM — не менша за 1.3 × оцінку (стеля 96 МБ; фізично v6e має 128 МБ на ядро)."""
    if interpret or pltpu is None:
        return {}
    lim = int(min(96 << 20, max(pi.VMEM_LIMIT, est * 1.3)))
    return dict(compiler_params=pltpu.CompilerParams(dimension_semantics=sem, vmem_limit_bytes=lim))


def _now_3d(g, m, scalars, interpret, block_cols):
    E, K, N = g.shape
    BN = _block_cols(K, N, block_cols)
    tile = pl.BlockSpec((1, K, BN), lambda e, j: (e, 0, j))
    rowsq = pl.pallas_call(
        functools.partial(_stat_cols, K=K, N=N, BN=BN),
        out_shape=jax.ShapeDtypeStruct((E, K, 1), jnp.float32),
        grid=(E, pl.cdiv(N, BN)),
        in_specs=[_smem_spec(interpret, 2), tile, tile],
        out_specs=pl.BlockSpec((1, K, 1), lambda e, j: (e, 0, 0)),
        interpret=interpret,
        **_params_vmem(("parallel", "arbitrary"), interpret, _vmem_3d(K, BN) - 2 * 2 * K * BN * 2),
    )(scalars, g, m)
    return rowsq / N


def _step_3d(g, m, w, col, scalars, interpret, block_cols):
    E, K, N = g.shape
    BN = _block_cols(K, N, block_cols)
    tile = pl.BlockSpec((1, K, BN), lambda e, j: (e, 0, j))
    kspec = pl.BlockSpec((1, K, 1), lambda e, j: (e, 0, 0))
    cinv = 1.0 / (jnp.sqrt(col) + EPS)
    m_new, w_new, rowsq = pl.pallas_call(
        functools.partial(_hs1_cols, K=K, N=N, BN=BN),
        out_shape=(jax.ShapeDtypeStruct((E, K, N), jnp.bfloat16), jax.ShapeDtypeStruct((E, K, N), jnp.bfloat16),
                   jax.ShapeDtypeStruct((E, K, 1), jnp.float32)),
        grid=(E, pl.cdiv(N, BN)),
        in_specs=[_smem_spec(interpret, 2), kspec, tile, tile, tile],
        out_specs=(tile, tile, kspec),
        input_output_aliases=_ALIASES,
        interpret=interpret,
        **_params_vmem(("parallel", "arbitrary"), interpret, _vmem_3d(K, BN)),
    )(scalars, cinv, g, m, w)
    return m_new, w_new, rowsq / N


def instinct_hs1_3d(g, m, w, cs, scalars, *, interpret=False, block_cols=512):
    """Крок OPT_HS_LAG для експертів (E, K, N), kind "hs3". cs: f32 (E, K, 1). → (m_new, w_new, cs_new)."""
    cs = cs.astype(jnp.float32)
    col = jax.lax.cond(jnp.min(cs) > 0, lambda: cs,
                       lambda: jnp.where(cs > 0, cs, _now_3d(g, m, scalars, interpret, block_cols)))
    return _step_3d(g, m, w, col, scalars, interpret, block_cols)


# ------------------------------------------------------------------------------------------------- еталон
def reference(g, m, w, cs, scalars, kind):
    """Та сама математика звичайним jnp, без округлення: → (mf, w_new fp32, cs_new)."""
    beta, lr_eff, wdmul = scalars[0], scalars[1], scalars[2]
    ro, co = (1, 0) if kind == "hs" else (1, 2)
    mf = m.astype(jnp.float32) * beta + g.astype(jnp.float32) * (1 - beta)
    u1 = mf / (_rms(mf, ro) + EPS)
    now = jnp.mean(u1 * u1, axis=co, keepdims=True)
    col = jnp.where(cs > 0, cs, now)
    u2 = u1 / (jnp.sqrt(col) + EPS)
    u3 = u2 / (_rms(u2, ro) + EPS)
    return mf, w.astype(jnp.float32) * wdmul - u3 * lr_eff, now


# ------------------------------------------------------------------------------------------------- міст у PyTorch
_OPS = {}


def torch_op(kind):
    """Операція PyTorch (torch.tpu.pallas.jax_op) memarch::instinct_hs1_{kind}, kind "hs" або "hs3":
    (g, m, w, cs, scalars) → (m_new, w_new, cs_new); g, m, w bf16, cs f32, scalars f32[4]. Створювати ДО
    torch.compile (як pallas_instinct.torch_op)."""
    if kind in _OPS:
        return _OPS[kind]
    import torch  # noqa: F401
    import torch_tpu  # noqa: F401
    from torch.tpu import pallas as tpp
    fn = instinct_hs1_2d if kind == "hs" else instinct_hs1_3d

    def f(g: jax.Array, m: jax.Array, w: jax.Array, cs: jax.Array,
          sc: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return fn(g, m, w, cs, sc)
    f.__name__ = f"instinct_hs1_{kind}"
    if pi.TORCH_DONATE:
        _OPS[kind] = tpp.jax_op(f"memarch::instinct_hs1_{kind}_d", f, donate_argnums=(1, 2, 3))
    elif pi.TORCH_ALIAS:
        _OPS[kind] = tpp.jax_op(f"memarch::instinct_hs1_{kind}_a", f, input_output_aliases={1: 0, 2: 1, 3: 2})
    else:
        _OPS[kind] = tpp.jax_op(f"memarch::instinct_hs1_{kind}", f)
    return _OPS[kind]
