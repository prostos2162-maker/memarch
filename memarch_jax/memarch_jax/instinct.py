"""Оптимізатор Instinct (семантика exp_opt.InstinctOptB) — чисті функції над pytree параметрів.

Порядок у кроці (як у torch): prepare(loss, sit) → update(усі ваги) → finish().
    st = init(params, d_sit, cfg, spec)
    params, st = step(params, grads, st, lr=..., situation=sit, loss=loss, spec=spec, cfg=cfg)

Види кроку (kind) задаються ЯВНО за іменами ваг (design §3, не за shape[0] ≥ 10000):
    vocab — emb/head/l_head1/book (нормування лише по рядку, рядки без градієнта не рухаються);
    hs3   — експерти (E, in, out): гомеостаз по входах і виходах;
    hs    — інші 2D (router, memory/reread): гомеостаз рядків і стовпців (OPT_HS_LAG: норма стовпця з минулого кроку);
    hsS   — light-лінійні (у torch порізані між чипами): свіжа норма стовпця щокроку, без lag;
    row   — ≥4D; tensor — 0D/1D.

Стохастичне округлення (SR) — лічильникове (хеш lowbias32 номера елемента й насіння кроку), як у Pallas-ядрах:
детерміноване й відтворюване; cfg.sr=False — округлення до найближчого (для тестів проти torch з OPT_SR_OFF=1).
Pallas-ядра (pallas_hs1 / pallas_instinct з the torch code root) — коли вага й інерція bf16 і SR увімкнено; на CPU —
interpret=True. Інакше — чистий jnp (та сама математика, у тому числі той самий хеш SR → побітово ті самі біти).
"""
import dataclasses
import math
import os
import sys

import jax
import jax.numpy as jnp

EPS = 1e-8
_TORCH_DIR = os.environ.get("MEMARCH_TORCH_DIR", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))


def _kernels():
    """Ліниво імпортуємо Pallas-ядра Стефана (вони лежать поруч із torch-кодом, але самі — чистий JAX)."""
    # копії лежать у пакеті (на TPU-VM немає torch-коду); оригінал — лише запасний шлях
    try:
        from . import pallas_instinct, pallas_hs1
    except ImportError:
        if _TORCH_DIR not in sys.path:
            sys.path.insert(0, _TORCH_DIR)
        import pallas_instinct, pallas_hs1
    return pallas_instinct, pallas_hs1


# ------------------------------------------------------------------------------------------------- налаштування
@dataclasses.dataclass(frozen=True)
class InstinctCfg:
    lr_peak: float = 6e-4
    weight_decay: float = 0.1
    cards: int = 4
    beta: float = 0.9
    new_card: float = 0.5
    ema: float = 0.98
    good: float = 0.05
    bad: float = 0.2
    top: float = 0.1
    relax: float = 0.01
    scale: float = 0.45
    m_bf16: bool = True
    rel_every: int = 8
    rel_cond: bool = True            # rel-норми під lax.cond лише на кроках want_rel (False — щокроку + where)
    hs_lag: bool = True              # OPT_HS_LAG
    sr: bool = True                  # стохастичне округлення bf16-записів (False — до найближчого)
    use_pallas: bool = True          # злиті ядра для hs/hs3/vocab (лише bf16 вага + bf16 інерція + SR)
    interpret: bool | None = None    # None — авто: interpret на не-TPU
    vocab_lr_mul: float = 1.0        # VOCAB_LR_MUL


# ------------------------------------------------------------------------------------------------- розклад lr
def lr_at(step, lr, steps, *, warmup=200, sched="wsd", wsd_frac=0.2, floor=0.1):
    """fastloop.lr_at (step 0-based, python float64). sched="wsd" — LR_SCHED=wsd, інакше косинус до 0.1."""
    up = min(1, (step + 1) / warmup)
    if sched == "wsd":
        d0 = (1 - wsd_frac) * steps
        return lr * up * (1.0 if step < d0 else 1 - (1 - floor) * min(1, (step - d0) / (steps - d0)))
    return lr * up * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, step / steps))))


def lr_scalars(lr, cfg):
    """(lr_t, lr_v, lr_ratio) як f32: рахуємо в float64 на хості, далі f32 (як _lr_sync/_ratio_sync у torch)."""
    if isinstance(lr, (int, float)):
        return (jnp.float32(lr * cfg.scale), jnp.float32(lr * cfg.scale * cfg.vocab_lr_mul),
                jnp.float32(lr / cfg.lr_peak))
    lr = jnp.asarray(lr, jnp.float32)
    return lr * cfg.scale, lr * (cfg.scale * cfg.vocab_lr_mul), lr / cfg.lr_peak


# ------------------------------------------------------------------------------------------------- spec / шляхи
def path_str(path):
    out = []
    for k in path:
        out.append(str(getattr(k, "key", getattr(k, "idx", getattr(k, "name", k)))))
    return ".".join(out)


_VOCAB_NAMES = ("emb", "head", "l_head1", "book")


def default_kind(name, shape, sharded=None):
    """Вид кроку за іменем (design §3). sharded(name) → True для light-лінійних (hsS)."""
    parts = name.split(".")
    nd = len(shape)
    if nd == 2 and any(p in _VOCAB_NAMES for p in parts):
        return "vocab"
    if nd == 3:
        return "hs3"
    if nd == 2:
        return "hsS" if (sharded is not None and sharded(name)) else "hs"
    if nd >= 4:
        return "row"
    return "tensor"


def make_spec(params, kinds=None, order=None, sharded=None, has_grad=None, stacked=("heavy",), frozen=()):
    """Статичний опис кроку: кортеж (name, kind, wd_flag, vocab, has_grad, leaf, li) у порядку `order` (порядок torch
    plist: від нього залежать номери rel/gain і послідовність насінь SR). kinds — словник name → kind (перевизначення).
    stacked — верхні ключі, чиї листки складені по осі шару (heavy: (L, ...)): кожен шар — окрема вага, як у torch
    (ім'я "heavy.{li}.{k}", вид за формою БЕЗ осі шару, свій rel/gain, свої норми). leaf/li — листок і номер шару
    (li=None — звичайний листок).
    frozen — імена torch-параметрів БЕЗ листка в params (lock_w, ind_a: requires_grad=True у torch, але grad None) —
    вони займають місце в plist torch, тож входять у P (rel/gain/quantile «винних» і номери насінь); тут — заглушки
    з has_grad=False (leaf=None): не оновлюються, rel лишається 0, як у torch."""
    flat = {path_str(p): x for p, x in jax.tree_util.tree_flatten_with_path(params)[0]}
    ent = {}
    for n, x in flat.items():
        top, _, rest = n.partition(".")
        if top in (stacked or ()) and rest:
            for li in range(x.shape[0]):
                ent[f"{top}.{li}.{rest}"] = (n, li, x.shape[1:])
        else:
            ent[n] = (n, None, x.shape)
    for n in frozen:
        ent[n] = (None, None, ())
    names = list(order) if order is not None else list(ent)
    assert set(names) == set(ent), "order має містити всі листки"
    spec = []
    for n in names:
        leaf, li, shp = ent[n]
        if leaf is None:                                         # заглушка frozen
            spec.append((n, "tensor", False, False, False, None, None))
            continue
        kind = (kinds or {}).get(n) or default_kind(n, shp, sharded)
        vocab = kind == "vocab"
        wd = len(shp) >= 2 and not vocab
        hg = True if has_grad is None else bool(has_grad.get(n, has_grad.get(leaf, True)))
        spec.append((n, kind, wd, vocab, hg, leaf, li))
    return tuple(spec)


# ------------------------------------------------------------------------------------------------- стан
def init(params, d_sit, cfg, spec):
    """Порожній стан (як InstinctOptB.__init__ + ліниві keys/cs). m і cs — словники name → масив."""
    flat = {path_str(p): x for p, x in jax.tree_util.tree_flatten_with_path(params)[0]}
    P, K = len(spec), cfg.cards
    mdt = jnp.bfloat16 if cfg.m_bf16 else jnp.float32
    m, cs = {}, {}
    for n, kind, _, _, _, leaf, li in spec:
        if leaf is None:                                         # заглушка frozen (make_spec(frozen=...))
            continue
        x = flat[leaf]
        if leaf not in m:
            m[leaf] = jnp.zeros(x.shape, mdt)
        pre = x.shape[:1] if li is not None else ()             # складений листок: cs теж зі віссю шару
        shp = x.shape[1:] if li is not None else x.shape
        if cfg.hs_lag and kind == "hs":
            cs[leaf] = jnp.zeros(pre + (1, shp[1]), jnp.float32)
        elif cfg.hs_lag and kind == "hs3":
            cs[leaf] = jnp.zeros(pre + (shp[0], shp[1], 1), jnp.float32)
    return dict(
        gain=jnp.ones((K, P), jnp.float32), keys=jnp.zeros((K, d_sit), jnp.float32),
        used=jnp.zeros((K,), bool), expect=jnp.zeros((K,), jnp.float32), seen=jnp.zeros((K,), jnp.float32),
        rel=jnp.zeros((P,), jnp.float32), seed_t=jnp.zeros((), jnp.int32), n_step=jnp.zeros((), jnp.int32),
        k=jnp.zeros((), jnp.int32), s=jnp.zeros((), jnp.float32), stake=jnp.zeros((), jnp.float32),
        g=jnp.ones((P,), jnp.float32), oh=jnp.zeros((K,), jnp.float32), m=m, cs=cs)


# ------------------------------------------------------------------------------------------------- prepare / finish
def _normalize(x):
    return x / jnp.maximum(jnp.sqrt(jnp.sum(x * x)), 1e-12)


def prepare(st, loss, situation, lr_ratio, cfg):
    """InstinctOptB.prepare (_prepare_impl): картка ситуації, очікування, несподіванка s, ставка, підсилювачі g."""
    keys, used, expect, seen, gain = st["keys"], st["used"], st["expect"], st["seen"], st["gain"]
    K = keys.shape[0]
    sit = _normalize(jnp.asarray(situation, jnp.float32).reshape(-1))
    any_used = jnp.any(used)
    sims = jnp.where(used, keys @ sit, -2.0)
    best = jnp.argmax(sims).astype(jnp.int32)                # перший індекс при рівних — як torch.max
    best_sim = sims[best]
    free = ~used
    first_free = jnp.argmax(free.astype(jnp.int32)).astype(jnp.int32)
    newer = (best_sim < cfg.new_card) & jnp.any(free)
    k = jnp.where(~any_used, 0, jnp.where(newer, first_free, best)).astype(jnp.int32)
    oh = jax.nn.one_hot(k, K, dtype=jnp.float32)
    was = jnp.sum(oh * used) > 0
    key_k = oh @ keys
    new_key = jnp.where(was, _normalize(0.95 * key_k + 0.05 * sit), sit)
    keys = keys + oh[:, None] * (new_key - key_k)[None]
    used = used | (oh > 0)
    loss = jnp.asarray(loss, jnp.float32)
    e = jnp.where(jnp.sum(oh * seen) > 0, jnp.sum(oh * expect), loss)
    s = (e - loss) / jnp.maximum(e, 1e-6)
    expect = expect + oh * (cfg.ema * e + (1 - cfg.ema) * loss - expect)
    seen = seen + oh * (1 - seen)
    stake = jnp.minimum(4 * jnp.abs(s) + 0.5 * lr_ratio, 1.0)
    g = oh @ gain
    return dict(st, keys=keys, used=used, expect=expect, seen=seen, s=s, stake=stake, k=k, g=g, oh=oh)


def finish(st, cfg):
    """InstinctOptB.finish (_finish_impl): n_step += 1; підсилювачі картки k вчаться за наслідком кроку."""
    s, gk, rel = st["s"], st["g"], st["rel"]
    culprit = (rel >= jnp.quantile(rel, 1 - cfg.top)).astype(jnp.float32)
    gk = gk + jnp.where(s < 0, culprit * cfg.bad * jnp.minimum(-s * 10, 1), 0.0)
    gk = gk + jnp.where(s > 0, cfg.good * jnp.minimum(s * 10, 1) * (1 - gk), 0.0)
    gk = gk + cfg.relax * (1 - gk)
    gk = jnp.clip(gk, 0.5, 3.0)
    gain = st["gain"] + st["oh"][:, None] * (gk / jnp.mean(gk) - st["gain"])
    return dict(st, gain=gain, n_step=st["n_step"] + 1)


# ------------------------------------------------------------------------------------------------- округлення
def _hash_u32(x):
    x = x ^ (x >> 16)
    x = x * jnp.uint32(0x7FEB352D)
    x = x ^ (x >> 15)
    x = x * jnp.uint32(0x846CA68B)
    x = x ^ (x >> 16)
    return x


def _sr_with(x, r16):
    xi = jax.lax.bitcast_convert_type(x, jnp.uint32)
    return jax.lax.bitcast_convert_type((xi + r16) & jnp.uint32(0xFFFF0000), jnp.float32).astype(jnp.bfloat16)


def _store(m32, w32, m_dtype, w_dtype, seed_u32, sr):
    """Запис інерції й ваги: bf16 — SR (той самий хеш, що в Pallas: молодші 16 біт → m, старші → w) або до
    найближчого; fp32 — без змін."""
    need = sr and (m_dtype == jnp.bfloat16 or w_dtype == jnp.bfloat16)
    if need:
        idx = jnp.arange(m32.size, dtype=jnp.uint32).reshape(m32.shape)
        h = _hash_u32(idx ^ (seed_u32 * jnp.uint32(0x9E3779B9)))
    m = (_sr_with(m32, h & jnp.uint32(0xFFFF)) if (need and m_dtype == jnp.bfloat16) else m32.astype(m_dtype))
    w = (_sr_with(w32, h >> 16) if (need and w_dtype == jnp.bfloat16) else w32.astype(w_dtype))
    return m, w


def _rms(x, axis):
    return jnp.sqrt(jnp.mean(x * x, axis=axis, keepdims=True))


# ------------------------------------------------------------------------------------------------- крок однієї ваги (jnp)
def _upd_jnp(kind, mf, grad, cs, axis_name=None, rows_total=None):
    """Напрям кроку (математика _instinct_step_impl). → (upd, cs_new або None)."""
    if kind == "vocab":
        hit = jnp.sum(jnp.abs(grad), axis=1, keepdims=True) > 0
        return jnp.where(hit, mf / (_rms(mf, 1) + EPS), 0.0), None
    if kind in ("hs", "hs3"):
        ro, co = (1, 0) if kind == "hs" else (1, 2)
        upd = mf / (_rms(mf, ro) + EPS)
        now = jnp.mean(upd * upd, axis=co, keepdims=True)
        col = now if cs is None else jnp.where(cs > 0, cs, now)
        upd = upd / (jnp.sqrt(col) + EPS)
        upd = upd / (_rms(upd, ro) + EPS)
        return upd, (None if cs is None else now)
    if kind == "hsS":
        # light-лінійні: свіжа норма стовпця (сума по всіх чипах / усього рядків); на W=1 — середнє по рядках
        upd = mf / (_rms(mf, 1) + EPS)
        col = jnp.sum(upd * upd, axis=0, keepdims=True)
        if axis_name is not None:
            col = jax.lax.psum(col, axis_name)
        col = col / (rows_total or mf.shape[0])
        upd = upd / (jnp.sqrt(col) + EPS)
        return upd / (_rms(upd, 1) + EPS), None
    if kind == "row":
        return mf / (jnp.sqrt(jnp.mean((mf * mf).reshape(mf.shape[0], -1), axis=1)).reshape(
            (-1,) + (1,) * (mf.ndim - 1)) + EPS), None
    return mf / (jnp.sqrt(jnp.mean(mf * mf)) + EPS), None


def _seed_bits(seed_t):
    """seed = int32(seed_t·1000003) (у torch int64 з обрізанням до int32 → ті самі молодші 32 біти), як u32."""
    return seed_t.astype(jnp.uint32) * jnp.uint32(1000003)


def _interp(cfg):
    if cfg.interpret is not None:
        return cfg.interpret
    return jax.default_backend() != "tpu"


def _pallas_call(kfn, args, sc, it, shard, leaf, li):
    """Виклик Pallas-ядра кроку. shard=(mesh, {leaf: PartitionSpec}) — для W>1 (dist.make_step).
    Pallas/Mosaic custom call GSPMD не розбиває: на шардованій вазі (vocab-рядки, EP-експерти) XLA зібрав би ПОВНИЙ
    тензор на кожен чип (24B → OOM) і рахував би його 8 разів. Тому шардовані листки йдуть через shard_map:
    ядро бачить свій шматок. Це точно, бо всі ядра локальні по шардованій осі (vocab — нормування рядка,
    hs3 — по кожному експерту окремо). Насіння SR змішуємо з номером чипа, щоб шматки не мали однакового шуму
    (індекс у хеші локальний). Репліковані листки — як є (кожен чип рахує те саме)."""
    spec = None
    if shard is not None:
        mesh, specs = shard
        spec = specs.get(leaf)
        if spec is not None and all(a is None for a in spec):
            spec = None
    from jax.sharding import PartitionSpec as P
    if spec is None and shard is not None:
        # репліковані листки на W>1: Mosaic-виклик GSPMD не розбиває навіть реплікований («Mosaic kernels cannot be
        # automatically partitioned») — теж через shard_map, усі входи/виходи P() (кожен чип рахує те саме, SR без зсуву)
        n_out = 3 if len(args) == 4 else 2
        def local_rep(*a):
            *xs, s = a
            return tuple(kfn(*xs, s, interpret=it))
        rep = tuple(P() for _ in range(len(args) + 1))
        try:
            sm = jax.shard_map(local_rep, mesh=shard[0], in_specs=rep, out_specs=tuple(P() for _ in range(n_out)),
                               check_vma=False)
        except AttributeError:
            from jax.experimental.shard_map import shard_map as _sm
            sm = _sm(local_rep, mesh=shard[0], in_specs=rep, out_specs=tuple(P() for _ in range(n_out)), check_rep=False)
        return sm(*args, sc)
    if spec is None:
        return tuple(kfn(*args, sc, interpret=it))
    parts = tuple(spec)
    if li is not None:                                         # шар складеного листка: перша вісь знята
        parts = parts[1:]
    ax = [a for a in parts if a is not None]
    assert len(ax) == 1, (leaf, spec)
    axis, dim = ax[0], parts.index(ax[0])
    def spec_for(x):
        return P(*([None] * dim + [axis] + [None] * (x.ndim - dim - 1)))
    in_specs = tuple(spec_for(a) for a in args) + (P(),)
    n_out = 3 if len(args) == 4 else 2
    out_specs = tuple(spec_for(args[j]) for j in ((1, 2, 3) if n_out == 3 else (1, 2)))

    def local(*a):
        *xs, s = a
        r = jax.lax.axis_index(axis).astype(jnp.uint32)
        sb = jax.lax.bitcast_convert_type(s[3], jnp.uint32)
        sb = _hash_u32(sb ^ ((r + jnp.uint32(1)) * jnp.uint32(0x85EBCA6B)))
        s = s.at[3].set(jax.lax.bitcast_convert_type(sb, jnp.float32))
        return tuple(kfn(*xs, s, interpret=it))
    try:
        sm = jax.shard_map(local, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
    except AttributeError:
        from jax.experimental.shard_map import shard_map as _sm
        sm = _sm(local, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)
    return sm(*args, sc)


def _layer_run(active, j):
    """Чи починається з active[j] блок складених шарів: K різних листків × L шарів (li = 0..L−1, li-major,
    кожен шар — ті самі листки в тому самому порядку й видах). → (L, K) або None (тоді — по одній вазі)."""
    e0 = active[j]
    if e0[6] != 0:
        return None
    K = 0
    while j + K < len(active) and active[j + K][6] == 0:
        K += 1
    leaves = [e[5] for e in active[j:j + K]]
    if len(set(leaves)) != K:
        return None
    L = 1
    while True:
        nxt = active[j + L * K:j + (L + 1) * K]
        if len(nxt) != K or any(e[6] != L or e[5] != lf or e[1:4] != b[1:4] for e, lf, b in zip(nxt, leaves, active[j:j + K])):
            break
        L += 1
    return (L, K) if L > 1 else None


def update(params, grads, st, lr_t, lr_v, cfg, spec, axis_name=None, rows_total=None, shard=None):
    """Крок усіх ваг (update_all) у фіксованому порядку spec. Ваги без градієнта (has_grad=False або grad None) —
    пропускаються повністю (без згасання m, без wd, rel не пишеться). → (params, st)."""
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    pflat = {path_str(p): x for p, x in leaves}
    gflat = {path_str(p): x for p, x in jax.tree_util.tree_flatten_with_path(grads, is_leaf=lambda v: v is None)[0]}
    m, cs = dict(st["m"]), dict(st["cs"])
    rel, seed_t, g_all, stake = st["rel"], st["seed_t"], st["g"], st["stake"]
    want_rel = (st["n_step"] % cfg.rel_every) == 0
    seed_base = _seed_bits(seed_t)                          # для generic-видів (у torch там свій randint)
    new_p = dict(pflat)
    beta = jnp.float32(cfg.beta)

    def one(i, kind, wdf, vocab, leaf, li, p, mi, csi, grad, rel, seed_t):
        """Крок однієї ваги (або одного шару складеного листка). i може бути traced (цикл по шарах).
        → (m_new, w_new, cs_new або None, rel, seed_t)."""
        lr = lr_v if vocab else lr_t
        wd = cfg.weight_decay / cfg.scale if (wdf and cfg.weight_decay) else 0.0
        eff = 1 + (g_all[i] - 1) * stake
        lr_eff = (lr * eff).astype(jnp.float32)
        wdmul = (1 - lr * wd).astype(jnp.float32) if wd else jnp.float32(1.0)
        rel_fn = lambda: jnp.linalg.norm(grad.astype(jnp.float32)) / (jnp.linalg.norm(p.astype(jnp.float32)) + EPS)
        if cfg.rel_cond:      # норми лише на кроках want_rel (звіт prof_b4_all п.7, ≈8–9 ms); значення ті самі
            rel = rel.at[i].set(jax.lax.cond(want_rel, rel_fn, lambda: rel[i].astype(jnp.float32)).astype(rel.dtype))
        else:                 # старий шлях: норми щокроку + where
            rel = rel.at[i].set(jnp.where(want_rel, rel_fn(), rel[i]))
        pallas_ok = (cfg.use_pallas and cfg.sr and p.dtype == jnp.bfloat16 and mi.dtype == jnp.bfloat16
                     and kind in ("hs", "hs3", "vocab"))
        if kind in ("hs", "hs3", "vocab") and cfg.sr and p.dtype == jnp.bfloat16 and mi.dtype == jnp.bfloat16:
            seed_t = seed_t + 1                              # лічильник крокує на кожну вагу «ядерного» шляху
            seed = _seed_bits(seed_t)
        else:                                            # (0x5BD1E995·(i+1)) mod 2³² — у uint32 (i може бути traced)
            seed = _hash_u32(seed_base ^ (jnp.uint32(0x5BD1E995) * (jnp.asarray(i).astype(jnp.uint32) + jnp.uint32(1))))
        cs_new = None
        if pallas_ok:
            pi, ph = _kernels()
            sc = jnp.stack([beta, lr_eff, wdmul, jax.lax.bitcast_convert_type(seed, jnp.float32)])
            gb, it = grad.astype(jnp.bfloat16), _interp(cfg)
            if kind == "vocab":
                kfn, args = pi.instinct_vocab, (gb, mi, p)
            elif csi is not None:
                kfn = ph.instinct_hs1_2d if kind == "hs" else ph.instinct_hs1_3d
                args = (gb, mi, p, csi)
            else:
                kfn = pi.instinct_hs_2d if kind == "hs" else pi.instinct_hs3
                args = (gb, mi, p)
            outs = _pallas_call(kfn, args, sc, it, shard, leaf, li)
            m_new, w_new = outs[0], outs[1]
            if len(outs) > 2:
                cs_new = outs[2]
        else:
            gf = grad.astype(jnp.float32)
            mf = mi.astype(jnp.float32) * beta + gf * jnp.float32(1 - cfg.beta)   # (1−β) у float64, як скаляр torch
            upd, cs_new = _upd_jnp(kind, mf, gf, csi, axis_name, rows_total)
            w32 = p.astype(jnp.float32)
            if wd:
                w32 = w32 * wdmul
            w32 = w32 - upd * lr_eff
            m_new, w_new = _store(mf, w32, mi.dtype, p.dtype, seed, cfg.sr)
        return m_new, w_new, cs_new, rel, seed_t

    traced = any(isinstance(x, jax.core.Tracer) for x in pflat.values())
    active = [e for e in spec if e[4] and gflat.get(e[5]) is not None]
    j = 0
    while j < len(active):
        n, kind, wdf, vocab, hg, leaf, li = active[j]
        i = spec.index(active[j])
        # цикл по шарах — лише під jit (traced): там він побітово = окремим шарам (перевірено jit-варіантом
        # test_stacked_heavy); поза jit (eager, тести) — старий розгорнутий шлях: eager-операції по одній
        # і скомпільоване тіло циклу можуть різнитися на 1 ulp (порядок редукцій норм).
        run = _layer_run(active, j) if (li is not None and traced) else None
        if run is not None:                                  # номери в spec мають іти з однаковим кроком по шарах
            L_, K_ = run
            ix = [[spec.index(active[j + l * K_ + k]) for k in range(K_)] for l in range(L_)]
            if any(ix[l][k] - ix[0][k] != l * (ix[1][0] - ix[0][0]) for l in range(L_) for k in range(K_)):
                run = None
        if run is not None:
            # Складені шари (heavy: L×K ваг підряд у порядку li-major) — fori_loop по шарах: стан (p, m, cs) складених
            # листків — carry, запис шару — dynamic_update_slice у той самий буфер (in-place, без копій усього
            # (L, ...) тензора на кожен шар, як було з розгорнутим .at[li].set). Порядок ваг/насінь/rel — той самий.
            L, K = run
            blk = active[j:j + K]
            i0s = [spec.index(e) for e in blk]
            stride = spec.index(active[j + K]) - i0s[0] if L > 1 else 0
            lv = [e[5] for e in blk]
            cs_keys = tuple(lf for lf in lv if cs.get(lf) is not None)

            def body(l, c, blk=blk, i0s=i0s, stride=stride, lv=lv, cs_keys=cs_keys):
                P_, M_, C_, rel_, sd_ = c
                P_, M_, C_ = dict(P_), dict(M_), dict(C_)
                for (n_, kind_, wdf_, vocab_, _, leaf_, _), i0 in zip(blk, i0s):
                    pl = jax.lax.dynamic_index_in_dim(P_[leaf_], l, 0, keepdims=False)
                    ml = jax.lax.dynamic_index_in_dim(M_[leaf_], l, 0, keepdims=False)
                    gl = jax.lax.dynamic_index_in_dim(gflat[leaf_], l, 0, keepdims=False)
                    cl = jax.lax.dynamic_index_in_dim(C_[leaf_], l, 0, keepdims=False) if leaf_ in C_ else None
                    m_new, w_new, cs_new, rel_, sd_ = one(i0 + l * stride, kind_, wdf_, vocab_, leaf_, 0, pl, ml, cl,
                                                          gl, rel_, sd_)
                    M_[leaf_] = jax.lax.dynamic_update_index_in_dim(M_[leaf_], m_new.astype(M_[leaf_].dtype), l, 0)
                    P_[leaf_] = jax.lax.dynamic_update_index_in_dim(P_[leaf_], w_new.astype(P_[leaf_].dtype), l, 0)
                    if cs_new is not None and leaf_ in C_:
                        C_[leaf_] = jax.lax.dynamic_update_index_in_dim(C_[leaf_], cs_new.astype(C_[leaf_].dtype), l, 0)
                return P_, M_, C_, rel_, sd_
            c0 = ({k: new_p[k] for k in lv}, {k: m[k] for k in lv}, {k: cs[k] for k in cs_keys}, rel, seed_t)
            P_, M_, C_, rel, seed_t = jax.lax.fori_loop(0, L, body, c0)
            new_p.update(P_); m.update(M_); cs.update(C_)
            j += L * K
            continue
        p, mi, csi = new_p[leaf], m[leaf], cs.get(leaf)
        grad = gflat[leaf]
        if li is not None:                                   # один шар складеного листка
            p, mi, grad = p[li], mi[li], grad[li]
            csi = None if csi is None else csi[li]
        m_new, w_new, cs_new, rel, seed_t = one(i, kind, wdf, vocab, leaf, li, p, mi, csi, grad, rel, seed_t)
        if cs_new is not None:
            cs[leaf] = cs_new if li is None else cs[leaf].at[li].set(cs_new)
        if li is None:
            m[leaf], new_p[leaf] = m_new, w_new
        else:
            m[leaf], new_p[leaf] = m[leaf].at[li].set(m_new), new_p[leaf].at[li].set(w_new)
        j += 1
    params = jax.tree_util.tree_unflatten(treedef, [new_p[path_str(p)] for p, _ in leaves])
    return params, dict(st, m=m, cs=cs, rel=rel, seed_t=seed_t)


# ------------------------------------------------------------------------------------------------- увесь крок
def step(params, grads, st, *, lr, situation, loss, spec, cfg, axis_name=None, rows_total=None, shard=None):
    """(params, grads, opt_state, lr, situation, loss) → (params, opt_state): prepare → update → finish.
    lr — python float (рахується lr_at на хості, float64 → f32) або traced скаляр."""
    lr_t, lr_v, ratio = lr_scalars(lr, cfg)
    st = prepare(st, loss, situation, ratio, cfg)
    params, st = update(params, grads, st, lr_t, lr_v, cfg, spec, axis_name, rows_total, shard)
    return params, finish(st, cfg)

