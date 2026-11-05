"""Повна модель FullLM2 (Fv2moeK40h14d2560) на JAX, один чіп (W=1): forward + loss + init.

Порядок (full2.py:1547-1814, train_loss 1953-1995, fastloop loss_fn):
  emb + INDUCT_IN → light_a → +memory reads → light_b → lh=l_norm → здивування (без grad) → якорі/tau
  → heavy MoE ×L (lax.scan) → scatter_add → final → +reread (REREAD_SEP/ANCH) → hn=norm
  loss = L_main(SHORT_V, з сумішшю INDUCT_MIX)/(B·T) + 0.01·mean(aux) + 0.3·light-head CE + read_cost·cost.

params — pytree (dict) з іменами як у torch; consts — статичні (python/numpy), передаються замиканням:
  consts = {'sc': ce.short_consts(...), 'lock_bucket': np, 'lock_w': (3,7), 'ind_a': (2,)}  (lock_w/ind_a заморожені).
prep — хостова підготовка induct.induct_raw (cand, on, bk, src)."""
import dataclasses

import numpy as np
import jax
import jax.numpy as jnp

from . import ce, induct, anchors, moe, memory, reread, hoop
from .attn import local_attention
from .light import flex_block, init_block, post_attn, pre_attn
from .prims import rmsnorm
from .config import mix_w_init

F32 = jnp.float32


# ---------------------------------------------------------------- конфіги підмодулів

def mem_cfg(cfg):
    return memory.MemCfg(d=cfg.d, Fn=cfg.mem_fresh, Dn=cfg.mem_dim, Dmax=cfg.mem_dim_max, nf=2, df=64,
                         dc=cfg.d // 4, W=cfg.mem_write, K=cfg.mem_focus, seg=cfg.seg, cdt=cfg.cdt, eb=cfg.mem_eb)


def rr_cfg(cfg):
    return reread.RereadCfg(df=64, k=4, R=4, seg=cfg.seg, frac=cfg.reread_anch, cdt=cfg.cdt, eb=cfg.mem_eb)


def moe_cfg(cfg):
    # обруч: усі A=budget·T місць зайняті (top-k без порожніх) → fill = 1; місткість рахується від B·A
    fill = 1.0 if getattr(cfg, "hoop", False) else cfg.frac / cfg.cap
    return moe.MoECfg(E=cfg.E, k=cfg.moe_k, group=cfg.moe_group, cap_align=cfg.cap_align, fill=fill,
                      fast=getattr(cfg, "moe_fast", True), a2a_chunks=getattr(cfg, "moe_a2a_chunks", 1))


def moe_h2(cfg):
    """Ширина експерта relu2 (BankMoE): hid = 2·d·2//k, h2 = 3·hid//2."""
    return 3 * (2 * cfg.d * 2 // cfg.moe_k) // 2


# ---------------------------------------------------------------- сталі

def build_consts(cfg, counts, lock_trust):
    """Сталі моделі з частот слів (numpy) і LOCK_TRUST (3,7)."""
    lt = np.asarray(lock_trust, np.float32)
    return dict(sc=ce.short_consts(counts, cfg.V, cfg.short_v, cfg.rare_cap),
                lock_bucket=induct.lock_bucket(counts, cfg.V),
                lock_w=jnp.asarray(lt).astype(cfg.cdt), ind_a=jnp.ones((2,), cfg.cdt))


def make_prep(x, cfg, consts):
    """Хостова підготовка (CPU_PREP): induct_raw → dict jnp-масивів."""
    cand, on, bk, src = induct.induct_raw(np.asarray(x), cfg.V, consts["lock_bucket"])
    return dict(cand=jnp.asarray(cand, jnp.int32), on=jnp.asarray(on), bk=jnp.asarray(bk, jnp.int32),
                src=jnp.asarray(src, jnp.int32))


# ---------------------------------------------------------------- init (ті самі розподіли, що в torch)

def _lin(k, out, inp):
    b = 1.0 / np.sqrt(inp)
    return jax.random.uniform(k, (out, inp), F32, -b, b)


def init_params(key, cfg, lock_trust=None):
    """Нові ваги (новий seed, розподіли torch): Embedding N(0,1), Linear U(±1/√in) (і зсув), RMSNorm 1,
    head[V-1]=0, експерти N·d^-½ / N·h2^-½·½, offset N·0.02, dim_weight −0.5, ind_in_g −2, mix_w з LOCK_TRUST.
    Усі float-листки у cfg.cdt (BF16_ALL)."""
    d, V, E, L = cfg.d, cfg.V, cfg.E, cfg.heavy
    ks = iter(jax.random.split(key, 64))
    nk = lambda: next(ks)
    h2 = moe_h2(cfg)
    head = jax.random.uniform(nk(), (V, d), F32, -1 / np.sqrt(d), 1 / np.sqrt(d)).at[V - 1].set(0)
    blocks = lambda n: [init_block(nk(), cfg) for _ in range(n)]
    dc, df = d // 4, 64
    ub = lambda k, n, inp: jax.random.uniform(k, (n,), F32, -1 / np.sqrt(inp), 1 / np.sqrt(inp))
    mem = {"norm_w.weight": jnp.ones(d), "face.weight": _lin(nk(), df, d), "content.weight": _lin(nk(), d, d),
           "importance.weight": _lin(nk(), 1, d), "importance.bias": ub(nk(), 1, d),
           "gate.weight": _lin(nk(), 1, 2 * d), "gate.bias": ub(nk(), 1, 2 * d),
           "down.weight": _lin(nk(), dc, d), "up.weight": _lin(nk(), d, dc), "norm_r.weight": jnp.ones(d),
           "norm_c.weight": jnp.ones(d), "query.weight": _lin(nk(), df, d), "value.weight": _lin(nk(), d, d),
           "out.weight": _lin(nk(), d, d), "dim_weight": jnp.asarray(-0.5, F32),
           "focus_gate.weight": _lin(nk(), 1, 2 * d), "focus_gate.bias": ub(nk(), 1, 2 * d)}
    rr = {"norm.weight": jnp.ones(d), "query.weight": _lin(nk(), df, d),
          "offset": jax.random.normal(nk(), (4, 128), F32) * 0.02,
          "halt_v.weight": _lin(nk(), 1, 128), "halt_v.bias": ub(nk(), 1, 128), "halt_h.weight": _lin(nk(), 1, d),
          "recog_r.weight": _lin(nk(), df, 128), "recog_c.weight": _lin(nk(), df, d), "out.weight": _lin(nk(), d, 128)}
    heavy = {"norm": jnp.ones((L, d)),
             "router": jax.random.uniform(nk(), (L, E, d), F32, -1 / np.sqrt(d), 1 / np.sqrt(d)),
             "up": jax.random.normal(nk(), (L, E, d, h2), F32) * d ** -0.5,
             "down": jax.random.normal(nk(), (L, E, h2, d), F32) * h2 ** -0.5 * 0.5}
    lt = np.ones((3, 7), np.float32) if lock_trust is None else np.asarray(lock_trust, np.float32)
    p = dict(emb=jax.random.normal(nk(), (V, d), F32), book=jax.random.normal(nk(), (V, 128), F32),
             light_a=blocks(cfg.light_a), light_b=blocks(cfg.light_b), final=blocks(cfg.final),
             memory=mem, reread=rr, l_norm=jnp.ones(d), l_head0=_lin(nk(), 128, d), l_head1=_lin(nk(), V, 128),
             heavy=heavy, norm=jnp.ones(d), head=head, ind_in_g=jnp.full((3,), -2.0, F32),
             mix_w=jnp.asarray(mix_w_init(lt)))
    if getattr(cfg, "hoop", False):
        p["hoop"] = hoop.init(nk(), d)
    return jax.tree_util.tree_map(lambda a: a.astype(cfg.cdt), p)


# ---------------------------------------------------------------- remat (jax.checkpoint по блоках)

# Мітки checkpoint_name: moe.py (агент MoE) — виходи обох a2a і dispatch; light.py/attn.py — легкі блоки.
MOE_SAVE_NAMES = ("moe_a2a_in", "moe_a2a_out", "moe_disp")
LIGHT_SAVE_NAMES = ("light_o", "light_up", "attn_out")


class LightSplit:
    """Режим легкого блоку «увага поза checkpoint»: pre_attn і post_attn — окремі jax.checkpoint (policy),
    а увага (splash, свій custom_vjp) — без remat, тож її залишки (q,k,v,out,lse) зберігаються і splash fwd
    не перераховується в bwd. Математика та сама (ті самі функції, лише інша межа checkpoint)."""

    def __init__(self, policy):
        self.policy = policy


def _policies():
    cp = jax.checkpoint_policies
    moe_save = cp.save_only_these_names(*MOE_SAVE_NAMES)
    return {"nothing": cp.nothing_saveable, "dots": cp.dots_saveable,
            "dots_nobatch": cp.dots_with_no_batch_dims_saveable,
            "moe_save": moe_save,
            # підмножини MOE_SAVE_NAMES — менше збережених залишків heavy (для B=5/6)
            "moe_save_out": cp.save_only_these_names("moe_a2a_out"),
            "moe_save_a2a": cp.save_only_these_names("moe_a2a_in", "moe_a2a_out"),
            "moe_save_disp": cp.save_only_these_names("moe_disp", "moe_a2a_out"),
            "light_save_noattn": LightSplit(cp.save_only_these_names("light_o", "light_up")),
            "moe_save_dots": cp.save_from_both_policies(moe_save, cp.dots_with_no_batch_dims_saveable),
            "light_save": LightSplit(cp.save_only_these_names(*LIGHT_SAVE_NAMES)),
            "light_save_dots": LightSplit(cp.save_from_both_policies(
                cp.save_only_these_names(*LIGHT_SAVE_NAMES), cp.dots_with_no_batch_dims_saveable))}


# назва → (light, heavy, memory); кожне: False (без checkpoint) | True (нічого не зберігати) | назва policy
REMAT_POLICIES = {
    "none": (False, False, False),
    "heavy": (False, True, True),            # MoE-шари (scan) і сегменти пам'яті; легкі — без remat
    "all": (True, True, True),               # усе перераховується в bwd
    "dots_saveable": ("dots", "dots", "dots"),            # зберігаємо результати матмулів
    # design §4: light — dots без batch-вимірів, heavy/mem — нічого. Увага: splash (custom call) і решта
    # легкого блоку все одно перераховуються, тому на TPU ≈ "all" (824 ms) — див. decisions.md.
    "dots_nobatch": ("dots_nobatch", True, True),
    "dots_nobatch_all": ("dots_nobatch", "dots_nobatch", "dots_nobatch"),   # policy доходить і до heavy/mem
    # звіт prof_b4_all п.1+п.5: heavy зберігає виходи a2a і dispatch; легкі — splash поза remat + light_*;
    "moe_save": ("light_save", "moe_save", True),
    "moe_save_dots": ("light_save_dots", "moe_save_dots", True),           # + dots без batch-вимірів
    "moe_save_heavy": (False, "moe_save", True),
    # економніші варіанти moe_save (менше temps → більший B): heavy зберігає лише частину назв
    "moe_save_out": ("light_save", "moe_save_out", True),
    "moe_save_a2a": ("light_save", "moe_save_a2a", True),
    "moe_save_disp": ("light_save", "moe_save_disp", True),
    "moe_save_lite": ("light_save_noattn", "moe_save", True),   # легкі не зберігають attn_out          # як "heavy", але heavy зберігає a2a/dispatch
}


def remat_parts(remat):
    """remat: bool (стара поведінка: True → heavy+memory) | назва з REMAT_POLICIES | (light, heavy, mem).
    → (light, heavy, mem), кожне False | True | policy-функція | LightSplit (лише light).
    Невідома назва частини — помилка (раніше кортеж з bench_tpu мовчки перетворював її на False)."""
    if remat is False or remat is None:
        return False, False, False
    if remat is True:
        return False, True, True
    parts = REMAT_POLICIES[remat] if isinstance(remat, str) else tuple(remat)
    pol = _policies()
    out = tuple(pol[x] if isinstance(x, str) else x for x in parts)
    if isinstance(out[1], LightSplit) or isinstance(out[2], LightSplit):
        raise ValueError("light_save* — лише для легких блоків")
    return out


def _light(blk, h, cfg, r):
    """Легкий блок, за потреби під jax.checkpoint (r: False | True | policy | LightSplit)."""
    if r is False:
        return flex_block(blk, h, cfg)
    if isinstance(r, LightSplit):
        q, k, v = jax.checkpoint(pre_attn, static_argnums=(2,), policy=r.policy)(blk, h, cfg)
        a = local_attention(q, k, v, cfg.window, cfg.attn_impl)
        return jax.checkpoint(post_attn, static_argnums=(3,), policy=r.policy)(blk, h, a, cfg)
    return jax.checkpoint(flex_block, static_argnums=(2,), policy=None if r is True else r)(blk, h, cfg)


# ---------------------------------------------------------------- forward

def forward(params, consts, x, prep, tau, cfg, train=True, remat=False, dist=None):
    """x (B,T) int. → (hn, lh), cost, mix (c1, g), tau_new, stats.
    dist — None (W=1) або dist.Dist (усередині shard_map): шардовані рядки словника, EP-обмін, спільний tau."""
    p = params
    take = None if dist is None else dist.vocab.take
    vocab = None if dist is None else dist.vocab
    exchange = moe.IDENTITY if dist is None else dist.exchange
    qhook = None if dist is None else dist.quantile_hook
    B, T = x.shape
    cdt = cfg.cdt
    r_light, r_heavy, r_mem = remat_parts(remat)
    stats = {}
    h = (p["emb"][x] if take is None else take(p["emb"], x)).astype(cdt)
    # --- INDUCT=lock: суміш і вхід
    cand, on, bk, src = prep["cand"], prep["on"], prep["bk"], prep["src"]
    lock_w = consts["lock_w"]
    val = induct.induct_val(on, bk, lock_w)
    c1, g, st_mix = induct.induct_mix(cand, on, bk, p["mix_w"])
    stats.update(st_mix)
    stats["trust"] = jnp.mean(lock_w.astype(F32))
    h = induct.induct_in(h, val, src, p["ind_in_g"])
    # --- легкі до пам'яті
    for blk in p["light_a"]:
        h = _light(blk, h, cfg, r_light)
    h_a = h                                           # для голови обруча при hoop_at="light_a"
    # --- пам'ять
    reads, snaps, st, _ = memory.memory_pass(p["memory"], h, mem_cfg(cfg), remat=r_mem)
    h = h + reads.astype(h.dtype)
    stats["mem_cards"] = st["valid"].astype(F32).sum(1).mean()
    for blk in p["light_b"]:
        h = _light(blk, h, cfg, r_light)
    lh = rmsnorm(h, p["l_norm"])
    # --- здивування → якорі
    s_ = ce.surprise(lh, p["l_head0"], p["l_head1"], x, cfg.ce_chunk, vocab)
    if getattr(cfg, "hoop", False):
        # --- обруч: модель сама передбачає свою помилку ДО важких шарів і віддає їм лише top-k «не з'їдених»
        # (локально на чипі, по кожній послідовності — у shard_map B локальне, комунікацій немає)
        feat = lh if cfg.hoop_at == "light_b" else rmsnorm(h_a, p["l_norm"])
        pred = hoop.predict(p["hoop"], feat, s_)
        Bq, Tq = pred.shape
        if cfg.hoop_thr >= 0:      # каузально: поріг на кожне слово окремо, без ранжування по послідовності
            idx = jnp.broadcast_to(jnp.arange(Tq)[None], (Bq, Tq))
            valid = (pred > cfg.hoop_thr).at[:, 0].set(True)
        else:
            idx, valid = hoop.select(pred, cfg.hoop_budget)
        tau_new = tau
        stats["anchors"] = jnp.mean(valid.astype(F32)) * idx.shape[1] / Tq
        stats["_hoop_pred"], stats["_hoop_idx"] = pred, idx      # забирає loss_terms (не йдуть у pmean)
        s_heavy = pred
    else:
        idx, valid, tau_new, st_a = anchors.select_anchors(s_, tau, cfg.frac, cfg.cap, train, quantile_hook=qhook)
        stats.update(st_a)
        s_heavy = s_
    # --- важкі (MoE) на якорях
    h, aux, drop = moe.heavy_stack(p["heavy"], h, idx, valid, s_heavy, moe_cfg(cfg), exchange, remat=r_heavy, stats=stats)
    stats["moe_drop"] = drop
    for blk in p["final"]:
        h = _light(blk, h, cfg, r_light)
    # --- перечитування
    o, cost = reread.reread_sep(p["reread"], h, s_, x, p["book"], snaps, rr_cfg(cfg), take)
    h = h + o.astype(h.dtype)
    hn = rmsnorm(h, p["norm"])
    return (hn, lh), cost, aux, (c1, g), tau_new, stats


def loss_terms(params, consts, xy, prep, tau, cfg, train=True, remat=False, dist=None):
    """xy (B,T+1). → dict доданків (main, aux, light, cost), total, tau_new, stats."""
    x, y = xy[:, :-1], xy[:, 1:]
    B, T = x.shape
    (hn, lh), cost, aux, (c1, g), tau_new, stats = forward(params, consts, x, prep, tau, cfg, train, remat, dist)
    vocab = None if dist is None else dist.vocab
    logp, st_r = ce.short_logp(hn.reshape(B * T, -1), params["head"], y.reshape(-1), consts["sc"], cfg.ce_chunk,
                               vocab)
    stats.update(st_r)
    main = induct.mix_nll(logp.reshape(B, T), c1, g, y) / (B * T)
    hl = None
    if "_hoop_pred" in stats:
        pred, idx = stats.pop("_hoop_pred"), stats.pop("_hoop_idx")
        # справжня помилка моделі на кожному слові (та сама суміш, що в main), зупинена — ціль обруча
        py = jnp.exp(logp.reshape(B, T).astype(F32))
        pm = (1 - g).astype(F32) * py + g.astype(F32) * (c1 == y).astype(F32)
        nll_tok = jax.lax.stop_gradient(-jnp.log(jnp.maximum(pm, 1e-30)))
        hl = hoop.loss(pred, nll_tok)
        stats.update(hoop.stats(pred, nll_tok, idx, cfg.hoop_eat))
    if not train:
        return dict(main=main), main, tau_new, stats
    light = ce.light_head_aux(lh, params["l_head0"], params["l_head1"], y, cfg.ce_chunk, vocab)
    terms = dict(main=main, aux=aux, light=light, cost=cost.astype(F32))
    total = main + 0.01 * aux + 0.3 * light + cfg.read_cost * cost.astype(F32)
    if hl is not None:
        terms["hoop"] = hl
        total = total + cfg.hoop_w * hl
    return terms, total, tau_new, stats


def train_loss(params, consts, xy, prep, tau, cfg, remat=False, dist=None):
    """Для value_and_grad(has_aux=True): loss, (tau_new, terms, stats). dist — див. forward (W>1: dist.py)."""
    terms, total, tau_new, stats = loss_terms(params, consts, xy, prep, tau, cfg, True, remat, dist)
    return total, (tau_new, terms, stats)
