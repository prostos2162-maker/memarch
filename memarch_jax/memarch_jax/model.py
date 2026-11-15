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

from . import ce, induct, anchors, moe, memory, reread, rope, uroboros, fishki
from .attn import local_attention
from .light import flex_block, init_block, post_attn, pre_attn, pre_attn_rope
from .prims import rmsnorm, dump_proof_norm
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
    fill = 1.0 if getattr(cfg, "uroboros", False) else cfg.frac / cfg.cap
    return moe.MoECfg(E=cfg.E, k=cfg.moe_k, group=cfg.moe_group, cap_align=cfg.cap_align, fill=fill,
                      fast=getattr(cfg, "moe_fast", True), a2a_chunks=getattr(cfg, "moe_a2a_chunks", 1))


def rope_cfg(cfg):
    return rope.RopeCfg(K=cfg.rope_k, links=cfg.rope_links, win=cfg.rope_win, span_cap=cfg.rope_span_cap,
                        gather_cap=cfg.rope_gather_cap or None, route=cfg.rope_route, G=cfg.rope_hier_g,
                        NB=cfg.rope_hier_nb, BM=cfg.rope_hier_bm, excl_win=cfg.rope_excl_win,
                        route_rot=cfg.rope_route_rot, rgrad=cfg.rope_rgrad, chunk=cfg.rope_chunk,
                        route_lazy=cfg.rope_route_lazy, dense_read=cfg.rope_dense_read, route_bf16=cfg.rope_route_bf16, pallas=cfg.rope_pallas,
                        read=getattr(cfg, "rope_read", "v3"), mass_pos=bool(getattr(cfg, "remind", False)),
                        mgrad=bool(getattr(cfg, "remind", False)) and getattr(cfg, "remind_use_w", 0.0) > 0)


def init_tau(cfg, tau=7.5):
    """Початковий стан порогів: скаляр (як було) або (2,) = [tau важких якорів, rope_tau] при cfg.rope;
    (3,) = [.., rope_qtau] при cfg.rope_qfrac < 1 (поріг слів, що читають трос)."""
    if not getattr(cfg, "rope", False):
        return jnp.asarray(tau, F32)
    if getattr(cfg, "rope_qfrac", 1.0) < 1.0:
        return jnp.asarray([tau, cfg.rope_tau0, cfg.rope_qtau0], F32)
    return jnp.asarray([tau, cfg.rope_tau0], F32)


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
    if getattr(cfg, "fish_n", 0) > 0:               # фішки замість важких MoE-шарів
        del p["heavy"]
        p["fish"] = fishki.init(nk(), cfg)
        p["fish_router"] = jnp.zeros((d, cfg.fish_kinds), F32)
    if getattr(cfg, "rope", False):                 # вагонетка замість mem2 і перечитування (book — лише для reread)
        for name in ("memory", "reread", "book"):
            del p[name]
    if getattr(cfg, "prev_in", False):
        p["prev_g"] = jnp.full((1,), -1.0, F32)
    if getattr(cfg, "uroboros", False):
        p["hoop"] = uroboros.init(nk(), d)        # ключ параметрів лишається "hoop": сумісність зі старими ckpt
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


def _light_rope(blk, h, cfg, r, rp, routed=None, qmask=None, docs=None):
    """Легкий блок із вагонеткою. Увага завжди поза зовнішнім checkpoint: її шматки вже мають свій jax.checkpoint.
    rp = (own, apos, avalid). → h, stats вагонетки."""
    if r is False:
        pre, post = pre_attn_rope, post_attn
    else:
        pol = r.policy if isinstance(r, LightSplit) else (None if r is True else r)
        pre = jax.checkpoint(pre_attn_rope, static_argnums=(2,), policy=pol)
        post = jax.checkpoint(post_attn, static_argnums=(3,), policy=pol)
    qr, kr, qu, ku, v = pre(blk, h, cfg)
    a, share, st = rope.rope_attention(qr, kr, qu, ku, v, *rp, rope_cfg(cfg), routed=routed, qmask=qmask, docs=docs)
    return post(blk, h, a, cfg), dict(st, rope_share=share)


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
    use_rope = getattr(cfg, "rope", False)
    rope_tau_new = None
    qf = getattr(cfg, "rope_qfrac", 1.0) if use_rope else 1.0
    rope_qtau = rope_qtau_new = None
    if use_rope:                                      # стан порогів (2,) або (3,): див. init_tau
        if qf < 1.0:
            rope_qtau = tau[2]
        tau, rope_tau = tau[0], tau[1]
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
    h0 = h
    h = induct.induct_in(h, val, src, p["ind_in_g"])
    if getattr(cfg, "prev_in", False):                # ПРОПОЗИЦІЯ (вимкнено): слово знає попереднє слово (як 2-слівний замок)
        pg = jax.nn.sigmoid(p["prev_g"].astype(F32)).astype(h.dtype)
        h = h + pg * jnp.concatenate([jnp.zeros_like(h0[:, :1]), h0[:, :-1]], 1)
    # --- легкі до пам'яті
    for blk in p["light_a"]:
        h = _light(blk, h, cfg, r_light)
    h_a = h                                           # для голови обруча при uroboros_at="light_a"
    # --- пам'ять: вагонетка (cfg.rope) замість mem2 і перечитування, інакше — як було
    if use_rope:
        # якорі — здивування легкої голови після light_a; порядок як у full2: спершу оновлення порогу, потім маска
        sr = ce.surprise(rmsnorm(h, p["l_norm"]), p["l_head0"], p["l_head1"], x, cfg.ce_chunk, vocab)
        rope_tau_new = rope_tau
        if train:
            q = jnp.quantile(sr[:, 1:].reshape(-1)[:100_000], 1 - (getattr(cfg, "rope_frac", 0.0) or cfg.frac))
            q = q if qhook is None else qhook(q)
            rope_tau_new = (rope_tau * 0.9 + 0.1 * q).astype(F32)
        rp = rope.anchors_from_surprise(sr, rope_tau_new, max(1, round(T * (getattr(cfg, "rope_cap", 0.0) or cfg.cap))),
                                        top=bool(getattr(cfg, "rope_anchor_top", False)))
        stats["rope_anchors"] = rp[2].astype(F32).sum(1).mean() / T
        # ДІАГНОСТИКА (без впливу на навчання): чи бачить маршрутизація слова голок. Ключі голок — токени
        # [k0, k0+512). key_link — частка таких слів у перших rope_links словах шматка свого якоря (лише їх бачить
        # пошук), key_anch — частка, що самі є якорями. Гіпотеза 13.11: у тексті/посеред рядка key_link ≈ 0.3–0.5.
        if getattr(cfg, "rope_key_diag", False):                # --rope_key_diag (вимкнено = як було)
            _k0 = int(getattr(cfg, "remind_tok0", 1000))
            _own, _apos = rp[0], rp[1]
            _isk = jax.lax.stop_gradient((x >= _k0) & (x < _k0 + 512))
            _a = jnp.take_along_axis(_apos, jnp.clip(_own, 0, _apos.shape[1] - 1), axis=1)
            _t = jnp.arange(T, dtype=jnp.int32)[None, :]
            _n = jnp.maximum(_isk.astype(F32).sum(), 1.0)
            stats["rope_key_link"] = (_isk & (_own >= 0) & (_t - _a < cfg.rope_links)).astype(F32).sum() / _n
            stats["rope_key_anch"] = (_isk & (_own >= 0) & (_t == _a)).astype(F32).sum() / _n
        qmask = None
        docs = None
        if getattr(cfg, "rope_doc_eos", -1) >= 0:              # межа документа для троса (лише маска вибору якорів)
            docs = jax.lax.stop_gradient(jnp.cumsum((x == cfg.rope_doc_eos).astype(jnp.int32), axis=1))
        if qf < 1.0:
            # ROPE_QFRAC (як full2, 10.11): трос читають лише слова зі здивуванням sr >= порога; решта — лише вікно.
            # Поріг — EMA квантиля (1-qf) у навчанні, заморожений в оцінці (причинно, як rope_tau)
            rope_qtau_new = rope_qtau
            if train:
                qq = jnp.quantile(sr[:, 1:].reshape(-1)[:100_000], 1 - qf)
                qq = qq if qhook is None else qhook(qq)
                rope_qtau_new = (rope_qtau * 0.9 + 0.1 * qq).astype(F32)
            qmask = jax.lax.stop_gradient(sr >= rope_qtau_new)
            stats["rope_qfrac_real"] = qmask.astype(F32).mean()
        routed = None
        mps = []
        for bi, blk in enumerate(p["light_b"]):
            if cfg.rope_layers and bi >= cfg.rope_layers:
                h = _light(blk, h, cfg, r_light)
                continue
            h, st_r = _light_rope(blk, h, cfg, r_light, rp, routed, qmask, docs)
            rt = st_r.pop("_route")
            mp = st_r.pop("_mass_pos", None)
            if mp is not None:                        # нагадування: частка троса по позиціях, середнє по шарах троса
                mps.append(mp)
            routed = rt if cfg.rope_share_route else None
            if bi == 0:                               # як full2: rope_share першого шару
                stats.update(st_r)
        if mps:
            stats["_rope_mass_pos"] = sum(mps) / len(mps)
    else:
        reads, snaps, st, _ = memory.memory_pass(p["memory"], h, mem_cfg(cfg), remat=r_mem)
        h = h + reads.astype(h.dtype)
        stats["mem_cards"] = st["valid"].astype(F32).sum(1).mean()
        for blk in p["light_b"]:
            h = _light(blk, h, cfg, r_light)
    lh = rmsnorm(h, p["l_norm"])
    # --- здивування → якорі
    s_ = ce.surprise(lh, p["l_head0"], p["l_head1"], x, cfg.ce_chunk, vocab)
    fish_on = getattr(cfg, "fish_n", 0) > 0
    if fish_on:
        # --- фішки (схема Стефана 11.11): увесь крок — один вид, текст іде через комбінацію фішок prep['fish'];
        # якорі для важкої частини не потрібні (трос бере свої якорі в легких шарах)
        r_ce, r_acc, r_logits = fishki.router_loss(p["fish_router"], lh, prep["kind"].reshape(-1))
        fax = None if dist is None else dist.vocab.axis
        if getattr(cfg, "fish_eval", False):
            # готова модель: роутер вибирає вид КОЖНОГО рядка, рядок іде через увесь набір цього виду
            kinds = r_logits.argmax(-1)
            out = jnp.zeros_like(h)
            ec = int(getattr(cfg, "fish_eval_c", 0))
            for k, pool in enumerate(fishki.pools(cfg)):
                if 0 < ec < len(pool):   # без фінальної фази: як на тренуванні — ec фішок виду (фіксована вибірка)
                    pool = [int(v) for v in np.random.default_rng(1000 + k).permutation(pool)[:ec]]
                hk = fishki.chain(p["fish"], h, jnp.asarray(pool, jnp.int32), fax, remat=r_heavy,
                                  center=getattr(cfg, "fish_center", False))
                out = out + jnp.where((kinds == k)[:, None, None], hk, jnp.zeros_like(hk))
            h = out
        else:
            order = prep["fish"].reshape(-1, prep["fish"].shape[-1])[0]
            h = fishki.chain(p["fish"], h, order, fax, remat=r_heavy,
                                  center=getattr(cfg, "fish_center", False))
        aux = 10.0 * r_ce                                  # у total іде 0.01·aux = 0.1·CE роутера
        drop = jnp.zeros((), F32)
        tau_new = tau
        stats["fish_router_acc"] = r_acc
    elif getattr(cfg, "uroboros", False):
        # --- обруч: модель сама передбачає свою помилку ДО важких шарів і віддає їм лише top-k «не з'їдених»
        # (локально на чипі, по кожній послідовності — у shard_map B локальне, комунікацій немає)
        feat = lh if cfg.uroboros_at == "light_b" else rmsnorm(h_a, p["l_norm"])
        pred = uroboros.predict(p["hoop"], feat, s_)
        Bq, Tq = pred.shape
        if cfg.uroboros_thr >= 0:      # каузально: поріг на кожне слово окремо, без ранжування по послідовності
            idx = jnp.broadcast_to(jnp.arange(Tq)[None], (Bq, Tq))
            valid = (pred > cfg.uroboros_thr).at[:, 0].set(True)
        else:
            idx, valid = uroboros.select(pred, cfg.uroboros_budget)
        tau_new = tau
        stats["anchors"] = jnp.mean(valid.astype(F32)) * idx.shape[1] / Tq
        stats["_uroboros_pred"], stats["_uroboros_idx"] = pred, idx      # забирає loss_terms (не йдуть у pmean)
        s_heavy = pred
    else:
        idx, valid, tau_new, st_a = anchors.select_anchors(s_, tau, cfg.frac, cfg.cap, train, quantile_hook=qhook)
        stats.update(st_a)
        s_heavy = s_
    # --- важкі (MoE) на якорях
    if fish_on:
        pass
    elif getattr(cfg, "heavy_all", False) or getattr(cfg, "heavy_rotate", False):
        fi = None if getattr(cfg, "heavy_all", False) else prep["hfull"].reshape(-1)[0]
        h, aux, drop = moe.heavy_stack_full(p["heavy"], h, idx, valid, s_heavy, moe_cfg(cfg), exchange, remat=r_heavy,
                                            stats=stats, full_idx=fi, all_full=getattr(cfg, "heavy_all", False))
    else:
        h, aux, drop = moe.heavy_stack(p["heavy"], h, idx, valid, s_heavy, moe_cfg(cfg), exchange, remat=r_heavy, stats=stats)
    stats["moe_drop"] = drop
    for blk in p["final"]:
        h = _light(blk, h, cfg, r_light)
    # --- перечитування (при вагонетці його немає: full2 no_mem → cost = 0)
    if use_rope:
        cost = jnp.zeros((), F32)
        tau_new = jnp.stack([tau_new, rope_tau_new] + ([rope_qtau_new] if qf < 1.0 else []))
    else:
        o, cost = reread.reread_sep(p["reread"], h, s_, x, p["book"], snaps, rr_cfg(cfg), take)
        h = h + o.astype(h.dtype)
    if getattr(cfg, "stream_m", False):                 # K1 (13.11): середнє h перед фінальною norm (dist: pmean)
        stats["stream_m"] = jax.lax.stop_gradient(h).astype(F32).mean((0, 1))
    if getattr(cfg, "norm_dump_proj", False):           # fix14 F1: norm гасить спільний вектор у будь-якому каналі
        hn = dump_proof_norm(h, p["norm"], prep.get("stream_u") if isinstance(prep, dict) else None)
    else:
        hn = rmsnorm(h, p["norm"])
    if getattr(cfg, "safe_norm_ratio", 0.0) > 0:            # fix16: спільний вектор після norm ≤ ratio·сигнал слів
        from .prims import cap_shared
        hn = cap_shared(hn, cfg.safe_norm_ratio, getattr(cfg, "safe_norm_t0", 256))
    return (hn, lh), cost, aux, (c1, g), tau_new, stats


def loss_terms(params, consts, xy, prep, tau, cfg, train=True, remat=False, dist=None):
    """xy (B,T+1). → dict доданків (main, aux, light, cost), total, tau_new, stats."""
    x, y = xy[:, :-1], xy[:, 1:]
    B, T = x.shape
    if getattr(cfg, "remind_blank", False) and isinstance(prep, dict) and "rem_y" in prep:
        # --remind_blank: у тексті на місці відповіді заглушка, справжня відповідь — лише ціль (бачить у «оцінці»)
        y = jnp.where(prep["rem"] > 0, prep["rem_y"].astype(y.dtype), y)
    (hn, lh), cost, aux, (c1, g), tau_new, stats = forward(params, consts, x, prep, tau, cfg, train, remat, dist)
    vocab = None if dist is None else dist.vocab
    _rp = prep.get("rem") if isinstance(prep, dict) else None
    # 15.11: k+1 — правильне слово може бути серед top-k, конкурентів беремо k без нього
    _iwk = int(cfg.iw_k) + 1 if (train and getattr(cfg, "inst_word", False) and getattr(cfg, "iw_rewrite", False)
                                 and not getattr(cfg, "iw_random", False)) else 0
    _iwr = bool(train and getattr(cfg, "inst_word", False)
                and (getattr(cfg, "iw_random", False) or getattr(cfg, "iw_rewrite", False)))
    _sl = ce.short_logp(hn.reshape(B * T, -1), params["head"], y.reshape(-1), consts["sc"], cfg.ce_chunk,
                        vocab, **({"prio": _rp} if _rp is not None else {}), **({"topk": _iwk} if _iwk else {}),
                        **({"want_lse": True} if _iwr else {}))
    logp, st_r = _sl[0], dict(_sl[1])
    lse_s = st_r.pop("_lse_s", None)           # (B·T,) lse короткого рівня з градієнтом (лише iw_random)
    iw_top = _sl[2] if _iwk else None          # 15.11: top-k переписаної відповіді — з того ж проходу main CE
    stats.update(st_r)
    _rm = prep.get("rem") if isinstance(prep, dict) else None
    if getattr(cfg, "remind_mask_main", False) and _rm is not None:
        # fix19: у рядках з голками токени голок (ключі/значення/мітка) не вчать main — вони випадкові, а повний
        # softmax на них тягне вгору всю групу токенів 1000..1576 (і кошик рідкісних) на всьому тексті
        _k0 = int(getattr(cfg, "remind_tok0", 1000))
        _row = (_rm > 0).any(axis=1, keepdims=True)
        _keep = 1.0 - ((y >= _k0) & (y <= _k0 + 576) & _row).astype(F32)
        _py = jnp.exp(logp.reshape(B, T).astype(F32))
        _p = (1 - g).astype(F32) * _py + g.astype(F32) * (c1 == y).astype(F32)
        main = (-jnp.log(jnp.maximum(_p, 1e-30)) * _keep).sum() / (B * T)
        stats["remind_main_masked"] = jax.lax.stop_gradient(1.0 - _keep.mean())
    else:
        main = induct.mix_nll(logp.reshape(B, T), c1, g, y) / (B * T)
    hl = None
    if "_uroboros_pred" in stats:
        pred, idx = stats.pop("_uroboros_pred"), stats.pop("_uroboros_idx")
        # справжня помилка моделі на кожному слові (та сама суміш, що в main), зупинена — ціль обруча
        py = jnp.exp(logp.reshape(B, T).astype(F32))
        pm = (1 - g).astype(F32) * py + g.astype(F32) * (c1 == y).astype(F32)
        nll_tok = jax.lax.stop_gradient(-jnp.log(jnp.maximum(pm, 1e-30)))
        hl = uroboros.loss(pred, nll_tok)
        stats.update(uroboros.stats(pred, nll_tok, idx, cfg.uroboros_eat))
    rem = prep.get("rem") if isinstance(prep, dict) else None
    rem_t = None
    if rem is not None:                                                 # нагадування: лише відповіді голок
        if getattr(cfg, "remind_cand", False):
            # fix19: softmax лише по 64 токенах-значеннях (v0..v0+63): спільна складова їхніх рядків голови
            # градієнта не отримує → немає глобального підсилення групи токенів голок і кошика V-1
            _v0 = int(getattr(cfg, "remind_tok0", 1000)) + 512
            _ids = jnp.arange(_v0, _v0 + 64, dtype=jnp.int32)
            Wv = params["head"][_ids] if vocab is None else vocab.take_same(params["head"], _ids)
            hv = hn.reshape(B * T, -1)
            lv = jnp.dot(hv.astype(jnp.bfloat16), Wv.astype(jnp.bfloat16).T, preferred_element_type=F32)
            lsm = jax.nn.log_softmax(lv, axis=-1)
            yi = jnp.clip(y.reshape(-1) - _v0, 0, 63)
            py_r = jnp.exp(jnp.take_along_axis(lsm, yi[:, None], 1)[:, 0]).reshape(B, T)
        else:
            py_r = jnp.exp(logp.reshape(B, T).astype(F32))
        pm_r = (1 - g).astype(F32) * py_r + g.astype(F32) * (c1 == y).astype(F32)
        mk = rem.astype(F32)
        den = float(B * cfg.remind_q)
        nll_r = -jnp.log(jnp.maximum(pm_r, 1e-30))
        rem_t = (nll_r * mk).sum() / den
        hit_r = (pm_r > 0.5).astype(F32)
        stats["remind_hit"] = (hit_r * mk).sum() / den   # p(відповідь)>0.5 ⇒ argmax правильний
        stats["remind_n"] = mk.sum() / den
        stats["remind_nll"] = rem_t
        if getattr(cfg, "remind_mean", False) or getattr(cfg, "remind_strict", False):
            # етап 1: точність окремо на 1..4-му питанні рядка (перебір дає 25/33/50/100%, чесний зв'язок — рівно)
            qn = jnp.cumsum(mk, axis=1) * mk
            for qk in range(1, 5):
                mq = (qn == qk).astype(F32)
                stats[f"remind_hit_q{qk}"] = (hit_r * mq).sum() / den
                stats[f"remind_n_q{qk}"] = mq.sum() / den
        strict = getattr(cfg, "remind_strict", False) and isinstance(prep, dict) and "rem_vals" in prep
        mk_ce = mk
        if strict:
            # «чесний тест» (Стефан 12.11): відповіді рядка йдуть по порядку; після першої помилки правильні
            # відповіді не винагороджуються (їх можна взяти перебором залишку) — keep; штраф pen діє на всі
            wrong = mk * (1.0 - jax.lax.stop_gradient(hit_r))
            prior = jnp.cumsum(wrong, axis=1) - wrong
            mk_ce = mk * (prior < 0.5).astype(F32)
            stats["remind_strict_keep"] = mk_ce.sum() / jnp.maximum(mk.sum(), 1.0)
            # штраф за перебір: −log(1 − P(інші значення свого рядка)) на кожній відповіді
            Q = cfg.remind_q
            ai = jax.lax.top_k(mk * (T - jnp.arange(T, dtype=F32))[None], Q)[1]          # (B,Q) позиції відповідей
            av = jnp.take_along_axis(mk, ai, 1)                                             # (B,Q) 1 — справжня
            hq = jnp.take_along_axis(hn.reshape(B, T, -1), ai[..., None], 1).reshape(B * Q, -1)
            yq = jnp.take_along_axis(y, ai, 1)                                              # (B,Q)
            rv = prep["rem_vals"].astype(jnp.int32)                                          # (B,n), −1 — немає
            p_oth = jnp.zeros((B, Q), F32)
            drop = jnp.zeros((), F32)
            for j in range(rv.shape[1]):
                vj = rv[:, j]
                lpj, stj = ce.short_logp(hq, params["head"], jnp.broadcast_to(jnp.maximum(vj, 0)[:, None], (B, Q)
                                                                              ).reshape(-1),
                                         consts["sc"], cfg.ce_chunk, vocab, prio=jnp.ones((B * Q,), jnp.int8))
                other = ((vj >= 0)[:, None] & (vj[:, None] != yq)).astype(F32)
                p_oth = p_oth + jnp.exp(lpj.reshape(B, Q).astype(F32)) * other
                drop = drop + stj["rare_drop"]
            pen = (-jnp.log(jnp.maximum(1.0 - p_oth, 1e-6)) * av).sum() / jnp.maximum(av.sum(), 1.0)
            stats["remind_strict_pen"] = jax.lax.stop_gradient(pen)
            stats["remind_strict_pother"] = jax.lax.stop_gradient((p_oth * av).sum() / jnp.maximum(av.sum(), 1.0))
            stats["remind_strict_rare_drop"] = jax.lax.stop_gradient(drop / rv.shape[1])
            rem_t = (nll_r * mk_ce).sum() / den
        if getattr(cfg, "remind_mean", False):                          # --remind_pack: середнє по відповідях (як torch CE)
            rem_t = rem_t * (den / jnp.maximum(mk_ce.sum(), 1.0))
        if strict:
            rem_t = rem_t + cfg.remind_strict_w * pen
        mpos = stats.pop("_rope_mass_pos", None)
        if mpos is not None:                                            # частка уваги троса саме на питаннях
            use = (mpos.astype(F32) * mk).sum() / den                       # Σ mass на питаннях / (B·q)
            stats["remind_rope_share"] = jax.lax.stop_gradient(use)
            if cfg.remind_use_w > 0:                                    # штраф: mean(1 − mass) по питаннях
                rem_use = (mk.sum() / den) - use
                stats["remind_use_pen"] = jax.lax.stop_gradient(rem_use)
    stats.pop("_rope_mass_pos", None)
    if not train:
        return dict(main=main), main, tau_new, stats
    light = ce.light_head_aux(lh, params["l_head0"], params["l_head1"], y, cfg.ce_chunk, vocab)
    terms = dict(main=main, aux=aux, light=light, cost=cost.astype(F32))
    total = main + 0.01 * aux + 0.3 * light + cfg.read_cost * cost.astype(F32)
    if getattr(cfg, "hn_guard_w", 0.0) > 0:
        # fix17 (14.11, анатомія r12_an): злам 15000→15250 — у стовбурі: середнє hn по позиціях рядка 9.8→45.9
        # (напрям новий, cos 0.04), довжина hn 35.5→58.1; голова 15250 на hn 15000 дає +0.017. Шарнірний штраф
        # лише понад пороги (на 15000 обидва нижче ⇒ доданок і градієнт = 0, forward/eval без змін).
        h32 = hn.astype(F32)
        mn = jnp.sqrt(jnp.sum(jnp.square(h32.mean(1)), -1) + 1e-12)                  # (B,) |середнє hn рядка|
        ln = jnp.sqrt(jnp.sum(jnp.square(h32), -1) + 1e-12).mean(1)                   # (B,) середня |hn|
        em = jnp.maximum(mn / cfg.hn_guard_mean - 1.0, 0.0)
        el = jnp.maximum(ln / cfg.hn_guard_len - 1.0, 0.0)
        hg = jnp.mean(jnp.square(em)) + jnp.mean(jnp.square(el))
        stats["hn_mean"] = jax.lax.stop_gradient(mn.mean())
        stats["hn_len"] = jax.lax.stop_gradient(ln.mean())
        terms["hn_guard"] = hg
        total = total + cfg.hn_guard_w * hg
    if rem_t is not None:
        terms["remind"] = rem_t
        total = total + cfg.remind_w * rem_t
        if getattr(cfg, "remind_only", False):                         # етап 1: як torch loss_and_pred — лише відповіді
            total = total - main
        if cfg.remind_use_w > 0 and "remind_use_pen" in stats:
            terms["remind_use"] = rem_use
            total = total + cfg.remind_use_w * rem_use
    if hl is not None:
        terms["uroboros"] = hl
        total = total + cfg.uroboros_w * hl
    if getattr(cfg, "inst_word", False):
        # 15.11 Instinct на словах (задум Стефана): слово вгадане нормально (p ≥ iw_ok) — нічого понад звичайне;
        # ймовірність розмазана (лідер < iw_spread) — легкий поштовх правильного до iw_target (не далі);
        # модель впевнена в неправильному (лідер ≠ y, p_лідера > iw_cw) — обережно знизити лише це одне слово до
        # iw_cw (не далі) — без розмазування (урок «чесного тесту» 12.11). prep["iw_mult"] — посилення на відкаті.
        hv = hn.reshape(B * T, -1)
        lp = logp.reshape(B, T).astype(F32)
        ok = jax.lax.stop_gradient(jnp.exp(lp)) >= cfg.iw_ok
        mult = prep.get("iw_mult", jnp.float32(1.0)) if isinstance(prep, dict) else jnp.float32(1.0)
        if getattr(cfg, "iw_random", False):
            # 15.11 Стефан (дешево): правильне слово відоме → підняти його +δ (через уже пораховане lp, без нових
            # множень); знизити −δ ВИПАДКОВЕ слово короткого словника (не правильне), одне на рядок — B рядків таблиці
            # замість вибірки на кожне слово і без пошуку top-k. Діє лише на словах поза нормою (p < iw_ok).
            sc = consts["sc"]
            yb = y.reshape(B, T)
            act = ~ok
            _h = (jnp.sum(x.astype(jnp.uint32), axis=1) * jnp.uint32(2654435761)
                  + jnp.uint32(40503) * jnp.arange(B, dtype=jnp.uint32))
            o_row = (_h % jnp.uint32(sc["K"])).astype(jnp.int32)                     # (B,) позиція в короткому
            o_ids = jnp.asarray(sc["sv_ids"])[o_row]                                    # (B,) номер у словнику
            # o_ids з локальних x — на кожному чипі різні → take (власні номери), не take_same (той — для однакових)
            # швидкість (15.11): лише B рядків — маскований шлях (all_gather B·W номерів + psum_scatter (W,B,d)),
            # а не vocab.take («ag»: all_gather ВСІЄЇ таблиці (V,d) щокроку заради 4 рядків). Значення й градієнт ті самі.
            Wo = (params["head"][o_ids] if vocab is None else vocab._take_masked(params["head"], o_ids)).astype(F32)
            zo = jnp.einsum("btd,bd->bt", hn.astype(F32), Wo)
            act = act & (o_ids[:, None] != yb)
            # 15.11 виправлення: було δ·(z_o − log p_c) — сирий логіт z_o без lse: градієнт по логітах
            # δ(e_o − e_y + p) має суму δ ≠ 0 (тисне ВНИЗ усі логіти одним напрямком у hn на всіх словах рядка) і
            # не має межі (z_o → −∞ «вигідно» завжди; на TPU доданок iw дійшов до ≈ −1.8, loss < main). Переписана
            # відповідь q = p + δ'(e_c − e_o) як мішень CE(q,p): градієнт p − q = δ'(e_o − e_c) — лише два слова,
            # сума 0. log p_o = z_o − lse_s (короткий рівень, той самий lse, що в log p_c) → lse скорочується.
            # δ' = min(δ, p_o): у переписаній відповіді не можна відняти більше, ніж слово має (q_o ≥ 0).
            lpo = zo - lse_s.reshape(B, T)
            d_eff = jnp.minimum(jnp.float32(cfg.iw_delta), jax.lax.stop_gradient(jnp.exp(lpo)))
            iw = mult * jnp.sum(d_eff * (lpo - lp) * act) / (B * T)
            stats["iw_po"] = jax.lax.stop_gradient(jnp.sum(jnp.exp(lpo) * act) / jnp.maximum(act.sum(), 1))
            stats["iw_act_frac"] = jax.lax.stop_gradient(act.mean())
            spread, cw = act, jnp.zeros_like(act)   # для статистик нижче (у цій гілці розмаз/хибне не розрізняємо)
        elif getattr(cfg, "iw_rewrite", False):
            # 15.11 рішення Стефана: ПЕРЕПИСАНА відповідь q як мішень CE(q,p) на словах поза нормою (p_c < iw_ok):
            # q = p, крім правильного (+δ') і top-k конкурентів короткого словника без правильного (−δ'·p_i/Σp_конк);
            # δ' = min(δ, Σp_конк) — жоден не віддає більше, ніж має; решта словника = p; Σq = 1.
            # Градієнт по логітах p − q = δ'(Σ w_i e_i − e_c), w_i = p_i/Σp_конк: сума 0, обмежений.
            # loss = δ'·(Σ w_i·log p_i − log p_c), log p_i = tv_i − lse_s (lse скорочується; δ', w — без градієнта).
            # Швидкість: tv — top-k логіти з того ж проходу main CE (chunk_ce_top), їхній градієнт додається в тому ж
            # зворотному проході по шматках — без вибірок рядків на слово і без нових множень на словник.
            sc = consts["sc"]
            pk, ipk, _Ws, tv = iw_top
            k1 = pk.shape[-1]
            pk, ipk, tv = pk.reshape(B, T, k1), ipk.reshape(B, T, k1), tv.reshape(B, T, k1)
            yb = y.reshape(B, T)
            c_pos = jnp.asarray(sc["sv_pos"])[yb]
            other = ipk != c_pos[..., None]
            sel = other & (jnp.cumsum(other.astype(jnp.int32), -1) <= int(cfg.iw_k))
            pc = jnp.where(sel, pk, 0.0)
            S = pc.sum(-1)
            act = (~ok) & (S > 0)
            d_eff = jnp.minimum(jnp.float32(cfg.iw_delta), S)
            w = pc / jnp.maximum(S, 1e-30)[..., None]
            lpk = tv.astype(F32) - lse_s.reshape(B, T, 1)
            d = jnp.sum(w * lpk, -1) - lp
            iw = mult * jnp.sum(d_eff * d * act) / (B * T)
            p_top = pk[..., 0]
            spread = (~ok) & (p_top < cfg.iw_spread)
            cw = (~ok) & (p_top > cfg.iw_cw) & (ipk[..., 0] != c_pos)
            stats["iw_act_frac"] = jax.lax.stop_gradient(act.mean())
            stats["iw_deff"] = jax.lax.stop_gradient(jnp.sum(d_eff * act) / jnp.maximum(act.sum(), 1))
        else:
            p_top, top_tok = ce.short_top(hv, params["head"], consts["sc"], cfg.ce_chunk, vocab)
            p_top, top_tok = p_top.reshape(B, T), top_tok.reshape(B, T)
            spread = (~ok) & (p_top < cfg.iw_spread)
            cw = (~ok) & (p_top > cfg.iw_cw) & (top_tok != y)
            l_sp = jnp.maximum(jnp.log(jnp.float32(cfg.iw_target)) - lp, 0.0) * spread
            lpt = ce.short_only_logp(hv, params["head"], top_tok.reshape(-1), consts["sc"], cfg.ce_chunk, vocab)
            pt = jnp.exp(lpt.reshape(B, T).astype(F32))
            l_cw = jnp.maximum(-jnp.log1p(-jnp.minimum(pt, 1 - 1e-6)) - (-jnp.log1p(-jnp.float32(cfg.iw_cw))), 0.0) * cw
            iw = (cfg.iw_spread_w * l_sp.sum() + cfg.iw_cw_w * mult * l_cw.sum()) / (B * T)
        terms["iw"] = iw
        stats["iw_spread_frac"] = jax.lax.stop_gradient(spread.mean())
        stats["iw_cw_frac"] = jax.lax.stop_gradient(cw.mean())
        stats["iw_ok_frac"] = jax.lax.stop_gradient(ok.mean())
        total = total + iw
    return terms, total, tau_new, stats


def train_loss(params, consts, xy, prep, tau, cfg, remat=False, dist=None):
    """Для value_and_grad(has_aux=True): loss, (tau_new, terms, stats). dist — див. forward (W>1: dist.py)."""
    terms, total, tau_new, stats = loss_terms(params, consts, xy, prep, tau, cfg, True, remat, dist)
    return total, (tau_new, terms, stats)
