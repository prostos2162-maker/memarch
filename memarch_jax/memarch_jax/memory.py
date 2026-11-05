"""Стовп пам'яті (порт mem2.Memory + hyperop/mempass._memory_pass_ingraph) на JAX.

Чисті функції над словником параметрів; ключі = суфікси torch state_dict модуля `memory.`
('norm_w.weight', 'face.weight', 'importance.weight', 'importance.bias', ...), тож конвертація 1:1.

Типи (MEM_BF16): лінійні шари — bf16-входи, fp32-накопичення, bf16-вихід (емуляція autocast); стан карток fp32;
решта — звичайна промоція (bf16 op f32 -> f32). einsum-и, де хоч один операнд fp32, рахуються у fp32 з
precision=HIGHEST (рішення в docs/decisions.md). cfg.cdt=float32 дає точний fp32-режим для тестів паритету.

Активні прапори запису: MEM_LEAN=1 (зміст лише для вибраних слів). Цикл по шматках — lax.scan зі сталими формами,
тіло під jax.checkpoint. Розриви рівних (top_k/argsort) — стабільні «менший індекс першим».
"""
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

HI = lax.Precision.HIGHEST
F32 = jnp.float32


@dataclass(frozen=True)
class MemCfg:
    """Локальна конфігурація пам'яті (типові значення mem2.Memory для d=2560). config.py (інший модуль) може
    будувати її зі свого Cfg; тут — мінімальна власна версія."""
    d: int = 2560
    Fn: int = 64          # свіжі картки
    Dn: int = 192         # тьмяні після прибирання
    Dmax: int = 256       # межа тьмяних
    nf: int = 2           # ланок лиця
    df: int = 64          # розмір лиця
    dc: int = 640         # стиснений зміст (d//4)
    W: int = 32           # записів на шматок
    K: int = 8            # фокус
    match: float = 0.8
    w_use: float = 2.0
    w_age: float = 1e-3
    seg: int = 512
    cdt: Any = jnp.bfloat16   # MEM_BF16=1 -> bf16; float32 -> точний режим
    eb: bool = False          # MEM_EINSUM_BF16: einsum-и як під torch autocast (входи й вихід bf16); типово — fp32

    @property
    def C(self):
        return self.Fn + self.Dmax


# ---- примітиви (мінімальні локальні версії; prims.py — інший модуль) ----------------------------------------
def rmsnorm(x, w, eps=1.1920929e-07):
    """nn.RMSNorm(eps=None) на fp32-вході: rms у fp32, множення на вагу з промоцією."""
    xf = x.astype(F32)
    y = xf * lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + eps)
    return y.astype(x.dtype) * w


_CPU = jax.default_backend() == "cpu"


def _lo(x, cdt):
    """Округлення входу множення до cdt. На CPU XLA не вміє bf16 x bf16 -> f32 (DotThunk), тож там округлені
    значення множаться у fp32 — добутки bf16 точні у fp32, тож результат той самий (з точністю до порядку суми)."""
    x = x.astype(cdt)
    return x.astype(F32) if (_CPU and cdt != F32) else x


def linear(x, w, b=None, cdt=jnp.bfloat16):
    """torch Linear під autocast(cdt): входи cdt, накопичення fp32, вихід cdt. w у torch-розкладці (out,in)."""
    y = lax.dot_general(_lo(x, cdt), _lo(w, cdt), (((x.ndim - 1,), (1,)), ((), ())),
                        precision=HI, preferred_element_type=F32)
    if b is not None:
        y = y + b.astype(cdt).astype(F32)
    return y.astype(cdt)


def sigmoid(x):
    """sigmoid як у torch для bf16: рахунок у fp32, одне округлення до типу входу."""
    return jax.nn.sigmoid(x.astype(F32)).astype(x.dtype)


def normalize(x, eps=1e-12):
    """F.normalize(x, dim=-1): x / max(||x||, eps) у типі x. Норма через max(sum², eps²) — без NaN-градієнта в нулі."""
    n2 = jnp.sum(jnp.square(x.astype(F32)), -1, keepdims=True)
    nrm = jnp.sqrt(jnp.maximum(n2, eps * eps)).astype(x.dtype)
    return x / nrm


def ein(spec, *ops, eb=False):
    """einsum із промоцією до спільного типу; fp32 — з повною точністю (HIGHEST).
    eb=True (MEM_EINSUM_BF16): як einsum під torch autocast — входи bf16, накопичення fp32, вихід bf16."""
    if eb:
        return jnp.einsum(spec, *[_lo(o, jnp.bfloat16) for o in ops], precision=HI,
                          preferred_element_type=F32).astype(jnp.bfloat16)
    dt = jnp.result_type(*ops)
    return jnp.einsum(spec, *[o.astype(dt) for o in ops], precision=HI, preferred_element_type=dt)


def bgather(t, idx):
    """t (B,C,...), idx (B,W,...) -> t[b, idx[b]] (B,W,...,  ...)."""
    return jax.vmap(lambda a, i: a[i])(t, idx)


def stable_topk(x, k):
    """top_k по останньому виміру; рівні — менший індекс першим (lax.top_k так і робить)."""
    return lax.top_k(x, k)


def argsort_stable(x, descending=False):
    if descending:
        # стабільний спадний порядок: рівні лишаються в порядку індексів
        return jnp.argsort(-x, axis=-1, stable=True)
    return jnp.argsort(x, axis=-1, stable=True)


# ---- стан ------------------------------------------------------------------------------------------------
def init_state(B, cfg: MemCfg):
    C = cfg.C
    z = lambda *s, dt=F32: jnp.zeros(s, dt)
    return dict(fc=z(B, cfg.Fn, cfg.d), dc=z(B, cfg.Dmax, cfg.dc), faces=z(B, C, cfg.nf, cfg.df),
                pos=z(B, C, dt=jnp.int32), valid=z(B, C, dt=jnp.bool_),
                bright=z(B, C), uses=z(B, C), touched=z(B, C))


def contents(p, st, cfg):
    """Зміст усіх карток (B,C,d) fp32: свіжі як є, тьмяні — up(dc) (bf16 при MEM_BF16, промоція до fp32)."""
    return jnp.concatenate([st["fc"], linear(st["dc"], p["up.weight"], cdt=cfg.cdt).astype(F32)], 1)


def _score(st, now, cfg):
    sc = st["bright"] + cfg.w_use * st["uses"] - cfg.w_age * (now - st["touched"])
    return jnp.where(st["valid"], sc, -1e9)


# ---- читання ---------------------------------------------------------------------------------------------
def mem_read(p, h, st, cfg: MemCfg):
    """h (B,S,d) fp32 -> (r (B,S,d) cdt, w (B,S,C) fp32 без градієнта)."""
    cdt = cfg.cdt
    n = rmsnorm(h, p["norm_r.weight"])
    q = normalize(linear(n, p["query.weight"], cdt=cdt))
    fa = normalize(st["faces"])
    sc = ein("bsf,bclf->bscl", q, fa, eb=cfg.eb).max(-1) * float(cfg.df) ** 0.5
    dimw = jax.nn.log_sigmoid(p["dim_weight"])
    sc = jnp.concatenate([sc[..., :cfg.Fn], sc[..., cfg.Fn:] + dimw.astype(sc.dtype)], -1)   # 0-вимірний тензор у torch не підвищує тип
    sc = jnp.where(st["valid"][:, None], sc, -1e4)
    anyv = st["valid"].any(-1)[:, None, None]
    w = jax.nn.softmax(sc.astype(F32), -1).astype(h.dtype) * anyv
    vals = linear(rmsnorm(contents(p, st, cfg), p["norm_c.weight"]), p["value.weight"], cdt=cdt)   # (B,C,d)
    mem = ein("bsc,bcd->bsd", w, vals, eb=cfg.eb)
    # фокус: картки, яких торкались останніми
    k = min(cfg.K, cfg.C)
    fidx = stable_topk(jnp.where(st["valid"], st["touched"], -1e9), k)[1]                        # (B,K)
    fsc = jnp.take_along_axis(sc, jnp.broadcast_to(fidx[:, None], sc.shape[:2] + (k,)), -1)
    fany = jnp.take_along_axis(st["valid"], fidx, 1).any(-1)[:, None, None]
    fw = jax.nn.softmax(fsc.astype(F32), -1).astype(h.dtype) * fany
    foc = ein("bsk,bkd->bsd", fw, bgather(vals, fidx), eb=cfg.eb)
    g = sigmoid(linear(jnp.concatenate([n, foc], -1), p["focus_gate.weight"], p["focus_gate.bias"], cdt=cdt))
    r = linear(g * foc + (1 - g) * mem, p["out.weight"], cdt=cdt)
    return r, lax.stop_gradient(w)


# ---- запис -----------------------------------------------------------------------------------------------
def put(t, idx, val, mask, eb=False):
    """Функційний запис t[b, idx] = val там, де mask; кілька записів в один слот — СЕРЕДНЄ (як у torch)."""
    C = t.shape[1]
    eb = eb and idx.shape[1] > 1    # torch CPU autocast: einsum зі згорткою розміру 1 іде не через bmm і лишається fp32
    oh = jax.nn.one_hot(idx, C, dtype=F32) * mask[..., None].astype(F32)       # (B,W,C)
    s = oh.sum(1)                                                                # (B,C)
    hit = jnp.minimum(s, 1.0)
    oh = oh / jnp.maximum(s, 1.0)[:, None, :]
    shp = hit.shape + (1,) * (t.ndim - 2)
    if t.dtype == jnp.bool_ or not jnp.issubdtype(t.dtype, jnp.floating):
        new = ein("bwc,bw...->bc...", oh, val.astype(F32), eb=eb)
        return jnp.where(hit.reshape(shp) > 0, jnp.round(new).astype(t.dtype), t)
    new = ein("bwc,bw...->bc...", oh.astype(t.dtype), val.astype(t.dtype), eb=eb)
    return t * (1 - hit.reshape(shp).astype(t.dtype)) + new


def mem_write(p, h, positions, st, now, w_read, cfg: MemCfg):
    """h (B,S,d) fp32 (= hs + r), positions (B,S) fp32, now — скаляр fp32, w_read (B,S,C) fp32. Повертає новий стан."""
    cdt, Fn = cfg.cdt, cfg.Fn
    B, S, d = h.shape
    n = rmsnorm(h, p["norm_w.weight"])
    f = linear(n, p["face.weight"], cdt=cdt)                                              # (B,S,df)
    imp = sigmoid(linear(n, p["importance.weight"], p["importance.bias"], cdt=cdt))  # (B,S,1)
    st = dict(st)
    used = lax.stop_gradient(w_read.astype(F32).sum(1))
    st["uses"] = st["uses"] + used
    st["touched"] = jnp.where(used > 0.5, jnp.zeros_like(st["touched"]) + now, st["touched"])
    valid = st["valid"]

    # --- без градієнта: вибір слів і слотів ---
    fs, imps = lax.stop_gradient(f), lax.stop_gradient(imp)
    fa = normalize(lax.stop_gradient(st["faces"]))
    sim = ein("bsf,bclf->bscl", normalize(fs).astype(fa.dtype), fa, eb=cfg.eb).max(-1)
    sim = jnp.where(valid[:, None], sim, -1.0).max(-1)                                     # (B,S)
    bright = (1 - jnp.maximum(sim, 0)) * imps[..., 0].astype(F32)
    Wn = min(cfg.W, S)
    cand = stable_topk(bright, Wn)[1]                                                     # (B,W)
    fc_ = bgather(fs, cand)
    cos = ein("bwf,bclf->bwcl", normalize(fc_).astype(fa.dtype), fa, eb=cfg.eb).max(-1)
    cos = jnp.where(valid[:, None], cos, -2.0)
    best_cos, best = cos.max(-1), jnp.argmax(cos, -1)
    upd = best_cos > cfg.match
    is_new = ~upd
    score = _score(st, now, cfg)
    key_f = jnp.where(valid[:, :Fn], score[:, :Fn], -2e9)
    rank_new = jnp.maximum(jnp.cumsum(is_new.astype(jnp.int32), -1) - 1, 0)
    slot = jnp.take_along_axis(argsort_stable(key_f), rank_new, 1)                        # (B,W) у [0,Fn)
    demote = is_new & jnp.take_along_axis(valid, slot, 1)
    key_d = jnp.where(valid[:, Fn:], score[:, Fn:], -2e9)
    rank_d = jnp.maximum(jnp.cumsum(demote.astype(jnp.int32), -1) - 1, 0)
    dslot = jnp.take_along_axis(argsort_stable(key_d), rank_d, 1)                         # у [0,Dmax)
    heat = jnp.where(valid[:, Fn:], w_read.astype(F32).sum(1)[:, Fn:], 0.0)
    hot_v, hot = heat.max(-1), jnp.argmax(heat, -1)
    promote = hot_v > 1.0
    weak = jnp.argmin(key_f, -1)
    slot, dslot, best, hot, weak = map(lax.stop_gradient, (slot, dslot, best, hot, weak))

    fw = bgather(f, cand)                                                                 # (B,W,df) cdt
    pw = jnp.take_along_axis(positions, cand, 1)
    bw = lax.stop_gradient(jnp.take_along_axis(bright, cand, 1))
    cw = linear(bgather(n, cand), p["content.weight"], cdt=cdt) * bgather(imp, cand)      # MEM_LEAN
    cont = contents(p, st, cfg)
    old = bgather(cont, best)
    g = sigmoid(linear(jnp.concatenate([cw.astype(F32), old], -1), p["gate.weight"], p["gate.bias"], cdt=cdt))
    new_val = old + g * (cw - old)

    pt = lambda t, i, v, m: put(t, i, v, m, eb=cfg.eb)
    down = lambda x: linear(x, p["down.weight"], cdt=cdt)
    up = lambda x: linear(x, p["up.weight"], cdt=cdt)
    fc, dc = st["fc"], st["dc"]
    faces, pos, valid, brightS, uses, touched = (st[k] for k in ("faces", "pos", "valid", "bright", "uses", "touched"))
    meta = lambda: (faces, pos, valid, brightS, uses, touched)

    # 1) витіснена свіжа -> тьмяна
    dfull = dslot + Fn
    dc = pt(dc, dslot, down(bgather(fc, slot)), demote)
    faces, pos, valid, brightS, uses, touched = [pt(t, dfull, bgather(t, slot), demote) for t in meta()]
    # 2) нові картки у свіжих місцях
    fc = pt(fc, slot, cw, is_new)
    chain0 = jnp.concatenate([fw[:, :, None], jnp.zeros((B, Wn, cfg.nf - 1, cfg.df), fw.dtype)], 2)
    faces = pt(faces, slot, chain0, is_new)
    pos = pt(pos, slot, pw, is_new)
    valid = pt(valid, slot, jnp.ones_like(is_new), is_new)
    brightS = pt(brightS, slot, bw, is_new)
    uses = pt(uses, slot, jnp.zeros_like(bw), is_new)
    touched = pt(touched, slot, jnp.zeros_like(bw) + now, is_new)
    # 3) оновлення наявних
    is_f, is_d = upd & (best < Fn), upd & (best >= Fn)
    fc = pt(fc, jnp.minimum(best, Fn - 1), new_val, is_f)
    dc = pt(dc, jnp.maximum(best - Fn, 0), down(new_val), is_d)
    chain = jnp.concatenate([fw[:, :, None].astype(F32), bgather(faces, best)[:, :, :-1]], 2)
    faces = pt(faces, best, chain, upd)
    pos = pt(pos, best, pw, upd)
    brightS = pt(brightS, best, jnp.maximum(bw, bgather(brightS, best)), upd)
    touched = pt(touched, best, jnp.zeros_like(bw) + now, upd)
    # 4) обмін: гаряча тьмяна <-> найслабша свіжа (маска promote)
    pm = promote[:, None]
    hf, wf = hot[:, None], weak[:, None]
    hot_c, weak_c = up(bgather(dc, hf)), down(bgather(fc, wf))
    fc, dc = pt(fc, wf, hot_c, pm), pt(dc, hf, weak_c, pm)
    hfull = hf + Fn
    sw = []
    for t in meta():
        a, b_ = bgather(t, wf), bgather(t, hfull)
        t = pt(t, wf, b_, pm)
        sw.append(pt(t, hfull, a, pm))
    faces, pos, valid, brightS, uses, touched = sw
    # 5) прибирання тьмяної (без градієнта)
    over = valid[:, Fn:].sum(1) > cfg.Dmax - cfg.W
    sc5 = (brightS + cfg.w_use * uses - cfg.w_age * (now - touched))[:, Fn:]
    sc5 = jnp.where(valid[:, Fn:], sc5, -1e9)
    keep = argsort_stable(argsort_stable(lax.stop_gradient(sc5), descending=True)) < cfg.Dn
    valid = jnp.concatenate([valid[:, :Fn], jnp.where(over[:, None], valid[:, Fn:] & keep, valid[:, Fn:])], 1)
    new_st = dict(fc=fc, dc=dc, faces=faces, pos=pos, valid=valid, bright=brightS, uses=uses, touched=touched)
    # діагностика шматка (B,): скільки оновили / нових / опущено, підйом, прибирання, середнє дублікатів
    stats = dict(upd=upd.sum(1), new=is_new.sum(1), demote=demote.sum(1), promote=promote, prune=over,
                 dup=upd.sum(1) - ((jax.nn.one_hot(best, cfg.C) * upd[..., None]).sum(1) > 0).sum(1))
    return new_st, stats


def seg_step(p, st, hs, s0, cfg: MemCfg):
    """Один шматок: читання + знімок (стан ДО запису) + запис. hs (B,S,d) fp32, s0 — fp32 скаляр."""
    r, w = mem_read(p, hs, st, cfg)
    snap = dict(faces=st["faces"], contents=contents(p, st, cfg), pos=st["pos"], valid=st["valid"])
    S = hs.shape[1]
    qpos = jnp.broadcast_to(s0 + jnp.arange(S, dtype=F32), hs.shape[:2])
    st, stats = mem_write(p, hs + r, qpos, st, s0 + S, w, cfg)
    return st, r, snap, stats


def memory_pass(p, h, cfg: MemCfg, remat=True):
    """h (B,T,d) -> (reads (B,T,d) cdt, snaps {faces (n,B,C,nf,df), contents (n,B,C,d), pos, valid}, st_last,
    stats {upd,new,demote,promote,prune,dup} (n,B) — діагностика шматків).
    Стан нульовий на кожен виклик (як init_state у torch). T кратне cfg.seg."""
    B, T, d = h.shape
    seg = cfg.seg
    assert T % seg == 0, (T, seg)
    n = T // seg
    hsegs = h.reshape(B, n, seg, d).swapaxes(0, 1)                                       # (n,B,seg,d)
    offs = jnp.arange(n, dtype=F32) * seg

    def body(st, xs):
        hs, s0 = xs
        st, r, snap, stats = seg_step(p, st, hs.astype(F32), s0, cfg)
        return st, (r, snap, stats)

    if remat:
        body = jax.checkpoint(body, policy=None if remat is True else remat)   # remat: True або policy
    st, (rs, snaps, stats) = lax.scan(body, init_state(B, cfg), (hsegs, offs))
    reads = rs.swapaxes(0, 1).reshape(B, T, d)
    return reads, snaps, st, stats
