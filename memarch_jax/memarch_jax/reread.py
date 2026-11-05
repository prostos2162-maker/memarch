"""Перечитування за закладкою (порт full2.Reread + FullLM2._reread_sep/_reread_core, REREAD_SEP=1,
REREAD_IDS=1, REC_LEAN=1, REREAD_LEAN, REREAD_ANCH) на JAX.

Ключі параметрів = суфікси torch state_dict модуля `reread.` ('norm.weight', 'query.weight', 'offset',
'halt_v.weight', 'halt_v.bias', 'halt_h.weight', 'recog_r.weight', 'recog_c.weight', 'out.weight').
Таблиця книжки (book.weight (V,128)) передається окремо: при W=1 — цілком (шардування — інший етап).
"""
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from .memory import F32, rmsnorm, linear, normalize, sigmoid, ein, bgather, stable_topk


@dataclass(frozen=True)
class RereadCfg:
    df: int = 64          # лице
    k: int = 4            # top_k карток
    R: int = 4            # max_read токенів за закладкою
    seg: int = 512
    frac: float = 0.25    # REREAD_ANCH
    cdt: Any = jnp.bfloat16
    eb: bool = False      # MEM_EINSUM_BF16 (див. memory.py)


def reread_core(p, hs, ids, table, faces, contents, pos, valid, qpos, rows, cfg: RereadCfg, take=None):
    """Reread.forward. hs (N,S,d) fp32 слова; ids (B,T) int номери слів; table (V,Db);
    faces (N,C,L,f), contents (N,C,d), pos/valid (N,C); qpos (N,S) int; rows (N,) — з якого тексту рядок.
    Повертає (o (N,S,d) cdt, cost — fp32 скаляр)."""
    cdt = cfg.cdt
    N, S, _ = hs.shape
    Tb = ids.shape[1]
    n = rmsnorm(hs, p["norm.weight"])
    q = normalize(linear(n, p["query.weight"], cdt=cdt))
    sc = ein("bsf,bclf->bscl", q, normalize(faces), eb=cfg.eb).max(-1) * float(cfg.df) ** 0.5
    sc = jnp.where(valid[:, None], sc, -1e4)
    k = min(cfg.k, sc.shape[-1])
    top_s, top_j = stable_topk(sc, k)                                                     # (N,S,k)
    w = jax.nn.softmax(top_s.astype(F32), -1).astype(hs.dtype) * (top_s > -1e3)
    mark = bgather(pos, top_j.reshape(N, S * k)).reshape(N, S, k)                         # закладки
    rpos = mark[..., None] + jnp.arange(cfg.R, dtype=mark.dtype)                          # (N,S,k,R)
    readable = rpos < qpos[:, :, None, None]                                              # лише раніші слова
    rpos = jnp.clip(rpos, 0, Tb - 1)
    tok = ids[rows[:, None, None, None], rpos]                                            # (N,S,k,R)
    v = (table[tok] if take is None else take(table, tok)).astype(F32) + p["offset"]   # take — шардована книжка (N,S,k,R,Db)
    hv = linear(v, p["halt_v.weight"], p["halt_v.bias"], cdt=cdt)[..., 0]
    hh = linear(n, p["halt_h.weight"], cdt=cdt)                                           # (N,S,1)
    halt = sigmoid(hv + hh[:, :, None])
    halt = jnp.where(readable, halt, jnp.ones_like(halt))
    halt = jnp.concatenate([halt[..., :-1], jnp.ones_like(halt[..., -1:])], -1)
    stay = 1 - halt[..., :-1]
    surv = [jnp.ones_like(halt[..., 0])]
    for i in range(stay.shape[-1]):                                                       # розгорнутий cumprod
        surv.append(surv[-1] * stay[..., i])
    survive = jnp.stack(surv, -1)
    rd = survive * halt * readable                                                        # dtype halt
    read = (rd[..., None] * v).sum(-2)                                                    # (N,S,k,Db) fp32
    cp = linear(contents, p["recog_c.weight"], cdt=cdt)                                   # REC_LEAN: (N,C,f)
    card_p = bgather(cp, top_j.reshape(N, S * k)).reshape(N, S, k, -1)
    rr = normalize(linear(read, p["recog_r.weight"], cdt=cdt))
    recog = sigmoid((rr * normalize(card_p)).sum(-1, keepdims=True) * 4)
    got = (w[..., None] * recog * read).sum(-2)
    cost = (w * (survive * readable).sum(-1)).sum(-1).mean()
    return linear(got, p["out.weight"], cdt=cdt), cost.astype(F32)


def reread_sep(p, h, s, ids, table, snaps, cfg: RereadCfg, take=None):
    """_reread_sep з REREAD_ANCH: h (B,T,d) залишок після final; s (B,T) здивування (s[:,0]=1e4), без градієнта;
    ids (B,T) номери слів; snaps — стос знімків memory_pass {faces (n,B,...), contents, pos, valid}.
    Рядки — шматок-major (j*B+b). Повертає (o (B,T,d) cdt, cost)."""
    B, T, d = h.shape
    seg = cfg.seg
    assert T % seg == 0, (T, seg)
    n = T // seg
    nB = n * B
    hs = h.astype(F32).reshape(B, n, seg, d).swapaxes(0, 1).reshape(nB, seg, d)
    fl = lambda t: t.reshape((nB,) + t.shape[2:])
    faces, contents, pos, valid = fl(snaps["faces"]), fl(snaps["contents"]), fl(snaps["pos"]), fl(snaps["valid"])
    qpos = jnp.repeat(jnp.arange(n, dtype=jnp.int32)[:, None] * seg + jnp.arange(seg, dtype=jnp.int32), B, 0)
    rows = jnp.tile(jnp.arange(B, dtype=jnp.int32), n)
    if cfg.frac:
        kk = max(1, int(round(seg * cfg.frac)))
        sp = lax.stop_gradient(s.astype(F32)).reshape(B, n, seg).swapaxes(0, 1).reshape(nB, seg)
        sel = jnp.sort(stable_topk(sp, kk)[1], -1)                                        # (nB,kk)
        hs_k = jnp.take_along_axis(hs, sel[..., None], 1)                                 # точне fp32-збирання
        qpos_k = jnp.take_along_axis(qpos, sel, 1)
        o_k, cost = reread_core(p, hs_k, ids, table, faces, contents, pos, valid, qpos_k, rows, cfg, take)
        o = jnp.zeros((nB, seg, d), o_k.dtype).at[jnp.arange(nB)[:, None], sel].set(o_k)  # розкладання на місця
    else:
        o, cost = reread_core(p, hs, ids, table, faces, contents, pos, valid, qpos, rows, cfg, take)
    o = o.reshape(n, B, seg, d).swapaxes(0, 1).reshape(B, T, d)
    return o, cost
