"""Pallas-ядра вагонетки (route_rot=True, exact route). Математика — як rope.py (dense_read), без рядкових gather:
індекси й межі спанів дістаються масками у VMEM. interpret=True на CPU; на TPU interpret=False.

  route_exact   — пошук top-K якорів по запитах (K ітерацій argmax), без grad;
  bias_forward  — q·k у позиції tpos (для спільного пошуку, де оцінка маршруту іншого шару);
  read          — причинна увага вікно+спани з власним bwd (custom_vjp).
Блоки: BQ запитів на BK ключів, лише причинний трикутник (таблиця пар у SMEM). Накопичення f32."""
from functools import partial
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import os
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32
I32 = jnp.int32


_DIAG_NOSPAN = os.environ.get("ROPE_DIAG_NOSPAN") == "1"


def _interp():
    if os.environ.get("PALLAS_FORCE_TPU") == "1":     # tools/tpu_lower_check: справжній Mosaic-lowering без TPU
        return False
    return jax.default_backend() != "tpu"


def _params(*sem):
    return pltpu.CompilerParams(dimension_semantics=sem)


def _sp(*shape_and_map):
    *shape, index_map = shape_and_map
    return pl.BlockSpec(tuple(shape), index_map)


# ---------------------------------------------------------------- пошук

def _route_kernel(q, kl, lval, lp, av, nx, apos, nxt, tidx, tval, tpos, tsc, st, nxo, *, A, L, K, BM, sc):
    """Без reshape/argmax/стовпцевих записів усередині ядра (Mosaic: «unsupported shape cast»):
    kl (L, A, hd), lval/lp (L, A); ланки — L окремих множень (BM, A) з максимумом у циклі; вихід (BM, K) збирається where."""
    n = pl.program_id(1)
    i = n * BM + jax.lax.broadcasted_iota(I32, (BM, 1), 0)
    qv = q[...]
    sbest = jnp.full((BM, A), -jnp.inf, F32)
    bpos = jnp.zeros((BM, A), I32)
    for l in range(L):                                       # перша з рівних ланок виграє (як argmax)
        sl = jnp.dot(qv, kl[l].T, preferred_element_type=F32) * sc
        sl = jnp.where(lval[l:l + 1, :] != 0, sl, -jnp.inf)
        up = sl > sbest
        bpos = jnp.where(up, lp[l:l + 1, :], bpos)
        sbest = jnp.where(up, sl, sbest)
    sa = jnp.where((av[...] != 0) & (nx[...] <= i), sbest, -jnp.inf)
    a_ids = jax.lax.broadcasted_iota(I32, (BM, A), 1)
    k_ids = jax.lax.broadcasted_iota(I32, (BM, K), 1)
    ap, nt = apos[...], nxt[...]
    o_i = jnp.zeros((BM, K), I32); o_v = o_i; o_p = o_i; o_s = jnp.zeros((BM, K), F32); o_st = o_i; o_nx = o_i
    for kk in range(K):
        val = jnp.max(sa, axis=1, keepdims=True)                                     # (BM, 1)
        top = jnp.min(jnp.where(sa == val, a_ids, A), axis=1, keepdims=True)         # перший максимум
        oh = a_ids == top
        ok = val > -jnp.inf
        pos = jnp.sum(jnp.where(oh, bpos, 0), axis=1, keepdims=True)
        sel = k_ids == kk
        o_i = jnp.where(sel, jnp.where(ok, top, 0), o_i)
        o_v = jnp.where(sel, ok.astype(I32), o_v)
        o_p = jnp.where(sel, jnp.where(ok, pos, 0), o_p)
        o_s = jnp.where(sel, jnp.where(ok, val, 0.0), o_s)
        o_st = jnp.where(sel, jnp.sum(jnp.where(oh, ap, 0), axis=1, keepdims=True), o_st)
        o_nx = jnp.where(sel, jnp.sum(jnp.where(oh, nt, 0), axis=1, keepdims=True), o_nx)
        sa = jnp.where(oh, -jnp.inf, sa)
    tidx[...] = o_i; tval[...] = o_v; tpos[...] = o_p; tsc[...] = o_s; st[...] = o_st; nxo[...] = o_nx


def route_exact(q, kl, lval, lp, avalid, nx, apos, nxt, K, BM, *, interpret=None):
    """q (B,H,T,hd); kl (B,H,A,L,hd); lval, lp (B,A,L); avalid, nx, apos, nxt (B,A). T ділиться на BM.
    → tidx, tval (bool), tpos, tscore, st, nxt_sel — кожне (B,H,T,K)."""
    interpret = _interp() if interpret is None else interpret
    B, H, T, hd = q.shape
    A, L = kl.shape[2], kl.shape[3]
    assert T % BM == 0
    sc = 1.0 / np.sqrt(hd)

    def one(q1, kl1, lv1, lp1, av1, nx1, ap1, nt1):
        klt = jnp.swapaxes(kl1, 1, 2)                       # (H, L, A, hd)
        mid = lambda h, n: (h, n, 0)
        tail = lambda h, n: (h, 0, 0, 0)
        whole = lambda h, n: (0, 0)                      # 1D/2D входи цілими: Mosaic хоче останні 2 виміри блоку = масиву
        kern = partial(_route_kernel, A=A, L=L, K=K, BM=BM, sc=sc)
        o = lambda d: jax.ShapeDtypeStruct((H, T, K), d)
        spec_o = _sp(pl.Squeezed(), BM, K, mid)
        return pl.pallas_call(
            kern, grid=(H, T // BM),
            in_specs=[_sp(pl.Squeezed(), BM, hd, mid), _sp(pl.Squeezed(), L, A, hd, tail),
                      _sp(L, A, whole), _sp(L, A, whole), _sp(1, A, whole), _sp(1, A, whole), _sp(1, A, whole), _sp(1, A, whole)],
            out_specs=[spec_o] * 6, out_shape=[o(I32), o(I32), o(I32), o(F32), o(I32), o(I32)],
            compiler_params=_params("arbitrary", "arbitrary"), interpret=interpret, name="rope_route")(
            q1, klt, lv1.T.astype(I32), lp1.T.astype(I32), av1[None].astype(I32), nx1[None].astype(I32),
            ap1[None].astype(I32), nt1[None].astype(I32))

    ti, tv, tp, ts, st, nn = jax.vmap(one)(q, kl, lval, lp, avalid, nx, apos, nxt)
    return ti, tv.astype(bool), tp, ts, st, nn


# ---------------------------------------------------------------- спільне для тайлів читання

class P(NamedTuple):
    H: int
    T: int
    hd: int
    K: int
    W: int
    BQ: int
    BK: int
    sc: float
    sc_bias: float


def _tile(q, k, st, en, gb, i0, t0, p):
    """→ s (BQ,BK) оцінки f32 з bias, mask (BQ,BK), in_k (BQ,K,BK) i32, i, t."""
    i = i0 + jnp.arange(p.BQ, dtype=I32)
    t = t0 + jnp.arange(p.BK, dtype=I32)
    s = jnp.dot(q, k.T, preferred_element_type=F32) * p.sc
    ii, tt = i[:, None], t[None, :]
    win = (tt <= ii) & (ii - tt < p.W)
    if _DIAG_NOSPAN:                                   # ДІАГНОСТИКА швидкості (tools/rope_kernel_bench): без маски прольотів
        z = jnp.zeros((p.BQ, p.BK), bool)
        return s, win, z, jnp.zeros((p.BQ, p.K, p.BK), I32), i, t
    tk = t[None, None, :]
    ink = ((st[:, :, None] <= tk) & (tk < en[:, :, None])).astype(I32)              # (BQ,K,BK)
    span = ink.sum(1) > 0
    s = s + (gb[:, :, None] * ink.astype(F32)).sum(1)
    return s, win | span, span, ink, i, t


def _pairs(p, kmajor=False):
    """Причинні пари (q-блок, k-блок) разом із first/last для осі-накопичувача."""
    nq = p.T // p.BQ
    pr = [(j, kb) for j in range(nq) for kb in range(((j + 1) * p.BQ - 1) // p.BK + 1)]
    if kmajor:
        pr = sorted(pr, key=lambda x: (x[1], x[0]))
    outer = [x[1] if kmajor else x[0] for x in pr]
    first = [int(i == 0 or outer[i] != outer[i - 1]) for i in range(len(pr))]
    last = [int(i == len(pr) - 1 or outer[i] != outer[i + 1]) for i in range(len(pr))]
    return (np.asarray([x[0] for x in pr], np.int32), np.asarray([x[1] for x in pr], np.int32),
            np.asarray(first, np.int32), np.asarray(last, np.int32))


def _call(kern, name, p, pairs, in_arrs, in_sp, out_shape, out_sp, scratch, interpret):
    pq, pk, fi, la = pairs
    n = len(pq)
    gs = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=4, grid=(p.H, n), in_specs=in_sp, out_specs=out_sp,
                                      scratch_shapes=scratch)
    return pl.pallas_call(kern, grid_spec=gs, out_shape=out_shape, compiler_params=_params("arbitrary", "arbitrary"),
                          interpret=interpret, name=name)(jnp.asarray(pq), jnp.asarray(pk), jnp.asarray(fi),
                                                          jnp.asarray(la), *in_arrs)


# маршрутні індекси блоків: імена аргументів мапи — (h, n, *scalars)
def _mq(p, dim):
    return _sp(pl.Squeezed(), p.BQ, dim, lambda h, n, pq, pk, f, l: (h, pq[n], 0))


def _mk(p, dim):
    return _sp(pl.Squeezed(), p.BK, dim, lambda h, n, pq, pk, f, l: (h, pk[n], 0))


# ---------------------------------------------------------------- forward

def _fwd_kernel(pq, pk, fi, la, q, k, v, st, en, gb, out, m_o, l_o, o32, mass, m_s, l_s, ls_s, o_s, *, p, pdt):
    n = pl.program_id(1)
    j, kb, first, last = pq[n], pk[n], fi[n], la[n]

    @pl.when(first == 1)
    def _():
        m_s[...] = jnp.full_like(m_s, -jnp.inf)
        l_s[...] = jnp.zeros_like(l_s)
        ls_s[...] = jnp.zeros_like(ls_s)
        o_s[...] = jnp.zeros_like(o_s)

    s, mask, span, _, _, _ = _tile(q[...], k[...], st[...], en[...], gb[...], j * p.BQ, kb * p.BK, p)
    s = jnp.where(mask, s, -jnp.inf)
    mb = jnp.max(s, axis=1, keepdims=True)
    mp = m_s[...]
    mn = jnp.maximum(mp, mb)
    safe = jnp.where(mn == -jnp.inf, 0.0, mn)
    alpha = jnp.where(mp == -jnp.inf, 0.0, jnp.exp(mp - safe))
    e = jnp.where(mask, jnp.exp(s - safe), 0.0)
    m_s[...] = mn
    l_s[...] = l_s[...] * alpha + e.sum(1, keepdims=True)
    ls_s[...] = ls_s[...] * alpha + jnp.where(span, e, 0.0).sum(1, keepdims=True)
    o_s[...] = o_s[...] * alpha + jnp.dot(e.astype(pdt), v[...], preferred_element_type=F32)

    @pl.when(last == 1)
    def _():
        o = o_s[...] / l_s[...]
        out[...] = o.astype(out.dtype)
        o32[...] = o
        m_o[...] = m_s[...]
        l_o[...] = l_s[...]
        mass[...] = ls_s[...] / l_s[...]


def _fwd(q, k, v, st, en, gb, p, interpret):
    dt = v.dtype

    def one(q1, k1, v1, st1, en1, gb1):
        o = lambda shp, d: jax.ShapeDtypeStruct(shp, d)
        H, T = p.H, p.T
        return _call(partial(_fwd_kernel, p=p, pdt=dt), "rope_read_fwd", p, _pairs(p), (q1, k1, v1, st1, en1, gb1),
                     [_mq(p, p.hd), _mk(p, p.hd), _mk(p, p.hd), _mq(p, p.K), _mq(p, p.K), _mq(p, p.K)],
                     [o((H, T, p.hd), dt), o((H, T, 1), F32), o((H, T, 1), F32), o((H, T, p.hd), F32), o((H, T, 1), F32)],
                     [_mq(p, p.hd), _mq(p, 1), _mq(p, 1), _mq(p, p.hd), _mq(p, 1)],
                     [pltpu.VMEM((p.BQ, 1), F32), pltpu.VMEM((p.BQ, 1), F32), pltpu.VMEM((p.BQ, 1), F32),
                      pltpu.VMEM((p.BQ, p.hd), F32)], interpret)
    return jax.vmap(one)(q, k, v, st, en, gb)


# ---------------------------------------------------------------- backward

def _weights(q, k, v, st, en, gb, m, l, o32, g, j, kb, p):
    s, mask, _, ink, i, t = _tile(q, k, st, en, gb, j * p.BQ, kb * p.BK, p)
    pr = jnp.where(mask, jnp.exp(s - m) / l, 0.0)
    a = jnp.dot(g.astype(v.dtype), v.T, preferred_element_type=F32)
    e = (g.astype(F32) * o32).sum(-1, keepdims=True)
    return pr, pr * (a - e), ink, t


def _bwd_q_kernel(pq, pk, fi, la, q, k, v, st, en, gb, m, l, o32, g, dq, BB, dq_s, b_s, *, p):
    n = pl.program_id(1)
    j, kb, first, last = pq[n], pk[n], fi[n], la[n]

    @pl.when(first == 1)
    def _():
        dq_s[...] = jnp.zeros_like(dq_s)
        b_s[...] = jnp.zeros_like(b_s)

    _, w, ink, _ = _weights(q[...], k[...], v[...], st[...], en[...], gb[...], m[...], l[...], o32[...], g[...], j, kb, p)
    dq_s[...] += jnp.dot(w.astype(k.dtype), k[...], preferred_element_type=F32) * p.sc
    b_s[...] += (w[:, None, :] * ink.astype(F32)).sum(-1)

    @pl.when(last == 1)
    def _():
        dq[...] = dq_s[...].astype(dq.dtype)
        BB[...] = b_s[...]


def _bwd_k_kernel(pq, pk, fi, la, q, k, v, st, en, gb, m, l, o32, g, BB, tpos, dk, dv, dk_s, dv_s, *, p):
    n = pl.program_id(1)
    j, kb, first, last = pq[n], pk[n], fi[n], la[n]

    @pl.when(first == 1)
    def _():
        dk_s[...] = jnp.zeros_like(dk_s)
        dv_s[...] = jnp.zeros_like(dv_s)

    pr, w, _, t = _weights(q[...], k[...], v[...], st[...], en[...], gb[...], m[...], l[...], o32[...], g[...], j, kb, p)
    dv_s[...] += jnp.dot(pr.T.astype(g.dtype), g[...], preferred_element_type=F32)
    u = (BB[...][:, :, None] * (tpos[...][:, :, None] == t[None, None, :]).astype(F32)).sum(1)
    dk_s[...] += jnp.dot((w * p.sc + u * p.sc_bias).T.astype(q.dtype), q[...], preferred_element_type=F32)

    @pl.when(last == 1)
    def _():
        dk[...] = dk_s[...].astype(dk.dtype)
        dv[...] = dv_s[...].astype(dv.dtype)


def _bias_q_kernel(pq, pk, fi, la, k, BB, tpos, dq, dq_s, *, p):
    n = pl.program_id(1)
    kb, first, last = pk[n], fi[n], la[n]

    @pl.when(first == 1)
    def _():
        dq_s[...] = jnp.zeros_like(dq_s)

    t = kb * p.BK + jnp.arange(p.BK, dtype=I32)
    u = (BB[...][:, :, None] * (tpos[...][:, :, None] == t[None, None, :]).astype(F32)).sum(1)
    dq_s[...] += jnp.dot(u.astype(k.dtype), k[...], preferred_element_type=F32) * p.sc_bias

    @pl.when(last == 1)
    def _():
        dq[...] = dq_s[...]


def _bwd(q, k, v, st, en, gb, tpos, m, l, o32, g, p, interpret, rgrad):
    def one(q1, k1, v1, st1, en1, gb1, tp1, m1, l1, o1, g1):
        s = lambda shp, d: jax.ShapeDtypeStruct(shp, d)
        H, T = p.H, p.T
        ins = (q1, k1, v1, st1, en1, gb1, m1, l1, o1, g1)
        sp = [_mq(p, p.hd), _mk(p, p.hd), _mk(p, p.hd), _mq(p, p.K), _mq(p, p.K), _mq(p, p.K), _mq(p, 1), _mq(p, 1),
              _mq(p, p.hd), _mq(p, p.hd)]
        dq, BB = _call(partial(_bwd_q_kernel, p=p), "rope_read_dq", p, _pairs(p), ins, sp,
                       [s((H, T, p.hd), q1.dtype), s((H, T, p.K), F32)], [_mq(p, p.hd), _mq(p, p.K)],
                       [pltpu.VMEM((p.BQ, p.hd), F32), pltpu.VMEM((p.BQ, p.K), F32)], interpret)
        BB2 = BB if rgrad else jnp.zeros_like(BB)
        dk, dv = _call(partial(_bwd_k_kernel, p=p), "rope_read_dk", p, _pairs(p, True), ins + (BB2, tp1),
                       sp + [_mq(p, p.K), _mq(p, p.K)],
                       [s((H, T, p.hd), k1.dtype), s((H, T, p.hd), v1.dtype)], [_mk(p, p.hd), _mk(p, p.hd)],
                       [pltpu.VMEM((p.BK, p.hd), F32), pltpu.VMEM((p.BK, p.hd), F32)], interpret)
        if rgrad:
            dqb = _call(partial(_bias_q_kernel, p=p), "rope_bias_dq", p, _pairs(p), (k1, BB, tp1),
                        [_mk(p, p.hd), _mq(p, p.K), _mq(p, p.K)], s((H, T, p.hd), F32), _mq(p, p.hd),
                        [pltpu.VMEM((p.BQ, p.hd), F32)], interpret)
            dq = (dq.astype(F32) + dqb).astype(q1.dtype)
        return dq, dk, dv
    return jax.vmap(one)(q, k, v, st, en, gb, tpos, m, l, o32, g)


# ---------------------------------------------------------------- bias (спільний пошук)

def _bias_fwd_kernel(pq, pk, fi, la, q, k, tpos, tval, gb, gb_s, *, p):
    n = pl.program_id(1)
    kb, first, last = pk[n], fi[n], la[n]

    @pl.when(first == 1)
    def _():
        gb_s[...] = jnp.zeros_like(gb_s)

    t = kb * p.BK + jnp.arange(p.BK, dtype=I32)
    s = jnp.dot(q[...], k[...].T, preferred_element_type=F32) * p.sc
    mt = (tpos[...][:, :, None] == t[None, None, :]) & (tval[...][:, :, None] != 0)
    gb_s[...] += (s[:, None, :] * mt.astype(F32)).sum(-1)

    @pl.when(last == 1)
    def _():
        gb[...] = gb_s[...]


def bias_forward(q, k, tpos, tval, p, interpret=None):
    """gb (B,H,T,K) = sc·q[i]·k[tpos[i,k]] при tval, інакше 0 (без множника rgrad_scale)."""
    interpret = _interp() if interpret is None else interpret

    def one(q1, k1, tp1, tv1):
        return _call(partial(_bias_fwd_kernel, p=p), "rope_bias_fwd", p, _pairs(p), (q1, k1, tp1, tv1.astype(I32)),
                     [_mq(p, p.hd), _mk(p, p.hd), _mq(p, p.K), _mq(p, p.K)],
                     jax.ShapeDtypeStruct((p.H, p.T, p.K), F32), _mq(p, p.K),
                     [pltpu.VMEM((p.BQ, p.K), F32)], interpret)
    return jax.vmap(one)(q, k, tpos, tval)


# ---------------------------------------------------------------- читання з власним bwd

@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9))
def read(q, k, v, st, en, gb, tpos, p, interpret, rgrad):
    """→ out (dtype v), mass (B,H,T,1) f32. Вхідні st/en/gb/tpos — сталі для grad; похідна bias — через rgrad у bwd."""
    out, _, _, _, mass = _fwd(q, k, v, st, en, gb, p, interpret)
    return out, lax.stop_gradient(mass)


def _read_fwd(q, k, v, st, en, gb, tpos, p, interpret, rgrad):
    out, m, l, o32, mass = _fwd(q, k, v, st, en, gb, p, interpret)
    return (out, lax.stop_gradient(mass)), (q, k, v, st, en, gb, tpos, m, l, o32)


def _read_bwd(p, interpret, rgrad, res, cot):
    q, k, v, st, en, gb, tpos, m, l, o32 = res
    dq, dk, dv = _bwd(q, k, v, st, en, gb, tpos, m, l, o32, cot[0], p, interpret, rgrad)
    return dq, dk, dv, None, None, None, None


read.defvjp(_read_fwd, _read_bwd)
