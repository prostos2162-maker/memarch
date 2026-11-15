"""Читання троса v3 (Pallas, TPU). Та сама математика, що rope_pallas.read (вікно + слова відкритих якорів,
спільний softmax у f32, rgrad), але дешевша на VPU і з меншою кількістю проходів:

  * маска прольотів — не 2·K порівнянь меж над (BQ,K,BK), а K порівнянь «якір слова == вибраний якір запиту»
    над (BQ,BK): kid[t] = номер якоря-власника слова t (або -2, якщо слово далі span_cap від якоря),
    qid[i,k] = вибраний запитом якір (або -1). Слово t входить у проліт запиту i  <=>  kid[t]==qid[i,k] і i-t >= W.
    Це рівно st <= t < min(nxt, st+cap, i-W+1) зі старого ядра (own[t]==a <=> apos[a] <= t < nxt[a]);
  * більші тайли (типово 1024x1024 замість 256x512) — менше кроків сітки;
  * назад — ОДИН причинний прохід (dq, dk, dv, BB разом; dk/dv накопичуються в резидентному рядку (T,hd) у VMEM)
    + один легкий прохід для rgrad (u·K та uᵀ·Q) замість трьох повних проходів;
  * D = sum(g·o) рахується один раз у XLA, а не в кожному тайлі.

Вибрані якорі одного запиту мають бути різними (route це гарантує: top-K без повторів).
interpret=True на CPU; tools/tpu_lower_check ловить помилки lowering на ПК."""
from functools import partial
import os
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32
I32 = jnp.int32
NEG = -jnp.inf


class P3(NamedTuple):
    T: int
    hd: int
    K: int
    W: int
    BQ: int
    BK: int
    sc: float
    sc_bias: float
    vmem: int = 100 << 20
    mgrad: bool = False          # диференційовна mass (частка троса): штраф нагадувань --remind_use_w; False — як було


def sizes(T, BQ=None, BK=None):
    """Розміри тайлів: ROPE_V3_BQ / ROPE_V3_BK (типово 1024 / 1024), обрізані до T; T має ділитися на обидва."""
    BQ = min(int(os.environ.get("ROPE_V3_BQ", 1024)) if BQ is None else BQ, T)
    BK = min(int(os.environ.get("ROPE_V3_BK", 1024)) if BK is None else BK, T)

    def _fit(n, b):                    # 12.11: короткі рядки етапу 1 (1280, 2304, …) — найбільший тайл ≤ b, що ділить n
        for c in (b, 512, 256, 128):
            if c <= b and n % c == 0:
                return c
        return b
    BQ, BK = _fit(T, BQ), _fit(T, BK)    # для T=8192 лишається 1024/1024
    assert T % BQ == 0 and T % BK == 0, (T, BQ, BK)
    return BQ, BK


def _pairs(p):
    """Причинні пари (q-блок, k-блок), q-блок зовнішній; first/last — межі q-блоку."""
    nq = p.T // p.BQ
    pr = [(j, kb) for j in range(nq) for kb in range(((j + 1) * p.BQ - 1) // p.BK + 1)]
    first = [int(i == 0 or pr[i][0] != pr[i - 1][0]) for i in range(len(pr))]
    last = [int(i == len(pr) - 1 or pr[i][0] != pr[i + 1][0]) for i in range(len(pr))]
    return tuple(jnp.asarray(np.asarray(a, np.int32)) for a in ([x[0] for x in pr], [x[1] for x in pr], first, last))


def _tile(q, k, kid, qids, gbs, j, kb, p):
    """-> sf (BQ,BK) f32 оцінки з маскою (-inf поза вікном і прольотами), ok, win, far, hits[K]."""
    s = jnp.dot(q, k.T, preferred_element_type=F32) * p.sc
    d = (j * p.BQ - kb * p.BK) + lax.broadcasted_iota(I32, (p.BQ, p.BK), 0) - lax.broadcasted_iota(I32, (p.BQ, p.BK), 1)
    win = (d >= 0) & (d < p.W)
    far = d >= p.W
    hits = [qids[kk] == kid for kk in range(p.K)]              # (BQ,1) == (1,BK)
    b = jnp.full((p.BQ, p.BK), NEG, F32)
    for kk in reversed(range(p.K)):                             # якорі різні -> щонайбільше одне влучання
        b = jnp.where(hits[kk], gbs[kk], b)
    sf = jnp.where(win, s, jnp.where(far, s + b, NEG))
    return sf, sf > NEG, win, far, hits


# ---------------------------------------------------------------- forward

def _fwd_kernel(pq, pk, fi, la, q, k, v, kid, *rest, p, pdt):
    K = p.K
    qids, gbs = [r[...] for r in rest[:K]], [r[...] for r in rest[K:2 * K]]
    out, o32, lse, mass, m_s, l_s, ls_s, o_s = rest[2 * K:]
    n = pl.program_id(1)
    j, kb = pq[n], pk[n]

    @pl.when(fi[n] == 1)
    def _():
        m_s[...] = jnp.full_like(m_s, NEG)
        l_s[...] = jnp.zeros_like(l_s)
        ls_s[...] = jnp.zeros_like(ls_s)
        o_s[...] = jnp.zeros_like(o_s)

    sf, ok, win, _, _ = _tile(q[...], k[...], kid[...], qids, gbs, j, kb, p)
    mb = jnp.max(sf, axis=1, keepdims=True)
    mp = m_s[...]
    mn = jnp.maximum(mp, mb)
    safe = jnp.where(mn == NEG, 0.0, mn)
    alpha = jnp.where(mp == NEG, 0.0, jnp.exp(mp - safe))
    e = jnp.where(ok, jnp.exp(sf - safe), 0.0)
    m_s[...] = mn
    l_s[...] = l_s[...] * alpha + e.sum(1, keepdims=True)
    ls_s[...] = ls_s[...] * alpha + jnp.where(win, 0.0, e).sum(1, keepdims=True)
    o_s[...] = o_s[...] * alpha + jnp.dot(e.astype(pdt), v[...], preferred_element_type=F32)

    @pl.when(la[n] == 1)
    def _():
        l = l_s[...]
        o = o_s[...] / l
        out[...] = o.astype(out.dtype)
        o32[...] = o
        lse[...] = m_s[...] + jnp.log(l)
        mass[...] = ls_s[...] / l


def _specs(p):
    qb = lambda w: pl.BlockSpec((None, p.BQ, w), lambda r, n, pq, pk, f, l: (r, pq[n], 0))
    kbk = lambda w: pl.BlockSpec((None, p.BK, w), lambda r, n, pq, pk, f, l: (r, pk[n], 0))
    kidb = pl.BlockSpec((None, 1, p.BK), lambda r, n, pq, pk, f, l: (r, 0, pk[n]))
    row = lambda w: pl.BlockSpec((None, p.T, w), lambda r, n, pq, pk, f, l: (r, 0, 0))
    return qb, kbk, kidb, row


def _cp(p):
    return pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"), vmem_limit_bytes=p.vmem)


def _cols(x):
    """(R,T,K) -> K масивів (R,T,1)."""
    return [x[..., kk:kk + 1] for kk in range(x.shape[-1])]


def _fwd(q, k, v, kid, qid, gb, p, interpret):
    R = q.shape[0]
    qb, kbk, kidb, _ = _specs(p)
    pairs = _pairs(p)
    o = lambda w, d: jax.ShapeDtypeStruct((R, p.T, w), d)
    gs = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4, grid=(R, int(pairs[0].shape[0])),
        in_specs=[qb(p.hd), kbk(p.hd), kbk(p.hd), kidb] + [qb(1)] * (2 * p.K),
        out_specs=[qb(p.hd), qb(p.hd), qb(1), qb(1)],
        scratch_shapes=[pltpu.VMEM((p.BQ, 1), F32)] * 3 + [pltpu.VMEM((p.BQ, p.hd), F32)])
    return pl.pallas_call(
        partial(_fwd_kernel, p=p, pdt=v.dtype), grid_spec=gs,
        out_shape=[o(p.hd, v.dtype), o(p.hd, F32), o(1, F32), o(1, F32)], compiler_params=_cp(p),
        interpret=interpret, name="rope_v3_fwd")(*pairs, q, k, v, kid, *_cols(qid), *_cols(gb))


# ---------------------------------------------------------------- backward A: dq, dk, dv, BB за один прохід

def _bwd_kernel(pq, pk, fi, la, q, k, v, kid, *rest, p):
    K = p.K
    qids, gbs = [r[...] for r in rest[:K]], [r[...] for r in rest[K:2 * K]]
    lse, D, g = rest[2 * K:2 * K + 3]
    o0 = 2 * K + 3
    if p.mgrad:                                                 # котангенс mass (R,T,1): mass = Σ_far p
        gm = rest[o0]
        o0 += 1
    dq, dk, dv = rest[o0:o0 + 3]
    bbs = rest[o0 + 3:o0 + 3 + K]
    dq_s = rest[o0 + 3 + K]
    bb_s = rest[o0 + 4 + K:]
    n = pl.program_id(1)
    j, kb = pq[n], pk[n]

    @pl.when(n == 0)
    def _():
        dk[...] = jnp.zeros_like(dk)
        dv[...] = jnp.zeros_like(dv)

    @pl.when(fi[n] == 1)
    def _():
        dq_s[...] = jnp.zeros_like(dq_s)
        for kk in range(K):
            bb_s[kk][...] = jnp.zeros_like(bb_s[kk])

    qv, kv, vv, gv = q[...], k[...], v[...], g[...]
    sf, ok, _, far, hits = _tile(qv, kv, kid[...], qids, gbs, j, kb, p)
    pr = jnp.where(ok, jnp.exp(sf - lse[...]), 0.0)
    a = jnp.dot(gv, vv.T, preferred_element_type=F32)
    if p.mgrad:                                                 # mass як додатковий канал значень: 1 на словах троса
        a = a + jnp.where(far, gm[...], 0.0)
    w = pr * (a - D[...])
    dq_s[...] += jnp.dot(w.astype(kv.dtype), kv, preferred_element_type=F32) * p.sc
    k0 = pl.multiple_of(kb * p.BK, p.BK)
    dk[pl.ds(k0, p.BK), :] += jnp.dot(w.T.astype(qv.dtype), qv, preferred_element_type=F32) * p.sc
    dv[pl.ds(k0, p.BK), :] += jnp.dot(pr.T.astype(gv.dtype), gv, preferred_element_type=F32)
    wf = jnp.where(far, w, 0.0)
    for kk in range(K):
        bb_s[kk][...] += jnp.where(hits[kk], wf, 0.0).sum(1, keepdims=True)

    @pl.when(la[n] == 1)
    def _():
        dq[...] = dq_s[...]
        for kk in range(K):
            bbs[kk][...] = bb_s[kk][...]


def _bwd(q, k, v, kid, qid, gb, lse, D, g, p, interpret, gm=None):
    R = q.shape[0]
    qb, kbk, kidb, row = _specs(p)
    pairs = _pairs(p)
    o = lambda w: jax.ShapeDtypeStruct((R, p.T, w), F32)
    gs = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4, grid=(R, int(pairs[0].shape[0])),
        in_specs=[qb(p.hd), kbk(p.hd), kbk(p.hd), kidb] + [qb(1)] * (2 * p.K) + [qb(1), qb(1), qb(p.hd)]
        + ([qb(1)] if p.mgrad else []),
        out_specs=[qb(p.hd), row(p.hd), row(p.hd)] + [qb(1)] * p.K,
        scratch_shapes=[pltpu.VMEM((p.BQ, p.hd), F32)] + [pltpu.VMEM((p.BQ, 1), F32)] * p.K)
    res = pl.pallas_call(
        partial(_bwd_kernel, p=p), grid_spec=gs, out_shape=[o(p.hd), o(p.hd), o(p.hd)] + [o(1)] * p.K,
        compiler_params=_cp(p), interpret=interpret, name="rope_v3_bwd")(
        *pairs, q, k, v, kid, *_cols(qid), *_cols(gb), lse, D, g, *([gm] if p.mgrad else []))
    return res[0], res[1], res[2], res[3:]


# ---------------------------------------------------------------- backward B: rgrad (похідна bias через q і k[tpos])

def _bias_kernel(pq, pk, fi, la, q, k, *rest, p):
    K = p.K
    tps, bbs = [r[...] for r in rest[:K]], [r[...] for r in rest[K:2 * K]]
    dq, dk, dq_s = rest[2 * K:]
    n = pl.program_id(1)
    kb = pk[n]

    @pl.when(n == 0)
    def _():
        dk[...] = jnp.zeros_like(dk)

    @pl.when(fi[n] == 1)
    def _():
        dq_s[...] = jnp.zeros_like(dq_s)

    qv, kv = q[...], k[...]
    t = kb * p.BK + lax.broadcasted_iota(I32, (p.BQ, p.BK), 1)
    u = jnp.zeros((p.BQ, p.BK), F32)
    for kk in range(K):
        u = u + jnp.where(tps[kk] == t, bbs[kk], 0.0)
    dq_s[...] += jnp.dot(u.astype(kv.dtype), kv, preferred_element_type=F32) * p.sc_bias
    k0 = pl.multiple_of(kb * p.BK, p.BK)
    dk[pl.ds(k0, p.BK), :] += jnp.dot(u.T.astype(qv.dtype), qv, preferred_element_type=F32) * p.sc_bias

    @pl.when(la[n] == 1)
    def _():
        dq[...] = dq_s[...]


def _bias(q, k, tpos, bbs, p, interpret):
    R = q.shape[0]
    qb, kbk, _, row = _specs(p)
    pairs = _pairs(p)
    o = jax.ShapeDtypeStruct((R, p.T, p.hd), F32)
    gs = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4, grid=(R, int(pairs[0].shape[0])),
        in_specs=[qb(p.hd), kbk(p.hd)] + [qb(1)] * (2 * p.K), out_specs=[qb(p.hd), row(p.hd)],
        scratch_shapes=[pltpu.VMEM((p.BQ, p.hd), F32)])
    return pl.pallas_call(partial(_bias_kernel, p=p), grid_spec=gs, out_shape=[o, o], compiler_params=_cp(p),
                          interpret=interpret, name="rope_v3_bias")(*pairs, q, k, *_cols(tpos), *bbs)


# ---------------------------------------------------------------- читання з власним bwd

def _flat(x):
    return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9))
def read(q, k, v, kid, qid, gb, tpos, p, interpret, rgrad):
    """q,k,v (B,H,T,hd); kid (B,T) i32; qid, tpos (B,H,T,K) i32; gb (B,H,T,K) f32.
    -> out (dtype v), mass (B,H,T,1) f32. kid/qid/gb/tpos — сталі для grad; похідна bias — через rgrad у bwd."""
    return _read_fwd(q, k, v, kid, qid, gb, tpos, p, interpret, rgrad)[0]


def _kid_rows(kid, H):
    B, T = kid.shape
    return jnp.broadcast_to(kid.astype(I32)[:, None, None, :], (B, H, 1, T)).reshape(B * H, 1, T)


def _read_fwd(q, k, v, kid, qid, gb, tpos, p, interpret, rgrad):
    B, H, T, hd = q.shape
    out, o32, lse, mass = _fwd(_flat(q), _flat(k), _flat(v), _kid_rows(kid, H), _flat(qid).astype(I32),
                               _flat(gb).astype(F32), p, interpret)
    res = out.reshape(B, H, T, hd), lax.stop_gradient(mass.reshape(B, H, T, 1))
    return res, (q, k, v, kid, qid, gb, tpos, o32, lse) + ((mass,) if p.mgrad else ())


def _read_bwd(p, interpret, rgrad, res, cot):
    q, k, v, kid, qid, gb, tpos, o32, lse = res[:9]
    B, H, T, hd = q.shape
    g = _flat(cot[0])
    D = (g.astype(F32) * o32).sum(-1, keepdims=True)
    gm = None
    if p.mgrad:                                                 # d mass/d s_j = p_j (far_j − mass): D += gm·mass
        gm = _flat(cot[1]).astype(F32)
        D = D + gm * res[9]
    qf, kf = _flat(q), _flat(k)
    dq, dk, dv, bbs = _bwd(qf, kf, _flat(v), _kid_rows(kid, H), _flat(qid).astype(I32), _flat(gb).astype(F32),
                           lse, D, g.astype(v.dtype), p, interpret, gm)
    if rgrad:
        dqb, dkb = _bias(qf, kf, _flat(tpos).astype(I32), bbs, p, interpret)
        dq, dk = dq + dqb, dk + dkb
    sh = lambda x, like: x.astype(like.dtype).reshape(like.shape)
    return sh(dq, q), sh(dk, k), sh(dv, v), None, None, None, None


read.defvjp(_read_fwd, _read_bwd)


def key_ids(own, apos, span_cap):
    """kid (B,T): номер якоря-власника слова, або -2, якщо слово далі span_cap від свого якоря."""
    B, T = own.shape
    A = apos.shape[1]
    st = jnp.take_along_axis(apos, jnp.clip(own, 0, A - 1), axis=1)
    t = jnp.arange(T, dtype=I32)[None, :]
    return jnp.where((own >= 0) & (own < A) & (t - st < span_cap), own, -2).astype(I32)
