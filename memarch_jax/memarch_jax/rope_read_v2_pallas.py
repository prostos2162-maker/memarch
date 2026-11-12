"""Rope read v2 — Pallas forward with gathers INSIDE the kernel (no XLA row gathers / unpermute).

XLA prep (sort_plan) is a single multi-operand lax.sort per row: real entries keyed by 2*bucket, bucket padding made of
dummy entries keyed 2*bucket+1 (exactly pcnt-cnt of them), the rest dropped to the end. Result: a bucket-padded entry
layout (Npad,) of (qid, st, en, gb) with each bucket starting at a chunk boundary — no gather, no scatter.

SPAN_FWD: grid (R, NCH). q row (T,hd) resident in VMEM; per-chunk SMEM blocks qid/st/en/gb (E,); fori_loop copies
q rows into a (E,hd) scratch; two key blocks lo/hi; per-entry (m,l,o); fori_loop online-LSE merge into resident
per-query accumulators o_acc (T,hd) f32, m_acc/l_acc (T,128). WIN_FWD: band kernel. Combine: elementwise XLA.
"""
from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .rope_read_v2 import plan_sizes

F32 = jnp.float32
I32 = jnp.int32


def _dedupe(st, en, gb, K):
    """(T,K) -> valid (dups removed), gb_ded (dup gb folded into the first identical (st,en))."""
    v0 = en > st
    same = (st[:, :, None] == st[:, None, :]) & (en[:, :, None] == en[:, None, :]) & v0[:, :, None] & v0[:, None, :]
    le = jnp.arange(K)[None, :] <= jnp.arange(K)[:, None]
    owner = jnp.where(v0, jnp.argmax(same & le[None], axis=-1), jnp.arange(K)[None, :]).astype(I32)
    gb_ded = jnp.einsum("ik,ikj->ij", gb.astype(F32), jax.nn.one_hot(owner, K, dtype=F32))
    return v0 & (owner == jnp.arange(K)[None, :]), gb_ded


def sort_plan(st, en, gb, T, K, BW=256, E=256, with_eid=False):
    """One row, (T,K) inputs. -> qid, st_p, en_p (Npad,) i32, gb_p (Npad,) f32, cb (NCH,) i32, nused () i32.
    Padding: qid=-1, st=T, en=0, gb=0. Chunks c >= nused repeat the last cb (no DMA) and are skipped."""
    nb, N, NCH, Npad = plan_sizes(T, K, BW, E)
    st, en = st.astype(I32), en.astype(I32)
    valid, gb_ded = _dedupe(st, en, gb, K)
    stf, enf, vf = st.reshape(N), en.reshape(N), valid.reshape(N)
    bkt = jnp.where(vf, stf // BW, nb)
    cnt = jax.ops.segment_sum(vf.astype(I32), bkt, nb + 1)[:nb]
    pcnt = (cnt + E - 1) // E * E
    poff = jnp.concatenate([jnp.zeros(1, I32), jnp.cumsum(pcnt)[:-1]]).astype(I32)
    nused = (jnp.sum(pcnt) // E).astype(I32)
    DROP = 2 * nb + 2
    nd = nb * (E - 1)
    tail_n = Npad - N - nd
    db = jnp.repeat(jnp.arange(nb, dtype=I32), E - 1)
    dj = jnp.tile(jnp.arange(E - 1, dtype=I32), nb)
    key = jnp.concatenate([jnp.where(vf, 2 * bkt, DROP), jnp.where(dj < (pcnt - cnt)[db], 2 * db + 1, DROP),
                           jnp.full(tail_n, DROP, I32)]).astype(I32)
    pad = lambda a, fill, dt: jnp.concatenate([a.astype(dt), jnp.full(nd + tail_n, fill, dt)])
    ops = (key, pad(jnp.where(vf, jnp.arange(N, dtype=I32) // K, -1), -1, I32),
           pad(jnp.where(vf, stf, T), T, I32), pad(jnp.where(vf, enf, 0), 0, I32),
           pad(jnp.where(vf, gb_ded.reshape(N), 0.0), 0.0, F32))
    if with_eid:                                  # entry id i*K+k (unique scatter slot), -1 for padding
        ops = ops + (pad(jnp.where(vf, jnp.arange(N, dtype=I32), -1), -1, I32),)
    srt = lax.sort(ops, num_keys=1, is_stable=True)
    qid, st_p, en_p, gb_p = srt[1:5]
    tail = jnp.arange(Npad) >= nused * E
    qid = jnp.where(tail, -1, qid)
    st_p = jnp.where(tail, T, st_p)
    en_p = jnp.where(tail, 0, en_p)
    gb_p = jnp.where(tail, 0.0, gb_p)
    c0 = jnp.arange(NCH, dtype=I32) * E
    cb = jnp.clip(jnp.searchsorted(poff, c0, side="right").astype(I32) - 1, 0, nb - 1)
    cb = jnp.where(jnp.arange(NCH) < nused, cb, cb[jnp.maximum(nused - 1, 0)])
    if with_eid:
        return qid, st_p, en_p, gb_p, cb, nused, jnp.where(tail, -1, srt[5])
    return qid, st_p, en_p, gb_p, cb, nused


# ---------------------------------------------------------------- SPAN_FWD

def _sub(dt):
    """Sublane tile height: 8 for 32-bit, 16 for bf16 (packed (16,128) tiles), 32 for 8-bit."""
    return 8 * (4 // jnp.dtype(dt).itemsize)


def _pick_row(tile, j):
    """Row j (dynamic, < tile rows) of a (S,w) value -> (1,w), via iota mask + sum (one nonzero row: exact)."""
    sel = lax.broadcasted_iota(I32, tile.shape, 0) == j
    return jnp.sum(jnp.where(sel, tile.astype(F32), 0.0), axis=0, keepdims=True)


def _groups(body, n, unroll):
    """for g in range(n): body(g) as a fori_loop with a MANUAL static unroll (Pallas ignores fori_loop(unroll=))."""
    u = max(d for d in range(1, max(unroll, 1) + 1) if n % d == 0)

    def step(gg, carry):
        for k in range(u):
            body(gg * u + k)
        return carry
    lax.fori_loop(0, n // u, step, 0)


def _span_fwd_kernel(cb_s, nu_s, qid, stp, enp, gbp, q, klo, khi, vlo, vhi, o_acc, m_acc, l_acc,
                     qs, sts, ens, gbs, oe, me, le, *, E, BW, sc, pdt, unroll=1):
    """Loop variant. Every dynamic sublane access is tile-aligned (pl.multiple_of): q/qs at _sub(q.dtype) rows,
    f32 accumulators at 8 rows. Gather: aligned (S,hd) tile + iota select; writes in groups of G rows at once;
    merge: per entry an aligned 8-row read-modify-write of o/m/l with a one-row where mask (sequential)."""
    r, c = pl.program_id(0), pl.program_id(1)
    SQ = _sub(q.dtype)                    # q tile rows (16 for bf16, 8 for f32)
    G = max(SQ, 8)                        # entries per group (scratch writes are G-row aligned)
    assert E % G == 0

    @pl.when(c == 0)
    def _():
        o_acc[...] = jnp.zeros_like(o_acc)
        m_acc[...] = jnp.full_like(m_acc, -jnp.inf)
        l_acc[...] = jnp.zeros_like(l_acc)

    @pl.when(c < nu_s[r])
    def _():
        cb = cb_s[r, c]
        rowsG = lax.broadcasted_iota(I32, (G, 1), 0)

        def cp1(g):
            e0 = pl.multiple_of(g * G, G)
            qg = jnp.zeros((G, qs.shape[1]), F32)
            sg = jnp.zeros((G, 128), I32)
            eg = jnp.zeros((G, 128), I32)
            bg = jnp.zeros((G, 128), F32)
            for j in range(G):                                    # static: assemble G rows, then one aligned store
                i = jnp.maximum(qid[e0 + j], 0)
                base = pl.multiple_of((i // SQ) * SQ, SQ)
                row = _pick_row(q[pl.ds(base, SQ), :], i - base)  # (1,hd) f32
                hit = rowsG == j
                qg = jnp.where(hit, row, qg)
                sg = jnp.where(hit, stp[e0 + j], sg)
                eg = jnp.where(hit, enp[e0 + j], eg)
                bg = jnp.where(hit, gbp[e0 + j], bg)
            qs[pl.ds(e0, G), :] = qg.astype(qs.dtype)
            sts[pl.ds(e0, G), :] = sg
            ens[pl.ds(e0, G), :] = eg
            gbs[pl.ds(e0, G), :] = bg
        _groups(cp1, E // G, unroll)
        qe = qs[...].astype(klo.dtype)
        st_c, en_c, gb_c = sts[:, 0:1], ens[:, 0:1], gbs[:, 0:1]
        t_lo = cb * BW + lax.broadcasted_iota(I32, (E, BW), 1)
        t_hi = t_lo + BW                         # clamped hi block at cb = nb-1: t >= T >= en -> fully masked
        mk_lo = (t_lo >= st_c) & (t_lo < en_c)
        mk_hi = (t_hi >= st_c) & (t_hi < en_c)
        s_lo = jnp.dot(qe, klo[...].T, preferred_element_type=F32) * sc + gb_c
        s_hi = jnp.dot(qe, khi[...].T, preferred_element_type=F32) * sc + gb_c
        m = jnp.maximum(jnp.max(jnp.where(mk_lo, s_lo, -jnp.inf), 1, keepdims=True),
                        jnp.max(jnp.where(mk_hi, s_hi, -jnp.inf), 1, keepdims=True))
        ms = jnp.where(m == -jnp.inf, 0.0, m)
        p_lo = jnp.where(mk_lo, jnp.exp(s_lo - ms), 0.0)
        p_hi = jnp.where(mk_hi, jnp.exp(s_hi - ms), 0.0)
        oe[...] = (jnp.dot(p_lo.astype(pdt), vlo[...], preferred_element_type=F32)
                   + jnp.dot(p_hi.astype(pdt), vhi[...], preferred_element_type=F32))
        me[...] = jnp.broadcast_to(m, me.shape)
        le[...] = jnp.broadcast_to(p_lo.sum(1, keepdims=True) + p_hi.sum(1, keepdims=True), le.shape)
        rows8 = lax.broadcasted_iota(I32, (8, 1), 0)

        def mg1(g):
            e0 = pl.multiple_of(g * G, G)
            og, mgt, lgt = oe[pl.ds(e0, G), :], me[pl.ds(e0, G), :], le[pl.ds(e0, G), :]
            for j in range(G):                                    # static, sequential (same tile may repeat)
                i = qid[e0 + j]

                @pl.when(i >= 0)
                def _():
                    m1, l1, o1 = mgt[j:j + 1, :], lgt[j:j + 1, :], og[j:j + 1, :]
                    base = pl.multiple_of((i // 8) * 8, 8)
                    hit = rows8 == (i - base)                      # (8,1)
                    m0 = m_acc[pl.ds(base, 8), :]
                    l0 = l_acc[pl.ds(base, 8), :]
                    o0 = o_acc[pl.ds(base, 8), :]
                    mn = jnp.maximum(m0, m1)
                    mns = jnp.where(mn == -jnp.inf, 0.0, mn)
                    a = jnp.where(m0 == -jnp.inf, 0.0, jnp.exp(m0 - mns))
                    b = jnp.where(l1 > 0, jnp.exp(m1 - mns), 0.0)
                    m_acc[pl.ds(base, 8), :] = jnp.where(hit, mn, m0)
                    l_acc[pl.ds(base, 8), :] = jnp.where(hit, a * l0 + b * l1, l0)
                    o_acc[pl.ds(base, 8), :] = jnp.where(hit, a[:, 0:1] * o0 + b[:, 0:1] * o1, o0)
        _groups(mg1, E // G, unroll)


NEG = -1e30
HI = lax.Precision.HIGHEST


def _col_to_row(x):
    """(N,1) f32 column -> (1,N) row via an A @ B.T dot (exact at HIGHEST: one x*1 term per output). x finite."""
    n = x.shape[0]
    xb = jnp.where(lax.broadcasted_iota(I32, (n, 128), 1) == 0, x, 0.0)
    e0 = (lax.broadcasted_iota(I32, (8, 128), 1) == 0).astype(F32)
    return lax.dot_general(e0, xb, (((1,), (1,)), ((), ())), precision=HI, preferred_element_type=F32)[0:1, :]


def _scatter(oht, x):
    """(T,E) bf16 one-hot @ (E,w) f32 values with a bf16 hi/lo split (~2^-16 relative), f32 accumulate."""
    hi = x.astype(jnp.bfloat16)
    lo = (x - hi.astype(F32)).astype(jnp.bfloat16)
    return jnp.dot(oht, hi, preferred_element_type=F32) + jnp.dot(oht, lo, preferred_element_type=F32)


def _span_fwd_onehot_kernel(cb_s, nu_s, qcol, qrow, stc, enc, gbc, q, klo, khi, vlo, vhi, o_acc, m_acc, l_acc,
                            *, E, BW, T, sc, pdt):
    r, c = pl.program_id(0), pl.program_id(1)

    @pl.when(c == 0)
    def _():
        o_acc[...] = jnp.zeros_like(o_acc)
        m_acc[...] = jnp.full_like(m_acc, -jnp.inf)
        l_acc[...] = jnp.zeros_like(l_acc)

    @pl.when(c < nu_s[r])
    def _():
        cb = cb_s[r, c]
        qc, qr = qcol[...], qrow[...]                                    # (E,1), (1,E); padding = -1
        oh = qc == lax.broadcasted_iota(I32, (E, T), 1)                  # (E,T)
        oht = lax.broadcasted_iota(I32, (T, E), 0) == qr                 # (T,E)
        if q.dtype == F32:
            qe = jnp.dot(oh.astype(F32), q[...], precision=HI, preferred_element_type=F32)
        else:                                                            # exact: one nonzero term per row
            qe = jnp.dot(oh.astype(q.dtype), q[...], preferred_element_type=F32).astype(q.dtype)
        st_c, en_c, gb_c = stc[...], enc[...], gbc[...]
        t_lo = cb * BW + lax.broadcasted_iota(I32, (E, BW), 1)
        t_hi = t_lo + BW
        mk_lo = (t_lo >= st_c) & (t_lo < en_c)
        mk_hi = (t_hi >= st_c) & (t_hi < en_c)
        s_lo = jnp.dot(qe, klo[...].T, preferred_element_type=F32) * sc + gb_c
        s_hi = jnp.dot(qe, khi[...].T, preferred_element_type=F32) * sc + gb_c
        m = jnp.maximum(jnp.max(jnp.where(mk_lo, s_lo, -jnp.inf), 1, keepdims=True),
                        jnp.max(jnp.where(mk_hi, s_hi, -jnp.inf), 1, keepdims=True))
        ms = jnp.where(m == -jnp.inf, 0.0, m)
        p_lo = jnp.where(mk_lo, jnp.exp(s_lo - ms), 0.0)
        p_hi = jnp.where(mk_hi, jnp.exp(s_hi - ms), 0.0)
        o = (jnp.dot(p_lo.astype(pdt), vlo[...], preferred_element_type=F32)
             + jnp.dot(p_hi.astype(pdt), vhi[...], preferred_element_type=F32))   # (E,hd), relative to m
        l = p_lo.sum(1, keepdims=True) + p_hi.sum(1, keepdims=True)                  # (E,1)
        has = l > 0
        mf = jnp.where(has, m, NEG)
        # per-query reference c_q = max(m_old, max of this chunk's entries of that query): exact with duplicates
        cmax = jnp.max(jnp.where(oht, _col_to_row(mf), NEG), 1, keepdims=True)        # (T,1)
        mold = m_acc[:, 0:1]
        moldf = jnp.where(mold == -jnp.inf, NEG, mold)
        ccol = jnp.maximum(moldf, cmax)                                                # (T,1)
        ce = jnp.max(jnp.where(oh, _col_to_row(ccol), NEG), 1, keepdims=True)         # (E,1) = c[qid]
        b = jnp.where(has, jnp.exp(mf - ce), 0.0)                                      # <= 1
        ohtb = oht.astype(jnp.bfloat16)
        o_add = _scatter(ohtb, b * o)
        l_add = _scatter(ohtb, jnp.broadcast_to(b * l, (E, 128)))
        a = jnp.where(mold == -jnp.inf, 0.0, jnp.exp(moldf - ccol))                    # (T,1)
        o_acc[...] = a * o_acc[...] + o_add
        l_acc[...] = a * l_acc[...] + l_add
        m_acc[...] = jnp.broadcast_to(jnp.where(cmax > 0.5 * NEG, ccol, mold), m_acc.shape)


def span_fwd(q, k, v, qid, stp, enp, gbp, cb, nused, *, BW, E, sc, interpret, mode="loop", unroll=1,
             qf32=False, vmem=100 << 20):
    """q,k,v (R,T,hd); qid/stp/enp/gbp (R,Npad); cb (R,NCH); nused (R,).
    mode "loop": SMEM scalars + fori_loop row gather / per-entry LSE merge (unroll = loop unroll).
    qf32 (loop only): keep q and its scratch in f32.
    mode "onehot": MXU one-hot gather (E,T)@q and transposed one-hot scatter (T,E)@values, exact LSE merge.
    -> o (R,T,hd) f32 unnormalized, m, l (R,T,128) lane-broadcast (m=-inf, l=0 where no span key)."""
    R, T, hd = q.shape
    Npad = qid.shape[1]
    NCH, nb = Npad // E, T // BW
    kb = lambda d: pl.BlockSpec((None, BW, hd), lambda r, c, cbs, nus: (r, jnp.minimum(cbs[r, c] + d, nb - 1), 0))
    row = lambda w: pl.BlockSpec((None, T, w), lambda r, c, cbs, nus: (r, 0, 0))
    outs = [jax.ShapeDtypeStruct((R, T, hd), F32), jax.ShapeDtypeStruct((R, T, 128), F32),
            jax.ShapeDtypeStruct((R, T, 128), F32)]
    cp = pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"), vmem_limit_bytes=vmem)
    kv = [row(hd), kb(0), kb(1), kb(0), kb(1)]
    if mode == "loop":
        if qf32:                                   # f32 q row + scratch (8-row tiles instead of packed bf16)
            q = q.astype(F32)
            kv = [row(hd), kb(0), kb(1), kb(0), kb(1)]
        resh = lambda a: a.reshape(R * NCH, 1, E)
        smem = lambda: pl.BlockSpec((None, None, E), lambda r, c, cbs, nus: (r * NCH + c, 0, 0),
                                    memory_space=pltpu.SMEM)
        gs = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2, grid=(R, NCH), in_specs=[smem()] * 4 + kv, out_specs=[row(hd), row(128), row(128)],
            scratch_shapes=[pltpu.VMEM((E, hd), q.dtype), pltpu.VMEM((E, 128), I32), pltpu.VMEM((E, 128), I32),
                            pltpu.VMEM((E, 128), F32), pltpu.VMEM((E, hd), F32), pltpu.VMEM((E, 128), F32),
                            pltpu.VMEM((E, 128), F32)])
        kern = partial(_span_fwd_kernel, E=E, BW=BW, sc=sc, pdt=v.dtype, unroll=unroll)
        ins = (resh(qid), resh(stp), resh(enp), resh(gbp))
    elif mode == "onehot":
        col = pl.BlockSpec((None, E, 1), lambda r, c, cbs, nus: (r, c, 0))
        rowE = pl.BlockSpec((None, 1, E), lambda r, c, cbs, nus: (r * NCH + c, 0, 0))
        gs = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2, grid=(R, NCH), in_specs=[col, rowE, col, col, col] + kv,
            out_specs=[row(hd), row(128), row(128)])
        kern = partial(_span_fwd_onehot_kernel, E=E, BW=BW, T=T, sc=sc, pdt=v.dtype)
        c1 = lambda a: a.reshape(R, Npad, 1)
        ins = (c1(qid), qid.reshape(R * NCH, 1, E), c1(stp), c1(enp), c1(gbp))
    else:
        raise ValueError(mode)
    return pl.pallas_call(kern, grid_spec=gs, out_shape=outs, compiler_params=cp, interpret=interpret,
                          name=f"rope_v2_span_fwd_{mode}")(cb, nused, *ins, q, k, k, v, v)


# ---------------------------------------------------------------- WIN_FWD

def _win_fwd_kernel(q, kp, kc, vp, vc, o, m_o, l_o, *, BQ, W, sc, pdt):
    j = pl.program_id(1)
    i = lax.broadcasted_iota(I32, (BQ, BQ), 0)
    t = lax.broadcasted_iota(I32, (BQ, BQ), 1)
    okc = (i - t >= 0) & (i - t < W)                       # same block
    okp = (i - t + BQ < W) & (j > 0)                       # previous block: t' = t - BQ
    qv = q[...]
    sp = jnp.dot(qv, kp[...].T, preferred_element_type=F32) * sc
    scur = jnp.dot(qv, kc[...].T, preferred_element_type=F32) * sc
    m = jnp.maximum(jnp.max(jnp.where(okp, sp, -jnp.inf), 1, keepdims=True),
                    jnp.max(jnp.where(okc, scur, -jnp.inf), 1, keepdims=True))   # finite: t = i always in window
    pp = jnp.where(okp, jnp.exp(sp - m), 0.0)
    pc = jnp.where(okc, jnp.exp(scur - m), 0.0)
    o[...] = (jnp.dot(pp.astype(pdt), vp[...], preferred_element_type=F32)
              + jnp.dot(pc.astype(pdt), vc[...], preferred_element_type=F32))
    m_o[...] = jnp.broadcast_to(m, m_o.shape)
    l_o[...] = jnp.broadcast_to(pp.sum(1, keepdims=True) + pc.sum(1, keepdims=True), l_o.shape)


def win_fwd(q, k, v, *, W, sc, interpret, BQ=256):
    R, T, hd = q.shape
    assert W <= BQ and T % BQ == 0
    blk = lambda w, f: pl.BlockSpec((None, BQ, w), f)
    cur = lambda r, j: (r, j, 0)
    prv = lambda r, j: (r, jnp.maximum(j - 1, 0), 0)
    return pl.pallas_call(
        partial(_win_fwd_kernel, BQ=BQ, W=W, sc=sc, pdt=v.dtype), grid=(R, T // BQ),
        in_specs=[blk(hd, cur), blk(hd, prv), blk(hd, cur), blk(hd, prv), blk(hd, cur)],
        out_specs=[blk(hd, cur), blk(128, cur), blk(128, cur)],
        out_shape=[jax.ShapeDtypeStruct((R, T, hd), F32), jax.ShapeDtypeStruct((R, T, 128), F32),
                   jax.ShapeDtypeStruct((R, T, 128), F32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary")),
        interpret=interpret, name="rope_v2_win_fwd")(q, k, k, v, v)


# ---------------------------------------------------------------- wrapper (forward only)

def plan_rows(st, en, gb, T, K, BW=256, E=256):
    """(R,T,K) -> sort_plan per row."""
    return jax.vmap(lambda a, b, c: sort_plan(a, b, c, T, K, BW, E))(st, en, gb)


def combine(o_s, m_s, l_s, o_w, m_w, l_w):
    """Elementwise LSE merge of span and window partials. -> out f32 (R,T,hd), mass (R,T,1)."""
    m_s, l_s, m_w, l_w = m_s[..., :1], l_s[..., :1], m_w[..., :1], l_w[..., :1]
    M = jnp.maximum(m_w, m_s)
    ew = jnp.exp(m_w - M)
    has = l_s > 0
    es = jnp.where(has, jnp.exp(jnp.where(has, m_s, 0.0) - M), 0.0)
    Lt = ew * l_w + es * l_s
    return (ew * o_w + es * o_s) / Lt, es * l_s / Lt


def read_fwd(q, k, v, st, en, gb, tpos, p, interpret=None, rgrad=True, BW=256, E=256, parts=False, mode="loop",
             unroll=1, qf32=False):
    """Same inputs/outputs as rope_pallas.read, forward only (no custom_vjp yet). rgrad/tpos unused in fwd."""
    from .rope_pallas import _interp
    interpret = _interp() if interpret is None else interpret
    B, H, T, hd = q.shape
    R = B * H
    assert T % BW == 0
    f = lambda a: a.reshape((R,) + a.shape[2:])
    st, en, gb = (lax.stop_gradient(f(x)) for x in (st, en, gb))
    qid, stp, enp, gbp, cb, nused = plan_rows(st, en, gb, T, p.K, BW, E)
    qr, kr, vr = f(q), f(k), f(v)
    o_s, m_s, l_s = span_fwd(qr, kr, vr, qid, stp, enp, gbp, cb, nused, BW=BW, E=E, sc=p.sc, interpret=interpret,
                             mode=mode, unroll=unroll, qf32=qf32)
    o_w, m_w, l_w = win_fwd(qr, kr, vr, W=p.W, sc=p.sc, interpret=interpret)
    out, mass = combine(o_s, m_s, l_s, o_w, m_w, l_w)
    mass = lax.stop_gradient(mass)
    res = out.astype(v.dtype).reshape(B, H, T, hd), mass.reshape(B, H, T, 1)
    if parts:
        return res, (qid, cb, nused)
    return res


# =====================================================================================================================
# "sc" path: SparseCore gathers q_e and scatters per-entry results; TensorCore only runs the dense chunk attention.
#   sort_plan(with_eid) -> SC gather q_e (R,Npad,hd) -> SPAN_DENSE (TC) -> per-entry row [o | m, l] (R,Npad,hd+128)
#   -> SC scatter to the unique slot eid = qid*K + k  (no add, no collisions) -> dense XLA LSE merge over K + window.

def _span_dense_kernel(cb_s, nu_s, qe_ref, stc, enc, gbc, klo, khi, vlo, vhi, out, *, E, BW, sc, hd, pdt):
    r, c = pl.program_id(0), pl.program_id(1)

    @pl.when(c < nu_s[r])
    def _():
        cb = cb_s[r, c]
        qe = qe_ref[...]
        st_c, en_c, gb_c = stc[...], enc[...], gbc[...]
        t_lo = cb * BW + lax.broadcasted_iota(I32, (E, BW), 1)
        t_hi = t_lo + BW
        mk_lo = (t_lo >= st_c) & (t_lo < en_c)
        mk_hi = (t_hi >= st_c) & (t_hi < en_c)
        s_lo = jnp.dot(qe, klo[...].T, preferred_element_type=F32) * sc + gb_c
        s_hi = jnp.dot(qe, khi[...].T, preferred_element_type=F32) * sc + gb_c
        s_lo = jnp.where(mk_lo, s_lo, -jnp.inf)           # also kills NaN/garbage q rows of padding entries
        s_hi = jnp.where(mk_hi, s_hi, -jnp.inf)
        m = jnp.maximum(jnp.max(s_lo, 1, keepdims=True), jnp.max(s_hi, 1, keepdims=True))
        ms = jnp.where(m == -jnp.inf, 0.0, m)
        p_lo = jnp.where(mk_lo, jnp.exp(s_lo - ms), 0.0)
        p_hi = jnp.where(mk_hi, jnp.exp(s_hi - ms), 0.0)
        out[:, :hd] = (jnp.dot(p_lo.astype(pdt), vlo[...], preferred_element_type=F32)
                       + jnp.dot(p_hi.astype(pdt), vhi[...], preferred_element_type=F32))
        l = p_lo.sum(1, keepdims=True) + p_hi.sum(1, keepdims=True)
        lane = lax.broadcasted_iota(I32, (E, 128), 1)
        out[:, hd:] = jnp.where(lane == 0, m, jnp.where(lane == 1, l, 0.0))


def span_dense(q_e, k, v, stp, enp, gbp, cb, nused, *, BW, E, sc, interpret, vmem=64 << 20):
    """q_e (R,Npad,hd) gathered entries; k,v (R,T,hd). -> (R,Npad,hd+128) f32: [:hd] o (unnormalized),
    [hd] m, [hd+1] l. Rows of chunks >= nused are unspecified (padding only)."""
    R, Npad, hd = q_e.shape
    T = k.shape[1]
    NCH, nb = Npad // E, T // BW
    ent = lambda w: pl.BlockSpec((None, E, w), lambda r, c, cbs, nus: (r, c, 0))
    kb = lambda d: pl.BlockSpec((None, BW, hd), lambda r, c, cbs, nus: (r, jnp.minimum(cbs[r, c] + d, nb - 1), 0))
    gs = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2, grid=(R, NCH),
        in_specs=[ent(hd), ent(1), ent(1), ent(1), kb(0), kb(1), kb(0), kb(1)], out_specs=ent(hd + 128))
    c1 = lambda a: a.reshape(R, Npad, 1)
    return pl.pallas_call(
        partial(_span_dense_kernel, E=E, BW=BW, sc=sc, hd=hd, pdt=v.dtype), grid_spec=gs,
        out_shape=jax.ShapeDtypeStruct((R, Npad, hd + 128), F32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"), vmem_limit_bytes=vmem),
        interpret=interpret, name="rope_v2_span_dense")(cb, nused, q_e, c1(stp), c1(enp), c1(gbp), k, k, v, v)


def combine_k(Y, valid, o_w, m_w, l_w, hd):
    """Y (R,T,K,hd+128) scattered per-entry rows (unwritten slots = garbage, masked by valid (R,T,K)).
    Exact LSE merge over the K entries and the window. -> out (R,T,hd) f32, mass (R,T,1)."""
    vm = valid[..., None]
    o_k = jnp.where(vm, Y[..., :hd], 0.0)
    m_k = jnp.where(valid, Y[..., hd], -jnp.inf)
    l_k = jnp.where(valid, Y[..., hd + 1], 0.0)
    m_w, l_w = m_w[..., 0], l_w[..., 0]
    M = jnp.maximum(m_w, jnp.max(m_k, -1))
    ew = jnp.exp(m_w - M)
    has = l_k > 0
    ek = jnp.where(has, jnp.exp(jnp.where(has, m_k, 0.0) - M[..., None]), 0.0)
    a_k = ek * l_k
    Lt = ew * l_w + a_k.sum(-1)
    out = (ew[..., None] * o_w + (ek[..., None] * o_k).sum(2)) / Lt[..., None]
    return out, (a_k.sum(-1) / Lt)[..., None]


def read_fwd_sc(q, k, v, st, en, gb, tpos, p, interpret=None, rgrad=True, BW=256, E=256, sc_impl="idx", parts=False):
    """Forward via SparseCore gather/scatter. sc_impl: "idx" (SC 32-bit indirect DMA), "db" (same, double-buffered),
    "row" (SC per-row DMA; CPU-interpret logic test only), "ref" (plain XLA take/scatter = sc_xla baseline)."""
    from .rope_pallas import _interp
    from . import rope_sc as SC
    interpret = _interp() if interpret is None else interpret
    B, H, T, hd = q.shape
    R, K = B * H, p.K
    f = lambda a: a.reshape((R,) + a.shape[2:])
    st, en, gb = (lax.stop_gradient(f(x)) for x in (st, en, gb))
    qid, stp, enp, gbp, cb, nused, eid = jax.vmap(
        lambda a, b, c: sort_plan(a, b, c, T, K, BW, E, with_eid=True))(st, en, gb)
    valid = jax.vmap(lambda a, b, c: _dedupe(a.astype(I32), b.astype(I32), c, K)[0])(st, en, gb)
    qr, kr, vr = f(q), f(k), f(v)
    if sc_impl == "ref":
        q_e = SC.ref_gather_rows(qr, qid)
    elif sc_impl in ("idx", "db"):                    # 32-bit SC indirect DMA (bf16 rows as uint32 pairs)
        lim = nused * E if sc_impl == "idx" else None          # skip the unused tail of each row (padding, QFRAC)
        q_e = SC.sc_gather_rows32(qr, qid, db=sc_impl == "db", nused_rows=lim)
    else:
        q_e = SC.sc_gather_rows(qr, qid, impl=sc_impl, interpret=interpret)
    ye = span_dense(q_e, kr, vr, stp, enp, gbp, cb, nused, BW=BW, E=E, sc=p.sc, interpret=interpret)
    if sc_impl == "ref":
        Y = SC.ref_scatter_rows(ye, eid, T * K)
    elif sc_impl in ("idx", "db"):
        lim = nused * E if sc_impl == "idx" else None
        Y = SC.sc_scatter_rows32(ye, eid, T * K, db=sc_impl == "db", nused_rows=lim)
    else:
        Y = SC.sc_scatter_rows(ye, eid, T * K, impl=sc_impl, interpret=interpret)
    o_w, m_w, l_w = win_fwd(qr, kr, vr, W=p.W, sc=p.sc, interpret=interpret)
    out, mass = combine_k(Y.reshape(R, T, K, hd + 128), valid, o_w, m_w, l_w, hd)
    res = out.astype(v.dtype).reshape(B, H, T, hd), lax.stop_gradient(mass).reshape(B, H, T, 1)
    if parts:
        return res, dict(qid=qid, eid=eid, cb=cb, nused=nused, stp=stp, enp=enp, gbp=gbp)
    return res
