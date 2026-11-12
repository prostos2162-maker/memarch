"""Rope read v2 (docs/rope_read_v2_plan.md, steps 2-3): bucket-sorted span entries + band window, LSE merge.

build_plan  — int-only plan per row (dedupe, buckets of width BW, chunks of E entries), vmappable.
read_ref    — pure-XLA reference of the whole v2 read; grads by plain autodiff. Same contract as rope_pallas.read:
              read_ref(q, k, v, st, en, gb, tpos, p, interpret, rgrad) -> (out (dtype v), mass (B,H,T,1) f32).

Semantics mirrored from rope_pallas.read:
  * score of key t for query i = sc*q_i.k_t + sum_k gb[i,k]*[st_k <= t < en_k]; window keys i-W+1..i (no bias);
  * a key inside two spans of the same query counts ONCE with summed bias -> dedupe of identical (st,en) entries,
    gb folded into the first one (partially overlapping distinct spans are not supported; route never makes them);
  * st/en/gb/tpos carry no gradient; mass is stop_gradient (its cotangent is ignored);
  * rgrad: bias gradient BB[i,k] = sum_{t in span k} dS[i,t] goes to q_i via sc_bias*k[tpos] and to k[tpos] via
    sc_bias*q_i, i.e. gb behaves like gb + sc_bias*(q_i.k[tpos] - stop_grad(same)). For a deduped duplicate the old
    kernel gives BB_dup = BB_first, routed through its own tpos — folding the differentiable gb reproduces that.
"""
import math

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

F32 = jnp.float32
I32 = jnp.int32
HI = lax.Precision.HIGHEST


def plan_sizes(T, K, BW, E):
    nb = T // BW
    N = T * K
    NCH = -(-(N + nb * (E - 1)) // E)
    return nb, N, NCH, NCH * E


def build_plan(st, en, gb, T, K, BW=256, E=256):
    """One row: st, en (T,K) i32, gb (T,K) f32. Returns (plan dict of ints, gb_ded (T,K)).
    gb_ded is differentiable in gb (dup gb added into the first entry); everything else is under stop_gradient."""
    assert T % BW == 0
    nb, N, NCH, Npad = plan_sizes(T, K, BW, E)
    st = lax.stop_gradient(st).astype(I32)
    en = lax.stop_gradient(en).astype(I32)
    v0 = en > st
    # 1. dedupe: owner[i,k] = first k' with identical valid (st,en)
    same = (st[:, :, None] == st[:, None, :]) & (en[:, :, None] == en[:, None, :]) & v0[:, :, None] & v0[:, None, :]
    le = jnp.arange(K)[None, :] <= jnp.arange(K)[:, None]                  # [k, k'] : k' <= k
    cand = same & le[None]
    owner = jnp.where(v0, jnp.argmax(cand, axis=-1), jnp.arange(K)[None, :]).astype(I32)
    dup = owner != jnp.arange(K)[None, :]
    gb_ded = jnp.einsum("ik,ikj->ij", gb.astype(F32), jax.nn.one_hot(owner, K, dtype=F32))
    valid = v0 & ~dup
    # 2-3. buckets
    stf, enf, vf = st.reshape(N), en.reshape(N), valid.reshape(N)
    bkt = jnp.where(vf, stf // BW, nb)
    cnt = jax.ops.segment_sum(vf.astype(I32), bkt, nb + 1)[:nb]
    pcnt = (cnt + E - 1) // E * E
    poff = jnp.concatenate([jnp.zeros(1, I32), jnp.cumsum(pcnt)[:-1]]).astype(I32)
    nused = (jnp.sum(pcnt) // E).astype(I32)
    # 4. stable rank within bucket
    perm = jnp.argsort(bkt, stable=True)
    sb = bkt[perm]
    start = jnp.concatenate([jnp.zeros(1, I32), jnp.cumsum(cnt)]).astype(I32)     # (nb+1)
    rank = jnp.zeros(N, I32).at[perm].set(jnp.arange(N, dtype=I32) - start[sb])
    slot = jnp.where(vf, poff[jnp.minimum(bkt, nb - 1)] + rank, Npad).astype(I32)
    # 5. src map
    src = jnp.full(Npad, -1, I32).at[slot].set(jnp.arange(N, dtype=I32), mode="drop")
    # 6. per chunk
    c0 = jnp.arange(NCH, dtype=I32) * E
    cb = jnp.clip(jnp.searchsorted(poff, c0, side="right").astype(I32) - 1, 0, nb - 1)
    used = jnp.arange(NCH) < nused
    cb = jnp.where(used, cb, cb[jnp.maximum(nused - 1, 0)])
    first = (used & (c0 == poff[cb])).astype(I32)
    last = (used & (c0 + E == poff[cb] + pcnt[cb])).astype(I32)
    # 7. gathered ints
    ok = src >= 0
    sc_ = jnp.maximum(src, 0)
    st_e = jnp.where(ok, stf[sc_], T)
    en_e = jnp.where(ok, enf[sc_], 0)
    plan = dict(src=src, slot=slot, cb=cb, first=first, last=last, nused=nused, cnt=cnt, pcnt=pcnt, poff=poff,
                valid=valid, owner=owner, st_e=st_e, en_e=en_e)
    return jax.tree.map(lax.stop_gradient, plan), gb_ded


def check_plan(plan, T, K, BW, E, cap):
    """Debug invariants (numpy, one row). Returns dict of checks; raises AssertionError on failure."""
    nb, N, NCH, Npad = plan_sizes(T, K, BW, E)
    assert cap <= BW
    pl_ = {k: np.asarray(v) for k, v in plan.items()}
    src, slot, cb, first, last, nused = (pl_[x] for x in ("src", "slot", "cb", "first", "last", "nused"))
    valid = pl_["valid"].reshape(N)
    nused = int(nused)
    assert nused <= NCH, (nused, NCH)
    ids = src[src >= 0]
    assert len(ids) == len(np.unique(ids)) == valid.sum(), "each valid entry exactly once"
    assert valid[ids].all()
    assert (src[slot[valid]] == np.nonzero(valid)[0]).all() and (slot[~valid] == Npad).all()
    assert (slot[valid] < nused * E).all(), "slots only in used chunks"
    cbe = np.repeat(cb, E)
    v = src >= 0
    assert (pl_["st_e"][v] >= BW * cbe[v]).all() and (pl_["en_e"][v] <= BW * cbe[v] + 2 * BW).all(), "bucket window"
    assert ((pl_["st_e"][v] // BW) == cbe[v]).all()
    # consecutive revisits: each bucket's chunks form one contiguous run with first at start, last at end
    cbu = cb[:nused]
    for b in np.unique(cbu):
        idx = np.nonzero(cbu == b)[0]
        assert (np.diff(idx) == 1).all(), "revisits consecutive"
        assert first[idx[0]] == 1 and first[idx[1:]].sum() == 0 and last[idx[-1]] == 1 and last[idx[:-1]].sum() == 0
    assert first[nused:].sum() == 0 and last[nused:].sum() == 0
    assert (cnt0 := pl_["cnt"]).sum() == valid.sum()
    return dict(nused=nused, NCH=NCH, valid=int(valid.sum()), empty_buckets=int((cnt0 == 0).sum()),
                max_chunks_per_bucket=int((pl_["pcnt"] // E).max()))


def _row(q, k, v, st, en, gb, tpos, *, T, K, W, BW, E, sc, sc_bias, rgrad):
    """One (b,h) row in f32. Returns out (T,hd) f32, mass (T,) f32."""
    nb, N, NCH, Npad = plan_sizes(T, K, BW, E)
    hd = q.shape[-1]
    if rgrad:          # zero-valued, differentiable copy of the bias: d gb / dq = sc_bias*k[tpos], d/dk[tpos] = sc_bias*q
        r = jnp.einsum("id,ikd->ik", q, k[jnp.clip(tpos, 0, T - 1)], precision=HI) * sc_bias
        gb = lax.stop_gradient(gb) + r - lax.stop_gradient(r)
    else:
        gb = lax.stop_gradient(gb)
    plan, gb_ded = build_plan(st, en, gb, T, K, BW, E)
    src, cb = plan["src"], plan["cb"]
    ok = src >= 0
    gb_e = jnp.where(ok, gb_ded.reshape(N)[jnp.maximum(src, 0)], 0.0)
    # ---- span part: per chunk dense over its two key blocks
    qe = q[jnp.maximum(src, 0) // K].reshape(NCH, E, hd)
    kb, vb = k.reshape(nb, BW, hd), v.reshape(nb, BW, hd)
    hi = jnp.minimum(cb + 1, nb - 1)
    Kc = jnp.concatenate([kb[cb], kb[hi]], 1)                    # (NCH, 2BW, hd)
    Vc = jnp.concatenate([vb[cb], vb[hi]], 1)
    t = cb[:, None] * BW + jnp.arange(2 * BW, dtype=I32)[None]  # unclamped: clamped hi gives t >= T -> masked
    st_e, en_e = plan["st_e"].reshape(NCH, E, 1), plan["en_e"].reshape(NCH, E, 1)
    msk = (t[:, None, :] >= st_e) & (t[:, None, :] < en_e)
    s = jnp.einsum("ced,cjd->cej", qe, Kc, precision=HI) * sc + gb_e.reshape(NCH, E, 1)
    s = jnp.where(msk, s, 0.0)
    m = lax.stop_gradient(jnp.max(jnp.where(msk, s, -jnp.inf), -1))
    p = jnp.where(msk, jnp.exp(s - jnp.where(jnp.isfinite(m), m, 0.0)[..., None]), 0.0)
    l_e = p.sum(-1).reshape(Npad)
    o_e = jnp.einsum("cej,cjd->ced", p, Vc, precision=HI).reshape(Npad, hd)
    m_e = m.reshape(Npad)
    # unpermute (sentinel slot Npad -> empty)
    m_s = jnp.concatenate([m_e, jnp.full(1, -jnp.inf)])[plan["slot"]].reshape(T, K)
    l_s = jnp.concatenate([l_e, jnp.zeros(1)])[plan["slot"]].reshape(T, K)
    o_s = jnp.concatenate([o_e, jnp.zeros((1, hd))])[plan["slot"]].reshape(T, K, hd)
    # ---- window part
    i = jnp.arange(T, dtype=I32)
    idx = i[:, None] - jnp.arange(W, dtype=I32)[None]
    wok = idx >= 0
    idx = jnp.maximum(idx, 0)
    sw = jnp.einsum("id,iwd->iw", q, k[idx], precision=HI) * sc
    sw = jnp.where(wok, sw, 0.0)
    m_w = lax.stop_gradient(jnp.max(jnp.where(wok, sw, -jnp.inf), -1))      # never empty (t = i)
    pw = jnp.where(wok, jnp.exp(sw - m_w[:, None]), 0.0)
    l_w = pw.sum(-1)
    o_w = jnp.einsum("iw,iwd->id", pw, v[idx], precision=HI)
    # ---- LSE combine (plan §2.4)
    M = jnp.maximum(m_w, jnp.max(m_s, -1))                                  # finite
    ew = jnp.exp(m_w - M)
    has = l_s > 0
    ek = jnp.where(has, jnp.exp(jnp.where(has, m_s, 0.0) - M[:, None]), 0.0)
    a_k = ek * l_s
    Lt = ew * l_w + a_k.sum(-1)
    out = (ew[:, None] * o_w + (ek[..., None] * o_s).sum(1)) / Lt[:, None]
    mass = lax.stop_gradient(a_k.sum(-1) / Lt)
    return out, mass


def read_ref(q, k, v, st, en, gb, tpos, p, interpret=None, rgrad=True, BW=256, E=256):
    """Same contract as rope_pallas.read. q,k,v (B,H,T,hd); st,en,gb,tpos (B,H,T,K). interpret is unused."""
    B, H, T, hd = q.shape
    f = lambda *a: _row(*a, T=T, K=p.K, W=p.W, BW=BW, E=E, sc=p.sc, sc_bias=p.sc_bias, rgrad=rgrad)
    f = jax.vmap(jax.vmap(f))
    st, en, gb, tpos = (lax.stop_gradient(x) for x in (st, en, gb, tpos))
    out, mass = f(q.astype(F32), k.astype(F32), v.astype(F32), st, en, gb.astype(F32), tpos)
    return out.astype(v.dtype), lax.stop_gradient(mass)[..., None]


build_plan_rows = lambda st, en, gb, T, K, BW=256, E=256: jax.vmap(
    lambda a, b, c: build_plan(a, b, c, T, K, BW, E))(st, en, gb)
