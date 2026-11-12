"""SparseCore row gather / row scatter for rope read v2 (jax 0.11.2, jax.experimental.pallas.tpu_sc).

v6e SparseCore (jax tpu_info): 2 SCs x 16 vector subcores, 8 lanes, 256 KiB VMEM per subcore, DMA granule 32 B.
The primitive used is the SC indirect DMA:
  gather : pltpu.async_copy(x_hbm.at[idx_vmem], buf_vmem, sem)          (HBM rows idx -> VMEM)
  scatter: pltpu.async_copy(buf_vmem, y_hbm.at[idx_vmem], sem[, add=True]) (VMEM -> HBM rows idx)
idx must be a 1-D int32 VMEM ref. (plsc.Indices(..., ignored_value=-1) also exists, but the CPU interpreter cannot
handle it, so padding is redirected instead: gather reads row 0, scatter writes into one extra sink row.)

Work split: the flat index list (M,) is cut into NW = cores*subcores equal contiguous parts; each worker runs
batches of BR rows: idx HBM->VMEM, indirect DMA, linear DMA. No add -> no write races (destinations are unique).

impl="idx": one indirect DMA per batch (the real SC gather/scatter path).
impl="row": BR single-row DMAs per batch (same partitioning; runs in the CPU interpreter, used for logic tests).

TPU probe 3 (v6e): indirect DMA supports ONLY 32-bit elements ("enqueue_indirect_transfer: only 32-bit elements")
-> use sc_gather_rows32 / sc_scatter_rows32 below (bf16 rows bitcast to uint32 pairs). impl="row" does not compile
on TPU (1-row DMA slices must be 8-row aligned); kept only as the CPU-interpretable logic test.
"""
import functools

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

I32 = jnp.int32


def _mesh():
    return plsc.VectorSubcoreMesh(core_axis_name="c", subcore_axis_name="s")


def _split(M, BRmax=128):
    m = _mesh()
    NW = m.num_cores * m.num_subcores
    BR = BRmax
    Mp = -(-M // (NW * BR)) * NW * BR
    return m, NW, BR, Mp


def _worker(t, nb, NW, mesh, interpret):
    """-> (worker id, batch). On TPU each subcore runs its own nb batches. The CPU interpreter does not model the
    SC mesh (only worker 0 runs), so there worker 0 loops over all NW workers' batches (same offsets)."""
    if interpret:
        return t // nb, t % nb
    return lax.axis_index("c") * mesh.num_subcores + lax.axis_index("s"), t


def _interp_arg(interpret):
    return pltpu.InterpretParams() if interpret else False


def sc_gather_rows_flat(x, idx, *, impl="idx", BR=128, interpret=False):
    """x (N, w), idx (M,) int32 in [0,N) or -1 -> (M, w); rows with idx=-1 are unspecified (read row 0)."""
    if interpret:                 # CPU interpreter: small batches (less padding; its DMA-id counter is int16)
        from jax.experimental.pallas import tpu as _t
        _t.reset_tpu_interpret_mode_state()
        BR = min(BR, 8)
    M = idx.shape[0]
    w = x.shape[1]
    mesh, NW, BR, Mp = _split(M, BR)
    idx = jnp.pad(jnp.maximum(idx, 0).astype(I32), (0, Mp - M))
    per = Mp // NW
    nb = per // BR

    def body(x_hbm, i_hbm, o_hbm, ib, buf, sem):
        def step(t, carry):
            wk, b = _worker(t, nb, NW, mesh, interpret)
            off = wk * per + b * BR
            pltpu.async_copy(i_hbm.at[pl.ds(off, BR)], ib, sem).wait()
            if impl == "idx":
                pltpu.async_copy(x_hbm.at[ib], buf, sem).wait()
            else:
                for j in range(BR):
                    pltpu.async_copy(x_hbm.at[pl.ds(ib[j], 1)], buf.at[pl.ds(j, 1)], sem).wait()
            pltpu.async_copy(buf, o_hbm.at[pl.ds(off, BR)], sem).wait()
            return carry
        lax.fori_loop(0, nb * (NW if interpret else 1), step, 0)

    out = pl.kernel(body, out_type=jax.ShapeDtypeStruct((Mp, w), x.dtype), mesh=mesh,
                    scratch_types=[(pltpu.VMEM if impl == "idx" else pltpu.SMEM)((BR,), I32), pltpu.VMEM((BR, w), x.dtype), pltpu.SemaphoreType.DMA],
                    interpret=_interp_arg(interpret), name=f"sc_gather_{impl}")(x, idx)
    return out[:M]


def sc_scatter_rows_flat(vals, idx, n_out, *, impl="idx", BR=128, interpret=False):
    """vals (M, w), idx (M,) unique in [0,n_out) or -1 -> (n_out, w); rows never written are UNSPECIFIED
    (caller masks them). idx=-1 rows go to a sink row n_out (dropped)."""
    if interpret:                 # CPU interpreter: small batches (less padding; its DMA-id counter is int16)
        from jax.experimental.pallas import tpu as _t
        _t.reset_tpu_interpret_mode_state()
        BR = min(BR, 8)
    M, w = vals.shape
    mesh, NW, BR, Mp = _split(M, BR)
    idx = jnp.pad(jnp.where(idx < 0, n_out, idx).astype(I32), (0, Mp - M), constant_values=n_out)
    vals = jnp.pad(vals, ((0, Mp - M), (0, 0)))
    per = Mp // NW
    nb = per // BR

    def body(v_hbm, i_hbm, o_hbm, ib, buf, sem):
        def step(t, carry):
            wk, b = _worker(t, nb, NW, mesh, interpret)
            off = wk * per + b * BR
            pltpu.async_copy(i_hbm.at[pl.ds(off, BR)], ib, sem).wait()
            pltpu.async_copy(v_hbm.at[pl.ds(off, BR)], buf, sem).wait()
            if impl == "idx":
                pltpu.async_copy(buf, o_hbm.at[ib], sem).wait()
            else:
                for j in range(BR):
                    pltpu.async_copy(buf.at[pl.ds(j, 1)], o_hbm.at[pl.ds(ib[j], 1)], sem).wait()
            return carry
        lax.fori_loop(0, nb * (NW if interpret else 1), step, 0)

    out = pl.kernel(body, out_type=jax.ShapeDtypeStruct((n_out + 1, w), vals.dtype), mesh=mesh,
                    scratch_types=[(pltpu.VMEM if impl == "idx" else pltpu.SMEM)((BR,), I32), pltpu.VMEM((BR, w), vals.dtype), pltpu.SemaphoreType.DMA],
                    interpret=_interp_arg(interpret), name=f"sc_scatter_{impl}")(vals, idx)
    return out[:n_out]


def sc_gather_rows(x, idx, **kw):
    """x (R,T,hd), idx (R,N) in [0,T) or -1 -> (R,N,hd)."""
    R, T, hd = x.shape
    N = idx.shape[1]
    g = jnp.where(idx >= 0, idx + (jnp.arange(R, dtype=I32) * T)[:, None], -1)
    return sc_gather_rows_flat(x.reshape(R * T, hd), g.reshape(-1), **kw).reshape(R, N, hd)


def sc_scatter_rows(vals, idx, n_out, **kw):
    """vals (R,N,w), idx (R,N) unique per row in [0,n_out) or -1 -> (R,n_out,w), unwritten rows unspecified."""
    R, N, w = vals.shape
    g = jnp.where(idx >= 0, idx + (jnp.arange(R, dtype=I32) * n_out)[:, None], -1)
    return sc_scatter_rows_flat(vals.reshape(R * N, w), g.reshape(-1), R * n_out, **kw).reshape(R, n_out, w)


# ---- pure-jnp references (XLA gather / scatter)

def ref_gather_rows(x, idx):
    return jnp.take_along_axis(x, jnp.maximum(idx, 0)[..., None], axis=1)


def ref_scatter_rows(vals, idx, n_out):
    R, N, w = vals.shape
    out = jnp.zeros((R, n_out + 1, w), vals.dtype)
    ii = jnp.where(idx < 0, n_out, idx)
    return out.at[jnp.arange(R)[:, None], ii].set(vals)[:, :n_out]


def cpu_sc_info():
    """Let get_sparse_core_info() work on CPU (interpret tests): register v6e info for device kind 'cpu'."""
    from jax._src import tpu_info as TI
    TI.registry.setdefault("cpu", lambda: TI._get_tpu_info_impl(TI.ChipVersion.TPU_V6E, 1))


# ---------------------------------------------------------------- 32-bit packing + double-buffered (impl="db")
# SC indirect DMA ("sc_tpu.enqueue_indirect_transfer") supports only 32-bit elements: bf16 rows travel as uint32
# pairs (exact bitcast). Double-buffered kernels: 2 slots (static ping-pong, loop over batch pairs), separate DMA
# semaphores per slot for index load / gather (or value load) / store (or scatter); the next batch's loads overlap
# the current indirect DMA and the previous store. Batch rows BR are sized to the 256 KiB subcore VMEM.

SC_VMEM_BUDGET = 160 * 1024


def _to32(x):
    """(..., w) any dtype -> (..., w*itemsize/4) uint32 (exact)."""
    it = jnp.dtype(x.dtype).itemsize
    if it == 4:
        return x if x.dtype == jnp.uint32 else lax.bitcast_convert_type(x, jnp.uint32)
    assert it in (1, 2) and (x.shape[-1] * it) % 4 == 0
    return lax.bitcast_convert_type(x.reshape(*x.shape[:-1], x.shape[-1] * it // 4, 4 // it), jnp.uint32)


def _from32(y, dtype, w):
    it = jnp.dtype(dtype).itemsize
    if it == 4:
        return lax.bitcast_convert_type(y, dtype)
    return lax.bitcast_convert_type(y, dtype).reshape(*y.shape[:-1], w)


def _br_for(w32, nslots, BRmax=256):
    """Largest power-of-2 batch (8..BRmax) with nslots*(BR*w32*4 + BR*4) <= SC_VMEM_BUDGET."""
    BR = BRmax
    while BR > 8 and nslots * BR * (w32 * 4 + 4) > SC_VMEM_BUDGET:
        BR //= 2
    return BR


def sc_gather32_flat(x32, idx, *, db=False, BR=None, lim=None, row_len=None):
    """x32 (N, w) uint32/f32/i32, idx (M,) in [0,N) or -1 (reads row 0) -> (M, w). TPU only (indirect DMA).
    lim (R,) int32 + row_len (simple kernel only): idx = R rows of row_len; batches at positions >= lim[r] are skipped
    (their output rows are unspecified). ROPE_QFRAC / padding: only the used prefix (nused*E) of each row moves."""
    M = idx.shape[0]
    w = x32.shape[1]
    BR = BR or _br_for(w, 2 if db else 1, 256 if db else 128)
    mesh = _mesh()
    NW = mesh.num_cores * mesh.num_subcores
    step = NW * BR * (2 if db else 1)
    Mp = -(-M // step) * step
    idx = jnp.pad(jnp.maximum(idx, 0).astype(I32), (0, Mp - M))
    per = Mp // NW
    nb = per // BR

    def wid():
        return lax.axis_index("c") * mesh.num_subcores + lax.axis_index("s")

    if not db and lim is not None:
        assert row_len % BR == 0
        Rr = lim.shape[0]

        def body(x_hbm, i_hbm, l_hbm, o_hbm, ib, buf, ls, sem):
            wk = wid()
            pltpu.async_copy(l_hbm, ls, sem).wait()

            def stp(b, carry):
                off = wk * per + b * BR
                r = jnp.minimum(off // row_len, Rr - 1)

                @pl.when((off < M) & (off - r * row_len < ls[r]))
                def _():
                    pltpu.async_copy(i_hbm.at[pl.ds(off, BR)], ib, sem).wait()
                    pltpu.async_copy(x_hbm.at[ib], buf, sem).wait()
                    pltpu.async_copy(buf, o_hbm.at[pl.ds(off, BR)], sem).wait()
                return carry
            lax.fori_loop(0, nb, stp, 0)
        scratch = [pltpu.VMEM((BR,), I32), pltpu.VMEM((BR, w), x32.dtype), pltpu.SMEM((Rr,), I32),
                   pltpu.SemaphoreType.DMA]
        out = pl.kernel(body, out_type=jax.ShapeDtypeStruct((Mp, w), x32.dtype), mesh=mesh, scratch_types=scratch,
                        name="sc_gather32_lim")(x32, idx, lim.astype(I32))
        return out[:M]
    if not db:
        def body(x_hbm, i_hbm, o_hbm, ib, buf, sem):
            wk = wid()

            def stp(b, carry):
                off = wk * per + b * BR
                pltpu.async_copy(i_hbm.at[pl.ds(off, BR)], ib, sem).wait()
                pltpu.async_copy(x_hbm.at[ib], buf, sem).wait()
                pltpu.async_copy(buf, o_hbm.at[pl.ds(off, BR)], sem).wait()
                return carry
            lax.fori_loop(0, nb, stp, 0)
        scratch = [pltpu.VMEM((BR,), I32), pltpu.VMEM((BR, w), x32.dtype), pltpu.SemaphoreType.DMA]
    else:
        def body(x_hbm, i_hbm, o_hbm, ib0, ib1, b0, b1, si, sg, so):
            wk = wid()
            ibs, bufs = (ib0, ib1), (b0, b1)
            off = lambda b: wk * per + b * BR
            ld = lambda s, b: pltpu.make_async_copy(i_hbm.at[pl.ds(off(b), BR)], ibs[s], si.at[s])
            st = lambda s, b: pltpu.make_async_copy(bufs[s], o_hbm.at[pl.ds(off(b), BR)], so.at[s])
            ld(0, 0).start()

            def pair(pp, carry):
                for s in (0, 1):                      # static slot
                    b = 2 * pp + s

                    @pl.when(b >= 2)
                    def _():
                        st(s, b - 2).wait()           # slot buffer free again
                    ld(s, b).wait()
                    g = pltpu.make_async_copy(x_hbm.at[ibs[s]], bufs[s], sg.at[s])
                    g.start()

                    @pl.when(b + 1 < nb)
                    def _():
                        ld(1 - s, b + 1).start()      # prefetch next indices during the gather
                    g.wait()
                    st(s, b).start()                  # store overlaps the next gather
                return carry
            lax.fori_loop(0, nb // 2, pair, 0)
            st(0, nb - 2).wait()
            st(1, nb - 1).wait()
        scratch = [pltpu.VMEM((BR,), I32), pltpu.VMEM((BR,), I32), pltpu.VMEM((BR, w), x32.dtype),
                   pltpu.VMEM((BR, w), x32.dtype), pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,)),
                   pltpu.SemaphoreType.DMA((2,))]
    out = pl.kernel(body, out_type=jax.ShapeDtypeStruct((Mp, w), x32.dtype), mesh=mesh, scratch_types=scratch,
                    name=f"sc_gather32{'_db' if db else ''}")(x32, idx)
    return out[:M]


def sc_scatter32_flat(vals, idx, n_out, *, db=False, BR=None, lim=None, row_len=None):
    """vals (M, w) 32-bit, idx (M,) unique in [0,n_out) or -1 (-> sink row) -> (n_out, w), unwritten unspecified.
    lim/row_len (simple kernel only): rows of row_len entries; batches at positions >= lim[r] are skipped."""
    M, w = vals.shape
    assert jnp.dtype(vals.dtype).itemsize == 4
    BR = BR or _br_for(w, 2 if db else 1, 256 if db else 128)
    mesh = _mesh()
    NW = mesh.num_cores * mesh.num_subcores
    step = NW * BR * (2 if db else 1)
    Mp = -(-M // step) * step
    idx = jnp.pad(jnp.where(idx < 0, n_out, idx).astype(I32), (0, Mp - M), constant_values=n_out)
    vals = jnp.pad(vals, ((0, Mp - M), (0, 0)))
    per = Mp // NW
    nb = per // BR

    def wid():
        return lax.axis_index("c") * mesh.num_subcores + lax.axis_index("s")

    if not db and lim is not None:
        assert row_len % BR == 0
        Rr = lim.shape[0]

        def body(v_hbm, i_hbm, l_hbm, o_hbm, ib, buf, ls, sem):
            wk = wid()
            pltpu.async_copy(l_hbm, ls, sem).wait()

            def stp(b, carry):
                off = wk * per + b * BR
                r = jnp.minimum(off // row_len, Rr - 1)

                @pl.when((off < M) & (off - r * row_len < ls[r]))
                def _():
                    pltpu.async_copy(i_hbm.at[pl.ds(off, BR)], ib, sem).wait()
                    pltpu.async_copy(v_hbm.at[pl.ds(off, BR)], buf, sem).wait()
                    pltpu.async_copy(buf, o_hbm.at[ib], sem).wait()
                return carry
            lax.fori_loop(0, nb, stp, 0)
        scratch = [pltpu.VMEM((BR,), I32), pltpu.VMEM((BR, w), vals.dtype), pltpu.SMEM((Rr,), I32),
                   pltpu.SemaphoreType.DMA]
        out = pl.kernel(body, out_type=jax.ShapeDtypeStruct((n_out + 1, w), vals.dtype), mesh=mesh,
                        scratch_types=scratch, name="sc_scatter32_lim")(vals, idx, lim.astype(I32))
        return out[:n_out]
    if not db:
        def body(v_hbm, i_hbm, o_hbm, ib, buf, sem):
            wk = wid()

            def stp(b, carry):
                off = wk * per + b * BR
                pltpu.async_copy(i_hbm.at[pl.ds(off, BR)], ib, sem).wait()
                pltpu.async_copy(v_hbm.at[pl.ds(off, BR)], buf, sem).wait()
                pltpu.async_copy(buf, o_hbm.at[ib], sem).wait()
                return carry
            lax.fori_loop(0, nb, stp, 0)
        scratch = [pltpu.VMEM((BR,), I32), pltpu.VMEM((BR, w), vals.dtype), pltpu.SemaphoreType.DMA]
    else:
        def body(v_hbm, i_hbm, o_hbm, ib0, ib1, b0, b1, si, sv, ss):
            wk = wid()
            ibs, bufs = (ib0, ib1), (b0, b1)
            off = lambda b: wk * per + b * BR
            li = lambda s, b: pltpu.make_async_copy(i_hbm.at[pl.ds(off(b), BR)], ibs[s], si.at[s])
            lv = lambda s, b: pltpu.make_async_copy(v_hbm.at[pl.ds(off(b), BR)], bufs[s], sv.at[s])
            sc = lambda s: pltpu.make_async_copy(bufs[s], o_hbm.at[ibs[s]], ss.at[s])
            li(0, 0).start()
            lv(0, 0).start()

            def pair(pp, carry):
                for s in (0, 1):
                    b = 2 * pp + s
                    li(s, b).wait()
                    lv(s, b).wait()
                    sc(s).start()

                    @pl.when(b + 1 < nb)
                    def _():
                        @pl.when(b >= 1)
                        def _():
                            sc(1 - s).wait()          # slot 1-s (batch b-1) scattered -> reusable
                        li(1 - s, b + 1).start()      # loads of b+1 overlap the scatter of b
                        lv(1 - s, b + 1).start()
                return carry
            lax.fori_loop(0, nb // 2, pair, 0)
            sc(0).wait()                              # batch nb-2 (slot 0) was never waited inside the loop
            sc(1).wait()                              # batch nb-1
        scratch = [pltpu.VMEM((BR,), I32), pltpu.VMEM((BR,), I32), pltpu.VMEM((BR, w), vals.dtype),
                   pltpu.VMEM((BR, w), vals.dtype), pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,)),
                   pltpu.SemaphoreType.DMA((2,))]
    out = pl.kernel(body, out_type=jax.ShapeDtypeStruct((n_out + 1, w), vals.dtype), mesh=mesh, scratch_types=scratch,
                    name=f"sc_scatter32{'_db' if db else ''}")(vals, idx)
    return out[:n_out]


def sc_gather_rows32(x, idx, *, db=False, nused_rows=None):
    """x (R,T,hd) any dtype (bf16 -> uint32 pairs), idx (R,N) in [0,T) or -1 -> (R,N,hd) in x.dtype."""
    R, T, hd = x.shape
    N = idx.shape[1]
    x32 = _to32(x)
    g = jnp.where(idx >= 0, idx + (jnp.arange(R, dtype=I32) * T)[:, None], -1)
    lim = None if nused_rows is None else nused_rows.astype(I32)
    y = sc_gather32_flat(x32.reshape(R * T, -1), g.reshape(-1), db=db, lim=lim, row_len=N).reshape(R, N, -1)
    return _from32(y, x.dtype, hd)


def sc_scatter_rows32(vals, idx, n_out, *, db=False, nused_rows=None):
    """vals (R,N,w) f32, idx (R,N) unique per row in [0,n_out) or -1 -> (R,n_out,w) f32, unwritten unspecified."""
    R, N, w = vals.shape
    g = jnp.where(idx >= 0, idx + (jnp.arange(R, dtype=I32) * n_out)[:, None], -1)
    lim = None if nused_rows is None else nused_rows.astype(I32)
    return sc_scatter32_flat(vals.reshape(R * N, w), g.reshape(-1), R * n_out, db=db, lim=lim,
                             row_len=N).reshape(R, n_out, w)
