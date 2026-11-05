"""M5: паритет пам'яті (memory_pass = mempass-цикл над mem2.Memory) з torch на CPU, W=1.
Запуск: cd memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_memory.py   (або pytest)"""
import _mem_ref as R  # noqa: E402  (ставить env/sys.path)
from _mem_ref import torch, jnp, jax, np, diff, tn, j2n, t2j, stable_torch

from memarch_jax.memory import MemCfg, memory_pass, put

D, T, SEG, B = 256, 1024, 64, 2
KW = dict(fresh=8, dim=12, dim_max=24, faces=2, d_face=64, write=8, focus=4)


def _cfg(cdt=jnp.float32, eb=False):
    return MemCfg(d=D, Fn=8, Dn=12, Dmax=24, nf=2, df=64, dc=D // 4, W=8, K=4, seg=SEG, cdt=cdt, eb=eb)


def _setup(seed=0):
    torch.manual_seed(seed)
    mem = R.mem2.Memory(D, **KW)
    with torch.no_grad():   # ненульові/неодиничні норми й dim_weight, щоб їхні градієнти щось перевіряли
        for nm in ("norm_w", "norm_r", "norm_c"):
            getattr(mem, nm).weight.add_(0.1 * torch.randn(D))
    h = R.structured_h(B, T, D, seed + 1)
    g = torch.Generator().manual_seed(seed + 2)
    n = T // SEG
    G = torch.randn(B, T, D, generator=g)
    Gc = torch.randn(n, B, KW["fresh"] + KW["dim_max"], D, generator=g)
    Gf = torch.randn(n, B, KW["fresh"] + KW["dim_max"], 2, 64, generator=g)
    return mem, h, G, Gc, Gf


def run_torch(mem, h, G, Gc, Gf, bf16):
    mem.zero_grad()
    h = h.clone().requires_grad_(True)
    with stable_torch():
        reads, snaps, st = R.torch_memory_pass(mem, h, SEG, bf16)
    loss = (reads.float() * G).sum() + sum((sn[1] * Gc[j]).sum() + (sn[0] * Gf[j]).sum() for j, sn in enumerate(snaps))
    loss.backward()
    grads = {k: tn(v.grad) for k, v in mem.named_parameters()}
    grads["h"] = tn(h.grad)
    out = dict(reads=tn(reads), faces=np.stack([tn(s[0]) for s in snaps]), contents=np.stack([tn(s[1]) for s in snaps]),
               pos=np.stack([tn(s[2]) for s in snaps]), valid=np.stack([tn(s[3]) for s in snaps]),
               st_pos=tn(st["pos"]), st_valid=tn(st["valid"]), st_fc=tn(st["fc"]), st_dc=tn(st["dc"]))
    return float(loss), out, grads


def run_jax(p, h, G, Gc, Gf, cfg):
    def lossf(p, h):
        reads, snaps, st, stats = memory_pass(p, h, cfg)
        l = (reads.astype(jnp.float32) * G).sum() + (snaps["contents"] * Gc).sum() + (snaps["faces"] * Gf).sum()
        return l, (reads, snaps, st, stats)
    (l, (reads, snaps, st, stats)), (gp, gh) = jax.jit(jax.value_and_grad(lossf, (0, 1), has_aux=True))(p, h)
    grads = {k: j2n(v) for k, v in gp.items()}
    grads["h"] = j2n(gh)
    out = dict(reads=j2n(reads), faces=j2n(snaps["faces"]), contents=j2n(snaps["contents"]),
               pos=j2n(snaps["pos"]), valid=j2n(snaps["valid"]), st_pos=j2n(st["pos"]), st_valid=j2n(st["valid"]),
               st_fc=j2n(st["fc"]), st_dc=j2n(st["dc"]))
    return float(l), out, grads, {k: np.asarray(v) for k, v in stats.items()}


def compare(cdt, eb, bf16_torch, verbose=True):
    mem, h, G, Gc, Gf = _setup()
    lt, ot, gt = run_torch(mem, h, G, Gc, Gf, bf16_torch)
    p = t2j(mem.state_dict())
    lj, oj, gj, stats = run_jax(p, jnp.asarray(h.numpy()), jnp.asarray(G.numpy()), jnp.asarray(Gc.numpy()),
                                jnp.asarray(Gf.numpy()), _cfg(cdt, eb))
    res = {"loss": diff(lj, lt)}
    for k in ot:
        res[k] = diff(oj[k], ot[k])
    res["pos_mismatch"] = int((oj["pos"] != ot["pos"]).sum())
    res["valid_mismatch"] = int((oj["valid"] != ot["valid"]).sum())
    for k in gt:
        res["grad:" + k] = diff(gj[k], gt[k])
    if verbose:
        print(f"--- memory  jax cdt={jnp.dtype(cdt).name} eb={eb}  vs torch bf16_autocast={bf16_torch}")
        print("  stats (сума по шматках/батчу):", {k: int(v.sum()) for k, v in stats.items()})
        for k, v in res.items():
            print(f"  {k:28s} {v}")
    return res, stats


def test_memory_fp32_parity():
    res, stats = compare(jnp.float32, False, False)
    # гілки справді працюють
    assert stats["upd"].sum() > 0 and stats["demote"].sum() > 0 and stats["promote"].sum() > 0
    assert stats["prune"].sum() > 0 and stats["dup"].sum() > 0
    assert res["pos_mismatch"] == 0 and res["valid_mismatch"] == 0
    assert res["st_pos"][0] == 0 and res["st_valid"][0] == 0
    for k, v in res.items():
        if isinstance(v, tuple):
            assert v[1] < 1e-5, (k, v)


def test_memory_bf16_report():
    """bf16 (MEM_BF16): звіт розбіжностей; eb=True — einsum як torch CPU autocast (bf16), eb=False — fp32 (наш вибір)."""
    r1, _ = compare(jnp.bfloat16, True, True)
    r2, _ = compare(jnp.bfloat16, False, True)
    assert np.isfinite(r1["reads"][0]) and np.isfinite(r2["reads"][0])


def test_memory_bf16_teacher_forced():
    """bf16 по шматку: JAX-читання й запис від ТОГО САМОГО стану, що в torch (стан щоразу з torch). Відокремлює
    помилки порту від хаосу дискретних виборів у вільному прогоні. eb=True (einsum як у torch CPU autocast)."""
    import torch.nn.functional as F  # noqa
    from memarch_jax import memory as mm
    mem, h, *_ = _setup()
    cfg = _cfg(jnp.bfloat16, True)
    p = t2j(mem.state_dict())
    st = mem.init_state(B, "cpu", torch.float32)
    rd = jax.jit(lambda p, x, s: mm.mem_read(p, x, s, cfg))
    wr = jax.jit(lambda p, x, q, s, now, w: mm.mem_write(p, x, q, s, now, w, cfg))
    worst = {}
    with torch.no_grad(), stable_torch(), torch.autocast("cpu", dtype=torch.bfloat16):
        for j in range(T // SEG):
            hs = h[:, j * SEG:(j + 1) * SEG].float()
            sj = {k: jnp.asarray(tn(v)) for k, v in st.items()}
            sj["pos"] = sj["pos"].astype(jnp.int32)
            r, w = mem.read(hs, st)
            rj, wj = rd(p, jnp.asarray(hs.numpy()), sj)
            x = hs + r
            qp = (j * SEG + torch.arange(float(SEG))).expand(B, -1)
            now = torch.tensor(float(SEG * (j + 1)))
            st = mem.write(x, qp, st, now=now, w_read=w)
            stj, _ = wr(p, jnp.asarray(x.float().numpy()), jnp.asarray(qp.numpy()), sj, jnp.float32(now), jnp.asarray(tn(w)))
            cur = {"r": diff(j2n(rj), tn(r)), "w": diff(j2n(wj), tn(w))}
            cur.update({k: diff(j2n(stj[k]), tn(st[k])) for k in st})
            for k, v in cur.items():
                worst[k] = max(worst.get(k, (0, 0)), v, key=lambda t: t[1])
    print("--- memory bf16 teacher-forced (гірший шматок, (max abs, max rel)):")
    for k, v in worst.items():
        print(f"  {k:10s} {v}")
    assert worst["pos"][0] == 0 and worst["valid"][0] == 0
    assert all(v[1] < 2e-2 for v in worst.values()), worst


# ---- юніт-тести -------------------------------------------------------------------------------------------
def test_put_average():
    t = jnp.zeros((1, 4, 2))
    idx = jnp.array([[1, 1, 3]])
    val = jnp.array([[[2., 4.], [4., 8.], [1., 1.]]])
    out = put(t, idx, val, jnp.array([[True, True, False]]))
    np.testing.assert_allclose(np.asarray(out[0]), [[0, 0], [3, 6], [0, 0], [0, 0]])
    ti = jnp.array([[5, 6, 7, 8]], jnp.int32)
    oi = put(ti, jnp.array([[2, 2]]), jnp.array([[1001., 1003.]]), jnp.array([[True, True]]))
    assert np.asarray(oi).tolist() == [[5, 6, 1002, 8]]


def test_seg0_and_events():
    """Шматок 0: нульове читання і W нових карток; далі — прибирання тримає тьмяну ≤ Dmax-W+... і Dn після."""
    mem, h, *_ = _setup()
    p = t2j(mem.state_dict())
    cfg = _cfg()
    reads, snaps, st, stats = jax.jit(lambda p, h: memory_pass(p, h, cfg))(p, jnp.asarray(h.numpy()))
    reads = np.asarray(reads)
    out0 = np.asarray(jnp.zeros((1, D)) @ p["out.weight"].T)
    assert np.abs(reads[:, :SEG] - out0).max() == 0
    assert (np.asarray(stats["new"][0]) == cfg.W).all() and (np.asarray(stats["upd"][0]) == 0).all()
    v = np.asarray(snaps["valid"])                     # (n,B,C)
    assert v[1, :, :cfg.Fn].sum() == B * cfg.W         # після шматка 0 — W свіжих карток (Fn=W тут)
    pr = np.asarray(stats["prune"])
    vd_after = np.concatenate([v[1:], np.asarray(st["valid"])[None]], 0)[..., cfg.Fn:].sum(-1)   # (n,B)
    assert (vd_after[pr] == cfg.Dn).all()              # після прибирання рівно Dn тьмяних
    assert (vd_after <= cfg.Dmax).all()


if __name__ == "__main__":
    test_put_average()
    test_seg0_and_events()
    test_memory_fp32_parity()
    test_memory_bf16_teacher_forced()
    test_memory_bf16_report()
    print("OK test_memory")
