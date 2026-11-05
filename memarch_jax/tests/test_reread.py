"""M6: паритет перечитування (_reread_sep + Reread, REREAD_ANCH=0.25, REREAD_IDS, REC_LEAN) з torch на CPU, W=1;
плюс комбінований тест пам'ять -> перечитування (градієнти крізь знімки в параметри пам'яті).
Запуск: cd memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_reread.py   (або pytest)"""
import _mem_ref as R  # noqa: E402
from _mem_ref import torch, jnp, jax, np, diff, tn, j2n, t2j, stable_torch

from memarch_jax.memory import MemCfg, memory_pass
from memarch_jax.reread import RereadCfg, reread_sep

D, T, SEG, V = 256, 1024, 64, 512
KW = dict(fresh=8, dim=12, dim_max=24, faces=2, d_face=64, write=8, focus=4)


def _mcfg(cdt=jnp.float32, eb=False):
    return MemCfg(d=D, Fn=8, Dn=12, Dmax=24, nf=2, df=64, dc=D // 4, W=8, K=4, seg=SEG, cdt=cdt, eb=eb)


def _rcfg(cdt=jnp.float32, eb=False):
    return RereadCfg(df=64, k=4, R=4, seg=SEG, frac=0.25, cdt=cdt, eb=eb)


def _modules(seed):
    torch.manual_seed(seed)
    mem = R.mem2.Memory(D, **KW)
    rr = R.full2.Reread(D, d_face=64, d_book=128, top_k=4, max_read=4)
    table = torch.nn.Parameter(torch.randn(V, 128))
    with torch.no_grad():
        for m, nm in ((mem, "norm_w"), (mem, "norm_r"), (mem, "norm_c"), (rr, "norm")):
            getattr(m, nm).weight.add_(0.1 * torch.randn(D))
        rr.halt_v.bias.add_(-1.0)   # частіше «читати далі» — працює весь ланцюг виживання
    return mem, rr, table


def _inputs(B, seed):
    g = torch.Generator().manual_seed(seed)
    h = R.structured_h(B, T, D, seed)
    h2 = torch.randn(B, T, D, generator=g)
    ids = torch.randint(0, V, (B, T), generator=g)
    s_ = torch.randn(B, T, generator=g)
    s_[:, 0] = 1e4
    G = torch.randn(B, T, D, generator=g)
    return h, h2, ids, s_, G


# ---- лише перечитування: знімки — листки з градієнтом -----------------------------------------------------
def reread_only(B, cdt=jnp.float32, eb=False, bf16_torch=False, seed=0, verbose=True):
    mem, rr, table = _modules(seed)
    h, h2, ids, s_, G = _inputs(B, seed + 7)
    with torch.no_grad(), stable_torch():
        _, snaps0, _ = R.torch_memory_pass(mem, h, SEG, False)
    faces = torch.stack([s[0] for s in snaps0]).clone().requires_grad_(True)
    conts = torch.stack([s[1] for s in snaps0]).clone().requires_grad_(True)
    pos = torch.stack([s[2] for s in snaps0])
    valid = torch.stack([s[3] for s in snaps0])
    n = T // SEG
    hq = h2.clone().requires_grad_(True)
    with stable_torch():
        snaps = [(faces[j], conts[j], pos[j], valid[j]) for j in range(n)]
        o, cost = R.torch_reread_sep(rr, hq, ids, table, snaps, SEG, s_, bf16_torch)
    loss = (o.float() * G).sum() + 100.0 * cost.float()
    loss.backward()
    gt = {k: tn(v.grad) for k, v in rr.named_parameters()}
    gt.update(table=tn(table.grad), h=tn(hq.grad), faces=tn(faces.grad), contents=tn(conts.grad))

    p = t2j(rr.state_dict())
    cfg = _rcfg(cdt, eb)
    sn = dict(pos=jnp.asarray(tn(pos)).astype(jnp.int32), valid=jnp.asarray(tn(valid)))

    def lossf(p, tab, hh, fa, co):
        oj, cj = reread_sep(p, hh, jnp.asarray(s_.numpy()), jnp.asarray(ids.numpy()), tab,
                            dict(sn, faces=fa, contents=co), cfg)
        return (oj.astype(jnp.float32) * jnp.asarray(G.numpy())).sum() + 100.0 * cj, (oj, cj)
    (lj, (oj, cj)), gs = jax.jit(jax.value_and_grad(lossf, (0, 1, 2, 3, 4), has_aux=True))(
        p, jnp.asarray(tn(table)), jnp.asarray(h2.numpy()), jnp.asarray(tn(faces)), jnp.asarray(tn(conts)))
    gj = {k: j2n(v) for k, v in gs[0].items()}
    gj.update(table=j2n(gs[1]), h=j2n(gs[2]), faces=j2n(gs[3]), contents=j2n(gs[4]))
    on, otn = j2n(oj), tn(o)
    res = {"o": diff(on, otn), "cost": diff(float(cj), float(cost)), "loss": diff(float(lj), float(loss)),
           "sel_mismatch": int(((on != 0).any(-1) != (otn != 0).any(-1)).sum()),
           "n_selected": int((otn != 0).any(-1).sum())}
    for k in gt:
        res["grad:" + k] = diff(gj[k], gt[k])
    if verbose:
        print(f"--- reread B={B} jax cdt={jnp.dtype(cdt).name} eb={eb} vs torch bf16_autocast={bf16_torch}  cost={float(cost):.6f}")
        for k, v in res.items():
            print(f"  {k:24s} {v}")
    return res


def test_reread_fp32_parity():
    for B in (1, 2):                                   # B=1 — REREAD_LEAN, B=2 — загальний (шматок-major) шлях
        res = reread_only(B)
        assert res["sel_mismatch"] == 0 and res["n_selected"] == B * (T // SEG - 1) * 16   # шматок 0: пам'ять порожня, o=0
        for k, v in res.items():
            if isinstance(v, tuple):
                assert v[1] < 1e-5, (B, k, v)


def test_reread_bf16():
    res = reread_only(2, jnp.bfloat16, True, True)
    assert res["sel_mismatch"] == 0
    assert res["o"][1] < 2e-2 and res["cost"][1] < 1e-2, res
    reread_only(2, jnp.bfloat16, False, True)          # наш вибір (fp32 einsum) — лише звіт


# ---- комбінований: пам'ять -> перечитування ---------------------------------------------------------------
def combined(B, cdt=jnp.float32, eb=False, bf16_torch=False, seed=1, verbose=True):
    mem, rr, table = _modules(seed)
    h, _, ids, s_, G = _inputs(B, seed + 7)
    hq = h.clone().requires_grad_(True)
    with stable_torch():
        reads, snaps, _ = R.torch_memory_pass(mem, hq, SEG, bf16_torch)
        h1 = hq + reads.float()
        o, cost = R.torch_reread_sep(rr, h1, ids, table, snaps, SEG, s_, bf16_torch)
    out = h1 + o.float()
    loss = (out * G).sum() + 100.0 * cost.float()
    loss.backward()
    gt = {"mem." + k: tn(v.grad) for k, v in mem.named_parameters()}
    gt.update({"rr." + k: tn(v.grad) for k, v in rr.named_parameters()})
    gt.update(table=tn(table.grad), h=tn(hq.grad))

    mc, rc = _mcfg(cdt, eb), _rcfg(cdt, eb)

    def lossf(pm, pr, tab, hh):
        rj, snj, _, _ = memory_pass(pm, hh, mc)
        h1j = hh + rj.astype(jnp.float32)
        oj, cj = reread_sep(pr, h1j, jnp.asarray(s_.numpy()), jnp.asarray(ids.numpy()), tab, snj, rc)
        outj = h1j + oj.astype(jnp.float32)
        return (outj * jnp.asarray(G.numpy())).sum() + 100.0 * cj, (outj, cj)
    (lj, (outj, cj)), gs = jax.jit(jax.value_and_grad(lossf, (0, 1, 2, 3), has_aux=True))(
        t2j(mem.state_dict()), t2j(rr.state_dict()), jnp.asarray(tn(table)), jnp.asarray(h.numpy()))
    gj = {"mem." + k: j2n(v) for k, v in gs[0].items()}
    gj.update({"rr." + k: j2n(v) for k, v in gs[1].items()})
    gj.update(table=j2n(gs[2]), h=j2n(gs[3]))
    res = {"out": diff(j2n(outj), tn(out)), "cost": diff(float(cj), float(cost)), "loss": diff(float(lj), float(loss))}
    for k in gt:
        res["grad:" + k] = diff(gj[k], gt[k])
    if verbose:
        print(f"--- combined memory+reread B={B} jax cdt={jnp.dtype(cdt).name} eb={eb} vs torch bf16={bf16_torch}")
        for k, v in res.items():
            print(f"  {k:28s} {v}")
    return res


def test_combined_fp32():
    res = combined(2)
    for k, v in res.items():
        # importance: малий градієнт зі взаємного знищення великих доданків (|g|max~0.2) — fp32-шум порядку
        # абсолютних похибок решти (~5e-5), тож відносний допуск ширший
        assert v[1] < (3e-3 if "importance" in k else 1e-5), (k, v)


def test_combined_bf16_report():
    combined(2, jnp.bfloat16, True, True)


# ---- юніт-тести -------------------------------------------------------------------------------------------
def test_halt_and_readable():
    """Ланцюг виживання і «лише раніші слова»: картка із закладкою на позиції qpos-2 -> читаються 2 слова."""
    from memarch_jax.reread import reread_core
    torch.manual_seed(0)
    rr = R.full2.Reread(D, d_face=64, d_book=128, top_k=1, max_read=4)
    p = t2j(rr.state_dict())
    p["halt_v.bias"] = p["halt_v.bias"] - 50.0          # halt ~ 0 там, де читається
    C = 3
    faces = jnp.zeros((1, C, 2, 64)).at[0, 0, 0, 0].set(1.0)
    hs = jnp.ones((1, 1, D))
    pos = jnp.array([[10, 0, 0]], jnp.int32)
    valid = jnp.array([[True, False, False]])
    ids = jnp.arange(32)[None]
    table = jnp.asarray(np.random.RandomState(0).randn(32, 128).astype(np.float32))
    cfg = RereadCfg(df=64, k=1, R=4, seg=1, cdt=jnp.float32)
    o, cost = reread_core(p, hs, ids, table, faces, jnp.zeros((1, C, D)), pos, valid, jnp.array([[12]]), jnp.array([0]), cfg)
    # читаються rpos 10, 11 (обидва < 12); halt≈0 -> обидва виживають; третій (12) нечитабельний -> halt=1, виживання 1
    # cost = sum_i survive_i * readable_i = 1 + 1 = 2 (w = 1 для єдиної картки)
    np.testing.assert_allclose(float(cost), 2.0, rtol=1e-5)


def test_anchor_selects_token0():
    """REREAD_ANCH: у шматку 0 вибирається слово 0 (s_[:,0] = 1e4)."""
    s = jnp.asarray(np.random.RandomState(1).randn(1, 2 * SEG).astype(np.float32)).at[:, 0].set(1e4)
    from memarch_jax.memory import stable_topk
    sel = jnp.sort(stable_topk(s.reshape(2, SEG), 16)[1], -1)
    assert int(sel[0, 0]) == 0


if __name__ == "__main__":
    test_halt_and_readable()
    test_anchor_selects_token0()
    test_reread_fp32_parity()
    test_reread_bf16()
    test_combined_fp32()
    test_combined_bf16_report()
    print("OK test_reread")
