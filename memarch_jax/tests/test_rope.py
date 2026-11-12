"""Паритет вагонетки (memarch_jax/rope.py) з PyTorch-еталоном (D:\\ai\\memarch\\rope_mem.py, rope_kernel.py), лише CPU.

Що справжнє, а що заміна:
  * rope_mem.anchors_from_surprise, rope_mem.rope_attention (щільна), rope_kernel.route_torch,
    rope_kernel.route_triton_hier (усе, крім ядер) і rope_kernel.rope_attention_kernel — справжній код еталона.
  * Triton-ядра працюють лише на GPU, тому тут підмінені: _group_score_kernel, _route_hier_kernel (та сама формула
    чесними torch-операціями, але f32 замість tf32) і _RopeAttn (щільний softmax над вікном ∪ зібраними словами).
Запуск: cd D:\\ai\\memarch_jax && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONIOENCODING=utf-8 python tests/test_rope.py
"""
import dataclasses
import os
import sys

sys.dont_write_bytecode = True                  # нічого не писати в D:\ai\memarch (__pycache__)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
os.environ.setdefault("ROPE_HALF", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, r"D:\ai\memarch")

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402
import jax                                      # noqa: E402
import jax.numpy as jnp                         # noqa: E402

import rope_mem                                 # noqa: E402
import rope_kernel                              # noqa: E402
from memarch_jax import rope as R               # noqa: E402
from memarch_jax.prims import rope_half, rope_cs  # noqa: E402

NEG = float("-inf")
REPORT = []


# ---------------------------------------------------------------- torch: RoPE і підміни Triton-ядер

def rope_fn(x):
    """Той самий RoPE, що prims.rope_half (пари — половинки): кути беремо з JAX, щоб порівнювати лише вагонетку."""
    B, H, T, D = x.shape
    cos, sin = (torch.tensor(np.asarray(a)) for a in rope_cs(T, D))
    xf = x.float()
    x1, x2 = xf[..., :D // 2], xf[..., D // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)


class _Fake:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        return lambda *a, **kw: self.fn(grid, *a, **kw)


def group_score_shim(grid, Q, S1, S2, OUT, T, NGP, NMB, D, BM, BG, QUEST, AGG, num_warps=4):
    assert not QUEST and AGG == 1
    B_H = Q.shape[0] * Q.shape[1]
    q = Q.reshape(B_H, T, D).float()
    s1 = S1.reshape(B_H, NGP, D).float()
    o = OUT.reshape(B_H, NMB, NGP)
    for bh in range(B_H):
        for m in range(NMB):
            qb = q[bh, m * BM:min(T, (m + 1) * BM)]
            o[bh, m] = (qb @ s1[bh].T).mean(0)


def route_hier_shim(grid, Q, KL, LV, LPOS, NXT, AVAL, NAB, NFB, SEL, TI, TV, TP, T, A, AP, H, NMB, sc, D, BM, BA, LKP, KK,
                    KP, PREC, POS, NB, GB, num_warps=4):
    B_H = Q.shape[0] * Q.shape[1]
    B = B_H // H
    q = Q.reshape(B_H, T, D).float()
    kl = KL.reshape(B_H, AP * LKP, D).float()
    lv = LV.reshape(B, AP * LKP).bool()
    lpos = LPOS.reshape(B, AP * LKP)
    nxt, aval = NXT.reshape(B, A), AVAL.reshape(B, A).bool()
    nab, nfb = NAB.reshape(B, NMB), NFB.reshape(B, NMB)
    sel = SEL.reshape(B_H, NMB, NB)
    ti, tv, tp = TI.reshape(B_H, T, KK), TV.reshape(B_H, T, KK), TP.reshape(B_H, T, KK)
    for bh in range(B_H):
        b = bh // H
        for m in range(NMB):
            cand = []
            for g in sel[bh, m].tolist():
                if g >= 0:
                    cand += list(range(g * GB * BA, (g + 1) * GB * BA))
            cand += list(range(int(nfb[b, m]) * BA, int(nab[b, m]) * BA))
            ii = torch.arange(m * BM, min(T, (m + 1) * BM))
            if not cand:
                tv[bh, ii] = 0
                continue
            ca = torch.tensor(cand)
            rows = (ca[:, None] // BA * LKP + torch.arange(LKP)[None]) * BA + (ca[:, None] % BA)   # (C,LKP)
            s = (q[bh, ii] @ kl[bh, rows.view(-1)].T * sc).view(len(ii), len(cand), LKP)
            s = s.masked_fill(~lv[b, rows.view(-1)].view(1, len(cand), LKP), NEG)
            vmax, li = s.max(-1)
            cvalid = ca < A
            cc = ca.clamp(max=A - 1)
            ok = aval[b, cc] & cvalid & (nxt[b, cc][None] <= ii[:, None])
            vmax = vmax.masked_fill(~ok, NEG)
            top = vmax.topk(min(KK, len(cand)), -1)
            pad = KK - top.indices.shape[1]
            ti[bh, ii, :top.indices.shape[1]] = ca[top.indices].to(ti.dtype)
            tv[bh, ii, :top.indices.shape[1]] = (top.values > NEG).to(tv.dtype)
            if pad:
                tv[bh, ii, top.indices.shape[1]:] = 0
            lbest = li.gather(1, top.indices)
            rb = (ca[top.indices] // BA * LKP + lbest) * BA + ca[top.indices] % BA
            tp[bh, ii, :top.indices.shape[1]] = lpos[b, rb].to(tp.dtype)


class AttnShim:
    """Щільний еквівалент _RopeAttn: один softmax над вікном ∪ зібраними словами, bias на зібраних, autograd."""

    @staticmethod
    def apply(qr, kr, v, gs, gl, gb, win, gi, segs, sege):
        B, H, T, hd = qr.shape
        s = (qr.float() @ kr.float().transpose(-1, -2)) * hd ** -0.5
        i = torch.arange(T)
        local = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < win)
        anyg = torch.zeros(B, H, T, T, dtype=torch.bool)
        bias = torch.zeros(B, H, T, T)
        for t in range(gs.shape[-1]):
            st, ln = gs[..., t].long()[..., None], gl[..., t].long()[..., None]
            m = (i[None, None, None] >= st) & (i[None, None, None] < st + ln)
            anyg = anyg | m
            if gb is not None:
                bias = bias + gb[..., t, None] * m
        s = (s + bias).masked_fill(~(local[None, None] | anyg), NEG)
        p = s.softmax(-1)
        return (p @ v.float()).to(v.dtype), (p * anyg).sum(-1).detach()


rope_kernel._group_score_kernel = _Fake(group_score_shim)
rope_kernel._route_hier_kernel = _Fake(route_hier_shim)
rope_kernel._RopeAttn = AttnShim


def env(**kw):
    defaults = dict(ROPE_FACE="chain", ROPE_IMPL="dense", ROPE_RGRAD="0", ROPE_EXCL_WIN="0", ROPE_ROUTE_ROT="0",
                    ROPE_KROUTE="torch", ROPE_HIER_G="64", ROPE_HIER_NB="16", ROPE_HIER_SUM="mean", ROPE_HIER_AGG="mean",
                    ROPE_HIER_BM="64")
    defaults.update({k: str(v) for k, v in kw.items()})
    os.environ.update(defaults)
    rope_mem.ROPE_FACE, rope_mem.ROPE_SOFT = "chain", False
    rope_mem.ROPE_LINKS, rope_mem.ROPE_WIN, rope_mem.ROPE_SPAN_CAP = 8, int(kw.get("WIN", 64)), int(kw.get("CAP", 256))


# ---------------------------------------------------------------- дані

def data(seed, B=2, H=2, T=2048, hd=8, cap=0.25, gap=True):
    g = np.random.default_rng(seed)
    q, k, v = (g.normal(0, 1, (B, H, T, hd)).astype(np.float32) * s for s in (2.0, 2.0, 1.0))
    # здивування з довгими проміжками без якорів: володіння понад span_cap і понад links
    s = g.exponential(1.0, (B, T)).astype(np.float32)
    if gap:
        for b in range(B):
            for _ in range(4):
                a = int(g.integers(1, T - 600))
                s[b, a:a + int(g.integers(280, 600))] = 0.0
    s[:, 0] = 1e4
    return q, k, v, s, max(1, round(T * cap))


def anchors_pair(s, tau, A):
    own0, apos0, av0 = rope_mem.anchors_from_surprise(torch.tensor(s), torch.tensor(tau), A)[1:]
    own1, apos1, av1 = R.anchors_from_surprise(jnp.asarray(s), jnp.asarray(tau), A)
    assert np.array_equal(own0.numpy(), np.asarray(own1)), "own"
    assert np.array_equal(apos0.numpy(), np.asarray(apos1)), "apos"
    assert np.array_equal(av0.numpy(), np.asarray(av1)), "avalid"
    return (own0, apos0, av0), (own1, apos1, av1)


def to_jax_cfg(**kw):
    base = dict(K=4, links=8, win=64, span_cap=256, gather_cap=None, route="exact", excl_win=False, route_rot=False,
                rgrad=False, G=64, NB=16, BM=64, chunk=256)
    base.update(kw)
    return R.RopeCfg(**base)


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def compare_route(tag, ti0, tv0, tp0, ti1, tv1, tp1):
    ti0, tv0, tp0 = (t.numpy().astype(int) for t in (ti0, tv0.int(), tp0))
    ti1, tv1, tp1 = (np.asarray(t).astype(int) for t in (ti1, tv1, tp1))
    n, bad, bad_pos = 0, 0, 0
    for idx in np.ndindex(ti0.shape[:3]):
        a0 = {int(ti0[idx][t]): int(tp0[idx][t]) for t in range(ti0.shape[-1]) if tv0[idx][t]}
        a1 = {int(ti1[idx][t]): int(tp1[idx][t]) for t in range(ti1.shape[-1]) if tv1[idx][t]}
        n += 1
        if set(a0) != set(a1):
            bad += 1
        elif a0 != a1:
            bad_pos += 1
    print(f"  {tag}: route queries {n}, anchor-set mismatches {bad}, best-link mismatches {bad_pos}")
    REPORT.append((tag, "route", n, bad, bad_pos))
    assert bad == 0 and bad_pos == 0, tag


def cotangent(shape, seed):
    return np.random.default_rng(seed + 123).normal(0, 1, shape).astype(np.float32)


def run_torch(fn, q, k, v, ct):
    qt, kt, vt = (torch.tensor(a, requires_grad=True) for a in (q, k, v))
    out, share = fn(qt, kt, vt)
    (out.float() * torch.tensor(ct)).sum().backward()
    return out.detach().float().numpy(), float(share), qt.grad.numpy(), kt.grad.numpy(), vt.grad.numpy()


def run_jax(fn, q, k, v, ct):
    def f(a, b, c):
        o, sh, st = fn(a, b, c)
        return (o.astype(jnp.float32) * ct).sum(), (o, sh, st)
    (val, (o, sh, st)), (gq, gk, gv) = jax.jit(jax.value_and_grad(f, (0, 1, 2), has_aux=True))(
        jnp.asarray(q), jnp.asarray(k), jnp.asarray(v))
    return np.asarray(o, np.float32), float(sh), np.asarray(gq), np.asarray(gk), np.asarray(gv), st


def compare_attn(tag, r0, r1, tol):
    names = ("out", "dq", "dk", "dv")
    errs = dict(share=abs(r0[1] - r1[1]))
    for name, a0, a1 in zip(names, (r0[0], r0[2], r0[3], r0[4]), (r1[0], r1[2], r1[3], r1[4])):
        errs[name] = rel(a1, a0)
    print(f"  {tag}: " + "  ".join(f"{k} {v:.2e}" for k, v in errs.items()) + f"  share {r0[1]:.4f}")
    REPORT.append((tag, "attn", errs))
    for name, e in errs.items():
        assert e <= tol, (tag, name, e)


# ---------------------------------------------------------------- перевірки

def jfn(rcfg):
    def f(q, k, v, own, apos, av):
        qr, kr = rope_half(q), rope_half(k)
        return R.rope_attention(qr, kr, q, k, v, own, apos, av, rcfg)
    return f


def test_anchors():
    for seed in range(4):
        q, k, v, s, A = data(seed)
        anchors_pair(s, 0.7, A)
    print("(1) anchors OK: own/apos/avalid exact on 4 seeds, B=2, T=2048")


def test_dense_exact_route():
    """Вимкнені нові прапорці: JAX route=exact проти справжньої щільної rope_mem.rope_attention (RGRAD 0 і 1)."""
    for seed, rg in [(0, 0), (1, 1), (2, 1)]:
        env(ROPE_RGRAD=rg)
        q, k, v, s, A = data(seed)
        (own0, apos0, av0), (own1, apos1, av1) = anchors_pair(s, 0.7, A)
        ct = cotangent(v.shape, seed)
        fn0 = lambda a, b, c: rope_mem.rope_attention(a, b, c, rope_fn, own0, apos0, av0, K=4, win=64, span_cap=256)
        r0 = run_torch(fn0, q, k, v, ct)
        rcfg = to_jax_cfg(rgrad=bool(rg))
        r1 = run_jax(lambda a, b, c: jfn(rcfg)(a, b, c, own1, apos1, av1), q, k, v, jnp.asarray(ct))
        compare_attn(f"dense exact rgrad={rg} seed={seed}", r0, r1, 2e-3)
        assert float(r1[5]["rope_overflow"]) == 0.0
    print("(2) dense reference parity OK")


def route_ref(q, k, own0, apos0, av0, excl, kk=4, Lk=8):
    """Справжній rope_kernel.route_torch у повернутих векторах (ROUTE_ROT=1)."""
    with torch.no_grad():
        return rope_kernel.route_torch(rope_fn(torch.tensor(q)).float(), rope_fn(torch.tensor(k)).float(), own0, apos0,
                                       av0, kk, Lk)


def test_kernel_exact(dtype="f32"):
    """Цільові прапорці без hier: справжня обгортка rope_attention_kernel (ROPE_KROUTE=torch), ROT=1, EXCL=1, RGRAD=1."""
    for seed in range(3):
        env(ROPE_RGRAD=1, ROPE_EXCL_WIN=1, ROPE_ROUTE_ROT=1, ROPE_KROUTE="torch")
        q, k, v, s, A = data(seed)
        (own0, apos0, av0), (own1, apos1, av1) = anchors_pair(s, 0.7, A)
        rcfg = to_jax_cfg(excl_win=True, route_rot=True, rgrad=True)
        qr, kr = rope_half(jnp.asarray(q)), rope_half(jnp.asarray(k))
        ti1, tv1, tp1 = R.route(qr, kr, jnp.asarray(q), jnp.asarray(k), own1, apos1, av1, rcfg)
        ti0, tv0, tp0 = route_ref(q, k, own0, apos0, av0, True)
        compare_route(f"kernel exact seed={seed}", ti0, tv0, tp0, ti1, tv1, tp1)
        ct = cotangent(v.shape, seed)
        fn0 = lambda a, b, c: rope_kernel.rope_attention_kernel(a, b, c, rope_fn, own0, apos0, av0, K=4, win=64, span_cap=256)
        r0 = run_torch(fn0, q, k, v, ct)
        r1 = run_jax(lambda a, b, c: jfn(rcfg)(a, b, c, own1, apos1, av1), q, k, v, jnp.asarray(ct))
        compare_attn(f"kernel exact seed={seed}", r0, r1, 2e-3)
    print("(3) kernel wrapper exact route parity OK")


def test_kernel_hier():
    """Цільовий режим: справжній route_triton_hier (ядра підмінені), G=64 NB=16 (усі групи — бо NG мало) і NB=2 (строгий відбір)."""
    for seed, nb, G, T in [(0, 16, 64, 2048), (1, 2, 64, 2048), (2, 2, 16, 2048), (3, 3, 32, 4096)]:
        env(ROPE_RGRAD=1, ROPE_EXCL_WIN=1, ROPE_ROUTE_ROT=1, ROPE_KROUTE="hier", ROPE_HIER_G=G, ROPE_HIER_NB=nb)
        q, k, v, s, A = data(seed, T=T)
        (own0, apos0, av0), (own1, apos1, av1) = anchors_pair(s, 0.7, A)
        rcfg = to_jax_cfg(route="hier", excl_win=True, route_rot=True, rgrad=True, G=G, NB=nb)
        ct = cotangent(v.shape, seed)
        fn0 = lambda a, b, c: rope_kernel.rope_attention_kernel(a, b, c, rope_fn, own0, apos0, av0, K=4, win=64, span_cap=256)
        qr, kr = rope_half(jnp.asarray(q)), rope_half(jnp.asarray(k))
        ti1, tv1, tp1 = R.route(qr, kr, jnp.asarray(q), jnp.asarray(k), own1, apos1, av1, rcfg)
        with torch.no_grad():
            kk, Lk = 4, 8
            qt, kt = torch.tensor(q), torch.tensor(k)
            ti0, tv0, tp0 = rope_kernel.route_triton_hier(rope_fn(qt).float(), rope_fn(kt).float(), own0, apos0, av0, kk, Lk,
                                                          need_pos=True, group_q=qt, group_k=kt)
        compare_route(f"kernel hier NB={nb} G={G} T={T} seed={seed}", ti0, tv0, tp0, ti1, tv1, tp1)
        r0 = run_torch(fn0, q, k, v, ct)
        r1 = run_jax(lambda a, b, c: jfn(rcfg)(a, b, c, own1, apos1, av1), q, k, v, jnp.asarray(ct))
        compare_attn(f"kernel hier NB={nb} G={G} T={T} seed={seed}", r0, r1, 2e-3)
    print("(4) kernel hier parity OK")


def test_bf16():
    """bf16-входи (як на TPU): ROT=1, exact."""
    env(ROPE_RGRAD=1, ROPE_EXCL_WIN=1, ROPE_ROUTE_ROT=1, ROPE_KROUTE="torch")
    q, k, v, s, A = data(5)
    (own0, apos0, av0), (own1, apos1, av1) = anchors_pair(s, 0.7, A)
    qb, kb, vb = (torch.tensor(a).to(torch.bfloat16) for a in (q, k, v))
    with torch.no_grad():
        o0, sh0 = rope_kernel.rope_attention_kernel(qb, kb, vb, rope_fn, own0, apos0, av0, K=4, win=64, span_cap=256)
    rcfg = to_jax_cfg(excl_win=True, route_rot=True, rgrad=True)
    qj, kj, vj = (jnp.asarray(a).astype(jnp.bfloat16) for a in (q, k, v))
    o1, sh1, _ = jax.jit(lambda a, b, c: jfn(rcfg)(a, b, c, own1, apos1, av1))(qj, kj, vj)
    e = rel(np.asarray(o1, np.float32), o0.float().numpy())
    print(f"(5) bf16 inputs: out rel {e:.2e}  share {float(sh0):.4f} vs {float(sh1):.4f}")
    REPORT.append(("bf16", "out", e))
    assert e < 2e-2


def test_static_shapes():
    """Той самий скомпільований код на різних якорях і щільностях: без перетрасування; виявлення переповнення."""
    q, k, v, s, A = data(0)
    rcfg = to_jax_cfg(route="hier", excl_win=True, route_rot=True, rgrad=True, NB=2)
    traces = {"n": 0}

    def f(a, b, c, own, apos, av):
        traces["n"] += 1
        return jfn(rcfg)(a, b, c, own, apos, av)
    fj = jax.jit(f)
    for seed, tau in ((0, 0.7), (1, 2.5), (2, 100.0)):
        q, k, v, s, A = data(seed)
        own, apos, av = R.anchors_from_surprise(jnp.asarray(s), jnp.asarray(tau), A)
        o, sh, st = fj(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), own, apos, av)
        assert np.isfinite(np.asarray(o)).all()
    assert traces["n"] == 1, traces
    # мала місткість: той самий результат для тексту без переповнення, stat > 0 — для тексту з довгими володіннями
    q, k, v, s, A = data(0)
    own, apos, av = R.anchors_from_surprise(jnp.asarray(s), jnp.asarray(0.7), A)
    ex = jfn(to_jax_cfg(excl_win=True, route_rot=True, rgrad=True))(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), own, apos, av)
    cut = jfn(to_jax_cfg(excl_win=True, route_rot=True, rgrad=True, gather_cap=8))(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v),
                                                                                  own, apos, av)
    assert float(ex[2]["rope_overflow"]) == 0.0 and float(cut[2]["rope_overflow"]) > 0.0
    print(f"(6) static shapes OK: 1 trace for 3 anchor densities; overflow exact=0, gather_cap=8 → "
          f"{float(cut[2]['rope_overflow']):.3f} of gathered words lost")


def test_model_dist():
    """Модель із cfg.rope: W=1 проти W=8 shard_map (loss і градієнти; один виклик, без оптимізатора і без кроків навчання)."""
    import test_dist as td
    from memarch_jax import model
    cfg = dataclasses.replace(td.CFG, rope=True, rope_win=16, rope_span_cap=32, rope_hier_g=16, rope_hier_nb=2,
                              rope_hier_bm=16, rope_chunk=64)
    td.CFG = cfg
    mesh = td.dist.make_mesh(td.W)
    params, consts = td.setup()
    assert all(n not in params for n in ("memory", "reread", "book")), "rope: mem2/reread/book мають бути відсутні"
    tau0 = model.init_tau(cfg, td.TAU)

    def w1(xy):
        prep = model.make_prep(xy[:, :-1], cfg, consts)
        f = lambda p: model.train_loss(p, consts, jnp.asarray(xy, jnp.int32), prep, tau0, cfg)
        (l, (tn, terms, st)), g = jax.jit(jax.value_and_grad(f, has_aux=True))(params)
        return float(l), g, np.asarray(tn), {k: float(st[k]) for k in ("rope_share", "rope_anchors", "rope_overflow")}

    xys = [td.make_xy(1, 20 + r) for r in range(td.W)]
    runs = [w1(x) for x in xys]
    xy8 = np.concatenate(xys)
    prep = model.make_prep(xy8[:, :-1], cfg, consts)
    gf = td.dist.make_grad_fn(cfg, consts, mesh, params)
    ps = td.dist.put(params, mesh, td.dist.param_specs(params))
    from jax.sharding import PartitionSpec as P
    (l8, (t8, terms8, st8, per, sit)), g8 = gf(ps, td.dist.put(jnp.asarray(xy8, jnp.int32), mesh, P("x")),
                                               td.dist.put(prep, mesh, P("x")), tau0)
    l8 = float(l8)
    lm = np.mean([r[0] for r in runs])
    print("    W=1 stats:", runs[0][3], " tau_new", runs[0][2])
    print(f"(7) rope model: W=8 loss {l8:.8f}  mean(W=1) {lm:.8f}  |d|={abs(l8 - lm):.2e}  tau_new {np.asarray(t8)}")
    assert abs(l8 - lm) <= 1e-5 * abs(lm)
    gmean = jax.tree_util.tree_map(lambda *g: sum(g) / td.W, *[r[1] for r in runs])
    bad = td.cmp_grads(g8, gmean, 2e-3)
    assert not bad, bad
    assert np.abs(np.asarray(g8["light_b"][0]["qkv"])).max() > 0


def test_fast_flags():
    """route_lazy і dense_read: ті самі вихід і градієнти, що й шлях із gather (місткість точна, без обрізання)."""
    q, k, v, s, A = data(0)
    own, apos, av = R.anchors_from_surprise(jnp.asarray(s), jnp.asarray(0.7), A)
    qj, kj, vj = jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)
    for dt in (jnp.float32, jnp.bfloat16):
        q2, k2, v2 = qj.astype(dt), kj.astype(dt), vj.astype(dt)

        def run(rcfg):
            def f(a, b, c):
                o, sh, st = jfn(rcfg)(a, b, c, own, apos, av)
                return (o.astype(jnp.float32) ** 2).sum(), (o, sh, st)
            (l, (o, sh, st)), g = jax.value_and_grad(f, argnums=(0, 1, 2), has_aux=True)(q2, k2, v2)
            return np.asarray(o.astype(jnp.float32)), float(sh), [np.asarray(x.astype(jnp.float32)) for x in g]

        base = dict(excl_win=True, route_rot=True, rgrad=True, NB=2)
        o0, s0, g0 = run(to_jax_cfg(route="hier", **base))
        tol = 1e-5 if dt == jnp.float32 else 2e-2
        for route_kind, flag in (("hier", "route_lazy"), ("hier", "dense_read"), ("exact", "route_lazy"),
                                 ("exact", "route_bf16"), ("hier", "route_bf16")):
            if route_kind == "exact":
                b2 = {k: v for k, v in base.items() if k != "NB"}
                o0, s0, g0 = run(to_jax_cfg(route="exact", **b2))
                rc = dataclasses.replace(to_jax_cfg(route="exact", **b2), **{flag: True})
            else:
                o0, s0, g0 = run(to_jax_cfg(route="hier", **base))
                rc = dataclasses.replace(to_jax_cfg(route="hier", **base), **{flag: True})
            o1, s1, g1 = run(rc)
            d = np.abs(o1 - o0).max() / max(np.abs(o0).max(), 1e-9)
            dg = max(np.abs(a - b).max() / max(np.abs(b).max(), 1e-9) for a, b in zip(g1, g0))
            print(f"(8) {route_kind} {flag} {dt.__name__}: rel out {d:.2e}, rel grad {dg:.2e}, share {s1:.5f} vs {s0:.5f}")
            assert d <= tol and dg <= tol and abs(s1 - s0) <= tol, (flag, dt, d, dg, s1, s0)


def test_pallas_vs_torch():
    """Ядра rope_pallas (interpret) проти справжньої обгортки PyTorch rope_attention_kernel: вибір, вихід, градієнти."""
    for seed in range(2):
        env(ROPE_RGRAD=1, ROPE_EXCL_WIN=1, ROPE_ROUTE_ROT=1, ROPE_KROUTE="torch")
        q, k, v, s, A = data(seed)
        (own0, apos0, av0), (own1, apos1, av1) = anchors_pair(s, 0.7, A)
        rcfg = dataclasses.replace(to_jax_cfg(excl_win=True, route_rot=True, rgrad=True), pallas=True)
        ct = cotangent(v.shape, seed)
        fn0 = lambda a, b, c: rope_kernel.rope_attention_kernel(a, b, c, rope_fn, own0, apos0, av0, K=4, win=64, span_cap=256)
        r0 = run_torch(fn0, q, k, v, ct)
        def f1(a, b, c):
            o, sh, st = R.rope_attention(rope_half(a), rope_half(b), a, b, c, own1, apos1, av1, rcfg)
            return o, sh, {"rope_overflow": st["rope_overflow"]}
        r1 = run_jax(f1, q, k, v, jnp.asarray(ct))
        compare_attn(f"pallas vs torch seed={seed}", r0, r1, 2e-3)
    print("(9) pallas kernels == PyTorch reference OK")


if __name__ == "__main__":
    print("devices:", jax.devices()[0].platform, len(jax.devices()))
    test_anchors()
    test_dense_exact_route()
    test_kernel_exact()
    test_kernel_hier()
    test_bf16()
    test_static_shapes()
    test_model_dist()
    print("OK")
    test_fast_flags()
    test_pallas_vs_torch()
