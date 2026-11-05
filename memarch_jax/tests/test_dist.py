"""M9: одна синхронна модель на 8 CPU-чипах (dist.py) проти W=1 (model.train_loss), fp32.

(a) однакові дані на всіх 8 шардах: loss W=8 == loss W=1, зібрані градієнти == градієнти W=1, tau_new той самий;
(b) різні дані на шардах: loss W=8 == середнє 8 окремих W=1 loss (кожен шард — свій прогін), градієнти == середнє
    8 окремих W=1 градієнтів; tau_new == 0.9·tau + 0.1·mean_r(q_r) (спільний tau, pmean квантиля);
    ситуація == справжнє середнє вектора слів по всьому батчу;
(c) повний крок з Instinct (fp32, без SR) на однакових даних: нові ваги W=8 == W=1.
Місткість MoE і якорі — на чип (від локального N), тож (b) збігається з окремими W=1 прогонами точно (див. decisions).
Запуск: cd memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_dist.py"""
import os
import sys

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                               # noqa: E402
import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
from jax.sharding import PartitionSpec as P                      # noqa: E402

from memarch_jax import config, model, dist, instinct            # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
W = 8
V, K, CH, T = 4096, 512, 512, 256
_r = np.random.default_rng(0)
COUNTS = np.floor(1e6 / np.arange(1, V + 1) ** 1.1 * _r.uniform(0.5, 1.5, V)).astype(np.int64)
COUNTS[V - 1] = 0
LT = _r.uniform(1.0, 4.0, (3, 7)).astype(np.float32)
CFG = config.small(d=128, head_dim=64, E=16, heavy=2, light_a=1, light_b=1, final=1, window=64, seg=64, V=V,
                   short_v=K, ce_chunk=CH, T=T, B=1, dtype="f32", cap_align=0, moe_group=512,
                   mem_fresh=8, mem_dim=12, mem_dim_max=24, mem_write=8, mem_focus=4, mem_eb=False)
TAU = 7.5


def make_xy(B, seed):
    g = np.random.default_rng(seed)
    xy = np.minimum(g.zipf(1.3, (B, T + 1)), V - 2).astype(np.int64)
    for b in range(B):
        for _ in range(8):
            L = g.integers(3, 12)
            s = g.integers(0, T // 2)
            d_ = g.integers(L, T + 1 - s - L)
            xy[b, s + d_:s + d_ + L] = xy[b, s:s + L]
    return xy


def setup():
    params = model.init_params(jax.random.PRNGKey(0), CFG, LT)
    k = iter(jax.random.split(jax.random.PRNGKey(1), 64))       # ворушимо норми й ворота — щоб grads були «живі»
    params = jax.tree_util.tree_map_with_path(
        lambda p, x: x + 0.1 * jax.random.normal(next(k), x.shape) if "norm" in instinct.path_str(p) else x, params)
    consts = model.build_consts(CFG, COUNTS, LT)
    return params, consts


def w1(params, consts, xy):
    """Еталон W=1: (loss, grads, tau_new, terms)."""
    prep = model.make_prep(xy[:, :-1], CFG, consts)
    f = lambda p: model.train_loss(p, consts, jnp.asarray(xy, jnp.int32), prep, jnp.float32(TAU), CFG)
    (l, (tn, terms, _)), g = jax.jit(jax.value_and_grad(f, has_aux=True))(params)
    return float(l), g, float(tn), {k: float(v) for k, v in terms.items()}


def w8(params, consts, xy, mesh):
    prep = model.make_prep(xy[:, :-1], CFG, consts)
    ps = dist.put(params, mesh, dist.param_specs(params))
    gf = dist.make_grad_fn(CFG, consts, mesh, params)
    (l, (tn, terms, stats, per, sit)), g = gf(ps, dist.put(jnp.asarray(xy, jnp.int32), mesh, P("x")),
                                              dist.put(prep, mesh, P("x")), jnp.float32(TAU))
    return float(l), g, float(tn), {k: float(v) for k, v in terms.items()}, np.asarray(per), np.asarray(sit), ps


def flat(t):
    return {instinct.path_str(p): np.asarray(x, np.float64) for p, x in jax.tree_util.tree_flatten_with_path(t)[0]}


def cmp_grads(ga, gb, tol):
    fa, fb = flat(ga), flat(gb)
    assert set(fa) == set(fb)
    worst = 0.0
    bad = []
    for n in sorted(fa):
        a, b = fa[n], fb[n]
        assert a.shape == b.shape, (n, a.shape, b.shape)
        sc = max(np.abs(b).max(), 1e-30)
        r = np.abs(a - b).max() / sc
        worst = max(worst, r)
        t = 5e-3 if n.split(".")[0] in dist.VOCAB_KEYS else tol   # bf16-G у chunk_ce: множники lse інші (див. decisions)
        if r > t and np.abs(a - b).max() > 1e-7:
            bad.append((n, r))
    print(f"  grads: {len(fa)} тензорів, max rel {worst:.3e}")
    return bad


def test_a_identical():
    mesh = dist.make_mesh(W)
    params, consts = setup()
    xy1 = make_xy(1, 1)
    l1, g1, t1, terms1 = w1(params, consts, xy1)
    l8, g8, t8, terms8, per, sit, _ = w8(params, consts, np.tile(xy1, (W, 1)), mesh)
    print(f"(a) loss W=1 {l1:.8f}  W=8 {l8:.8f}  per-chip {per.min():.8f}..{per.max():.8f}  tau {t1:.6f}/{t8:.6f}")
    for k in terms1:
        print(f"    term {k:6s} {terms1[k]:+.8e} {terms8[k]:+.8e}")
    assert abs(l8 - l1) <= 1e-5 * abs(l1), (l1, l8)
    assert abs(t8 - t1) <= 1e-5 * abs(t1)
    sit1 = np.asarray(params["emb"][xy1[:, :-1]].astype(jnp.float32).mean((0, 1)))
    assert np.abs(sit - sit1).max() < 1e-5, np.abs(sit - sit1).max()
    bad = cmp_grads(g8, g1, 2e-3)     # шум fp32: W=1 з ce_chunk 512↔1024 сам дає 1.5e-3
    assert not bad, bad


def test_b_different():
    mesh = dist.make_mesh(W)
    params, consts = setup()
    xys = [make_xy(1, 10 + r) for r in range(W)]
    runs = [w1(params, consts, x) for x in xys]
    l8, g8, t8, _, per, sit, _ = w8(params, consts, np.concatenate(xys), mesh)
    lm = np.mean([r[0] for r in runs])
    print(f"(b) loss W=8 {l8:.8f}  mean(W=1) {lm:.8f}  max|per-chip − W=1| "
          f"{max(abs(per[i] - runs[i][0]) for i in range(W)):.3e}")
    assert abs(l8 - lm) <= 1e-5 * abs(lm)
    for i in range(W):
        assert abs(per[i] - runs[i][0]) <= 1e-5 * abs(runs[i][0]), (i, per[i], runs[i][0])
    # tau_new: W=1 дає 0.9·tau + 0.1·q_r → спільний = середнє
    tm = float(np.mean([r[2] for r in runs]))
    print(f"    tau W=8 {t8:.6f}  mean(W=1 tau_new) {tm:.6f}  розкид W=1 {np.ptp([r[2] for r in runs]):.4f}")
    assert abs(t8 - tm) <= 1e-5 * abs(tm)
    xa = np.concatenate(xys)[:, :-1]
    sit_true = np.asarray(params["emb"][xa].astype(jnp.float32).mean((0, 1)))
    assert np.abs(sit - sit_true).max() < 1e-5
    gmean = jax.tree_util.tree_map(lambda *g: sum(g) / W, *[r[1] for r in runs])
    bad = cmp_grads(g8, gmean, 2e-3)
    assert not bad, bad


def test_c_train_step():
    mesh = dist.make_mesh(W)
    params, consts = setup()
    ocfg = instinct.InstinctCfg(m_bf16=False, sr=False, use_pallas=False)
    spec = instinct.make_spec(params)
    d = CFG.d
    lr = 3e-4
    xy1 = make_xy(1, 1)
    # W=1
    l1, g1, t1, _ = w1(params, consts, xy1)
    sit1 = params["emb"][xy1[:, :-1]].astype(jnp.float32).mean((0, 1))
    st = instinct.init(params, d, ocfg, spec)
    p1, _ = jax.jit(lambda p, g, s: instinct.step(p, g, s, lr=lr, situation=sit1, loss=l1, spec=spec, cfg=ocfg))(
        params, g1, st)
    # W=8
    xy8 = np.tile(xy1, (W, 1))
    prep = model.make_prep(xy8[:, :-1], CFG, consts)
    pspec = dist.param_specs(params)
    ps = dist.put(params, mesh, pspec)
    st8 = instinct.init(params, d, ocfg, spec)
    step = dist.make_train_step(CFG, consts, mesh, params, ocfg, spec)
    p8, o8, t8, l8, _, _ = step(ps, st8, dist.put(jnp.asarray(xy8, jnp.int32), mesh, P("x")),
                                dist.put(prep, mesh, P("x")), jnp.float32(TAU), lr)
    print(f"(c) крок: loss {l1:.8f}/{float(l8):.8f}  шардування emb після кроку: {p8['emb'].sharding.spec}")
    fa, fb, f0 = flat(p8), flat(p1), flat(params)
    worst, rows = 0.0, []
    fg = flat(g1)
    noise = sorted(n for n in fa if np.abs(fg[n]).max() < 1e-7)   # grad ~1e-8 = шум fp32; нормований крок — випадковий
    print("    пропущено (max|grad| < 1e-7):", noise)
    for n in fa:
        if n in noise:
            continue
        dd = np.abs(fb[n] - f0[n]).max()                         # величина кроку
        e = np.abs(fa[n] - fb[n]).max()
        worst = max(worst, e / max(dd, 1e-12))
        rows.append((e / max(dd, 1e-12), n, e, dd))
    for r_, n, e, dd in sorted(rows, reverse=True)[:5]:
        print(f"    {n:28s} max|W8−W1| {e:.3e}  max|крок| {dd:.3e}  відн. {r_:.3e}")
    print(f"    нові ваги W=8 vs W=1: max |Δ| / |крок| = {worst:.3e}")
    assert worst < 1e-2, worst      # шум градієнтів (≤2e-3 відн., vocab — bf16-G) після нормування кроку


if __name__ == "__main__":
    print("пристроїв:", len(jax.devices()))
    test_a_identical()
    test_b_different()
    test_c_train_step()
    print("OK")
