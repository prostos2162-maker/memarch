"""--opt_gearbox у Pallas-ядрах (15.11): ||g||² — з проходу А двопрохідних ядер (instinct_hs_2d / instinct_hs3), без
окремого проходу по g. Перевіряємо (CPU, interpret=True), кілька кроків зі сплеском градієнта (передача вниз):
  (1) ядро з Σg² == старий ядерний шлях (окремий jnp.sum(g²) перед ядром, GB_KERN=0) — ваги, m, df, gidx;
  (2) ядро == чистий jnp-шлях (use_pallas=False) до точності округлення SR;
  (3) без gearbox — ядро побітово як було (gear=None дає ті самі m, w, що й старий виклик);
  (4) шардований hs3 (EP по експертах, shard_map, psum Σg²) — те саме, що й без шардування для df/gidx.
Запуск: PYTHONIOENCODING=utf-8 python tests/test_gearbox_pallas.py"""
import dataclasses
import os
import sys

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                               # noqa: E402
import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402

from memarch_jax import instinct, pallas_instinct as pi          # noqa: E402

CALLS = []
_orig = instinct._pallas_call


def _spy(kfn, args, sc, it, shard, leaf, li, gear=None, dst=None):
    CALLS.append((leaf, gear is not None))
    return _orig(kfn, args, sc, it, shard, leaf, li, gear=gear, dst=dst)


instinct._pallas_call = _spy

STEPS = 7
SPIKE = {3: 8.0, 5: 0.2}          # крок → множник градієнта (сплеск → вниз; тиша → вгору)


def make(seed=0):
    r = np.random.default_rng(seed)
    params = {"a": {"w": jnp.asarray(r.normal(0, 0.05, (64, 256)), jnp.bfloat16),
                    "b": jnp.asarray(r.normal(0, 0.05, (128, 128)), jnp.bfloat16)},
              "ex": jnp.asarray(r.normal(0, 0.05, (4, 64, 256)), jnp.bfloat16)}
    grads = [jax.tree_util.tree_map(lambda x, k=k: jnp.asarray(
        r.normal(0, 1e-3, x.shape) * SPIKE.get(k, 1.0) * (1 + 0.3 * r.uniform()), jnp.bfloat16), params)
        for k in range(STEPS)]
    return params, grads


def run(use_pallas, gearbox=True, gb_kern=True, shard=None, seed=0):
    instinct.GB_KERN = gb_kern
    params, grads = make(seed)
    spec = instinct.make_spec(params)
    cfg = instinct.InstinctCfg(m_bf16=True, sr=True, use_pallas=use_pallas, interpret=True, hs_lag=False,
                               gearbox=gearbox, damp=2.0 if gearbox else 0.0)
    st = instinct.init(params, 8, cfg, spec)
    sit = jnp.ones((8,), jnp.float32)

    def f(p, g, st, loss):
        return instinct.step(p, g, st, lr=3e-3, situation=sit, loss=loss, spec=spec, cfg=cfg, shard=shard)
    f = jax.jit(f)
    if shard is not None:
        from jax.sharding import NamedSharding
        mesh, specs = shard
        params = {k: (jax.device_put(v, NamedSharding(mesh, specs["ex"])) if k == "ex" else v)
                  for k, v in params.items()}
    hist = []
    for k in range(STEPS):
        params, st = f(params, grads[k], st, jnp.float32(3.0 - 0.1 * k))
        hist.append((np.asarray(st["df"]).copy() if "df" in st else None, np.asarray(st["gidx"]).copy() if "gidx" in st else None))
    out = {instinct.path_str(p): np.asarray(x.astype(jnp.float32)) for p, x in
           jax.tree_util.tree_flatten_with_path(params)[0]}
    mm = {k: np.asarray(v.astype(jnp.float32)) for k, v in st["m"].items()}
    return out, mm, hist


def close(a, b, what, ulps=2.0):
    worst = 0.0
    for k in a:
        tol = np.abs(a[k]) * 2.0 ** -7 * ulps + 1e-30
        frac = float((np.abs(a[k] - b[k]) > tol).mean())
        worst = max(worst, frac)
    print(f"  {what}: частка елементів поза {ulps:g} ulp bf16 — {worst:.2e}")
    return worst


def main():
    CALLS.clear()
    pk, mk, hk = run(True)
    used = sorted({l for l, g in CALLS if g})
    print(f"ядро з Σg²: листки {used}")
    assert set(used) == {"a.w", "a.b", "ex"}, used
    gears = [h[1].tolist() for h in hk]
    print("передачі по кроках:", gears)
    print("множники df:", [np.round(h[0], 4).tolist() for h in hk])
    assert len({tuple(g) for g in gears}) > 1, "коробка жодного разу не перемкнулась — тест нічого не перевіряє"

    # (1) проти старого ядерного шляху (окремий прохід ||g||²)
    CALLS.clear()
    po, mo, ho = run(True, gb_kern=False)
    assert not any(g for _, g in CALLS)
    for (d1, g1), (d2, g2) in zip(hk, ho):
        assert (g1 == g2).all(), (g1, g2)
        np.testing.assert_allclose(d1, d2, rtol=1e-6, atol=0)
    nb = sum(int((pk[k] != po[k]).sum()) for k in pk) + sum(int((mk[k] != mo[k]).sum()) for k in mk)
    print(f"(1) проти окремого проходу: gidx/df збіг; різних bf16-елементів ваг+m: {nb}")
    assert close(pk, po, "ваги") < 1e-4 and close(mk, mo, "m") < 1e-4

    # (2) проти чистого jnp
    pj, mj, hj = run(False)
    for (d1, g1), (d2, g2) in zip(hk, hj):
        assert (g1 == g2).all(), (g1, g2)
        np.testing.assert_allclose(d1, d2, rtol=1e-5, atol=0)
    print("(2) проти jnp: gidx/df збіг")
    assert close(pk, pj, "ваги") < 1e-3 and close(mk, mj, "m") < 1e-3

    # (3) без gearbox: ядро gear=None побітово як до зміни (gear-гілка не чіпає старий шлях)
    CALLS.clear()
    pn, mn, _ = run(True, gearbox=False)
    assert CALLS and not any(g for _, g in CALLS)
    r = np.random.default_rng(5)
    g = jnp.asarray(r.normal(0, 1e-3, (64, 256)), jnp.bfloat16)
    m = jnp.asarray(r.normal(0, 1e-3, (64, 256)), jnp.bfloat16)
    w = jnp.asarray(r.normal(0, 5e-2, (64, 256)), jnp.bfloat16)
    sc = jnp.asarray([0.9, 3e-3, 0.999, 1.5], jnp.float32)
    a = pi.instinct_hs_2d(g, m, w, sc, interpret=True)
    one = lambda gg, aux: jnp.float32(1.0)
    b = pi.instinct_hs_2d(g, m, w, sc, interpret=True, gear=(one, jnp.ones((5,), jnp.float32)))
    assert len(a) == 2 and len(b) == 3
    assert all((np.asarray(x) == np.asarray(y)).all() for x, y in zip(a, b[:2])), "множник 1.0 не дав того самого"
    gref = float(jnp.sum(g.astype(jnp.float32) ** 2))
    assert abs(float(b[2]) - gref) <= 1e-5 * gref, (float(b[2]), gref)
    g3 = jnp.asarray(r.normal(0, 1e-3, (4, 64, 256)), jnp.bfloat16)
    m3, w3 = g3 * 0.5, (g3 * 40).astype(jnp.bfloat16)
    a3 = pi.instinct_hs3(g3, m3, w3, sc, interpret=True)
    b3 = pi.instinct_hs3(g3, m3, w3, sc, interpret=True, gear=(one, jnp.ones((5,), jnp.float32)))
    assert all((np.asarray(x) == np.asarray(y)).all() for x, y in zip(a3, b3[:2]))
    gref3 = float(jnp.sum(g3.astype(jnp.float32) ** 2))
    assert abs(float(b3[2]) - gref3) <= 1e-5 * gref3
    print("(3) gear=None / множник 1.0 — побітово те саме; Σg² ядра = jnp.sum(g²) (2D і hs3)")

    # (4) шардований hs3 (4 «чипи», EP): Σg² через psum — коробка та сама
    from jax.sharding import Mesh, PartitionSpec as P
    mesh = Mesh(np.array(jax.devices()[:4]), ("x",))
    CALLS.clear()
    ps, ms, hs = run(True, shard=(mesh, {"ex": P("x", None, None)}))
    assert ("ex", True) in CALLS
    for (d1, g1), (d2, g2) in zip(hk, hs):
        assert (g1 == g2).all(), (g1, g2)
        np.testing.assert_allclose(d1, d2, rtol=1e-6, atol=0)
    print("(4) шардований hs3: gidx/df збіг з нешардованим")
    po4, mo4, _ = run(True, gb_kern=False, shard=(mesh, {"ex": P("x", None, None)}))
    nb4 = sum(int((ps[k] != po4[k]).sum()) for k in ps) + sum(int((ms[k] != mo4[k]).sum()) for k in ms)
    print(f"  шардований: проти окремого проходу на тій самій розкладці різних bf16-елементів ваг+m: {nb4}")
    assert close(ps, po4, "ваги") < 1e-4 and close(ms, mo4, "m") < 1e-4
    print("GEARBOX_PALLAS_OK")


if __name__ == "__main__":
    main()
