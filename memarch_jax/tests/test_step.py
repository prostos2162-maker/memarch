"""M10 (W=1): step.train_step — 5 jit-кроків на повтореному батчі зменшують loss; jit == eager (loss і grads);
усі remat-політики дають ті самі loss/grads; jit-крок == ручний value_and_grad + instinct.step.
Запуск: cd memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_step.py"""
import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                       # noqa: E402
import jax                               # noqa: E402
import jax.numpy as jnp                  # noqa: E402

from memarch_jax import config, model, instinct, step as S   # noqa: E402

V = 1024
CFG = config.small(V=V, short_v=256, ce_chunk=512, T=128, d=128, head_dim=64, E=8, window=64, seg=64)
_r = np.random.default_rng(0)
COUNTS = np.floor(1e6 / np.arange(1, V + 1) ** 1.1 * _r.uniform(0.5, 1.5, V)).astype(np.int64)
COUNTS[V - 1] = 0
LT = _r.uniform(1.0, 4.0, (3, 7)).astype(np.float32)
ICFG = instinct.InstinctCfg(use_pallas=False, sr=False)


def _setup(B=2, seed=0):
    consts = model.build_consts(CFG, COUNTS, LT)
    params = model.init_params(jax.random.PRNGKey(seed), CFG, LT)
    spec = instinct.make_spec(params)
    xy = np.random.default_rng(seed + 1).integers(0, V - 1, (B, CFG.T + 1))
    return consts, params, spec, xy


def _copy(t):
    return jax.tree_util.tree_map(lambda a: jnp.array(a, copy=True), t)


def _maxdiff(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    return max(float(jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32)))) if x.size else 0.0
               for x, y in zip(la, lb))


def _relmax(a, b):
    """max|a−b| по всіх листках / max|b| по всіх листках (глобальна шкала: крихітні grad, напр. memory.importance
    ~1e-9, інакше дають «відносний» шум порядку 1 від іншого порядку сумування XLA)."""
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    d = max(float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64)))) for x, y in zip(la, lb) if x.size)
    sc = max(float(np.max(np.abs(np.asarray(y, np.float64)))) for y in lb if y.size)
    return d / sc


def test_loss_decreases():
    consts, params, spec, xy = _setup()
    batch = S.make_batch(xy, CFG, consts, 3e-3)
    ts = S.make_train_step(CFG, consts, spec, ICFG, remat="heavy")
    st = S.init_state(params, spec, ICFG)
    losses = []
    for _ in range(6):
        st, m = ts(st, batch)
        losses.append(float(m["loss"]))
    print("losses:", [round(x, 4) for x in losses], "step", int(st["step"]), "tau", float(st["tau"]))
    assert int(st["step"]) == 6
    assert all(np.isfinite(losses))
    assert losses[5] < losses[0] - 0.05, losses
    assert sum(b < a for a, b in zip(losses, losses[1:])) >= 4, losses


def test_jit_equals_eager():
    consts, params, spec, xy = _setup()
    batch = S.make_batch(xy, CFG, consts, 3e-3)
    tau = jnp.float32(7.5)
    lf = S.default_loss_fn(CFG, consts)
    vg = jax.value_and_grad(lf, has_aux=True)
    (l_e, (t_e, _, _)), g_e = vg(params, batch["xy"], batch["prep"], tau)
    (l_j, (t_j, _, _)), g_j = jax.jit(vg)(params, batch["xy"], batch["prep"], tau)
    dl, dg = abs(float(l_e) - float(l_j)), _relmax(g_j, g_e)
    print(f"jit vs eager: |dloss|={dl:.3e} grads rel max={dg:.3e} tau {float(t_e)} {float(t_j)}")
    assert dl < 1e-5 and dg < 1e-3 and float(t_e) == float(t_j)
    # повний крок: jit (donate) == неjit-функція кроку == ручне value_and_grad + instinct.step
    st0 = S.init_state(params, spec, ICFG)
    st_e, m_e = S.make_train_step(CFG, consts, spec, ICFG, jit=False)(_copy(st0), batch)
    st_j, m_j = S.make_train_step(CFG, consts, spec, ICFG)(_copy(st0), batch)
    p_m, o_m = instinct.step(params, g_e, _copy(st0["opt"]), lr=batch["lr"], situation=S.situation_full(params, batch["xy"]),
                             loss=l_e, spec=spec, cfg=ICFG)
    # memory.importance.*: grad ~1e-9 (шум порядку сумування), а Instinct нормує крок → знак шуму дає ±lr;
    # тому ці листки виключаємо з порівняння jit↔eager (ручний шлях з тими самими grads — точний для всіх)
    noisy = lambda t: {k: v for k, v in t.items() if k != "memory"} | {
        "memory": {k: v for k, v in t["memory"].items() if not k.startswith("importance")}}
    # eager на CPU не побітово відтворюваний між викликами (порядок сумування в потоках): ~1e-8 у grads, а в
    # memory.importance це ±lr після нормування → теж без noisy-листків, поріг 1e-6 (раніше вимагали == 0)
    d1, d2 = _maxdiff(noisy(st_j["params"]), noisy(st_e["params"])), _maxdiff(noisy(st_e["params"]), noisy(p_m))
    print(f"step jit vs eager params max|d|={d1:.3e}; eager vs manual={d2:.3e}; k={int(m_j['k'])},{int(o_m['k'])} "
          f"s={float(m_j['s']):.6f},{float(o_m['s']):.6f}")
    assert d1 < 2e-5 and d2 < 1e-6 and int(m_j["k"]) == int(o_m["k"])
    assert abs(float(m_j["s"]) - float(o_m["s"])) < 1e-6


def test_remat_policies():
    consts, params, spec, xy = _setup()
    batch = S.make_batch(xy, CFG, consts, 3e-3)
    tau = jnp.float32(7.5)
    ref = None
    for pol in ("none", "heavy", "all", "dots_saveable", "dots_nobatch"):
        vg = jax.jit(jax.value_and_grad(S.default_loss_fn(CFG, consts, remat=pol), has_aux=True))
        (l, _), g = vg(params, batch["xy"], batch["prep"], tau)
        if ref is None:
            ref = (l, g)
            print(f"remat {pol}: loss={float(l):.6f}")
            continue
        dl, dg = abs(float(l) - float(ref[0])), _relmax(g, ref[1])
        print(f"remat {pol}: loss={float(l):.6f} |dloss|={dl:.3e} grads rel max={dg:.3e}")
        assert dl < 1e-5 and dg < 1e-3, pol


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
