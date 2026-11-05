r"""M11: step.make_train_step на сітці з 8 CPU-чипів (dist.make_step) — fwd+bwd(shard_map)+Instinct в ОДНІЙ
jit-програмі з donate і remat. 10 кроків на повтореному батчі: loss спадає, після кроку 1 жодних перекомпіляцій
(лічильник backend_compile + _cache_size), розкладка стану зберігається, крок 1 == W=1 step.py на тих самих даних.
Запуск: cd memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_dist_step.py"""
import os
import sys

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time                                                      # noqa: E402
import numpy as np                                               # noqa: E402
import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402

from memarch_jax import config, model, dist, instinct, step as S  # noqa: E402

W = 8
V, T = 2048, 128
_r = np.random.default_rng(0)
COUNTS = np.floor(1e6 / np.arange(1, V + 1) ** 1.1 * _r.uniform(0.5, 1.5, V)).astype(np.int64)
COUNTS[V - 1] = 0
LT = _r.uniform(1.0, 4.0, (3, 7)).astype(np.float32)
CFG = config.small(d=128, head_dim=64, E=16, V=V, short_v=256, ce_chunk=512, T=T, B=1, window=64, seg=64,
                   dtype="f32", cap_align=0)
ICFG = instinct.InstinctCfg(m_bf16=False, sr=False, use_pallas=False)

COMPILES = []
jax.monitoring.register_event_duration_secs_listener(
    lambda ev, d, **kw: COMPILES.append(ev) if ev == "/jax/core/compile/backend_compile_duration" else None)


def test_mesh_step(steps=10, remat="heavy"):
    mesh = dist.make_mesh(W)
    consts = model.build_consts(CFG, COUNTS, LT)
    params = model.init_params(jax.random.PRNGKey(0), CFG, LT)
    spec = instinct.make_spec(params)
    xy = np.random.default_rng(1).integers(0, V - 1, (W, T + 1))
    hb = S.make_batch(xy, CFG, consts, 3e-3)

    # еталон W=1: кожен рядок окремо на одному чипі (B=1) — design §10: loss W=8 = середнє loss чипів
    lf1 = jax.jit(S.default_loss_fn(CFG, consts, remat))
    ref = np.mean([float(lf1(params, jnp.asarray(xy[i:i + 1], jnp.int32), model.make_prep(xy[i:i + 1, :-1], CFG, consts),
                             jnp.float32(7.5))[0]) for i in range(W)])
    ts, st, place = dist.make_step(CFG, consts, mesh, params, spec, ICFG, remat=remat)
    batch = place(hb)
    sh0 = jax.tree_util.tree_map(lambda x: x.sharding, st)
    losses, per_step_compiles, times = [], [], []
    for i in range(steps):
        n0 = len(COMPILES)
        t0 = time.perf_counter()
        st, m = ts(st, batch)
        losses.append(float(m["loss"]))
        times.append(time.perf_counter() - t0)
        per_step_compiles.append(len(COMPILES) - n0)
    sh1 = jax.tree_util.tree_map(lambda x: x.sharding, st)
    print(f"losses: {[round(x, 4) for x in losses]}")
    print(f"backend compiles per step: {per_step_compiles}  jit cache size: {ts._cache_size()}")
    print(f"step s: first {times[0]:.1f}, rest mean {np.mean(times[1:]):.2f}")
    print(f"emb {st['params']['emb'].sharding.spec}  heavy.up {st['params']['heavy']['up'].sharding.spec}  "
          f"step {int(st['step'])} tau {float(st['tau']):.5f}  per_chip {np.asarray(m['per_chip']).round(4)}")
    # W=1 на W рядках vs W=8 по рядку на чип: різні MoE-місткість/якорі на чип → лише близько (спільний батч)
    print(f"step1 loss mean W=1(B=1 per row) {ref:.6f}  W=8 {losses[0]:.6f}  |d| {abs(ref - losses[0]):.2e}")
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0] - 0.05 and sum(b < a for a, b in zip(losses, losses[1:])) >= 7, losses
    assert sum(per_step_compiles[1:]) == 0, per_step_compiles
    assert ts._cache_size() == 1
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda a, b: a == b, sh0, sh1))
    assert int(st["step"]) == steps
    assert abs(ref - losses[0]) <= 1e-5 * ref


def test_offload_opt(steps=3, remat="moe_save"):
    """--offload_opt: Instinct m у pinned_host — ті самі loss і params, що без offload (біт-у-біт), m на хості."""
    mesh = dist.make_mesh(W)
    consts = model.build_consts(CFG, COUNTS, LT)
    params = model.init_params(jax.random.PRNGKey(0), CFG, LT)
    spec = instinct.make_spec(params)
    xy = np.random.default_rng(1).integers(0, V - 1, (W, T + 1))
    hb = S.make_batch(xy, CFG, consts, 3e-3)
    out = []
    for off in (False, True):
        params = model.init_params(jax.random.PRNGKey(0), CFG, LT)       # donate видаляє попередні
        ts, st, place = dist.make_step(CFG, consts, mesh, params, spec, ICFG, remat=remat, offload_opt=off)
        b = place(hb)
        ls = []
        for _ in range(steps):
            st, m = ts(st, b)
            ls.append(float(m["loss"]))
        kinds = {x.sharding.memory_kind for x in jax.tree_util.tree_leaves(st["opt"]["m"])}
        out.append((ls, st, kinds))
    print(f"offload: loss {out[0][0]} vs {out[1][0]}  m memory_kind {out[0][2]} → {out[1][2]}")
    assert out[0][0] == out[1][0]
    assert out[1][2] == {"pinned_host"}
    for a, b in zip(jax.tree_util.tree_leaves(out[0][1]["params"]), jax.tree_util.tree_leaves(out[1][1]["params"])):
        assert np.array_equal(np.asarray(a), np.asarray(b))


if __name__ == "__main__":
    print("пристроїв:", len(jax.devices()))
    test_mesh_step()
    test_offload_opt()
    print("OK")
