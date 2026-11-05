r"""Review M9-M12: Pallas-ядра Instinct у 8-чиповому кроці (dist.make_step). На TPU Pallas custom call GSPMD
не розбиває — шардовані листки (vocab-рядки, EP-експерти) йдуть через shard_map (instinct._pallas_call).
Тут (CPU, interpret=True) перевіряємо: крок з ядрами == крок чистим jnp з точністю до шуму SR (насіння шматків
змішане з номером чипа), розкладка стану збережена, ядерний шлях справді пройдено.
Запуск: cd memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_dist_pallas.py"""
import os
import sys

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                               # noqa: E402
import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402

from memarch_jax import config, model, dist, instinct, step as S  # noqa: E402

W, V, T = 8, 2048, 64
_r = np.random.default_rng(0)
COUNTS = np.floor(1e6 / np.arange(1, V + 1) ** 1.1 * _r.uniform(0.5, 1.5, V)).astype(np.int64)
COUNTS[V - 1] = 0
LT = _r.uniform(1.0, 4.0, (3, 7)).astype(np.float32)
CFG = config.small(d=128, head_dim=64, E=16, V=V, short_v=256, ce_chunk=512, T=T, B=1, window=32, seg=32,
                   dtype="bf16", cap_align=0)

CALLS = []
_orig = instinct._pallas_call
def _spy(kfn, args, sc, it, shard, leaf, li):
    CALLS.append((leaf, shard is not None and shard[1].get(leaf) is not None and any(a for a in shard[1][leaf])))
    return _orig(kfn, args, sc, it, shard, leaf, li)
instinct._pallas_call = _spy


def run(use_pallas):
    mesh = dist.make_mesh(W)
    consts = model.build_consts(CFG, COUNTS, LT)
    params = model.init_params(jax.random.PRNGKey(0), CFG, LT)
    spec = instinct.make_spec(params)
    icfg = instinct.InstinctCfg(m_bf16=True, sr=True, use_pallas=use_pallas, interpret=True)
    xy = np.random.default_rng(1).integers(0, V - 1, (W, T + 1))
    ts, st, place = dist.make_step(CFG, consts, mesh, params, spec, icfg, remat="none")
    sh0 = jax.tree_util.tree_map(lambda x: x.sharding, st)
    p0 = jax.tree_util.tree_map(lambda x: np.asarray(x.astype(jnp.float32)), st["params"])
    st, m = ts(st, place(S.make_batch(xy, CFG, consts, 3e-3)))
    sh1 = jax.tree_util.tree_map(lambda x: x.sharding, st)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda a, b: a == b, sh0, sh1)), "розкладка змінилась"
    p1 = jax.tree_util.tree_map(lambda x: np.asarray(x.astype(jnp.float32)), st["params"])
    return float(m["loss"]), p0, p1


def main():
    l_j, p0, pj = run(False)
    CALLS.clear()
    l_p, _, pp = run(True)
    sharded = sorted({l for l, s in CALLS if s})
    print(f"loss jnp {l_j:.6f} pallas {l_p:.6f}; ядерних викликів {len(CALLS)}, шардованих листків: {sharded}")
    assert any(s for _, s in CALLS), "жоден шардований листок не пішов через ядро"
    assert l_j == l_p
    worst = 0.0
    for (path, a), b, c in zip(jax.tree_util.tree_flatten_with_path(p0)[0], jax.tree_util.tree_leaves(pj),
                               jax.tree_util.tree_leaves(pp)):
        n = instinct.path_str(path)
        if n.startswith("memory.importance") or n.startswith("memory.gate"):
            continue                                             # шумові градієнти (див. decisions)
        dj, dp = b - a, c - a
        step_sz = np.abs(dj).max() + 1e-12
        # SR: кожен елемент може відрізнятися на 1 ulp bf16 від ваги — порівнюємо з ulp і кроком
        ulp = np.abs(a) * 2.0 ** -7 + 1e-30
        bad = np.abs(dp - dj) > (2 * ulp + 0.05 * step_sz)
        frac = bad.mean()
        worst = max(worst, frac)
        if frac > 1e-3:
            print(f"  {n}: частка розбіжних {frac:.4f}, max|d| {np.abs(dp - dj).max():.3e}, крок {step_sz:.3e}")
    print(f"найгірша частка елементів поза 2 ulp: {worst:.2e}")
    assert worst < 1e-3
    print("OK")


if __name__ == "__main__":
    main()
