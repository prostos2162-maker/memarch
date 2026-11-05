"""Паритет Instinct (memarch_jax.instinct) проти torch exp_opt.InstinctOptB на CPU, W=1, SR вимкнено.

Запуск: cd memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_instinct.py   (або python -m pytest)
"""
import os
import sys

os.environ.update(DEVICE="cpu", NO_COMPILE="1", OPT_HS_LAG="1", OPT_SR_OFF="1", OPT_FN="1", OPT_FUSED="1")
os.environ.pop("BF16_ALL", None)
os.environ.pop("PALLAS_OPT", None)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # repo root: torch reference code

import numpy as np
import torch
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)
import torch.distributed as _dist
if not _dist.is_initialized():                 # hsS у torch робить all_reduce (W=1 — тотожність)
    _dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29533", rank=0, world_size=1)
import exp_opt
from memarch_jax import instinct as I

RESULTS = {}


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(jnp.asarray(x, jnp.float32))


def _diff(a, b):
    a, b = _np(a), _np(b)
    d = np.abs(a - b)
    return float(d.max()) if d.size else 0.0, float((d / (np.abs(b) + 1e-6)).max()) if d.size else 0.0


# ------------------------------------------------------------------------------------------------- prepare/finish
def test_prepare_finish_20():
    rng = np.random.default_rng(0)
    K, d, P = 4, 64, 7
    cfg = I.InstinctCfg(lr_peak=1e-3)
    tk = dict(keys=torch.zeros(K, d), used=torch.zeros(K, dtype=torch.bool), expect=torch.zeros(K),
              seen=torch.zeros(K), gain=torch.ones(K, P))
    st = dict(gain=jnp.ones((K, P)), keys=jnp.zeros((K, d)), used=jnp.zeros(K, bool), expect=jnp.zeros(K),
              seen=jnp.zeros(K), rel=jnp.zeros(P), g=jnp.ones(P), oh=jnp.zeros(K), s=jnp.float32(0),
              stake=jnp.float32(0), k=jnp.int32(0), n_step=jnp.int32(0))
    base = rng.standard_normal((3, d)).astype(np.float32)          # три «ситуації» + шум: картки повторюються
    worst = {}
    prep_j = jax.jit(lambda st, l, s, r: I.prepare(st, l, s, r, cfg))
    fin_j = jax.jit(lambda st: I.finish(st, cfg))
    for t in range(20):
        sit = base[t % 3] + 0.3 * rng.standard_normal(d).astype(np.float32) if t % 5 else rng.standard_normal(d).astype(np.float32)
        loss = np.float32(5.0 - 0.1 * t + rng.standard_normal() * 0.3)
        ratio = np.float32(min(1, (t + 1) / 10))
        rel = rng.random(P).astype(np.float32)
        s, stake, k, g, oh = exp_opt._prepare_impl(tk["keys"], tk["used"], tk["expect"], tk["seen"], tk["gain"],
                                                   torch.tensor(sit), torch.tensor(loss), torch.tensor(ratio), 0.5, 0.98)
        exp_opt._finish_impl(s, g, torch.tensor(rel), tk["gain"], oh, 0.1, 0.2, 0.05, 0.01)
        st = prep_j(st, loss, sit, ratio)
        assert int(st["k"]) == int(k), (t, int(st["k"]), int(k))
        assert np.array_equal(np.asarray(st["used"]), tk["used"].numpy())
        st = fin_j(dict(st, rel=jnp.asarray(rel)))
        for name, a, b in [("s", st["s"], s), ("stake", st["stake"], stake), ("g", st["g"], g),
                           ("keys", st["keys"], tk["keys"]), ("expect", st["expect"], tk["expect"]),
                           ("gain", st["gain"], tk["gain"])]:
            ab, rl = _diff(a, b)
            worst[name] = max(worst.get(name, 0), ab)
            assert ab < 1e-5, (t, name, ab)
    RESULTS["prepare_finish_20 max_abs"] = worst


# ------------------------------------------------------------------------------------------------- повний крок
SHAPES = [("emb", (10000, 32)), ("router", (48, 64)), ("up", (3, 32, 40)), ("light.w", (64, 48)),
          ("norm", (48,)), ("alpha", ()), ("unused", (16, 24)), ("head", (10000, 32))]
KINDS = {"light.w": "hsS"}


def _setup(dtype_t, rng):
    tp = []
    for n, shp in SHAPES:
        x = torch.tensor(rng.standard_normal(shp).astype(np.float32) * 0.05).to(dtype_t)
        p = torch.nn.Parameter(x)
        if n == "light.w":
            p._sharded, p._shard_W, p._rows_total = True, 1, shp[0]
        tp.append(p)
    return tp


def _grads(rng, t):
    gs = {}
    for n, shp in SHAPES:
        g = rng.standard_normal(shp).astype(np.float32) * 0.01
        if n in ("emb", "head"):
            g[rng.random(shp[0]) < 0.7] = 0                      # більшість слів у кроці не трапились
        if n == "router" and t == 0:
            g[:, :5] = 0                                         # частина стовпців без градієнта
        gs[n] = g
    return gs


def _run(m_bf16, dtype_t, dtype_j, steps=5, rel_every=2, sr=False, use_pallas=False, torch_too=True, seed=1):
    rng = np.random.default_rng(seed)
    tp = _setup(dtype_t, rng)
    names = [n for n, _ in SHAPES]
    params = {}
    for n, p in zip(names, tp):
        d = params
        *pre, last = n.split(".")
        for q in pre:
            d = d.setdefault(q, {})
        d[last] = jnp.asarray(np.array(p.detach().float().numpy(), copy=True), dtype_j)
    lr_peak, d_sit = 1e-2, 32
    cfg = I.InstinctCfg(lr_peak=lr_peak, weight_decay=0.1, scale=0.45, m_bf16=m_bf16, rel_every=rel_every,
                        hs_lag=True, sr=sr, use_pallas=use_pallas)
    spec = I.make_spec(params, kinds=KINDS, order=names, has_grad={"unused": False})
    st = I.init(params, d_sit, cfg, spec)
    if torch_too:
        opt = exp_opt.InstinctOptB(tp, lr=lr_peak, weight_decay=0.1, scale=0.45, norm="hs", m_bf16=m_bf16,
                                   rel_every=rel_every)
    stepj = jax.jit(lambda p, g, s, lr, sit, loss: I.step(p, g, s, lr=lr, situation=sit, loss=loss, spec=spec, cfg=cfg))
    for t in range(steps):
        lr = I.lr_at(t, lr_peak, 50, warmup=3)
        gs = _grads(rng, t)
        sit = rng.standard_normal(d_sit).astype(np.float32)
        loss = np.float32(4.0 - 0.2 * t + 0.1 * rng.standard_normal())
        if torch_too:
            opt.param_groups[0]["lr"] = lr
            opt.prepare(torch.tensor(loss), torch.tensor(sit))
            for n, p in zip(names, tp):
                p.grad = None if n == "unused" else torch.tensor(gs[n]).to(p.dtype)
            opt.update_all()
            opt.finish()
        gj = {}
        for n in names:
            d = gj
            *pre, last = n.split(".")
            for q in pre:
                d = d.setdefault(q, {})
            d[last] = jnp.asarray(gs[n], dtype_j)
        params, st = stepj(params, gj, st, jnp.float32(lr), sit, loss)
    pf = {I.path_str(p): x for p, x in jax.tree_util.tree_flatten_with_path(params)[0]}
    return names, tp, (opt if torch_too else None), pf, st


def _compare(tag, names, tp, opt, pf, st, tol=None):
    rep = {}
    for i, (n, p) in enumerate(zip(names, tp)):
        rep[f"w.{n}"] = _diff(pf[n], opt.w[i] if opt.w[i] is not None else p)
        rep[f"m.{n}"] = _diff(st["m"][n], opt.m[i])
        if opt.cs[i] is not None:
            rep[f"cs.{n}"] = _diff(st["cs"][n], opt.cs[i])
    for k in ("rel", "gain", "s", "stake", "keys", "expect"):
        rep[k] = _diff(st[k], getattr(opt, k))
    assert int(st["k"]) == int(opt.k)
    worst = max(v[0] for v in rep.values())
    RESULTS[tag] = dict(max_abs=worst, max_rel_weights=max(v[1] for k, v in rep.items() if k.startswith("w.")),
                        per=dict((k, f"{v[0]:.2e}") for k, v in rep.items()))
    if tol is not None:
        bad = {k: v for k, v in rep.items() if v[0] > tol}
        assert not bad, bad


def test_full_fp32_m32():
    _compare("full_fp32_m_fp32 (5 steps)", *_run(False, torch.float32, jnp.float32), tol=1e-5)


def test_full_fp32_mbf16():
    # bf16-інерція: рідкісні перевертання округлення m на 1 ulp bf16 (XLA зливає mul+add у FMA → mf різниться на 1 ulp fp32)
    _compare("full_fp32_m_bf16 (5 steps)", *_run(True, torch.float32, jnp.float32), tol=2e-4)


def test_full_bf16_report():
    # torch: bf16-ваги з fp32-копією (без BF16_ALL); jax: bf16 без копії, SR вимкнено — лише звіт
    _compare("full_bf16_weights_m_bf16 (5 steps, report)", *_run(True, torch.bfloat16, jnp.bfloat16))


def test_pallas_vs_jnp_sr():
    """Pallas (interpret) проти jnp-запасного шляху, SR увімкнено (той самий хеш): біти мають збігатися до 1 ulp."""
    _, _, _, pa, sa = _run(True, torch.bfloat16, jnp.bfloat16, steps=3, sr=True, use_pallas=True, torch_too=False)
    _, _, _, pb, sb = _run(True, torch.bfloat16, jnp.bfloat16, steps=3, sr=True, use_pallas=False, torch_too=False)
    rep = {}
    for n in pa:
        a, b = np.asarray(pa[n], np.float32), np.asarray(pb[n], np.float32)
        ulp = np.abs(a - b) / np.maximum(np.abs(b), 1e-30) * 128     # bf16: 1 ulp ≈ 2^-7 відносно
        rep[n] = (float(np.abs(a - b).max()), float((a != b).mean()), float(ulp.max()))
        ma, mb = np.asarray(sa["m"][n], np.float32), np.asarray(sb["m"][n], np.float32)
        rep["m." + n] = (float(np.abs(ma - mb).max()), float((ma != mb).mean()), 0.0)
    assert int(sa["seed_t"]) == int(sb["seed_t"]) == 3 * 4
    RESULTS["pallas_interpret_vs_jnp (SR on, 3 steps) max_abs/frac_neq/ulp"] = {k: tuple(f"{x:.2e}" for x in v) for k, v in rep.items()}
    for n in pa:
        assert rep[n][2] <= 2.01, (n, rep[n])


def test_lr_at():
    import fastloop  # noqa  (порівняння з torch-розкладом)
    class A: lr, steps = 6e-4, 1000
    os.environ.update(LR_SCHED="wsd", WARMUP="200")
    for s in [0, 1, 199, 200, 799, 800, 801, 900, 999, 1000]:
        assert I.lr_at(s, 6e-4, 1000) == fastloop.lr_at(s, A), s


def test_stacked_heavy():
    """Складений по шару heavy (L,...) == окремі ваги heavy.i.* (вид за формою без осі шару, свій rel/gain/cs)."""
    L, E, d, h = 3, 4, 32, 40
    rng = np.random.default_rng(5)
    mk = lambda *s: jnp.asarray(rng.standard_normal(s) * 0.1, jnp.bfloat16)
    st_p = {"heavy": {"norm": mk(L, d), "router": mk(L, E, d), "up": mk(L, E, d, h), "down": mk(L, E, h, d)},
            "w": mk(24, 16)}
    st_g = jax.tree_util.tree_map(lambda a: jnp.asarray(rng.standard_normal(a.shape), jnp.float32), st_p)
    sep = lambda t: dict({f"h{i}": {k: v[i] for k, v in t["heavy"].items()} for i in range(L)}, w=t["w"])
    order = [f"heavy.{i}.{k}" for i in range(L) for k in ("norm", "router", "up", "down")] + ["w"]
    sp_a = I.make_spec(st_p, order=order)
    sp_b = I.make_spec(sep(st_p), order=[o.replace("heavy.", "h").replace("h", "h", 1) if o != "w" else o
                                         for o in [f"h{i}.{k}" for i in range(L) for k in ("norm", "router", "up", "down")] + ["w"]],
                       stacked=())
    kinds_a = {n: k for n, k, *_ in sp_a}
    assert kinds_a["heavy.0.norm"] == "tensor" and kinds_a["heavy.1.router"] == "hs" and kinds_a["heavy.2.up"] == "hs3"
    assert [k for _, k, *_ in sp_a] == [k for _, k, *_ in sp_b]
    for pal in (False, True):
        cfg = I.InstinctCfg(rel_every=1, use_pallas=pal)
        sa, sb = I.init(st_p, 8, cfg, sp_a), I.init(sep(st_p), 8, cfg, sp_b)
        pa, pb = st_p, sep(st_p)
        for t in range(3):
            sit = jnp.asarray(rng.standard_normal(8), jnp.float32)
            pa, sa = I.step(pa, st_g, sa, lr=1e-3, situation=sit, loss=jnp.float32(3.0 - t), spec=sp_a, cfg=cfg)
            pb, sb = I.step(pb, sep(st_g), sb, lr=1e-3, situation=sit, loss=jnp.float32(3.0 - t), spec=sp_b, cfg=cfg)
        for a, b in zip(jax.tree_util.tree_leaves(sep(pa)), jax.tree_util.tree_leaves(pb)):
            assert np.array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))
        assert np.array_equal(np.asarray(sa["rel"]), np.asarray(sb["rel"])) and int(sa["seed_t"]) == int(sb["seed_t"])
    RESULTS["stacked_heavy == separate layers"] = "bitwise"


def test_stacked_heavy_jit():
    """Під jit складені шари йдуть через fori_loop (in-place по шарах) — побітово = окремим вагам під jit."""
    L, E, d, h = 3, 4, 32, 40
    rng = np.random.default_rng(6)
    mk = lambda *s: jnp.asarray(rng.standard_normal(s) * 0.1, jnp.bfloat16)
    st_p = {"heavy": {"norm": mk(L, d), "router": mk(L, E, d), "up": mk(L, E, d, h), "down": mk(L, E, h, d)},
            "w": mk(24, 16)}
    st_g = jax.tree_util.tree_map(lambda a: jnp.asarray(rng.standard_normal(a.shape), jnp.float32), st_p)
    sep = lambda t: dict({f"h{i}": {k: v[i] for k, v in t["heavy"].items()} for i in range(L)}, w=t["w"])
    ks = ("norm", "router", "up", "down")
    sp_a = I.make_spec(st_p, order=[f"heavy.{i}.{k}" for i in range(L) for k in ks] + ["w"])
    sp_b = I.make_spec(sep(st_p), order=[f"h{i}.{k}" for i in range(L) for k in ks] + ["w"], stacked=())
    js = jax.jit(I.step, static_argnames=("spec", "cfg"))
    for pal in (False, True):
        cfg = I.InstinctCfg(rel_every=1, use_pallas=pal)
        sa, sb = I.init(st_p, 8, cfg, sp_a), I.init(sep(st_p), 8, cfg, sp_b)
        pa, pb = st_p, sep(st_p)
        for t in range(3):
            sit = jnp.asarray(rng.standard_normal(8), jnp.float32)
            pa, sa = js(pa, st_g, sa, lr=1e-3, situation=sit, loss=jnp.float32(3.0 - t), spec=sp_a, cfg=cfg)
            pb, sb = js(pb, sep(st_g), sb, lr=1e-3, situation=sit, loss=jnp.float32(3.0 - t), spec=sp_b, cfg=cfg)
        for a, b in zip(jax.tree_util.tree_leaves(sep(pa)), jax.tree_util.tree_leaves(pb)):
            assert np.array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))
        assert np.array_equal(np.asarray(sa["rel"]), np.asarray(sb["rel"])) and int(sa["seed_t"]) == int(sb["seed_t"])
    RESULTS["stacked_heavy (jit, fori по шарах) == separate layers"] = "bitwise"


if __name__ == "__main__":
    import json
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("OK  ", name)
            except Exception as e:
                fails += 1
                import traceback; traceback.print_exc()
                print("FAIL", name, type(e).__name__, e)
    print(json.dumps(RESULTS, indent=1, ensure_ascii=False))
    sys.exit(1 if fails else 0)


