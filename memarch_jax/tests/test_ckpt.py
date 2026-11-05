"""M11: ckpt — побітовий round-trip (params bf16/f32, стан Instinct з m bf16, cs, keys, used bool, tau, step,
data_state) для бекендів npz і orbax; продовження тренування з відновленого стану дає побітово той самий крок;
torch↔jax (export_torch → import_torch) побітово.
Запуск: cd memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_ckpt.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np                       # noqa: E402
import jax                               # noqa: E402
import jax.numpy as jnp                  # noqa: E402

from memarch_jax import ckpt, instinct, model, step as S   # noqa: E402
import test_step as TS                   # noqa: E402

DATA_ST = {"sources": [{"epoch": 1, "si": 2, "di": 3, "off": 4}], "taken": [12.0], "buf": [1, 2, 3], "n_docs": 9}


def _bits_equal(a, b):
    sa, sb = jax.tree_util.tree_structure(a), jax.tree_util.tree_structure(b)
    assert sa == sb, (sa, sb)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype and x.shape == y.shape, (x.dtype, y.dtype, x.shape, y.shape)
        assert x.tobytes() == y.tobytes()


def _bf16_state():
    cfg = TS.CFG.__class__(**{**TS.CFG.__dict__, "dtype": "bf16"})
    params = model.init_params(jax.random.PRNGKey(1), cfg, TS.LT)
    icfg = instinct.InstinctCfg(use_pallas=False, sr=True)
    spec = instinct.make_spec(params)
    st = S.init_state(params, spec, icfg)
    g = jax.tree_util.tree_map(lambda p: jax.random.normal(jax.random.PRNGKey(2), p.shape, jnp.float32).astype(p.dtype),
                               params)
    p2, o2 = instinct.step(params, g, st["opt"], lr=1e-3, situation=jnp.ones(cfg.d), loss=3.0, spec=spec, cfg=icfg)
    return dict(params=p2, opt=o2, tau=jnp.float32(7.25), step=jnp.int32(17))


def test_roundtrip():
    st = _bf16_state()
    assert any(x.dtype == jnp.bfloat16 for x in jax.tree_util.tree_leaves(st))
    assert isinstance(st["params"]["light_a"], list)
    for backend in ("npz", "orbax"):
        d = os.path.join(tempfile.mkdtemp(), "step_17")
        ckpt.save(d, st, data_state=DATA_ST, extra={"cfg": "small"}, backend=backend)
        got, ds, ex = ckpt.restore(d)
        _bits_equal(st, got)
        assert ds == DATA_ST and ex == {"cfg": "small"}
        assert ckpt.latest(os.path.dirname(d)) == d
        n = len(jax.tree_util.tree_leaves(st))
        print(f"{backend}: {n} leaves bitwise equal; cs keys {sorted(got['opt']['cs'])[:3]}…")


def test_resume_step_identical():
    consts, params, spec, xy = TS._setup()
    batch = S.make_batch(xy, TS.CFG, consts, 3e-3)
    ts = S.make_train_step(TS.CFG, consts, spec, TS.ICFG, donate=False)
    st, _ = ts(S.init_state(params, spec, TS.ICFG), batch)
    d = os.path.join(tempfile.mkdtemp(), "step_1")
    ckpt.save(d, st, backend="npz")
    st_r, _, _ = ckpt.restore(d)
    _bits_equal(st, st_r)
    a, ma = ts(st, batch)
    b, mb = ts(st_r, batch)
    _bits_equal(a, b)
    print(f"resume: next step bitwise equal, loss {float(ma['loss']):.6f} == {float(mb['loss']):.6f}")


def test_torch_roundtrip():
    st = _bf16_state()
    frozen = {"lock_w": jnp.asarray(TS.LT, jnp.bfloat16), "ind_a": jnp.ones((2,), jnp.bfloat16)}
    sd = ckpt.export_torch(st["params"], frozen, st["tau"])
    p, fr, tau = ckpt.import_torch(sd, jnp.bfloat16)
    _bits_equal(st["params"], p)
    _bits_equal(frozen, fr)
    assert float(tau) == float(st["tau"])
    try:
        import torch
        sd_t = {k: torch.from_numpy(np.array(v, copy=True)) for k, v in sd.items()}
        p2, _, _ = ckpt.import_torch(sd_t, jnp.bfloat16)
        _bits_equal(p, p2)
        print("torch tensors: OK")
    except ImportError:
        pass
    print(f"torch↔jax: {len(sd)} torch keys, bitwise equal")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
