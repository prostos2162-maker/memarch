"""M0+M1: паритет FlexBlock (relu2, qkv3, LIGHT_FFN=2, ROPE_HALF, ATTN_IMPL=block) torch ↔ JAX на CPU, W=1.
Запуск: cd memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_light.py   (або python -m pytest ...)"""
import os
import sys

os.environ.update(DEVICE="cpu", NO_COMPILE="1", ATTN_IMPL="block", ROPE_HALF="1", BLK="relu2,qkv3", LIGHT_FFN="2")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # repo root: torch reference code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import full  # noqa: E402
from baseline import rope as t_rope  # noqa: E402
from lm import _block_window_attention  # noqa: E402

from memarch_jax import config, light, prims  # noqa: E402
from memarch_jax.attn import dense_ref  # noqa: E402

RES = {}


def _np(t):
    return t.detach().float().numpy()


def _cmp(name, a, b):
    a, b = np.asarray(jnp.asarray(a, jnp.float32)), np.asarray(b, np.float32)
    ad = float(np.max(np.abs(a - b)))
    rd = ad / max(float(np.max(np.abs(b))), 1e-30)
    RES[name] = (ad, rd)
    return ad, rd


def _build(cfg, dtype, seed=0):
    torch.manual_seed(seed)
    blk = full.FlexBlock(cfg.d, cfg.heads)
    with torch.no_grad():                       # ненульові/неодиничні норми — перевіряємо множення на вагу
        blk.n1.weight.uniform_(0.5, 1.5)
        blk.n2.weight.uniform_(0.5, 1.5)
    blk = blk.to(dtype)
    p = light.from_torch(blk.state_dict())
    return blk, p


def _run(cfg, tdt, jdt, T, seed=0):
    blk, p = _build(cfg, tdt, seed)
    assert blk.up.weight.shape == (cfg.light_hid, cfg.d)
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(2, T, cfg.d, generator=g).to(tdt).requires_grad_(True)
    gy = torch.randn(2, T, cfg.d, generator=g).to(tdt)
    mask = ("block", cfg.window) if T > cfg.window else None
    y = blk(x, mask)
    (y.float() * gy.float()).sum().backward()

    xj = jnp.asarray(_np(x)).astype(jdt)
    gyj = jnp.asarray(_np(gy)).astype(jnp.float32)
    f = lambda p, x: jnp.sum(light.flex_block(p, x, cfg).astype(jnp.float32) * gyj)
    yj = jax.jit(lambda p, x: light.flex_block(p, x, cfg))(p, xj)
    gp, gx = jax.jit(jax.grad(f, (0, 1)))(p, xj)
    tag = f"{jnp.dtype(jdt).name}_T{T}"
    out = {"y": _cmp(f"{tag}/y", yj, _np(y)), "dx": _cmp(f"{tag}/dx", gx, _np(x.grad))}
    tsd = dict(blk.named_parameters())
    for k, n in dict(n1="n1.weight", qkv="qkv.weight", o="o.weight", n2="n2.weight", up="up.weight",
                     down="down.weight").items():
        out["d" + k] = _cmp(f"{tag}/d{k}", gp[k], _np(tsd[n].grad))
    return out


def test_prims_f32():
    torch.manual_seed(0)
    x = torch.randn(3, 4, 64, 128)
    a, r = _cmp("rope_f32", prims.rope_half(jnp.asarray(x.numpy())), t_rope(x).numpy())
    assert r < 1e-6
    w = torch.rand(128) + .5
    n = torch.nn.RMSNorm(128)
    n.weight.data = w
    a, r = _cmp("rms_f32", prims.rmsnorm(jnp.asarray(x.numpy()), jnp.asarray(w.numpy())), n(x).detach().numpy())
    assert r < 1e-6
    xb, wb = x.bfloat16(), w.bfloat16()
    n = n.bfloat16()
    a, r = _cmp("rms_bf16", prims.rmsnorm(jnp.asarray(_np(xb)).astype(jnp.bfloat16), jnp.asarray(_np(wb)).astype(jnp.bfloat16)), _np(n(xb)))
    a, r = _cmp("rope_bf16", prims.rope_half(jnp.asarray(_np(xb)).astype(jnp.bfloat16)), _np(t_rope(xb)))


def test_window_attn_f32():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 300, 32) for _ in range(3))
    ref = _block_window_attention(q, k, v, 64)
    a, r = _cmp("attn_f32", dense_ref(*(jnp.asarray(t.numpy()) for t in (q, k, v)), 64), ref.numpy())
    assert r < 1e-5


def test_consts():
    if not (os.path.exists(config.DEFAULT_COUNTS) and os.path.exists(config.DEFAULT_LOCK_TRUST)):
        print("SKIP test_consts: phrasebook files not found (set MEMARCH_PHR)")
        return
    cfg = config.target()
    c = config.load_consts(cfg)
    cnt = config.load_counts(cfg.V)
    assert c["sv_ids"].shape == (8192,) and c["sv_pos"][c["sv_ids"]].tolist() == list(range(8192))
    assert c["sv_mask"][-1] == -1e4 and c["lock_bucket"].max() <= 6
    assert c["lock_trust"].shape == (3, 7)
    assert cfg.heads == 10 and cfg.light_hid == 5120
    # перехресно з torch-конструкцією (full2.py:1314-1333) на тих самих файлах
    top = torch.from_numpy(np.argsort(-cnt, kind="stable")[:8192].copy())
    assert (top.numpy() == c["sv_ids"]).all()


def test_flexblock_f32():
    cfg = config.small(d=256, head_dim=64, window=64)
    for T in (256, 48):                         # вікно (T > window) і коротко (каузальна SDPA)
        r = _run(cfg, torch.float32, jnp.float32, T)
        for k, (ad, rd) in r.items():
            assert rd < 1e-5, (T, k, ad, rd)


def test_flexblock_bf16():
    cfg = config.small(d=256, head_dim=64, window=64, dtype="bf16")
    r = _run(cfg, torch.bfloat16, jnp.bfloat16, 256)
    for k, (ad, rd) in r.items():
        assert rd < 5e-2, (k, ad, rd)


if __name__ == "__main__":
    for f in (test_prims_f32, test_window_attn_f32, test_consts, test_flexblock_f32, test_flexblock_bf16):
        try:
            f()
            print("OK  ", f.__name__)
        except AssertionError as e:
            print("FAIL", f.__name__, e)
    for k, (a, r) in RES.items():
        print(f"{k:22s} max_abs={a:.3e} max_rel={r:.3e}")
