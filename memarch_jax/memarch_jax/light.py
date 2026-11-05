"""Легкий блок FlexBlock (full.py:113-175) під BLK=relu2,qkv3 і LIGHT_FFN=2:
x += o(attn(rope(q), rope(k), v)) ; x += down(relu(up(n2 x))²). Параметри — у розкладці torch (out, in)."""
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from .attn import local_attention
from .prims import linear, relu2, rmsnorm, rope_half

KEYS = ("n1", "qkv", "o", "n2", "up", "down")


def init_block(key, cfg, dtype=jnp.float32):
    """Init як у torch: Linear ~ U(±1/sqrt(in)), RMSNorm = 1."""
    d, hid = cfg.d, cfg.light_hid
    ks = jax.random.split(key, 4)
    u = lambda k, shape: jax.random.uniform(k, shape, jnp.float32, -1, 1) / jnp.sqrt(shape[1])
    p = dict(n1=jnp.ones(d), qkv=u(ks[0], (3 * d, d)), o=u(ks[1], (d, d)), n2=jnp.ones(d),
             up=u(ks[2], (hid, d)), down=u(ks[3], (d, hid)))
    return {k: v.astype(dtype) for k, v in p.items()}


def from_torch(sd, prefix=""):
    """state_dict FlexBlock → pytree (numpy → jnp, dtype зберігається)."""
    import numpy as np
    m = dict(n1="n1.weight", qkv="qkv.weight", o="o.weight", n2="n2.weight", up="up.weight", down="down.weight")
    out = {}
    for k, n in m.items():
        t = sd[prefix + n].detach()
        out[k] = jnp.asarray(t.float().numpy()).astype(jnp.bfloat16 if str(t.dtype) == "torch.bfloat16" else jnp.float32)
    return out


# RoPE у розкладці (B,T,H,hd) до транспозиції (звіт prof_b4_all п.6, ≈15–20 ms). False — старий шлях
# (транспозиція, потім rope у (B,H,T,hd)). Поелементно ті самі операції → побітово однаково.
ROPE_BTHD = True


def pre_attn(p, x, cfg):
    """n1 → qkv → rope. x (B,T,d) → q, k, v (B,H,T,hd)."""
    B, T, D = x.shape
    H, hd = cfg.heads, D // cfg.heads
    n1 = rmsnorm(x, p["n1"])
    # qkv3: три множення на зрізи однієї ваги
    q, k, v = (linear(n1, p["qkv"][i * D:(i + 1) * D]).reshape(B, T, H, hd) for i in range(3))
    if ROPE_BTHD:
        q, k = rope_half(q, cfg.rope_base, t_axis=-3), rope_half(k, cfg.rope_base, t_axis=-3)
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
    else:
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        q, k = rope_half(q, cfg.rope_base), rope_half(k, cfg.rope_base)
    return q, k, v


def post_attn(p, x, a, cfg):
    """x += o(a) ; x += down(relu(up(n2 x))²). Мітки checkpoint_name — для remat-політик (model.REMAT_POLICIES)."""
    B, T, D = x.shape
    x = x + checkpoint_name(linear(a.transpose(0, 2, 1, 3).reshape(B, T, D), p["o"]).astype(x.dtype), "light_o")
    h = checkpoint_name(linear(rmsnorm(x, p["n2"]), p["up"]), "light_up")
    return x + linear(relu2(h), p["down"]).astype(x.dtype)


def flex_block(p, x, cfg):
    """x (B, T, d) у dtype залишкового потоку (bf16 на TPU). Повертає x того ж dtype."""
    q, k, v = pre_attn(p, x, cfg)
    a = local_attention(q, k, v, cfg.window, cfg.attn_impl)
    return post_attn(p, x, a, cfg)
