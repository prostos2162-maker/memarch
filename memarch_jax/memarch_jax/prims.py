"""Примітиви: RMSNorm як у torch, лінійний шар (емуляція autocast bf16), RoPE (ROPE_HALF), relu², bf16-політика."""
import jax
import jax.numpy as jnp
from jax import lax

EPS = 1.1920929e-07          # finfo(fp32).eps — opmath-епсилон aten для nn.RMSNorm(eps=None)


RMS_CAST_FIRST = False       # True: старий composite aten (cast до dtype x, потім ·w у bf16)


def rmsnorm(x, w, eps=EPS, cast_first=None):
    """За замовчуванням як злитий rms_norm torch 2.14 (перевірено на CPU побітово):
    y = (x.f32·rsqrt(mean(x.f32²)+eps)·w.f32).astype(x.dtype).  cast_first=True — варіант
    (x.f32·rsqrt(...)).astype(x.dtype)·w. Для f32-входу (memory) обидва варіанти однакові (вихід f32)."""
    cf = RMS_CAST_FIRST if cast_first is None else cast_first
    xf = x.astype(jnp.float32)
    y = xf * lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + eps)
    if cf:
        return y.astype(x.dtype) * w
    return (y * w.astype(jnp.float32)).astype(jnp.result_type(x.dtype, w.dtype))


def linear(x, W, out_dtype=None):
    """x @ W.T, W у розкладці torch (out, in). Вхід/вага приводяться до dtype x (bf16 → як autocast:
    f32-накопичення, bf16-вихід). У f32 — точність HIGHEST (паритет з torch CPU)."""
    dt = x.dtype
    y = lax.dot_general(x, W.astype(dt), (((x.ndim - 1,), (1,)), ((), ())),
                        precision=lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    return y.astype(out_dtype or dt)


def rope_cs(T, D, base=10000.0, offset=0):
    """cos, sin (T, D/2) у fp32 — як baseline.rope."""
    inv = 1.0 / base ** (jnp.arange(0, D, 2, dtype=jnp.float32) / D)
    ang = (jnp.arange(T, dtype=jnp.float32) + offset)[:, None] * inv[None]
    return jnp.cos(ang), jnp.sin(ang)


def rope_half(x, base=10000.0, t_axis=-2):
    """RoPE з ROPE_HALF=1: пари — половинки вимірів. x (..., T, D) (t_axis=-2) або (B, T, H, D) (t_axis=-3:
    рахунок до транспозиції в (B,H,T,D), без зайвих копій розкладки). Поелементно ті самі операції → побітово
    однаково для будь-якої розкладки. Рахунок у fp32, потім cast назад."""
    T, D = x.shape[t_axis], x.shape[-1]
    cos, sin = rope_cs(T, D, base)
    if t_axis == -3:                      # (T, D/2) → (T, 1, D/2) для мовлення по головах
        cos, sin = cos[:, None], sin[:, None]
    else:
        assert t_axis == -2, t_axis
    xf = x.astype(jnp.float32)
    x1, x2 = xf[..., :D // 2], xf[..., D // 2:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).astype(x.dtype)


def relu2(x):
    """relu(x)² у dtype входу (bf16 на TPU)."""
    r = jnp.maximum(x, 0)
    return r * r


def normalize(x, eps=1e-12, axis=-1):
    """F.normalize: x / max(‖x‖, eps)."""
    n = jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)), axis, keepdims=True))
    return (x / jnp.maximum(n, eps).astype(x.dtype)).astype(x.dtype)


def sr_bf16(x, key):
    """Стохастичне округлення f32 → bf16 (додаємо випадкові 16 молодших біт і відкидаємо їх)."""
    bits = lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    r = jax.random.bits(key, x.shape, jnp.uint32) & jnp.uint32(0xFFFF)
    out = (bits + r) & jnp.uint32(0xFFFF0000)
    return lax.bitcast_convert_type(out, jnp.float32).astype(jnp.bfloat16)


def cast_tree(tree, dtype):
    """Приводить усі float-листки до dtype (BF16_ALL)."""
    return jax.tree_util.tree_map(lambda a: a.astype(dtype) if jnp.issubdtype(a.dtype, jnp.floating) else a, tree)
