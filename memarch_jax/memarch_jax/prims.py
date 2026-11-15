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


def dump_proof_norm(x, w, u, eps=EPS):
    """fix14 F1 (--norm_dump_proj): фінальна norm без спільного вектора потоку («звалища» m) у будь-якому каналі.
    y = (x − (x·û)û)·rsqrt(mean(x²)+eps)·w, û = stop_grad(норм(u)); u — EMA напряму середнього h (opt.U, K1).
    Знаменник — по ПОВНОМУ x (масштаб/температура як у rmsnorm); u=0 (перший крок resume) → середнє по батчу."""
    xf = x.astype(jnp.float32)
    mean_x = lax.stop_gradient(xf.reshape(-1, xf.shape[-1]).mean(0))
    u = jnp.asarray(u, jnp.float32) if u is not None else mean_x
    u = jnp.where(jnp.sum(u * u) > 0, u, mean_x)
    uh = lax.stop_gradient(u * lax.rsqrt(jnp.sum(u * u) + 1e-30))
    y = xf * lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + eps)
    c = (y @ uh)[..., None]
    y = y - c * uh
    return (y * w.astype(jnp.float32)).astype(jnp.result_type(x.dtype, w.dtype))


def linear(x, W, out_dtype=None):
    """x @ W.T, W у розкладці torch (out, in). Вхід/вага приводяться до dtype x (bf16 → як autocast:
    f32-накопичення, bf16-вихід). У f32 — точність HIGHEST (паритет з torch CPU)."""
    dt = x.dtype
    y = lax.dot_general(x, W.astype(dt), (((x.ndim - 1,), (1,)), ((), ())),
                        precision=lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    return y.astype(out_dtype or dt)


def rope_cs(T, D, base=10000.0, offset=0, pcut=0.0):
    """cos, sin (T, D/2) у fp32 — як baseline.rope.
    pcut > 0 (p-RoPE, лише трос): частоти з довжиною хвилі 2π/inv > pcut слів не обертаються (inv = 0) — ці виміри
    стають незалежними від відстані, а решта за будь-якої далекої відстані має вже бачений розподіл фаз
    (довжина хвилі ≤ pcut ≪ найменша далека відстань навчання). 0 — як було, побітово."""
    inv = 1.0 / base ** (jnp.arange(0, D, 2, dtype=jnp.float32) / D)
    if pcut > 0:
        inv = jnp.where(2 * jnp.pi / inv <= pcut, inv, 0.0)
    ang = (jnp.arange(T, dtype=jnp.float32) + offset)[:, None] * inv[None]
    return jnp.cos(ang), jnp.sin(ang)


def rope_half(x, base=10000.0, t_axis=-2, pcut=0.0):
    """RoPE з ROPE_HALF=1: пари — половинки вимірів. x (..., T, D) (t_axis=-2) або (B, T, H, D) (t_axis=-3:
    рахунок до транспозиції в (B,H,T,D), без зайвих копій розкладки). Поелементно ті самі операції → побітово
    однаково для будь-якої розкладки. Рахунок у fp32, потім cast назад."""
    T, D = x.shape[t_axis], x.shape[-1]
    cos, sin = rope_cs(T, D, base, pcut=pcut)
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


def cap_shared(hn, ratio, t0=256):
    """fix16: обмеження спільного вектора після фінальної norm, ПРИЧИННО. m_t = середнє hn по позиціях ≤ t,
    var_t = середнє |hn − m|² по позиціях ≤ t; якщо |m_t| > ratio·sqrt(var_t) — спільну складову стискаємо:
    out_t = hn_t − (1 − s_t)·m_t, s_t = ratio·sqrt(var_t)/|m_t|. Позиції < t0 і s_t ≥ 1 — побітово без змін."""
    hf = hn.astype(jnp.float32)
    T = hf.shape[-2]
    n = jnp.arange(1, T + 1, dtype=jnp.float32)[:, None]
    m = jnp.cumsum(hf, axis=-2) / n
    e2 = jnp.cumsum(jnp.sum(hf * hf, -1, keepdims=True), axis=-2) / n
    m2 = jnp.sum(m * m, -1, keepdims=True)
    var = jnp.maximum(e2 - m2, 0.0)
    s = jnp.minimum(1.0, ratio * jnp.sqrt(var + 1e-6) / jnp.sqrt(m2 + 1e-6))   # зсув під коренем: без inf-похідної в 0
    s = jnp.where((jnp.arange(T) >= t0)[:, None], s, 1.0)
    out = hf - m * (1.0 - s)
    return jnp.where(s < 1.0, out, hf).astype(hn.dtype)
