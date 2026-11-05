"""Каузальна увага з локальним вікном: ключ j видно запиту i ⇔ j ≤ i і i−j < window (window ключів разом із собою).

dense_ref — щільна маскована (CPU-еталон, f32 softmax). splash — Pallas splash_attention для TPU.
Масштаб 1/sqrt(hd) (за замовчуванням SDPA / splash_sdpa)."""
import functools
import math

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name


def dense_ref(q, k, v, window):
    """q, k, v: (B, H, T, hd). Оцінки й softmax у f32, вихід — у dtype v."""
    T, hd = q.shape[-2], q.shape[-1]
    s = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32,
                   precision=jax.lax.Precision.HIGHEST) * (1.0 / math.sqrt(hd))
    i = jnp.arange(T)[:, None]
    j = jnp.arange(T)[None]
    ok = (j <= i) & (i - j < window)
    s = jnp.where(ok, s, -jnp.inf)
    p = jax.nn.softmax(s, -1)
    o = jnp.einsum("bhqk,bhkd->bhqd", p, v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
    return o.astype(v.dtype)


@functools.lru_cache(maxsize=None)
def _splash_kernel(H, T, window, bq, bkv, bkvc):
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm
    # splash: «window−1 попередніх ключів + сам» = наше вікно (q − kv < window)
    mask = sm.MultiHeadMask([sm.LocalMask((T, T), (window - 1, 0), 0) for _ in range(H)])
    bs = sk.BlockSizes(block_q=bq, block_kv_compute=bkvc, block_kv=bkv,
                       block_q_dkv=bq, block_kv_dkv=bkv, block_kv_dkv_compute=bkvc,
                       block_q_dq=bq, block_kv_dq=bkv)
    # будується під трасуванням (перший виклик усередині jit) → масиви маски ставали трейсерами й
    # потрапляли в lru_cache (UnexpectedTracerError / «compiled for 231 inputs but called with 217»).
    with jax.ensure_compile_time_eval():
        return sk.make_splash_mha(mask=mask, block_sizes=bs, head_shards=1, q_seq_shards=1)


def splash(q, k, v, window, bq=1024, bkv=1024, bkvc=512):
    """TPU: splash_attention (Pallas, свій bwd). q заздалегідь множиться на 1/sqrt(hd) — у JAX-splash масштаб
    не вбудований. (Не перевірено на CPU; паритет з dense_ref — тест на TPU.)"""
    B, H, T, hd = q.shape
    bq, bkv, bkvc = min(bq, T), min(bkv, T), min(bkvc, T)
    kern = _splash_kernel(H, T, window, bq, bkv, bkvc)
    q = (q.astype(jnp.float32) * (1.0 / math.sqrt(hd))).astype(q.dtype)
    return jax.vmap(kern)(q, k, v)


def local_attention(q, k, v, window, impl="dense"):
    """Вибір реалізації. При T ≤ window обидві дають звичайну каузальну увагу (як SDPA is_causal у torch)."""
    if impl == "splash":
        return checkpoint_name(splash(q, k, v, window), "attn_out")
    return checkpoint_name(dense_ref(q, k, v, window), "attn_out")
