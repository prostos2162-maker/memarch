"""Вибір якорів за здивуванням і оновлення порогу tau (full2.py:1668-1677), один чіп.
quantile_hook — місце для майбутнього pmean квантиля по чіпах (рішення Стефана: спільний tau)."""
import jax
import jax.numpy as jnp

F32 = jnp.float32


def select_anchors(s, tau, frac=0.2, cap=0.25, train=True, quantile_hook=None, n_q=100_000):
    """s (B,T) fp32 здивування, tau скаляр fp32 (СТАРИЙ поріг для маски).
    → idx (B,A) позиції (спершу збережені якорі за зростанням, далі решта), valid (B,A), tau_new, stats."""
    s = jax.lax.stop_gradient(s)
    B, T = s.shape
    anchor = s > tau
    if train:
        q = jnp.quantile(s[:, 1:].reshape(-1)[:n_q], 1 - frac)    # лінійна інтерполяція, як torch.quantile
        if quantile_hook is not None:
            q = quantile_hook(q)                                   # напр. lambda q: lax.pmean(q, 'x')
        tau_new = (tau * 0.9 + 0.1 * q).astype(F32)
    else:
        tau_new = tau
    A = max(1, round(T * cap))
    keep = anchor & (jnp.cumsum(anchor, 1) <= A)
    pos = jnp.arange(T)
    idx = jnp.argsort((~keep).astype(jnp.int32) * T + pos, axis=-1)[:, :A]   # ключі унікальні
    valid = jnp.take_along_axis(keep, idx, 1)
    stats = {"anchors": valid.astype(F32).mean() * A / T}
    return idx, valid, tau_new, stats
