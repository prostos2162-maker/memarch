"""Конвертація torch FullLM2 state_dict → JAX pytree (model.py) і назад (для перевірок). W=1, повний (не-SHARD) ckpt.

Імена 1:1: emb.weight→emb, head.weight→head, l_head.0/1.weight→l_head0/1, light_a.i.*→light_a[i][*],
heavy.i.{norm.weight,router.weight,up,down}→heavy[*] (стек по шару), memory.X→memory['X'], reread.X→reread['X'],
book.weight→book, mix_w, ind_in_g. Заморожені lock_w/ind_a і буфер tau повертаються окремо.
Невикористані буфери (err, learned, up_snap, s_mean, ...) відкидаються (design §9 п.8)."""
import numpy as np
import jax.numpy as jnp

from .light import KEYS as BLK_KEYS

_TOP = {"emb": "emb.weight", "book": "book.weight", "head": "head.weight", "norm": "norm.weight",
        "l_norm": "l_norm.weight", "l_head0": "l_head.0.weight", "l_head1": "l_head.1.weight",
        "mix_w": "mix_w", "ind_in_g": "ind_in_g"}
_HEAVY = {"norm": "norm.weight", "router": "router.weight", "up": "up", "down": "down"}


def _np(t):
    if hasattr(t, "detach"):
        t = t.detach()
        if t.is_floating_point():
            t = t.float()
        return t.numpy()
    return np.asarray(t)


def _count(sd, prefix):
    ids = {int(k[len(prefix):].split(".")[0]) for k in sd if k.startswith(prefix)}
    return max(ids) + 1 if ids else 0


def torch_to_jax(sd, dtype=jnp.float32):
    """sd — state_dict (або dict градієнтів з тими ж іменами). → (params, frozen {'lock_w','ind_a'}, tau | None).
    Верхньорівневі ключі, відсутні в sd (напр. градієнт None), пропускаються."""
    a = lambda k: jnp.asarray(_np(sd[k])).astype(dtype)
    p = {k: a(n) for k, n in _TOP.items() if n in sd}
    for grp in ("light_a", "light_b", "final"):
        p[grp] = [{k: a(f"{grp}.{i}.{k}.weight") for k in BLK_KEYS} for i in range(_count(sd, grp + "."))]
    L = _count(sd, "heavy.")
    p["heavy"] = {k: jnp.stack([a(f"heavy.{i}.{n}") for i in range(L)]) for k, n in _HEAVY.items()}
    for grp in ("memory", "reread"):
        p[grp] = {k[len(grp) + 1:]: a(k) for k in sd if k.startswith(grp + ".")}
    frozen = {k: a(k) for k in ("lock_w", "ind_a") if k in sd}
    tau = jnp.asarray(_np(sd["tau"]), jnp.float32) if "tau" in sd else None
    return p, frozen, tau


def grads_to_jax(named_grads, dtype=jnp.float32):
    """{torch-ім'я: grad} → pytree тієї ж структури, що params (без lock_w/ind_a)."""
    return torch_to_jax(named_grads, dtype)[0]


def jax_to_torch(params, frozen=None, tau=None):
    """Зворотна мапа → {torch-ім'я: numpy fp32}."""
    f = lambda x: np.asarray(jnp.asarray(x, jnp.float32))
    out = {n: f(params[k]) for k, n in _TOP.items() if k in params}
    for grp in ("light_a", "light_b", "final"):
        for i, blk in enumerate(params[grp]):
            for k in BLK_KEYS:
                out[f"{grp}.{i}.{k}.weight"] = f(blk[k])
    for k, n in _HEAVY.items():
        for i in range(params["heavy"][k].shape[0]):
            out[f"heavy.{i}.{n}"] = f(params["heavy"][k][i])
    for grp in ("memory", "reread"):
        for k, v in params[grp].items():
            out[f"{grp}.{k}"] = f(v)
    for k, v in (frozen or {}).items():
        out[k] = f(v)
    if tau is not None:
        out["tau"] = f(tau)
    return out
