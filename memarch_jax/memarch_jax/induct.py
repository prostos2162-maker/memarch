"""Рефлекс впізнавання INDUCT=lock: «ключі й замки» на 1/2/3 слова (full2.py _induct_raw/_induct/forward).
induct_raw — на хості в numpy (CPU_PREP, ключі int64), решта — чисті jnp-функції."""
import numpy as np
import jax
import jax.numpy as jnp

F32 = jnp.float32


def lock_bucket(counts, V):
    """Кошик рідкісності слова: clip(floor(log10(cnt+1)), 0, 6), cnt доповнено нулями до V."""
    cnt = np.asarray(counts).astype(np.int64)[:V]
    cnt = np.pad(cnt, (0, V - len(cnt)))
    return np.clip(np.floor(np.log10(cnt + 1)), 0, 6).astype(np.int64)


def induct_raw(x, V, bucket):
    """Хостова підготовка (numpy, без градієнта). x (B,T) int → cand (B,T,3), on (B,T,3) bool, bk (B,T), src (B,T,3).
    Для кожного ключа (слово / пара / трійка) — остання попередня поява; кандидат — слово після неї.
    Діє лише найдовший замок."""
    x = np.asarray(x).astype(np.int64)
    B, T = x.shape
    prev = np.concatenate([np.full((B, 1), V, np.int64), x[:, :-1]], 1)
    prev2 = np.concatenate([np.full((B, 2), V, np.int64), x[:, :-2]], 1)
    keys = (x, prev * (V + 1) + x, (prev2 * (V + 1) + prev) * (V + 1) + x)
    rows = np.arange(B)[:, None]
    cands, ons, srcs = [], [], []
    for key in keys:
        order = np.argsort(key, axis=1, kind="stable")    # однакові ключі — за позицією
        ks = np.take_along_axis(key, order, 1)
        ps = order
        same = np.concatenate([np.zeros((B, 1), bool), ks[:, 1:] == ks[:, :-1]], 1)
        lastp = np.concatenate([np.full((B, 1), -1, np.int64), ps[:, :-1]], 1)
        last = np.full((B, T), -1, np.int64)
        last[rows, ps] = np.where(same, lastp, -1)
        ok = last >= 0
        src = np.minimum(last + 1, T - 1)
        cand = np.take_along_axis(x, src, 1)
        cands.append(np.where(ok, cand, 0)); srcs.append(np.where(ok, src, 0)); ons.append(ok)
    for k in range(len(ons) - 1):                         # лише найдовший замок, що спрацював
        ons[k] = ons[k] & ~np.stack(ons[k + 1:], 0).any(0)
    bk = np.asarray(bucket)[np.minimum(x, len(bucket) - 1)]
    return np.stack(cands, -1), np.stack(ons, -1), bk, np.stack(srcs, -1)


def _sig(z):
    """sigmoid як torch на bf16: рахунок у fp32, одне округлення до dtype входу."""
    return jax.nn.sigmoid(z.astype(F32)).astype(z.dtype)


def induct_val(on, bk, lock_w):
    """Довіра замка: on·lock_w[k, bk] (fp32). lock_w заморожена (у consts) — використовується лише val≠0 і статистика."""
    lw = jax.lax.stop_gradient(lock_w).astype(F32)
    return on.astype(F32) * jnp.stack([lw[k][bk] for k in range(on.shape[-1])], -1)


def induct_in(h, val, src, ind_in_g):
    """INDUCT_IN + INDUCT_POS у формі вибірки: h + Σ_k [val_k≠0]·σ(g_k)·h[src_k]. h — ще чисте emb(x) (B,T,d).
    Замки взаємовиключні (≤1 ненульовий доданок на рядок), тож уперед побітово як bmm-форма torch;
    градієнт — та сама математика (інший порядок fp32-сумування при розсипанні в h[src])."""
    g = _sig(ind_in_g[:val.shape[-1]])                              # у dtype параметра (bf16 при BF16_ALL)
    wsel = (val != 0).astype(F32) * g.astype(F32)                   # (B,T,k) fp32, як у torch
    acc = 0.0
    for k in range(val.shape[-1]):
        hk = jnp.take_along_axis(h, src[..., k:k + 1], axis=1)      # (B,T,d)
        acc = acc + wsel[..., k:k + 1].astype(h.dtype).astype(F32) * hk.astype(F32)
    return h + acc.astype(h.dtype)


def induct_mix(cand, on, bk, mix_w):
    """INDUCT_MIX: «легка голова перебиває». c1 (B,T) — кандидат замка, g (B,T) — вага суміші (dtype mix_w)."""
    anyon = on.any(-1)
    kk = jnp.argmax(on.astype(F32), -1)
    c1 = (cand * on).sum(-1)
    g = _sig(mix_w[kk, bk]) * anyon.astype(mix_w.dtype)
    mix_g = g.astype(F32).sum() / jnp.maximum(anyon.sum(), 1).astype(F32)
    return c1, g, {"mix_g": jax.lax.stop_gradient(mix_g)}


def mix_nll(logp, c1, g, y):
    """Головна втрата з сумішшю: Σ −log max((1−g)·P(y) + g·[c1=y], 1e-30). logp, c1, g, y — (B,T)."""
    py = jnp.exp(logp.astype(F32))
    p = (1 - g).astype(F32) * py + g.astype(F32) * (c1 == y).astype(F32)
    return -jnp.log(jnp.maximum(p, 1e-30)).sum()
