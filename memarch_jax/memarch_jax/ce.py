"""Вихід на словник шматками (chunk_ce з власним VJP, як _ChunkCE у full2.py), здивування легкої голови
(SURP_INGRAPH, _surp_lse_impl) і дворівневий короткий словник SHORT_V (_short_logp, один чіп).

Точки округлення як у torch: логіти шматка — bf16-вихід множення (fp32-накопичення), lse — у fp32;
логіт правильного слова — fp32-скалярний добуток bf16-рядків (без bf16-округлення); назад — bf16 G."""
from functools import partial
import math

import numpy as np
import jax
import jax.numpy as jnp

BF = jnp.bfloat16
F32 = jnp.float32


def _mm_bf16(a, b):
    """bf16 @ bf16 → bf16 (fp32-накопичення, одне округлення) — як torch-множення bf16-тензорів."""
    return jnp.dot(a, b, preferred_element_type=F32).astype(BF)


def _chunks(V, ch):
    return [(c0, min(c0 + ch, V)) for c0 in range(0, V, ch)]


def _lse(x, axis=-1):
    return jax.nn.logsumexp(x, axis=axis)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def chunk_ce(h, W, b, y, ch):
    """(lse, логіт правильного слова) для h @ Wᵀ + b шматками по ch слів. h (N,d), W (V,d), b (V,) fp32, y (N,) int.
    b — без градієнта в моделі (сталі зсуви), але VJP його рахує (як torch)."""
    return _chunk_ce_fwd(h, W, b, y, ch)[0]


# Блок слів (рядків h) для chunk_ce: при N > TOK рахуємо lse/логіт і VJP по блоках TOK слів (lax.map / scan),
# щоб у пам'яті жила одна матриця логітів (TOK, ch), а не (N, ch). Для кожного слова числа ТІ САМІ (lse слова не
# залежить від інших слів); dW/db накопичуються у fp32 по блоках і округлюються в bf16 один раз (та сама точка
# округлення, що й у одному великому множенні). Хвіст доповнюємо нулями з g=0 → внесок 0.
TOK = 8192


def _lse_rows(hb, Wb, bf, ch):
    lses = []
    for c0, c1 in _chunks(Wb.shape[0], ch):
        lo = _mm_bf16(hb, Wb[c0:c1].T).astype(F32) + bf[c0:c1]
        lses.append(_lse(lo, -1))
    return _lse(jnp.stack(lses, -1), -1)


def _ly_rows(hb, Wb, bf, y):
    return (hb.astype(F32) * Wb[y].astype(F32)).sum(-1) + bf[y]


def _blocks(x, nb, fill=0):
    """(N, ...) → (nb, TOK, ...) з доповненням хвоста."""
    pad = nb * TOK - x.shape[0]
    if pad:
        x = jnp.concatenate([x, jnp.full((pad,) + x.shape[1:], fill, x.dtype)])
    return x.reshape((nb, TOK) + x.shape[1:])


def _chunk_ce_fwd(h, W, b, y, ch):
    hb, Wb = h.astype(BF), W.astype(BF)
    bf = b.astype(F32)
    N = hb.shape[0]
    if N <= TOK:
        lse, ly = _lse_rows(hb, Wb, bf, ch), _ly_rows(hb, Wb, bf, y)
    else:                                                # блоки слів: одна (TOK, ch) матриця логітів за раз
        nb = -(-N // TOK)
        lse, ly = jax.lax.map(lambda a: (_lse_rows(a[0], Wb, bf, ch), _ly_rows(a[0], Wb, bf, a[1])),
                              (_blocks(hb, nb), _blocks(y, nb)))
        lse, ly = lse.reshape(-1)[:N], ly.reshape(-1)[:N]
    return (lse, ly), (hb, Wb, bf, y, lse, h.dtype, W.dtype, b.dtype)


def _chunk_ce_fwd_vjp(h, W, b, y, ch):
    out, res = _chunk_ce_fwd(h, W, b, y, ch)
    hb, Wb, bf, y, lse, hd, wd, bd = res
    return out, (hb, Wb, bf, y, lse, jnp.zeros((), hd), jnp.zeros((), wd), jnp.zeros((), bd))


def _bwd_rows(hb, Wb, bf, y, lse, g_lse, g_ly, ch):
    """→ dh (n,d) fp32, dW (V,d) fp32 (ще без bf16-округлення), db (V,) fp32."""
    dh = jnp.zeros(hb.shape, F32)
    dW, db = [], []
    for c0, c1 in _chunks(Wb.shape[0], ch):
        lo = _mm_bf16(hb, Wb[c0:c1].T).astype(F32) + bf[c0:c1]          # перерахунок, як у torch
        hit = (jnp.arange(c0, c1)[None] == y[:, None])
        G = g_lse[:, None] * jnp.exp(lo - lse[:, None]) + g_ly[:, None] * hit
        Gb = G.astype(BF)
        dh = dh + _mm_bf16(Gb, Wb[c0:c1]).astype(F32)
        dW.append(jnp.dot(Gb.T, hb, preferred_element_type=F32))       # bf16-округлення — у викликача, один раз
        db.append(G.sum(0))
    return dh, jnp.concatenate(dW), jnp.concatenate(db)


def _chunk_ce_bwd(ch, res, g):
    hb, Wb, bf, y, lse, hz, wz, bz = res
    g_lse, g_ly = g
    N = hb.shape[0]
    if N <= TOK:
        dh, dW, db = _bwd_rows(hb, Wb, bf, y, lse, g_lse, g_ly, ch)
    else:                                                # блоки слів; dW/db — fp32-накопичення в carry
        nb = -(-N // TOK)
        xs = (_blocks(hb, nb), _blocks(y, nb), _blocks(lse, nb), _blocks(g_lse, nb), _blocks(g_ly, nb))

        def body(c, a):
            dh_b, dW_b, db_b = _bwd_rows(a[0], Wb, bf, a[1], a[2], a[3], a[4], ch)
            # dh блоку одразу в dtype h: той самий convert, але scan не пише fp32-стек (nb,TOK,d) (менше пам'яті й DUS)
            return (c[0] + dW_b, c[1] + db_b), dh_b.astype(hz.dtype)
        (dW, db), dh = jax.lax.scan(body, (jnp.zeros(Wb.shape, F32), jnp.zeros(Wb.shape[:1], F32)), xs)
        dh = dh.reshape(-1, hb.shape[1])[:N]
    # bf16-округлення dW завжди (як Gbᵀ@hb у torch)
    return (dh.astype(hz.dtype), dW.astype(BF).astype(wz.dtype), db.astype(bz.dtype), None)


chunk_ce.defvjp(_chunk_ce_fwd_vjp, _chunk_ce_bwd)


def chunk_ce_nb(h, W, y, ch):
    """chunk_ce без зсуву (b=None у torch): нульовий fp32-зсув дає ті самі числа."""
    return chunk_ce(h, W, jnp.zeros((W.shape[0],), F32), y, ch)


def surprise(lh, w0, W2, x, ch, vocab=None):
    """Здивування легкої голови (SURP_INGRAPH): s (B,T) fp32, s[:,0]=1e4, s[:,1:] = lse − логіт слова x[:,1:].
    lh (B,T,d) — вихід l_norm; w0 (128,d), W2 (V,128). Без градієнта.
    vocab — None (W=1, уся таблиця) або dist.VocabShard (W2 — свої рядки, двоетапний lse)."""
    lh, w0, W2 = (jax.lax.stop_gradient(a) for a in (lh, w0, W2))
    B, T = x.shape
    a = lh[:, :T - 1]
    if a.dtype == BF:                                   # autocast: bf16-вихід лінійного шару
        h128 = jnp.dot(a, w0.astype(BF).T, preferred_element_type=F32)
    else:                                               # fp32-модель (тест): fp32-множення
        h128 = jnp.dot(a, w0.astype(a.dtype).T, precision="highest")
    hb = h128.reshape(B * (T - 1), -1).astype(BF)
    tg = x[:, 1:].reshape(-1)
    if vocab is not None:                               # нульовий зсув: lo + 0 і ly + 0 — ті самі числа
        lse, ly = vocab.ce(hb, W2, jnp.zeros((vocab.V,), F32), tg, ch)
        s = jnp.full((B, T), 1e4, F32)
        return s.at[:, 1:].set(jax.lax.stop_gradient(lse - ly).reshape(B, T - 1))
    Wb = W2.astype(BF)
    lse = _lse(jnp.stack([_lse(_mm_bf16(hb, Wb[c0:c1].T).astype(F32), -1)
                          for c0, c1 in _chunks(W2.shape[0], ch)], -1), -1)
    ly = (hb.astype(F32) * Wb[tg].astype(F32)).sum(-1)
    s = jnp.full((B, T), 1e4, F32)
    return s.at[:, 1:].set((lse - ly).reshape(B, T - 1))


# ---------- дворівневий вихід SHORT_V ----------

def short_consts(counts, V, K, rare_cap=0.3):
    """Сталі короткого словника з частот (numpy), як буфери FullLM2 (full2.py:1310-1338)."""
    cnt = np.zeros(V, np.int64)
    if counts is not None:
        c0 = np.asarray(counts).astype(np.int64)[:V]
        cnt[:len(c0)] = c0
    top = np.argsort(-cnt, kind="stable")[:K]
    pos = np.full(V, K, np.int64)
    pos[top] = np.arange(K)
    bucket = V - 1
    assert cnt[bucket] == 0 and pos[bucket] == K, "останній рядок словника має бути порожнім"
    msk = (pos < K).astype(np.float32) * -1e4
    msk[bucket] = -1e4
    rare = 1 - cnt[top].sum() / max(cnt.sum(), 1) if cnt.sum() else 0.18
    rare = min(max(rare, 1e-3), 0.999)
    return dict(sv_ids=top.astype(np.int32), sv_pos=pos.astype(np.int32), sv_mask=msk, bucket=bucket, K=K,
                prior0=float(np.log(rare * K / (1 - rare))), rare_cap=rare_cap)


def short_logp(h, W, y, sc, ch, vocab=None):
    """log P(y) дворівневого виходу. h (N,d), W (V,d) голова, y (N,) int, sc — short_consts.
    Короткий словник (K+1 рядків, один шматок) + повний словник на C=місткість місцях (рідкісні — першими).
    vocab=None — один чіп; dist.VocabShard — W лише свої рядки (як shard.short_logp_vocab: короткі рядки — psum
    своїх частин, рідкісні — all_gather h[idx] + двоетапний lse)."""
    K, N = sc["K"], h.shape[0]
    b = sc["bucket"]
    if vocab is None:
        Ws = jnp.concatenate([W[jnp.asarray(sc["sv_ids"])], W[b:b + 1]])
    else:
        Ws = vocab.take_same(W, jnp.asarray(np.concatenate([sc["sv_ids"], [b]]).astype(np.int32)))
    bs = jnp.concatenate([jnp.zeros((K,), F32), jnp.full((1,), sc["prior0"], F32)])
    ys = jnp.asarray(sc["sv_pos"])[y]
    lse_s, ly_s = chunk_ce(h, Ws, bs, ys, min(ch, K + 1))
    logp = ly_s - lse_s
    rare = ys == K
    C = min(N, -(-int(round(sc["rare_cap"] * N)) // 256) * 256)
    key = rare.astype(F32) * 2 - jnp.arange(N, dtype=F32) / N
    idx = jax.lax.top_k(jax.lax.stop_gradient(key), C)[1]
    hit = rare[idx]
    vce = chunk_ce if vocab is None else vocab.ce
    lse_r, ly_r = vce(h[idx], W, jnp.asarray(sc["sv_mask"]), y[idx], ch)
    logp = logp.at[idx].add(jnp.where(hit, ly_r - lse_r, 0.0))
    stats = {"rare_drop": (rare.sum() - hit.sum()).astype(F32) / N}
    return logp, stats


def light_head_aux(lh, w0, W2, y, ch, vocab=None):
    """Допоміжна втрата легкої голови: mean(lse − ly) на кожному 4-му слові (градієнт іде в lh і l_head)."""
    a = lh[:, ::4]
    if a.dtype == BF:
        h4 = jnp.dot(a, w0.astype(BF).T, preferred_element_type=F32).astype(BF)
    else:
        h4 = jnp.dot(a, w0.astype(a.dtype).T, precision="highest")
    h4, y4 = h4.reshape(-1, h4.shape[-1]), y[:, ::4].reshape(-1)
    if vocab is None:
        lse, ly = chunk_ce_nb(h4, W2, y4, ch)
    else:
        lse, ly = vocab.ce(h4, W2, jnp.zeros((vocab.V,), F32), y4, ch)
    return (lse - ly).mean()
