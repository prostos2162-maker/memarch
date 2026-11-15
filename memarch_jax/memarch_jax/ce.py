"""Вихід на словник шматками (chunk_ce з власним VJP, як _ChunkCE у full2.py), здивування легкої голови
(SURP_INGRAPH, _surp_lse_impl) і дворівневий короткий словник SHORT_V (_short_logp, один чіп).

Точки округлення: логіти шматка — fp32-накопичення bf16-множення БЕЗ округлення до bf16 (16.11, див. CE_LOGITS),
lse — у fp32; логіт правильного слова — fp32-скалярний добуток bf16-рядків; назад — bf16 G."""
from functools import partial
import math
import os

import numpy as np
import jax
import jax.numpy as jnp

BF = jnp.bfloat16
F32 = jnp.float32


def _mm_bf16(a, b):
    """bf16 @ bf16 → bf16 (fp32-накопичення, одне округлення) — як torch-множення bf16-тензорів."""
    return jnp.dot(a, b, preferred_element_type=F32).astype(BF)


# 16.11 (fable52g) — ПРИЧИНА зламів 34014 / 34750 / 35750 (і всіх resume з ckpt 34000). Було: логіти = bf16-округлений
# вихід множення (як torch), і вперед, і у власному зворотному проході (перерахунок). На TPU ці два місця рахуються
# НЕоднаково: lse (вперед) — від округлених логітів, імовірності (назад) — від неокруглених (компілятор має право
# прибрати пару f32→bf16→f32 «з надлишковою точністю» і робить це не всюди). Тоді Σ_i p_i ≠ 1, і градієнт перестає
# бути градієнтом softmax: з'являється сила вздовж НУЛЬ-ПРОСТОРУ softmax (спільний зсув усіх логітів слова).
# У легкій голові (l_head0·l_head1) спільний зсув логітів = h4·(середній рядок l_head1); похибка округлення росте
# разом із зсувом (крок bf16 при |логіті| 40 — 0.25, при 1200 — 8, при 2150 — 16) → сила росте → зсув росте: петля.
# Докази (<local path>, чекпойнти 30000/34000/35250/35750):
#   • найбільше сингулярне l_head0: 5.6 → 9.8 (уже в «чистому» 34000) → 22.6 → 28.4, решта спектра 5.4 → 5.2;
#     спільний зсув логітів (rms) 79 → 1190 → 2150 за розкиду корисних логітів 2.5; light CE 3.7 → 11.8 → 21.9;
#   • сума рядків інерції l_head1 (у справжнього softmax ≡ 0): 0.0005 → 0.12 → 0.99;
#   • градієнт «lse від округлених, p від неокруглених» на стенді має cos +0.987 зі справжньою інерцією l_head0 з
#     ckpt 35250 і тягне моду ВГОРУ (−34.8); узгоджений градієнт (як на CPU) — cos −0.06 і тягне її ВНИЗ (+0.29).
#     Тому стенди на CPU/f32 злам не відтворювали ніколи.
# Тепер логіти — fp32-накопичення без округлення до bf16 в ОБОХ проходах (один і той самий вираз, прибирати нічого):
# градієнт точно узгоджений із втратою, а великий спільний зсув більше не з'їдає точність (fp32: 24 біти).
# CE_LOGITS=bf16 — старий шлях (A/B); CE_LOGITS=tpuA — лише для стенду: емуляція TPU (вперед округлено, назад ні).
CE_LOGITS = os.environ.get("CE_LOGITS", "f32")
assert CE_LOGITS in ("f32", "bf16", "tpuA"), CE_LOGITS


def _logits(hb, Wt, bwd=False):
    """Логіти шматка hb @ Wt (bf16-входи) → fp32. bwd — виклик із зворотного проходу (різниця лише в tpuA)."""
    z = jnp.dot(hb, Wt, preferred_element_type=F32)
    if CE_LOGITS == "f32":
        return z
    if CE_LOGITS == "bf16":
        return z.astype(BF).astype(F32)
    return z if bwd else jax.lax.reduce_precision(z, 8, 7)      # tpuA: bf16 = 8 біт порядку, 7 біт мантиси


def _gw(Gb, Wc):
    """Внесок шматка в dh: Gb @ Wc → fp32 (старий шлях — з округленням до bf16 на кожен шматок)."""
    z = jnp.dot(Gb, Wc, preferred_element_type=F32)
    return z.astype(BF).astype(F32) if CE_LOGITS == "bf16" else z


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
        lo = _logits(hb, Wb[c0:c1].T) + bf[c0:c1]
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


def _bwd_rows(hb, Wb, bf, y, lse, g_lse, g_ly, ch, ti=None, g_tv=None):
    """→ dh (n,d) fp32, dW (V,d) fp32 (ще без bf16-округлення), db (V,) fp32.
    ti/g_tv (n,k) (15.11, chunk_ce_top): котангенси top-k логітів додаються в G тих самих стовпців — без множень."""
    dh = jnp.zeros(hb.shape, F32)
    dW, db = [], []
    for c0, c1 in _chunks(Wb.shape[0], ch):
        lo = _logits(hb, Wb[c0:c1].T, bwd=True) + bf[c0:c1]              # перерахунок — той самий вираз, що вперед
        col = jnp.arange(c0, c1)[None]
        hit = (col == y[:, None])
        G = g_lse[:, None] * jnp.exp(lo - lse[:, None]) + g_ly[:, None] * hit
        if g_tv is not None:
            for j in range(ti.shape[1]):
                G = G + g_tv[:, j:j + 1] * (col == ti[:, j:j + 1])
        Gb = G.astype(BF)
        dh = dh + _gw(Gb, Wb[c0:c1])
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


# 15.11 (швидкість --inst_word --iw_rewrite): той самий chunk_ce, але прохід вперед ще й віддає k найбільших логітів
# стовпців < kmax (короткий словник без кошика) — top-k переписаної відповіді без другого множення h·Wsᵀ.
# Градієнт: chunk_ce + котангенси top-k логітів (15.11: переписана відповідь на top-k конкурентах) — у тому ж
# зворотному проході по шматках (додавання в G тих самих стовпців), без нових множень на словник.
def _lse_top_rows(hb, Wb, bf, ch, k, kmax):
    lses, tv, ti = [], [], []
    for c0, c1 in _chunks(Wb.shape[0], ch):
        lo = _logits(hb, Wb[c0:c1].T) + bf[c0:c1]
        lses.append(_lse(lo, -1))
        if c0 < kmax:
            lom = lo[:, :kmax - c0] if c1 > kmax else lo           # стовпці ≥ kmax (кошик) — не кандидати
            # 15.11: approx_max_k — апаратний TPU-пошук (top_k на TPU сортує, +~28 мс/крок); на CPU = точний
            v, i = jax.lax.approx_max_k(lom, min(k, lom.shape[-1]), recall_target=0.95)
            tv.append(v); ti.append(i + c0)
    v, i = jnp.concatenate(tv, -1), jnp.concatenate(ti, -1)
    if len(tv) > 1:
        v, j = jax.lax.top_k(v, k)
        i = jnp.take_along_axis(i, j, 1)
    return _lse(jnp.stack(lses, -1), -1), v, i


def _chunk_ce_top_fwd(h, W, b, y, ch, k, kmax):
    hb, Wb = h.astype(BF), W.astype(BF)
    bf = b.astype(F32)
    N = hb.shape[0]
    if N <= TOK:
        lse, tv, ti = _lse_top_rows(hb, Wb, bf, ch, k, kmax)
        ly = _ly_rows(hb, Wb, bf, y)
    else:
        nb = -(-N // TOK)
        lse, ly, tv, ti = jax.lax.map(
            lambda a: (lambda r: (r[0], _ly_rows(a[0], Wb, bf, a[1]), r[1], r[2]))(_lse_top_rows(a[0], Wb, bf, ch, k, kmax)),
            (_blocks(hb, nb), _blocks(y, nb)))
        lse, ly = lse.reshape(-1)[:N], ly.reshape(-1)[:N]
        tv, ti = tv.reshape(-1, tv.shape[-1])[:N], ti.reshape(-1, ti.shape[-1])[:N]
    ti = ti.astype(jnp.int32)
    return (lse, ly, tv, ti), (hb, Wb, bf, y, lse, ti, jnp.zeros((), h.dtype), jnp.zeros((), W.dtype),
                               jnp.zeros((), b.dtype))


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def chunk_ce_top(h, W, b, y, ch, k, kmax):
    """(lse, логіт слова, top-k логітів стовпців < kmax (N,k) fp32, їхні номери (N,k)) — див. chunk_ce."""
    return _chunk_ce_top_fwd(h, W, b, y, ch, k, kmax)[0]


def _chunk_ce_top_bwd(ch, k, kmax, res, g):
    hb, Wb, bf, y, lse, ti, hz, wz, bz = res
    g_lse, g_ly, g_tv = g[0], g[1], g[2].astype(F32)
    N = hb.shape[0]
    if N <= TOK:
        dh, dW, db = _bwd_rows(hb, Wb, bf, y, lse, g_lse, g_ly, ch, ti, g_tv)
    else:
        nb = -(-N // TOK)
        xs = (_blocks(hb, nb), _blocks(y, nb), _blocks(lse, nb), _blocks(g_lse, nb), _blocks(g_ly, nb),
              _blocks(ti, nb), _blocks(g_tv, nb))

        def body(c, a):
            dh_b, dW_b, db_b = _bwd_rows(a[0], Wb, bf, a[1], a[2], a[3], a[4], ch, a[5], a[6])
            return (c[0] + dW_b, c[1] + db_b), dh_b.astype(hz.dtype)
        (dW, db), dh = jax.lax.scan(body, (jnp.zeros(Wb.shape, F32), jnp.zeros(Wb.shape[:1], F32)), xs)
        dh = dh.reshape(-1, hb.shape[1])[:N]
    return (dh.astype(hz.dtype), dW.astype(BF).astype(wz.dtype), db.astype(bz.dtype), None)


chunk_ce_top.defvjp(_chunk_ce_top_fwd, _chunk_ce_top_bwd)


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
    lse = _lse(jnp.stack([_lse(_logits(hb, Wb[c0:c1].T), -1)
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


def short_logp(h, W, y, sc, ch, vocab=None, prio=None, topk=0, want_lse=False):
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
    if topk:   # 15.11: top-k короткого словника з того ж проходу (для переписаної відповіді, див. short_topk)
        lse_s, ly_s, tv, ti = chunk_ce_top(h, Ws, bs, ys, min(ch, K + 1), int(topk), K)
        # (p top-k без градієнта, номери, Ws, сирі логіти top-k З градієнтом — для переписаної відповіді)
        top = (jnp.exp(jax.lax.stop_gradient(tv - lse_s[:, None])), jax.lax.stop_gradient(ti), Ws, tv)
    else:
        lse_s, ly_s = chunk_ce(h, Ws, bs, ys, min(ch, K + 1))
    logp = ly_s - lse_s
    rare = ys == K
    C = min(N, -(-int(round(sc["rare_cap"] * N)) // 256) * 256)
    key = rare.astype(F32) * 2 - jnp.arange(N, dtype=F32) / N
    if prio is not None:   # нагадування: відповіді — першими в місткості рідкісних (інакше лише log P(кошика), без слова)
        key = key + (rare & (prio.reshape(-1) > 0)).astype(F32) * 4
    idx = jax.lax.top_k(jax.lax.stop_gradient(key), C)[1]
    hit = rare[idx]
    vce = chunk_ce if vocab is None else vocab.ce
    lse_r, ly_r = vce(h[idx], W, jnp.asarray(sc["sv_mask"]), y[idx], ch)
    logp = logp.at[idx].add(jnp.where(hit, ly_r - lse_r, 0.0))
    stats = {"rare_drop": (rare.sum() - hit.sum()).astype(F32) / N}
    if want_lse:   # 15.11 (--inst_word iw_random): lse короткого рівня З ГРАДІЄНТОМ — для log p_o = z_o − lse_s
        stats["_lse_s"] = lse_s
    if topk:
        return logp, stats, top
    return logp, stats


def short_only_logp(h, W, y_short_tok, sc, ch, vocab=None):
    """15.11: log P(слова) лише по короткому словнику (+кошик у lse), без рідкісної частини — для лідера short_top
    (він завжди з короткого словника). Дешевше за short_logp (немає повного словника на місткості рідкісних)."""
    K = sc["K"]
    b = sc["bucket"]
    if vocab is None:
        Ws = jnp.concatenate([W[jnp.asarray(sc["sv_ids"])], W[b:b + 1]])
    else:
        Ws = vocab.take_same(W, jnp.asarray(np.concatenate([sc["sv_ids"], [b]]).astype(np.int32)))
    bs = jnp.concatenate([jnp.zeros((K,), F32), jnp.full((1,), sc["prior0"], F32)])
    ys = jnp.asarray(sc["sv_pos"])[y_short_tok]
    lse_s, ly_s = chunk_ce(h, Ws, bs, ys, min(ch, K + 1))
    return ly_s - lse_s


def short_top(h, W, sc, ch, vocab=None):
    """15.11 (Instinct на словах): без градієнта — найбільша ймовірність короткого словника і її слово.
    h (N,d) → (p_top (N,) fp32, top_tok (N,) int32 — номер у повному словнику). Кошик «рідкісне» в lse є,
    але лідером бути не може. Ті самі множення, що в short_logp (XLA зливає однакові dot)."""
    K, N = sc["K"], h.shape[0]
    b = sc["bucket"]
    if vocab is None:
        Ws = jnp.concatenate([W[jnp.asarray(sc["sv_ids"])], W[b:b + 1]])
    else:
        Ws = vocab.take_same(W, jnp.asarray(np.concatenate([sc["sv_ids"], [b]]).astype(np.int32)))
    h, Ws = jax.lax.stop_gradient(h), jax.lax.stop_gradient(Ws)
    hb, Wb = h.astype(BF), Ws.astype(BF)
    mx, am, ls = [], [], []
    for c0, c1 in _chunks(K + 1, min(ch, K + 1)):
        lo = _logits(hb, Wb[c0:c1].T)
        if c1 > K:                                            # кошик: зсув prior0, лідером не рахується
            lo = lo.at[:, K - c0].add(jnp.float32(sc["prior0"]))
            lom = lo.at[:, K - c0].set(-1e30)
        else:
            lom = lo
        ls.append(_lse(lo, -1)); mx.append(lom.max(-1)); am.append(lom.argmax(-1) + c0)
    lse = _lse(jnp.stack(ls, -1), -1)
    mxa, ama = jnp.stack(mx, -1), jnp.stack(am, -1)
    j = mxa.argmax(-1)
    top_s = jnp.take_along_axis(ama, j[:, None], 1)[:, 0]
    p_top = jnp.exp(jnp.take_along_axis(mxa, j[:, None], 1)[:, 0] - lse)
    return p_top, jnp.asarray(sc["sv_ids"])[top_s].astype(jnp.int32)


def short_topk(h, W, sc, ch, k=4, vocab=None, with_ws=False):
    """15.11 (переписана відповідь): без градієнта — k найбільших імовірностей короткого словника і їхні слова
    (N,k) за спаданням. Кошик лідером/конкурентом не буває. with_ws=True — ще й рядки короткого словника Ws
    (K+1,d) З градієнтом: переписування рахує δ·h·(Ws[o]−Ws[c]) двома скалярними добутками на слово, без
    другого множення на весь словник (швидкість)."""
    K = sc["K"]
    b = sc["bucket"]
    if vocab is None:
        Ws_g = jnp.concatenate([W[jnp.asarray(sc["sv_ids"])], W[b:b + 1]])
    else:
        Ws_g = vocab.take_same(W, jnp.asarray(np.concatenate([sc["sv_ids"], [b]]).astype(np.int32)))
    h, Ws = jax.lax.stop_gradient(h), jax.lax.stop_gradient(Ws_g)
    hb, Wb = h.astype(BF), Ws.astype(BF)
    # 15.11: шматки ≤2048 стовпців по черзі (lax.map — живий лише один шматок f32[N,c]); раніше одним f32[N,K+1] +
    # дві повні копії .at[].add/.set — OOM на v6e. Поправка кошика — jnp.where по глобальному індексу стовпця;
    # доповнення останнього шматка — −1e30 (exp → 0 у lse, у top-k не потрапляє). Злиття lse/top-k — як раніше.
    c = min(ch, 2048, K + 1)
    n = -(-(K + 1) // c)
    kk = min(k, c)
    Wp = jnp.pad(Wb, ((0, n * c - (K + 1)), (0, 0))).reshape(n, c, Wb.shape[-1])

    def one(a):
        j, Wc = a
        col = j * c + jnp.arange(c)
        lo = _logits(hb, Wc.T)
        lo = jnp.where((col == K)[None, :], lo + jnp.float32(sc["prior0"]), lo)
        lo = jnp.where((col > K)[None, :], jnp.float32(-1e30), lo)
        lom = jnp.where((col >= K)[None, :], jnp.float32(-1e30), lo)
        vv, ii = jax.lax.top_k(lom, kk)
        return _lse(lo, -1), vv, ii + j * c
    ls, vs, ix = jax.lax.map(one, (jnp.arange(n), Wp))          # (n,N), (n,N,kk), (n,N,kk)
    lse = _lse(jnp.moveaxis(ls, 0, -1), -1)
    v = jnp.moveaxis(vs, 0, 1).reshape(vs.shape[1], -1)
    i = jnp.moveaxis(ix, 0, 1).reshape(ix.shape[1], -1)
    v2, j = jax.lax.top_k(v, k)
    i2 = jnp.take_along_axis(i, j, 1)
    if with_ws:
        return jnp.exp(v2 - lse[:, None]), i2.astype(jnp.int32), Ws_g
    return jnp.exp(v2 - lse[:, None]), jnp.asarray(sc["sv_ids"])[i2].astype(jnp.int32)


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
