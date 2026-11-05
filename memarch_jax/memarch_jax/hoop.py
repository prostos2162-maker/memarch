"""Uroboros (internal name: "hoop") -- surprise-proportional compute routing; experimental.

«Обруч» (ідея Стефана, Уроборос): обчислення ∝ здивуванню, і рішення приймає сама модель.

Змія/обруч = помилки (здивування на слово), чорна діра = знання (ваги). Те, що модель уже передбачає,
«вже з'їдене» — воно не має крутитися у важких шарах. Тому:

  1. Перед важкими MoE-шарами маленька голова (на lh — виході легких шарів, ДО важких) передбачає ВЛАСНУ
     майбутню помилку моделі на кожному слові: ŝ[t] ≈ −log P_моделі(y[t]) (головна втрата цього слова).
     Ціль — справжня помилка, зупинена (stop_gradient): «обруч усередині горизонту» — знання вчиться
     передбачати власні помилки.
  2. Важкі шари бачать лише A = budget·T слів з найбільшим ŝ (жорсткий top-k на послідовність, статична
     форма для TPU). Решта йде залишково (h без змін) — їх «уже з'їдено».
  3. Бюджет стискається з навчанням (обруч звужується): next_budget() дивиться, яка частка слів має
     передбачену помилку нижче порогу «з'їденого», і переходить на нижчий рівень (кілька рівнів → кілька
     перекомпіляцій, форма в межах рівня статична).

Сама вибірка не диференційована (top-k), градієнт важких шарів іде лише через вибрані слова; голова
обруча вчиться окремою MSE-втратою (cfg.hoop_w)."""
import numpy as np
import jax
import jax.numpy as jnp

F32 = jnp.float32
S_MAX = 20.0          # обрізання цілі/прогнозу (нат): дуже рідкісні слова не мають розхитувати регресію
HID = 64              # ширина прихованого шару голови обруча


def init(key, d):
    """Параметри голови обруча: [lh ‖ log1p(s_)] → HID → 1. Вихідний зсув = 3 нати (типова помилка на старті)."""
    k1, k2 = jax.random.split(key)
    b1 = 1.0 / np.sqrt(d + 1)
    b2 = 1.0 / np.sqrt(HID)
    return {"w1": jax.random.uniform(k1, (HID, d + 1), F32, -b1, b1), "b1": jnp.zeros((HID,), F32),
            "w2": jax.random.uniform(k2, (1, HID), F32, -b2, b2), "b2": jnp.full((1,), 3.0, F32)}


def predict(p, lh, s_cur):
    """Передбачена помилка моделі на кожному слові: ŝ (B,T) fp32 ≥ 0.
    lh (B,T,d) — вихід легких шарів (вхід зупинено: голова не тягне легкі шари під себе);
    s_cur (B,T) — уже пораховане здивування легкої голови на ПОТОЧНОМУ слові (дешева підказка, без градієнта)."""
    lh = jax.lax.stop_gradient(lh).astype(F32)
    sc = jnp.log1p(jnp.clip(jax.lax.stop_gradient(s_cur), 0.0, S_MAX))[..., None]
    z = jnp.concatenate([lh, sc], -1)
    hid = jax.nn.gelu(z @ p["w1"].astype(F32).T + p["b1"].astype(F32))
    out = hid @ p["w2"].astype(F32).T + p["b2"].astype(F32)
    return jax.nn.softplus(out[..., 0])                      # помилка невід'ємна


def n_slots(T, budget):
    """Скільки слів на послідовність іде у важкі шари (статично)."""
    return max(1, min(T, int(round(T * budget))))


def select(pred, budget):
    """Жорсткий top-k за передбаченою помилкою: idx (B,A) за зростанням позиції, valid (B,A) — усі True.
    Слово 0 (без контексту) завжди вибране — як і в якорях (s[:,0]=1e4)."""
    B, T = pred.shape
    A = n_slots(T, budget)
    key = jax.lax.stop_gradient(pred).at[:, 0].set(1e9)
    _, idx = jax.lax.top_k(key, A)
    idx = jnp.sort(idx, axis=-1)
    return idx, jnp.ones((B, A), bool)


def loss(pred, nll_tok):
    """MSE між передбаченою і справжньою (зупиненою) помилкою, обидві обрізані до S_MAX."""
    tgt = jnp.clip(jax.lax.stop_gradient(nll_tok).astype(F32), 0.0, S_MAX)
    return jnp.mean((jnp.clip(pred, 0.0, S_MAX) - tgt) ** 2)


def stats(pred, nll_tok, idx, eat):
    """Лог: частка «з'їдених» слів (ŝ < eat), кореляція ŝ зі справжньою помилкою, помилка на вибраних/пропущених."""
    pred = jax.lax.stop_gradient(pred).astype(F32)
    tgt = jax.lax.stop_gradient(nll_tok).astype(F32)
    B, T = pred.shape
    sel = jnp.zeros((B, T), F32).at[jnp.arange(B)[:, None], idx].set(1.0)
    pc, tc = pred - pred.mean(), tgt - tgt.mean()
    corr = (pc * tc).mean() / jnp.maximum(jnp.sqrt((pc ** 2).mean() * (tc ** 2).mean()), 1e-9)
    return {"hoop_eaten": (pred < eat).astype(F32).mean(),
            "hoop_corr": corr,
            "hoop_nll_sel": (tgt * sel).sum() / jnp.maximum(sel.sum(), 1.0),
            "hoop_nll_skip": (tgt * (1 - sel)).sum() / jnp.maximum((1 - sel).sum(), 1.0)}


def next_budget(cur, eaten, levels, margin=1.25, corr=None, min_corr=0.5):
    """Адаптивний бюджет (обруч стискається з навчанням), на хості між кроками.
    Потреба = (1 − eaten)·margin — частка слів, які ще не з'їдено, з запасом. Береться найменший рівень ≥ потреби,
    але не більше ніж на один рівень униз за раз і ніколи не вище за поточний (обруч лише звужується).
    levels — спадний кортеж статичних бюджетів (кожен рівень = одна компіляція).
    eaten/corr — краще середні за вікно кроків. corr < min_corr → голова ще не знає своїх помилок, не стискаємо
    (інакше на старті, коли прогноз — константа, обруч стиснувся б наосліп)."""
    if corr is not None and float(corr) < min_corr:
        return cur
    levels = sorted(levels, reverse=True)
    need = (1.0 - float(eaten)) * margin
    i = levels.index(cur) if cur in levels else 0
    if i + 1 < len(levels) and levels[i + 1] >= need:
        return levels[i + 1]
    return levels[i]
