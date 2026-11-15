"""«Фішки» (схема Стефана 11.11): важка частина = багато однакових невеликих FFN-шарів (relu², без уваги).
Увесь крок — тексти одного виду; текст іде через КОМБІНАЦІЮ фішок свого виду (порядок — дані кроку, без
перекомпіляцій для того самого розміру комбінації). Роутер (лінійний, за середнім станом після легких шарів)
вчиться вгадувати вид — у готовій моделі він вибирає набір фішок.
Шардування: ваги фішок порізані по d між чипами (FSDP); перед використанням вибрана фішка збирається all_gather
(«їздять ваги, а не слова»), градієнт повертається reduce_scatter (транспозиція all_gather)."""
import jax
import jax.numpy as jnp
from jax import lax

F32 = jnp.float32


def pools(cfg):
    """Набори фішок за видом: own своїх + over спільних із сусіднім видом (кільце). → list[list[int]]."""
    K, NF, P = cfg.fish_kinds, cfg.fish_n, cfg.fish_pool
    own = NF // K
    over = max(0, P - own)
    return [list(range(own * k, own * k + own))[:P] + [(own * (k + 1) + j) % NF for j in range(over)]
            for k in range(K)]


def init(key, cfg):
    d, NF, HF = cfg.d, cfg.fish_n, cfg.fish_hf
    k1, k2 = jax.random.split(key)
    dt = cfg.cdt
    # по ОДНІЙ фішці (lax.map): інакше випадкові числа всіх 110 фішок разом (u32 + f32 + bf16) брали ~17 ГБ
    # тимчасової пам'яті на чип v6e-8 (tools/init_compile_check.py, 12.11)
    up = lax.map(lambda k: (jax.random.normal(k, (d, HF), F32) * d ** -0.5).astype(dt), jax.random.split(k1, NF))
    down = lax.map(lambda k: (jax.random.normal(k, (HF, d), F32) * HF ** -0.5 * 0.5).astype(dt), jax.random.split(k2, NF))
    return {"norm": jnp.ones((NF, d), dt), "up": up, "down": down}


def _center(xn):
    """fish_center (14.11, fw_s2): прибрати зі входу фішки спільний вектор потоку — каузальне середнє x̂ по
    попередніх словах рядка (cumsum/t, без зазирання вперед). Причина: на ck34000 |mean x̂|/rms x̂ = 0.42, а стовпці
    up усіх фішок дрейфнули проти цього вектора (a_mu = mean·up ≈ −6 при std решти 1.3) → relu² мертвий для 99.9%
    пар нейрон×слово, градієнта немає, фішки не вчаться. Вартість O(B·T·d) проти 2·d·HF матмулу (~1e-4)."""
    xf = xn.astype(F32)
    t = jnp.arange(1, xf.shape[1] + 1, dtype=F32)[None, :, None]
    return (xf - jnp.cumsum(xf, 1) / t).astype(xn.dtype)


def _one(norm, up, down, h, center=False):
    x = h.astype(F32)
    xn = (x * lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)).astype(h.dtype)
    if center:
        xn = _center(xn)
    xn = xn * norm.astype(h.dtype)
    a = jnp.einsum("btd,dh->bth", xn, up.astype(h.dtype), preferred_element_type=F32).astype(h.dtype)
    return h + jnp.einsum("bth,hd->btd", jnp.square(jax.nn.relu(a)), down.astype(h.dtype),
                          preferred_element_type=F32).astype(h.dtype)


def chain(pf, h, order, axis=None, remat=False, center=False):
    """h (B,T,d); order (C,) int — номери фішок цього кроку по черзі. axis — ім'я осі shard_map (ваги порізані по d)."""
    def step(h, i):
        up = lax.dynamic_index_in_dim(pf["up"], i, 0, keepdims=False)          # (d[/W], HF)
        down = lax.dynamic_index_in_dim(pf["down"], i, 0, keepdims=False)      # (HF[/W], d)
        norm = lax.dynamic_index_in_dim(pf["norm"], i, 0, keepdims=False)
        if axis is not None:                                                   # зібрати вибрану фішку з усіх чипів
            up = lax.all_gather(up, axis, axis=0, tiled=True)
            down = lax.all_gather(down, axis, axis=0, tiled=True)
        return _one(norm, up, down, h, center)
    if remat:
        step = jax.checkpoint(step, policy=None if remat is True else remat)
    for c in range(order.shape[0]):
        h = step(h, order[c])
    return h


def router_loss(w, lh, kind):
    """w (d,K); lh (B,T,d) (стан після легких шарів, нормований); kind (B,) int → (CE, точність)."""
    logits = jnp.einsum("bd,dk->bk", lax.stop_gradient(lh).astype(F32).mean(1), w.astype(F32))
    lp = jax.nn.log_softmax(logits, -1)
    ce = -jnp.take_along_axis(lp, kind[:, None], 1).mean()
    acc = (logits.argmax(-1) == kind).astype(F32).mean()
    return ce, acc, logits
