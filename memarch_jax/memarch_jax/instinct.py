"""Оптимізатор Instinct (семантика exp_opt.InstinctOptB) — чисті функції над pytree параметрів.

Порядок у кроці (як у torch): prepare(loss, sit) → update(усі ваги) → finish().
    st = init(params, d_sit, cfg, spec)
    params, st = step(params, grads, st, lr=..., situation=sit, loss=loss, spec=spec, cfg=cfg)

Види кроку (kind) задаються ЯВНО за іменами ваг (design §3, не за shape[0] ≥ 10000):
    vocab — emb/head/l_head1/book (нормування лише по рядку, рядки без градієнта не рухаються);
    hs3   — експерти (E, in, out): гомеостаз по входах і виходах;
    hs    — інші 2D (router, memory/reread): гомеостаз рядків і стовпців (OPT_HS_LAG: норма стовпця з минулого кроку);
    hsS   — light-лінійні (у torch порізані між чипами): свіжа норма стовпця щокроку, без lag;
    row   — ≥4D; tensor — 0D/1D.

Стохастичне округлення (SR) — лічильникове (хеш lowbias32 номера елемента й насіння кроку), як у Pallas-ядрах:
детерміноване й відтворюване; cfg.sr=False — округлення до найближчого (для тестів проти torch з OPT_SR_OFF=1).
Pallas-ядра (pallas_hs1 / pallas_instinct з D:\\ai\\memarch) — коли вага й інерція bf16 і SR увімкнено; на CPU —
interpret=True. Інакше — чистий jnp (та сама математика, у тому числі той самий хеш SR → побітово ті самі біти).
"""
import dataclasses
import functools
import math
import os
import sys

import jax
import jax.numpy as jnp

EPS = 1e-8
_TORCH_DIR = os.environ.get("MEMARCH_TORCH_DIR", r"D:\ai\memarch")


def _kernels():
    """Ліниво імпортуємо Pallas-ядра Стефана (вони лежать поруч із torch-кодом, але самі — чистий JAX)."""
    # копії лежать у пакеті (на TPU-VM немає D:i\memarch); оригінал — лише запасний шлях
    try:
        from . import pallas_instinct, pallas_hs1
    except ImportError:
        if _TORCH_DIR not in sys.path:
            sys.path.insert(0, _TORCH_DIR)
        import pallas_instinct, pallas_hs1
    return pallas_instinct, pallas_hs1


# ------------------------------------------------------------------------------------------------- налаштування
@dataclasses.dataclass(frozen=True)
class InstinctCfg:
    lr_peak: float = 6e-4
    weight_decay: float = 0.1
    cards: int = 4
    beta: float = 0.9
    new_card: float = 0.5
    ema: float = 0.98
    good: float = 0.05
    bad: float = 0.2
    top: float = 0.1
    relax: float = 0.01
    scale: float = 0.45
    m_bf16: bool = True
    rel_every: int = 8
    rel_cond: bool = True            # rel-норми під lax.cond лише на кроках want_rel (False — щокроку + where)
    hs_lag: bool = True              # OPT_HS_LAG
    sr: bool = True                  # стохастичне округлення bf16-записів (False — до найближчого)
    use_pallas: bool = True          # злиті ядра для hs/hs3/vocab (лише bf16 вага + bf16 інерція + SR)
    interpret: bool | None = None    # None — авто: interpret на не-TPU
    vocab_lr_mul: float = 1.0        # VOCAB_LR_MUL
    # ПРОПОЗИЦІЯ 12.11 (вимкнено): градієнт кожної ваги ділиться на свій rms ПЕРЕД інерцією — крок з великим
    # градієнтом (текст) не забиває в інерції кроки з малим (нагадування); напрям кроку той самий
    gnorm: bool = False
    # 12.11 (Стефан: «винне те, що діяло саме в момент помилки»): "abs" — винні = top-частка за rel=|g|/|w| (як було;
    # на ділі завжди ті самі тензори з малими вагами: fish_router, prev_g, light_a/0 — їхній крок ріс до 3×);
    # "ema" — винні = ті, чий rel ЦЬОГО кроку найбільше перевищує їхнє власне середнє (стан rel_avg)
    culprit: str = "abs"
    # 13.11 (кут оптимізатора, ПРОПОЗИЦІЯ, вимкнено): «впевненість кроку». Ваги троса (light_b/0.qkv) у змішаному режимі
    # мають ||m||/sqrt(E||g||²) ≈ 0.16–0.23 — рівень чистого шуму (для β=0.9 шум дає 0.229, сталий сигнал ≈ 1), але
    # нормування дає їм ПОВНИЙ крок → випадкове блукання: 52% норми q/k за 250 кроків (ckpt 500→750 v3), ||w|| росте
    # 64→78 рівно як у блуканні. snr=True: крок тензора × clip(c / snr_ref, snr_floor, 1), c = ||m_new|| / sqrt(v),
    # v — EMA(||g||²) з snr_beta (скаляр на тензор, стан opt.snr_v). Напрям кроку (гомеостаз рядків/стовпців) той самий;
    # коли градієнт сталий — крок як був; коли кроки сперечаються/шум — тензор майже стоїть (пам'ять не розмивається).
    snr: bool = False
    snr_beta: float = 0.95
    snr_ref: float = 1.0
    snr_floor: float = 0.1
    snr_skip: tuple = ("fish.",)    # фішки: 2 з 110 за крок, свій _freeze_idle; норма по всьому стеку — зайвий прохід
    snr_only: tuple = ()            # непорожній — лише тензори з цими префіксами (напр. ("light_a.", "light_b.") — шлях троса)
    # 13.11 (Стефан: «картка на кожен вид»): з фішками кожен крок — один вид тексту, рівень втрати різний для видів
    # (код ~3.2, англ ~3.5, укр ~3.9, синтетика інша), а 4 картки за схожістю входу змішували види → знак несподіванки
    # скакав щокроку і «винні» підсилювались даремно (стрибок mix 15000→15250: фінальні шари). card_by_kind=True —
    # картку ситуації задає ВІДОМИЙ вид кроку (prepare(card=...)), cards = видів + 1 (останя — синтетика).
    card_by_kind: bool = False
    # 13.11 (фікс зламу 15000→15250, раунд r7, Стефан: «вбудувати і тренувати»). Злам: final/0 пише в потік спільний
    # вектор m; його поворот (з light_b/light_a), підсилений увагою final/*, виводить m з каналу, який гасить фінальна
    # norm → петля. stream_proj=True (K1): з градієнта й інерції матриць light_a/light_b/final (.o/.down/.qkv)
    # прибирається складова вздовж U — одиничного EMA(0.9) середнього h перед фінальною norm (стан opt.U); тобто
    # оптимізатор не тягне сам спільний вектор. norm_anchor=True (K2): 1D norm-ваги цих шарів після кроку тягнуться
    # до 1 (w ← w − lr·wd·(w−1)), як wd для 2D. Разом з gnorm на стенді справжніх кроків: 15000 nll −0.04 і 15250
    # −0.22 проти бази (<local path>).
    stream_proj: bool = False
    # fix14 (13.11, вимкнено): gnorm_rawrel — при gnorm rel=|g|/|w| (винні, картки) рахується з СИРОГО градієнта
    # (з gnorm ||g_norm||≡sqrt(N) → rel ставав константою тензора ~150, винні = ті, в кого спадає |w|).
    # snr_dz — dead-zone впевненості замість c/snr_ref: v=c̄²=EMA(||m_new||²/||g||²), k=(1−β)/(1+β) (шум),
    # r̂=max(c̄²−k,0)/(1−c̄²), fac=clip(sqrt(r̂·H/k), floor, 1), H=snr_horizon: гальмує лише блукання.
    gnorm_rawrel: bool = False
    snr_dz: bool = False
    snr_horizon: float = 250.0
    norm_anchor: bool = False
    # fix15 (13.11, ідея Стефана «не давати m вилазити більше, ніж модель може прийняти; вихід розтягнути на кілька
    # кроків»): m_rate=True — після кроку U-складова руху матриць потоку (STREAM_W×STREAM_KEYS) c=ΔW·U обмежується за
    # КОГЕРЕНТНОЮ швидкістю v=EMA_0.9(e) ≤ κ·base (base — повільна EMA_0.998 |v|, власна норма тензора); надлишок → банк,
    # віддається наступними кроками (Σe = Σc). Перші m_rate_warm кроків τ=∞ (лише вчить base) → resume без стрибка.
    m_rate: bool = False
    m_rate_kappa: float = 2.0
    m_rate_warm: int = 100
    # 13.11 (Стефан: «сходи, на яких дивишся під ноги»): відкладене віддається лише там, де свіжий градієнт ще згоден
    # (bank·(g·U) < 0, бо крок іде проти градієнта); де модель уже каже «назад» — відкладене гаситься.
    m_rate_gate: bool = False
    # fix16 (13.11, розбір зовнішньої моделі): для light_a/light_b/final і норм (norm, l_norm) — jnp-шлях, де
    # (1) інерція, що йде ПРОТИ свіжого градієнта, проєктується (рядок / весь 1D-тензор), (2) координати кроку проти
    # градієнта маскуються (cautious), (3) 1D-норми крокують відносно |w| (≤2×), (4) підсилювачі карток ≤ g_max.
    # Збережена інерція m — справжня EMA (без змін).
    safe16: bool = False
    safe16_prefix: tuple = ("light_a.", "light_b.", "final.")
    g_max: float = 3.0
    # 14.11 (Slingshot / Peri-LN: ріст норми ваг без wd → нестабільність): wd_norm>0 — decoupled weight decay ДО НУЛЯ
    # для 1D норм-ваг (norm, l_norm, *.n1, *.n2), яким Instinct дає кроки сталої довжини без wd (ріст без меж, напр.
    # канал 1249 фінальної norm 1.5→3.06). Усередині кроку Instinct → стохастичне округлення bf16 (малі зміни не губляться).
    wd_norm: float = 0.0
    # fix18 (14.11, r13_an + відкати r2: 55.6% шкоди 15000→15250 — qkv final/1, 30.1% — o final/0; масштаб логітів
    # уваги final/* шкоду НЕ повертає, тобто зламався зміст q/k/v/o, а не гострота): множник кроку (lr_eff, без wd)
    # для тензорів з цими префіксами, ((префікс, множник), ...). () — вимкнено. Пряма модель не змінюється → resume
    # без стрибка; статичний python-множник на листок → без перекомпіляцій і без ціни на TPU.
    lr_mul: tuple = ()
    # fix20 (14.11, r20_s1): гальмо несподіванки. s=(очікування−loss)/очікування картки; якщо крок МНОГО гірший за
    # очікуване (−s > surge_thr), lr_eff *= (surge_thr/−s)². 0 — вимкнено. Звичайні кроки (|s|<0.15) не чіпає.
    surge_thr: float = 0.0
    # fix21 (14.11, fable52; відтворено на ПК: <local path>): гасіння сплеску градієнта ПО
    # ТЕНЗОРУ. Instinct нормує крок по миттєвій інерції й не знає масштабу градієнта: коли вздовж гострого напряму
    # (спільний вектор потоку × relu²-нейрони) росте коливання, ||g|| тензора росте, а крок лишається повним; snr його
    # ще й ПІДНІМАЄ (впевненість = ||m||/sqrt(EMA g²) → 1). Злам 34014: дані звичайні, ||g|| light_b.1.down ×28 за
    # 20 кроків. damp=k>0 (стан opt.dv (K, P) — типова ||g||² тензора на картці-виді кроку, EMA damp_beta; opt.df (P,) —
    # множник останнього кроку для журналу): якщо ||g|| > k·типової, то (1) в інерцію градієнт входить обрізаним до
    # k·типової — один батч не захоплює m на десятки кроків; (2) крок тензора × (k·типова/||g||)^damp_p; snr рахується
    # вже від обрізаного градієнта (сплеск не піднімає впевненість). Поки ||g|| ≤ k·типової — крок побітово як був
    # (lax.cond: без зайвого проходу по g). Лише тензори damp_only (легкі/фінальні/норми — дешево; фішки й словникові
    # таблиці не чіпаємо: зайвий прохід по 11B/0.3B ваг).
    damp: float = 0.0
    damp_p: float = 1.0
    damp_beta: float = 0.99
    damp_cap: float = 2.0            # у EMA типової норми сплеск входить не більше ніж (cap·типова)²: гальмо не «звикає»
    damp_only: tuple = ("light_a.", "light_b.", "final.", "norm", "l_norm", "l_head0")
    # 15.11 err_step — задум Стефана (29.10, уточнено 15.11): у момент здивування виправити прогноз до БІЛЬШ точного.
    # Довжина кроку кожного тензора = lr × (похибка зараз / очікувана похибка) × (звична ||g|| тензора / ||g|| зараз).
    # Похибка = loss кроку, очікувана = expect картки (вже є: 1 − s); звична ||g|| — стан dv (як fix21, EMA damp_beta).
    # Звичайний крок ≈ як був; більша похибка → більший крок; сплеск чутливості (||g|| ×28) → крок ×1/28.
    # err_cap — мій захист (не з задуму): множник не більше err_cap, щоб при ||g||→0 крок не летів у нескінченність.
    # Потребує damp>0 (стан dv); діє на ВСІ тензори; градієнт в інерцію не обрізається.
    err_step: bool = False
    err_cap: float = 3.0
    # 15.11 запобіжник-відкатник (Стефан): loss кроку > rb_ratio × звичний (EMA rb_beta на картці-виді) → ваги й
    # інерція повертаються на стан до попереднього кроку, оновлення цього кроку не застосовується; НАСТУПНИЙ крок
    # повторно вчиться на батчі попереднього кроку (збережений на пристрої, opt["rb_b"]) з кроком × rb_lr (варіант А).
    # Стан: rb_p (копія ваг), rb_m (копія m), rb_lema (K,) — у opt (шардування успадковується за формою).
    rollback: bool = False
    rb_ratio: float = 1.5
    rb_beta: float = 0.98
    rb_lr: float = 0.5
    rb_warm: int = 50
    gearbox: bool = False     # 15.11 «механіка» (Стефан) замість err_step: передачі за готовою інструкцією
    gb_down: float = 2.0      # ||g|| > 2× звичної → передача вниз
    gb_clutch: float = 5.0    # > 5× → зчеплення: цей тензор цього кроку не крутимо
    gb_up: float = 1.3        # спокійно = < 1.3× звичної
    gb_calm: int = 3          # (не вживається з 15.11)
    gb_acc: float = 0.05      # «потужність мотора»: на скільки множник кроку змінюється за крок
    rb_fish_n: int = 0       # 15.11 (Стефан): копія лише активних фішок кроку — загальна кількість фішок
    rb_fish_k: int = 0        # скільки фішок активні за крок (зазвичай 2); 0 = копія всієї моделі


# ------------------------------------------------------------------------------------------------- розклад lr
def lr_at(step, lr, steps, *, warmup=200, sched="wsd", wsd_frac=0.2, floor=0.1):
    """fastloop.lr_at (step 0-based, python float64). sched="wsd" — LR_SCHED=wsd, інакше косинус до 0.1."""
    up = min(1, (step + 1) / warmup)
    if sched == "wsd":
        d0 = (1 - wsd_frac) * steps
        return lr * up * (1.0 if step < d0 else 1 - (1 - floor) * min(1, (step - d0) / (steps - d0)))
    return lr * up * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, step / steps))))


def lr_scalars(lr, cfg):
    """(lr_t, lr_v, lr_ratio) як f32: рахуємо в float64 на хості, далі f32 (як _lr_sync/_ratio_sync у torch)."""
    if isinstance(lr, (int, float)):
        return (jnp.float32(lr * cfg.scale), jnp.float32(lr * cfg.scale * cfg.vocab_lr_mul),
                jnp.float32(lr / cfg.lr_peak))
    lr = jnp.asarray(lr, jnp.float32)
    return lr * cfg.scale, lr * (cfg.scale * cfg.vocab_lr_mul), lr / cfg.lr_peak


# ------------------------------------------------------------------------------------------------- spec / шляхи
def path_str(path):
    out = []
    for k in path:
        out.append(str(getattr(k, "key", getattr(k, "idx", getattr(k, "name", k)))))
    return ".".join(out)


_VOCAB_NAMES = ("emb", "head", "l_head1", "book")


def default_kind(name, shape, sharded=None):
    """Вид кроку за іменем (design §3). sharded(name) → True для light-лінійних (hsS)."""
    parts = name.split(".")
    nd = len(shape)
    if nd == 2 and any(p in _VOCAB_NAMES for p in parts):
        return "vocab"
    if nd == 3:
        return "hs3"
    if nd == 2:
        return "hsS" if (sharded is not None and sharded(name)) else "hs"
    if nd >= 4:
        return "row"
    return "tensor"


def make_spec(params, kinds=None, order=None, sharded=None, has_grad=None, stacked=("heavy",), frozen=()):
    """Статичний опис кроку: кортеж (name, kind, wd_flag, vocab, has_grad, leaf, li) у порядку `order` (порядок torch
    plist: від нього залежать номери rel/gain і послідовність насінь SR). kinds — словник name → kind (перевизначення).
    stacked — верхні ключі, чиї листки складені по осі шару (heavy: (L, ...)): кожен шар — окрема вага, як у torch
    (ім'я "heavy.{li}.{k}", вид за формою БЕЗ осі шару, свій rel/gain, свої норми). leaf/li — листок і номер шару
    (li=None — звичайний листок).
    frozen — імена torch-параметрів БЕЗ листка в params (lock_w, ind_a: requires_grad=True у torch, але grad None) —
    вони займають місце в plist torch, тож входять у P (rel/gain/quantile «винних» і номери насінь); тут — заглушки
    з has_grad=False (leaf=None): не оновлюються, rel лишається 0, як у torch."""
    flat = {path_str(p): x for p, x in jax.tree_util.tree_flatten_with_path(params)[0]}
    ent = {}
    for n, x in flat.items():
        top, _, rest = n.partition(".")
        if top in (stacked or ()) and rest:
            for li in range(x.shape[0]):
                ent[f"{top}.{li}.{rest}"] = (n, li, x.shape[1:])
        else:
            ent[n] = (n, None, x.shape)
    for n in frozen:
        ent[n] = (None, None, ())
    names = list(order) if order is not None else list(ent)
    assert set(names) == set(ent), "order має містити всі листки"
    spec = []
    for n in names:
        leaf, li, shp = ent[n]
        if leaf is None:                                         # заглушка frozen
            spec.append((n, "tensor", False, False, False, None, None))
            continue
        kind = (kinds or {}).get(n) or default_kind(n, shp, sharded)
        vocab = kind == "vocab"
        wd = len(shp) >= 2 and not vocab
        hg = True if has_grad is None else bool(has_grad.get(n, has_grad.get(leaf, True)))
        spec.append((n, kind, wd, vocab, hg, leaf, li))
    return tuple(spec)


# ------------------------------------------------------------------------------------------------- стан
# --opt_gearbox у Pallas: ||g||² — з проходу А двопрохідних ядер; GB_KERN=0 — старий окремий прохід (для порівняння)
GB_KERN = os.environ.get("GB_KERN", "1") != "0"
GB_LATE = lambda leaf: str(leaf).startswith("fish.")
# 15.11 швидкість: ||g||² усіх тензорів — ОДНИМ проходом до циклу ваг (одна злита редукція + одне all-reduce вектора
# замість psum усередині ядра на КОЖЕН тензор/шар — бар'єр між чипами, що серіалізує оптимізатор), коробка
# передач — векторно над (P,) (замість 3 dynamic-update-slice на тензор). Правила ті самі. GB_PRE=0 — старий шлях.
GB_PRE = os.environ.get("GB_PRE", "1") != "0"
GB_PRE_LATE = os.environ.get("GB_PRE_LATE", "1") != "0"   # 15.11: фішки — ||g||² у проході оновлення (не в передпроході)
# 15.11 (fable52f): GB_TOP — верхня передача (за задумом 1.25); GB_TOP=1.0 — без перевищення ліміту (для порівняння:
# у спокійному режимі 47 з 49 тензорів за ~12 кроків виходять на 1.25, тобто весь крок моделі ×1.25)
GEARS = (0.25, 0.5, 1.0, float(os.environ.get("GB_TOP", "1.25")))   # 15.11 передачі Instinct; 1.0 = ліміт, 1.25 = водій сам вирішив перевищити; старт 2


def rb_is_fish(key, x, cfg):
    """Тензор фішок з провідною віссю NF (відкатник тримає лише активні рядки)."""
    nf = int(getattr(cfg, "rb_fish_n", 0))
    return (nf > 0 and int(getattr(cfg, "rb_fish_k", 0)) > 0 and str(key).startswith("fish.")
            and x.ndim >= 1 and x.shape[0] == nf)


def rb_cut(key, x, cfg, ids=None):
    """Копія для відкатника: фішки — лише рядки ids (k штук), решта — повністю."""
    if not rb_is_fish(key, x, cfg):
        return x
    if ids is None:
        return x[:int(cfg.rb_fish_k)]
    return rb_rows(x, ids)


def rb_rows(x, ids):
    """Рядки ids (k,) провідної осі: k×dynamic_slice (без gather/fill-маски на весь тензор)."""
    return jnp.concatenate([jax.lax.dynamic_slice_in_dim(x, ids[j], 1, 0) for j in range(ids.shape[0])], 0)


def rb_set_rows(x, ids, rows):
    """x з рядками ids ← rows: k×dynamic_update_slice (на місці, без scatter); порядок j — останній перемагає."""
    for j in range(ids.shape[0]):
        x = jax.lax.dynamic_update_slice_in_dim(x, rows[j:j + 1].astype(x.dtype), ids[j], 0)
    return x


def rb_paste(key, x, saved, cfg, ids):
    """Відкат: фішки — повернути рядки ids, решта — замінити повністю."""
    if not rb_is_fish(key, x, cfg):
        return saved.astype(x.dtype)
    return x.at[ids].set(saved.astype(x.dtype))


def init(params, d_sit, cfg, spec):
    """Порожній стан (як InstinctOptB.__init__ + ліниві keys/cs). m і cs — словники name → масив."""
    flat = {path_str(p): x for p, x in jax.tree_util.tree_flatten_with_path(params)[0]}
    P, K = len(spec), cfg.cards
    mdt = jnp.bfloat16 if cfg.m_bf16 else jnp.float32
    m, cs = {}, {}
    for n, kind, _, _, _, leaf, li in spec:
        if leaf is None:                                         # заглушка frozen (make_spec(frozen=...))
            continue
        x = flat[leaf]
        if leaf not in m:
            m[leaf] = jnp.zeros(x.shape, mdt)
        pre = x.shape[:1] if li is not None else ()             # складений листок: cs теж зі віссю шару
        shp = x.shape[1:] if li is not None else x.shape
        if cfg.hs_lag and kind == "hs":
            cs[leaf] = jnp.zeros(pre + (1, shp[1]), jnp.float32)
        elif cfg.hs_lag and kind == "hs3":
            cs[leaf] = jnp.zeros(pre + (shp[0], shp[1], 1), jnp.float32)
    return dict(
        gain=jnp.ones((K, P), jnp.float32), keys=jnp.zeros((K, d_sit), jnp.float32),
        used=jnp.zeros((K,), bool), expect=jnp.zeros((K,), jnp.float32), seen=jnp.zeros((K,), jnp.float32),
        rel=jnp.zeros((P,), jnp.float32), seed_t=jnp.zeros((), jnp.int32), n_step=jnp.zeros((), jnp.int32),
        k=jnp.zeros((), jnp.int32), s=jnp.zeros((), jnp.float32), stake=jnp.zeros((), jnp.float32),
        g=jnp.ones((P,), jnp.float32), oh=jnp.zeros((K,), jnp.float32), m=m, cs=cs,
        **({"rel_avg": jnp.zeros((P,), jnp.float32)} if cfg.culprit == "ema" else {}),
        **({"snr_v": jnp.zeros((P,), jnp.float32), "snr_f": jnp.ones((P,), jnp.float32)} if cfg.snr else {}),
        **({"dv": jnp.zeros((K, P), jnp.float32), "df": jnp.ones((P,), jnp.float32)}
           if getattr(cfg, "damp", 0.0) > 0 else {}),
        **({"gidx": jnp.full((P,), 2, jnp.int32), "gcalm": jnp.zeros((P,), jnp.int32)}
           if (getattr(cfg, "gearbox", False) and getattr(cfg, "damp", 0.0) > 0) else {}),
        **({"U": jnp.zeros((d_sit,), jnp.float32)} if (getattr(cfg, "stream_proj", False)
                                                       or getattr(cfg, "m_rate", False)) else {}),
        **(m_rate_init(params, d_sit) if getattr(cfg, "m_rate", False) else {}),
        **({"rb_p": {k: rb_cut(k, jnp.asarray(v), cfg) for k, v in flat.items()},
            "rb_m": {k: rb_cut(k, jnp.zeros_like(v), cfg) for k, v in m.items()},
            "rb_ids": jnp.zeros((max(int(getattr(cfg, "rb_fish_k", 0)), 1),), jnp.int32),
            "rb_lema": jnp.zeros((K,), jnp.float32), "rb_n": jnp.zeros((), jnp.int32),
            # 15.11 (fable52f): rb_cool[k] — попередній крок картки-виду k був відхилений відкатником; такий вид
            # наступного разу вчиться обов'язково (без відкату), інакше вид з піднятою втратою не вчиться зовсім
            "rb_cool": jnp.zeros((K,), bool),
            # 15.11 варіант А: rb_rep — наступний крок повторює збережений батч rb_b (кладе dist.make_step);
            # rb_src — вид кроку, що зберіг батч (0 текст, 1 синтетика); rb_valid — rb_p/rb_b від попереднього кроку
            "rb_rep": jnp.zeros((), bool), "rb_src": jnp.zeros((), jnp.int32), "rb_valid": jnp.zeros((), bool)}
           if getattr(cfg, "rollback", False) else {}))


# ------------------------------------------------------------------------------------------------- K1 / K2 (13.11)
STREAM_W = ("light_a.", "light_b.", "final.")
STREAM_KEYS = (".o", ".down", ".qkv")
NORM_KEYS = (".n1", ".n2")


def _proj_off(x, U):
    """Прибрати з x складову вздовж U по першій осі розміру len(U) (як r7_combo._proj)."""
    d = U.shape[0]
    for ax in range(x.ndim):
        if x.shape[ax] == d:
            xf = jnp.moveaxis(x.astype(jnp.float32), ax, -1)
            xf = xf - (xf @ U)[..., None] * U
            return jnp.moveaxis(xf, -1, ax).astype(x.dtype)
    return x


def stream_project(grads, st, stream_m, update_u=True, beta_u=0.9, project=True):
    """K1: U ← норм(β·U + (1−β)·норм(stream_m)) (перший крок — норм(stream_m)); з градієнтів і інерції матриць
    STREAM_W×STREAM_KEYS прибирається складова вздовж U. → (grads, st)."""
    U0 = st["U"]
    mv = stream_m.astype(jnp.float32)
    mvn = mv / jnp.maximum(jnp.linalg.norm(mv), 1e-8)
    n0 = jnp.linalg.norm(U0)
    if update_u:
        U = jnp.where(n0 > 0, beta_u * U0 + (1 - beta_u) * mvn, mvn)
    else:
        U = jnp.where(n0 > 0, U0, mvn)
    U = U / jnp.maximum(jnp.linalg.norm(U), 1e-8)
    if not project:                                    # fix15 без K1: лише стан U (EMA-напрям m)
        return grads, dict(st, U=U)

    def hit(name, x):
        return x is not None and x.ndim >= 2 and name.startswith(STREAM_W) and name.endswith(STREAM_KEYS)
    grads = jax.tree_util.tree_map_with_path(
        lambda p, g: _proj_off(g, U) if hit(path_str(p), g) else g, grads, is_leaf=lambda v: v is None)
    m = {k: (_proj_off(v, U) if hit(k, v) else v) for k, v in st["m"].items()}
    return grads, dict(st, m=m, U=U)


def norm_anchor(params, lr, cfg):
    """K2: 1D norm-ваги light_a/light_b/final (.n1/.n2) → w − lr·wd·(w−1)."""
    def f(p, w):
        n = path_str(p)
        if w.ndim == 1 and n.startswith(STREAM_W) and n.endswith(NORM_KEYS):
            w32 = w.astype(jnp.float32)
            return (w32 - lr * cfg.weight_decay * (w32 - 1.0)).astype(w.dtype)
        return w
    return jax.tree_util.tree_map_with_path(f, params)


def _uaxis(x, d):
    for ax in range(x.ndim):
        if x.shape[ax] == d:
            return ax
    return None


def _mrl_hit(name, x, d):
    return x.ndim >= 2 and name.startswith(STREAM_W) and name.endswith(STREAM_KEYS) and _uaxis(x, d) is not None


def m_rate_init(params, d):
    """Стан fix15: банк і когерентна швидкість (форма тензора без осі d), base (скаляр/тензор), лічильник кроків."""
    flat = {path_str(p): x for p, x in jax.tree_util.tree_flatten_with_path(params)[0]}
    bank, v, base = {}, {}, {}
    for k, x in flat.items():
        if _mrl_hit(k, x, d):
            ax = _uaxis(x, d)
            shp = x.shape[:ax] + x.shape[ax + 1:]
            bank[k] = jnp.zeros(shp, jnp.float32)
            v[k] = jnp.zeros(shp, jnp.float32)
            base[k] = jnp.zeros((), jnp.float32)
    return dict(mrl_bank=bank, mrl_v=v, mrl_base=base, mrl_n=jnp.zeros((), jnp.int32))


def m_rate(p_old, p_new, st, cfg, beta_v=0.9, beta_b=0.998, grads=None):
    """fix15: обмеження швидкості виходу m. c = ΔW·U (U-складова кроку тензора потоку = скільки m він дописує за крок).
    e* = c + bank; v' = β_v·v + (1−β_v)·e*; |v'| > τ=κ·base → v' ← v'·τ/|v'|; e = (v' − β_v·v)/(1−β_v) — випущено;
    bank ← e* − e (нічого не викидається). Замінюється лише U-складова: W ← W_new + (e − c)⊗U.
    → (params, st, info{bank, cut})."""
    U = st["U"].astype(jnp.float32)
    d = U.shape[0]
    n = st["mrl_n"]
    kappa = cfg.m_rate_kappa
    po = {path_str(p): x for p, x in jax.tree_util.tree_flatten_with_path(p_old)[0]}
    bank_o, v_o, base_o = dict(st["mrl_bank"]), dict(st["mrl_v"]), dict(st["mrl_base"])
    bsum, cut, drop = jnp.zeros((), jnp.float32), jnp.zeros((), jnp.float32), jnp.zeros((), jnp.float32)
    warm = n < cfg.m_rate_warm
    gate = bool(getattr(cfg, "m_rate_gate", False)) and grads is not None
    gf = {path_str(p): x for p, x in jax.tree_util.tree_flatten_with_path(grads)[0]} if gate else {}

    def f(path, w1):
        nonlocal bsum, cut, drop
        k = path_str(path)
        if k not in bank_o:
            return w1
        ax = _uaxis(w1, d)
        w1f = jnp.moveaxis(w1.astype(jnp.float32), ax, -1)
        dW = w1f - jnp.moveaxis(po[k].astype(jnp.float32), ax, -1)
        c = dW @ U
        bank, v, base = bank_o[k], v_o[k], base_o[k]
        if gate and k in gf:                               # «дивись під ноги»: відкладене лише там, куди модель ще хоче
            gU = jnp.moveaxis(gf[k].astype(jnp.float32), ax, -1) @ U
            keep = bank * gU < 0
            drop = drop + jnp.linalg.norm(jnp.where(keep, 0.0, bank))
            bank = jnp.where(keep, bank, 0.0)
        e_star = c + bank
        v_p = beta_v * v + (1 - beta_v) * e_star
        nv = jnp.linalg.norm(v_p)
        tau = jnp.where(warm, jnp.inf, kappa * base)
        s = jnp.minimum(1.0, tau / jnp.maximum(nv, 1e-30))
        v_n = v_p * s
        e = jnp.where(s < 1.0, (v_n - beta_v * v) / (1 - beta_v), e_star)
        bank_n = e_star - e
        nvn = jnp.linalg.norm(v_n)                        # розігрів: швидка EMA_0.95 (вивчити рівень), далі повільна,
        base_o[k] = jnp.where(warm, jnp.where(n == 0, nvn, 0.95 * base + 0.05 * nvn),   # вхід ≤ κ·base (без храповика)
                              beta_b * base + (1 - beta_b) * jnp.minimum(nvn, base))   # без храповика: межа під тиском не росте
        bank_o[k], v_o[k] = bank_n, v_n
        bsum = bsum + jnp.linalg.norm(bank_n)
        cut = cut + (s < 1.0).astype(jnp.float32)
        dlt = e - c
        out = w1f + dlt[..., None] * U                    # у розігріві dlt ≡ 0 точно → ваги бітово ті самі
        return jnp.moveaxis(out, -1, ax).astype(w1.dtype)

    params = jax.tree_util.tree_map_with_path(f, p_new)
    st = dict(st, mrl_bank=bank_o, mrl_v=v_o, mrl_base=base_o, mrl_n=n + 1)
    return params, st, dict(mrl_bank=bsum, mrl_cut=cut, mrl_drop=drop)


# ------------------------------------------------------------------------------------------------- prepare / finish
def _normalize(x):
    return x / jnp.maximum(jnp.sqrt(jnp.sum(x * x)), 1e-12)


def prepare(st, loss, situation, lr_ratio, cfg, card=None):
    """InstinctOptB.prepare (_prepare_impl): картка ситуації, очікування, несподіванка s, ставка, підсилювачі g.
    card (int32 скаляр, лише з cfg.card_by_kind) — картку задає вид кроку, а не схожість входу."""
    keys, used, expect, seen, gain = st["keys"], st["used"], st["expect"], st["seen"], st["gain"]
    K = keys.shape[0]
    sit = _normalize(jnp.asarray(situation, jnp.float32).reshape(-1))
    if cfg.card_by_kind and card is not None:
        k = jnp.clip(jnp.asarray(card, jnp.int32).reshape(()), 0, K - 1)
    else:
        any_used = jnp.any(used)
        sims = jnp.where(used, keys @ sit, -2.0)
        best = jnp.argmax(sims).astype(jnp.int32)            # перший індекс при рівних — як torch.max
        best_sim = sims[best]
        free = ~used
        first_free = jnp.argmax(free.astype(jnp.int32)).astype(jnp.int32)
        newer = (best_sim < cfg.new_card) & jnp.any(free)
        k = jnp.where(~any_used, 0, jnp.where(newer, first_free, best)).astype(jnp.int32)
    oh = jax.nn.one_hot(k, K, dtype=jnp.float32)
    was = jnp.sum(oh * used) > 0
    key_k = oh @ keys
    new_key = jnp.where(was, _normalize(0.95 * key_k + 0.05 * sit), sit)
    keys = keys + oh[:, None] * (new_key - key_k)[None]
    used = used | (oh > 0)
    loss = jnp.asarray(loss, jnp.float32)
    e = jnp.where(jnp.sum(oh * seen) > 0, jnp.sum(oh * expect), loss)
    s = (e - loss) / jnp.maximum(e, 1e-6)
    expect = expect + oh * (cfg.ema * e + (1 - cfg.ema) * loss - expect)
    seen = seen + oh * (1 - seen)
    stake = jnp.minimum(4 * jnp.abs(s) + 0.5 * lr_ratio, 1.0)
    g = oh @ gain
    return dict(st, keys=keys, used=used, expect=expect, seen=seen, s=s, stake=stake, k=k, g=g, oh=oh)


def finish(st, cfg):
    """InstinctOptB.finish (_finish_impl): n_step += 1; підсилювачі картки k вчаться за наслідком кроку."""
    s, gk, rel = st["s"], st["g"], st["rel"]
    extra = {}
    if cfg.culprit == "ema" and "rel_avg" in st:
        ra = st["rel_avg"]
        ra0 = jnp.where(ra > 0, ra, rel)                        # перший раз — без «несподіванки»
        ratio = rel / (ra0 + 1e-12)
        culprit = (ratio >= jnp.quantile(ratio, 1 - cfg.top)).astype(jnp.float32)
        fresh = (st["n_step"] % cfg.rel_every) == 0             # rel перераховано саме на цьому кроці
        extra["rel_avg"] = jnp.where(fresh, 0.9 * ra0 + 0.1 * rel, ra0)
    else:
        culprit = (rel >= jnp.quantile(rel, 1 - cfg.top)).astype(jnp.float32)
    gk = gk + jnp.where(s < 0, culprit * cfg.bad * jnp.minimum(-s * 10, 1), 0.0)
    gk = gk + jnp.where(s > 0, cfg.good * jnp.minimum(s * 10, 1) * (1 - gk), 0.0)
    gk = gk + cfg.relax * (1 - gk)
    gk = jnp.clip(gk, 0.5, cfg.g_max)
    gain = st["gain"] + st["oh"][:, None] * (gk / jnp.mean(gk) - st["gain"])
    return dict(st, gain=gain, n_step=st["n_step"] + 1, **extra)


# ------------------------------------------------------------------------------------------------- округлення
def _hash_u32(x):
    x = x ^ (x >> 16)
    x = x * jnp.uint32(0x7FEB352D)
    x = x ^ (x >> 15)
    x = x * jnp.uint32(0x846CA68B)
    x = x ^ (x >> 16)
    return x


def _sr_with(x, r16):
    xi = jax.lax.bitcast_convert_type(x, jnp.uint32)
    return jax.lax.bitcast_convert_type((xi + r16) & jnp.uint32(0xFFFF0000), jnp.float32).astype(jnp.bfloat16)


def _store(m32, w32, m_dtype, w_dtype, seed_u32, sr, row_off=None):
    """Запис інерції й ваги: bf16 — SR (той самий хеш, що в Pallas: молодші 16 біт → m, старші → w) або до
    найближчого; fp32 — без змін. row_off (k,) — рядки ids провідної осі повного тензора (фішки, лише активні):
    індекс елемента = ids·(розмір рядка) + локальний, mod 2³² — рівно той, що мав би повний тензор (iota u32)."""
    need = sr and (m_dtype == jnp.bfloat16 or w_dtype == jnp.bfloat16)
    if need and row_off is not None:
        per = math.prod(m32.shape[1:])
        loc = jnp.arange(per, dtype=jnp.uint32).reshape((1,) + m32.shape[1:])
        off = row_off.astype(jnp.uint32) * jnp.uint32(per % (1 << 32))
        idx = loc + off.reshape((-1,) + (1,) * (m32.ndim - 1))
        h = _hash_u32(idx ^ (seed_u32 * jnp.uint32(0x9E3779B9)))
    elif need:
        idx = jnp.arange(m32.size, dtype=jnp.uint32).reshape(m32.shape)
        h = _hash_u32(idx ^ (seed_u32 * jnp.uint32(0x9E3779B9)))
    m = (_sr_with(m32, h & jnp.uint32(0xFFFF)) if (need and m_dtype == jnp.bfloat16) else m32.astype(m_dtype))
    w = (_sr_with(w32, h >> 16) if (need and w_dtype == jnp.bfloat16) else w32.astype(w_dtype))
    return m, w


def _rms(x, axis):
    return jnp.sqrt(jnp.mean(x * x, axis=axis, keepdims=True))


# ------------------------------------------------------------------------------------------------- крок однієї ваги (jnp)
def _rms_gg(x, g, axis):
    """(_rms(x, axis), Σg²) одним variadic-reduce: ||g||² коробки передач у тому ж проході, що й норма рядків."""
    ax = (axis,) if isinstance(axis, int) else tuple(axis)
    ax = tuple(a % x.ndim for a in ax)
    s1, s2 = jax.lax.reduce((x * x, g * g), (jnp.float32(0), jnp.float32(0)),
                            lambda a, b: (a[0] + b[0], a[1] + b[1]), ax)
    n = math.prod(x.shape[a] for a in ax)
    shp = tuple(1 if a in ax else d for a, d in enumerate(x.shape))
    return jnp.sqrt((s1 / jnp.float32(n)).reshape(shp)), jnp.sum(s2)


def _upd_jnp_gg(kind, mf, grad, cs, axis_name=None, rows_total=None):
    """_upd_jnp + ||g||² (fp32) для --opt_gearbox: норма градієнта — в одному проході з першою нормою рядків mf.
    → (upd, cs_new, gg)."""
    g = grad.astype(jnp.float32)
    if kind in ("hs", "hs3", "hsS"):
        ro = 1
        r, gg = _rms_gg(mf, g, ro)
        upd = mf / (r + EPS)
        if kind == "hsS":
            col = jnp.sum(upd * upd, axis=0, keepdims=True)
            if axis_name is not None:
                col = jax.lax.psum(col, axis_name)
            col = col / (rows_total or mf.shape[0])
            upd = upd / (jnp.sqrt(col) + EPS)
            return upd / (_rms(upd, 1) + EPS), None, gg
        co = 0 if kind == "hs" else 2
        now = jnp.mean(upd * upd, axis=co, keepdims=True)
        col = now if cs is None else jnp.where(cs > 0, cs, now)
        upd = upd / (jnp.sqrt(col) + EPS)
        upd = upd / (_rms(upd, ro) + EPS)
        return upd, (None if cs is None else now), gg
    if kind == "row" and mf.ndim >= 2:
        r, gg = _rms_gg(mf, g, tuple(range(1, mf.ndim)))
        return mf / (r + EPS), None, gg
    if kind not in ("vocab", "row"):
        r, gg = _rms_gg(mf, g, tuple(range(mf.ndim)))
        return mf / (r + EPS), None, gg
    upd, cs_new = _upd_jnp(kind, mf, grad, cs, axis_name, rows_total)
    return upd, cs_new, jnp.sum(g * g)


def _upd_jnp(kind, mf, grad, cs, axis_name=None, rows_total=None):
    """Напрям кроку (математика _instinct_step_impl). → (upd, cs_new або None)."""
    if kind == "vocab":
        hit = jnp.sum(jnp.abs(grad), axis=1, keepdims=True) > 0
        return jnp.where(hit, mf / (_rms(mf, 1) + EPS), 0.0), None
    if kind in ("hs", "hs3"):
        ro, co = (1, 0) if kind == "hs" else (1, 2)
        upd = mf / (_rms(mf, ro) + EPS)
        now = jnp.mean(upd * upd, axis=co, keepdims=True)
        col = now if cs is None else jnp.where(cs > 0, cs, now)
        upd = upd / (jnp.sqrt(col) + EPS)
        upd = upd / (_rms(upd, ro) + EPS)
        return upd, (None if cs is None else now)
    if kind == "hsS":
        # light-лінійні: свіжа норма стовпця (сума по всіх чипах / усього рядків); на W=1 — середнє по рядках
        upd = mf / (_rms(mf, 1) + EPS)
        col = jnp.sum(upd * upd, axis=0, keepdims=True)
        if axis_name is not None:
            col = jax.lax.psum(col, axis_name)
        col = col / (rows_total or mf.shape[0])
        upd = upd / (jnp.sqrt(col) + EPS)
        return upd / (_rms(upd, 1) + EPS), None
    if kind == "row":
        return mf / (jnp.sqrt(jnp.mean((mf * mf).reshape(mf.shape[0], -1), axis=1)).reshape(
            (-1,) + (1,) * (mf.ndim - 1)) + EPS), None
    return mf / (jnp.sqrt(jnp.mean(mf * mf)) + EPS), None


def _seed_bits(seed_t):
    """seed = int32(seed_t·1000003) (у torch int64 з обрізанням до int32 → ті самі молодші 32 біти), як u32."""
    return seed_t.astype(jnp.uint32) * jnp.uint32(1000003)


def _interp(cfg):
    if cfg.interpret is not None:
        return cfg.interpret
    return jax.default_backend() != "tpu"


def _pallas_call(kfn, args, sc, it, shard, leaf, li, gear=None, dst=None):
    """Виклик Pallas-ядра кроку. shard=(mesh, {leaf: PartitionSpec}) — для W>1 (dist.make_step).
    Pallas/Mosaic custom call GSPMD не розбиває: на шардованій вазі (vocab-рядки, EP-експерти) XLA зібрав би ПОВНИЙ
    тензор на кожен чип (24B → OOM) і рахував би його 8 разів. Тому шардовані листки йдуть через shard_map:
    ядро бачить свій шматок. Це точно, бо всі ядра локальні по шардованій осі (vocab — нормування рядка,
    hs3 — по кожному експерту окремо). Насіння SR змішуємо з номером чипа, щоб шматки не мали однакового шуму
    (індекс у хеші локальний). Репліковані листки — як є (кожен чип рахує те саме)."""
    if dst is not None:
        # 15.11 відкатник: dst=(hit, sm, sw) — m/w-виходи у буфер знімка (див. pallas_instinct._with_dst); shard_map —
        # ті самі розрізи, що в m/w, hit — P(). Gear разом із dst не йде (викликач перевіряє).
        assert gear is None
        _h, _sm, _sw = dst
        _kfn0 = kfn

        def kfn(*a, interpret=False, **kw):
            *xs, s_ = a
            return _kfn0(*xs[:-3], s_, interpret=interpret, dst=(xs[-1], xs[-3], xs[-2]), **kw)
        n_base = len(args)
        args = tuple(args) + (_sm, _sw, jnp.asarray(_h, jnp.float32).reshape(1))
    else:
        n_base = len(args)
    spec = None
    if shard is not None:
        mesh, specs = shard
        spec = specs.get(leaf)
        if spec is not None and all(a is None for a in spec):
            spec = None
    from jax.sharding import PartitionSpec as P
    if spec is None and shard is not None:
        # репліковані листки на W>1: Mosaic-виклик GSPMD не розбиває навіть реплікований («Mosaic kernels cannot be
        # automatically partitioned») — теж через shard_map, усі входи/виходи P() (кожен чип рахує те саме, SR без зсуву)
        n_out = 3 if (n_base == 4 or gear is not None) else 2
        if gear is not None:                                   # (gear: fn — статична, aux — ще один вхід P())
            gfn = gear[0]
            def local_rep(*a):
                *xs, s, ax_ = a
                return tuple(kfn(*xs, s, interpret=it, gear=(gfn, ax_)))
            args = tuple(args) + (sc,)
            sc = gear[1]
        else:
            def local_rep(*a):
                *xs, s = a
                return tuple(kfn(*xs, s, interpret=it))
        rep = tuple(P() for _ in range(len(args) + 1))
        try:
            sm = jax.shard_map(local_rep, mesh=shard[0], in_specs=rep, out_specs=tuple(P() for _ in range(n_out)),
                               check_vma=False)
        except AttributeError:
            from jax.experimental.shard_map import shard_map as _sm
            sm = _sm(local_rep, mesh=shard[0], in_specs=rep, out_specs=tuple(P() for _ in range(n_out)), check_rep=False)
        return sm(*args, sc)
    if spec is None:
        if gear is not None:
            return tuple(kfn(*args, sc, interpret=it, gear=gear))
        return tuple(kfn(*args, sc, interpret=it))
    parts = tuple(spec)
    if li is not None:                                         # шар складеного листка: перша вісь знята
        parts = parts[1:]
    ax = [a for a in parts if a is not None]
    assert len(ax) == 1, (leaf, spec)
    axis, dim = ax[0], parts.index(ax[0])
    def spec_for(x):
        return P(*([None] * dim + [axis] + [None] * (x.ndim - dim - 1)))
    in_specs = (tuple(spec_for(a) for a in args[:n_base])
                + ((spec_for(args[n_base]), spec_for(args[n_base + 1]), P()) if dst is not None else ())
                + (P(),) + ((P(),) if gear is not None else ()))
    n_out = 3 if n_base == 4 else 2
    out_specs = tuple(spec_for(args[j]) for j in ((1, 2, 3) if n_out == 3 else (1, 2)))
    if gear is not None:
        out_specs = out_specs + (P(),)                         # Σg² уже глобальна (psum у ядрі)

    def local(*a):
        if gear is not None:
            *xs, s, ax_ = a
        else:
            *xs, s = a
        r = jax.lax.axis_index(axis).astype(jnp.uint32)
        sb = jax.lax.bitcast_convert_type(s[3], jnp.uint32)
        sb = _hash_u32(sb ^ ((r + jnp.uint32(1)) * jnp.uint32(0x85EBCA6B)))
        s = s.at[3].set(jax.lax.bitcast_convert_type(sb, jnp.float32))
        if gear is not None:
            return tuple(kfn(*xs, s, interpret=it, gear=(gear[0], ax_), axis_name=axis))
        return tuple(kfn(*xs, s, interpret=it))
    try:
        sm = jax.shard_map(local, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
    except AttributeError:
        from jax.experimental.shard_map import shard_map as _sm
        sm = _sm(local, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)
    return sm(*args, sc, gear[1]) if gear is not None else sm(*args, sc)


def _gear_core(_gg, _dv, _v, _gi, _s, cfg):
    """15.11 коробка передач одного тензора: (||g||², звична dv, швидкість v, передача gi, s) → (cf, gin, dvn).
    Спільна для jnp-шляху й Pallas-ядер (там — усередині обгортки ядра, між проходами А і Б)."""
    _r = jnp.where((_dv > 0) & (_gg > 0), jnp.sqrt(_gg / jnp.maximum(_dv, 1e-38)), 1.0)
    _G = jnp.asarray(GEARS, jnp.float32)
    # 15.11 Стефан (водій): передачі лише по одній; сплеск > gb_down → вниз; спокійно → вгору (до 1.25 —
    # водій сам вирішує перевищити); «камера»: їде вище ліміту 1.0 і loss гірший за очікуваний (s<0) → штраф,
    # вниз. Швидкість (множник кроку) не стрибає: щокроку підтягується до передачі на ≤ gb_acc (розгін 0→100).
    _fine = (_v > 1.0) & (_s.astype(jnp.float32) < 0)
    _up = (_r < cfg.gb_up) & ~_fine & (_v >= _G[_gi] - 1e-3)   # вгору — лише коли розігнався до поточної
    _dn = (_r > cfg.gb_down) | _fine
    _gin = jnp.clip(_gi + jnp.where(_dn, -1, jnp.where(_up, 1, 0)), 0, len(GEARS) - 1).astype(_gi.dtype)
    _vt = _G[_gin]
    _cf = (_v + jnp.clip(_vt - _v, -cfg.gb_acc, cfg.gb_acc)).astype(jnp.float32)
    _cf = jnp.where(_gg > 0, _cf, _v)
    _gin = jnp.where(_gg > 0, _gin, _gi)
    _in = jnp.minimum(_gg, jnp.float32(cfg.damp_cap * cfg.damp_cap) * _dv)
    _dvn = jnp.where(_dv > 0, cfg.damp_beta * _dv + (1 - cfg.damp_beta) * _in, _gg)
    _dvn = jnp.where(_gg > 0, _dvn, _dv)
    return _cf, _gin, _dvn


def _gear_cf(gg, aux, cfg):
    """Множник передачі для ядра: aux = f32[5] (dv, v, gi, s, множник після передачі — snr)."""
    return _gear_core(gg, aux[0], aux[1], aux[2].astype(jnp.int32), aux[3], cfg)[0]


def _layer_run(active, j):
    """Чи починається з active[j] блок складених шарів: K різних листків × L шарів (li = 0..L−1, li-major,
    кожен шар — ті самі листки в тому самому порядку й видах). → (L, K) або None (тоді — по одній вазі)."""
    e0 = active[j]
    if e0[6] != 0:
        return None
    K = 0
    while j + K < len(active) and active[j + K][6] == 0:
        K += 1
    leaves = [e[5] for e in active[j:j + K]]
    if len(set(leaves)) != K:
        return None
    L = 1
    while True:
        nxt = active[j + L * K:j + (L + 1) * K]
        if len(nxt) != K or any(e[6] != L or e[5] != lf or e[1:4] != b[1:4] for e, lf, b in zip(nxt, leaves, active[j:j + K])):
            break
        L += 1
    return (L, K) if L > 1 else None


def fish_fast_ok(leaf, p, csi, cfg, fish_ids):
    """15.11 швидкість: фішка fish.up/fish.down (NF, ., .) оновлюється лише в рядках активних фішок кроку fish_ids
    (решта NF−k — простій, _freeze_idle однаково повертає їх як були). Математика кожної фішки окрема (hs3: норми
    по осях 1/2 всередині фішки), SR — ті самі індекси, rel — повні норми. Інакше (gnorm, safe16, snr на фішках,
    cs не по фішках) — старий шлях по всьому стеку."""
    if fish_ids is None or str(leaf) not in ("fish.up", "fish.down") or p.ndim != 3:
        return False
    if FISH_FAST == 0 or cfg.gnorm or getattr(cfg, "safe16", False):
        return False
    if getattr(cfg, "snr", False) and not str(leaf).startswith(tuple(cfg.snr_skip)):
        return False
    if csi is not None and csi.shape[:1] != p.shape[:1]:
        return False
    return fish_ids.shape[0] < p.shape[0]


FISH_FAST = int(os.environ.get("FISH_FAST", "1"))


def update(params, grads, st, lr_t, lr_v, cfg, spec, axis_name=None, rows_total=None, shard=None, rb_dst=None,
           fish_ids=None, fish_done=None):
    """Крок усіх ваг (update_all) у фіксованому порядку spec. Ваги без градієнта (has_grad=False або grad None) —
    пропускаються повністю (без згасання m, без wd, rel не пишеться). → (params, st)."""
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    pflat = {path_str(p): x for p, x in leaves}
    gflat = {path_str(p): x for p, x in jax.tree_util.tree_flatten_with_path(grads, is_leaf=lambda v: v is None)[0]}
    m, cs = dict(st["m"]), dict(st["cs"])
    rel, seed_t, g_all, stake = st["rel"], st["seed_t"], st["g"], st["stake"]
    use_snr = bool(getattr(cfg, "snr", False)) and "snr_v" in st
    sv = (st["snr_v"], st["snr_f"]) if use_snr else (jnp.zeros((1,), jnp.float32), jnp.ones((1,), jnp.float32))
    use_damp = getattr(cfg, "damp", 0.0) > 0 and "dv" in st
    if use_damp:                                            # fix21: рядок типових норм картки цього кроку + множники
        sv = sv + (st["dv"][st["k"]], st["df"] if "gidx" in st else jnp.ones_like(st["df"]))
        if "gidx" in st:                                    # 15.11 коробка передач: стан між кроками
            sv = sv + (st["gidx"], st["gcalm"])
    gb_pre = None
    gb_late_leaves = set()
    if use_damp and getattr(cfg, "gearbox", False) and "gidx" in st and GB_PRE:
        _P = len(spec)
        _gflat0 = {path_str(pp): x for pp, x in jax.tree_util.tree_flatten_with_path(
            grads, is_leaf=lambda v: v is None)[0]}
        _sums, _idx = {}, []
        _vals = []
        for _j, (_n, _k, _w, _v, _hg, _leaf, _li) in enumerate(spec):
            _g = _gflat0.get(_leaf) if _leaf is not None else None
            if not _hg or _g is None:
                continue
            if GB_PRE_LATE and GB_LATE(_leaf) and _li is None:
                # 15.11 швидкість: фішки (jnp-шлях) — ||g||² у тому ж проході, що й їхнє оновлення (gear_late), а не
                # окремим читанням усіх (NF, ...) градієнтів тут. Стан коробки цих тензорів передпрохід не змінює
                # (gg=0 → _gear_core віддає стан як був), правила ті самі.
                gb_late_leaves.add(_leaf)
                continue
            if _leaf not in _sums:
                _g32 = _g.astype(jnp.float32)
                _ax = tuple(range(1, _g.ndim)) if _li is not None else None
                _sums[_leaf] = jnp.sum(_g32 * _g32, axis=_ax)
            _idx.append(_j)
            _vals.append(_sums[_leaf][_li] if _li is not None else _sums[_leaf])
        _gg = jnp.zeros((_P,), jnp.float32)
        if _idx:
            _gg = _gg.at[jnp.asarray(_idx, jnp.int32)].set(jnp.stack(_vals).astype(jnp.float32))
        _cfv, _ginv, _dvnv = _gear_core(_gg, sv[2], sv[3], sv[4], st["s"], cfg)
        gb_pre = _cfv
        sv = sv[:2] + (_dvnv, _cfv, _ginv.astype(sv[4].dtype), sv[5])
    want_rel = (st["n_step"] % cfg.rel_every) == 0
    seed_base = _seed_bits(seed_t)                          # для generic-видів (у torch там свій randint)
    new_p = dict(pflat)
    beta = jnp.float32(cfg.beta)

    def one(i, kind, wdf, vocab, leaf, li, p, mi, csi, grad, rel, seed_t, sv, row_off=None, rel_pg=None):
        """Крок однієї ваги (або одного шару складеного листка). i може бути traced (цикл по шарах).
        → (m_new, w_new, cs_new або None, rel, seed_t)."""
        lr = lr_v if vocab else lr_t
        g_ms = None
        if cfg.gnorm:
            g32 = grad.astype(jnp.float32)
            _ms = jnp.mean(g32 * g32)
            grad = g32 * jax.lax.rsqrt(_ms + 1e-30)
            if getattr(cfg, "gnorm_rawrel", False):
                g_ms = _ms                                     # fix14: rel від сирого градієнта
        _nrm = (getattr(cfg, "wd_norm", 0.0) > 0 and p.ndim == 1
                and (str(leaf) in ("norm", "l_norm") or str(leaf).endswith((".n1", ".n2"))))
        wd = (cfg.weight_decay / cfg.scale if (wdf and cfg.weight_decay)
              else (cfg.wd_norm / cfg.scale if _nrm else 0.0))
        eff = 1 + (g_all[i] - 1) * stake
        lr_eff = (lr * eff).astype(jnp.float32)
        if getattr(cfg, "surge_thr", 0.0) > 0:                # fix20: поганий батч не робить повного кроку
            _ns = jnp.maximum(-st["s"].astype(jnp.float32), 1e-6)
            lr_eff = (lr_eff * jnp.minimum(1.0, (jnp.float32(cfg.surge_thr) / _ns) ** 2)).astype(jnp.float32)
        for _pref, _mul in getattr(cfg, "lr_mul", ()):         # fix18: гальмо кроку обраних тензорів
            if str(leaf).startswith(_pref):
                lr_eff = (lr_eff * jnp.float32(_mul)).astype(jnp.float32)
                break
        safe = bool(getattr(cfg, "safe16", False)) and (str(leaf).startswith(tuple(cfg.safe16_prefix))
                                                         or str(leaf) in ("norm", "l_norm"))
        pallas_ok = (cfg.use_pallas and cfg.sr and p.dtype == jnp.bfloat16 and mi.dtype == jnp.bfloat16
                     and kind in ("hs", "hs3", "vocab") and not str(leaf).startswith("fish.") and not safe)
        gear_late = None

        def gear(_gg, sv):
            """15.11 коробка передач тензора i за ||g||² цього кроку → (множник кроку _cf, sv)."""
            _cf, _gin, _dvn = _gear_core(_gg, sv[2][i], sv[3][i], sv[4][i], st["s"], cfg)
            return _cf, sv[:2] + (sv[2].at[i].set(_dvn), sv[3].at[i].set(_cf), sv[4].at[i].set(_gin),
                                  sv[5].at[i].set(sv[5][i]))
        # ||g||² — з проходу А двопрохідних ядер (hs/hs3 без cs): без окремого проходу по g
        gear_kern = bool(use_damp and getattr(cfg, "gearbox", False) and pallas_ok and kind in ("hs", "hs3")
                         and csi is None and GB_KERN)
        snr_post = jnp.float32(1.0)
        if gb_pre is not None and leaf not in gb_late_leaves:  # множник уже пораховано (передпрохід), стан коробки — теж
            gear_kern = False
            lr_eff = (lr_eff * gb_pre[i]).astype(jnp.float32)
        elif use_damp and getattr(cfg, "gearbox", False) and not pallas_ok and not safe and GB_LATE(leaf):
            # швидкість: ||g||² — у тому ж проході, що й норми напряму (_upd_jnp_gg); множник — перед записом ваги
            gear_late = gear
        elif gear_kern:
            pass                                             # множник — у ядрі (_pallas_call(gear=...))
        elif use_damp and getattr(cfg, "gearbox", False):
            _g32 = grad.astype(jnp.float32)
            _cf, sv = gear(jnp.sum(_g32 * _g32), sv)
            lr_eff = (lr_eff * _cf).astype(jnp.float32)
        elif use_damp and getattr(cfg, "err_step", False):
            # err_step (задум Стефана): крок = lr · (loss/очікуване) · (звична ||g|| / ||g|| зараз), ≤ err_cap
            _g32 = grad.astype(jnp.float32)
            _gg = jnp.sum(_g32 * _g32)
            _dv = sv[2][i]
            # 15.11: множник похибки кроку (loss/expect) прибрано — на стенді роздував крок усім тензорам у провалі
            # (eval 5.40 проти 4.52); похибку тепер несуть правила на словах (inst_word). Лишається «÷ чутливість».
            _err = jnp.float32(1.0)
            _sens = jnp.where((_dv > 0) & (_gg > 0), jnp.sqrt(_dv / jnp.maximum(_gg, 1e-38)), 1.0)
            _cf = jnp.clip(_err * _sens, 0.0, jnp.float32(cfg.err_cap)).astype(jnp.float32)
            _cf = jnp.where(_gg > 0, _cf, 1.0)
            _in = jnp.minimum(_gg, jnp.float32(cfg.damp_cap * cfg.damp_cap) * _dv)
            _dvn = jnp.where(_dv > 0, cfg.damp_beta * _dv + (1 - cfg.damp_beta) * _in, _gg)
            _dvn = jnp.where(_gg > 0, _dvn, _dv)
            lr_eff = (lr_eff * _cf).astype(jnp.float32)
            sv = sv[:2] + (sv[2].at[i].set(_dvn), sv[3].at[i].set(_cf))
        elif use_damp and str(leaf).startswith(tuple(cfg.damp_only)):
            # fix21: ||g|| проти типової норми цього тензора на цій картці (до snr: він бачить уже обрізаний градієнт)
            _g32 = grad.astype(jnp.float32)
            _gg = jnp.sum(_g32 * _g32)
            _dv = sv[2][i]
            _lim = jnp.float32(cfg.damp * cfg.damp) * _dv                    # (k·типова)²
            _hot = (_dv > 0) & (_gg > _lim)
            _cf = jnp.where(_hot, jnp.sqrt(_lim / jnp.maximum(_gg, 1e-38)), 1.0).astype(jnp.float32)   # ≤ 1
            _in = jnp.minimum(_gg, jnp.float32(cfg.damp_cap * cfg.damp_cap) * _dv)
            _dvn = jnp.where(_dv > 0, cfg.damp_beta * _dv + (1 - cfg.damp_beta) * _in, _gg)
            _dvn = jnp.where(_gg > 0, _dvn, _dv)
            _gd = grad.dtype
            grad = jax.lax.cond(_hot, lambda: (_g32 * _cf).astype(_gd), lambda: grad)   # (1) в інерцію — обрізаний
            lr_eff = (lr_eff * _cf ** jnp.float32(cfg.damp_p)).astype(jnp.float32)      # (2) крок коротшає зі сплеском
            sv = sv[:2] + (sv[2].at[i].set(_dvn), sv[3].at[i].set(_cf))
        if use_snr and not str(leaf).startswith(tuple(cfg.snr_skip)) and (
                not cfg.snr_only or str(leaf).startswith(tuple(cfg.snr_only))):
            # впевненість: ||β·m + (1−β)·g|| / sqrt(EMA ||g||²) — три скалярні редукції по g і m (один прохід)
            g32, m32 = grad.astype(jnp.float32), mi.astype(jnp.float32)
            gg, mg, m2 = jnp.sum(g32 * g32), jnp.sum(m32 * g32), jnp.sum(m32 * m32)
            b_ = jnp.float32(cfg.beta)
            mn2 = b_ * b_ * m2 + 2 * b_ * (1 - b_) * mg + (1 - b_) * (1 - b_) * gg
            v_old = sv[0][i]
            if getattr(cfg, "snr_dz", False):                   # fix14: dead-zone (гальмує лише блукання)
                c2 = jnp.maximum(mn2, 0.0) / (gg + 1e-30)
                ok = (v_old > 0) & (v_old <= 1.0)               # старий стан (EMA ||g||² з ckpt) — не c̄², почати знову
                v_new = jnp.where(ok, cfg.snr_beta * v_old + (1 - cfg.snr_beta) * c2, c2)
                k_ = (1 - b_) / (1 + b_)
                r_hat = jnp.maximum(v_new - k_, 0.0) / jnp.maximum(1 - v_new, 1e-6)
                fac = jnp.clip(jnp.sqrt(r_hat * cfg.snr_horizon / k_), cfg.snr_floor, 1.0)
            else:
                v_new = jnp.where(v_old > 0, cfg.snr_beta * v_old + (1 - cfg.snr_beta) * gg, gg)
                c = jnp.sqrt(jnp.maximum(mn2, 0.0)) / (jnp.sqrt(v_new) + 1e-30)
                fac = jnp.clip(c / cfg.snr_ref, cfg.snr_floor, 1.0)
            fac = jnp.where(gg > 0, fac, 1.0)                    # без градієнта (простій) — нічого не міняємо
            v_new = jnp.where(gg > 0, v_new, v_old)
            if gear_kern:                                    # порядок як у jnp-шляху: (lr·передача)·snr — у ядрі
                snr_post = fac.astype(jnp.float32)
            else:
                lr_eff = (lr_eff * fac).astype(jnp.float32)
            sv = (sv[0].at[i].set(v_new), sv[1].at[i].set(fac)) + sv[2:]
        wdmul = (1 - lr * wd).astype(jnp.float32) if wd else jnp.float32(1.0)
        _rp, _rg = (p, grad) if rel_pg is None else rel_pg
        if g_ms is not None:   # fix14 gnorm_rawrel: ||g_raw|| = sqrt(N·mean(g_raw²)) — без зайвого проходу по g
            _n = float(math.prod(grad.shape))
            rel_fn = lambda: jnp.sqrt(g_ms * _n) / (jnp.linalg.norm(_rp.astype(jnp.float32)) + EPS)
        else:
            rel_fn = lambda: jnp.linalg.norm(_rg.astype(jnp.float32)) / (jnp.linalg.norm(_rp.astype(jnp.float32)) + EPS)
        if cfg.rel_cond:      # норми лише на кроках want_rel (звіт prof_b4_all п.7, ≈8–9 ms); значення ті самі
            rel = rel.at[i].set(jax.lax.cond(want_rel, rel_fn, lambda: rel[i].astype(jnp.float32)).astype(rel.dtype))
        else:                 # старий шлях: норми щокроку + where
            rel = rel.at[i].set(jnp.where(want_rel, rel_fn(), rel[i]))
        # фішки (fish.*) порізані між чипами ВСЕРЕДИНІ матриці (по d / HF): гомеостаз ядра нормує рядки й стовпці
        # лише свого шматка → не те саме, що глобально; чистий jnp-шлях (GSPMD) рахує норми по всій матриці — точно
        if kind in ("hs", "hs3", "vocab") and cfg.sr and p.dtype == jnp.bfloat16 and mi.dtype == jnp.bfloat16:
            seed_t = seed_t + 1                              # лічильник крокує на кожну вагу «ядерного» шляху
            seed = _seed_bits(seed_t)
        else:                                            # (0x5BD1E995·(i+1)) mod 2³² — у uint32 (i може бути traced)
            seed = _hash_u32(seed_base ^ (jnp.uint32(0x5BD1E995) * (jnp.asarray(i).astype(jnp.uint32) + jnp.uint32(1))))
        cs_new = None
        if pallas_ok:
            pi, ph = _kernels()
            sc = jnp.stack([beta, lr_eff, wdmul, jax.lax.bitcast_convert_type(seed, jnp.float32)])
            gb, it = grad.astype(jnp.bfloat16), _interp(cfg)
            if kind == "vocab":
                kfn, args = pi.instinct_vocab, (gb, mi, p)
            elif csi is not None:
                kfn = ph.instinct_hs1_2d if kind == "hs" else ph.instinct_hs1_3d
                args = (gb, mi, p, csi)
            else:
                kfn = pi.instinct_hs_2d if kind == "hs" else pi.instinct_hs3
                args = (gb, mi, p)
            _dst = None
            if (rb_dst is not None and not gear_kern and li is None and (kind == "vocab" or csi is not None)
                    and leaf in rb_dst["p"] and leaf in rb_dst["m"]
                    and rb_dst["p"][leaf].shape == p.shape and rb_dst["m"][leaf].shape == mi.shape):
                # 15.11 відкатник: вихід — у буфер знімка, вибір hit у ядрі (step._rb_finish цей листок не чіпає)
                _dst = (rb_dst["hit"], rb_dst["m"][leaf], rb_dst["p"][leaf])
                rb_dst["done"].add(leaf)
            if gear_kern:
                aux = jnp.stack([sv[2][i], sv[3][i], sv[4][i].astype(jnp.float32),
                                 st["s"].astype(jnp.float32), snr_post]).astype(jnp.float32)
                outs = _pallas_call(kfn, args, sc, it, shard, leaf, li,
                                    gear=(functools.partial(_gear_cf, cfg=cfg), aux))
                m_new, w_new = outs[0], outs[1]
                _, sv = gear(outs[2], sv)                    # той самий множник; тут — лише стан коробки
            else:
                outs = _pallas_call(kfn, args, sc, it, shard, leaf, li, dst=_dst)
                m_new, w_new = outs[0], outs[1]
                if len(outs) > 2:
                    cs_new = outs[2]
        else:
            gf = grad.astype(jnp.float32)
            mf = mi.astype(jnp.float32) * beta + gf * jnp.float32(1 - cfg.beta)   # (1−β) у float64, як скаляр torch
            lr_x = lr_eff
            if safe:                                     # fix16: напрям кроку без руху проти свіжого градієнта
                m32 = mi.astype(jnp.float32)
                ax = tuple(range(1, gf.ndim)) if gf.ndim >= 2 else None
                mg = jnp.sum(m32 * gf, axis=ax, keepdims=gf.ndim >= 2)
                gg = jnp.sum(gf * gf, axis=ax, keepdims=gf.ndim >= 2) + 1e-30
                mstep = (m32 - jnp.minimum(mg, 0.0) / gg * gf) * beta + gf * jnp.float32(1 - cfg.beta)
                mstep = jnp.where(mstep * gf >= 0, mstep, 0.0)
                upd, cs_new = _upd_jnp(kind, mstep, gf, csi, axis_name, rows_total)
                if p.ndim == 1:                          # 1D-норми: крок відносно |w| (канали-стоки ~0.004 не вибиваються)
                    w0 = p.astype(jnp.float32)
                    lr_x = lr_eff * jnp.minimum(jnp.abs(w0) / (jnp.sqrt(jnp.mean(w0 * w0)) + EPS), 2.0)
            elif gear_late is not None:
                upd, cs_new, _gg = _upd_jnp_gg(kind, mf, gf, csi, axis_name, rows_total)
                _cf, sv = gear_late(_gg, sv)
                lr_x = (lr_x * _cf).astype(jnp.float32)
            else:
                upd, cs_new = _upd_jnp(kind, mf, gf, csi, axis_name, rows_total)
            w32 = p.astype(jnp.float32)
            if wd:
                w32 = w32 * wdmul
            w32 = w32 - upd * lr_x
            m_new, w_new = _store(mf, w32, mi.dtype, p.dtype, seed, cfg.sr, row_off=row_off)
        return m_new, w_new, cs_new, rel, seed_t, sv

    traced = any(isinstance(x, jax.core.Tracer) for x in pflat.values())
    active = [e for e in spec if e[4] and gflat.get(e[5]) is not None]
    j = 0
    while j < len(active):
        n, kind, wdf, vocab, hg, leaf, li = active[j]
        i = spec.index(active[j])
        # цикл по шарах — лише під jit (traced): там він побітово = окремим шарам (перевірено jit-варіантом
        # test_stacked_heavy); поза jit (eager, тести) — старий розгорнутий шлях: eager-операції по одній
        # і скомпільоване тіло циклу можуть різнитися на 1 ulp (порядок редукцій норм).
        run = _layer_run(active, j) if (li is not None and traced) else None
        if run is not None:                                  # номери в spec мають іти з однаковим кроком по шарах
            L_, K_ = run
            ix = [[spec.index(active[j + l * K_ + k]) for k in range(K_)] for l in range(L_)]
            if any(ix[l][k] - ix[0][k] != l * (ix[1][0] - ix[0][0]) for l in range(L_) for k in range(K_)):
                run = None
        if run is not None:
            # Складені шари (heavy: L×K ваг підряд у порядку li-major) — fori_loop по шарах: стан (p, m, cs) складених
            # листків — carry, запис шару — dynamic_update_slice у той самий буфер (in-place, без копій усього
            # (L, ...) тензора на кожен шар, як було з розгорнутим .at[li].set). Порядок ваг/насінь/rel — той самий.
            L, K = run
            blk = active[j:j + K]
            i0s = [spec.index(e) for e in blk]
            stride = spec.index(active[j + K]) - i0s[0] if L > 1 else 0
            lv = [e[5] for e in blk]
            cs_keys = tuple(lf for lf in lv if cs.get(lf) is not None)

            def body(l, c, blk=blk, i0s=i0s, stride=stride, lv=lv, cs_keys=cs_keys):
                P_, M_, C_, rel_, sd_, sv_ = c
                P_, M_, C_ = dict(P_), dict(M_), dict(C_)
                for (n_, kind_, wdf_, vocab_, _, leaf_, _), i0 in zip(blk, i0s):
                    pl = jax.lax.dynamic_index_in_dim(P_[leaf_], l, 0, keepdims=False)
                    ml = jax.lax.dynamic_index_in_dim(M_[leaf_], l, 0, keepdims=False)
                    gl = jax.lax.dynamic_index_in_dim(gflat[leaf_], l, 0, keepdims=False)
                    cl = jax.lax.dynamic_index_in_dim(C_[leaf_], l, 0, keepdims=False) if leaf_ in C_ else None
                    m_new, w_new, cs_new, rel_, sd_, sv_ = one(i0 + l * stride, kind_, wdf_, vocab_, leaf_, 0, pl, ml, cl,
                                                               gl, rel_, sd_, sv_)
                    M_[leaf_] = jax.lax.dynamic_update_index_in_dim(M_[leaf_], m_new.astype(M_[leaf_].dtype), l, 0)
                    P_[leaf_] = jax.lax.dynamic_update_index_in_dim(P_[leaf_], w_new.astype(P_[leaf_].dtype), l, 0)
                    if cs_new is not None and leaf_ in C_:
                        C_[leaf_] = jax.lax.dynamic_update_index_in_dim(C_[leaf_], cs_new.astype(C_[leaf_].dtype), l, 0)
                return P_, M_, C_, rel_, sd_, sv_
            c0 = ({k: new_p[k] for k in lv}, {k: m[k] for k in lv}, {k: cs[k] for k in cs_keys}, rel, seed_t, sv)
            P_, M_, C_, rel, seed_t, sv = jax.lax.fori_loop(0, L, body, c0)
            new_p.update(P_); m.update(M_); cs.update(C_)
            j += L * K
            continue
        p, mi, csi = new_p[leaf], m[leaf], cs.get(leaf)
        grad = gflat[leaf]
        if li is None and fish_fast_ok(leaf, p, csi, cfg, fish_ids):
            # 15.11 швидкість: лише активні фішки (k рядків з NF) — читання/запис k/NF стеку замість усього;
            # простій (рядок без градієнта) — як _freeze_idle: ваги/m/cs як були
            ids = fish_ids.astype(jnp.int32)
            ps, ms_, gs = rb_rows(p, ids), rb_rows(mi, ids), rb_rows(grad, ids)
            cs_s = None if csi is None else rb_rows(csi, ids)
            m_s, w_s, c_s, rel, seed_t, sv = one(i, kind, wdf, vocab, leaf, li, ps, ms_, cs_s, gs, rel, seed_t, sv,
                                                 row_off=ids, rel_pg=(p, grad))
            # rel (lax.cond) читає повний p ДО запису рядків: бар'єр дає порядок «cond → запис на місці», інакше XLA
            # копіює весь стек (операнд cond і буфер DUS — один)
            _, p = jax.lax.optimization_barrier((rel, p))
            act = (jnp.abs(gs.astype(jnp.float32)).reshape(gs.shape[0], -1).sum(1) > 0).reshape(
                (-1,) + (1,) * (gs.ndim - 1))
            m[leaf] = rb_set_rows(mi, ids, jnp.where(act, m_s.astype(mi.dtype), ms_))
            new_p[leaf] = rb_set_rows(p, ids, jnp.where(act, w_s.astype(p.dtype), ps))
            if c_s is not None and csi is not None:
                cs[leaf] = rb_set_rows(csi, ids, jnp.where(act, c_s.astype(csi.dtype), cs_s))
            if fish_done is not None:
                fish_done.add(leaf)
            j += 1
            continue
        if li is not None:                                   # один шар складеного листка
            p, mi, grad = p[li], mi[li], grad[li]
            csi = None if csi is None else csi[li]
        m_new, w_new, cs_new, rel, seed_t, sv = one(i, kind, wdf, vocab, leaf, li, p, mi, csi, grad, rel, seed_t, sv)
        if cs_new is not None:
            cs[leaf] = cs_new if li is None else cs[leaf].at[li].set(cs_new)
        if li is None:
            m[leaf], new_p[leaf] = m_new, w_new
        else:
            m[leaf], new_p[leaf] = m[leaf].at[li].set(m_new), new_p[leaf].at[li].set(w_new)
        j += 1
    params = jax.tree_util.tree_unflatten(treedef, [new_p[path_str(p)] for p, _ in leaves])
    extra = {"snr_v": sv[0], "snr_f": sv[1]} if use_snr else {}
    if use_damp:                                            # fix21: типові норми — у рядок картки цього кроку
        extra["dv"] = st["dv"].at[st["k"]].set(sv[2])
        extra["df"] = sv[3]
        if "gidx" in st:
            extra["gidx"], extra["gcalm"] = sv[4], sv[5]
    return params, dict(st, m=m, cs=cs, rel=rel, seed_t=seed_t, **extra)


# ------------------------------------------------------------------------------------------------- увесь крок
def step(params, grads, st, *, lr, situation, loss, spec, cfg, axis_name=None, rows_total=None, shard=None, card=None,
         rb_dst=None, fish_ids=None):
    """(params, grads, opt_state, lr, situation, loss) → (params, opt_state): prepare → update → finish.
    lr — python float (рахується lr_at на хості, float64 → f32) або traced скаляр. card — див. prepare."""
    lr_t, lr_v, ratio = lr_scalars(lr, cfg)
    p_old, m_old, cs_old = params, dict(st["m"]), dict(st["cs"])
    st = prepare(st, loss, situation, ratio, cfg, card=card)
    fdone = set()
    params, st = update(params, grads, st, lr_t, lr_v, cfg, spec, axis_name, rows_total, shard, rb_dst=rb_dst,
                        fish_ids=fish_ids, fish_done=fdone)
    params, st = _freeze_idle(p_old, m_old, cs_old, params, grads, st, skip=fdone)
    return params, finish(st, cfg)


FREEZE_IDLE = ("fish.up", "fish.down", "fish.norm")       # стек (NF, ...): фішки, яких крок не торкався, не міняються


def _freeze_idle(p_old, m_old, cs_old, params, grads, st, skip=frozenset()):
    """Фішки (Стефан 11.11): за крок працює лише 2 з NF. Фішка без градієнта цього кроку лишається як була —
    ні згасання ваг (wd), ні руху за інерцією (як у torch AdamW, де невикористаний параметр має grad None).
    Без цього за ~97% «простою» кожної фішки wd стер би її ваги за тренування."""
    gf = {path_str(p): g for p, g in jax.tree_util.tree_flatten_with_path(grads)[0]}
    if not any(n in gf for n in FREEZE_IDLE):
        return params, st
    act = {n: jnp.abs(gf[n].astype(jnp.float32)).reshape(gf[n].shape[0], -1).sum(1) > 0 for n in FREEZE_IDLE
           if n in gf and n not in skip}            # skip — фішки, вже оновлені лише в активних рядках (update)
    if not act:
        return params, st

    def pick(n, new, old):
        a = act[n].reshape((-1,) + (1,) * (new.ndim - 1))
        return jnp.where(a, new, old.astype(new.dtype))
    po = {path_str(p): x for p, x in jax.tree_util.tree_flatten_with_path(p_old)[0]}
    params = jax.tree_util.tree_map_with_path(
        lambda p, x: pick(path_str(p), x, po[path_str(p)]) if path_str(p) in act else x, params)
    m, cs = dict(st["m"]), dict(st["cs"])
    nf = {n: act[n].shape[0] for n in act}
    for n in act:                     # лише стан, що йде ПО ФІШКАХ (перша вісь = NF); напр. cs fish.norm має форму (1, d)
        if n in m and n in m_old and m[n].shape[:1] == (nf[n],):
            m[n] = pick(n, m[n], m_old[n])
        if n in cs and n in cs_old and cs[n].shape[:1] == (nf[n],) and cs[n].shape == cs_old[n].shape:
            cs[n] = pick(n, cs[n], cs_old[n])
    return params, dict(st, m=m, cs=cs)

