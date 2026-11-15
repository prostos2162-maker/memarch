"""Конфігурація моделі (Fv2moeK40h14d2560 і малі тестові) + сталі з файлів (sv_*, lock_bucket, LOCK_TRUST).

Сталі будуються numpy так само, як буфери FullLM2 (full2.py:1310-1338 і 1405-1418): ті самі файли, що на VM
(_tmp/phr/counts_fineweb.npy, _tmp/phr/lock_trust.pt) → побітово ті самі масиви."""
import dataclasses
import math
import os

import numpy as np

_PHR = os.environ.get("MEMARCH_PHR", "phr")
DEFAULT_COUNTS = os.path.join(_PHR, "counts_fineweb.npy")
DEFAULT_LOCK_TRUST = os.path.join(_PHR, "lock_trust.pt")


@dataclasses.dataclass(frozen=True)
class Cfg:
    # --- форма моделі
    d: int = 2560
    head_dim: int = 256
    E: int = 40                  # експертів у важкому шарі
    heavy: int = 14
    light_a: int = 2
    light_b: int = 2
    final: int = 2
    light_ffn: float = 2.0       # LIGHT_FFN: ширина FFN легких шарів у d (relu², без гейта)
    window: int = 2048
    seg: int = 512
    frac: float = 0.2
    cap: float = 0.25
    V: int = 131328
    # --- вихід і втрата
    short_v: int = 8192
    ce_chunk: int = 16768
    rare_cap: float = 0.3
    read_cost: float = 0.01
    reread_anch: float = 0.25
    # --- MoE
    moe_k: int = 4
    cap_align: int = 256
    moe_group: int = 512
    moe_fast: bool = True          # moe.MoECfg.fast: обмін у (W,El,c,d) без транспозицій; False — старий шлях (A/B)
    moe_a2a_chunks: int = 1        # >1 — шматки для перекриття a2a з матмулами (wgrad експертів ≈1 ulp bf16)
    moe_a2a_impl: str = "a2a"      # 'a2a' | 'ppermute' (кільце асинхронних collective-permute)
    vocab_take: str = "ag"         # VocabShard.take: 'ag' (новий) | 'masked' (старий)
    vocab_take_bwd: str = "sort"   # бекворд take: 'sort' (sort+segment_sum) | 'scatter' (старий scatter-add)
    vocab_ce_a2a: bool = True      # VocabShard.ce: a2a/psum_scatter замість AG/psum
    # --- пам'ять (mem2.Memory за замовчуванням: fresh/dim/dim_max/write/focus)
    mem_fresh: int = 64
    mem_dim: int = 192
    mem_dim_max: int = 256
    mem_write: int = 32
    mem_focus: int = 8
    mem_eb: bool = False         # MEM_EINSUM_BF16 (лише паритет з torch CPU autocast)
    # --- запуск
    W: int = 1                   # чипів (етап паритету — 1)
    B: int = 1
    T: int = 8192
    # --- точність: "bf16" (як на TPU: BF16_ALL + autocast) | "f32" (точні перевірки паритету)
    dtype: str = "bf16"
    attn_impl: str = "dense"     # "dense" (CPU-еталон) | "splash" (TPU, Pallas)
    rope_base: float = 10000.0
    # --- «обруч» (uroboros.py): важкі шари лише для слів, де модель сама чекає своєї помилки (замість якорів)
    uroboros: bool = False           # False — старі якорі за здивуванням легкої голови (tau)
    uroboros_budget: float = 0.25    # частка слів у важкі шари (статичний top-k; рівень = компіляція)
    uroboros_levels: tuple = (0.25, 0.18, 0.12, 0.08)   # спадні рівні бюджету для uroboros.next_budget
    uroboros_eat: float = 1.0        # поріг «з'їденого» слова: передбачена помилка < uroboros_eat нат
    uroboros_w: float = 0.1          # вага MSE голови обруча в loss
    uroboros_thr: float = -1.0       # ≥0: каузальний відбір (eval): слово йде у важкі, якщо ŝ > поріг (без top-k)
    uroboros_at: str = "light_b"     # де стоїть голова: "light_b" (lh, після пам'яті) | "light_a" (до пам'яті)

    # --- «вагонетка» (rope.py): пам'ять-якорі замість mem2 і перечитування, лише в light_b. False — модель без змін
    rope: bool = False
    rope_k: int = 4
    rope_links: int = 8
    rope_win: int = 64
    remind: bool = False             # нагадування (голки ключ→значення в тексті): prep["rem"] + доданок remind
    remind_w: float = 1.0            # вага доданку remind (сума NLL відповідей / (B·remind_q))
    remind_use_w: float = 0.0        # штраф на питаннях нагадувань: w·mean(1 − частка троса); 0 — немає
    remind_only: bool = False        # етап 1 (--remind_synth): втрата лише на відповідях (+aux/light), без main
    norm_dump_proj: bool = False     # fix14 F1 (--norm_dump_proj): фінальна norm прибирає складову вздовж opt.U (prims.dump_proof_norm)
    stream_m: bool = False
    # fix16 (13.11): після фінальної norm спільний вектор (причинне середнє по попередніх позиціях рядка) обмежено
    # |m| ≤ safe_norm_ratio·|r| (r = відхилення слів); 0 — вимкнено; перші safe_norm_t0 позицій не чіпаються
    safe_norm_ratio: float = 0.0
    # fix17 (14.11): шарнірний штраф на стовбур — |середнє hn по позиціях рядка| > hn_guard_mean або середня |hn| >
    # hn_guard_len (після фінальної norm); w·(відносний надлишок)²; 0 — вимкнено. Пороги вище значень кроку 15000 (9.8 / 35.5)
    hn_guard_w: float = 0.0
    hn_guard_mean: float = 15.0
    hn_guard_len: float = 42.0
    safe_norm_t0: int = 256           # 13.11 (--opt_stream_proj): forward кладе в stats["stream_m"] середнє h перед фінальною norm
    remind_mean: bool = False        # доданок remind = середнє по відповідях батчу (--remind_pack: відповідей у рядку багато)
    # «чесний тест» (Стефан 12.11, етап 1): після першої помилки в епізоді правильні відповіді не винагороджуються
    # (їх можна взяти перебором), а ймовірність на ІНШИХ значеннях рядка штрафується завжди: w·−log(1−P(інші))
    remind_strict: bool = False
    # «відповіді бачиш лише після тесту» (Стефан 12.11): prep має rem_y — ціль на місцях відповідей (у тексті заглушка)
    remind_blank: bool = False
    remind_strict_w: float = 1.0
    remind_q: int = 16               # питань на рядок-нагадування (нормування як у torch: середнє за кроки = p·CE)
    # 14.11 fix19 (злам після 15000, слід — голки після зміни словника): (1) remind_cand — втрата відповіді голки =
    # softmax лише по 64 токенах-значеннях (рядки голови v0..v0+63), а не по всьому словнику: спільна частина рядків
    # значень градієнта не отримує → немає глобального підсилення «токенів голок» і поштовху кошика рідкісних V-1;
    # (2) remind_mask_main — у рядках з голками main не вчиться на токенах голок (ключі/значення випадкові → шум,
    # що тягне всю групу токенів угору). У рядках без голок звичайні токени 1000..1576 вчаться як завжди.
    # 15.11 Instinct на словах (задум Стефана; див. model.loss_terms): пороги й ваги
    inst_word: bool = False
    iw_ok: float = 0.5
    iw_spread: float = 0.35
    iw_target: float = 0.3
    iw_spread_w: float = 0.2
    iw_cw: float = 0.5
    iw_cw_w: float = 0.1
    iw_rewrite: bool = False   # 15.11 Стефан: переписана відповідь як мішень (лише 2 слова: правильне +δ, «найтупіше» −δ)
    iw_delta: float = 0.05
    iw_random: bool = False   # 15.11 Стефан: −δ випадковому слову (не найтупішому) — без top-k і вибірок
    iw_k: int = 4              # скільки конкурентів дивитися для «найтупішого»
    remind_cand: bool = False
    remind_mask_main: bool = False
    remind_tok0: int = 1000
    rope_span_cap: int = 256
    rope_gather_cap: int = 0         # слів на запит у зібраній частині; 0 → k·span_cap (точно, без переповнення)
    rope_route: str = "hier"         # "hier" | "exact"
    rope_hier_g: int = 64
    rope_hier_nb: int = 16
    rope_hier_bm: int = 64
    rope_excl_win: bool = True
    rope_route_rot: bool = True
    rope_rgrad: bool = True
    rope_chunk: int = 256            # запитів на шматок уваги (lax.map + checkpoint)
    rope_route_lazy: bool = False    # rope.RopeCfg.route_lazy
    rope_dense_read: bool = False    # rope.RopeCfg.dense_read
    rope_route_bf16: bool = False    # rope.RopeCfg.route_bf16
    rope_pallas: bool = False        # rope.RopeCfg.pallas (Pallas-ядра, exact route)
    fish_n: int = 0                  # «фішки» (Стефан 11.11): >0 — важкі MoE замінено на fish_n однакових FFN-шарів
    fish_hf: int = 19456             # ширина фішки (2·d·hf ≈ 0.1B при d=2560)
    fish_kinds: int = 5              # видів тексту (комбінацій)
    fish_pool: int = 17              # фішок у наборі виду (готова модель: ~17·0.1B ≈ 1.7B + легкі ≈ 2B на слово)
    fish_eval: bool = False          # оцінка готової моделі: вид рядка вибирає роутер, рядок іде через увесь набір
    fish_c: int = 2                  # фішок за крок на тренуванні (випадкові з набору виду)
    # eval без фінальної фази (12.11): рядок іде через fish_eval_c фішок свого виду (фіксована випадкова підмножина
    # набору), як на тренуванні; 0 — через увесь набір (готова модель, лише після фінальної фази)
    fish_eval_c: int = 0
    fish_center: bool = False        # 14.11 fw_s2: вхід фішки без спільного вектора (каузальне середнє) — фікс мертвих relu²
    heavy_rotate: bool = False       # ідея Стефана 11.11: важкий шар номер prep['hfull'] цього кроку — на ВСІХ словах (тасування)
    heavy_all: bool = False          # усі важкі шари на всіх словах (фінальна фаза перед SFT / готова модель)
    rope_doc_eos: int = -1           # >=0: id токена EOS; трос не відкриває якорі з іншого документа (лише маска, без змін ядер)
    rope_read: str = "v3"            # rope.RopeCfg.read: "v1" старе ядро читання, "v3" — rope_read_v3 (швидше, та сама математика)
    rope_layers: int = 0             # у скількох перших шарах light_b вагонетка (0 — у всіх); решта — звичайна локальна увага
    rope_share_route: bool = False   # другий і далі шари беруть вибір якорів першого (зміна семантики — потрібна перевірка пригадування)
    rope_tau0: float = 5.0           # початковий поріг якорів (full2.rope_tau)
    rope_qfrac: float = 1.0          # частка слів, що читають трос (решта — лише вікно); 1.0 = як було (ROPE_QFRAC у full2)
    rope_qtau0: float = 5.0          # початковий поріг QFRAC (EMA квантиля 1-qfrac у навчанні, заморожений в оцінці)
    # ПРОПОЗИЦІЇ 12.11 (вимкнено; проти застрягання «ключ→значення» на широких моделях):
    attn_gain: float = 1.0           # множник оцінок уваги легких шарів (q·gain); 1.0 — як було
    rope_anchor_top: bool = False    # стеля якорів троса: True — A найбільш здивованих по всьому рядку; False — перші A (як було)
    rope_pcut: float = 0.0           # p-RoPE у шарі троса: обертаються лише частоти з хвилею ≤ rope_pcut слів (0 — як було)
    rope_frac: float = 0.0           # частка якорів ТРОСА (квантиль порогу); 0 — cfg.frac, як було (важкі якорі не чіпає)
    rope_cap: float = 0.0            # стеля якорів троса (A = rope_cap·T); 0 — cfg.cap, як було
    rope_key_diag: bool = False      # 13.11 діагностика klink/kanch (частка ключів голок у ланках/якорях); без впливу на навчання
    prev_in: bool = False            # вхід «попереднє слово»: h[t] += σ(prev_g)·emb(x[t−1]) (поруч із замками; новий параметр prev_g)

    # старі назви (hoop*) — лише читання, для сумісності зі старим кодом
    hoop = property(lambda self: self.uroboros)
    hoop_budget = property(lambda self: self.uroboros_budget)
    hoop_levels = property(lambda self: self.uroboros_levels)
    hoop_eat = property(lambda self: self.uroboros_eat)
    hoop_w = property(lambda self: self.uroboros_w)
    hoop_thr = property(lambda self: self.uroboros_thr)
    hoop_at = property(lambda self: self.uroboros_at)

    @property
    def heads(self):
        return self.d // self.head_dim

    @property
    def light_hid(self):
        """Ширина FFN легкого блоку (full.py:130-134): max(256, round(LIGHT_FFN·d/256)·256); без LIGHT_FFN — 6d."""
        if self.light_ffn:
            return max(256, int(round(self.light_ffn * self.d / 256)) * 256)
        return 6 * self.d

    @property
    def cdt(self):
        """dtype обчислень/залишкового потоку."""
        import jax.numpy as jnp
        return jnp.bfloat16 if self.dtype == "bf16" else jnp.float32


def target():
    """Fv2moeK40h14d2560 під цільовим env (tpu_m23 + G22)."""
    return Cfg()


def small(**kw):
    """Мала конфігурація для паритету на CPU (Fv2moeKs8-подібна)."""
    base = dict(d=256, head_dim=64, E=8, heavy=2, light_a=1, light_b=1, final=1, window=64, seg=64,
                V=4096, short_v=512, ce_chunk=1024, T=256, dtype="f32")
    base.update(kw)
    return Cfg(**base)


def tiny_k40(**kw):
    """«tiny K40»: E=40, d=128 — перевірка EP 40/8."""
    base = dict(d=128, head_dim=64, E=40, heavy=2, light_a=1, light_b=1, final=1, window=64, seg=64,
                V=4096, short_v=512, ce_chunk=1024, T=256, dtype="f32")
    base.update(kw)
    return Cfg(**base)


def from_env(env=None):
    """Cfg з тих самих змінних середовища, що читає torch (HEAD_DIM, LIGHT, LIGHT_FFN, SEG, VOCAB, SHORT_V, ...)."""
    e = os.environ if env is None else env
    g = lambda k, dflt, f=int: f(e[k]) if e.get(k) else dflt
    return Cfg(head_dim=g("HEAD_DIM", 256), light_a=g("LIGHT", 2), light_b=g("LIGHT", 2),
               light_ffn=g("LIGHT_FFN", 2.0, float), seg=g("SEG", 512), V=g("VOCAB", 131328),
               short_v=g("SHORT_V", 8192), ce_chunk=g("CE_CHUNK", 16768), rare_cap=g("RARE_CAP", 0.3, float),
               read_cost=g("READ_COST", 0.01, float), reread_anch=g("REREAD_ANCH", 0.25, float),
               attn_impl="splash" if e.get("ATTN_IMPL") == "splash" else "dense")


# ---------------------------------------------------------------- сталі з файлів

def load_counts(V, path=DEFAULT_COUNTS):
    """TOKEN_COUNTS, обрізані/доповнені нулями до V (int64)."""
    cnt = np.zeros(V, np.int64)
    if path:
        c0 = np.load(path).astype(np.int64)[:V]
        cnt[:len(c0)] = c0
    return cnt


def load_lock_trust(path=DEFAULT_LOCK_TRUST):
    """LOCK_TRUST (3,7) як fp32 numpy. Файл — torch .pt, тож потрібен torch (лише тут, на хості)."""
    if not path:
        return np.ones((3, 7), np.float32)
    import torch
    return torch.load(path).float().numpy()


def short_vocab_consts(cnt, K):
    """sv_ids, sv_pos, sv_mask, sv_bucket, sv_prior0/1 — точно як full2.py:1310-1338."""
    V = len(cnt)
    top = np.argsort(-cnt, kind="stable")[:K].astype(np.int64)
    pos = np.full(V, K, np.int64)
    pos[top] = np.arange(K)
    bucket = V - 1
    assert cnt[bucket] == 0 and pos[bucket] == K, "останній рядок словника має бути порожнім"
    msk = (pos < K).astype(np.float32) * np.float32(-1e4)
    msk[bucket] = -1e4
    rare = 1 - cnt[top].sum() / max(cnt.sum(), 1) if cnt.sum() else 0.18
    rare = min(max(rare, 1e-3), 0.999)
    return dict(sv_ids=top, sv_pos=pos, sv_mask=msk, sv_bucket=bucket,
                sv_prior0=float(np.log(rare * K / (1 - rare))), sv_prior1=float(np.log(rare)))


def lock_bucket(cnt):
    """Кошик рідкісності 0..6: clip(floor(log10(cnt+1)), 0, 6) (full2.py:1411)."""
    return np.clip(np.floor(np.log10(cnt + 1)), 0, 6).astype(np.int64)


def mix_w_init(lw):
    """mix_w = logit(clamp(exp(lw)·e^-4.5, .005, .95)) у fp32 (далі округлюється до bf16 при BF16_ALL)."""
    lw = np.asarray(lw, np.float32)
    pm = np.clip(np.exp(lw) * np.float32(math.exp(-4.5)), 0.005, 0.95).astype(np.float32)
    return np.log(pm / (1 - pm)).astype(np.float32)


def load_consts(cfg, counts_path=DEFAULT_COUNTS, lock_trust_path=DEFAULT_LOCK_TRUST):
    """Усі сталі моделі (numpy). Великі масиви перетворює на jnp викликач."""
    cnt = load_counts(cfg.V, counts_path)
    c = short_vocab_consts(cnt, cfg.short_v) if cfg.short_v else {}
    c["lock_bucket"] = lock_bucket(cnt)
    c["lock_trust"] = load_lock_trust(lock_trust_path)
    return c
