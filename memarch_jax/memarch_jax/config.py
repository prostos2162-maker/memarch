"""Конфігурація моделі (Fv2moeK40h14d2560 і малі тестові) + сталі з файлів (sv_*, lock_bucket, LOCK_TRUST).

Сталі будуються numpy так само, як буфери FullLM2 (full2.py:1310-1338 і 1405-1418): ті самі файли, що на VM
(phr/counts_fineweb.npy, phr/lock_trust.pt) → побітово ті самі масиви."""
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
    # --- «обруч» (hoop.py): важкі шари лише для слів, де модель сама чекає своєї помилки (замість якорів)
    hoop: bool = False           # False — старі якорі за здивуванням легкої голови (tau)
    hoop_budget: float = 0.25    # частка слів у важкі шари (статичний top-k; рівень = компіляція)
    hoop_levels: tuple = (0.25, 0.18, 0.12, 0.08)   # спадні рівні бюджету для hoop.next_budget
    hoop_eat: float = 1.0        # поріг «з'їденого» слова: передбачена помилка < hoop_eat нат
    hoop_w: float = 0.1          # вага MSE голови обруча в loss
    hoop_thr: float = -1.0       # ≥0: каузальний відбір (eval): слово йде у важкі, якщо ŝ > поріг (без top-k)
    hoop_at: str = "light_b"     # де стоїть голова: "light_b" (lh, після пам'яті) | "light_a" (до пам'яті)

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
