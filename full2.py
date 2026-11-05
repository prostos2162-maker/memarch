"""Повна модель v2 (ARCHITECTURE.md, «Уточнення Стефана»): дві незалежні системи.

  легкі шари (вікно)  →  читання пам'яті (фокус + картки)  →  легкі шари (вікно)
     → здивування → якорі (поріг + стеля)  →  важкі шари: лише думають над якорем (без уваги)
     → фінальні шари (вікно)  →  перечитування за закладкою + впізнавання  →  вихід

Пам'ять (mem2.Memory) — окрема система: свій сигнал яскравості, фіксований розмір
(свіжа + тьмяна з прибиранням), картки з ланцюжком лиць і закладками. Якорі —
лише рішення, де думати важко. Ціна на слово не залежить від довжини тексту.

Fv2 — ~300M (ширина 1024: 2+2 легкі, 14 важких, 2 фінальні), Fv2s — маленька для перевірок.
"""
import int8mm  # noqa: E402  (INT8_MM=1 підміняє F.linear до побудови моделі)
import math

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import dev
from full import FlexBlock, _COMPILE
from lm import FFN
from mem2 import Memory
from mem3 import VolumeMemory

ATTN_IMPL = os.environ.get("ATTN_IMPL", "")
MOE_CAP_ALIGN = int(os.environ.get("MOE_CAP_ALIGN", "256" if dev.TYPE in ("tpu", "xla") else "0"))   # кратність місткості експерта
# MOE_GMM=1: експерти одним torch._grouped_mm над якорями, впорядкованими за експертом (як megablox): без місткості,
# без доповнення й відкидань; форми сталі (N·k рядків), пари «нікуди» стоять у хвості й не рахуються
MOE_GMM = os.environ.get("MOE_GMM") == "1"
MOE_GATHER = bool(os.environ.get("MOE_GATHER", "1" if dev.TYPE in ("tpu", "xla") else ""))   # розсилка/складання MoE без scatter
# MOE_DISP=inv (лише разом з MOE_GATHER): розсилка й складання як пара взаємно обернених gather — назад теж gather
# (без index_put/scatter-add), складання злите з вагами місць. Уперед бітово те саме; назад — лише порядок сум fp.
MOE_DISP = os.environ.get("MOE_DISP", "")
# MOE_DISP=onehot: розсилка/складання як у GShard — слова діляться на групи по MOE_GROUP, у кожній групі експерт має
# Cg місць; розсилка = einsum з one-hot маскою D (G, g, E, Cg), складання = einsum з D·вага. Жодного gather/scatter
# над d-широкими даними (і назад теж — лише множення на MXU). Cg = ceil(C / G): G·Cg ≥ C старої місткості.
MOE_GROUP = int(os.environ.get("MOE_GROUP", "512"))
# MOE_IDX32=1: лічильники one-hot-розсилки (cumsum/rank/D) в int32 замість int64 — та сама арифметика, менше байтів
MOE_IDX32 = os.environ.get("MOE_IDX32") == "1"
CKPT_NO_RNG = os.environ.get("CKPT_NO_RNG") == "1"   # див. FullLM2._run
_MOE_INV_CHECK = os.environ.get("CHECK") == "1" and dev.TYPE == "cpu"   # перевірка взаємної оберненості (лише CPU)
# DIAG_LAYER=1: час кожного важкого шару по частинах (увага / розсилка / all-to-all / експерти / складання),
# і вперед, і назад (хуки на градієнтах у тих самих межах). Кожна мітка чекає пристрій (dev.sync) — це ламає
# перекриття, тож цифри — «скільки коштує частина сама по собі», не для заміру швидкості кроку. Вимкнено — нуль змін.
DIAG_LAYER = os.environ.get("DIAG_LAYER") == "1"
_LD = {"fwd": {}, "bwd": {}, "f_last": None, "b_last": None, "n": 0}


def _ld_now():
    import time
    if getattr(dev, "XLA", False):                 # лінивий XLA: спершу виконати зібраний шматок графа
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    dev.sync()
    return time.perf_counter()


def _ld_add(kind, li, name, dt):
    k = (li, name)
    _LD[kind][k] = _LD[kind].get(k, 0.0) + dt


def _ld_bwd_hook(li, nxt):
    def hook(g):
        t = _ld_now()
        if nxt is not None and _LD["b_last"] is not None:
            _ld_add("bwd", li, nxt, t - _LD["b_last"])     # назад: між цією міткою і попередньою — частина ПІСЛЯ мітки
        _LD["b_last"] = t
        return None
    return hook


@torch.compiler.disable
def _ld(t, li, done, nxt, li_done=None):
    """Мітка межі: вперед — час від попередньої мітки йде частині `done` (що щойно скінчилась);
    назад — частині `nxt` (що йде після мітки). done/nxt=None — лише точка відліку.
    post = місцеве складання (gather/index_add) + залишок; comb = зворотний all-to-all (лише EP)."""
    now = _ld_now()
    if done is not None and _LD["f_last"] is not None:
        _ld_add("fwd", li if li_done is None else li_done, done, now - _LD["f_last"])
    _LD["f_last"] = now
    if torch.is_grad_enabled() and t.requires_grad:
        t.register_hook(_ld_bwd_hook(li, nxt))
    return t


def layer_diag_report(step):
    """Друк LAYERDIAG (мс, сума по шарах з часу останнього звіту, поділена на кількість кроків) і скидання."""
    if not DIAG_LAYER:
        return
    n = max(_LD["n"], 1)
    names = ("attn", "disp", "a2a", "exp", "comb", "post")
    parts = []
    for kind in ("fwd", "bwd"):
        tot = {}
        for (li, nm), v in _LD[kind].items():
            tot[nm] = tot.get(nm, 0.0) + v
        allms = sum(tot.values()) * 1e3 / n
        parts.append(kind + " " + " ".join(f"{nm}={tot.get(nm, 0.0) * 1e3 / n:.1f}" for nm in names)
                     + f" sum={allms:.1f}")
    print(f"LAYERDIAG step {step} (мс/крок, n={_LD['n']}) " + " | ".join(parts), flush=True)
    for kind in ("fwd", "bwd"):
        top = sorted(_LD[kind].items(), key=lambda kv: -kv[1])[:3]
        print(f"LAYERDIAG {kind} top3: " + "  ".join(f"L{li}.{nm}={v * 1e3 / n:.1f}" for (li, nm), v in top),
              flush=True)
    _LD["fwd"].clear(); _LD["bwd"].clear()
    _LD["n"] = 0
    _LD["f_last"] = _LD["b_last"] = None


_HC1, _HC2 = -7046029254386353131, -4658895280553007687   # непарні множники (int64, переповнення — як у numpy)


def _h1(c, mask):
    return ((c * _HC1) >> 24) & mask


def _h2(c, mask):
    return ((c * _HC2) >> 24) & mask


def _cuckoo(keys, max_kick=500):
    """Хеш-таблиця cuckoo для ключів словосполучень: два місця на ключ, заповнення ≤1/8. Повертає (ключі, номери, маска);
    порожнє місце — ключ -2**62 (такого ключа немає). Пошук — 2 перевірки, результат той самий, що у searchsorted."""
    M = 1 << max(10, int(np.ceil(np.log2(len(keys) * 8))))
    mask = M - 1
    with np.errstate(over="ignore"):
        h1 = ((keys * np.int64(_HC1)) >> 24) & mask
        h2 = ((keys * np.int64(_HC2)) >> 24) & mask
    hk = np.full(M, -(1 << 62), dtype=np.int64)
    hv = np.zeros(M, dtype=np.int64)
    owner = np.full(M, -1, dtype=np.int64)
    for i in range(len(keys)):
        cur, slot = i, h1[i]
        for _ in range(max_kick):
            if owner[slot] < 0:
                owner[slot] = cur
                break
            owner[slot], cur = cur, owner[slot]                    # витіснити: той, хто сидів, іде на своє друге місце
            slot = h2[cur] if slot == h1[cur] else h1[cur]
        else:
            raise RuntimeError("cuckoo: не вдалося розмістити ключ — збільште таблицю")
    used = owner >= 0
    hk[used] = keys[owner[used]]
    hv[used] = owner[used]
    return hk, hv, mask              # "block" — увага у вікні без FlexAttention (TPU/XLA)


@torch.compiler.disable
def _window_mask(T, window, device, block=128):
    """Маска вікна, складена одразу з правила «блок бачить себе і попередні window/block блоків».
    Лінійна ціна: не перевіряємо кожну пару «слово × слово» (create_block_mask так робить, і це квадрат)."""
    from torch.nn.attention.flex_attention import BlockMask
    nb = -(-T // block)
    wb = -(-window // block)                                    # скільки блоків назад сягає вікно
    q = torch.arange(nb, device=device)[:, None]
    offs = torch.arange(wb + 1, device=device)[None]            # сам блок + wb попередніх
    kv = q - offs                                               # (nb, wb+1), спершу найближчі
    ok = kv >= 0
    full_kv = kv[:, 1:wb] if wb > 1 else kv[:, :0]              # блоки цілком у вікні (без маски)
    part = torch.cat([kv[:, :1], kv[:, wb:wb + 1]], 1)          # діагональ і край вікна — з маскою
    part_ok = torch.cat([ok[:, :1], ok[:, wb:wb + 1]], 1)
    full_ok = ok[:, 1:wb] if wb > 1 else ok[:, :0]

    def pack(idx, valid):
        n = valid.sum(1).int()
        # дійсні блоки — першими і за зростанням (як у create_block_mask; ядру так зручніше читати пам'ять)
        key = torch.where(valid, idx, torch.full_like(idx, 1 << 30))
        idx = key.sort(dim=1).values.masked_fill(~valid.sort(dim=1, descending=True).values, 0).int()
        # скомпільоване ядро чекає таблицю на всю ширину (стільки стовпців, скільки блоків)
        idx = torch.cat([idx, torch.zeros(nb, max(0, nb - idx.shape[1]), dtype=idx.dtype, device=device)], 1)[:, :nb]
        return n[None, None].contiguous(), idx[None, None].contiguous()

    pn, pi = pack(part, part_ok)
    fn, fi = pack(full_kv, full_ok)
    mask_mod = lambda b, h, qi, ki: (ki <= qi) & (qi - ki < window)
    return BlockMask.from_kv_blocks(pn, pi, fn, fi, BLOCK_SIZE=block, mask_mod=mask_mod, seq_lengths=(T, T))


class Reread(nn.Module):
    """Перечитування за закладкою + впізнавання: слово вибирає картку (за лицем),
    стрибає за закладкою, перечитує кілька сирих токенів (з ціною за кожен),
    а картка перевіряє, чи прочитане збігається з нею."""

    def __init__(self, d, d_face=64, d_book=128, top_k=4, max_read=4):
        super().__init__()
        self.k, self.R, self.df = top_k, max_read, d_face
        self.norm = nn.RMSNorm(d)
        self.query = nn.Linear(d, d_face, bias=False)
        self.offset = nn.Parameter(torch.randn(max_read, d_book) * 0.02)
        self.halt_v = nn.Linear(d_book, 1)
        self.halt_h = nn.Linear(d, 1, bias=False)
        self.recog_r = nn.Linear(d_book, d_face, bias=False)
        self.recog_c = nn.Linear(d, d_face, bias=False)
        self.out = nn.Linear(d_book, d, bias=False)

    def forward(self, h, book, faces, contents, pos, valid, qpos, rows=None):
        """h: (B,S,d) слова шматка; картки: faces (B,C,L,f), contents (B,C,d), pos/valid (B,C)."""
        n = self.norm(h)
        q = F.normalize(self.query(n), dim=-1)
        rb = _RR_BISECT
        if "rr_min" in rb:                           # лише норма + вихід: чи падіння в обгортці, чи всередині модуля
            return self.out(self.norm(h)[..., :self.out.in_features] if self.out.in_features <= h.shape[-1]
                            else F.pad(self.norm(h), (0, self.out.in_features - h.shape[-1]))), h.new_zeros(())
        if "nosc" in rb:                             # без оцінок карток (запит, нормування лиць, множення, amax)
            sc = torch.zeros(B0 := h.shape[0], h.shape[1], faces.shape[1], device=h.device, dtype=h.dtype) + n.mean() * 0
        elif "bmm" in rb:                              # те саме, звичайним пакетним множенням (без einsum)
            Bf, Cf, Lf, Ff = faces.shape
            fz = F.normalize(faces, dim=-1).reshape(Bf, Cf * Lf, Ff)
            sc = torch.bmm(q, fz.transpose(1, 2)).view(Bf, -1, Cf, Lf).amax(-1) * math.sqrt(self.df)
        else:
            sc = torch.einsum("bsf,bclf->bscl", q, F.normalize(faces, dim=-1)).amax(-1) * math.sqrt(self.df)
        sc = sc.masked_fill(~valid[:, None], -1e4)
        k = min(self.k, sc.shape[-1])
        if "notopk" in rb:                           # лише для пошуку збою XLA: перші k карток замість найкращих
            top_s = sc[..., :k]
            top_j = torch.arange(k, device=sc.device).expand(*sc.shape[:-1], k)
        else:
            top_s, top_j = sc.topk(k, -1)
        w = F.softmax(top_s.float(), -1).type_as(h) * (top_s > -1e3)
        B, S = h.shape[:2]
        if "nomark" in rb:
            mark = torch.zeros(B, S, k, device=h.device, dtype=pos.dtype)
        else:
            mark = pos.gather(1, top_j.reshape(B, -1)).view(B, S, k)                 # закладки
        rpos = mark[..., None] + torch.arange(self.R, device=h.device)
        readable = (rpos < qpos[:, :, None, None]) if "noqpos" not in rb else torch.ones_like(rpos, dtype=torch.bool)
        ids = None
        if isinstance(book, tuple):                    # REREAD_IDS: номери слів (Bt, T) + таблиця (V, Db)
            ids, table = book
            Tb, Db = ids.shape[1], table.shape[1]
        else:
            Tb, Db = book.shape[1], book.shape[-1]
        rpos = rpos.clamp(0, Tb - 1)
        if rows is None:
            rows = torch.arange(B, device=h.device)
        flat = (rows[:, None, None, None] * Tb + rpos).reshape(-1)  # рядок книжки = свій текст
        if "nobook" in rb:
            v = torch.zeros(B, S, k, self.R, Db, device=h.device, dtype=torch.float32) + self.offset
        elif ids is not None:                          # цілі номери не мають градієнта — розсилки по тексту немає
            v = F.embedding(ids.reshape(-1)[flat], table).float().view(B, S, k, self.R, Db) + self.offset
        else:
            v = book.reshape(-1, Db)[flat].view(B, S, k, self.R, Db) + self.offset
        if "nohaltlin" in rb:
            halt = torch.full(v.shape[:-1], 0.5, device=h.device, dtype=v.dtype)
        else:
            halt = torch.sigmoid(self.halt_v(v).squeeze(-1) + self.halt_h(n)[:, :, None])
        halt = torch.where(readable, halt, torch.ones_like(halt))
        halt = torch.cat([halt[..., :-1], torch.ones_like(halt[..., -1:])], -1)
        # cumprod по кількох читаннях — розгорнутим циклом (на TPU cumprod — reduce_window з доповненням,
        # компілятор XLA на ньому падає: «split of a padded window dim is not supported»)
        stay = 1 - halt[..., :-1]
        surv = [torch.ones_like(halt[..., 0])]
        for i in range(stay.shape[-1]):
            surv.append(surv[-1] * stay[..., i])
        survive = torch.stack(surv, -1)
        if "nohalt" in rb:
            survive, halt = torch.ones_like(halt), torch.ones_like(halt)
        read = ((survive * halt * readable)[..., None] * v).sum(-2) if "noread" not in rb else v[..., 0, :] * 0 + self.offset[0]
        # впізнавання: чи збігається прочитане з карткою
        if "norecog" in rb:
            recog = torch.ones_like(read[..., :1])
        else:
            # проєкція карток — ДО вибірки (лінійна, порядкова: той самий результат), тож вибираються d_face чисел на
            # картку, а не d (2560 → 64: у ~40 разів менше даних у вибірці та її зворотному scatter)
            if REC_LEAN:
                cp = self.recog_c(contents)                                               # (B, C, d_face)
                card_p = cp.gather(1, top_j.reshape(B, -1, 1).expand(-1, -1, cp.shape[-1])).view(B, S, k, -1)
            else:
                card = contents.gather(1, top_j.reshape(B, -1, 1).expand(-1, -1, contents.shape[-1])).view(B, S, k, -1)
                card_p = self.recog_c(card)
            recog = torch.sigmoid((F.normalize(self.recog_r(read), dim=-1)
                                   * F.normalize(card_p, dim=-1)).sum(-1, keepdim=True) * 4)
        got = (w[..., None] * recog * read).sum(-2)
        cost = (w * (survive * readable).sum(-1)).sum(-1).mean()
        return self.out(got), cost


class _SlotGather(torch.autograd.Function):
    """MOE_DISP=inv: місця експертів беруть свої слова (xd[table]); назад — кожне слово бере градієнти своїх
    k місць (fi) і складає їх у fp32. Обидва напрямки — gather, scatter-add немає."""
    @staticmethod
    def forward(ctx, xd, table, fi, EC):
        N = xd.shape[0]
        m = (table < N)[:, None]
        xe = torch.where(m, xd[table.clamp(max=N - 1)], xd.new_zeros(()))       # те саме, що cat(xd, 0)[table]
        ctx.save_for_backward(table, fi)
        ctx.N, ctx.EC = N, EC
        return xe

    @staticmethod
    def backward(ctx, g):
        table, fi = ctx.saved_tensors
        EC = ctx.EC
        mk = (fi < EC)[..., None]
        dx = torch.where(mk, g[fi.clamp(max=EC - 1)], g.new_zeros(())).float().sum(1).to(g.dtype)
        return dx, None, None, None


class _CombineW(torch.autograd.Function):
    """MOE_DISP=inv: складання з вагами одним кроком: out[n] = Σ_j (ye[fi[n,j]] · wtab[fi[n,j]]) у fp32.
    Порядок як раніше (bf16 добуток → точне розширення → сума по k), тож уперед бітово те саме.
    Назад: місце бере градієнт свого слова (go[table]) — gather замість scatter-add."""
    @staticmethod
    def forward(ctx, ye, wtab, fi, table, N, out_dtype):
        EC = ye.shape[0]
        c = fi.clamp(max=EC - 1)
        mk = fi < EC
        yg = ye[c]                                                                # (N,k,d)
        wg = torch.where(mk, wtab[c], wtab.new_zeros(()))                       # (N,k) — порожні вибори дають 0
        out = (yg * wg[..., None]).to(out_dtype).sum(1)
        ctx.save_for_backward(ye, wtab, fi, table)
        ctx.N = N
        return out

    @staticmethod
    def backward(ctx, go):
        ye, wtab, fi, table = ctx.saved_tensors
        N = ctx.N
        m = (table < N)[:, None]
        gs = torch.where(m, go[table.clamp(max=N - 1)], go.new_zeros(())).to(ye.dtype)   # (E·C,d)
        dye = gs * wtab[:, None]
        dwtab = (gs.float() * ye.float()).sum(-1).to(wtab.dtype)
        return dye, dwtab, None, None, None, None


class _ErrProbe(torch.autograd.Function):
    """Тотожність уперед; назад — ковзне середнє норми градієнта на виході кожного експерта (його «помилка»:
    як сильно loss хоче змінити його роботу). Усе на відеокарті, без зупинок."""

    @staticmethod
    def forward(ctx, y, buf):
        ctx.buf = buf
        return y.view_as(y)

    @staticmethod
    def backward(ctx, g):
        with torch.no_grad():
            ctx.buf.mul_(0.9).add_(0.1 * g.float().norm(dim=-1).mean(1))
        return g, None


class _ErrProbeTok(torch.autograd.Function):
    """Те саме, що _ErrProbe, але з градієнта на виході слів (N, d) (EP_A2A_DEDUP: виходи експертів уже складені на
    власнику): градієнт місця (e, c) = вага місця × градієнт його слова, тож норма = вага × норма градієнта слова."""

    @staticmethod
    def forward(ctx, out, buf, table, wtab, E, C):
        ctx.buf, ctx.E, ctx.C = buf, E, C
        ctx.save_for_backward(table, wtab)
        return out.view_as(out)

    @staticmethod
    def backward(ctx, g):
        table, wtab = ctx.saved_tensors
        with torch.no_grad():
            gn = g.float().norm(dim=-1)
            gn = torch.cat([gn, gn.new_zeros(1)])[table]             # (E·C,) порожнє місце (N) → 0
            ctx.buf.mul_(0.9).add_(0.1 * (gn * wtab.float().abs()).view(ctx.E, ctx.C).mean(1))
        return g, None, None, None, None, None


# EP_A2A_DEDUP=1 (EP, кілька чипів): пара (слово, чип) їде all-to-all один раз, а не раз на вибраного експерта
# того чипа; EP_DEDUP_CAP — стеля місць посилки на чип як частка El·C (1 — точно як без прапора)
EP_A2A_DEDUP = os.environ.get("EP_A2A_DEDUP") == "1"
EP_DEDUP_CAP = float(os.environ.get("EP_DEDUP_CAP", "1.0"))


def _ep_rows(E):
    """EP_ALLOC_LOCAL=1 (разом з EP=1, кілька процесів): кожен процес одразу створює лише СВОЇХ E/W експертів —
    повна таблиця 40 експертів справжньої моделі (~44 ГБ fp32) не влазить ні в чип, ні 8 разів у хост."""
    import os
    W = int(os.environ.get("WORLD_SIZE", 1))
    if os.environ.get("EP") == "1" and os.environ.get("EP_ALLOC_LOCAL") == "1" and W > 1:
        assert E % W == 0
        return E // W
    return E


class BudgetMoE(nn.Module):
    """Важкий шар як MoE зі спільним бюджетом (ARCHITECTURE.md, «Ідея 4»).

    E експертів, кожен — вужчий FFN (hidden = 4d·k_avg/E... тут 2d при k_avg=2), тож у середньому
    обчислень на якір стільки ж, скільки в щільному FFN, а параметрів у E/k_avg разів більше.
    budget=True: кількість експертів на якір від здивування (сильніше → більше, до k_max; слабше → 1),
    у середньому по батчу ≈ k_avg (спільний запас фіксований). budget=False — рівно k_avg на кожен.
    """

    def __init__(self, d, E=8, k_avg=2, k_max=4, budget=True, wave=None, wave_every=50, students=5, hint_w=1.0,
                 frozen=0):
        super().__init__()
        self.E, self.k_avg, self.k_max, self.budget = E, k_avg, k_max, budget
        self.fill = 1.0                   # частка справжніх якорів серед місць (FullLM2 ставить frac / cap)
        hid = 2 * d                       # 8 експертів × 2 активних × 2d = як щільний FFN 4d
        self.norm = nn.RMSNorm(d)
        self.router = nn.Linear(d, E, bias=False)
        # frozen > 0 — «заморожений банк» (замір ціни хвиль): останні `frozen` експертів лежать у bf16 без градієнтів
        # ваг і без оптимізатора, рахуються окремим множенням; вчаться лише перші E - frozen
        Ea = E - frozen
        if getattr(type(self), "_ep_ok", False):          # BankMoE з EP: лише свої експерти
            Ea = _ep_rows(Ea)
        self.n_frozen = frozen
        self.up = nn.Parameter(torch.randn(Ea, d, 2 * hid) * d ** -0.5)
        self.down = nn.Parameter(torch.randn(Ea, hid, d) * hid ** -0.5 * 0.5)
        if frozen:
            self.register_buffer("up_f", (torch.randn(frozen, d, 2 * hid) * d ** -0.5).bfloat16())
            self.register_buffer("down_f", (torch.randn(frozen, hid, d) * hid ** -0.5 * 0.5).bfloat16())
        self.aux = torch.zeros(())
        self.register_buffer("s_mean", torch.tensor(5.0))      # ковзне середнє здивування (не з цього батчу)
        # навчання хвилями (Ідея 4, Стефан): навчений експерт перестає вчитись і час від часу коротко підказує
        # ненавченим (до `students` учнів на вчителя); якщо знову почав помилятись — знову учень.
        # wave="err": «навчився» = помилка впала нижче 30% від свого піку; wave="still" = ваги майже не рухаються.
        self.wave, self.wave_every, self.students, self.hint_w = wave, wave_every, students, hint_w
        self.warm = int(__import__("os").environ.get("WAVE_WARM", 500))                   # перші кроки вчаться всі
        self.err_thr = 0.3                 # «мала помилка» = нижче цієї частки від піку
        self.hint_now = False
        if wave:
            for nm in ("err", "err_peak", "err_freeze", "move_peak", "last_move", "hint_avg"):
                self.register_buffer(nm, torch.zeros(E))
            self.register_buffer("learned", torch.zeros(E, dtype=torch.bool))
            if wave == "still":
                self.register_buffer("up_snap", self.up.detach().clone())
        self.hint = torch.zeros(())

    def forward(self, x, surprise=None):
        if MOE_DISP == "onehot":
            return self._forward_onehot(x, surprise)
        xf, xe, wtab, table, C = self._dispatch(x, surprise)
        N, d = xf.shape
        li = getattr(self, "_ld_li", -1)
        if DIAG_LAYER:
            xe = _ld(xe, li, "disp", "a2a" if getattr(self, "ep_W", 1) > 1 else "exp")
        ye = self._experts(xe, wtab, C)
        if DIAG_LAYER and getattr(self, "ep_W", 1) == 1:
            ye = _ld(ye, li, "exp", "post")
        if MOE_GATHER and MOE_DISP == "inv":            # складання + ваги одним кроком, назад — gather
            out = _CombineW.apply(ye.reshape(self.E * C, d), wtab, self._flat_idx, table, N, xf.dtype)
            return x + out.view_as(x)
        ye = ye.reshape(self.E * C, d) * wtab[:, None]
        if MOE_GATHER:                                   # TPU: складання без scatter — кожен токен бере свої місця
            fi = self._flat_idx
            yp = torch.cat([ye.to(xf.dtype), ye.new_zeros(1, d, dtype=xf.dtype)])
            out = yp[fi.clamp(max=self.E * C)].sum(1)
        else:
            out = xf.new_zeros(N + 1, d).index_add(0, table, ye.to(xf.dtype))[:N]     # складання — у fp32
        return x + out.view_as(x)

    def _forward_onehot(self, x, surprise):
        """MOE_DISP=onehot (GShard): групи по g слів, у групі експерт має Cg місць; номер у черзі — cumsum по групі
        (порядок пар (слово, вибір) той самий, що в _route). Розсилка/складання — einsum з one-hot маскою."""
        xf, keep, w, top_e, oh, C = self._route(x, surprise)
        N, d = xf.shape
        E = self.E
        g = min(MOE_GROUP, N) if MOE_GROUP > 0 else N
        G = -(-N // g)
        Np = G * g
        Cg = -(-C // G)                                   # G·Cg ≥ C: та сама (або трохи більша) місткість
        Ct = G * Cg
        li = getattr(self, "_ld_li", -1)
        dv = x.device
        xd = xf.to(torch.get_autocast_dtype(dev.TYPE)) if torch.is_autocast_enabled() else xf
        kk = oh.shape[1]
        if MOE_IDX32:                                     # лічильники місць — int32 (TPU: int64 емулюється, удвічі байтів)
            oh = oh.to(torch.int32)
        with torch.no_grad():
            ohp = F.pad(oh, (0, 0, 0, 0, 0, Np - N)).view(G, g * kk, E)          # доповнення — без виборів
            rank = (ohp.cumsum(1) - 1).view(G, g, kk, E)                          # номер у черзі експерта в групі
            place = ohp.view(G, g, kk, E).bool() & (rank < Cg)
            self.dropped = 1 - place.sum() / keep.sum().clamp(min=1)
            # експерти в top-k різні → для (слово, експерт) не більше одного вибору: сума по k = вибір
            pe = place.any(2)                                                     # (G,g,E)
            re = (rank * place).sum(2)                                            # (G,g,E) місце в групі
            D = (re[..., None] == torch.arange(Cg, device=dv, dtype=rank.dtype)) & pe[..., None]    # (G,g,E,Cg) one-hot
            D = D.to(xd.dtype)
        wp = F.pad(w, (0, 0, 0, Np - N)).view(G, g, kk, 1)
        we = (wp * place.to(wp.dtype)).sum(2)                                     # (G,g,E) вага слова в експерті
        # вага кожного місця (E·Ct,): у місці не більше одного слова, тож сума по s — це його вага (поелементно, дешево)
        wtab = (D * we[..., None].to(D.dtype)).sum(1).permute(1, 0, 2).reshape(E * Ct)
        xg = F.pad(xd, (0, 0, 0, Np - N)).view(G, g, d)
        xe = torch.einsum("gsec,gsd->egcd", D, xg).reshape(E, Ct, d)             # розсилка множенням
        if DIAG_LAYER:
            xe = _ld(xe, li, "disp", "a2a" if getattr(self, "ep_W", 1) > 1 else "exp")
        ye = self._experts(xe, wtab, Ct)
        if DIAG_LAYER and getattr(self, "ep_W", 1) == 1:
            ye = _ld(ye, li, "exp", "post")
        # ваги — на місця (як ye·wtab у старому шляху), складання — з тією ж 0/1 маскою D: назад лише 1 великий einsum
        # (dye), а не 2 (dye і dCw), і маска Cw (G,g,E,Cg) не зберігається
        yw = ye * wtab.view(E, Ct, 1).to(ye.dtype)
        out = torch.einsum("gsec,egcd->gsd", D.to(ye.dtype), yw.view(E, G, Cg, d))     # складання множенням
        out = out.reshape(Np, d)[:N].to(xf.dtype)
        return x + out.view_as(x)

    def _route(self, x, surprise):
        """Роутер: нормоване xf (N,d), маска виборів keep, ваги w (N,kmax), експерти top_e, one-hot oh, місткість C."""
        N, d = x.shape[0] * x.shape[1], x.shape[-1]
        xf = self.norm(x).reshape(N, d)
        logits = self.router(xf).float()
        probs = logits.softmax(-1)
        # здивування < 0 позначає порожнє місце (якорів у тексті менше за стелю) — воно не йде до експертів
        real = surprise.reshape(N) >= 0 if surprise is not None else torch.ones(N, dtype=torch.bool, device=x.device)
        if self.budget and surprise is not None:
            s = surprise.reshape(N).float().clamp(min=0)
            # ділимо на середнє з минулих кроків: бюджет слова не залежить від майбутніх слів
            k_tok = (self.k_avg * s / self.s_mean.clamp(min=1e-6)).round().clamp(1, self.k_max).long()
            if self.training:
                with torch.no_grad():
                    self.s_mean.mul_(0.99).add_(0.01 * (s * real).sum() / real.sum().clamp(min=1))
        else:
            k_tok = torch.full((N,), self.k_avg, device=x.device, dtype=torch.long)
        k_tok = k_tok * real
        top_p, top_e = probs.topk(self.k_max, -1)                               # (N,kmax)
        keep = torch.arange(self.k_max, device=x.device)[None] < k_tok[:, None]
        w = (top_p * keep) / (top_p * keep).sum(-1, keepdim=True).clamp(min=1e-9)
        # розсилка з фіксованою місткістю (як GShard): усі експерти — одним множенням, без циклу і розривів графа
        # місць у кожного експерта: від очікуваної кількості СПРАВЖНІХ якорів (порожні місця не рахуються), запас ×1.25
        cf = 1.6 if self.budget else 1.25           # з бюджетом якорю дістається до k_max експертів — запас більший
        C = int(cf * self.fill * N * self.k_avg / self.E) + 1
        if MOE_CAP_ALIGN:                                # прес TPU — плитки 256: місця експерта кратні плитці (1025 → 1024)
            C = max(MOE_CAP_ALIGN, round(C / MOE_CAP_ALIGN) * MOE_CAP_ALIGN)
        e_flat = top_e.masked_fill(~keep, self.E)                                # E = «нікуди»
        oh = F.one_hot(e_flat, self.E + 1)[..., :self.E]                         # (N,kmax,E)
        # навантаження: щоб експерти не простоювали (як у Switch Transformer)
        frac = oh.sum((0, 1)).float()
        pm = (probs * real[:, None]).sum(0) / real.sum().clamp(min=1)
        self.aux = self.E * (frac / frac.sum().clamp(min=1) * pm).sum()
        return xf, keep, w, top_e, oh, C

    def _dispatch(self, x, surprise):
        """Роутер + розсилка якорів по місцях експертів: xe (E, C, d), ваги місць wtab, таблиця місць → якір."""
        xf, keep, w, top_e, oh, C = self._route(x, surprise)
        N, d = xf.shape
        e_flat = top_e.masked_fill(~keep, self.E)
        rank = (oh.reshape(-1, self.E).cumsum(0) - 1).view_as(oh)                # номер якоря в черзі експерта
        place = (oh.bool() & (rank < C))                                          # влізло в місткість
        self.dropped = 1 - place.sum() / keep.sum().clamp(min=1)                 # частка, якій не вистачило місця
        slot = (torch.arange(self.E, device=x.device) * C)[None, None] + rank    # (N,kmax,E) місце в загальній таблиці
        flat_idx = torch.where(place, slot, torch.full_like(slot, self.E * C)).amin(-1)   # одне місце на (якір, вибір)
        tok = torch.arange(N, device=x.device)[:, None].expand(-1, self.k_max)
        ok = flat_idx < self.E * C
        # розсилка в bf16 (як решта множень під autocast): удвічі менше байтів на копіювання якорів
        xd = xf.to(torch.get_autocast_dtype(dev.TYPE)) if torch.is_autocast_enabled() else xf
        if MOE_GATHER:
            # TPU: без scatter. Місця експерта заповнюються підряд у порядку появи (rank), тож після стабільного
            # сортування пар (якір, вибір) за експертом місце (e, r) — це елемент «початок e + r» списку.
            self._flat_idx = flat_idx
            ef = e_flat.reshape(-1)
            order = torch.sort(ef, stable=True).indices
            cnt = oh.sum((0, 1))                                                    # (E,) черга кожного експерта
            start = torch.cumsum(cnt, 0) - cnt
            r = torch.arange(C, device=x.device)
            src = (start[:, None] + r[None]).clamp(max=ef.numel() - 1)              # (E, C) місце у відсортованому
            valid = (r[None] < cnt[:, None]).reshape(-1)
            item = order[src.reshape(-1)]
            table = torch.where(valid, tok.reshape(-1)[item], N)
            wtab = torch.where(valid, (w * ok).reshape(-1)[item], 0).to(xd.dtype)
        else:
            table = torch.full((self.E * C + 1,), N, device=x.device, dtype=torch.long)   # N = порожнє місце
            table = table.scatter(0, flat_idx.reshape(-1).clamp(max=self.E * C), torch.where(ok, tok, N).reshape(-1))[:-1]
            wtab = torch.zeros(self.E * C + 1, device=x.device, dtype=xd.dtype)
            wtab = wtab.scatter(0, flat_idx.reshape(-1).clamp(max=self.E * C), (w * ok).reshape(-1).to(xd.dtype))[:-1]
        if MOE_GATHER and MOE_DISP == "inv":
            EC = self.E * C
            if _MOE_INV_CHECK:                           # кожен вибір, що влiз, і його місце вказують одне на одного
                okf = flat_idx < EC
                assert (table[flat_idx[okf]] == tok[okf]).all()
                assert int((table < N).sum()) == int(okf.sum())
            xe = _SlotGather.apply(xd, table, flat_idx, EC).view(self.E, C, d)
            return xf, xe, wtab, table, C
        xpad = torch.cat([xd, xd.new_zeros(1, d)])
        xe = xpad[table].view(self.E, C, d)
        return xf, xe, wtab, table, C

    def _experts(self, xe, wtab, C):
        """Робота експертів над своїми місцями: (E, C, d) → (E, C, d)."""
        up, down = self.up, self.down
        if self.wave:                                    # навчені експерти працюють, але не вчаться
            fr = self.learned[:, None, None]
            up, down = torch.where(fr, up.detach(), up), torch.where(fr, down.detach(), down)
        if self.n_frozen:                                # банк заморожених: окреме множення, без градієнтів ваг
            Ea = self.E - self.n_frozen
            g, u = torch.bmm(xe[:Ea], up.type_as(xe)).chunk(2, -1)
            ya = torch.bmm(F.silu(g) * u, down.type_as(xe))
            g, u = torch.bmm(xe[Ea:], self.up_f.type_as(xe)).chunk(2, -1)
            ye = torch.cat([ya, torch.bmm(F.silu(g) * u, self.down_f.type_as(xe))])
        else:
            g, u = torch.bmm(xe, up.type_as(xe)).chunk(2, -1)
            ye = torch.bmm(F.silu(g) * u, down.type_as(xe))                     # (E,C,d)
        if self.wave and self.training:
            ye = _ErrProbe.apply(ye, self.err)
            if self.hint_now:
                ye_raw = ye
                self.hint = self._hint(xe, ye_raw, wtab.view(self.E, C))
                with torch.no_grad():         # для логу: середня сила підказок саме в кроки візиту вчителя
                    self.hint_avg[0].mul_(0.9).add_(0.1 * self.hint.detach())
            else:
                self.hint = torch.zeros((), device=xe.device)
            self.aux = self.aux + self.hint_w * self.hint
        return ye

    def _hint(self, xe, ye, wt):
        """Учитель коротко підказує учневі: на найважчих якорях учня (де вони розходяться найбільше,
        10% місць) м'яко тягне вихід учня до свого. Без вибірки за маскою — форми сталі."""
        learned = self.learned
        n_t = learned.sum()
        order = (~learned).long().argsort(stable=True)                      # спершу навчені (вчителі)
        rank = (~learned).long().cumsum(0) - 1                              # номер учня
        teacher = order[(rank // self.students) % n_t.clamp(min=1)]         # кожен учитель — до `students` учнів
        with torch.no_grad():
            g, u = torch.bmm(xe, self.up[teacher].type_as(xe)).chunk(2, -1)
            yt = torch.bmm(F.silu(g) * u, self.down[teacher].type_as(xe))
        diff = (ye.float() - yt.float()).pow(2).mean(-1) * (wt > 0)          # (E,C), лише справжні місця
        k = max(1, diff.shape[1] // 10)
        worst = diff.topk(k, dim=1).values.mean(1)                          # де учень помиляється найбільше
        is_student = (~learned) & (n_t > 0)
        return (worst * is_student).sum() / is_student.sum().clamp(min=1)

    @torch.no_grad()
    def before_step(self, step):
        """Поза графом, раз на крок: чи час підказки, і раз на wave_every кроків — хто навчився / знову учень.
        Лише тензорні операції на відеокарті (без зупинок)."""
        if not self.wave:
            return
        self.up._frozen_rows = self.down._frozen_rows = self.learned   # оптимізатор не зсуває навчених за інерцією
        self.hint_now = step > 0 and step % self.wave_every == 0 and bool(self.training)
        if step % self.wave_every or step < self.warm:                      # спершу всі вчаться
            return
        self.err_peak.copy_(torch.maximum(self.err_peak, self.err))
        if self.wave == "err":
            cand = self.err < self.err_thr * self.err_peak
        else:
            move = (self.up - self.up_snap).flatten(1).norm(dim=1) / (self.up_snap.flatten(1).norm(dim=1) + 1e-8)
            self.move_peak.copy_(torch.maximum(self.move_peak, torch.where(self.learned, 0.0, move)))
            self.last_move.copy_(move)
            cand = (move < 0.3 * self.move_peak) & ~self.learned
            self.up_snap.copy_(self.up)
        relearn = self.learned & (self.err > 1.5 * self.err_freeze)        # знову почав помилятись
        newly = ~self.learned & cand
        self.err_freeze.copy_(torch.where(newly, self.err, self.err_freeze))
        self.learned.copy_((self.learned | newly) & ~relearn)


# INDUCT_POS=1: вектор кандидата замка — з уже порахованого emb(x) за позицією (one-hot матмул), без gather зі словника
INDUCT_POS = os.environ.get("INDUCT_POS", "1" if dev.TYPE in ("tpu", "xla") else "") == "1"
# книжка перечитування за номерами слів (див. forward); 0 — як було (тензор book(x))
REREAD_IDS = os.environ.get("REREAD_IDS", "1") == "1"


def _book_in(book):
    """Вхід книжки для перечитування: (номери слів, таблиця) як є, або старий тензор book(x) у fp32."""
    if isinstance(book, tuple):
        return book[0].contiguous(), book[1]
    return book.float().contiguous()
_RR_BISECT = set(filter(None, os.environ.get("RR_BISECT", "").split(",")))   # пошук збою XLA у перечитуванні
RESID_BF16 = os.environ.get("RESID_BF16") == "1"
MEM_BF16 = os.environ.get("MEM_BF16") == "1"        # множення пам'яті й перечитування в bf16 (стан карток лишається fp32)   # експеримент: залишковий потік легкої частини в bf16, а не fp32
CE_CHUNK = int(os.environ.get("CE_CHUNK", "0"))   # >0: вихід на словник шматками словника, логіти не матеріалізуються
# CE_SAVE=1: bf16-логіти шматків зберігаються з прямого проходу — зворотний не повторює множення на весь словник
# (бітово те саме: і там, і там логіт — bf16-вихід того самого множення); ціна — N×V×2 байт пам'яті (8192×131k ≈ 2.1 ГБ)
CE_SAVE = os.environ.get("CE_SAVE") == "1"
# REREAD_LEAN=1: перечитування без зайвих pad / перестановки / clone на вході (див. _reread_sep)
REREAD_LEAN = os.environ.get("REREAD_LEAN", "1") == "1"
REC_LEAN = os.environ.get("REC_LEAN", "1") == "1"     # перечитування: проєкція карток до вибірки
# SHARD_VOCAB=1 (hyperop/shard.py ставить short_logp_vocab): дворівневий вихід на порізаній голові без збирання таблиці
_SHORT_LOGP_HOOK = None
# SURP_INGRAPH=1: здивування (logsumexp легкої голови шматками словника) — однією скомпільованою програмою, а не
# ~40 eager-операціями на крок між скомпільованими шарами. На TPU з батчем 2 (m14/V5) eager-зріз Wb[c0:c0+CE_CHUNK]
# (16768,128) потрапляв у кешовану as_strided-програму torch_tpu, зібрану під буфер (33536,128) — E0102; у графі
# зрізів поза програмою немає. Рахунок той самий (ті самі операції й порядок).
SURP_INGRAPH = os.environ.get("SURP_INGRAPH") == "1"
# CE_SLICE_CLONE=1: шматок таблиці слів у chunk-CE — окремим буфером (.clone()), а не видом зі зсувом. На TPU
# torch_tpu відтворює вид кешованою as_strided-програмою, а при батчі >1 вона отримує базовий буфер іншого
# розміру (E0102 «expected bf16[33536,128] got bf16[16768,128]», знайдено DIAG_ASTRIDED у m16/X0). Значення ті самі.
CE_SLICE_CLONE = os.environ.get("CE_SLICE_CLONE") == "1"


def _wsl(W, a, b):
    return W[a:b].clone() if CE_SLICE_CLONE else W[a:b]

_SURP_FN = None
# EXP_SAVE_CONTIG=1: експертний FFN relu² своєю autograd.Function. На TPU (m16/X0, DIAG_ASTRIDED) ~100 as_strided-програм
# на крок — це транспоновані вигляди активацій експертів (hᵀ (5,3840,2048), xᵀ (5,2560,2048)) зі зворотного проходу
# bmm, які SplitCompiler torch_tpu виносить за межу програми при all-to-all і матеріалізує окремими програмами
# (63 мс/крок у профілі V8). Тут транспонування активацій — суцільними копіями в тому ж графі; зберігається лише
# a = x·up (а не relu і relu² окремо, як робить AOT): −78 МБ на шар. Рахунок побітово той самий (ті самі bmm у тому ж порядку).
EXP_SAVE_CONTIG = os.environ.get("EXP_SAVE_CONTIG") == "1"
# EXP_SAVE_Q8=1 (разом з EXP_SAVE_CONTIG): збережене a = x·up лежить в int8 з масштабом на рядок — ТА САМА формула, що в
# hyperop ACT_Q8 (_ActQ8.pack), але всередині графа (XLA), а не хуком saved_tensors: хуки під torch.compile виконуються
# eager на кожен збережений тензор (перевірено на CPU: pack викликається щоразу для кожного тензора) — на torch_tpu це
# тисячі окремих програм на крок (m17/Y1: 8915 мс при 2×8192). Економія: −39 МБ на шар (78 → 39) на 8192 слів.
EXP_SAVE_Q8 = os.environ.get("EXP_SAVE_Q8") == "1"


class _Relu2FFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up, down):
        a = torch.bmm(x, up)
        y = torch.bmm(F.relu(a).square(), down)
        if EXP_SAVE_Q8:
            af = a.float()
            s = af.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127.0
            q = torch.round(af / s).clamp(-127, 127).to(torch.int8)
            ctx.save_for_backward(x, q, s.to(torch.bfloat16), up, down)
            ctx.a_dtype = a.dtype
        else:
            ctx.save_for_backward(x, a, up, down)
        return y

    @staticmethod
    def backward(ctx, g):
        if EXP_SAVE_Q8:
            x, q, s, up, down = ctx.saved_tensors
            a = (q.float() * s.float()).to(ctx.a_dtype)
        else:
            x, a, up, down = ctx.saved_tensors
        r = F.relu(a)
        h = r.square()
        # TPU: транспонована активація — суцільна копія в графі (XLA це лише розкладка, той самий dot); на CPU/GPU —
        # вигляд, як у autograd (ядро bmm з суцільним операндом округлює інакше → перевірки рівності побітові лише так)
        tT = (lambda t: t.transpose(1, 2).contiguous()) if dev.TPU else (lambda t: t.transpose(1, 2))
        ddown = torch.bmm(tT(h), g)
        dh = torch.bmm(g, down.transpose(1, 2))
        da = torch.where(r > 0, dh * (2 * r), torch.zeros_like(dh))     # як autograd: pow → relu
        dx = torch.bmm(da, up.transpose(1, 2))
        dup = torch.bmm(tT(x), da)
        return dx, dup, ddown


def _surp_lse_impl(lhd, w0, W2, x):
    """Здивування легкої голови: (B, T−1) = lse − логіт правильного слова; lhd (B,T,d) без градієнта, w0 (128,d),
    W2 (V,128), x (B,T) слова. Зрізи (lhd[:, :T−1], x[:, 1:]) — усередині графа."""
    T = x.shape[1]
    h128 = F.linear(lhd[:, :T - 1], w0)
    Bq, Tq = h128.shape[:2]
    hb, Wb = h128.reshape(Bq * Tq, -1).to(torch.bfloat16), W2.to(torch.bfloat16)
    lse = torch.logsumexp(torch.stack([torch.logsumexp((hb @ Wb[c0:c0 + CE_CHUNK].t()).float(), -1)
                                       for c0 in range(0, W2.shape[0], CE_CHUNK)], -1), -1)
    tg = x[:, 1:].reshape(-1)
    ly = (hb.float() * Wb[tg].float()).sum(-1)
    return (lse - ly).view(Bq, Tq)


def _surp_fn():
    global _SURP_FN
    if _SURP_FN is None:
        _SURP_FN = _surp_lse_impl if os.environ.get("NO_COMPILE") else torch.compile(_surp_lse_impl, dynamic=False)
    return _SURP_FN


class _ChunkCE(torch.autograd.Function):
    """logsumexp і логіт правильного слова для h @ Wᵀ + b шматками словника по `ch` слів: уперед і назад логіти
    живуть лише шматком (bf16-множення, fp32-накопичення), повна таблиця N×V не пишеться в пам'ять."""

    @staticmethod
    def forward(ctx, h, W, b, y, ch):
        hb, Wb = h.to(torch.bfloat16), W.to(torch.bfloat16)
        V = W.shape[0]
        lses, los = [], []
        i8 = int8mm.INT8 and int8mm.HEAD
        save = CE_SAVE and not i8
        if i8:                                            # INT8_MM: h — масштаб на рядок, W — на рядок слова
            hq, hs = int8mm.fq(hb, 1)
        for c0 in range(0, V, ch):
            if i8 and int8mm.on_real(hb):            # TPU: справжнє int8-множення (JAX через jax_op)
                lo = int8mm._jops()["lin_fwd_f32"](hb, Wb[c0:c0 + ch])
            elif i8:
                wq, ws = int8mm.fq(Wb[c0:c0 + ch], 1)
                lo = (hq @ wq.t()).float() * hs * ws.t()
            else:
                lo = hb @ _wsl(Wb, c0, c0 + ch).t()       # bf16 прямо з множення; fp32 — лише всередині злиття з reduce
            if save:
                los.append(lo)
            lof = lo.float() + b[c0:c0 + ch].float() if b is not None else lo.float()
            lses.append(torch.logsumexp(lof, -1))
        lse = torch.logsumexp(torch.stack(lses, -1), -1)
        # логіт правильного слова — окремим скалярним добутком h·W[y] (рядки вибираються з таблиці, 84 МБ),
        # а не вибіркою з fp32-логітів кожного шматка (на v6e це було ~12 ГБ на крок)
        if i8:                                            # той самий квантований логіт, що й у lse
            wyq, wys = int8mm.fq(Wb[y], 1)
            ly = (hq.float() * wyq.float()).sum(-1) * hs[:, 0] * wys[:, 0]
        else:
            ly = (hb.float() * Wb[y].float()).sum(-1)
        if b is not None:
            ly = ly + b[y].float()
        ctx.save_for_backward(hb, Wb, b, y, lse, *los)
        ctx.ch, ctx.h_dtype, ctx.W_dtype, ctx.saved_lo = ch, h.dtype, W.dtype, bool(los)
        return lse, ly

    @staticmethod
    def backward(ctx, g_lse, g_ly):
        hb, Wb, b, y, lse, *los = ctx.saved_tensors
        ch, V = ctx.ch, Wb.shape[0]
        dh = torch.zeros(hb.shape, device=hb.device, dtype=torch.float32)
        dW, db = [], []
        i8 = int8mm.INT8 and int8mm.HEAD
        if i8:
            hq, hs = int8mm.fq(hb, 1)
        for ci, c0 in enumerate(range(0, V, ch)):
            if los:
                lo = los[ci].float()                      # CE_SAVE: логіти з прямого проходу
            elif i8 and int8mm.on_real(hb):
                lo = int8mm._jops()["lin_fwd_f32"](hb, Wb[c0:c0 + ch])
            elif i8:
                wq, ws = int8mm.fq(Wb[c0:c0 + ch], 1)
                lo = (hq @ wq.t()).float() * hs * ws.t()
            else:
                lo = (hb @ _wsl(Wb, c0, c0 + ch).t()).float()
            if b is not None:
                lo = lo + b[c0:c0 + ch].float()
            n = lo.shape[1]                               # (fp32 тут лише поелементно — XLA зливає з exp і множенням)
            hit = torch.arange(c0, c0 + n, device=y.device)[None] == y[:, None]
            G = g_lse[:, None] * (lo - lse[:, None]).exp() + g_ly[:, None] * hit   # ∂/∂логітів
            Gb = G.to(torch.bfloat16)
            if i8 and int8mm.on_real(Gb):
                dh = dh + int8mm._jops()["lin_dgrad_f32"](Gb, Wb[c0:c0 + n])
            elif i8:                                      # градієнт за h — int8 (G — на рядок, W — на стовпець d)
                gq, gs = int8mm.fq(Gb, 1)
                wcq, wcs = int8mm.fq(Wb[c0:c0 + n], 0)
                dh = dh + (gq @ wcq).float() * gs * wcs
            else:
                dh = dh + (Gb @ _wsl(Wb, c0, c0 + n)).float()
            # bf16-таблиця: множення накопичує у fp32 й округлює раз — те саме, що .float() і потім .to(bf16), але без
            # fp32-копії градієнта всієї таблиці (1.3 ГБ запису + читання на крок)
            dW.append((Gb.t() @ hb) if ctx.W_dtype == torch.bfloat16 else (Gb.t() @ hb).float())
            if b is not None:
                db.append(G.sum(0))
        return (dh.to(ctx.h_dtype), torch.cat(dW).to(ctx.W_dtype),
                torch.cat(db).to(b.dtype) if b is not None else None, None, None)


def chunk_ce(h, W, b, y, ch=CE_CHUNK):
    """(lse, логіт правильного) для кожного рядка h; h (N,d), W (V,d), b (V,) або None, y (N,)."""
    return _ChunkCE.apply(h, W, b, y, ch)


class _ContigGrad(torch.autograd.Function):
    """Тотожність уперед; назад — суцільна копія градієнта (TPU: вигляд-градієнт на вході скомпільованого
    зворотного проходу програма не приймає, E0102)."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.contiguous().clone()


class _ErrProbeSeg(torch.autograd.Function):
    """Як _ErrProbe, але рядки йдуть підряд за експертом (grouped_mm): середня норма градієнта по рядках кожного
    експерта — множенням на one-hot (MXU), без scatter."""

    @staticmethod
    def forward(ctx, y, buf, oh_t):
        ctx.buf, ctx.oh_t = buf, oh_t
        return y.view_as(y)

    @staticmethod
    def backward(ctx, g):
        with torch.no_grad():
            n = ctx.oh_t @ g.float().norm(dim=-1, keepdim=True)              # (E,1) сума норм
            cnt = ctx.oh_t.sum(1, keepdim=True).clamp(min=1)
            ctx.buf.mul_(0.9).add_(0.1 * (n / cnt).squeeze(1))
        return g, None, None


class BankMoE(BudgetMoE):
    _ep_ok = True
    """MoE з хвилями «перестав мінятись» і справжнім заморожений банком (ціна «1 до N» у тренуванні).

    Два банки: «вчаться» (параметр fp32 з оптимізатором) і «заморожені» (bf16-буфер: без градієнтів ваг, без
    оптимізатора й без щокрокового перетворення). Роутер бачить усіх E експертів, як і раніше. Навчені переїжджають
    у заморожений банк групами раз на `rebank_every` кроків (одночасно в усіх шарах — щоб граф перекомпілювався
    один раз), а той, хто знову почав помилятись, повертається. Підказок учителя немає (у тесті вони нічого не дали).
    """

    def __init__(self, d, E=8, **kw):
        import os
        # MOE_K=4: кожен якір іде до k експертів, кожен — у k/2 рази вужчий (активних обчислень стільки ж, що при 2):
        # рядків на експерта в k/2 разів більше, а ваг у кожного в k/2 менше — множення стає менш пам'ятно-обмеженим
        k = int(os.environ.get("MOE_K", 2))
        super().__init__(d, E=E, k_avg=k, k_max=max(4, k), budget=False, wave="still", hint_w=0.0, **kw)
        hid = 2 * d * 2 // k
        self.L, self.Fz = list(range(E)), []                  # хто вчиться / хто заморожений (номери експертів)
        self.has_L, self.has_F = True, False
        self.eager_experts = False                            # True — поза графом: без перекомпіляцій, але крок +28%
        self.register_buffer("L_idx", torch.arange(E))
        self.register_buffer("F_idx", torch.zeros(0, dtype=torch.long))
        self.register_buffer("up_f", torch.zeros(0, d, 2 * hid, dtype=torch.bfloat16))
        self.register_buffer("down_f", torch.zeros(0, hid, d, dtype=torch.bfloat16))
        # BANK_Q=int8 / int4: заморожений банк зберігається вузько (масштаб на вихідний стовпець), перед множенням
        # розгортається в bf16 — ваги читаються з HBM у 2 / 4 рази менше байтів (банк пам'ятно-обмежений)
        self.bank_q = os.environ.get("BANK_Q", "")
        self.register_buffer("up_fs", torch.zeros(0, 1, 2 * hid, dtype=torch.bfloat16))
        self.register_buffer("down_fs", torch.zeros(0, 1, d, dtype=torch.bfloat16))
        # BF16_EXPERTS=1: ваги «вчаться» теж у bf16 (точна fp32-копія — в оптимізаторі): без щокрокового
        # перетворення fp32→bf16 і з удвічі легшими градієнтами — це ~90% усіх параметрів моделі
        self.w_dtype = torch.bfloat16 if os.environ.get("BF16_EXPERTS") == "1" else torch.float32
        # MOE_ACT=relu2: експерт relu(x·up)² · down (як у modded-nanogpt) замість SwiGLU; ширина 1.5× — параметрів
        # і обчислень стільки ж, але без гейта (менше поелементної роботи й байтів активацій)
        self.relu2 = os.environ.get("MOE_ACT") == "relu2"
        if self.relu2:
            h2 = 3 * hid // 2
            self.up = nn.Parameter(torch.randn(_ep_rows(E), d, h2) * d ** -0.5)
            self.down = nn.Parameter(torch.randn(_ep_rows(E), h2, d) * h2 ** -0.5 * 0.5)
            self.up_f = torch.zeros(0, d, h2, dtype=torch.bfloat16)
            self.down_f = torch.zeros(0, h2, d, dtype=torch.bfloat16)
            self.up_fs = torch.zeros(0, 1, h2, dtype=torch.bfloat16)
        elif k != 2:                                          # SwiGLU, вужчі експерти під MOE_K
            self.up = nn.Parameter(torch.randn(_ep_rows(E), d, 2 * hid) * d ** -0.5)
            self.down = nn.Parameter(torch.randn(_ep_rows(E), hid, d) * hid ** -0.5 * 0.5)
        self.up = nn.Parameter(self.up.detach().to(self.w_dtype))
        self.down = nn.Parameter(self.down.detach().to(self.w_dtype))
        self.up_snap = self.up.detach().clone()

    def _fq(self, w):
        """Ваги для замороженого банку (float, (F, вхід, вихід)) → (як зберігати, масштаб на вихідний стовпець)."""
        if not self.bank_q:
            return w.bfloat16().contiguous(), w.new_zeros(w.shape[0], 1, w.shape[2]).bfloat16()
        if self.bank_q == "int8":
            sc = (w.abs().amax(1, keepdim=True) / 127).clamp(min=1e-8)
            return (w / sc).round().clamp(-127, 127).to(torch.int8).contiguous(), sc.bfloat16()
        # int4: масштаб на групу з 128 рядків входу (один на стовпець — похибка ~20%), пари рядків → один байт
        F_, K, N = w.shape
        wg = w.view(F_, K // 128, 128, N)
        sc = (wg.abs().amax(2, keepdim=True) / 7).clamp(min=1e-8)
        q = (wg / sc).round().clamp(-7, 7).to(torch.int8).view(F_, K, N)
        q = ((q[:, 0::2] & 0x0F) | (q[:, 1::2] << 4)).to(torch.int8)
        return q.contiguous(), sc[:, :, 0].bfloat16().contiguous()          # масштаб (F, K/128, N)

    def _ffn_frozen(self, x):
        """Заморожені експерти. int8: множиться на вузькі ваги (int8→bf16 XLA зливає в пролог множення — з HBM
        читається 1 байт на вагу), масштаб стовпця — на вихід (на v6e −34% проти bf16-банку; масштаб до множення
        створює окрему bf16-копію ваг і виграшу немає). Масштаб стовпця up — на вихід першого множення до relu²."""
        if self.bank_q != "int8":
            return self._ffn(x, self._fw("up"), self._fw("down"))
        h = torch.bmm(x, self.up_f.to(x.dtype)) * self.up_fs.to(x.dtype)
        if self.relu2:
            h = F.relu(h).square()
        else:
            g, u = h.chunk(2, -1)
            h = F.silu(g) * u
        return torch.bmm(h, self.down_f.to(x.dtype)) * self.down_fs.to(x.dtype)

    def _fw(self, which):
        """Заморожені ваги «up» / «down» у bf16 (з вузького зберігання — розгортання перед множенням)."""
        q = getattr(self, which + "_f")
        if not self.bank_q:
            return q
        sc = getattr(self, which + "_fs")
        if self.bank_q == "int4":
            F_, K2, N = q.shape
            q = torch.stack([(q << 4) >> 4, q >> 4], 2).reshape(F_, K2 * 2 // 128, 128, N)
            return (q.to(torch.bfloat16) * sc[:, :, None]).reshape(F_, K2 * 2, N)
        return q.to(torch.bfloat16) * sc

    def forward(self, x, surprise=None):
        if MOE_GMM:
            return self._forward_gmm(x, surprise)
        if EP_A2A_DEDUP and MOE_DISP != "onehot" and getattr(self, "ep_W", 1) > 1 and not self.has_F:
            return self._forward_ep_dedup(x, surprise)
        return super().forward(x, surprise)

    def _forward_ep_dedup(self, x, surprise):
        """EP_A2A_DEDUP=1: пара (слово, чип) їде all-to-all ОДИН раз, хоч слово вибрало кількох експертів того чипа
        (top-4 з 40, по 5 на чипі — часто). На чипі-власнику слово розмножується по місцях своїх експертів; назад —
        зважена сума виходів його експертів цього чипа, теж раз на (слово, чип).
        Місткість експертів і відкидання — ті самі (та сама _dispatch). Стеля місць на чип Cc = EP_DEDUP_CAP × El·C
        (EP_DEDUP_CAP=1 — точно як без прапора, бо різних слів на чип не більше за El·C; <1 — менше байтів, а пари
        понад стелю відкидаються ЦІЛИМИ (усі вибори слова на тому чипі) — вони додаються до moe_drop).
        Усе — збиранням (gather/sort/searchsorted), без scatter; форми сталі."""
        from torch.distributed._functional_collectives import all_to_all_single_autograd, all_to_all_single
        import torch.distributed as tdist
        xf, _, wtab, table, C = self._dispatch(x, surprise)        # xe не потрібен (граф його викине)
        N, d = xf.shape
        W, E = self.ep_W, self.E
        El = E // W
        S = El * C                                                  # місць експертів одного чипа
        # різних слів на чип ≤ min(S, N): стеля понад N — лише порожні місця (точність та сама, байтів менше)
        Cc = min(S, N, max(1, math.ceil(EP_DEDUP_CAP * S)))
        if MOE_CAP_ALIGN and Cc < S:
            Cc = min(S, max(MOE_CAP_ALIGN, math.ceil(Cc / MOE_CAP_ALIGN) * MOE_CAP_ALIGN))
        g = tdist.group.WORLD
        dv = x.device
        xd = xf.to(torch.get_autocast_dtype(dev.TYPE)) if torch.is_autocast_enabled() else xf
        with torch.no_grad():
            tb = table.view(W, S)                                   # слово кожного місця, згруповано за чипом
            Wi = torch.arange(W, device=dv)[:, None]
            # які слова їдуть на чип s: прапор (W, N+1) через порівняння відсортованих (без scatter)
            ts = torch.sort(tb, dim=1).values                       # (W,S)
            nn_ = torch.arange(N, device=dv)[None].expand(W, -1).contiguous()
            lo = torch.searchsorted(ts, nn_)
            hi = torch.searchsorted(ts, nn_, right=True)
            hit = hi > lo                                            # (W,N) слово n має місце на чипі s
            cnt = hit.sum(1)                                         # різних слів на чип
            pos = hit.long().cumsum(1) - 1                           # місце слова в посилці на чип s
            pos = torch.where(hit & (pos < Cc), pos, Cc)            # Cc = «не їде» (немає або понад стелю)
            order = torch.sort((~hit).long(), dim=1, stable=True).indices   # спершу слова, що їдуть, у порядку n
            q = torch.arange(Cc, device=dv)[None]
            if Cc > N:                                               # вирівнювання MOE_CAP_ALIGN вийшло за N
                order = torch.cat([order, order.new_full((W, Cc - N), N)], 1)
            ctok = torch.where(q < cnt[:, None], order[:, :Cc], N)  # (W,Cc) слово кожного місця посилки
            # місце посилки для кожного місця експерта; порожнє місце або слово понад стелю → Cc
            posp = torch.cat([pos, pos.new_full((W, 1), Cc)], 1)    # слово N (порожнє) → Cc
            m = posp.gather(1, tb)                                   # (W,S)
            lost = ((m == Cc) & (tb < N)).sum()                     # вибори, відкинуті стелею Cc
            placed = (tb < N).sum().clamp(min=1)
            # обернена таблиця для складання: місця експертів кожного місця посилки (до K = min(k_max, El))
            K = min(self.k_max, El)
            ms, mo = torch.sort(m, dim=1, stable=True)
            st = torch.searchsorted(ms, q.expand(W, -1).contiguous())        # (W,Cc)
            en = torch.searchsorted(ms, q.expand(W, -1).contiguous(), right=True)
            t = torch.arange(K, device=dv)
            ii = (st[..., None] + t).clamp(max=S - 1)                 # (W,Cc,K)
            inv = torch.where(st[..., None] + t < en[..., None], mo.gather(1, ii.view(W, -1)).view(W, Cc, K), S)
        self.dropped = 1 - (1 - self.dropped) * (placed - lost) / placed
        if not hasattr(self, "_dd_stat"):
            self._dd_stat = {}
        self._dd_stat.update(uniq=cnt.detach(), picks=(tb < N).sum(1).detach(), S=S, Cc=Cc)
        # посилка: (W, Cc, d) слів + метадані (номер місця посилки для місць експертів, обернена таблиця, ваги)
        xpad = torch.cat([xd, xd.new_zeros(1, d)])
        xs = xpad[ctok.reshape(-1)]                                  # (W·Cc, d)
        xr = all_to_all_single_autograd(xs.contiguous(), None, None, g).view(W, Cc, d)
        mr = all_to_all_single(m.reshape(-1).contiguous(), None, None, g).view(W, S)
        invr = all_to_all_single(inv.reshape(-1).contiguous(), None, None, g).view(W, Cc, K)
        wr = all_to_all_single_autograd(wtab.reshape(-1).contiguous(), None, None, g).view(W, S)
        # на власнику: розмножити слово по місцях своїх експертів (W джерел × El експертів × C місць)
        xrp = torch.cat([xr, xr.new_zeros(W, 1, d)], 1)             # Cc — порожнє місце
        xe = xrp.gather(1, mr[..., None].expand(-1, -1, d))         # (W,S,d)
        xe = xe.view(W, El, C, d).transpose(0, 1).reshape(El, W * C, d)
        y = self._ffn(xe, self.up, self.down)                        # (El, W·C, d)
        y = y.view(El, W, C, d).transpose(0, 1).reshape(W, S, d)
        # зважена сума виходів експертів цього чипа для кожного (слово, чип) — у fp32, назад у типі посилки
        yw = y.float() * wr[..., None].float()
        ywp = torch.cat([yw, yw.new_zeros(W, 1, d)], 1)             # S — порожнє
        yc = ywp.gather(1, invr.view(W, Cc * K, 1).expand(-1, -1, d)).view(W, Cc, K, d).sum(2)
        yb = all_to_all_single_autograd(yc.to(xd.dtype).reshape(W * Cc, d).contiguous(), None, None, g).view(W, Cc, d)
        # у себе: слово бере свої місця посилок з усіх чипів
        ybp = torch.cat([yb, yb.new_zeros(W, 1, d)], 1).to(xf.dtype)
        out = ybp[Wi.expand(-1, N), pos].sum(0)                      # (N,d)
        if self.training:
            out = _ErrProbeTok.apply(out, self.err, table, wtab, E, C)
        return x + out.view_as(x)

    def _forward_gmm(self, x, surprise):
        N, d = x.shape[0] * x.shape[1], x.shape[-1]
        E, k = self.E, self.k_avg
        xf = self.norm(x).reshape(N, d)
        probs = self.router(xf).float().softmax(-1)
        real = surprise.reshape(N) >= 0 if surprise is not None else torch.ones(N, dtype=torch.bool, device=x.device)
        top_p, top_e = probs.topk(k, -1)                                          # (N,k)
        w = top_p / top_p.sum(-1, keepdim=True).clamp(min=1e-9) * real[:, None]
        e_flat = top_e.masked_fill(~real[:, None], E).reshape(-1)                 # (M,) E = «нікуди»
        M = e_flat.numel()
        oh = F.one_hot(e_flat, E + 1)                                             # (M,E+1)
        cnt = oh.sum(0)                                                           # черга кожного експерта
        frac = cnt[:E].float()
        pm = (probs * real[:, None]).sum(0) / real.sum().clamp(min=1)
        self.aux = E * (frac / frac.sum().clamp(min=1) * pm).sum()
        self.dropped = torch.zeros((), device=x.device)                           # відкидань немає
        # місце пари у впорядкованому списку = початок її експерта + її номер у черзі (без сортування)
        start = torch.cumsum(cnt, 0) - cnt
        rank = (oh.cumsum(0) * oh).sum(-1) - 1
        pos = start[e_flat] + rank                                                # (M,) перестановка
        order = torch.empty_like(pos).scatter_(0, pos, torch.arange(M, device=x.device))   # пара на місці i
        tok = order // k
        xd = xf.to(torch.get_autocast_dtype(dev.TYPE)) if torch.is_autocast_enabled() else xf
        offs = torch.cumsum(cnt[:E], 0).to(torch.int32)
        # хвіст («нікуди») grouped_mm не рахує — сміття і вперед, і назад; маска відрізає його в обидва боки
        live = (torch.arange(M, device=x.device) < offs[-1])[:, None]
        xs = torch.where(live, xd[tok], 0)                                        # (M,d) за експертом
        # where, а не множення на 0: хвіст виходу grouped_mm не визначений (може бути NaN, а 0·NaN = NaN)
        ys = torch.where(live, self._ffn_gmm(xs, offs), 0)
        if self.training:
            es = e_flat[order]
            oh_t = F.one_hot(es, E + 1)[:, :E].t().float()                       # (E,M)
            ys = _ErrProbeSeg.apply(ys, self.err, oh_t)
        y = ys[pos].view(N, k, d) * w.reshape(N, k, 1).to(ys.dtype)               # назад у порядок якорів
        return x + y.sum(1).to(x.dtype).view_as(x)

    def _ffn_gmm(self, xs, offs):
        """Усі експерти одним grouped_mm; банки «вчаться»/«заморожені» складаються в одну таблицю за номером."""
        if self.has_F:
            dt = xs.dtype
            up = torch.zeros(self.E, *self.up.shape[1:], device=xs.device, dtype=dt)
            up = up.index_copy(0, self.L_idx, self.up.to(dt)).index_copy(0, self.F_idx, self._fw("up").to(dt))                 if self.has_L else up.index_copy(0, self.F_idx, self._fw("up").to(dt))
            down = torch.zeros(self.E, *self.down.shape[1:], device=xs.device, dtype=dt)
            down = down.index_copy(0, self.L_idx, self.down.to(dt)).index_copy(0, self.F_idx, self._fw("down").to(dt))                 if self.has_L else down.index_copy(0, self.F_idx, self._fw("down").to(dt))
        else:
            up, down = self.up.to(xs.dtype), self.down.to(xs.dtype)
        h = torch._grouped_mm(xs, up, offs=offs)
        if self.relu2:
            h = F.relu(h).square()
        else:
            g, u = h.chunk(2, -1)
            h = F.silu(g) * u
        return torch._grouped_mm(h.to(xs.dtype), down, offs=offs)       # grouped_mm поза autocast — тип вручну

    def _ffn(self, x, up, down):
        if self.relu2:
            if int8mm.INT8 and int8mm.EXP:        # INT8_MM=1: експерти в int8 (прямий + градієнт за входом)
                return int8mm.bmm(F.relu(int8mm.bmm(x, up.type_as(x))).square(), down.type_as(x))
            if EXP_SAVE_CONTIG:                   # зворотний прохід без транспонованих виглядів активацій (TPU)
                return _Relu2FFN.apply(x, up.type_as(x), down.type_as(x))
            return torch.bmm(F.relu(torch.bmm(x, up.type_as(x))).square(), down.type_as(x))
        g, u = torch.bmm(x, up.type_as(x)).chunk(2, -1)
        return torch.bmm(F.silu(g) * u, down.type_as(x))

    def _experts(self, xe, wtab, C):
        if self.eager_experts:
            return self._experts_eager(xe, wtab, C)
        return self._experts_body(xe, wtab, C)

    @torch.compiler.disable
    def _experts_eager(self, xe, wtab, C):
        """Поза графом: розміри банків можуть мінятись без перекомпіляції моделі (три множення — і так великі)."""
        return self._experts_body(xe, wtab, C)

    def ep_shard(self, rank, W):
        """Шардинг експертів (EP=1): чип rank тримає лише своїх E/W експертів; слова їдуть до експерта (all-to-all),
        а не ваги до слова. Градієнт експерта — сума з усіх чипів, тому ділимо на W (як усереднення у FSDP/DDP)."""
        assert self.E % W == 0 and not self.has_F, "EP: E має ділитись на кількість чипів; банки — до шардингу"
        El = self.E // W
        sl = slice(rank * El, (rank + 1) * El) if self.up.shape[0] == self.E else slice(0, El)   # EP_ALLOC_LOCAL — уже свої
        self.up = nn.Parameter(self.up.detach()[sl].to(self.w_dtype).clone())     # bf16 (BF16_EXPERTS): FSDP їх не
        self.down = nn.Parameter(self.down.detach()[sl].to(self.w_dtype).clone()) # бачить, тож не перетворює сам
        self.up_snap = self.up.detach().clone()
        self.ep_W, self.ep_rank = W, rank
        for p in (self.up, self.down):
            if not os.environ.get("EP_NOHOOK"):        # SHARD=1 (hyperop): ділення на W робить sync_grads у графі
                p.register_hook(lambda g, W=W: g / W)
            p._ep_local = True

    def _ep_ffn(self, xe):
        """xe (E, C, d) — місця під експертів з МОЇХ слів → чипи власників → (E/W, W·C, d) → назад."""
        from torch.distributed._functional_collectives import all_to_all_single_autograd
        import torch.distributed as tdist
        W, E, (_, C, d) = self.ep_W, self.E, xe.shape
        El = E // W
        g = tdist.group.WORLD
        li = getattr(self, "_ld_li", -1)
        r = all_to_all_single_autograd(xe.contiguous(), None, None, g)          # блок s — слова чипа s для моїх
        if DIAG_LAYER:
            r = _ld(r, li, "a2a", "exp")
        r = r.view(W, El, C, d).transpose(0, 1).reshape(El, W * C, d)
        y = self._ffn(r, self.up, self.down)
        if DIAG_LAYER:
            y = _ld(y, li, "exp", "comb")
        y = y.view(El, W, C, d).transpose(0, 1).reshape(E, C, d).contiguous()
        y = all_to_all_single_autograd(y, None, None, g)                       # блок s — мої слова від експертів s
        if DIAG_LAYER:
            y = _ld(y, li, "comb", "post")                                     # comb = зворотний all-to-all
        return y

    def _experts_body(self, xe, wtab, C):
        # (розміри банків — динамічні виміри графа: переїзд не перебудовує граф; гілки — лише «є/нема заморожених»)
        if getattr(self, "ep_W", 1) > 1:
            ye = self._ep_ffn(xe)
        elif not self.has_F:
            ye = self._ffn(xe, self.up, self.down)
        else:
            parts = []
            if self.has_L:
                parts.append((self.L_idx, self._ffn(xe[self.L_idx], self.up, self.down)))
            parts.append((self.F_idx, self._ffn_frozen(xe[self.F_idx])))
            ye = xe.new_zeros(xe.shape)
            for idx, y in parts:
                ye = ye.index_copy(0, idx, y.to(ye.dtype))
        if self.training:
            ye = _ErrProbe.apply(ye, self.err)
        return ye

    @torch.no_grad()
    def before_step(self, step):
        """Хто навчився («перестав мінятись») — як у BudgetMoE, але рух міряється лише в банку «вчаться»."""
        self.hint_now = False
        if step % self.wave_every or step < self.warm:
            return
        self.err_peak.copy_(torch.maximum(self.err_peak, self.err))
        move = torch.zeros(self.E, device=self.err.device)
        if self.L:
            mL = (self.up.float() - self.up_snap.float()).flatten(1).norm(dim=1) / (self.up_snap.float().flatten(1).norm(dim=1) + 1e-8)
            move = move.index_copy(0, self.L_idx, mL)
            self.up_snap.copy_(self.up)
        self.move_peak.copy_(torch.maximum(self.move_peak, torch.where(self.learned, 0.0, move)))
        self.last_move.copy_(move)
        cand = (move < 0.3 * self.move_peak) & ~self.learned
        in_L = torch.zeros(self.E, dtype=torch.bool, device=move.device).index_fill(0, self.L_idx, True)
        cand &= in_L                                   # заморожені не рухаються — їх «навченість» тримає err
        relearn = self.learned & (self.err > 1.5 * self.err_freeze)
        newly = ~self.learned & cand
        self.err_freeze.copy_(torch.where(newly, self.err, self.err_freeze))
        self.learned.copy_((self.learned | newly) & ~relearn)

    @torch.no_grad()
    def wants_rebank(self, learned_cpu):
        """Чи варто переселяти (на хості, з уже прочитаною маскою): група ≥ E/4 навчених у «вчаться» або хтось
        із заморожених знову учень, або навчились усі, хто лишився."""
        to_freeze = [e for e in self.L if learned_cpu[e]]
        to_thaw = [e for e in self.Fz if not learned_cpu[e]]
        return bool(to_thaw) or len(to_freeze) >= max(1, self.E // 4) or (to_freeze and len(to_freeze) == len(self.L))

    @torch.no_grad()
    def rebank(self, learned_cpu):
        """Переселення між банками. Повертає [(старий параметр, новий, відповідність місць)] — для оптимізатора."""
        to_freeze = {e for e in self.L if learned_cpu[e]}
        to_thaw = {e for e in self.Fz if not learned_cpu[e]}
        newL = sorted((set(self.L) - to_freeze) | to_thaw)
        newF = sorted((set(self.Fz) - to_thaw) | to_freeze)
        changes = []
        for nm, fnm in (("up", "up_f"), ("down", "down_f")):
            old = getattr(self, nm)
            frz = getattr(self, fnm)
            full = torch.empty(self.E, *old.shape[1:], device=old.device, dtype=torch.float32)
            if self.L:
                full[torch.tensor(self.L, device=old.device)] = old.detach().float()
            if self.Fz:
                full[torch.tensor(self.Fz, device=old.device)] = self._fw(nm).float()
            new = nn.Parameter((full[torch.tensor(newL, device=old.device, dtype=torch.long)] if newL else full[:0])
                               .to(self.w_dtype).contiguous())
            setattr(self, nm, new)
            if newF:
                q, sc = self._fq(full[torch.tensor(newF, device=old.device, dtype=torch.long)])
            else:
                q, sc = frz[:0], getattr(self, fnm + "s")[:0]
            setattr(self, fnm, q)
            setattr(self, fnm + "s", sc)
            changes.append((old, new, [self.L.index(e) if e in self.L else None for e in newL]))
        self.L, self.Fz = newL, newF
        self.has_L, self.has_F = bool(newL), bool(newF)
        dv = self.up.device                              # (не «dev» — це ім'я модуля пристрою)
        self.L_idx = torch.tensor(newL, device=dv, dtype=torch.long)
        self.F_idx = torch.tensor(newF, device=dv, dtype=torch.long)
        self.up_snap = self.up.detach().clone()
        for t in (self.up, self.down, self.up_f, self.down_f, self.up_fs, self.down_fs, self.L_idx, self.F_idx):
            # TPU (torch_tpu): лише сталі форми — динамічний вимір валить рантайм (Aborted); перекомпіляція раз на переїзд
            if t.shape[0] > 1 and not dev.TPU:
                torch._dynamo.maybe_mark_dynamic(t, 0)   # кількість експертів у банку — змінна, а не стала графа
        return changes


class HeavyCtxAttn(nn.Module):
    """Важкий шар дивиться у прочитаний текст (стани легкої частини попередніх слів), не на інших якорів.
    Без позицій — впізнавання за змістом («Кирил» знаходить «Кирила», хоч де той був вище)."""

    def __init__(self, d, heads=8):
        super().__init__()
        self.h = heads
        self.nq, self.nk = nn.RMSNorm(d), nn.RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.o.weight)

    def forward(self, a, ctx, idx):
        B, A, D = a.shape
        T = ctx.shape[1]
        hd = D // self.h
        q = self.q(self.nq(a)).view(B, A, self.h, hd).transpose(1, 2)
        k, v = self.kv(self.nk(ctx)).view(B, T, 2, self.h, hd).permute(2, 0, 3, 1, 4)
        mask = (torch.arange(T, device=a.device)[None, None, :] <= idx[:, :, None])[:, None]   # лише назад
        y = F.scaled_dot_product_attention(q, k.type_as(q), v.type_as(q), attn_mask=mask)
        return a + self.o(y.transpose(1, 2).reshape(B, A, D))


class FullLM2(nn.Module):
    def __init__(self, vocab, d=1024, light_a=2, light_b=2, heavy=14, final=2, heads=16, frac=0.2, cap=0.25,
                 window=2048, seg=512, ckpt=False, mem_kw=None, use_memory=True, seg_graph=False,
                 volume=False, moe=None, wave=None, n_exp=8, n_frozen=0, bank=False):
        super().__init__()
        self.vocab = vocab                                       # з доповненням (кратне 64/256); справжніх слів — PHR_V
        self.seg_graph = seg_graph                               # крок пам'яті записаний графом (CUDA graphs)
        self.use_memory = use_memory                             # False — для перевірки, що задача чесна
        cap = float(__import__("os").environ.get("ANCHOR_CAP", cap))          # стеля якорів (частка слів, що йдуть у важкі шари)
        frac = float(__import__("os").environ.get("ANCHOR_FRAC", frac))       # бажана частка якорів (поріг підлаштовується)
        self.frac, self.cap, self.window, self.seg, self.ckpt = frac, cap, window, seg, ckpt
        # CKPT=1: перерахунок активацій усіх шарів у зворотному проході замість зберігання (пам'ять ↓, час ↑ ~на прохід
        # уперед); CKPT=heavy / CKPT=light — лише важкі / лише легкі й фінальні шари
        self.ckpt_kind = __import__("os").environ.get("CKPT", "")
        if self.ckpt_kind == "1":
            self.ckpt = True
        self.emb = nn.Embedding(vocab, d)
        self.book = nn.Embedding(vocab, 128)                     # сирий текст «на полиці»
        self.light_a = nn.ModuleList(FlexBlock(d, heads) for _ in range(light_a))
        # volume: пам'ять v3 — обсяг комірок (шматочки), картки будь-якого розміру
        self.memory = (VolumeMemory if volume else Memory)(d, **(mem_kw or {}))
        self.light_b = nn.ModuleList(FlexBlock(d, heads) for _ in range(light_b))
        self.l_norm = nn.RMSNorm(d)
        self.l_head = nn.Sequential(nn.Linear(d, 128, bias=False), nn.Linear(128, vocab, bias=False))
        # без уваги: лише думають над якорем; moe: "fixed" / "budget" — важкі шари як MoE (Ідея 4)
        self.moe = moe
        self.heavy = nn.ModuleList((BankMoE(d, E=n_exp) if bank else
                                    BudgetMoE(d, E=n_exp, budget=moe == "budget", wave=wave, frozen=n_frozen)) if moe
                                   else FFN(d)
                                   for _ in range(heavy))
        for b in self.heavy:
            b.fill = frac / cap
        self.final = nn.ModuleList(FlexBlock(d, heads) for _ in range(final))
        self.reread = Reread(d)
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        # SHORT_V=8192 (Стефан, 02.11): «короткий словник на кожному слові + повний лише там, де треба». Вихід
        # дворівневий: K найчастіших слів + «скринька рідкісне»; P(рідкісне w) = P(скринька)·P(w | рідкісні).
        # Повний словник рахується лише на місцях, де правильне слово рідкісне (~18%), зі сталою місткістю
        # (основна порція + друга порція для надлишку; RARE_CAP — разом, частка всіх місць)
        import os as _os
        self.short_v = int(_os.environ.get("SHORT_V", 0) or 0)
        if self.short_v:
            import numpy as _np
            cnt = _np.zeros(vocab, _np.int64)
            if _os.environ.get("TOKEN_COUNTS"):
                c0 = _np.load(_os.environ["TOKEN_COUNTS"]).astype(_np.int64)[:vocab]
                cnt[:len(c0)] = c0
            top = torch.from_numpy(_np.argsort(-cnt, kind="stable")[:self.short_v].copy())
            pos = torch.full((vocab,), self.short_v, dtype=torch.long)
            pos[top] = torch.arange(self.short_v)
            self.register_buffer("sv_ids", top, persistent=False)              # слова короткого словника
            self.register_buffer("sv_pos", pos, persistent=False)              # місце слова в короткому (K = рідкісне)
            # «скринька рідкісне» — останній (порожній, доповнення словника) рядок таблиці виходу: оптимізатор вчить
            # його як рядок слова (масштаб на рядок); сталий апріорний зсув = частка рідкісних у корпусі
            self.sv_bucket = vocab - 1
            assert cnt[self.sv_bucket] == 0 and pos[self.sv_bucket] == self.short_v, "останній рядок словника має бути порожнім"
            msk = (pos < self.short_v).float() * -1e4
            msk[self.sv_bucket] = -1e4
            self.register_buffer("sv_mask", msk, persistent=False)        # короткі й скринька — геть з повного
            rare = 1 - cnt[top.numpy()].sum() / max(cnt.sum(), 1) if cnt.sum() else 0.18
            # апріорний логіт скриньки: P(скринька) = частка рідкісних на старті. Без PHRASE_INIT логіти слів нульові →
            # log(частка·K/(1−частка)); з PHRASE_INIT (head_bias = log p слова) — уточнюється в _short_logp
            rare = min(max(rare, 1e-3), 0.999)
            self.sv_prior0 = float(_np.log(rare * self.short_v / (1 - rare)))
            self.sv_prior1 = float(_np.log(rare))
            with torch.no_grad():
                self.head.weight[self.sv_bucket].zero_()
            self.rare_cap = float(_os.environ.get("RARE_CAP", "0.3"))
        self.register_buffer("tau", torch.tensor(5.0))
        self.stats = {}
        from full import _BLK
        self.blk = _BLK
        n_light = light_a + light_b
        if "ve" in _BLK:                                         # рецепт суперника: окремі таблиці значень
            self.ve = nn.ModuleList(nn.Embedding(vocab, d) for _ in range(n_light))
        import os
        if os.environ.get("PHRASE_INIT"):                        # «народжені з готовими словосполученнями»
            ph = torch.load(os.environ["PHRASE_INIT"])
            E, H = ph["emb"], ph["head"]
            with torch.no_grad():
                rms = E.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-6)
                n = E.shape[0]                                   # словник може бути доповнений (50304)
                self.emb.weight[:n].copy_(E / rms)               # рядки входу — як звичайна ініціалізація (rms 1)
                self.head.weight[:n].copy_(H * float(os.environ.get("PHRASE_C", 0.05)))  # логіти ≈ PPMI(попереднє, наступне)
                self.head.weight[n:].zero_()
            lp = torch.full((vocab,), float(ph["logp_next"].min()) - 5)
            lp[:n] = ph["logp_next"]
            self.head_bias = nn.Parameter(lp)                    # наскільки слово часте саме по собі
            if os.environ.get("OPAQUE"):                         # ентропія прогнозу розмовника після кожного слова
                self.register_buffer("ent_t", ph["ent"].float(), persistent=False)
            if os.environ.get("SURP_PHRASE"):                    # здивування = зламане звичне словосполучення
                En = E / rms
                self.register_buffer("sp_E", (En * float(os.environ.get("PHRASE_C", 0.05))).to(torch.bfloat16), persistent=False)
                self.register_buffer("sp_H", H.to(torch.bfloat16), persistent=False)
                self.register_buffer("sp_lp", ph["logp_next"].float(), persistent=False)
                self.register_buffer("sp_lse", ph["lse"].float(), persistent=False)
        if os.environ.get("PHRASE3"):                            # словосполучення з двох слів → що буде третім
            tg = torch.load(os.environ["PHRASE3"])
            K = len(tg["keys"])
            self.register_buffer("phr_keys", tg["keys"].long(), persistent=False)
            if os.environ.get("PHR_HASH", "1" if dev.TYPE in ("tpu", "xla") else ""):
                # TPU: бінарний пошук (searchsorted) — ~18 gather на слово; хеш-таблиця cuckoo — 4 gather
                hk, hv, self.phr_hmask = _cuckoo(tg["keys"].long().numpy())
                self.register_buffer("phr_hkey", torch.from_numpy(hk), persistent=False)
                self.register_buffer("phr_hval", torch.from_numpy(hv), persistent=False)
            self.phr = nn.Embedding(K + 1, d)                    # рядок K — «пари немає в таблиці» (нулі)
            with torch.no_grad():
                self.phr.weight.zero_()
                self.phr.weight[:K].copy_(tg["vec"] / float(os.environ.get("PHRASE_C", 0.05)))
            self.phr_V = int(os.environ.get("PHR_V", 50257))
        if os.environ.get("PHRASE4"):                            # словосполучення з трьох слів → четверте
            tg = torch.load(os.environ["PHRASE4"])
            K = len(tg["keys"])
            self.register_buffer("phr4_keys", tg["keys"].long(), persistent=False)
            self.phr4 = nn.Embedding(K + 1, d)
            with torch.no_grad():
                self.phr4.weight.zero_()
                self.phr4.weight[:K].copy_(tg["vec"] / float(os.environ.get("PHRASE_C", 0.05)))
        if os.environ.get("SKIP"):                               # чернетка наперед: що буває через 2..4 слова після цього
            sk = torch.load(os.environ["SKIP"])
            g = float(os.environ.get("SKIP_G", 0.25)) / float(os.environ.get("PHRASE_C", 0.05))
            self.skip_o = sorted(sk)
            self.skip = nn.ModuleList(nn.Embedding(vocab, d) for _ in self.skip_o)
            with torch.no_grad():
                for t, o in zip(self.skip, self.skip_o):
                    t.weight.zero_()
                    t.weight[:sk[o].shape[0]].copy_(sk[o] * g)
        if os.environ.get("TONGUE"):                             # «слова на язиці»: продовження з розмовника — прямо на виході
            pt = torch.load(os.environ["TONGUE"])
            for k in ("bi_idx", "bi_val", "tri_idx", "tri_val"):
                self.register_buffer("tg_" + k, pt[k], persistent=False)
            self.register_buffer("tg_keys", pt["keys"].long(), persistent=False)
            self.tg_a = nn.Parameter(torch.tensor([float(os.environ.get("TONGUE_A", 0.5)), float(os.environ.get("TONGUE_A", 0.5))]))
        if os.environ.get("INDUCT"):                             # рефлекс впізнавання: що йшло після цього слова минулого разу
            self.ind_a = nn.Parameter(torch.tensor([1.0, 1.0]))
            self.ind_in_g = nn.Parameter(torch.full((3,), -2.0))  # ворота «впізнав → подумай про це» (спершу слабкі)
            if os.environ.get("INDUCT") == "lock":               # «ключі й замки» (Стефан): замки 1/2/3 слова, довіра — від рідкісності
                if os.environ.get("TOKEN_COUNTS"):               # частоти слів, пораховані наперед (корпус тренування)
                    cnt = np.load(os.environ["TOKEN_COUNTS"]).astype(np.int64)[:vocab]
                    cnt = np.pad(cnt, (0, vocab - len(cnt)))
                else:
                    cnt = np.bincount(np.load("fineweb.npy", mmap_mode="r")[2_000_000:].astype(np.int64), minlength=vocab)
                bucket = np.clip(np.floor(np.log10(cnt + 1)), 0, 6).astype(np.int64)   # 0: <10 … 6: ≥1M появ
                self.register_buffer("lock_bucket", torch.tensor(bucket), persistent=False)
                lw = torch.load(os.environ["LOCK_TRUST"]) if os.environ.get("LOCK_TRUST") else torch.ones(3, 7)
                self.lock_w = nn.Parameter(lw.clone())           # довіра [довжина замка, рідкісність]; наперед — з даних
                if os.environ.get("INDUCT_MIX"):                 # «легка голова перебиває»: вага суміші = частка влучань
                    pm = (lw.exp() * math.exp(-4.5)).clamp(0.005, 0.95)
                    self.mix_w = nn.Parameter(torch.log(pm / (1 - pm)))
        if os.environ.get("HEAVY_ATTN"):                         # важкі шари дивляться в прочитаний текст
            self.h_attn = nn.ModuleList(HeavyCtxAttn(d, heads) for _ in range(heavy))
        self.n_draft = int(os.environ.get("DRAFT", 0))            # якір накидає чернетку на n слів уперед
        if self.n_draft:
            self.draft = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(self.n_draft))
            for lin in self.draft:
                nn.init.zeros_(lin.weight)                       # спершу чернетка порожня
            self.draft_norm = nn.RMSNorm(d)
        if "smear" in _BLK:                                      # слово підглядає попереднє ще до уваги
            self.smear = nn.Linear(d, 1, bias=False)
            nn.init.zeros_(self.smear.weight)

    def _run(self, blk, *a, heavy=False):
        on = self.ckpt or (self.ckpt_kind == "heavy" and heavy) or (self.ckpt_kind == "light" and not heavy)
        # CKPT_NO_RNG=1: без збереження/відновлення стану генератора (preserve_rng_state=False). Dropout'у немає — рахунок
        # той самий; на torch_tpu checkpoint з preserve_rng_state тягне torch.<tpu>.get_rng_state, якого dynamo не трасує,
        # а розрив усередині тіла checkpoint неможливий → увесь шар виконується eager (m19/S1: tt_jit_bmm/pow/all_to_all
        # ×14–56 на крок, calls_captured 23.8k проти 42.6k без CKPT) — повільно і з NaN у eager-режимі torch_tpu
        if on and self.training:
            return checkpoint(blk, *a, use_reentrant=False, preserve_rng_state=not CKPT_NO_RNG)
        return blk(*a)

    def _phr_idx(self, x):
        """Номер рядка словосполучення (попереднє, поточне) у таблиці; K — «немає в таблиці»."""
        ctx = torch.cat([torch.full_like(x[:, :1], -1), x[:, :-1]], 1) * self.phr_V + x
        K = self.phr_keys.numel()
        if hasattr(self, "phr_hkey"):
            s1, s2 = _h1(ctx, self.phr_hmask), _h2(ctx, self.phr_hmask)
            return torch.where(self.phr_hkey[s1] == ctx, self.phr_hval[s1],
                               torch.where(self.phr_hkey[s2] == ctx, self.phr_hval[s2], K))
        pos = torch.searchsorted(self.phr_keys, ctx).clamp(max=K - 1)
        return torch.where(self.phr_keys[pos] == ctx, pos, K)

    def _phr4_idx(self, x):
        V = self.phr_V
        p1 = torch.cat([torch.full_like(x[:, :1], -1), x[:, :-1]], 1)
        p2 = torch.cat([torch.full_like(x[:, :2], -1), x[:, :-2]], 1)
        ctx = (p2 * V + p1) * V + x
        K = self.phr4_keys.numel()
        pos = torch.searchsorted(self.phr4_keys, ctx).clamp(max=K - 1)
        return torch.where(self.phr4_keys[pos] == ctx, pos, K)

    @torch.no_grad()
    def prep(self, x):
        """Усе, що залежить лише від слів, а не від ваг, — наперед («вода вже набрана»): рахується для наступного
        батчу на окремому потоці, поки йде поточний крок."""
        p = {}
        if hasattr(self, "phr"):
            p["phr"] = self._phr_idx(x)
        if hasattr(self, "phr4"):
            p["phr4"] = self._phr4_idx(x)
        if hasattr(self, "ind_a"):
            if INDUCT_POS:
                p["ind_cand"], p["ind_on"], p["ind_bk"], p["ind_src"] = self._induct_raw(x, True)
            else:
                p["ind_cand"], p["ind_on"], p["ind_bk"] = self._induct_raw(x)
        if hasattr(self, "sp_lse"):
            p["surp"] = self._phr_surp(x)
        return p

    def _phr_surp(self, x):
        B, T = x.shape
        s_ = torch.full((B, T), 1e4, device=x.device)
        xp, xc = x[:, :-1], x[:, 1:]
        n = self.sp_lse.numel()
        ok = (xp < n) & (xc < n)
        xp, xc = xp.clamp(max=n - 1), xc.clamp(max=n - 1)
        lo = (self.sp_E[xp].float() * self.sp_H[xc].float()).sum(-1) + self.sp_lp[xc]
        s_[:, 1:] = torch.where(ok, self.sp_lse[xp] - lo, 1e4)
        return s_

    def _induct(self, x, prep=None):
        """Кандидати рефлексу + довіра (довіра — від ваг, решта — від слів, може прийти готовою з prep)."""
        cand, on, bk = (prep["ind_cand"], prep["ind_on"], prep["ind_bk"]) if prep else self._induct_raw(x)
        if hasattr(self, "lock_w"):
            val = torch.stack([on[..., k].float() * self.lock_w[k][bk] for k in range(on.shape[-1])], -1)
        else:
            val = on.float() * self.ind_a[:on.shape[-1]]
        return cand, val

    @torch.no_grad()
    def _induct_raw(self, x, want_src=False):
        """Вроджене впізнавання (закладка на попередню появу): для кожного слова — де воно (і пара «попереднє+це»)
        траплялось останній раз у тексті; слово, що йшло тоді далі, — готовий варіант на виході. Без навчання,
        лише вага довіри ind_a. Причинно: попередня поява j < t, кандидат x[j+1] уже прочитаний."""
        B, T = x.shape
        V = self.vocab                                   # «немає попереднього» — номер поза справжніми словами
        pos = torch.arange(T, device=x.device).expand(B, T)
        cands, vals, srcs = [], [], []
        prev = torch.cat([torch.full_like(x[:, :1], V), x[:, :-1]], 1)
        keys = (x, prev * (V + 1) + x)
        if hasattr(self, "lock_w"):
            prev2 = torch.cat([torch.full_like(x[:, :2], V), x[:, :-2]], 1)
            keys = keys + ((prev2 * (V + 1) + prev) * (V + 1) + x,)
            bk = self.lock_bucket[x.clamp(max=self.lock_bucket.numel() - 1)]
        for k, key in enumerate(keys):
            with torch.no_grad():
                order = key.argsort(dim=1, stable=True)         # стабільне: однакові ключі — за позицією (без key*T, що
                # переповнює int64 на словнику 131k і тексті 8k)
                ks, ps = key.gather(1, order), pos.gather(1, order)
                same = torch.cat([torch.zeros_like(ks[:, :1], dtype=torch.bool), ks[:, 1:] == ks[:, :-1]], 1)
                lastp = torch.cat([torch.full_like(ps[:, :1], -1), ps[:, :-1]], 1)
                last = torch.full_like(x, -1).scatter(1, ps, torch.where(same, lastp, -1))
                ok = last >= 0
                src = (last + 1).clamp(max=T - 1)          # де в цьому ж тексті стоїть кандидат
                cand = x.gather(1, src)
            cands.append(torch.where(ok, cand, 0))
            srcs.append(torch.where(ok, src, 0))
            vals.append(ok)
        if hasattr(self, "lock_w"):                      # лише найдовший замок, що спрацював (без подвійного рахунку)
            for k in range(len(vals) - 1):
                vals[k] = vals[k] & ~torch.stack(vals[k + 1:], 0).any(0)
        else:
            bk = torch.zeros_like(x)
        if want_src:
            return torch.stack(cands, -1), torch.stack(vals, -1), bk, torch.stack(srcs, -1)
        return torch.stack(cands, -1), torch.stack(vals, -1), bk

    @torch.no_grad()
    def table_rows(self, x):
        """Які рядки таблиць задіяні в цьому кроці (оптимізатор оновлює лише їх). Розмір сталий — без зупинок хоста."""
        rows = {id(self.emb.weight): x.reshape(-1)}
        if hasattr(self, "phr"):
            rows[id(self.phr.weight)] = self._phr_idx(x).reshape(-1)
        if hasattr(self, "phr4"):
            rows[id(self.phr4.weight)] = self._phr4_idx(x).reshape(-1)
        return rows

    def forward(self, x, prep=None):
        if dev.TPU:
            x = x.contiguous()                        # (E0102: вигляд зі зсувом через межу графа навколо проходу пам'яті)
        B, T = x.shape
        win = None
        if T > self.window:                               # ATTN_IMPL=block: вікно без FlexAttention (TPU/XLA)
            win = (ATTN_IMPL, self.window) if ATTN_IMPL in ("block", "block2", "splash") else _window_mask(T, self.window, x.device)
        h = self.emb(x)
        if RESID_BF16 and torch.is_autocast_enabled():
            h = h.to(torch.bfloat16)                     # залишковий потік у bf16 — удвічі менше байтів на кожне додавання/норму
        if hasattr(self, "ind_a"):
            self._tongue = self._induct(x, prep)
            if hasattr(self, "mix_w"):
                cand, on, bk = (prep["ind_cand"], prep["ind_on"], prep["ind_bk"]) if prep else self._induct_raw(x)
                anyon = on.any(-1)
                kk = on.float().argmax(-1)
                c1 = (cand * on).sum(-1)
                g = torch.sigmoid(self.mix_w[kk, bk]) * anyon
                self._mix = (c1, g)
                self.stats["mix_g"] = (g.sum() / anyon.sum().clamp(min=1)).detach()
            tw = self.lock_w if hasattr(self, "lock_w") else self.ind_a
            self.stats["trust"] = tw.detach().float().mean()
            if os.environ.get("INDUCT_IN"):                  # усвідомлене впізнавання: кандидат — і на вхід думки
                cand, val = self._tongue
                g = torch.sigmoid(self.ind_in_g[:cand.shape[-1]])
                if INDUCT_POS:
                    # кандидат — слово з цього ж тексту (позиція src), тож його вектор уже є в emb(x): «який рядок
                    # взяти» — one-hot за позиціями, одне множення (B,T,T)@(B,T,d) на MXU замість вибірки з таблиці
                    src = prep["ind_src"] if prep else self._induct_raw(x, True)[3]
                    wsel = ((val != 0).float() * g[None, None])                              # (B,T,k)
                    sel = (src[..., None] == torch.arange(T, device=x.device)).to(wsel.dtype)   # (B,T,k,T)
                    mixm = (sel * wsel[..., None]).sum(2)                                   # (B,T,T)
                    h = h + torch.bmm(mixm.to(h.dtype), h).type_as(h)       # h тут — ще чисте emb(x)
                else:
                    h = h + (self.emb(cand) * (val[..., None] != 0).to(h.dtype) * g[None, None, :, None]).sum(2).type_as(h)
        if hasattr(self, "tg_a"):
            V0 = int(os.environ.get("PHR_V", 50257))
            xc = x.clamp(max=V0 - 1)
            ctx = torch.cat([torch.full_like(x[:, :1], -1), x[:, :-1]], 1) * V0 + x
            K = self.tg_keys.numel()
            pos = torch.searchsorted(self.tg_keys, ctx).clamp(max=K - 1)
            hit = (self.tg_keys[pos] == ctx)[..., None]
            cand = torch.where(hit, self.tg_tri_idx[pos], self.tg_bi_idx[xc])
            val = torch.where(hit, self.tg_tri_val[pos].float() * self.tg_a[1], self.tg_bi_val[xc].float() * self.tg_a[0])
            self._tongue = (cand, val)
        if hasattr(self, "phr"):
            h = h + self.phr(prep["phr"] if prep else self._phr_idx(x)).type_as(h)
        if hasattr(self, "phr4"):
            h = h + self.phr4(prep["phr4"] if prep else self._phr4_idx(x)).type_as(h)
        if hasattr(self, "skip"):
            for t, o in zip(self.skip, self.skip_o):
                xs = torch.cat([x[:, :1].expand(-1, o - 1), x[:, :-(o - 1)]], 1)        # слово на o-1 позицій позаду
                m = (torch.arange(x.shape[1], device=x.device) >= o - 1).to(h.dtype)[None, :, None]
                h = h + t(xs).type_as(h) * m
        # REREAD_IDS=1: «книжка» для перечитування — не тензор (B,T,128) з градієнтом, а номери слів + таблиця: рядки
        # беруться за цілими номерами лише там, де перечитуємо (зворотний прохід — у таблицю напряму, без розсилки
        # градієнта по всьому тексту). Значення й градієнти таблиці — ті самі (book(x)[pos] == book.weight[x[pos]]).
        book = (x, self.book.weight) if REREAD_IDS else self.book(x)
        if "smear" in self.blk:
            h = torch.cat([h[:, :1], h[:, 1:] + torch.sigmoid(self.smear(h[:, 1:])) * h[:, :-1]], 1)
        if hasattr(self, "ent_t"):
            # непрозорі слова — де розмовник не певен у наступному (ентропія > порогу); стеля OPAQUE частки, причинно
            keep_f = float(os.environ["OPAQUE"])
            n = self.ent_t.numel()
            ent = self.ent_t[x.clamp(max=n - 1)]
            K = max(16, int(T * keep_f) // 16 * 16)
            thr = float(os.environ.get("OPAQUE_THR", 5.0))
            op = ent > thr
            keep = op & (op.cumsum(1) <= K)
            pos_ = torch.arange(T, device=x.device)
            sidx = ((~keep).long() * T + pos_).argsort(-1)[:, :K]
            sel = (sidx, keep.gather(1, sidx))
            for b in list(self.light_a) + list(self.light_b) + list(self.final):
                b.cur_sel = sel
            self.stats["opaque"] = keep.float().mean()
        lights = list(self.light_a) + list(self.light_b) if ("ve" in self.blk or "x0" in self.blk) else []
        for i, b in enumerate(lights):
            b.cur_ve = self.ve[i](x) if "ve" in self.blk else None
            b.cur_x0 = h if "x0" in self.blk else None
        for b in self.light_a:
            h = self._run(b, h, win)

        # пам'ять: шматки по seg слів; шматок k читає стан після шматків < k, потім пише
        seg = min(self.seg, T)
        # пам'ять: шматки по seg слів; шматок k читає стан після шматків < k, потім пише
        seg = min(self.seg, T)
        no_mem = bool(os.environ.get("NO_MEM"))                  # короткі тексти: вікно бачить усе, пам'ять не потрібна
        if not no_mem:
            reads, snaps, st = self._memory_pass(h, seg)
            self.last_mem = (snaps, st)                          # для діагностики
            h = h + reads.type_as(h)
            self.stats["mem_cards"] = st["valid"].float().sum(1).mean()

        for b in self.light_b:
            h = self._run(b, h, win)
        lh = self.l_norm(h)

        # здивування → якорі (окремо від пам'яті: свій сигнал — помилка прогнозу легкої частини)
        with torch.no_grad():
            s_ = torch.full((B, T), 1e4, device=x.device)
            lhd = lh.detach()
            if hasattr(self, "sp_lse"):
                s_ = prep["surp"] if prep else self._phr_surp(x)
            if CE_CHUNK and not hasattr(self, "sp_lse") and SURP_INGRAPH:
                # SURP_INGRAPH=1: той самий рахунок, але однією скомпільованою програмою (див. _surp_lse_impl)
                s_[:, 1:] = _surp_fn()(lhd, self.l_head[0].weight, self.l_head[1].weight, x)
            elif CE_CHUNK and not hasattr(self, "sp_lse"):
                # без повної таблиці логітів (B×T×V): logsumexp шматками словника + логіт слова через h·W[y]
                h128 = self.l_head[0](lhd[:, :T - 1])
                Bq, Tq = h128.shape[:2]
                W2 = self.l_head[1].weight
                hb, Wb = h128.reshape(Bq * Tq, -1).to(torch.bfloat16), W2.to(torch.bfloat16)
                lse = torch.logsumexp(torch.stack([torch.logsumexp((hb @ Wb[c0:c0 + CE_CHUNK].t()).float(), -1)
                                                   for c0 in range(0, W2.shape[0], CE_CHUNK)], -1), -1)
                tg = x[:, 1:].reshape(-1)
                ly = (hb.float() * Wb[tg].float()).sum(-1)
                s_[:, 1:] = (lse - ly).view(Bq, Tq)
            for j in (range(0, T - 1, 8192) if not hasattr(self, "sp_lse") and not CE_CHUNK else []):
                lo = self.l_head(lhd[:, j:min(j + 8192, T - 1)])
                tgt = x[:, j + 1:j + 1 + lo.shape[1], None]
                s_[:, j + 1:j + 1 + lo.shape[1]] = (torch.logsumexp(lo, -1) - lo.gather(-1, tgt).squeeze(-1)).float()
            anchor = s_ > self.tau
            if self.training:
                q = torch.quantile(s_[:, 1:].reshape(-1)[:100_000], 1 - self.frac)
                self.tau.mul_(0.9).add_(0.1 * q)
            A = max(1, round(T * self.cap))
            keep = anchor & (anchor.cumsum(1) <= A)
            pos = torch.arange(T, device=x.device)
            idx = ((~keep).long() * T + pos).argsort(-1)[:, :A]
            valid = keep.gather(1, idx)
            self.stats["anchors"] = valid.float().mean() * A / T

        # важкі шари: лише якорі, без уваги — глибоко думають над тим, що принесла легка частина
        d = h.shape[-1]
        a_in = h.gather(1, idx[..., None].expand(-1, -1, d))
        a = a_in
        sa = s_.gather(1, idx).clamp(0, 50) if self.moe else None    # здивування якоря — бюджет експертів
        if self.moe:
            sa = torch.where(valid, sa, -1.0)                          # порожні місця не йдуть до експертів
        ctx = h.detach() if os.environ.get("HEAVY_ATTN_DETACH") else h
        if DIAG_LAYER:
            _LD["n"] += 1
            _LD["b_last"] = None                                       # назад цього кроку — з власної точки відліку
            a = _ld(a, -1, None, None)                                 # точка відліку (і вперед, і назад)
        for li, b in enumerate(self.heavy):
            if DIAG_LAYER:
                b._ld_li = li
                a = _ld(a, li, "post" if li else None, "attn" if hasattr(self, "h_attn") else "disp", li - 1)  # кінець попереднього шару → «post» того шару
            if hasattr(self, "h_attn"):
                a = self.h_attn[li](a, ctx, idx)
            if DIAG_LAYER and hasattr(self, "h_attn"):
                a = _ld(a, li, "attn", "disp")
            a = self._run(b, a, sa, heavy=True) if self.moe else self._run(b, a, heavy=True)
        if DIAG_LAYER:
            a = _ld(a, len(self.heavy) - 1, "post", "post")
        h = h.scatter_add(1, idx[..., None].expand(-1, -1, d), (a - a_in) * valid[..., None])
        if self.n_draft:
            # «напівпрозорі» слова: думка якоря після важких шарів — підказка на 1..n позицій уперед (лише майбутнім)
            an = self.draft_norm(a)
            self._drafts = []
            for j, lin in enumerate(self.draft, 1):
                pj = idx + j
                okj = valid & (pj < T)
                dj = lin(an)
                self._drafts.append((dj, pj.clamp(max=T - 1), okj))
                h = h.scatter_add(1, pj.clamp(max=T - 1)[..., None].expand(-1, -1, d), (dj * okj[..., None]).type_as(h))
        if self.moe:
            self.stats["moe_drop"] = sum(b.dropped for b in self.heavy) / len(self.heavy)
            if self.heavy[0].wave:
                self.stats["learned"] = sum(b.learned.float().mean() for b in self.heavy) / len(self.heavy)
                self.stats["hint"] = sum(b.hint_avg[0] for b in self.heavy) / len(self.heavy)
                # де проходять критерії «навчився»: помилка / пік і рух / пік (мінімум по експертах, середнє по шарах)
                self.stats["err_r"] = sum((b.err / b.err_peak.clamp(min=1e-12)).min() for b in self.heavy) / len(self.heavy)
                if self.heavy[0].wave == "still":
                    self.stats["move_r"] = sum((b.last_move / b.move_peak.clamp(min=1e-12)).min() for b in self.heavy) / len(self.heavy)

        for b in self.final:
            h = self._run(b, h, win)

        # перечитування за закладкою + впізнавання: усі шматки одним викликом
        if no_mem or os.environ.get("NO_REREAD"):         # NO_REREAD — лише для пошуку збою компілятора TPU
            return (self.norm(h), lh), torch.zeros((), device=x.device)
        o, cost = (self._reread_sep(h, book, snaps, seg, s_) if os.environ.get("REREAD_SEP")
                   else self._reread_pass(h, book, snaps, seg))
        h = h + o.type_as(h)
        return (self.norm(h), lh), cost

    def _seg_step(self, hs, st, s0):
        """Один шматок: читання пам'яті + запис. Скомпільований як один граф (розміри пам'яті сталі),
        інакше процесор запускає тисячі дрібних операцій і відеокарта чекає."""
        with torch.autocast(dev.TYPE, dtype=torch.bfloat16, enabled=MEM_BF16):   # MEM_BF16: множення пам'яті в bf16
            r, w = self.memory.read(hs, st)
            if not self.use_memory:
                r, w = r * 0, w * 0
            snap = (st["faces"], self.memory._contents(st), st["pos"], st["valid"])
            qpos = s0 + torch.arange(hs.shape[1], device=hs.device).expand(hs.shape[0], -1)
            st = self.memory.write(hs + r, qpos, st, now=s0 + hs.shape[1], w_read=w)
        return r, snap, st

    @torch.compiler.disable
    def _memory_pass(self, h, seg):
        """Цикл по шматках поза зовнішньою компіляцією (інакше компілятор розгортає сотні ітерацій);
        тіло циклу — скомпільоване."""
        B, T, _ = h.shape
        if not hasattr(self, "_seg_c"):
            import os
            mode = "reduce-overhead" if self.seg_graph else None
            self._seg_c = self._seg_step if os.environ.get("NO_COMPILE") else torch.compile(self._seg_step, mode=mode)
        st = self.memory.init_state(B, h.device, torch.float32)
        reads, snaps = [], []
        offs = torch.arange(0, T, seg, device=h.device, dtype=torch.float32)   # номери шматків — одразу на відеокарті
        for j, s in enumerate(range(0, T, seg)):
            s0 = offs[j]
            r, snap, st = self._seg_c(h[:, s:s + seg].float().contiguous(), st, s0)
            # CUDA-графи: виходи перезаписуються наступним запуском; TPU: вихід-вигляд зі зсувом наступна скомпільована
            # програма на вході не приймає (E0102) — суцільні копії стану
            if self.seg_graph or dev.TPU:
                r, snap, st = r.clone(), tuple(t.clone() for t in snap), {k: v.clone() for k, v in st.items()}
            reads.append(r)
            snaps.append(snap)
        return torch.cat(reads, 1), snaps, st

    @torch.compiler.disable
    def _reread_sep(self, h, book, snaps, seg, surp=None):
        """Перечитування окремим скомпільованим графом (REREAD_SEP): XLA не зливає його з рештою кроку
        (у спільному графі на TPU компілятор падав: «split of a padded window dim»)."""
        # підготовка входів (склеювання знімків шматків, pad, перекладки) — поза графом, суцільними копіями:
        # інакше на вхід програми потрапляє вигляд на половину склеєного буфера (E0102)
        B, T, d = h.shape
        n = len(snaps)
        pad = n * seg - T
        if REREAD_LEAN and pad == 0 and B == 1:
            # REREAD_LEAN: без pad (T кратне шматку), без перестановки (B = 1 — вона тотожна) і без зайвого clone —
            # h.float() і так нова суцільна копія; знімки — свіжий stack (суцільний повний буфер)
            hs = h.float().view(n, seg, d)
        else:
            hs = F.pad(h.float(), (0, 0, 0, pad)).view(B, n, seg, d).transpose(0, 1).reshape(n * B, seg, d).contiguous().clone()
        cg = lambda t: _ContigGrad.apply(t) if t.requires_grad else t   # градієнт кожного знімка — суцільний
        if REREAD_LEAN:
            cat = lambda i: torch.stack([cg(sn[i]) for sn in snaps]).reshape(n * B, *snaps[0][i].shape[1:])
        else:
            cat = lambda i: torch.stack([cg(sn[i]) for sn in snaps]).reshape(n * B, *snaps[0][i].shape[1:]).contiguous().clone()
        faces, contents, pos, valid = cat(0), cat(1), cat(2), cat(3)
        qpos = (torch.arange(n, device=h.device)[:, None] * seg + torch.arange(seg, device=h.device)).repeat_interleave(B, 0).contiguous()
        rows = torch.arange(B, device=h.device).repeat(n).contiguous()
        if not hasattr(self, "_reread_c"):
            self._reread_c = self._reread_core if os.environ.get("NO_COMPILE") else torch.compile(self._reread_core)
        frac = float(os.environ.get("REREAD_ANCH", 0) or 0)
        if frac and surp is not None:
            # перечитування лише на словах зі здивуванням (Стефан): у кожному шматку — частка frac найдивніших слів;
            # результат розкладається на свої місця one-hot множенням (сталі форми, без scatter)
            k = max(1, int(round(seg * frac)))
            sp = F.pad(surp.float(), (0, pad), value=-1e9).view(B, n, seg).transpose(0, 1).reshape(n * B, seg)
            sel = sp.detach().topk(k, -1).indices.sort(-1).values                            # (nB, k)
            # збирання і розкладання — one-hot множенням в обидва боки (без gather/scatter: на TPU їхній
            # зворотний прохід давав «split of a padded window dim»)
            oh = (sel[..., None] == torch.arange(seg, device=h.device)).to(hs.dtype).contiguous()  # (nB, k, seg)
            with torch.autocast(dev.TYPE, enabled=False):                                   # збирання точне, fp32
                hs_k = torch.bmm(oh, hs).contiguous()
            qpos_k = (oh * qpos[:, None, :].to(hs.dtype)).sum(-1).round().long().contiguous()
            o_k, cost = self._reread_c(hs_k, _book_in(book), faces, contents, pos, valid, qpos_k, rows)
            o = torch.bmm(oh.transpose(1, 2).to(o_k.dtype).contiguous(), o_k)            # (nB, seg, d)
        else:
            o, cost = self._reread_c(hs, _book_in(book), faces, contents, pos, valid, qpos, rows)
        if not self.use_memory:
            o, cost = o * 0, cost * 0
        o = o.view(n, B, seg, d).transpose(0, 1).reshape(B, n * seg, d)[:, :T]
        return o, cost

    def _reread_core(self, hs, book, faces, contents, pos, valid, qpos, rows):
        with torch.autocast(dev.TYPE, dtype=torch.bfloat16, enabled=MEM_BF16):
            return self.reread(hs, book, faces, contents, pos, valid, qpos, rows)

    def _reread_pass(self, h, book, snaps, seg):
        """Перечитування для всіх шматків одним батчем: шматок k бачить знімок пам'яті k."""
        B, T, d = h.shape
        n = len(snaps)
        pad = n * seg - T
        hs = F.pad(h.float(), (0, 0, 0, pad)).view(B, n, seg, d).transpose(0, 1).reshape(n * B, seg, d)
        cat = lambda i: torch.stack([sn[i] for sn in snaps]).reshape(n * B, *snaps[0][i].shape[1:])
        faces, contents, pos, valid = cat(0), cat(1), cat(2), cat(3)
        if dev.TPU:                                   # суцільні буфери на вході скомпільованої програми (E0102)
            hs, faces, contents, pos, valid = (t.contiguous().clone() for t in (hs, faces, contents, pos, valid))
        if "noqi" in _RR_BISECT:                      # без repeat_interleave: те саме через expand + reshape
            qpos = (torch.arange(n, device=h.device)[:, None] * seg + torch.arange(seg, device=h.device))[:, None].expand(n, B, seg).reshape(n * B, seg)
        else:
            qpos = (torch.arange(n, device=h.device)[:, None] * seg + torch.arange(seg, device=h.device)).repeat_interleave(B, 0)
        rows = torch.arange(B, device=h.device).repeat(n)                       # з якого тексту рядок
        with torch.autocast(dev.TYPE, dtype=torch.bfloat16, enabled=MEM_BF16):
            o, cost = self.reread(hs, _book_in(book), faces, contents, pos, valid, qpos, rows)
        if not self.use_memory:
            o, cost = o * 0, cost * 0
        o = o.view(n, B, seg, d).transpose(0, 1).reshape(B, n * seg, d)[:, :T]
        return o, cost

    @torch.compiler.disable                          # поза графом: Liger не вбудовується в torch.compile; кінець кроку
    def _head_w(self, hn):
        """Вихідна таблиця; з head_bias — частота слова як додатковий вимір (злитий кернел зсуву не знає)."""
        if not hasattr(self, "head_bias"):
            return hn, self.head.weight
        V = self.head.weight.shape[0]
        hn = torch.cat([hn, torch.ones_like(hn[..., :1]), torch.zeros_like(hn[..., :15])], -1)
        w = torch.cat([self.head.weight, self.head_bias[:, None].to(self.head.weight.dtype),
                       self.head.weight.new_zeros(V, 15)], 1)
        return hn, w

    def _fused_ce(self, hn, y):
        from lm import FUSED
        hn, w = self._head_w(hn)
        if FUSED == "ours":
            import fused_ops
            if hasattr(self, "_mix"):
                c1, g = self._mix
                return fused_ops.linear_ce_mix(hn, w, y, g, (c1 == y).float())
            if hasattr(self, "_tongue"):
                cand, val = self._tongue
                return fused_ops.linear_ce_sparse(hn, w, y, cand, val)
            return fused_ops.linear_ce(hn, w, y)
        from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
        return liger_fused_linear_cross_entropy(hn.reshape(-1, hn.shape[-1]), w, y.reshape(-1),
                                                reduction="sum")

    def _ce_tok(self, hn, y, j=None):
        lo = self.head(hn).float()
        if hasattr(self, "head_bias"):
            lo = lo + self.head_bias.float()
        return F.cross_entropy(lo.reshape(-1, lo.shape[-1]), y.reshape(-1), reduction="none")

    def _short_logp(self, h, y):
        """log P(y) дворівневого виходу (SHORT_V). h (N, d), y (N,)."""
        if _SHORT_LOGP_HOOK is not None:                 # SHARD_VOCAB=1 (hyperop/shard.py): голова порізана, не збирається
            return _SHORT_LOGP_HOOK(self, h, y)
        W, b = self.head.weight, getattr(self, "head_bias", None)
        K, N = self.short_v, h.shape[0]
        Ws = torch.cat([W[self.sv_ids], W[self.sv_bucket:self.sv_bucket + 1]])            # (K+1, d)
        bs = torch.cat([b[self.sv_ids].float(), h.new_full((1,), self.sv_prior1, dtype=torch.float32)]) if b is not None             else torch.cat([h.new_zeros(K, dtype=torch.float32), h.new_full((1,), self.sv_prior0, dtype=torch.float32)])
        ys = self.sv_pos[y]                                                           # K — «скринька рідкісне»
        ch = CE_CHUNK or 16768
        lse_s, ly_s = chunk_ce(h, Ws, bs, ys, min(ch, K + 1))
        logp = ly_s - lse_s
        rare = ys == K
        # місця під повний словник: сталою місткістю, рідкісні — першими (за порядком тексту)
        C = min(N, -(-int(round(self.rare_cap * N)) // 256) * 256)
        key = rare.float() * 2 - torch.arange(N, device=h.device, dtype=torch.float32) / N
        idx = key.detach().topk(C).indices
        hit = rare[idx]
        br = (b.float() if b is not None else W.new_zeros(W.shape[0], dtype=torch.float32)) + self.sv_mask
        lse_r, ly_r = chunk_ce(h[idx], W, br, y[idx], ch)
        logp = logp.index_add(0, idx, torch.where(hit, ly_r - lse_r, torch.zeros_like(ly_r)))
        if not torch.compiler.is_compiling():
            self.stats["rare_drop"] = (rare.sum() - hit.sum()).float() / N             # надлишок понад місткість
        return logp

    def _ce_part(self, hn, y, j=None):
        # без повного log_softmax у пам'яті: logsumexp (fp32 — усередині згортки) + логіт правильного слова.
        # На TPU (без злитого кернела) вихід на словник був ~третиною кроку.
        if CE_CHUNK and not (hasattr(self, "_tongue") and j is not None and not hasattr(self, "_mix")):
            B_, T_ = y.shape
            if self.short_v:
                logp = self._short_logp(hn.reshape(B_ * T_, -1), y.reshape(-1)).view(B_, T_)
                lse, ly = torch.zeros_like(logp), logp                       # ly − lse = log P(y)
            else:
                lse, ly = chunk_ce(hn.reshape(B_ * T_, -1), self.head.weight, getattr(self, "head_bias", None), y.reshape(-1))
                lse, ly = lse.view(B_, T_), ly.view(B_, T_)
            if hasattr(self, "_mix") and j is not None:
                c1, g = self._mix
                c1, g = c1[:, j:j + T_], g[:, j:j + T_]
                py = (ly - lse).exp()
                return -torch.log(((1 - g) * py + g * (c1 == y).float()).clamp(min=1e-30)).sum()
            return (lse - ly).sum()
        lo = self.head(hn).float()
        if hasattr(self, "head_bias"):
            lo = lo + self.head_bias.float()
        if hasattr(self, "_tongue") and j is not None and not hasattr(self, "_mix"):
            cand, val = self._tongue
            lo = lo.scatter_add(-1, cand[:, j:j + lo.shape[1]].long(), val[:, j:j + lo.shape[1]])
        lse = torch.logsumexp(lo, -1)
        ly = lo.gather(-1, y[..., None])[..., 0]
        if hasattr(self, "_mix") and j is not None:      # суміш «легка голова перебиває» — та сама, що на тренуванні
            c1, g = self._mix
            c1, g = c1[:, j:j + lo.shape[1]], g[:, j:j + lo.shape[1]]
            py = (ly - lse).exp()
            return -torch.log(((1 - g) * py + g * (c1 == y).float()).clamp(min=1e-30)).sum()
        return (lse - ly).sum()

    @torch.no_grad()
    def rebank(self, step, every=1000):
        """Раз на `every` кроків: переселити експертів між банками в усіх шарах разом (одна перекомпіляція).
        Єдине читання маски на хост за `every` кроків."""
        banks = [b for b in self.heavy if isinstance(b, BankMoE)]
        if not banks or step == 0 or step % every:
            return []
        masks = torch.stack([b.learned for b in banks]).cpu().tolist()
        if not any(b.wants_rebank(mk) for b, mk in zip(banks, masks)):
            return []
        changes = []
        for b, mk in zip(banks, masks):
            changes += b.rebank(mk)
        self.stats["frozen"] = torch.tensor(sum(len(b.Fz) for b in banks) / sum(b.E for b in banks))
        return changes

    def before_step(self, step):
        for b in self.heavy:
            if hasattr(b, "before_step"):
                b.before_step(step)

    def train_loss(self, out, y, train=True):
        hn, lh = out
        from lm import FUSED
        if train and self.training and getattr(self, "panic", 0):
            # «паніка перед іспитом»: вага слова — наскільки погано модель його зараз знає (лише те, що завалиш)
            loss = 0
            for j in range(0, hn.shape[1], 8192):
                ce = checkpoint(self._ce_tok, hn[:, j:j + 8192], y[:, j:j + 8192], j, use_reentrant=False)
                w = (ce / ce.mean()).detach().clamp(0.25, 4.0) ** self.panic
                loss = loss + (ce * (w / w.mean())).sum()   # (ділення на кількість слів — нижче, спільне)
        elif FUSED and self.training and not self.short_v:   # злитий вихід + похибка (з SHORT_V — свій дворівневий вихід)
            loss = self._fused_ce(hn, y)
        else:
            loss = 0
            for j in range(0, hn.shape[1], 8192):
                part = (hn[:, j:j + 8192], y[:, j:j + 8192])
                loss = loss + (checkpoint(self._ce_part, *part, j, use_reentrant=False) if self.training and not CE_CHUNK
                               else self._ce_part(*part, j))     # CE_CHUNK сам перераховує шматки назад
        loss = loss / y.numel()
        if train and self.moe:
            loss = loss + 0.01 * sum(b.aux for b in self.heavy) / len(self.heavy)   # рівномірне навантаження експертів
        if train and hasattr(self.light_a[0], "moe"):
            lb = [b for b in list(self.light_a) + list(self.light_b) + list(self.final)]
            loss = loss + 0.01 * sum(b.moe.aux for b in lb) / len(lb)
        if train and getattr(self, "n_draft", 0) and os.environ.get("DRAFT_LOSS"):
            # явна чернетка: підказка якоря на позицію p+j має вгадати слово, яке там треба сказати (y[p+j])
            wd = float(os.environ.get("DRAFT_LOSS"))
            dl = 0
            for dj, pj, okj in self._drafts:
                tgt = y.gather(1, pj)
                lo = self.head(self.norm(dj)).float()
                if hasattr(self, "head_bias"):
                    lo = lo + self.head_bias.float()
                ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tgt.reshape(-1), reduction="none")
                dl = dl + (ce * okj.reshape(-1)).sum() / okj.sum().clamp(min=1)
            loss = loss + wd * dl / len(self._drafts)
        if train and not hasattr(self, "sp_lse"):
            V = self.head.out_features
            if CE_CHUNK:                                 # легка голова — теж шматками словника
                h4 = self.l_head[0](lh[:, ::4])
                lse, ly = chunk_ce(h4.reshape(-1, h4.shape[-1]), self.l_head[1].weight, None, y[:, ::4].reshape(-1))
                loss = loss + 0.3 * (lse - ly).mean()
            else:
                loss = loss + 0.3 * F.cross_entropy(self.l_head(lh[:, ::4]).float().reshape(-1, V), y[:, ::4].reshape(-1))
        return loss


def build_full2(name, vocab):
    if name == "Fv2":
        return FullLM2(vocab)
    if name == "Fv2_nomem":                          # та сама модель без пам'яті (контроль для довгих книжок)
        return FullLM2(vocab, ckpt=True, use_memory=False)
    if name == "Fv2ck":
        return FullLM2(vocab, ckpt=True)
    if name in ("Fv2moe", "Fv2moeB"):                # важкі шари як MoE: рівно 2 експерти / бюджет за здивуванням
        return FullLM2(vocab, moe="budget" if name == "Fv2moeB" else "fixed")
    if name in ("Fv2moeW", "Fv2moeWs", "Fv2moeW5", "Fv2moeWsN"):  # + хвилі: «навчився» = мала помилка / не рухається
        m = FullLM2(vocab, moe="fixed", wave="still" if name in ("Fv2moeWs", "Fv2moeWsN") else "err")
        if name == "Fv2moeWsN":                      # без учителя: лише заморозка навчених
            for blk in m.heavy:
                blk.hint_w = 0.0
        if name == "Fv2moeW5":                       # м'якший поріг «малої помилки»: 60% від піку
            for blk in m.heavy:
                blk.err_thr = 0.6
        return m
    if name == "Fv2ckg":                             # те саме + крок пам'яті записаний графом
        return FullLM2(vocab, ckpt=True, seg_graph=True)
    if name in ("Fv3s", "Fv3s_big16"):              # пам'ять v3 (обсяг); Fv3s — обсяг як у Fv2s, big16 — як у big16
        big = name == "Fv3s_big16"
        return FullLM2(vocab, d=256, light_a=1, light_b=1, heavy=2, final=1, heads=4, window=64, seg=32,
                       mem_kw=dict(fresh=64 if big else 16, dim=128 if big else 32, dim_max=160 if big else 48,
                                   write=16 if big else 8, focus=4), volume=True)
    if name.startswith("Fv2moeKs"):                  # мала для ПК: d=256, Fv2moeKs8 — 8 експертів у банках
        return FullLM2(vocab, d=256, light_a=1, light_b=1, heavy=3, final=1, heads=4, window=128, seg=64,
                       moe="fixed", n_exp=int(name[8:]), bank=True,
                       mem_kw=dict(fresh=64, dim=128, dim_max=160, write=16, focus=4))
    if name.startswith("Fv2moeK"):                   # хвилі + справжні банки: Fv2moeK32 — d=768, 32 експерти
        # Fv2moeK32h4 — те саме, але 4 важкі шари (профіль на 3060); Fv2moeK32d1024 — ширина 1024 (як у суперника)
        rest, _, dd = name[7:].partition("d")
        e, _, h = rest.partition("h")
        d = int(dd) if dd else 768
        L = int(os.environ.get("LIGHT", 2))            # легких шарів до / після пам'яті (кожне слово)
        return FullLM2(vocab, d=d, heads=d // int(os.environ.get("HEAD_DIM", 64)), moe="fixed", n_exp=int(e), bank=True, heavy=int(h) if h else 14,
                       light_a=L, light_b=L, seg=int(os.environ.get("SEG", 512)))   # SEG — шматок пам'яті (вказівник бачить попередні)
    if name.startswith("Fv2moeE"):                   # «ціна 1 до N»: d=768, 2 активні експерти, змінюється лише їх кількість
        # Fv2moeE64 — усі вчаться; Fv2moeE64f90 — 90% експертів у замороженому банку
        e, _, fr = name[7:].partition("f")
        E = int(e)
        return FullLM2(vocab, d=768, heads=12, moe="fixed", n_exp=E, n_frozen=int(E * int(fr) / 100) if fr else 0)
    if name in ("Fns", "Fns_nomem"):                 # мала модель для «голки» на ПК: d=384, вікно 128, велика пам'ять
        return FullLM2(vocab, d=384, light_a=1, light_b=1, heavy=3, final=1, heads=6, window=128, seg=64, ckpt=True,
                       mem_kw=dict(fresh=64, dim=128, dim_max=160, write=16, focus=4), use_memory=name == "Fns")
    if name in ("Fn", "Fn_nomem"):                   # середня модель для «голки» на книжках: d=512, вікно 256, велика пам'ять
        return FullLM2(vocab, d=512, light_a=2, light_b=2, heavy=4, final=1, heads=8, window=256, seg=128, ckpt=True,
                       mem_kw=dict(fresh=64, dim=128, dim_max=160, write=16, focus=4), use_memory=name == "Fn")
    if name == "Fv2s_big16s64":                      # гальмо на довжину: шматок = вікно (64), запис ×2 — та сама частка слів
        return FullLM2(vocab, d=256, light_a=1, light_b=1, heavy=2, final=1, heads=4, window=64, seg=64,
                       mem_kw=dict(fresh=64, dim=128, dim_max=160, write=32, focus=4))
    if name in ("Fv2s_big", "Fv2s_big16"):          # маленька модель, але більша пам'ять: 64 свіжі + 128 тьмяних
        return FullLM2(vocab, d=256, light_a=1, light_b=1, heavy=2, final=1, heads=4, window=64, seg=32,
                       mem_kw=dict(fresh=64, dim=128, dim_max=160, write=16 if name == "Fv2s_big16" else 8, focus=4))
    if name in ("Fv2s", "Fv2s_nomem", "Fv2s_w16"):  # маленька: вікно 64, шматки 32, пам'ять 16+32
        return FullLM2(vocab, d=256, light_a=1, light_b=1, heavy=2, final=1, heads=4, window=64, seg=32,
                       mem_kw=dict(fresh=16, dim=32, dim_max=48, write=16 if name == "Fv2s_w16" else 8, focus=4),
                       use_memory=name != "Fv2s_nomem")
    raise ValueError(name)
