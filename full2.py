"""Повна модель v2 (ARCHITECTURE.md, «Уточнення Стефана»): дві незалежні системи.

  легкі шари (вікно)  →  читання пам'яті (фокус + картки)  →  легкі шари (вікно)
     → здивування → якорі (поріг + стеля)  →  важкі шари: лише думають над якорем (без уваги)
     → фінальні шари (вікно)  →  перечитування за закладкою + впізнавання  →  вихід

Пам'ять (mem2.Memory) — окрема система: свій сигнал яскравості, фіксований розмір
(свіжа + тьмяна з прибиранням), картки з ланцюжком лиць і закладками. Якорі —
лише рішення, де думати важко. Ціна на слово не залежить від довжини тексту.

Fv2 — ~300M (ширина 1024: 2+2 легкі, 14 важких, 2 фінальні), Fv2s — маленька для перевірок.
"""
import math

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from full import FlexBlock, _COMPILE
from lm import FFN
from mem2 import Memory
from mem3 import VolumeMemory


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
        sc = torch.einsum("bsf,bclf->bscl", q, F.normalize(faces, dim=-1)).amax(-1) * math.sqrt(self.df)
        sc = sc.masked_fill(~valid[:, None], -1e4)
        k = min(self.k, sc.shape[-1])
        top_s, top_j = sc.topk(k, -1)
        w = F.softmax(top_s.float(), -1).type_as(h) * (top_s > -1e3)
        B, S = h.shape[:2]
        mark = pos.gather(1, top_j.reshape(B, -1)).view(B, S, k)                 # закладки
        rpos = mark[..., None] + torch.arange(self.R, device=h.device)
        readable = rpos < qpos[:, :, None, None]                                  # лише вже прочитане
        rpos = rpos.clamp(0, book.shape[1] - 1)
        Db = book.shape[-1]
        if rows is None:
            rows = torch.arange(B, device=h.device)
        flat = (rows[:, None, None, None] * book.shape[1] + rpos).reshape(-1)  # рядок книжки = свій текст
        v = book.reshape(-1, Db)[flat].view(B, S, k, self.R, Db) + self.offset
        halt = torch.sigmoid(self.halt_v(v).squeeze(-1) + self.halt_h(n)[:, :, None])
        halt = torch.where(readable, halt, torch.ones_like(halt))
        halt = torch.cat([halt[..., :-1], torch.ones_like(halt[..., -1:])], -1)
        survive = torch.cumprod(torch.cat([torch.ones_like(halt[..., :1]), 1 - halt[..., :-1]], -1), -1)
        read = ((survive * halt * readable)[..., None] * v).sum(-2)              # (B,S,k,Db)
        # впізнавання: чи збігається прочитане з карткою
        card = contents.gather(1, top_j.reshape(B, -1, 1).expand(-1, -1, contents.shape[-1])).view(B, S, k, -1)
        recog = torch.sigmoid((F.normalize(self.recog_r(read), dim=-1)
                               * F.normalize(self.recog_c(card), dim=-1)).sum(-1, keepdim=True) * 4)
        got = (w[..., None] * recog * read).sum(-2)
        cost = (w * (survive * readable).sum(-1)).sum(-1).mean()
        return self.out(got), cost


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
        xf, xe, wtab, table, C = self._dispatch(x, surprise)
        N, d = xf.shape
        ye = self._experts(xe, wtab, C)
        ye = ye.reshape(self.E * C, d) * wtab[:, None]
        out = xf.new_zeros(N + 1, d).index_add(0, table, ye.to(xf.dtype))[:N]     # складання — у fp32
        return x + out.view_as(x)

    def _dispatch(self, x, surprise):
        """Роутер + розсилка якорів по місцях експертів: xe (E, C, d), ваги місць wtab, таблиця місць → якір."""
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
        e_flat = top_e.masked_fill(~keep, self.E)                                # E = «нікуди»
        oh = F.one_hot(e_flat, self.E + 1)[..., :self.E]                         # (N,kmax,E)
        # навантаження: щоб експерти не простоювали (як у Switch Transformer)
        frac = oh.sum((0, 1)).float()
        pm = (probs * real[:, None]).sum(0) / real.sum().clamp(min=1)
        self.aux = self.E * (frac / frac.sum().clamp(min=1) * pm).sum()
        rank = (oh.reshape(-1, self.E).cumsum(0) - 1).view_as(oh)                # номер якоря в черзі експерта
        place = (oh.bool() & (rank < C))                                          # влізло в місткість
        self.dropped = 1 - place.sum() / keep.sum().clamp(min=1)                 # частка, якій не вистачило місця
        slot = (torch.arange(self.E, device=x.device) * C)[None, None] + rank    # (N,kmax,E) місце в загальній таблиці
        flat_idx = torch.where(place, slot, torch.full_like(slot, self.E * C)).amin(-1)   # одне місце на (якір, вибір)
        tok = torch.arange(N, device=x.device)[:, None].expand(-1, self.k_max)
        ok = flat_idx < self.E * C
        table = torch.full((self.E * C + 1,), N, device=x.device, dtype=torch.long)   # N = порожнє місце
        table = table.scatter(0, flat_idx.reshape(-1).clamp(max=self.E * C), torch.where(ok, tok, N).reshape(-1))[:-1]
        # розсилка в bf16 (як решта множень під autocast): удвічі менше байтів на копіювання якорів
        xd = xf.to(torch.get_autocast_dtype("cuda")) if torch.is_autocast_enabled() else xf
        wtab = torch.zeros(self.E * C + 1, device=x.device, dtype=xd.dtype)
        wtab = wtab.scatter(0, flat_idx.reshape(-1).clamp(max=self.E * C), (w * ok).reshape(-1).to(xd.dtype))[:-1]
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


class BankMoE(BudgetMoE):
    """MoE з хвилями «перестав мінятись» і справжнім заморожений банком (ціна «1 до N» у тренуванні).

    Два банки: «вчаться» (параметр fp32 з оптимізатором) і «заморожені» (bf16-буфер: без градієнтів ваг, без
    оптимізатора й без щокрокового перетворення). Роутер бачить усіх E експертів, як і раніше. Навчені переїжджають
    у заморожений банк групами раз на `rebank_every` кроків (одночасно в усіх шарах — щоб граф перекомпілювався
    один раз), а той, хто знову почав помилятись, повертається. Підказок учителя немає (у тесті вони нічого не дали).
    """

    def __init__(self, d, E=8, **kw):
        super().__init__(d, E=E, budget=False, wave="still", hint_w=0.0, **kw)
        hid = 2 * d
        self.L, self.Fz = list(range(E)), []                  # хто вчиться / хто заморожений (номери експертів)
        self.has_L, self.has_F = True, False
        self.eager_experts = False                            # True — поза графом: без перекомпіляцій, але крок +28%
        self.register_buffer("L_idx", torch.arange(E))
        self.register_buffer("F_idx", torch.zeros(0, dtype=torch.long))
        self.register_buffer("up_f", torch.zeros(0, d, 2 * hid, dtype=torch.bfloat16))
        self.register_buffer("down_f", torch.zeros(0, hid, d, dtype=torch.bfloat16))
        # BF16_EXPERTS=1: ваги «вчаться» теж у bf16 (точна fp32-копія — в оптимізаторі): без щокрокового
        # перетворення fp32→bf16 і з удвічі легшими градієнтами — це ~90% усіх параметрів моделі
        import os
        self.w_dtype = torch.bfloat16 if os.environ.get("BF16_EXPERTS") == "1" else torch.float32
        # MOE_ACT=relu2: експерт relu(x·up)² · down (як у modded-nanogpt) замість SwiGLU; ширина 1.5× — параметрів
        # і обчислень стільки ж, але без гейта (менше поелементної роботи й байтів активацій)
        self.relu2 = os.environ.get("MOE_ACT") == "relu2"
        if self.relu2:
            h2 = 3 * hid // 2
            self.up = nn.Parameter(torch.randn(E, d, h2) * d ** -0.5)
            self.down = nn.Parameter(torch.randn(E, h2, d) * h2 ** -0.5 * 0.5)
            self.up_f = torch.zeros(0, d, h2, dtype=torch.bfloat16)
            self.down_f = torch.zeros(0, h2, d, dtype=torch.bfloat16)
        self.up = nn.Parameter(self.up.detach().to(self.w_dtype))
        self.down = nn.Parameter(self.down.detach().to(self.w_dtype))
        self.up_snap = self.up.detach().clone()

    def _ffn(self, x, up, down):
        if self.relu2:
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

    def _experts_body(self, xe, wtab, C):
        # (розміри банків — динамічні виміри графа: переїзд не перебудовує граф; гілки — лише «є/нема заморожених»)
        if not self.has_F:
            ye = self._ffn(xe, self.up, self.down)
        else:
            parts = []
            if self.has_L:
                parts.append((self.L_idx, self._ffn(xe[self.L_idx], self.up, self.down)))
            parts.append((self.F_idx, self._ffn(xe[self.F_idx], self.up_f, self.down_f)))
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
                full[torch.tensor(self.Fz, device=old.device)] = frz.float()
            new = nn.Parameter((full[torch.tensor(newL, device=old.device, dtype=torch.long)] if newL else full[:0])
                               .to(self.w_dtype).contiguous())
            setattr(self, nm, new)
            setattr(self, fnm, full[torch.tensor(newF, device=old.device, dtype=torch.long)].bfloat16().contiguous()
                    if newF else frz[:0])
            changes.append((old, new, [self.L.index(e) if e in self.L else None for e in newL]))
        self.L, self.Fz = newL, newF
        self.has_L, self.has_F = bool(newL), bool(newF)
        dev = self.up.device
        self.L_idx = torch.tensor(newL, device=dev, dtype=torch.long)
        self.F_idx = torch.tensor(newF, device=dev, dtype=torch.long)
        self.up_snap = self.up.detach().clone()
        for t in (self.up, self.down, self.up_f, self.down_f, self.L_idx, self.F_idx):
            if t.shape[0] > 1:
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
        self.seg_graph = seg_graph                               # крок пам'яті записаний графом (CUDA graphs)
        self.use_memory = use_memory                             # False — для перевірки, що задача чесна
        self.frac, self.cap, self.window, self.seg, self.ckpt = frac, cap, window, seg, ckpt
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

    def _run(self, blk, *a):
        return checkpoint(blk, *a, use_reentrant=False) if self.ckpt and self.training else blk(*a)

    def _phr_idx(self, x):
        """Номер рядка словосполучення (попереднє, поточне) у таблиці; K — «немає в таблиці»."""
        ctx = torch.cat([torch.full_like(x[:, :1], -1), x[:, :-1]], 1) * self.phr_V + x
        K = self.phr_keys.numel()
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
    def _induct_raw(self, x):
        """Вроджене впізнавання (закладка на попередню появу): для кожного слова — де воно (і пара «попереднє+це»)
        траплялось останній раз у тексті; слово, що йшло тоді далі, — готовий варіант на виході. Без навчання,
        лише вага довіри ind_a. Причинно: попередня поява j < t, кандидат x[j+1] уже прочитаний."""
        B, T = x.shape
        V = 50304
        pos = torch.arange(T, device=x.device).expand(B, T)
        cands, vals = [], []
        prev = torch.cat([torch.full_like(x[:, :1], V), x[:, :-1]], 1)
        keys = (x, prev * (V + 1) + x)
        if hasattr(self, "lock_w"):
            prev2 = torch.cat([torch.full_like(x[:, :2], V), x[:, :-2]], 1)
            keys = keys + ((prev2 * (V + 1) + prev) * (V + 1) + x,)
            bk = self.lock_bucket[x.clamp(max=self.lock_bucket.numel() - 1)]
        for k, key in enumerate(keys):
            with torch.no_grad():
                order = (key * T + pos).argsort(1)
                ks, ps = key.gather(1, order), pos.gather(1, order)
                same = torch.cat([torch.zeros_like(ks[:, :1], dtype=torch.bool), ks[:, 1:] == ks[:, :-1]], 1)
                lastp = torch.cat([torch.full_like(ps[:, :1], -1), ps[:, :-1]], 1)
                last = torch.full_like(x, -1).scatter(1, ps, torch.where(same, lastp, -1))
                ok = last >= 0
                cand = x.gather(1, (last + 1).clamp(max=T - 1))
            cands.append(torch.where(ok, cand, 0))
            vals.append(ok)
        if hasattr(self, "lock_w"):                      # лише найдовший замок, що спрацював (без подвійного рахунку)
            for k in range(len(vals) - 1):
                vals[k] = vals[k] & ~torch.stack(vals[k + 1:], 0).any(0)
        else:
            bk = torch.zeros_like(x)
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
        B, T = x.shape
        win = _window_mask(T, self.window, x.device) if T > self.window else None
        h = self.emb(x)
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
                h = h + (self.emb(cand) * (val[..., None] != 0).to(h.dtype) * g[None, None, :, None]).sum(2).type_as(h)
        if hasattr(self, "tg_a"):
            V0 = 50257
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
        book = self.book(x)
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
            for j in (range(0, T - 1, 8192) if not hasattr(self, "sp_lse") else []):
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
        for li, b in enumerate(self.heavy):
            if hasattr(self, "h_attn"):
                a = self.h_attn[li](a, ctx, idx)
            a = self._run(b, a, sa) if self.moe else self._run(b, a)
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
        if no_mem:
            return (self.norm(h), lh), torch.zeros((), device=x.device)
        o, cost = self._reread_pass(h, book, snaps, seg)
        h = h + o.type_as(h)
        return (self.norm(h), lh), cost

    def _seg_step(self, hs, st, s0):
        """Один шматок: читання пам'яті + запис. Скомпільований як один граф (розміри пам'яті сталі),
        інакше процесор запускає тисячі дрібних операцій і відеокарта чекає."""
        with torch.autocast("cuda", enabled=False):
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
            if self.seg_graph:                                   # виходи графа перезаписуються наступним запуском
                r, snap, st = r.clone(), tuple(t.clone() for t in snap), {k: v.clone() for k, v in st.items()}
            reads.append(r)
            snaps.append(snap)
        return torch.cat(reads, 1), snaps, st

    def _reread_pass(self, h, book, snaps, seg):
        """Перечитування для всіх шматків одним батчем: шматок k бачить знімок пам'яті k."""
        B, T, d = h.shape
        n = len(snaps)
        pad = n * seg - T
        hs = F.pad(h.float(), (0, 0, 0, pad)).view(B, n, seg, d).transpose(0, 1).reshape(n * B, seg, d)
        cat = lambda i: torch.stack([sn[i] for sn in snaps]).reshape(n * B, *snaps[0][i].shape[1:])
        faces, contents, pos, valid = cat(0), cat(1), cat(2), cat(3)
        qpos = (torch.arange(n, device=h.device)[:, None] * seg + torch.arange(seg, device=h.device)).repeat_interleave(B, 0)
        rows = torch.arange(B, device=h.device).repeat(n)                       # з якого тексту рядок
        with torch.autocast("cuda", enabled=False):
            o, cost = self.reread(hs, book.float(), faces, contents, pos, valid, qpos, rows)
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

    def _ce_part(self, hn, y, j=None):
        lo = self.head(hn).float()
        if hasattr(self, "head_bias"):
            lo = lo + self.head_bias.float()
        if hasattr(self, "_mix") and j is not None:      # суміш «легка голова перебиває» — та сама, що на тренуванні
            c1, g = self._mix
            c1, g = c1[:, j:j + lo.shape[1]], g[:, j:j + lo.shape[1]]
            py = lo.log_softmax(-1).gather(-1, y[..., None])[..., 0].exp()
            return -torch.log(((1 - g) * py + g * (c1 == y).float()).clamp(min=1e-30)).sum()
        if hasattr(self, "_tongue") and j is not None:
            cand, val = self._tongue
            lo = lo.scatter_add(-1, cand[:, j:j + lo.shape[1]].long(), val[:, j:j + lo.shape[1]])
        return F.cross_entropy(lo.view(-1, self.head.out_features), y.reshape(-1), reduction="sum")

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
        elif FUSED and self.training:                  # злитий вихід + похибка: логіти не зберігаються й не рахуються двічі
            loss = self._fused_ce(hn, y)
        else:
            loss = 0
            for j in range(0, hn.shape[1], 8192):
                part = (hn[:, j:j + 8192], y[:, j:j + 8192])
                loss = loss + (checkpoint(self._ce_part, *part, j, use_reentrant=False) if self.training
                               else self._ce_part(*part, j))
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
        return FullLM2(vocab, d=d, heads=d // 64, moe="fixed", n_exp=int(e), bank=True, heavy=int(h) if h else 14,
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
