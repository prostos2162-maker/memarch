"""Повна модель: усі ідеї разом (ARCHITECTURE.md).

  1. Легкі шари бачать лише вікно з останніх `window` слів (близький контекст;
     ціна на слово не росте з довжиною тексту).
  2. Здивування легкої частини → якорі (поріг + стеля 25% на текст).
  3. Якорі — це картки: важкі шари працюють лише на якорях і бачать усі попередні
     якорі тексту, тобто шукають картки за лицем, хоч як далеко в минулому.
  4. Результат важких шарів повертається на місця якорів; фінальні шари з вікном
     дають сусідам підглянути, що надумала важка частина.
  5. Перечитування за закладкою: на виході кожне слово може вибрати картку-якір
     і перечитати сирий текст біля неї (з ціною за кожен прочитаний токен).

F300 — ~300M (ширина 1024: 4 легкі, 14 важких, 2 фінальні), як S300c_20.
T300ck / F300ck — те саме з перерахунком активацій (для довгого контексту).
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import dev
from baseline import Block as AttnBlock
from baseline import rope
from lm import ImageOut, TransformerLM, _block_window_attention, _block_window_attention2, _splash_window_attention, _flex_attention, _rope_at  # noqa: F401  (ImageOut — на майбутнє)


_COMPILE = not os.environ.get("NO_COMPILE")


@torch.compiler.disable
def _masks(anchor_valid, T, A, window, device):
    """Маски уваги (поза компіляцією): вікно для легких шарів і «попередні якорі тексту» для важких."""
    from torch.nn.attention.flex_attention import create_block_mask
    win = None
    if T > window:
        # _compile: маска будується поблоково; без цього на 128k+ створюється повна таблиця T×T
        win = create_block_mask(lambda b, h, q, kv: (kv <= q) & (q - kv < window), None, None, T, T,
                                device=device, _compile=_COMPILE)
    valid = anchor_valid
    heavy = create_block_mask(lambda b, h, q, kv: ((kv <= q) & valid[b, kv]) | (q == kv),
                              valid.shape[0], None, A, A, device=device, _compile=_COMPILE)
    return win, heavy


_BLK = set(filter(None, os.environ.get("BLK", "").split(",")))   # перемикачі блоку для гонки: qk, zero


class FineMoE(nn.Module):
    """Легкий MoE (Стефан): E малих експертів SwiGLU, активні k на слово; обчислень — як у звичайного FFN
    (експерт = 1/k від ширини), знань — у E/k разів більше. Фіксована місткість, без зупинок хоста."""

    def __init__(self, d, E=32, k=4, hid=None):
        super().__init__()
        self.E, self.k = E, k
        h = hid or 4 * d // k
        self.router = nn.Linear(d, E, bias=False)
        self.up = nn.Parameter(torch.randn(E, d, 2 * h) * d ** -0.5)
        self.down = nn.Parameter(torch.randn(E, h, d) * h ** -0.5)
        self.aux = torch.zeros(())

    def forward(self, x):
        shp = x.shape
        xf = x.reshape(-1, shp[-1])
        N = xf.shape[0]
        logits = self.router(xf).float()
        gate, ex = logits.topk(self.k, -1)                         # (N, k)
        gate = gate.softmax(-1)
        prob = logits.softmax(-1)
        frac = F.one_hot(ex, self.E).float().sum(1).mean(0) / self.k
        self.aux = self.E * (frac * prob.mean(0)).sum()            # рівномірне навантаження
        C = int(1.25 * N * self.k / self.E) + 1
        e_flat = ex.reshape(-1)                                    # (N*k)
        order = e_flat.argsort(stable=True)
        es = e_flat[order]
        start = torch.searchsorted(es, torch.arange(self.E, device=x.device))
        slot = torch.arange(N * self.k, device=x.device) - start[es]
        keep = slot < C
        tok = order // self.k
        dest = torch.where(keep, es * C + slot, self.E * C)        # зайве — в «нікуди»
        buf = xf.new_zeros(self.E * C + 1, shp[-1])
        buf[dest] = xf[tok]
        hb = buf[:-1].view(self.E, C, -1)
        g, u = torch.bmm(hb, self.up.type_as(hb)).chunk(2, -1)
        yb = torch.bmm(F.silu(g) * u, self.down.type_as(hb)).reshape(self.E * C, -1)
        yb = torch.cat([yb, yb.new_zeros(1, yb.shape[-1])])
        w = gate.reshape(-1)[order] * keep
        y = xf.new_zeros(N, shp[-1]).index_add(0, tok, (yb[dest] * w[:, None].to(yb.dtype)).to(xf.dtype))
        return y.view(shp)


_LMOE = os.environ.get("LIGHT_MOE")                              # "32:4" — легкі шари як MoE
_LFFN = float(os.environ.get("LIGHT_FFN", 0) or 0)                 # ширина FFN легких шарів у d (0 — як було: 4d SwiGLU / 6d relu²)


class _Relu2(torch.autograd.Function):
    """relu(h)²: зберігається лише h (вперед пишеться один тензор), похідна 2·relu(h) — з h на зворотному."""

    @staticmethod
    def forward(ctx, h):
        ctx.save_for_backward(h)
        r = F.relu(h)
        return r * r

    @staticmethod
    def backward(ctx, g):
        h, = ctx.saved_tensors
        return g * (2 * F.relu(h)).to(g.dtype)


class FlexBlock(AttnBlock):
    """Блок уваги з довільною маскою FlexAttention і позиціями RoPE."""

    def __init__(self, d, heads):
        super().__init__(d, heads)
        if "zero" in _BLK:                    # вихід уваги й FFN з нуля: блок спершу — тотожність
            nn.init.zeros_(self.o.weight)
            nn.init.zeros_(self.down.weight)
        if "ve" in _BLK:                      # value embeddings: ворота на голову (з нуля → множник 1)
            self.ve_gate = nn.Linear(d, heads, bias=False)
            nn.init.zeros_(self.ve_gate.weight)
        if "x0" in _BLK:                      # сирі вбудовування слова до входу кожного шару
            self.x0_l = nn.Parameter(torch.zeros(()))
        self.cur_ve = self.cur_x0 = None
        if "relu2" in _BLK or "relu2c" in _BLK:   # FFN relu(x·up)²·down, ширина 6d: параметрів і множень як у SwiGLU 4d,
            self.up = nn.Linear(d, 6 * d, bias=False)       # без гейта (менше поелементного й байтів)
            self.down = nn.Linear(6 * d, d, bias=False)
        if _LFFN:                             # LIGHT_FFN=2 (Стефан, 02.11): легкі шари лише читають — FFN вужчий
            hid = max(256, int(round(_LFFN * d / 256)) * 256)   # (думання — у важких шарах на якорях)
            r2 = "relu2" in _BLK or "relu2c" in _BLK
            self.up = nn.Linear(d, hid if r2 else 2 * hid, bias=False)
            self.down = nn.Linear(hid, d, bias=False)
        if _LMOE:
            E, k = map(int, _LMOE.split(":"))
            self.moe = FineMoE(d, E, k)
            del self.up, self.down

    def forward(self, x, block_mask=None, pos=None):
        B, T, D = x.shape
        if self.cur_x0 is not None:
            x = x + self.x0_l * self.cur_x0
        if "qkv3" in _BLK:                   # три множення на зрізи тієї ж ваги: результат одразу в розкладці уваги
            n1 = self.n1(x)
            hd = D // self.heads
            q, k, v = (F.linear(n1, self.qkv.weight[i * D:(i + 1) * D]).view(B, T, self.heads, hd).transpose(1, 2)
                       for i in range(3))
        else:
            q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        if self.cur_ve is not None:           # v: (B, heads, T, hd)
            g = 2 * torch.sigmoid(self.ve_gate(x)).transpose(1, 2)[..., None]
            v = v + g * self.cur_ve.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        if "qk" in _BLK:                      # QK-norm: масштаб уваги не розповзається
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        if block_mask is None:                                        # коротко: звичайна каузальна (Flash)
            a = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        else:
            if pos is None:
                q, k = rope(q), rope(k)
            else:
                q, k = _rope_at(q, pos), _rope_at(k, pos)
            a = (({"block2": _block_window_attention2, "splash": _splash_window_attention}.get(block_mask[0], _block_window_attention))(q, k, v, block_mask[1])
                 if isinstance(block_mask, tuple) else _flex_attention(q, k, v, block_mask))
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        if hasattr(self, "moe"):
            return x + self.moe(self.n2(x))
        sel = getattr(self, "cur_sel", None)
        if sel is not None:                   # «напівпрозорі» слова (розмовник певен) — без FFN; рахуємо лише непрозорі
            idx, ok = sel
            xs = self.n2(x).gather(1, idx[..., None].expand(-1, -1, D))
            y = self.down(self._act(self.up(xs))) * ok[..., None].to(x.dtype)
            return x.scatter_add(1, idx[..., None].expand(-1, -1, D), y.type_as(x))
        return x + self.down(self._act(self.up(self.n2(x))))

    def _act(self, h):
        if "relu2c" in _BLK:                  # свій backward (зберігає лише h) — на v6e виявився повільнішим
            return _Relu2.apply(h)
        if "relu2" in _BLK or "relu2c" in _BLK:
            return F.relu(h).square()
        g, u = h.chunk(2, -1)
        return F.silu(g) * u


class AnchorCardRead(nn.Module):
    """Перечитування за закладкою: слово вибирає серед попередніх якорів-карток
    (за лицем — станом якоря після важких шарів) і перечитує сирий текст біля закладки.
    Рахується шматками по `chunk` слів, щоб таблиця «слово × якір» не займала всю пам'ять."""

    def __init__(self, d, d_book=128, top_k=4, max_read=4, d_face=64, chunk=4096):
        super().__init__()
        self.top_k, self.max_read, self.d_face, self.chunk = top_k, max_read, d_face, chunk
        self.norm = nn.RMSNorm(d)
        self.face = nn.Linear(d, d_face, bias=False)
        self.query = nn.Linear(d, d_face, bias=False)
        self.offset = nn.Parameter(torch.randn(max_read, d_book) * 0.02)
        self.halt_v = nn.Linear(d_book, 1)
        self.halt_h = nn.Linear(d, 1, bias=False)
        self.out = nn.Linear(d_book, d, bias=False)

    def forward(self, x, book, cards, idx, valid):
        with torch.autocast(dev.TYPE, enabled=False):
            outs, costs = [], []
            for s in range(0, x.shape[1], self.chunk):
                args = (x[:, s:s + self.chunk].float(), s, book.float(), cards.float(), idx, valid)
                # з перерахунком: таблиця «слово × картка» не зберігається для зворотного проходу
                o, c = checkpoint(self._part, *args, use_reentrant=False) if self.training else self._part(*args)
                outs.append(o)
                costs.append(c)
            return torch.cat(outs, 1), torch.stack(costs).mean()

    def _part(self, x, start, book, cards, idx, valid):
        B, Tq, _ = x.shape
        T = book.shape[1]
        n = self.norm(x)
        faces = self.face(self.norm(cards))                                    # (B,A,f)
        scores = self.query(n) @ faces.transpose(1, 2) / math.sqrt(self.d_face)  # (B,Tq,A)
        qpos = torch.arange(start, start + Tq, device=x.device)
        ok = valid[:, None, :] & (idx[:, None, :] < qpos[None, :, None])     # лише минулі картки
        scores = scores.masked_fill(~ok, -1e4)
        k = min(self.top_k, scores.shape[-1])
        top_s, top_j = scores.topk(k, -1)
        w = F.softmax(top_s, -1) * (top_s > -1e3)
        mark = idx.gather(1, top_j.reshape(B, -1)).view(B, Tq, k)             # закладки = позиції якорів

        R = self.max_read
        rpos = mark[..., None] + torch.arange(R, device=x.device)
        readable = rpos <= qpos[None, :, None, None]
        rpos = rpos.clamp(max=T - 1)
        Db = book.shape[-1]
        v = book.gather(1, rpos.reshape(B, -1, 1).expand(-1, -1, Db)).view(B, Tq, k, R, Db) + self.offset
        halt = torch.sigmoid(self.halt_v(v).squeeze(-1) + self.halt_h(n)[:, :, None])
        halt = torch.where(readable, halt, torch.ones_like(halt))
        halt = torch.cat([halt[..., :-1], torch.ones_like(halt[..., -1:])], -1)
        survive = torch.cumprod(torch.cat([torch.ones_like(halt[..., :1]), 1 - halt[..., :-1]], -1), -1)
        read = ((survive * halt * readable)[..., None] * v).sum(-2)
        got = (w[..., None] * read).sum(-2)
        cost = (w * (survive * readable).sum(-1)).sum(-1).mean()
        return x + self.out(got), cost


class FullLM(nn.Module):
    def __init__(self, vocab, d=1024, light=4, heavy=14, final=2, heads=16, frac=0.2, cap=0.25,
                 window=2048, ckpt=False):
        super().__init__()
        self.frac, self.cap, self.window, self.ckpt = frac, cap, window, ckpt
        self.emb = nn.Embedding(vocab, d)
        self.book = nn.Embedding(vocab, 128)                          # сирий текст «на полиці»
        self.light = nn.ModuleList(FlexBlock(d, heads) for _ in range(light))
        self.l_norm = nn.RMSNorm(d)
        self.l_head = nn.Sequential(nn.Linear(d, 128, bias=False), nn.Linear(128, vocab, bias=False))
        self.heavy = nn.ModuleList(FlexBlock(d, heads) for _ in range(heavy))
        self.final = nn.ModuleList(FlexBlock(d, heads) for _ in range(final))
        self.cards = AnchorCardRead(d)
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.register_buffer("tau", torch.tensor(5.0))
        self.stats = {}

    def _run(self, blk, *a):
        return checkpoint(blk, *a, use_reentrant=False) if self.ckpt and self.training else blk(*a)

    def forward(self, x):
        B, T = x.shape
        A = max(1, round(T * self.cap))
        h = self.emb(x)

        # 1. легкі шари з вікном
        with torch.no_grad():
            win, _ = _masks(torch.ones(B, A, dtype=torch.bool, device=x.device), T, A, self.window, x.device) \
                if T > self.window else (None, None)
        for b in self.light:
            h = self._run(b, h, win)
        lh = self.l_norm(h)

        # 2. здивування → якорі (поріг + стеля, перші за порядком)
        with torch.no_grad():
            s = torch.full((B, T), 1e4, device=x.device)
            lhd = lh.detach()
            step = max(1, min(B, 32768 // T))
            for i in range(0, B, step):
                for j in range(0, T - 1, 8192):                         # шматками: не тримати B×T×V цілим
                    lo = self.l_head(lhd[i:i + step, j:min(j + 8192, T - 1)])
                    tgt = x[i:i + step, j + 1:j + 1 + lo.shape[1], None]
                    s[i:i + lo.shape[0], j + 1:j + 1 + lo.shape[1]] = \
                        (torch.logsumexp(lo, -1) - lo.gather(-1, tgt).squeeze(-1)).float()
            anchor = s > self.tau
            if self.training:
                q = torch.quantile(s[:, 1:].reshape(-1)[:100_000], 1 - self.frac)
                self.tau.mul_(0.9).add_(0.1 * q)
            keep = anchor & (anchor.cumsum(1) <= A)
            pos = torch.arange(T, device=x.device)
            idx = ((~keep).long() * T + pos).argsort(-1)[:, :A]           # закладки якорів, по порядку
            valid = keep.gather(1, idx)
            _, heavy_mask = _masks(valid, T, A, self.window, x.device)
            self.stats["anchors"] = valid.float().mean() * A / T

        # 3. якорі-картки: важкі шари бачать усі попередні якорі тексту
        d = h.shape[-1]
        a_in = h.gather(1, idx[..., None].expand(-1, -1, d))
        a = a_in
        for b in self.heavy:
            a = self._run(b, a, heavy_mask)          # RoPE за номером слоту = номер якоря в тексті
        h = h.scatter_add(1, idx[..., None].expand(-1, -1, d), (a - a_in) * valid[..., None])

        # 4. фінальні шари з вікном
        for b in self.final:
            h = self._run(b, h, win)

        # 5. перечитування за закладкою (картки = якорі)
        h, cost = self.cards(h, self.book(x), a, idx, valid)
        return (self.norm(h), lh), cost

    def _ce_part(self, hn, y):
        return F.cross_entropy(self.head(hn).float().view(-1, self.head.out_features), y.reshape(-1), reduction="sum")

    def train_loss(self, out, y, train=True):
        hn, lh = out
        V = self.head.out_features
        # вихід на словник шматками по 8k слів (з перерахунком): на 128k+ повні логіти не влазять у пам'ять
        loss = 0
        for j in range(0, hn.shape[1], 8192):
            part = (hn[:, j:j + 8192], y[:, j:j + 8192])
            loss = loss + (checkpoint(self._ce_part, *part, use_reentrant=False) if self.training
                           else self._ce_part(*part))
        loss = loss / y.numel()
        if train:
            l_logits = self.l_head(lh[:, ::4])
            loss = loss + 0.3 * F.cross_entropy(l_logits.float().reshape(-1, V), y[:, ::4].reshape(-1))
        return loss


class CkptTransformer(TransformerLM):
    """Трансформер з перерахунком активацій (для довгого контексту)."""

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = checkpoint(b, h, use_reentrant=False) if self.training else b(h)
        return self.head(self.norm(h)), torch.zeros((), device=x.device)


def build_full(name, vocab):
    if name == "F300":
        return FullLM(vocab)
    if name == "F300ck":
        return FullLM(vocab, ckpt=True)
    if name == "T300ck":
        return CkptTransformer(vocab, d=1024, layers=16, heads=16)
    if name == "Fsmall":                            # для локальних перевірок
        return FullLM(vocab, d=256, light=1, heavy=2, final=1, heads=4, window=64)
    raise ValueError(name)
