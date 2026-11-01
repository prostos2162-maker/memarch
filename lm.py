"""Мовні моделі для порівняння на справжньому тексті (FineWeb-Edu).

transformer  — Llama-подібний, увага через SDPA (FlashAttention).
cards        — без уваги: короткий хвіст (згортка) у кожному шарі + картки з
               закладками й перечитуванням у кількох шарах.
tail         — те саме, що cards, але без карток (щоб бачити, що дають саме картки).
img_sampled  — трансформер, вихід через готові образи слів + пачка неправильних слів.
planK        — планувальник + мовець, думка по K токенів (plan8, plan32, plan128).
planwK       — те саме, але мовець бачить останні 2K токенів (свою і попередню думку).
planwcK      — planwK + картки в мовці (дістати точне з далекого минулого).
T300 / S300_NN — те саме на ~300M (ширина 1024): трансформер 16 шарів проти здивувань.
surpNN       — думати на здивуваннях: важкі шари лише на NN% найнесподіваніших слів.
surpimgNN    — surpNN + вихід через образи слів GPT-2 (повний словник). Порівнювати з img_full.
sketchK      — планувальник дає ескіз (8 векторів) і меню (256 слів), мовець вибирає з меню.
planwimgK    — planwK + вихід через образи слів і пачку неправильних слів.
img_full     — те саме, але порівняння з усіма словами (щоб відділити «позичені образи»
               від «дешевого виходу»).

Деталі реалізації прототипу карток (не з початкового задуму, переглянути):
  - картки стоять у шарах card_layers, а не в кожному;
  - «книжка на полиці» = окремі вбудовування сирих токенів розміру d_book;
  - перечитування починається з самої закладки (зсув 0) і йде до max_read-1.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline import Block as AttnBlock


FUSED = os.environ.get("FUSED", "")                  # "", "ours", "liger" — злиті кернели для SwiGLU і похибки


@torch.compiler.disable                               # Liger не вбудовується в torch.compile (падає dynamo)
def _liger_swiglu(gu):
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    g, u = gu.chunk(2, -1)
    return LigerSiLUMulFunction.apply(g.contiguous(), u.contiguous())


class FFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.up = nn.Linear(d, 2 * 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        gu = self.up(self.norm(x))
        if FUSED == "ours":                          # злиті кернели (fused_ops.py) / Liger — порівняння
            import fused_ops
            return x + self.down(fused_ops.swiglu(gu))
        if FUSED == "liger":
            return x + self.down(_liger_swiglu(gu))
        g, u = gu.chunk(2, -1)
        return x + self.down(F.silu(g) * u)


class Tail(nn.Module):
    """Короткий хвіст: бачить лише останні `width` токенів."""

    def __init__(self, d, width=4):
        super().__init__()
        self.width = width
        self.norm = nn.RMSNorm(d)
        self.conv = nn.Conv1d(d, d, width, groups=d)
        self.gate = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        n = self.norm(x)
        c = self.conv(F.pad(n.transpose(1, 2), (self.width - 1, 0))).transpose(1, 2)
        return x + self.out(c * F.silu(self.gate(n)))


class CardRead(nn.Module):
    """Пошук картки за лицем і перечитування сирого тексту за закладкою."""

    def __init__(self, d, d_book, top_k=4, max_read=4, d_face=64):
        super().__init__()
        self.top_k, self.max_read, self.d_face = top_k, max_read, d_face
        self.norm = nn.RMSNorm(d)
        self.face = nn.Linear(d, d_face, bias=False)
        self.query = nn.Linear(d, d_face, bias=False)
        self.offset = nn.Parameter(torch.randn(max_read, d_book) * 0.02)
        self.halt_v = nn.Linear(d_book, 1)
        self.halt_h = nn.Linear(d, 1, bias=False)
        self.out = nn.Linear(d_book, d, bias=False)

    def forward(self, x, book):
        # усе в fp32: на torch 2.11 змішані bf16/fp32 тут ламають backward
        with torch.autocast("cuda", enabled=False):
            return self._forward(x.float(), book.float())

    def _forward(self, x, book):
        B, T, _ = x.shape
        n = self.norm(x)
        scores = self.query(n) @ self.face(n).transpose(1, 2) / math.sqrt(self.d_face)
        pos = torch.arange(T, device=x.device)
        scores = scores.masked_fill(pos[None, :] >= pos[:, None], -1e4)
        top_s, top_i = scores.topk(self.top_k, -1)                    # (B,T,K)
        w = F.softmax(top_s.float(), -1).type_as(x) * (top_s > -1e3)

        R = self.max_read
        rpos = top_i[..., None] + torch.arange(R, device=x.device)     # (B,T,K,R)
        readable = rpos <= pos[None, :, None, None]
        rpos = rpos.clamp(max=T - 1)
        Db = book.shape[-1]
        v = book.gather(1, rpos.reshape(B, -1, 1).expand(-1, -1, Db)).view(B, T, self.top_k, R, Db)
        v = v + self.offset

        halt = torch.sigmoid(self.halt_v(v).squeeze(-1) + self.halt_h(n)[:, :, None])
        halt = torch.where(readable, halt, torch.ones_like(halt))
        halt = torch.cat([halt[..., :-1], torch.ones_like(halt[..., -1:])], -1)
        survive = torch.cumprod(torch.cat([torch.ones_like(halt[..., :1]), 1 - halt[..., :-1]], -1), -1)
        read = ((survive * halt * readable)[..., None] * v).sum(-2)  # (B,T,K,Db)
        n_read = (survive * readable).sum(-1)

        got = (w[..., None] * read).sum(-2)
        cost = (w * n_read).sum(-1).mean()
        return x + self.out(got), cost


class TransformerLM(nn.Module):
    def __init__(self, vocab, d=512, layers=8, heads=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(AttnBlock(d, heads) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h)), torch.zeros((), device=x.device)


class CardsLM(nn.Module):
    def __init__(self, vocab, d=512, layers=8, d_book=128, card_layers=(3, 6), use_cards=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.book = nn.Embedding(vocab, d_book)       # сирий текст «на полиці»
        self.tails = nn.ModuleList(Tail(d) for _ in range(layers))
        self.ffns = nn.ModuleList(FFN(d) for _ in range(layers))
        self.cards = nn.ModuleDict(
            {str(i): CardRead(d, d_book) for i in card_layers} if use_cards else {})
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        h = self.emb(x)
        book = self.book(x) if len(self.cards) else None
        cost = torch.zeros((), device=x.device)
        for i, (tail, ffn) in enumerate(zip(self.tails, self.ffns)):
            h = tail(h)
            if str(i) in self.cards:
                h, c = self.cards[str(i)](h, book)
                cost = cost + c
            h = ffn(h)
        return self.head(self.norm(h)), cost


def load_word_images(vocab):
    """Готові образи слів: вбудовування GPT-2 (768 чисел на токен), заморожені."""
    if os.environ.get("FAKE_IMAGES"):                 # для локального тесту без завантаження
        return torch.randn(vocab, 768) * 0.1
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    wte = load_file(hf_hub_download("gpt2", "model.safetensors"))["wte.weight"]
    out = torch.zeros(vocab, wte.shape[1])
    out[:len(wte)] = wte
    return out


class ImageOut(nn.Module):
    """Вихід через готові образи слів.

    Модель видає образ «що хочу сказати» (proj), він порівнюється з фіксованими
    образами слів. sampled: на тренуванні лише з правильним словом і n_neg
    випадковими неправильними (з поправкою на частоту слова); full: з усіма.
    На оцінці завжди з усіма, тож val loss чесний і порівнянний з трансформером.
    """

    def __init__(self, d, vocab, sampled, n_neg=1024, n_real=50257):
        super().__init__()
        self.sampled, self.n_neg, self.n_real = sampled, n_neg, n_real
        img = load_word_images(vocab)
        self.register_buffer("img", img / img.norm(dim=-1, keepdim=True).mean())
        self.proj = nn.Linear(d, img.shape[1], bias=False)
        self.register_buffer("log_q", torch.full((vocab,), -math.log(n_real)))

    def set_unigram(self, counts):
        q = (counts.float() + 1) / (counts.float() + 1)[:self.n_real].sum()
        q[self.n_real:] = 1e-12
        self.log_q.copy_(q.log())

    def logits(self, h):
        lo = self.proj(h) @ self.img.T.type_as(h)
        lo[..., self.n_real:] = -1e4
        return lo

    def loss(self, h, y, train):
        if not (train and self.sampled):
            return F.cross_entropy(self.logits(h).float().view(-1, self.img.shape[0]), y.reshape(-1))
        z = self.proj(h).reshape(-1, self.img.shape[1])                       # (N, 768)
        y = y.reshape(-1)
        neg = torch.multinomial(self.log_q.exp(), self.n_neg, replacement=True)
        pos_lo = (z * self.img[y].type_as(z)).sum(-1, keepdim=True) - self.log_q[y, None]
        neg_lo = z @ self.img[neg].T.type_as(z) - self.log_q[neg][None]
        neg_lo = neg_lo.masked_fill(neg[None] == y[:, None], -1e4)            # випадково влучили в правильне
        lo = torch.cat([pos_lo, neg_lo], -1).float()
        return F.cross_entropy(lo, torch.zeros(len(y), dtype=torch.long, device=y.device))


class ImagePlannerLM(nn.Module):
    """planwK + вихід через готові образи слів і пачку неправильних слів."""

    def __init__(self, vocab, k):
        super().__init__()
        self.core = PlannerSpeakerLM(vocab, k=k, window=2 * k)
        del self.core.head
        self.core.head = nn.Identity()                 # ядро віддає стан, а не логіти
        self.out = ImageOut(512, vocab, sampled=True)

    def forward(self, x):
        return self.core(x)

    def train_loss(self, h, y, train=True):
        return self.out.loss(h, y, train)


class ImageTransformerLM(TransformerLM):
    """Той самий трансформер, змінено лише вихід на словник."""

    def __init__(self, vocab, sampled):
        super().__init__(vocab)
        del self.head
        self.out = ImageOut(self.norm.normalized_shape[0], vocab, sampled)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.norm(h), torch.zeros((), device=x.device)

    def train_loss(self, h, y, train=True):
        return self.out.loss(h, y, train)


class WindowBlock(AttnBlock):
    """Блок уваги, що бачить лише останні `window` токенів (ковзне вікно)."""

    def forward(self, x, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        from baseline import rope
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        g, u = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)


class PlannerSpeakerLM(nn.Module):
    """Планувальник + мовець.

    Текст ділиться на думки по `k` токенів (поки фіксовано; далі — за «зарядом»).
    Планувальник (великий, `p_layers`) працює раз на думку: бачить усі попередні
    думки і видає намір для наступної. Мовець (маленький, `s_layers`) працює на
    кожному токені, але бачить лише свою думку і намір, і нарощує слова навколо нього.
    Намір вчиться сам: він має допомогти мовцю вгадати всі слова думки.
    """

    def __init__(self, vocab, k, d=512, p_layers=8, s_layers=3, heads=8, window=0, cards=False):
        super().__init__()
        self.k = k
        self.window = window      # 0: мовець бачить лише свою думку; >0: останні `window` токенів
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(k, d) * 0.02)          # місце слова в думці
        self.pool_q = nn.Parameter(torch.randn(4, d) * 0.02)       # стискає думку в один вектор
        self.pool_out = nn.Linear(4 * d, d, bias=False)
        self.start = nn.Parameter(torch.randn(d) * 0.02)           # намір для першої думки
        self.planner = nn.ModuleList(AttnBlock(d, heads) for _ in range(p_layers))
        self.p_norm = nn.RMSNorm(d)
        self.intent = nn.Linear(d, d, bias=False)
        # картки: мовець після першого шару може дістати точне з далекого минулого
        self.book = nn.Embedding(vocab, 128) if cards else None
        self.cards = CardRead(d, 128) if cards else None
        Blk = WindowBlock if window else AttnBlock
        self.speaker = nn.ModuleList(Blk(d, heads) for _ in range(s_layers))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        B, T = x.shape
        k, n = self.k, T // self.k
        e = self.emb(x).view(B, n, k, -1) + self.pos               # (B,n,k,d)

        # стиснути кожну думку в вектор (увага 4 запитами по словах думки)
        att = torch.einsum("bnkd,qd->bnqk", e, self.pool_q).softmax(-1)
        blocks = self.pool_out(torch.einsum("bnqk,bnkd->bnqd", att, e).flatten(2))  # (B,n,d)

        # планувальник: думка j бачить лише думки 0..j-1
        p = blocks
        for b in self.planner:
            p = b(p)
        p = self.intent(self.p_norm(p))
        plan = torch.cat([self.start.expand(B, 1, -1).type_as(p), p[:, :-1]], 1)  # (B,n,d)

        # мовець: з наміром на кожному слові
        h = e + plan[:, :, None]
        if self.window:
            # бачить останні `window` токенів, тож не сліпне на межі думки
            h = h.reshape(B, T, -1)
            pos = torch.arange(T, device=x.device)
            dist = pos[:, None] - pos[None, :]
            mask = (dist >= 0) & (dist < self.window)
            cost = torch.zeros((), device=x.device)
            for i, b in enumerate(self.speaker):
                h = b(h, mask)
                if i == 0 and self.cards is not None:
                    h, cost = self.cards(h, self.book(x))
            return self.head(self.norm(h)), cost
        h = h.reshape(B * n, k, -1)                                # кожна думка окремо
        for b in self.speaker:
            h = b(h)
        return self.head(self.norm(h.view(B, T, -1))), torch.zeros((), device=x.device)


class SketchLM(nn.Module):
    """Планувальник дає ескіз і меню слів, мовець вибирає з меню.

    Раз на думку (k токенів) планувальник, бачачи лише попередні думки, видає:
      - ескіз: M векторів (якорі й підказки, що має бути між ними);
      - меню: `menu` слів, які можуть знадобитись у цій думці.
    Мовець (ковзне вікно 2k) на кожному слові дивиться на потрібну частину ескізу
    і вибирає слово з меню. Якщо потрібного слова в меню немає, він каже «інше»,
    і тоді (лише для таких слів) вибір іде з усього словника.
    """

    def __init__(self, vocab, k=32, d=512, p_layers=8, s_layers=3, heads=8, M=8, menu=256):
        super().__init__()
        self.k, self.M, self.n_menu, self.window = k, M, menu, 2 * k
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(k, d) * 0.02)
        self.pool_q = nn.Parameter(torch.randn(4, d) * 0.02)
        self.pool_out = nn.Linear(4 * d, d, bias=False)
        self.planner = nn.ModuleList(AttnBlock(d, heads) for _ in range(p_layers))
        self.p_norm = nn.RMSNorm(d)
        self.start = nn.Parameter(torch.randn(d) * 0.02)             # план перед першою думкою
        self.sketch = nn.Linear(d, M * d, bias=False)
        self.menu_head = nn.Linear(d, vocab, bias=False)
        self.speaker = nn.ModuleList(WindowBlock(d, heads) for _ in range(s_layers))
        self.c_norm = nn.ModuleList(nn.RMSNorm(d) for _ in range(s_layers))
        self.c_q = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(s_layers))
        self.c_out = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(s_layers))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)                   # образи слів мовця
        self.other = nn.Linear(d, 1)                                  # «потрібного слова в меню немає»
        self.stats = {}

    def forward(self, x):
        B, T = x.shape
        k, n, M = self.k, T // self.k, self.M
        e = self.emb(x).view(B, n, k, -1) + self.pos
        att = torch.einsum("bnkd,qd->bnqk", e, self.pool_q).softmax(-1)
        blocks = self.pool_out(torch.einsum("bnqk,bnkd->bnqd", att, e).flatten(2))

        p = blocks
        for b in self.planner:
            p = b(p)
        p = self.p_norm(p)
        # план для думки j — з думок 0..j-1
        p = torch.cat([self.start.expand(B, 1, -1).type_as(p), p[:, :-1]], 1)   # (B,n,d)
        sk = self.sketch(p).view(B, n, M, -1)                                   # ескіз
        menu_logits = self.menu_head(p)                                         # (B,n,V), раз на думку
        menu_idx = menu_logits.detach().float().topk(self.n_menu, -1).indices   # меню

        h = (e + sk.mean(2)[:, :, None]).reshape(B, T, -1)
        pos = torch.arange(T, device=x.device)
        dist = pos[:, None] - pos[None, :]
        mask = (dist >= 0) & (dist < self.window)
        scale = 1 / math.sqrt(h.shape[-1])
        for blk, cn, cq, co in zip(self.speaker, self.c_norm, self.c_q, self.c_out):
            h = blk(h, mask)
            q = cq(cn(h)).view(B, n, k, -1)                                     # на яку частину ескізу дивитись
            a = (torch.einsum("bnkd,bnmd->bnkm", q, sk) * scale).softmax(-1)
            h = h + co(torch.einsum("bnkm,bnmd->bnkd", a, sk).reshape(B, T, -1))
        return (self.norm(h), menu_idx, menu_logits), torch.zeros((), device=x.device)

    def train_loss(self, out, y, train=True):
        hn, menu_idx, menu_logits = out
        B, T, d = hn.shape
        k, n = self.k, T // self.k
        yb = y.reshape(B, n, k)
        hb = hn.view(B, n, k, d)

        # меню вчиться містити слова своєї думки
        menu_loss = -F.log_softmax(menu_logits.float(), -1).gather(-1, yb).mean()

        hit = menu_idx[:, :, None, :] == yb[..., None]                         # (B,n,k,menu)
        inmenu = hit.any(-1)
        slot = hit.float().argmax(-1)
        Wm = self.head.weight[menu_idx]                                         # (B,n,menu,d)
        lo = torch.cat([torch.einsum("bnkd,bnmd->bnkm", hb, Wm.type_as(hb)), self.other(hb)], -1).float()
        tgt = torch.where(inmenu, slot, torch.full_like(slot, self.n_menu))
        logp_menu = F.log_softmax(lo, -1)
        self.stats["miss"] = (~inmenu).float().mean().item()

        miss = ~inmenu
        nm = int(miss.sum())
        tok_menu = menu_idx[:, :, None, :].expand(B, n, k, self.n_menu)
        if train:
            loss = F.nll_loss(logp_menu.view(-1, self.n_menu + 1), tgt.view(-1))
            if nm:
                full = (hb[miss] @ self.head.weight.T.type_as(hb)).float()      # лише для слів поза меню
                full = full.scatter(1, tok_menu[miss], -1e4)
                loss = loss + F.cross_entropy(full, yb[miss], reduction="sum") / y.numel()
            return loss + 0.5 * menu_loss
        # оцінка: точна ймовірність кожного слова, порівнянна з трансформером
        lp = logp_menu.gather(-1, tgt[..., None]).squeeze(-1)
        if nm:
            full = (hb[miss] @ self.head.weight.T.type_as(hb)).float()
            full = full.scatter(1, tok_menu[miss], -1e4)
            lp_miss = F.log_softmax(full, -1).gather(-1, yb[miss][:, None]).squeeze(-1)
            lp = lp.clone()
            lp[miss] = lp[miss] + lp_miss
        return -lp.mean()


def _rope_at(x, pos, base=10000.0):
    """RoPE з довільними позиціями (pos: (N,)), x: (B,H,N,D)."""
    D = x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, D, 2, device=x.device).float() / D)
    ang = pos.float()[:, None] * inv[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).type_as(x)


_flex = None


def _flex_attention(q, k, v, block_mask):
    global _flex
    if _flex is None:
        from torch.nn.attention.flex_attention import flex_attention
        # без Triton (Windows, NO_COMPILE=1) — повільний, але правильний режим для перевірки
        _flex = flex_attention if os.environ.get("NO_COMPILE") else torch.compile(flex_attention, dynamic=False)
    return _flex(q, k, v, block_mask=block_mask)


@torch.compiler.disable
def _pack_plan(anchor):
    """План упаковки якорів (поза компіляцією, щоб довжина була звичайним числом):
    які слова, їхні номери у своєму тексті, і маска «той самий текст і не пізніше»."""
    from torch.nn.attention.flex_attention import create_block_mask
    B, T = anchor.shape
    dev = anchor.device
    flat = anchor.reshape(-1).nonzero().squeeze(1)                # усі якорі батчу, по порядку
    N = flat.numel()
    # довжина округлюється до кратного 512: різних форм лише кілька, кожна компілюється раз
    Np = max(512, -(-N // 512) * 512)
    seq = flat // T
    counts = anchor.sum(1)
    pos = torch.arange(N, device=dev) - (counts.cumsum(0) - counts)[seq]   # номер якоря у своєму тексті
    pad = Np - N
    seq_p = torch.cat([seq, -1 - torch.arange(pad, device=dev)])           # доповнення бачить лише себе
    pos_p = torch.cat([pos, torch.zeros(pad, dtype=torch.long, device=dev)])
    idx_p = torch.cat([flat, torch.zeros(pad, dtype=torch.long, device=dev)])
    bm = create_block_mask(lambda b, hh, q, kv: (seq_p[q] == seq_p[kv]) & (kv <= q),
                           None, None, Np, Np, device=dev)
    return flat, idx_p, pos_p, bm


class PackedBlock(AttnBlock):
    """Блок уваги для якорів з усього батчу, упакованих в одну щільну послідовність.
    Маска (FlexAttention): той самий текст і не пізніше. Порожнечі немає."""

    def forward(self, x, block_mask, pos):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = _flex_attention(_rope_at(q, pos), _rope_at(k, pos), v, block_mask)
        x = x + self.o(a.transpose(1, 2).reshape(B, N, D))
        g, u = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)


class SurpriseLM(nn.Module):
    """Думати на здивуваннях.

    1. Легка частина (`light` шарів) читає кожне слово і пробує вгадати наступне.
    2. Здивування на слові t = наскільки погано легка частина його вгадала.
       Якщо воно вище порогу, слово — якір. Поріг підлаштовується так, щоб
       якорів була частка `frac` (вирішується по кожному слову окремо, без
       погляду в майбутнє).
    3. Важка частина (`heavy` шарів) працює лише на якорях і бачить лише
       попередні якорі.
    4. Фінальний легкий шар дає всім словам підглянути, що надумала важка частина.
    """

    def __init__(self, vocab, d=512, light=2, heavy=8, final=1, heads=8, frac=0.2, images=False, packed=False,
                 select="threshold", chunk=128, cap=0.25, sit_cards=0):
        super().__init__()
        # sit_cards: картки досвіду в моделі (Ідея 5б) — за ситуацією (що прочитано досі) трохи
        # підкручують роботу легких шарів, як «ще до повороту тримав кермо інакше»
        self.sit_cards = sit_cards
        if sit_cards:
            self.sit_keys = nn.Parameter(torch.randn(sit_cards, d) * 0.02)
            self.sit_gain = nn.Parameter(torch.zeros(sit_cards, d))
        self.frac, self.packed = frac, packed
        # select: як вибирати якорі
        #   threshold — здивування вище порогу (змінна кількість якорів)
        #   cap       — поріг, але не більше `cap` частки тексту, перші за порядком (стала форма, причинно)
        #   quota     — рівно `frac` найнесподіваніших у кожному шматку з `chunk` слів (стала форма;
        #               на тренуванні вибір у шматку бачить наступні слова, тож оцінка йде правилом cap)
        self.select, self.chunk, self.cap = select, chunk, cap
        self.emb = nn.Embedding(vocab, d)
        self.light = nn.ModuleList(AttnBlock(d, heads) for _ in range(light))
        self.l_norm = nn.RMSNorm(d)
        self.l_head = nn.Sequential(nn.Linear(d, 128, bias=False), nn.Linear(128, vocab, bias=False))
        # packed: якорі всього батчу в одній щільній послідовності (без доповнення порожнечею)
        self.heavy = nn.ModuleList((PackedBlock if packed else WindowBlock)(d, heads) for _ in range(heavy))
        self.final = nn.ModuleList(AttnBlock(d, heads) for _ in range(final))
        self.norm = nn.RMSNorm(d)
        # вихід: власні образи слів або готові з GPT-2 (повний словник)
        self.out = ImageOut(d, vocab, sampled=False) if images else None
        self.head = None if images else nn.Linear(d, vocab, bias=False)
        self.register_buffer("tau", torch.tensor(5.0))               # поріг здивування
        self.stats = {}

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x)
        if self.sit_cards:
            # ситуація на слові t — середнє прочитаного до t включно (без погляду вперед)
            # (середнє через трикутну матрицю, а не cumsum: на cumsum падає компілятор inductor)
            tri = torch.ones(T, T, device=x.device).tril_()
            sit = (tri / tri.sum(1, keepdim=True)).type_as(h) @ h.detach()
            p = (F.normalize(sit, dim=-1) @ F.normalize(self.sit_keys, dim=-1).T * 8).softmax(-1)
            h = h * (1 + (p @ self.sit_gain).type_as(h))
        for b in self.light:
            h = b(h)
        lh = self.l_norm(h)

        # здивування на слові t: як погано легка частина вгадала його на кроці t-1
        with torch.no_grad():
            s = torch.full((B, T), 1e4, device=x.device)
            # -log p(x_t) = logsumexp - logit, шматками і без збереження для градієнта:
            # повна таблиця ймовірностей усіх слів не живе в пам'яті цілком
            lhd = lh.detach()
            for i in range(0, B, max(1, B // 4)):
                lo = self.l_head(lhd[i:i + max(1, B // 4), :-1])
                s[i:i + lo.shape[0], 1:] = (torch.logsumexp(lo, -1)
                                            - lo.gather(-1, x[i:i + lo.shape[0], 1:, None]).squeeze(-1)).float()
            anchor = s > self.tau            # спершу вибір старим порогом: він не знає цього батчу
            if self.training:
                q = torch.quantile(s[:, 1:].reshape(-1)[:100_000], 1 - self.frac)
                self.tau.mul_(0.9).add_(0.1 * q)
            self.stats["anchors"] = anchor.float().mean()          # тензор: без зупинки на хост
        if self.select != "threshold":
            h = self._heavy_static(h, s, anchor)
        elif self.packed:
            h = self._heavy_packed(h, anchor)
        else:
            h = self._heavy_padded(h, anchor)
        for b in self.final:
            h = b(h)
        hn = self.norm(h)
        logits = self.out.logits(hn) if self.out is not None else self.head(hn)
        return (logits, lh), torch.zeros((), device=x.device)

    def _heavy_packed(self, h, anchor):
        B, T, d = h.shape
        flat, idx_p, pos_p, bm = _pack_plan(anchor)
        flat_h = h.reshape(B * T, d)
        a_in = flat_h[idx_p][None]
        a = a_in
        for blk in self.heavy:
            a = blk(a, bm, pos_p)
        return flat_h.index_add(0, flat, (a - a_in)[0, :flat.shape[0]]).view(B, T, d)

    def _heavy_static(self, h, s, anchor):
        """Стала кількість якорів на текст: без змінних розмірів, тож крок можна записати графом."""
        B, T, d = h.shape
        with torch.no_grad():
            pos = torch.arange(T, device=h.device)
            if self.select == "quota" and self.training:
                k = max(1, round(self.chunk * self.frac))
                sc = s.view(B, T // self.chunk, self.chunk)
                top = sc.topk(k, -1).indices.sort(-1).values             # найнесподіваніші в шматку
                idx = (top + torch.arange(0, T, self.chunk, device=h.device)[None, :, None]).view(B, -1)
                valid = torch.ones_like(idx, dtype=torch.bool)
            else:
                C = max(1, round(T * self.cap))
                keep = anchor & (anchor.cumsum(1) <= C)                   # перші C якорів за порядком
                idx = ((~keep).long() * T + pos).argsort(-1)[:, :C]
                valid = keep.gather(1, idx)
            self.stats["anchors"] = valid.float().mean() * idx.shape[1] / T
        A = idx.shape[1]
        a_in = h.gather(1, idx[..., None].expand(-1, -1, d))
        causal = torch.ones(A, A, dtype=torch.bool, device=h.device).tril()
        mask = (causal[None] & valid[:, None, :]) | torch.eye(A, dtype=torch.bool, device=h.device)[None]
        a = a_in
        for b in self.heavy:
            a = b(a, mask[:, None])
        return h.scatter_add(1, idx[..., None].expand(-1, -1, d), (a - a_in) * valid[..., None])

    def _heavy_padded(self, h, anchor):
        B, T, d = h.shape
        x = h
        with torch.no_grad():
            pos = torch.arange(T, device=x.device)
            order = ((~anchor).long() * T + pos).argsort(-1)          # спершу якорі, по порядку
            A = max(1, int(anchor.sum(1).max()))
            idx = order[:, :A]
            valid = anchor.gather(1, idx)

        # важка частина: лише якорі, і кожен бачить лише попередні якорі
        a_in = h.gather(1, idx[..., None].expand(-1, -1, d))
        causal = torch.ones(A, A, dtype=torch.bool, device=x.device).tril()
        mask = (causal[None] & valid[:, None, :]) | torch.eye(A, dtype=torch.bool, device=x.device)[None]
        a = a_in
        for b in self.heavy:
            a = b(a, mask[:, None])
        return h.scatter_add(1, idx[..., None].expand(-1, -1, d), (a - a_in) * valid[..., None])

    def train_loss(self, out, y, train=True):
        logits, lh = out
        V = logits.shape[-1]
        loss = F.cross_entropy(logits.float().view(-1, V), y.reshape(-1))
        if train:   # легка частина вчиться вгадувати, інакше здивування нічого не значить
            # (на кожній 4-й позиції — щоб не платити за другий повний вихід на словник)
            l_logits = self.l_head(lh[:, ::4])
            loss = loss + 0.3 * F.cross_entropy(l_logits.float().reshape(-1, V), y[:, ::4].reshape(-1))
        return loss


def build(name, vocab):
    if name in ("F300", "F300ck", "T300ck", "Fsmall"):   # повна модель v1 — full.py
        from full import build_full
        return build_full(name, vocab)
    if name.startswith(("Fv2", "Fv3")) or name in ("Fn", "Fn_nomem", "Fns", "Fns_nomem"):                    # повна модель v2 / v3 — full2.py
        from full2 import build_full2
        return build_full2(name, vocab)
    # ~300M: ширина 1024. Трансформер 16 шарів; здивування: 4 легкі + 14 важких + 2 фінальні
    if name == "T300":
        return TransformerLM(vocab, d=1024, layers=16, heads=16)
    if name.startswith("Td") and "L" in name:      # Td512L16 — звичайний трансформер (контроль у гонці на ПК)
        d, L = name[2:].split("L")
        return TransformerLM(vocab, d=int(d), layers=int(L), heads=int(d) // 128)
    if name.startswith("S300_"):                    # S300_20: частка якорів 20%
        return SurpriseLM(vocab, d=1024, light=4, heavy=14, final=2, heads=16, frac=int(name[5:]) / 100)
    if name.startswith("S300p_"):                   # S300p_20: те саме, якорі упаковані (без порожнечі)
        return SurpriseLM(vocab, d=1024, light=4, heavy=14, final=2, heads=16, frac=int(name[6:]) / 100, packed=True)
    if name.startswith("S300c_"):                   # S300c_20: поріг + стеля 25% (стала форма, причинно)
        return SurpriseLM(vocab, d=1024, light=4, heavy=14, final=2, heads=16, frac=int(name[6:]) / 100, select="cap")
    if name.startswith("S300q_"):                   # S300q_20: квота 20% у кожному шматку з 128 слів
        return SurpriseLM(vocab, d=1024, light=4, heavy=14, final=2, heads=16, frac=int(name[6:]) / 100, select="quota")
    if name.startswith("surpcx"):                   # surpcxNN: + картки досвіду в моделі (Ідея 5б)
        return SurpriseLM(vocab, frac=int(name[6:]) / 100, select="cap", sit_cards=4)
    if name.startswith("surpc"):
        return SurpriseLM(vocab, frac=int(name[5:]) / 100, select="cap")
    if name.startswith("surpq"):
        return SurpriseLM(vocab, frac=int(name[5:]) / 100, select="quota")
    if name.startswith("surpp"):                    # surppNN: маленька упакована версія (для перевірок)
        return SurpriseLM(vocab, frac=int(name[5:]) / 100, packed=True)
    if name.startswith("surpimg"):                  # surpimgNN: здивування + образи слів GPT-2
        return SurpriseLM(vocab, frac=int(name[7:]) / 100, images=True)
    if name.startswith("surp"):                     # surpNN: частка якорів NN% (surp20)
        return SurpriseLM(vocab, frac=int(name[4:]) / 100)
    if name.startswith("sketch"):                   # sketchK: ескіз + меню, думка по K токенів
        return SketchLM(vocab, k=int(name[6:]))
    if name.startswith("planwimg"):                 # planwimgK: planwK + образи слів
        return ImagePlannerLM(vocab, k=int(name[8:]))
    if name.startswith("planwc"):                   # planwcK: + картки в мовці
        k = int(name[6:])
        return PlannerSpeakerLM(vocab, k=k, window=2 * k, cards=True)
    if name.startswith("planw"):                    # planwK: думка K, мовець бачить 2K токенів
        k = int(name[5:])
        return PlannerSpeakerLM(vocab, k=k, window=2 * k)
    if name.startswith("plan"):
        return PlannerSpeakerLM(vocab, k=int(name[4:]))
    if name == "transformer":
        return TransformerLM(vocab)
    if name == "img_sampled":
        return ImageTransformerLM(vocab, sampled=True)
    if name == "img_full":
        return ImageTransformerLM(vocab, sampled=False)
    if name == "cards":
        return CardsLM(vocab)
    if name == "tail":
        return CardsLM(vocab, use_cards=False)
    raise ValueError(name)
