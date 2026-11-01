"""Прототип v0: картки + закладки + точкове перечитування.

На кожному токені:
  1. короткий хвіст (каузальна згортка на кілька токенів) дає стан h;
  2. з h виходить лице картки (за чим її знайдуть) і запит (що шукаємо зараз);
     закладка картки = її позиція в тексті;
  3. запит порівнюється з лицями всіх попередніх карток, береться top-K.
     Вчиться міра збігу: м'який вибір серед K найкращих;
  4. за закладкою модель перечитує сирі токени після неї. На кожному кроці
     вирішує, чи зупинитись; кожен прочитаний токен коштує штраф (cost).

v0 порівнює запит з усіма картками (як увага, O(n^2)). З memory=MemoryConfig(...)
пошук іде лише серед карток, що лишились у свіжій і тьмяній пам'яті (див. memory.py).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory import MemoryConfig, simulate


class Tail(nn.Module):
    """Короткий хвіст: бачить лише останні `width` токенів."""

    def __init__(self, d, width=4):
        super().__init__()
        self.width = width
        self.norm = nn.RMSNorm(d)
        self.conv = nn.Conv1d(d, d, width, groups=d)
        self.up = nn.Linear(d, 2 * 2 * d, bias=False)
        self.down = nn.Linear(2 * d, d, bias=False)

    def forward(self, h):
        c = F.pad(self.norm(h).transpose(1, 2), (self.width - 1, 0))
        h = h + self.conv(c).transpose(1, 2)
        g, u = self.up(h).chunk(2, -1)
        return h + self.down(F.silu(g) * u)


class Cards(nn.Module):
    def __init__(self, vocab, d=128, layers=2, top_k=4, max_read=8, memory=None):
        super().__init__()
        self.d, self.top_k, self.max_read = d, top_k, max_read
        self.emb = nn.Embedding(vocab, d)
        self.book = nn.Embedding(vocab, d)          # сирий текст «на полиці»
        self.tails = nn.ModuleList(Tail(d) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        self.face = nn.Linear(d, d, bias=False)
        self.query = nn.Linear(d, d, bias=False)
        self.offset = nn.Parameter(torch.randn(max_read, d) * 0.02)  # де від закладки
        self.halt_v = nn.Linear(d, 1)
        self.halt_h = nn.Linear(d, 1, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.n_out = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.memory = memory
        self.mem_stats = {}

    def encode(self, x):
        h = self.emb(x)
        for t in self.tails:
            h = t(h)
        hn = self.norm(h)
        return h, hn, self.query(hn), self.face(hn), self.book(x)

    def read(self, h, hn, scores, qpos, book):
        """h, hn: стан запитів (B,Tq,d); scores: (B,Tq,T), заборонене = -1e4; qpos: позиції запитів."""
        B, Tq, T = scores.shape
        k = min(self.top_k, T)
        top_s, top_i = scores.topk(k, -1)                             # (B,Tq,K)
        has_card = top_s > -1e3
        w = F.softmax(top_s, -1) * has_card

        # 4. перечитування за закладкою
        R = self.max_read
        rpos = top_i[..., None] + torch.arange(1, R + 1, device=scores.device)  # (B,Tq,K,R)
        readable = rpos < qpos[None, :, None, None]                  # не заглядати в майбутнє
        rpos = rpos.clamp(max=T - 1)
        v = book.gather(1, rpos.view(B, -1, 1).expand(-1, -1, self.d)).view(B, Tq, k, R, self.d)
        v = v + self.offset

        halt = torch.sigmoid(self.halt_v(v).squeeze(-1) + self.halt_h(hn)[:, :, None])
        halt = torch.where(readable, halt, torch.ones_like(halt))
        halt = torch.cat([halt[..., :-1], torch.ones_like(halt[..., -1:])], -1)
        survive = torch.cumprod(torch.cat([torch.ones_like(halt[..., :1]), 1 - halt[..., :-1]], -1), -1)
        p = survive * halt                                            # де зупинився
        read = (p[..., None] * v * readable[..., None]).sum(-2)       # (B,Tq,K,d)
        n_read = (survive * readable).sum(-1)                         # скільки токенів прочитано

        got = (w[..., None] * read).sum(-2)
        cost = (w * n_read).sum(-1).mean()
        return self.head(self.n_out(h + self.out(got))), cost, top_i, w

    def forward(self, x):
        T = x.shape[1]
        mem = None
        if self.memory is not None:
            mem, self.mem_stats = simulate(self, x, self.memory)
        h, hn, q, face, book = self.encode(x)

        # 3. пошук картки
        scores = q @ face.transpose(1, 2) / math.sqrt(self.d)
        pos = torch.arange(T, device=x.device)
        allowed = pos[None, :] < pos[:, None]                         # лише минулі картки
        if mem is not None:
            allowed = allowed & mem                                   # і лише ті, що в пам'яті
        scores = scores.masked_fill(~allowed, -1e4)
        logits, cost, _, _ = self.read(h, hn, scores, pos, book)
        return logits, cost
