"""Свіжа і тьмяна пам'ять з прибиранням.

Комірка зберігає закладку (позицію токена) і три числа для оцінки:
здивування при записі, скільки разів її діставали, коли записана.

Кожен крок:
  1. модель шукає картку лише серед тих, що зараз у пам'яті;
  2. якщо найкраща картка лежала в тьмяній, вона піднімається у свіжу,
     а найслабша свіжа опускається на її місце;
  3. новий токен пишеться у свіжу; якщо там повно, найслабша свіжа опускається в тьмяну;
  4. коли тьмяна розтяглась до межі, прибирання викидає найслабші до норми.

Рішення про пам'ять дискретні, тому симуляція йде без градієнтів і повертає
маску «які позиції були в пам'яті на кожному кроці». Модель потім вчиться
з цією маскою паралельно, як у v0.
"""
import math

import torch
import torch.nn.functional as F


class MemoryConfig:
    def __init__(self, fresh=64, dim=128, dim_max=160, w_use=2.0, w_age=1 / 128):
        self.fresh, self.dim, self.dim_max = fresh, dim, dim_max
        self.w_use, self.w_age = w_use, w_age


@torch.no_grad()
def simulate(model, x, cfg):
    B, T = x.shape
    dev = x.device
    Fr, S = cfg.fresh, cfg.fresh + cfg.dim_max
    if T <= cfg.fresh + cfg.dim:
        # усе влазить, прибирання не спрацює ні разу: пам'ять = весь минулий текст
        return torch.ones(B, T, T, dtype=torch.bool, device=dev), {"evicted": 0}
    ar = torch.arange(B, device=dev)

    pos = torch.full((B, S), -1, dtype=torch.long, device=dev)
    sur = torch.zeros(B, S, device=dev)
    uses = torch.zeros(B, S, device=dev)
    birth = torch.zeros(B, S, device=dev)
    fields = [pos, sur, uses, birth]

    def score(t):
        sc = sur + cfg.w_use * uses - cfg.w_age * (t - birth)
        return sc.masked_fill(pos < 0, float("inf"))

    def move(rows, src, dst):
        for f in fields:
            f[ar, dst] = torch.where(rows, f[ar, src], f[ar, dst])

    def swap(rows, a, b):
        for f in fields:
            va, vb = f[ar, a].clone(), f[ar, b].clone()
            f[ar, a] = torch.where(rows, vb, va)
            f[ar, b] = torch.where(rows, va, vb)

    h, hn, q, face, book = model.encode(x)
    mask = torch.zeros(B, T, T, dtype=torch.bool, device=dev)
    prev_logits = None
    stats = {"evicted": 0}

    for t in range(T):
        valid = pos >= 0
        row = torch.zeros(B, T, dtype=torch.long, device=dev)
        row = row.scatter_add_(1, pos.clamp(min=0), valid.long()) > 0
        mask[:, t] = row

        # 1. пошук і читання на цьому кроці
        scores = (q[:, t:t + 1] @ face.transpose(1, 2)) / math.sqrt(model.d)
        scores = scores.masked_fill(~row[:, None], -1e4)
        logits, _, top_i, w = model.read(h[:, t:t + 1], hn[:, t:t + 1], scores,
                                         torch.tensor([t], device=dev), book)

        # здивування на токені t = наскільки погано його передбачили на кроці t-1
        if prev_logits is None:
            s_t = torch.full((B,), math.log(logits.shape[-1]), device=dev)
        else:
            s_t = -F.log_softmax(prev_logits.float(), -1)[ar, x[:, t]]
        prev_logits = logits[:, 0]

        # 2. використання і підйом з тьмяної
        strong = (w[:, 0, 0] > 0.5) & row.any(1)
        hit = (pos == top_i[:, 0, 0][:, None]) & strong[:, None]
        uses += hit.float()
        hit_slot = hit.float().argmax(1)
        promote = hit.any(1) & (hit_slot >= Fr)
        if promote.any():
            weakest_fresh = score(t)[:, :Fr].argmin(1)
            swap(promote, hit_slot, weakest_fresh)

        # 3. запис нового токена у свіжу
        empty_f = pos[:, :Fr] < 0
        has_empty = empty_f.any(1)
        tgt = torch.where(has_empty, empty_f.long().argmax(1), score(t)[:, :Fr].argmin(1))
        demote = ~has_empty
        if demote.any():
            dst = (pos[:, Fr:] < 0).long().argmax(1) + Fr
            move(demote, tgt, dst)
        pos[ar, tgt] = t
        sur[ar, tgt] = s_t
        uses[ar, tgt] = 0
        birth[ar, tgt] = float(t)

        # 4. прибирання тьмяної
        dpos = pos[:, Fr:]
        cnt = (dpos >= 0).sum(1)
        over = cnt >= cfg.dim_max
        if over.any():
            rank = score(t)[:, Fr:].argsort(1).argsort(1)
            drop = (rank < (cnt - cfg.dim)[:, None]) & over[:, None] & (dpos >= 0)
            stats["evicted"] += drop.sum().item()
            pos[:, Fr:] = torch.where(drop, -1, dpos)

    return mask, stats
