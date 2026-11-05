"""Крихітний Llama-подібний трансформер: RMSNorm, RoPE, SwiGLU, SDPA (flash на GPU)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ROPE_HALF=1: пари вимірів — половинки (x[:D/2], x[D/2:]), а не сусіди (0::2, 1::2): та сама RoPE з іншим
# (однаковим для q і k) порядком вимірів, але без кроку 2 і stack — на TPU це суцільні зрізи замість копій
ROPE_HALF = __import__("os").environ.get("ROPE_HALF", "1" if __import__("os").environ.get("DEVICE") in ("tpu", "xla")
                                         else "") == "1"


def rope_cs(x, cos, sin):
    """Поворот x (..., D) кутами (cos, sin) форми (..., D/2), що мовчки розтягуються на x."""
    if ROPE_HALF:
        x1, x2 = x.chunk(2, -1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).type_as(x)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).type_as(x)


def rope(x, base=10000.0):
    # x: (B, H, T, D)
    T, D = x.shape[-2], x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, D, 2, device=x.device).float() / D)
    ang = torch.arange(T, device=x.device).float()[:, None] * inv[None]
    cos, sin = ang.cos(), ang.sin()
    if ROPE_HALF:
        return rope_cs(x, cos, sin)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
    return out.flatten(-2).type_as(x)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.n1 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.n2 = nn.RMSNorm(d)
        self.up = nn.Linear(d, 2 * 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        g, u = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)


class Transformer(nn.Module):
    def __init__(self, vocab, d=128, layers=2, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))
