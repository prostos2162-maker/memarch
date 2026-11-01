"""Скільки якорів обрізала б стеля «на шматок» замість стелі «на текст».

1. Коротко тренує схему здивувань (поріг + стеля на текст) і зберігає ваги.
2. На справжніх текстах (val) рахує якорі за порогом, без жодної стелі,
   і дивиться, яку частку з них відкинула б кожна стеля.

python anchor_chunks.py --steps 2000
"""
import argparse
import os

import numpy as np
import torch

import fastloop
from lm_bench import prepare


@torch.no_grad()
def anchors_without_cap(model, val, seq, n_texts):
    """Якорі за порогом (без стелі) для n_texts текстів з val. Повертає (n_texts, seq) bool."""
    model.eval()
    g = torch.Generator(device="cuda").manual_seed(0)
    out = []
    for _ in range(n_texts // 16):
        start = torch.randint(0, len(val.t) - seq - 1, (16,), device="cuda", generator=g)
        x = val.t[start[:, None] + torch.arange(seq, device="cuda")].long()
        with torch.autocast("cuda", torch.bfloat16):
            h = model.emb(x)
            for b in model.light:
                h = b(h)
            lo = model.l_head(model.l_norm(h))[:, :-1]
        s = torch.full(x.shape, 1e4, device="cuda")
        s[:, 1:] = (torch.logsumexp(lo.float(), -1) - lo.float().gather(-1, x[:, 1:, None]).squeeze(-1))
        out.append(s > model.tau)
    return torch.cat(out)


def dropped(anchor, chunk, cap):
    """Частка якорів, відкинутих стелею `cap` у шматках по `chunk` слів (перші за порядком лишаються)."""
    B, T = anchor.shape
    a = anchor.view(B, T // chunk, chunk)
    kept = a & (a.cumsum(-1) <= cap)
    return 1 - kept.sum().item() / max(1, a.sum().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="S300c_20")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--micro", type=int, default=16)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--read_cost", type=float, default=0.01)
    p.add_argument("--tokens", type=int, default=70_000_000)
    p.add_argument("--texts", type=int, default=512)
    args = p.parse_args()

    prepare("fineweb_small.npy", args.tokens)
    allt = np.load("fineweb_small.npy", mmap_mode="r")
    val, data = fastloop.GPUData(allt[:2_000_000], 0), fastloop.GPUData(allt[2_000_000:], 1)

    ckpt = f"results/{args.model}_probe.pt"
    tr = fastloop.Trainer(args.model, "ABC", args)
    if os.path.exists(ckpt):
        tr.model.load_state_dict(torch.load(ckpt))
    else:
        for step in range(args.steps):
            tr.step(data.batch(args.batch, args.seq), step)
            if (step + 1) % 500 == 0:
                print(f"крок {step + 1}  val {fastloop.val_loss(tr.model, val, args.seq):.3f}", flush=True)
        os.makedirs("results", exist_ok=True)
        torch.save(tr.model.state_dict(), ckpt)

    a = anchors_without_cap(tr.model, val, args.seq, args.texts)
    print(f"\nякорів без стелі: {a.float().mean().item():.1%} слів  ({args.texts} текстів по {args.seq})")
    for chunk in (64, 128, 256):
        frac = a.view(len(a), -1, chunk).float().mean(-1)
        print(f"  шматок {chunk:4d}: частка якорів у шматку — медіана {frac.median().item():.0%}, "
              f"90-й перцентиль {frac.quantile(0.9).item():.0%}, максимум {frac.max().item():.0%}")
    print("\nчастка відкинутих якорів:")
    print(f"  стеля на текст 25% (як зараз):        {dropped(a, args.seq, round(args.seq * 0.25)):.2%}")
    for chunk in (64, 128, 256):
        for share in (0.20, 0.25, 0.30):
            print(f"  стеля на шматок {chunk:4d}, {share:.0%} ({round(chunk * share):3d} місць): "
                  f"{dropped(a, chunk, round(chunk * share)):.2%}")


if __name__ == "__main__":
    main()
