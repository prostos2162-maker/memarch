"""Тренування на associative recall і оцінка на різних довжинах.

python train.py --model transformer
python train.py --model cards
"""
import argparse
import time

import torch
import torch.nn.functional as F

from data import VOCAB, IGNORE, make_batch
from baseline import Transformer
from functools import partial

from cards import Cards
from memory import MemoryConfig

MODELS = {
    "transformer": Transformer,
    "cards": Cards,
    "cards_mem": partial(Cards, memory=MemoryConfig()),
}


def run(model, x):
    """Повертає (logits, cost). У трансформера ціни за перечитування немає."""
    out = model(x)
    return out if isinstance(out, tuple) else (out, torch.zeros((), device=x.device))


@torch.no_grad()
def evaluate(model, n_pairs, n_queries, device, batches=8):
    model.eval()
    g = torch.Generator().manual_seed(1234)
    correct = total = 0
    for _ in range(batches):
        x, y = make_batch(64, n_pairs, n_queries, device, g)
        with torch.autocast("cuda", torch.bfloat16):
            pred = run(model, x)[0].argmax(-1)
        mask = y != IGNORE
        correct += (pred[mask] == y[mask]).sum().item()
        total += mask.sum().item()
    model.train()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="transformer", choices=MODELS)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--pairs", type=int, default=64)
    p.add_argument("--queries", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--read_cost", type=float, default=0.01, help="штраф за кожен перечитаний токен")
    args = p.parse_args()

    torch.manual_seed(0)
    device = "cuda"
    model = MODELS[args.model](VOCAB, d=args.d, layers=args.layers).to(device)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M параметрів")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200))

    # поступове ускладнення: на 64 парах одразу трансформер застрягає,
    # тож починаємо з 8 і подвоюємо, коли поточний рівень розв'язано
    level = min(8, args.pairs)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        n = int(torch.randint(4, level + 1, ()).item())
        x, y = make_batch(args.batch, n, min(args.queries, n), device)
        with torch.autocast("cuda", torch.bfloat16):
            logits, cost = run(model, x)
        loss = F.cross_entropy(logits.float().view(-1, VOCAB), y.view(-1), ignore_index=IGNORE)
        loss = loss + args.read_cost * cost.float()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0:
            acc = evaluate(model, level, args.queries, device, batches=2)
            print(f"крок {step:5d}  рівень {level:3d} пар  loss {loss.item():.3f}  точність {acc:.3f}  читає {cost.item():.2f} ток.  {time.time() - t0:.0f}с")
            if acc > 0.95 and level < args.pairs:
                level = min(2 * level, args.pairs)

    print("\nТочність за кількістю пар (тренували на", args.pairs, "):")
    for n in [16, 32, 64, 128, 256]:
        acc = evaluate(model, n, args.queries, device)
        stats = getattr(model, "mem_stats", None)
        extra = f"  викинуто з пам'яті: {stats['evicted'] // 64} на текст" if stats else ""
        print(f"  {n:4d} пар: {acc:.3f}{extra}")


if __name__ == "__main__":
    main()
