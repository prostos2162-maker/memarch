"""Порівняння на FineWeb-Edu: хто швидше (за часом) доходить до того самого val loss.

python lm_bench.py --models transformer cards tail --steps 3000
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from lm import build

VOCAB = int(__import__('os').environ.get('VOCAB', 50304))  # gpt2 (50257), округлено вгору до кратного 64; словосполучення — 100288


def prepare(path, n_tokens):
    if os.path.exists(path):
        return
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    out = np.empty(n_tokens, dtype=np.uint16)
    filled, buf = 0, []
    for ex in ds:
        buf.append(ex["text"])
        if len(buf) == 1024:
            for ids in enc.encode_ordinary_batch(buf, num_threads=os.cpu_count()):
                ids = ids + [enc.eot_token]
                k = min(len(ids), n_tokens - filled)
                out[filled:filled + k] = ids[:k]
                filled += k
            buf = []
            print(f"токенізовано {filled / 1e6:.0f}M", flush=True)
            if filled >= n_tokens:
                break
    np.save(path, out)


def batches(data, batch, seq, rng):
    while True:
        idx = rng.integers(0, len(data) - seq - 1, batch)
        x = np.stack([data[i:i + seq + 1] for i in idx]).astype(np.int64)
        yield torch.from_numpy(x).cuda(non_blocking=True)


def step_loss(model, xy, train):
    out, cost = model(xy[:, :-1])
    if hasattr(model, "train_loss"):           # моделі з власним виходом (образи слів)
        return model.train_loss(out, xy[:, 1:], train), cost
    return F.cross_entropy(out.float().view(-1, VOCAB), xy[:, 1:].reshape(-1)), cost


@torch.no_grad()
def val_loss(model, val, batch, seq, n=20):
    model.eval()
    rng = np.random.default_rng(0)
    it = batches(val, 8, seq, rng)
    total = 0.0
    for _ in range(n):
        xy = next(it)
        with torch.autocast("cuda", torch.bfloat16):
            total += step_loss(model, xy, train=False)[0].item()
    model.train()
    return total / n


def train(name, data, val, args):
    torch.manual_seed(0)
    model = build(name, VOCAB).cuda()
    if hasattr(model, "out") and hasattr(model.out, "set_unigram"):
        model.out.set_unigram(torch.from_numpy(np.bincount(data[:20_000_000], minlength=VOCAB)).cuda())
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name}: {n_params / 1e6:.1f}M параметрів ===", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps)))))
    it = batches(data, args.batch, args.seq, np.random.default_rng(1))
    log, train_time = [], 0.0
    for step in range(1, args.steps + 1):
        torch.cuda.synchronize()
        t0 = time.time()
        xy = next(it)
        # батч ділиться на мікро-батчі, інакше логіти 32×1024×50k не влазять у пам'ять
        for mb in xy.split(args.micro):
            with torch.autocast("cuda", torch.bfloat16):
                loss, cost = step_loss(model, mb, train=True)
            ((loss + args.read_cost * cost.float()) * len(mb) / len(xy)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        train_time += time.time() - t0      # час оцінки не рахуємо
        if step % args.eval_every == 0 or step == args.steps:
            vl = val_loss(model, val, args.batch, args.seq)
            log.append({"step": step, "time": train_time, "val": vl, "read": cost.item()})
            extra = "".join(f"  {k} {v:.2f}" for k, v in getattr(model, "stats", {}).items())
            print(f"{name}  крок {step:5d}  val {vl:.3f}  читає {cost.item():.2f}{extra}  {train_time / 60:.1f} хв", flush=True)
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump({"params": n_params, "log": log}, f)
    return log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["transformer", "cards", "tail"])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--micro", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--read_cost", type=float, default=0.01)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--tokens", type=int, default=120_000_000)
    args = p.parse_args()

    prepare("fineweb.npy", args.tokens)
    allt = np.load("fineweb.npy", mmap_mode="r")
    val, data = allt[:2_000_000], allt[2_000_000:]

    logs = {}
    for m in args.models:
        if os.path.exists(f"results/{m}.json"):      # уже натреновано раніше
            with open(f"results/{m}.json") as f:
                logs[m] = json.load(f)["log"]
            print(f"{m}: беру готовий результат з results/{m}.json")
        else:
            logs[m] = train(m, data, val, args)

    if "transformer" in logs:
        target = logs["transformer"][-1]["val"]
        print(f"\nЦіль: val {target:.3f} (кінець трансформера)")
        for m, log in logs.items():
            hit = next((r for r in log if r["val"] <= target), None)
            if hit:
                print(f"  {m:12s} дійшов на кроці {hit['step']:5d} за {hit['time'] / 60:.1f} хв")
            else:
                print(f"  {m:12s} не дійшов; кінцевий val {log[-1]['val']:.3f} за {log[-1]['time'] / 60:.1f} хв")


if __name__ == "__main__":
    main()
