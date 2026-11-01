"""Швидкість на довгому контексті: повна модель проти трансформера (FlashAttention).

На кожній довжині крок має ~32k токенів (батч = 32768 // довжина, мінімум 1).
Обидві моделі з перерахунком активацій, етап А циклу (torch.compile).

python long_speed.py --lens 1024 4096 16384 32768 65536
"""
import argparse
import time

import torch

import fastloop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["T300ck", "F300ck"])
    p.add_argument("--lens", nargs="+", type=int, default=[1024, 4096, 16384, 32768, 65536])
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--n", type=int, default=6)
    # batch: «tokens» — ~32k токенів на крок; «one» — 1 текст; «fill» — стільки текстів, скільки влазить
    p.add_argument("--batch", default="tokens", choices=["tokens", "one", "fill"])
    a = p.parse_args()
    for T in a.lens:
        if a.batch == "fill":
            for name in a.models:
                best, B = None, 1
                while True:                                   # подвоюємо, поки влазить
                    r = run(name, T, B, a)
                    if r is None:
                        break
                    best, B = r, B * 2
                print(f"FILL {name:7s} T={T:6d}  макс. текстів {best[0] if best else 0}  "
                      f"{best[1] if best else 0:7.1f}k ток/с", flush=True)
            continue
        B = 1 if a.batch == "one" else max(1, 32768 // T)
        for name in a.models:
            args = argparse.Namespace(lr=6e-4, steps=4000, micro=B, batch=B, seq=T, read_cost=0.01)
            torch.cuda.reset_peak_memory_stats()
            try:
                tr = fastloop.Trainer(name, "A", args)
                for s in range(a.warmup):
                    tr.step(torch.randint(0, 50257, (B, T + 1), device="cuda"), s)
                torch.cuda.synchronize()
                t = time.time()
                for s in range(a.n):
                    tr.step(torch.randint(0, 50257, (B, T + 1), device="cuda"), a.warmup + s)
                torch.cuda.synchronize()
                dt = (time.time() - t) / a.n
                print(f"LONG {name:7s} T={T:6d} B={B:2d}  {dt * 1000:8.1f} ms/крок  "
                      f"{B * T / dt / 1000:7.1f}k ток/с  mem {torch.cuda.max_memory_allocated() / 2**30:4.0f}GB  "
                      f"{ {k: round(float(v), 2) for k, v in getattr(tr.model, 'stats', {}).items()} }", flush=True)
                del tr
            except torch.OutOfMemoryError:
                print(f"LONG {name:7s} T={T:6d} B={B:2d}  OOM", flush=True)
            torch.cuda.empty_cache()


def run(name, T, B, a):
    """Один замір; None — якщо не влізло в пам'ять."""
    args = argparse.Namespace(lr=6e-4, steps=4000, micro=B, batch=B, seq=T, read_cost=0.01)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        tr = fastloop.Trainer(name, "A", args)
        for s in range(a.warmup):
            tr.step(torch.randint(0, 50257, (B, T + 1), device="cuda"), s)
        torch.cuda.synchronize()
        t = time.time()
        for s in range(a.n):
            tr.step(torch.randint(0, 50257, (B, T + 1), device="cuda"), a.warmup + s)
        torch.cuda.synchronize()
        dt = (time.time() - t) / a.n
        print(f"LONG {name:7s} T={T:6d} B={B:2d}  {dt * 1000:8.1f} ms/крок  {B * T / dt / 1000:7.1f}k ток/с  "
              f"mem {torch.cuda.max_memory_allocated() / 2**30:4.0f}GB", flush=True)
        del tr
        return B, B * T / dt / 1000
    except torch.OutOfMemoryError:
        print(f"LONG {name:7s} T={T:6d} B={B:2d}  OOM", flush=True)
        return None
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
