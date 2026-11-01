"""Порівняння злитих кернелів: звичайний PyTorch (як зараз у моделі) / Liger / наші (fused_ops.py).

Перевіряє однаковість результатів і градієнтів, міряє час прямого+зворотного проходу і пам'ять.
python bench_fused.py [N_rows] [d] [V]
"""
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import fused_ops

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
d = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
V = int(sys.argv[3]) if len(sys.argv) > 3 else 50304
dev = "cuda"


def bench(fn, n=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1e3, torch.cuda.max_memory_allocated() / 2**30


def ce_torch(h, w, y):            # як у моделі: шматки по 8192 з checkpoint (логіти рахуються двічі)
    def part(hh, yy):
        return F.cross_entropy((hh @ w.T).float(), yy, reduction="sum")
    return sum(checkpoint(part, h[i:i + 8192], y[i:i + 8192], use_reentrant=False) for i in range(0, len(h), 8192))


def ce_liger(h, w, y):
    from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
    return liger_fused_linear_cross_entropy(h, w, y, reduction="sum")


def ce_ours(h, w, y):
    return fused_ops.linear_ce(h, w, y)


torch.manual_seed(0)
print(f"== вихід на словник + похибка: N={N}, d={d}, V={V}")
h0 = (torch.randn(N, d, device=dev) * 0.5).bfloat16()
w0 = (torch.randn(V, d, device=dev) * 0.02).bfloat16()
y = torch.randint(0, V, (N,), device=dev)
ref = None
for name, f in [("torch+checkpoint", ce_torch), ("liger", ce_liger), ("ours", ce_ours)]:
    h = h0.clone().requires_grad_(); w = w0.clone().requires_grad_()
    try:
        loss = f(h, w, y); loss.backward()
        out = (float(loss), h.grad.float(), w.grad.float())
        if ref is None:
            ref = out
        dl = abs(out[0] - ref[0]) / abs(ref[0])
        dh = ((out[1] - ref[1]).norm() / ref[1].norm()).item()
        dw = ((out[2] - ref[2]).norm() / ref[2].norm()).item()

        def run():
            hh = h0.clone().requires_grad_(); ww = w0.clone().requires_grad_()
            f(hh, ww, y).backward()
        ms, mem = bench(run)
        print(f"{name:18s} {ms:8.2f} мс  пік {mem:5.2f} ГБ   різниця: loss {dl:.1e}  grad h {dh:.1e}  grad w {dw:.1e}")
    except Exception as e:
        print(f"{name:18s} ПОМИЛКА {type(e).__name__}: {str(e)[:200]}")

H = 4 * d
print(f"== SwiGLU: рядків {N}, H={H}")
gu0 = torch.randn(N, 2 * H, device=dev).bfloat16()


def sw_torch(gu):
    g, u = gu.chunk(2, -1)
    return F.silu(g) * u


def sw_liger(gu):
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    g, u = gu.chunk(2, -1)
    return LigerSiLUMulFunction.apply(g.contiguous(), u.contiguous())


sw_torch_c = torch.compile(sw_torch)
ref = None
for name, f in [("torch", sw_torch), ("torch.compile", sw_torch_c), ("liger", sw_liger), ("ours", fused_ops.swiglu)]:
    gu = gu0.clone().requires_grad_()
    try:
        o = f(gu); go = torch.ones_like(o); o.backward(go)
        out = (o.float(), gu.grad.float())
        if ref is None:
            ref = out
        do = ((out[0] - ref[0]).norm() / ref[0].norm()).item()
        dg = ((out[1] - ref[1]).norm() / ref[1].norm()).item()

        def run():
            gg = gu0.clone().requires_grad_()
            f(gg).backward(go)
        ms, mem = bench(run, 30)
        print(f"{name:18s} {ms:8.3f} мс  пік {mem:5.2f} ГБ   різниця: out {do:.1e}  grad {dg:.1e}")
    except Exception as e:
        print(f"{name:18s} ПОМИЛКА {type(e).__name__}: {str(e)[:200]}")
