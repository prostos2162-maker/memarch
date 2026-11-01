"""Куди йде час у кроці: профайлер PyTorch, топ операцій за часом на відеокарті.

python profile_step.py S300_20 AB
"""
import sys

import argparse
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

import fastloop

name, stage = sys.argv[1], sys.argv[2]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 32          # батч (на 3060 — менший)
args = argparse.Namespace(lr=6e-4, steps=4000, micro=B, batch=B, seq=1024, read_cost=0.01)
data = fastloop.GPUData(np.random.randint(0, 50257, 5_000_000, dtype=np.int32), seed=1)
tr = fastloop.Trainer(name, stage, args)
for s in range(40):
    tr.step(data.batch(B, 1024), s)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    for s in range(5):
        tr.step(data.batch(B, 1024), 40 + s)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=45, max_name_column_width=70))

# підсумок за категоріями: лише кернели відеокарти (без подвійного рахунку операцій aten), час на крок
from collections import defaultdict
cat = defaultdict(float)
for e in prof.events():
    if e.device_type != torch.autograd.DeviceType.CUDA:
        continue
    n = e.name
    k = ("множення" if any(s in n for s in ("gemm", "cutlass", "sm90_xmma", "sm80_xmma", "cublas")) else
         "увага" if "flash" in n or "fmha" in n else
         "оптимізатор" if "rsub_sqrt" in n or "vector_norm" in n else
         "triton інше" if n.startswith("triton") else
         "копіювання" if "copy" in n.lower() or "Memcpy" in n or "Memset" in n else
         "aten поелементні" if "elementwise" in n or "reduce_kernel" in n else "інше")
    cat[k] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
tot = sum(cat.values())
print(f"\nКЕРНЕЛИ на крок: {tot / 5 / 1000:.1f} мс")
for k, v in sorted(cat.items(), key=lambda x: -x[1]):
    print(f"  {k:18s} {v / 5 / 1000:7.1f} мс  {v / tot:5.1%}")
names = defaultdict(float)
for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        names[e.name[:60]] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
print("топ кернелів:")
for k, v in sorted(names.items(), key=lambda x: -x[1])[:40]:
    print(f"  {v / 5 / 1000:6.1f} мс  {k}")
