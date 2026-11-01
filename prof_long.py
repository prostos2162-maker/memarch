"""Куди йде час кроку повної моделі на довгому контексті (профайлер: відеокарта і процесор).

python prof_long.py 32768
"""
import argparse
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import fastloop

T = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
B = int(sys.argv[2]) if len(sys.argv) > 2 else 1
args = argparse.Namespace(lr=6e-4, steps=4000, micro=B, batch=B, seq=T, read_cost=0.01)
tr = fastloop.Trainer("Fv2ck", "A", args)
x = torch.randint(0, 50257, (B, T + 1), device="cuda")
for s in range(3):
    tr.step(x, s)
torch.cuda.synchronize()

# грубий розклад за частинами моделі: вимірюємо forward частин окремо
m = tr.model
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    def t(fn, n=3):
        fn(); torch.cuda.synchronize(); a = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.time() - a) / n * 1000
    print(f"PART T={T} B={B} forward цілком: {t(lambda: m(x[:, :-1])):8.1f} ms", flush=True)
    h = torch.randn(B, T, m.emb.embedding_dim, device="cuda")
    seg = min(m.seg, T)
    print(f"PART T={T} B={B} пам'ять (цикл по {T // seg} шматках): {t(lambda: m._memory_pass(h, seg)):8.1f} ms", flush=True)
    _, snaps, _ = m._memory_pass(h, seg)
    book = m.book(x[:, :-1])
    print(f"PART T={T} B={B} перечитування: {t(lambda: m._reread_pass(h, book, snaps, seg)):8.1f} ms", flush=True)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    tr.step(x, 10)
    torch.cuda.synchronize()
ka = prof.key_averages()
print(f"TOTAL T={T} B={B} self CPU / CUDA (мс):",
      round(sum(e.self_cpu_time_total for e in ka) / 1000), round(sum(e.self_device_time_total for e in ka) / 1000))
print("TOP за часом відеокарти:")
print(ka.table(sort_by="self_cuda_time_total", row_limit=15, max_name_column_width=60))
print("TOP за часом процесора:")
print(ka.table(sort_by="self_cpu_time_total", row_limit=15, max_name_column_width=60))
