"""Де процесор зупиняє відеокарту в кроці тренування (Стефан: «це не дует, це соло»).

python sync_audit.py Fv2 AB  — друкує кожне місце, де хост чекає на відеокарту (з компіляцією, як у тренуванні).
"""
import argparse
import sys
import warnings

import numpy as np
import torch

import fastloop

warnings.simplefilter("always")
name, stage = sys.argv[1], sys.argv[2]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 32                  # батч (на 3060 — менший)
args = argparse.Namespace(lr=6e-4, steps=4000, micro=B, batch=B, seq=1024, read_cost=0.01)
data = fastloop.GPUData(np.random.randint(0, 50257, 5_000_000, dtype=np.int32), seed=1)
tr = fastloop.Trainer(name, stage, args)
for s in range(5):
    tr.step(data.batch(B, 1024), s)
torch.cuda.synchronize()
torch.cuda.set_sync_debug_mode(1)
tr.step(data.batch(B, 1024), 5)
torch.cuda.set_sync_debug_mode(0)
torch.cuda.synchronize()
print("AUDIT DONE", name, stage, flush=True)
