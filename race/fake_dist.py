# Заглушка torch.distributed для одного процесу (Windows без NCCL): усі колективи — тотожні.
import torch
class ReduceOp: AVG = "avg"; SUM = "sum"; MAX = "max"
class _Work:
    def get_future(self):
        f = torch.futures.Future(); f.set_result(None); return f
    def wait(self): return True
def init_process_group(*a, **k): pass
def is_initialized(): return True
def get_world_size(*a): return 1
def get_rank(*a): return 0
def barrier(*a, **k): pass
def destroy_process_group(*a): pass
def broadcast(t, src=0, async_op=False): return _Work()
def all_reduce(t, op=None, async_op=False): return _Work()
def reduce(t, dst=0, op=None, async_op=False): return _Work()
@torch.compiler.disable
def reduce_scatter_tensor(out, inp, op=None, async_op=False):
    if out.data_ptr() != inp.data_ptr(): out.copy_(inp.view_as(out) if inp.numel() == out.numel() else inp[:out.size(0)])
    return _Work()
@torch.compiler.disable
def all_gather_into_tensor(out, inp, async_op=False):
    with torch.no_grad():
        _ag(out, inp)
    return _Work()
def _ag(out, inp):
    if out.data_ptr() != inp.data_ptr(): out[:inp.size(0)].copy_(inp) if inp.dim() and out.size(0) != inp.size(0) else out.copy_(inp.view_as(out))
    return _Work()
@torch.compiler.disable
def all_gather(lst, t, async_op=False):
    lst[0].copy_(t); return _Work()
