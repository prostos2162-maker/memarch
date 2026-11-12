"""KindCorpus (фішки): увесь батч — один вид; вид кроку і комбінація фішок детерміновані (однакові на всіх процесах);
частки видів дотримано; stan → відновлення дає той самий наступний батч; HostLoader кладе kind/fish у prep.
Запуск: PYTHONIOENCODING=utf-8 python tests/test_kind_corpus.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np                       # noqa: E402

from memarch_jax import data             # noqa: E402

ROOT = r"D:\ai\_cpu_t1\tok"
KINDS = {"code": {"stack_v3": 0.4}, "math": {"nemotron_math4": 0.15}, "uk": {"hplt_uk": 0.18},
         "en": {"fineweb_edu": 0.2}}
POOLS = [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8], [8, 9, 10, 11, 0], [11, 12, 13, 14, 15]]
mk = lambda rank: data.KindCorpus(KINDS, POOLS, 2, root=ROOT, seq=256, seed=3, rank=rank, world=2, eos=131072,
                                   cache=r"D:\ai\memarch\_tmp\kc_cache")
a, b = mk(0), mk(1)
ks, orders = [], []
for i in range(40):
    a.batch(2)
    b.batch(2)
    ea, eb = a.extra(), b.extra()
    assert (ea["kind"] == eb["kind"]).all() and (ea["fish"] == eb["fish"]).all(), "процеси не узгоджені"
    k = int(ea["kind"][0, 0])
    assert set(ea["fish"][0].tolist()) <= set(POOLS[k]) and len(set(ea["fish"][0].tolist())) == 2
    ks.append(k)
    orders.append(tuple(ea["fish"][0]))
frac = np.bincount(ks, minlength=4) / len(ks)
print("частки видів:", dict(zip(a.names, np.round(frac, 3))), "ціль", dict(zip(a.names, np.round(a.p, 3))))
assert np.abs(frac - a.p).max() < 0.05
print("різних комбінацій:", len(set(orders)))
st = a.state_dict()
x1 = a.batch(2)
e1 = a.extra()
c = mk(0)
c.load_state_dict(st)
x2 = c.batch(2)
assert (x1 == x2).all() and (c.extra()["fish"] == e1["fish"]).all(), "resume не той самий"
a.full = True
a.batch(2)
k = int(a.extra()["kind"][0, 0])
assert a.extra()["fish"][0].tolist() == POOLS[k]
hl = data.HostLoader(mk(0), 2, prep_fn=lambda x: {"on": np.zeros_like(x)}, depth=2)
xy, prep = hl.next()
hl.close()
assert set(prep) >= {"on", "kind", "fish"} and prep["fish"].shape == (2, 2)
print("KIND_CORPUS_OK")
