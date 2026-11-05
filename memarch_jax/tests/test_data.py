r"""M11-дані: data.Corpus на крихітних локальних шматках — збіг з torch-лоадером (data/loader.py)
батч у батч, маски drop/cdrop/score, resume зі state_dict (Corpus і HostLoader з префетчем + induct prep).
Запуск: cd memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_data.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np                       # noqa: E402
import zstandard as zstd                 # noqa: E402

from memarch_jax import data             # noqa: E402

EOS, V = 999, 1000


def _make_root():
    root = tempfile.mkdtemp()
    r = np.random.default_rng(0)
    for src, n_sh, code in (("web", 3, False), ("code", 4, True)):
        os.makedirs(os.path.join(root, src))
        for i in range(n_sh):
            docs = [np.append(r.integers(0, EOS, r.integers(5, 300)), EOS) for _ in range(40)]
            arr = np.concatenate(docs).astype(np.uint32)
            base = os.path.join(root, src, f"s{i:03d}")
            with open(base + ".u32.zst", "wb") as f:
                f.write(zstd.ZstdCompressor().compress(arr.tobytes()))
            with open(base + ".json", "w") as f:
                json.dump({"eos": EOS}, f)
            np.save(base + ".drop.npy", np.array([0, 3, 7], np.int64))
            if i == 0:
                np.save(base + ".cdrop.npy", np.array([1], np.int64))
            if code:
                np.save(base + ".score.npy", r.uniform(0, 5, 40).astype(np.float32))
    return root


ROOT = _make_root()
W = {"web": 0.4, "code": 0.6}


def test_matches_torch_loader():
    import importlib.util
    spec = importlib.util.spec_from_file_location("torch_loader", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "loader.py"))
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    ref = L.Corpus(W, root=ROOT, seq=64, seed=3)
    c = data.Corpus(W, root=ROOT, seq=64, seed=3)
    assert c.eos == EOS
    for i in range(400):                                 # > епохи джерел (кілька обертів)
        a, b = c.batch(4), ref.batch(4)
        assert a.dtype == np.int32 and a.shape == (4, 65)
        assert np.array_equal(a, b), i
    assert c.mix() == ref.mix()
    ep = [s.epoch for s in c.sources]
    print("torch-loader parity: 400 batches equal; mix", c.mix(), "epochs", ep)
    assert max(ep) >= 1


def test_masks():
    c = data.Corpus({"web": 1.0}, root=ROOT, seq=64, seed=0)
    toks = np.concatenate([c.batch(8).ravel() for _ in range(5)])
    assert toks.max() <= EOS and (toks == EOS).sum() > 0
    # кількість документів шматка після drop/cdrop
    s = c.sources[0]
    d0 = data._docs(data._fetch(ROOT, "web", "s000", None), np.random.default_rng(0), EOS)
    assert len(d0) == 40 - 3 - 1
    print("masks OK: s000 docs", len(d0))


def test_resume():
    c = data.Corpus(W, root=ROOT, seq=64, seed=5)
    for _ in range(7):
        c.batch(3)
    st = json.loads(json.dumps(c.state_dict()))          # через JSON — як у ckpt
    want = [c.batch(3) for _ in range(10)]
    c2 = data.Corpus(W, root=ROOT, seq=64, seed=5)
    c2.load_state_dict(st)
    got = [c2.batch(3) for _ in range(10)]
    assert all(np.array_equal(a, b) for a, b in zip(want, got))
    print("Corpus resume OK")


def test_hostloader_resume():
    from memarch_jax import induct
    bucket = induct.lock_bucket(np.arange(V, 0, -1).astype(np.int64), V)
    prep = data.induct_prep(V, bucket)
    c = data.Corpus(W, root=ROOT, seq=64, seed=7)
    ref = [c.batch(2) for _ in range(9)]
    hl = data.HostLoader(data.Corpus(W, root=ROOT, seq=64, seed=7), B=2, prep_fn=prep, depth=3)
    got = [hl.next() for _ in range(4)]
    st = hl.state_dict()                                  # префетч уже пішов далі — стан має бути «після 4-го»
    hl.close()
    for (xy, p), r in zip(got, ref):
        assert np.array_equal(xy, r) and set(p) == {"cand", "on", "bk", "src"} and p["cand"].shape == (2, 64, 3)
    hl2 = data.HostLoader(data.Corpus(W, root=ROOT, seq=64, seed=7), B=2, prep_fn=prep)
    hl2.load_state_dict(json.loads(json.dumps(st)))
    rest = [hl2.next()[0] for _ in range(5)]
    hl2.close()
    assert all(np.array_equal(a, b) for a, b in zip(rest, ref[4:]))
    print("HostLoader resume OK (prefetch depth 3)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
