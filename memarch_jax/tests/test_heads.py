"""Паритет M2+M3 (ce, surprise, short_logp, induct, anchors) проти torch full2.py на CPU, W=1.
Запуск: cd memarch_jax && PYTHONIOENCODING=utf-8 python tests/test_heads.py  (або python -m pytest)."""
import os, sys, tempfile
import numpy as np

V, K, CH, D = 4096, 512, 1000, 256
_tmp = tempfile.mkdtemp()
_rng0 = np.random.default_rng(0)
COUNTS = np.floor(1e6 / np.arange(1, V + 1) ** 1.1 * _rng0.uniform(0.5, 1.5, V)).astype(np.int64)
COUNTS[V - 1] = 0
np.save(os.path.join(_tmp, "cnt.npy"), COUNTS)
os.environ.update(DEVICE="cpu", NO_COMPILE="1", ATTN_IMPL="block", CE_CHUNK=str(CH), SURP_INGRAPH="1",
                  SHORT_V=str(K), INDUCT="lock", INDUCT_IN="1", INDUCT_MIX="1", INDUCT_POS="1",
                  TOKEN_COUNTS=os.path.join(_tmp, "cnt.npy"))
os.environ.pop("BF16_ALL", None)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # repo root: torch reference code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import full2
import jax
import jax.numpy as jnp
from memarch_jax import ce, induct, anchors

torch.manual_seed(0)
RES = []
_M = None


def _np(t):
    return t.detach().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(jnp.asarray(t, jnp.float32))


def cmp(name, a, b, tol):
    a, b = _np(a), _np(b)
    ad = float(np.abs(a - b).max()) if a.size else 0.0
    rd = ad / max(float(np.abs(b).max()), 1e-30)
    RES.append((name, ad, rd))
    print(f"  {name:44s} max_abs={ad:.3e} max_rel={rd:.3e}")
    assert rd <= tol, f"{name}: rel {rd} > {tol}"


def _model():
    global _M
    if _M is None:
        _M = full2.FullLM2(V, d=64, light_a=1, light_b=1, heavy=1, final=1, heads=1)
    return _M


def _tj(t, dt=None):
    a = jnp.asarray(t.detach().float().numpy())
    return a.astype(dt) if dt is not None else a


# ---------------- chunk_ce ----------------
def _chunk_case(dt, ch, use_b, tol):
    N, Vw = 300, 2600
    h = torch.randn(N, D).to(dt).requires_grad_()
    W = (torch.randn(Vw, D) * 0.05).to(dt).requires_grad_()
    b = (torch.randn(Vw) * 0.5) if use_b else None
    y = torch.randint(0, Vw, (N,))
    a1, a2 = torch.randn(N), torch.randn(N)
    lse, ly = full2.chunk_ce(h, W, b, y, ch)
    ((lse * a1).sum() + (ly * a2).sum()).backward()
    jd = jnp.bfloat16 if dt == torch.bfloat16 else jnp.float32
    hj, Wj, yj = _tj(h, jd), _tj(W, jd), jnp.asarray(y.numpy())
    bj = _tj(b) if use_b else jnp.zeros((Vw,), jnp.float32)
    A1, A2 = _tj(a1), _tj(a2)
    lj, lyj = ce.chunk_ce(hj, Wj, bj, yj, ch)

    def f(h_, W_):
        l_, y_ = ce.chunk_ce(h_, W_, bj, yj, ch)
        return (l_ * A1).sum() + (y_ * A2).sum()
    gh, gW = jax.grad(f, (0, 1))(hj, Wj)
    tag = f"chunk_ce[{str(dt)[6:]},ch={ch},b={int(use_b)}]"
    cmp(tag + " lse", lj, lse, tol); cmp(tag + " ly", lyj, ly, tol)
    cmp(tag + " dh", gh, h.grad, tol); cmp(tag + " dW", gW, W.grad, tol)


def test_chunk_ce():
    print("chunk_ce")
    for dt, tol in ((torch.float32, 5e-3), (torch.bfloat16, 2e-2)):
        for ch in (CH, 2601):
            for use_b in (False, True):
                _chunk_case(dt, ch, use_b, tol)


# ---------------- surprise ----------------
def test_surprise():
    print("surprise (SURP_INGRAPH)")
    B, T = 2, 513
    for dt, tol in ((torch.float32, 1e-3), (torch.bfloat16, 1e-2)):   # h128 → bf16: фліпи округлення
        lh = torch.randn(B, T, D).to(dt)
        w0 = (torch.randn(128, D) * 0.06).to(dt)
        W2 = (torch.randn(V, 128) * 0.09).to(dt)
        x = torch.randint(0, V, (B, T))
        if dt == torch.bfloat16:
            with torch.autocast("cpu", torch.bfloat16):
                st = full2._surp_lse_impl(lh, w0, W2, x)
        else:
            st = full2._surp_lse_impl(lh, w0, W2, x)
        jd = jnp.bfloat16 if dt == torch.bfloat16 else jnp.float32
        sj = ce.surprise(_tj(lh, jd), _tj(w0, jd), _tj(W2, jd), jnp.asarray(x.numpy()), CH)
        assert float(sj[0, 0]) == 1e4
        cmp(f"surprise[{str(dt)[6:]}]", sj[:, 1:], st, tol)


# ---------------- short_logp ----------------
def test_short_logp():
    print("short_logp (SHORT_V, один чіп)")
    m = _model()
    sc = ce.short_consts(COUNTS, V, K, 0.3)
    assert np.array_equal(sc["sv_ids"], m.sv_ids.numpy()) and np.array_equal(sc["sv_pos"], m.sv_pos.numpy())
    assert np.array_equal(sc["sv_mask"], m.sv_mask.numpy()) and sc["prior0"] == m.sv_prior0
    p = COUNTS / COUNTS.sum()
    N = 1024
    for mode in ("zipf", "uniform"):              # uniform: рідкісних > місткості (перевірка rare_drop)
        for dt, tol in ((torch.float32, 5e-3), (torch.bfloat16, 2e-2)):
            yy = _rng0.choice(V, N, p=p) if mode == "zipf" else _rng0.integers(0, V - 1, N)
            y = torch.from_numpy(yy)
            h = torch.randn(N, D).to(dt).requires_grad_()
            W = (torch.randn(V, D) * 0.05).to(dt)
            W[V - 1] = 0
            m.head.weight = torch.nn.Parameter(W.clone())
            lp = m._short_logp(h, y)
            a = torch.randn(N)
            (lp * a).sum().backward()
            jd = jnp.bfloat16 if dt == torch.bfloat16 else jnp.float32
            hj, Wj, yj, Aj = _tj(h, jd), _tj(W, jd), jnp.asarray(yy), _tj(a)
            lpj, st = ce.short_logp(hj, Wj, yj, sc, CH)
            gh, gW = jax.grad(lambda h_, W_: (ce.short_logp(h_, W_, yj, sc, CH)[0] * Aj).sum(), (0, 1))(hj, Wj)
            tag = f"short_logp[{mode},{str(dt)[6:]}]"
            cmp(tag + " logp", lpj, lp, tol)
            cmp(tag + " dh", gh, h.grad, tol); cmp(tag + " dW", gW, m.head.weight.grad, tol)
            cmp(tag + " rare_drop", st["rare_drop"], m.stats["rare_drop"], 1e-6)


# ---------------- induct ----------------
def _xrep(B=3, T=512):
    r = np.random.default_rng(1)
    x = r.integers(0, 40, (B, T))                  # малий словник → повтори слів/пар/трійок
    x[:, 100:140] = x[:, 10:50]                    # повторений фрагмент
    x[0, 200:210] = 7                              # «a a a»
    x[1, :] = r.integers(0, V, T)                  # майже без повторів
    return x


def test_induct_raw():
    print("induct_raw (хост, numpy)")
    m = _model()
    bucket = induct.lock_bucket(COUNTS, V)
    assert np.array_equal(bucket, m.lock_bucket.numpy())
    x = _xrep()
    ct, ot, bt, st = m._induct_raw(torch.from_numpy(x), True)
    cj, oj, bj, sj = induct.induct_raw(x, V, bucket)
    for n, a, b in (("cand", cj, ct), ("on", oj, ot), ("bk", bj, bt), ("src", sj, st)):
        assert np.array_equal(a, b.numpy()), n
    print(f"  exact match cand/on/bk/src; on-rate per lock = {oj.mean((0, 1))}")


def test_induct_in_mix():
    print("induct_in + induct_mix")
    m = _model()
    bucket = induct.lock_bucket(COUNTS, V)
    x = _xrep()
    xt = torch.from_numpy(x)
    cand, on, bk, src = m._induct_raw(xt, True)
    B, T = x.shape
    lw0 = (torch.rand(3, 7) * 4).float()
    lw0[1, 2] = 0.0                                # нульова довіра → замок не діє
    for dt, tol in ((torch.float32, 1e-5), (torch.bfloat16, 1e-2)):
        emb = (torch.randn(V, D)).to(dt).requires_grad_()
        gpar = torch.tensor([-2.0, -1.0, 0.5]).to(dt).requires_grad_()
        lw = lw0.to(dt)
        mix_w = (torch.randn(3, 7)).to(dt).requires_grad_()
        # --- torch: код із FullLM2.forward (INDUCT_POS) ---
        h = emb[xt]
        val = torch.stack([on[..., k].float() * lw[k][bk] for k in range(3)], -1)
        g = torch.sigmoid(gpar[:3])
        wsel = ((val != 0).float() * g[None, None])
        sel = (src[..., None] == torch.arange(T)).to(wsel.dtype)
        mixm = (sel * wsel[..., None]).sum(2)
        ht = h + torch.bmm(mixm.to(h.dtype), h).type_as(h)
        anyon = on.any(-1); kk = on.float().argmax(-1); c1 = (cand * on).sum(-1)
        gm = torch.sigmoid(mix_w[kk, bk]) * anyon
        mixg = gm.sum() / anyon.sum().clamp(min=1)
        R = torch.randn(B, T, D); Rg = torch.randn(B, T)
        ((ht.float() * R).sum() + (gm.float() * Rg).sum()).backward()
        # --- jax ---
        jd = jnp.bfloat16 if dt == torch.bfloat16 else jnp.float32
        cj, oj, bj, sj = (jnp.asarray(a) for a in induct.induct_raw(x, V, bucket))
        xj, Rj, Rgj, lwj = jnp.asarray(x), _tj(R), _tj(Rg), _tj(lw, jd)

        def f(emb_, g_, mw_):
            hj = emb_[xj]
            vj = induct.induct_val(oj, bj, lwj)
            hh = induct.induct_in(hj, vj, sj, g_)
            c1j, gj, stt = induct.induct_mix(cj, oj, bj, mw_)
            return (hh.astype(jnp.float32) * Rj).sum() + (gj.astype(jnp.float32) * Rgj).sum(), (hh, c1j, gj, stt)
        (_, (hh, c1j, gj, stt)), grads = jax.value_and_grad(f, (0, 1, 2), has_aux=True)(
            _tj(emb, jd), _tj(gpar, jd), _tj(mix_w, jd))
        tag = f"[{str(dt)[6:]}]"
        fw = float(np.abs(_np(hh) - _np(ht)).max())
        print(f"  induct_in fwd{tag} max_abs={fw:.3e} (bitwise={fw == 0.0})")
        RES.append(("induct_in fwd" + tag, fw, 0.0))
        assert np.array_equal(np.asarray(c1j), c1.numpy())
        cmp("induct_in d_emb" + tag, grads[0], emb.grad, tol)
        cmp("induct_in d_ind_in_g" + tag, grads[1], gpar.grad, tol)
        cmp("induct_mix g" + tag, gj, gm, 1e-6)
        cmp("induct_mix d_mix_w" + tag, grads[2], mix_w.grad, tol)
        cmp("induct_mix mix_g" + tag, stt["mix_g"], mixg, max(tol, 1e-5))   # torch: bf16-сума (стат.)


# ---------------- anchors ----------------
def test_anchors():
    print("anchors")
    B, T, frac, cap = 2, 1024, 0.2, 0.25
    for spread in (1.0, 3.0):                      # 3.0: якорів більше, ніж місткість A
        s = torch.randn(B, T).abs() * 3 * spread + 2
        s[:, 0] = 1e4
        tau = torch.tensor(5.0)
        anchor = s > tau
        q = torch.quantile(s[:, 1:].reshape(-1)[:100_000], 1 - frac)
        tau_t = tau.clone().mul_(0.9).add_(0.1 * q)
        A = max(1, round(T * cap))
        keep = anchor & (anchor.cumsum(1) <= A)
        idx = ((~keep).long() * T + torch.arange(T)).argsort(-1)[:, :A]
        valid = keep.gather(1, idx)
        st = valid.float().mean() * A / T
        ij, vj, tj, stj = anchors.select_anchors(jnp.asarray(s.numpy()), jnp.float32(5.0), frac, cap, True)
        assert np.array_equal(np.asarray(ij), idx.numpy()) and np.array_equal(np.asarray(vj), valid.numpy())
        cmp(f"anchors tau_new[spread={spread}]", tj, tau_t, 1e-7)
        cmp(f"anchors stat[spread={spread}]", stj["anchors"], st, 1e-6)
        _, _, te, _ = anchors.select_anchors(jnp.asarray(s.numpy()), jnp.float32(5.0), frac, cap, False)
        assert float(te) == 5.0
    print("  idx/valid exact")


def test_chunk_ce_token_blocks():
    """chunk_ce з блоками слів (N > ce.TOK, з неповним хвостом) == без блоків: ly/dh побітово, lse ≤2 ulp,
    dW — те саме fp32-накопичення з одним bf16-округленням (допуск 1 bf16-ulp на порядок сум)."""
    rng = np.random.default_rng(3)
    N, V, d, ch = 300, 96, 32, 40
    h = jnp.asarray(rng.standard_normal((N, d)), jnp.bfloat16)
    W = jnp.asarray(rng.standard_normal((V, d)) * 0.3, jnp.bfloat16)
    b = jnp.asarray(rng.standard_normal(V), jnp.float32)
    y = jnp.asarray(rng.integers(0, V, N), jnp.int32)
    gl, gy = (jnp.asarray(rng.standard_normal(N), jnp.float32) for _ in range(2))

    def run(tok):
        old, ce.TOK = ce.TOK, tok
        try:
            f = lambda h, W, b: ce.chunk_ce(h, W, b, y, ch)
            out, vjp = jax.vjp(jax.jit(f), h, W, b)
            return out, vjp((gl, gy))
        finally:
            ce.TOK = old
    (l0, y0), (dh0, dW0, db0) = run(10 ** 9)
    (l1, y1), (dh1, dW1, db1) = run(128)                  # 3 блоки, хвіст 84
    for a, c in ((y0, y1), (dh0, dh1)):
        assert np.array_equal(np.asarray(a, np.float32), np.asarray(c, np.float32))
    # lse: та сама формула на слово, але тіло lax.map компілюється окремо (інше злиття exp/суми на CPU) → ≤ 1-2 ulp fp32
    np.testing.assert_allclose(np.asarray(l0), np.asarray(l1), rtol=3e-7, atol=0)
    np.testing.assert_allclose(np.asarray(db0), np.asarray(db1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(dW0, np.float32), np.asarray(dW1, np.float32), rtol=1e-2, atol=1e-2)
    print("  chunk_ce token blocks: ly/dh exact, lse ≤2 ulp, dW/db fp32-accumulated")


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
    print("ALL OK")
