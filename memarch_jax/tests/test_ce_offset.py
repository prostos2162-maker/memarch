"""16.11 (fable52g): chunk_ce за великого СПІЛЬНОГО зсуву логітів (нуль-простір softmax) — причина зламів 34014/35750.
Вимоги до типового шляху (ce.CE_LOGITS=f32):
  1) втрата й градієнти = точному fp32-CE за будь-якого зсуву (зсув точність не з'їдає);
  2) градієнт по рядках таблиці має нульову суму (Σ_i (p_i − y_i) = 0) → немає сили вздовж спільного зсуву;
  3) градієнт по h не має складової вздовж середнього рядка понад шум округлення G до bf16.
Для порівняння — емуляція TPU до виправлення (CE_LOGITS=tpuA в окремому процесі): сума рядків ≠ 0.
python tests/test_ce_offset.py"""
import os, subprocess, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax, jax.numpy as jnp
from memarch_jax import ce

F32, BF = jnp.float32, jnp.bfloat16


def _case(off, N=96, V=1536, D=128):
    """h зі спільним вектором; рядки W = спільний вектор r̄ + мала корисна частина → усі логіти слова ≈ off."""
    k = jax.random.split(jax.random.PRNGKey(0), 4)
    u = jax.random.normal(k[0], (D,)); u = u / jnp.linalg.norm(u)
    h = (jax.random.normal(k[1], (N, D)) + off * u).astype(BF)
    W = (jax.random.normal(k[2], (V, D)) * 0.3 + u).astype(BF)
    y = jax.random.randint(k[3], (N,), 0, V)
    return h, W, y, u


def _stats(off):
    h, W, y, u = _case(off)
    f = lambda h, W: (lambda r: (r[0] - r[1]).mean())(ce.chunk_ce_nb(h, W, y, 512))
    L, (gh, gW) = jax.value_and_grad(f, argnums=(0, 1))(h, W)

    def ex(h, W):
        z = h.astype(F32) @ W.astype(F32).T
        return (jax.nn.logsumexp(z, -1) - jnp.take_along_axis(z, y[:, None], 1)[:, 0]).mean()
    Le, (ghe, gWe) = jax.value_and_grad(ex, argnums=(0, 1))(h, W)
    gW, gh = np.asarray(gW.astype(F32)), np.asarray(gh.astype(F32))
    rs = float(np.linalg.norm(gW.sum(0)) / np.sqrt((gW * gW).sum(1)).sum())
    cs = lambda a, b: float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return float(L), float(Le), rs, cs(gW, np.asarray(gWe)), cs(gh, np.asarray(ghe))


def test_offset_exact():
    assert ce.CE_LOGITS == "f32", ce.CE_LOGITS
    for off in (0.0, 40.0, 300.0, 2000.0):
        L, Le, rs, cW, ch = _stats(off)
        print(f"  зсув логітів ≈ {off:6.0f}: loss {L:.5f} точний {Le:.5f} | сума рядків dW {rs:.5f} | cos(dW, точний) {cW:.4f} "
              f"cos(dh, точний) {ch:.4f}")
        assert abs(L - Le) <= 2e-3 * max(1.0, abs(Le)), (off, L, Le)
        assert rs < 5e-3, (off, rs)
        assert cW > 0.995 and ch > 0.99, (off, cW, ch)
    print("OK test_offset_exact")


def test_tpu_emulation_breaks():
    """До виправлення на TPU (lse від bf16-округлених логітів, p — від неокруглених): сума рядків градієнта ≠ 0."""
    code = ("import sys; sys.path.insert(0, r'%s'); import tests.test_ce_offset as t; "
            "print('RS', *[t._stats(o)[2] for o in (40.0, 300.0, 2000.0)])" % os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    env = dict(os.environ, CE_LOGITS="tpuA", PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, encoding="utf-8").stdout
    rs = [float(x) for x in [l for l in out.splitlines() if l.startswith("RS")][0].split()[1:]]
    print(f"  емуляція TPU до виправлення: сума рядків dW за зсуву 40/300/2000 = {rs[0]:.4f} / {rs[1]:.4f} / {rs[2]:.4f}")
    assert rs[1] > 0.02 and rs[2] > 0.1, rs
    print("OK test_tpu_emulation_breaks")


if __name__ == "__main__":
    test_offset_exact()
    test_tpu_emulation_breaks()
    print("ALL OK")
