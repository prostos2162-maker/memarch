"""Бенчмарк кроку тренування на TPU (v6e-8) або на CPU (smoke): одна jit-програма dist.make_step
(fwd+bwd у shard_map + Instinct, remat, donate) на всіх чипах сітки.

  python -m memarch_jax.bench_tpu --model Fv2moeK40h14d2560 --batch 1 --seq 8192 --remat heavy --steps 20
  python -m memarch_jax.bench_tpu --model small --batch 1 --seq 256 --steps 5 --cpu8      # smoke на 8 CPU-пристроях

Друкує: час компіляції (lower/compile окремо), мс на крок (кожен + медіана після прогріву), tok/s (усі чипи),
MFU = STEP_FLOPS·tok/s / (W·PEAK), пік HBM на чип (device.memory_stats), XLA memory_analysis, кількість
перекомпіляцій після кроку 1 (має бути 0), опційно trace jax.profiler (--profile DIR). Останній рядок —
"RESULT {json}" для скриптів запуску.

--batch — рядків НА ЧИП (глобальний батч = W·batch), як --batch у hyperop.tpu8."""
import argparse
import dataclasses
import json
import os
import sys
import time


def _args(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="Fv2moeK40h14d2560", help="Fv2moeK40h14d2560 | small | tiny_k40")
    a.add_argument("--batch", type=int, default=1, help="рядків на чип")
    a.add_argument("--seq", type=int, default=8192)
    a.add_argument("--remat", default="heavy", help="none|heavy|all|dots_saveable|dots_nobatch|dots_nobatch_all|moe_save|moe_save_dots|moe_save_heavy|moe_save_out|moe_save_a2a|moe_save_disp|moe_save_lite або light,heavy,mem (частина: False|True|назва policy з model._policies)")
    a.add_argument("--offload_opt", action="store_true",
                   help="Instinct m у pinned_host (хост), на чип лише на час оновлення — −≈3.4 GiB args/чип")
    a.add_argument("--steps", type=int, default=20)
    a.add_argument("--warmup", type=int, default=3, help="кроків без обліку в медіані (після компіляції)")
    a.add_argument("--lr", type=float, default=6e-4)
    a.add_argument("--dtype", default=None, help="bf16|f32 (за замовчуванням: bf16 для цілі, f32 для small)")
    a.add_argument("--attn", default=None, help="splash|dense (за замовчуванням: splash на TPU)")
    a.add_argument("--no_pallas", action="store_true", help="Instinct без Pallas-ядер")
    a.add_argument("--counts", default=None, help="counts_fineweb.npy (інакше синтетичний Zipf)")
    a.add_argument("--lock_trust", default=None, help="lock_trust.pt або .npy (інакше одиниці)")
    a.add_argument("--step_flops", type=float, default=float(os.environ.get("STEP_FLOPS", 4.4e9)),
                   help="FLOP на токен (fwd+bwd), для MFU")
    a.add_argument("--peak_tflops", type=float, default=float(os.environ.get("PEAK_TFLOPS", 918.0)),
                   help="пік bf16 на чип, TFLOP/s (v6e = 918)")
    a.add_argument("--profile", default=None, help="тека для jax.profiler trace")
    a.add_argument("--profile_steps", type=int, default=3)
    a.add_argument("--chips", type=int, default=None, help="W (за замовчуванням усі пристрої)")
    a.add_argument("--cpu8", action="store_true", help="8 CPU-пристроїв (XLA_FLAGS) — smoke без TPU")
    a.add_argument("--moe_chunks", type=int, default=None, help="cfg.moe_a2a_chunks (шматки a2a для перекриття)")
    a.add_argument("--moe_impl", default=None, help="cfg.moe_a2a_impl: a2a|ppermute")
    a.add_argument("--moe_old", action="store_true", help="cfg.moe_fast=False — старий шлях MoE (A/B)")
    a.add_argument("--vocab_take", default=None, help="cfg.vocab_take: ag|masked")
    a.add_argument("--vocab_take_bwd", default=None, help="cfg.vocab_take_bwd: sort|scatter (бекворд take, A/B)")
    a.add_argument("--no_ce_a2a", action="store_true", help="cfg.vocab_ce_a2a=False (A/B)")
    a.add_argument("--rope_old", action="store_true", help="light.ROPE_BTHD=False (A/B)")
    a.add_argument("--hoop", action="store_true", help="cfg.hoop=True: обруч замість якорів (важкі лише top budget·T)")
    a.add_argument("--hoop_budget", type=float, default=0.125, help="cfg.hoop_budget (частка слів у важкі)")
    a.add_argument("--host_init", action="store_true", help="ініціалізація на хості (інакше jit з out_shardings)")
    return a.parse_args(argv)


def _cfg(args):
    from . import config
    base = {"Fv2moeK40h14d2560": config.target, "small": config.small, "tiny_k40": config.tiny_k40}[args.model]()
    kw = dict(B=args.batch, T=args.seq)
    if args.dtype:
        kw["dtype"] = args.dtype
    if args.model != "Fv2moeK40h14d2560":
        kw.setdefault("dtype", "f32")
        kw["window"], kw["seg"] = min(base.window, args.seq), min(base.seg, args.seq)
    if args.attn:
        kw["attn_impl"] = args.attn
    # прапорці швидкого шляху (звіт prof_b4_all); за замовчуванням — значення з config
    if args.moe_chunks:
        kw["moe_a2a_chunks"] = args.moe_chunks
    if args.moe_impl:
        kw["moe_a2a_impl"] = args.moe_impl
    if args.moe_old:
        kw["moe_fast"] = False
    if args.vocab_take:
        kw["vocab_take"] = args.vocab_take
    if args.vocab_take_bwd:
        kw["vocab_take_bwd"] = args.vocab_take_bwd
    if args.no_ce_a2a:
        kw["vocab_ce_a2a"] = False
    if args.hoop:
        kw["hoop"], kw["hoop_budget"] = True, args.hoop_budget
    if args.rope_old:
        from . import light
        light.ROPE_BTHD = False
    return dataclasses.replace(base, **kw)


def _consts(cfg, args):
    import numpy as np
    from . import model
    if args.counts and os.path.exists(args.counts):
        cnt = np.load(args.counts).astype(np.int64)[:cfg.V]
        cnt = np.pad(cnt, (0, cfg.V - len(cnt)))
    else:
        r = np.random.default_rng(0)
        cnt = np.floor(1e9 / np.arange(1, cfg.V + 1) ** 1.1 * r.uniform(0.5, 1.5, cfg.V)).astype(np.int64)
        cnt[cfg.V - 1] = 0
        print("[bench] counts: синтетичний Zipf (немає --counts)")
    lt = np.ones((3, 7), np.float32)
    if args.lock_trust and os.path.exists(args.lock_trust):
        if args.lock_trust.endswith(".npy"):
            lt = np.load(args.lock_trust).astype(np.float32)
        else:
            from . import config
            lt = config.load_lock_trust(args.lock_trust)
    return model.build_consts(cfg, cnt, lt), cnt, lt


def _batches(cfg, consts, W, n, lr, seed=0):
    """n різних хостових батчів (W·B, T+1): Zipf + повтори фраз (щоб induct/якорі працювали як на тексті)."""
    import numpy as np
    from . import step as S
    g = np.random.default_rng(seed)
    out, t_prep = [], 0.0
    for _ in range(n):
        xy = np.minimum(g.zipf(1.2, (W * cfg.B, cfg.T + 1)), cfg.V - 2).astype(np.int64)
        for b in range(xy.shape[0]):
            for _ in range(cfg.T // 256):
                L = g.integers(3, 16)
                s = g.integers(0, cfg.T // 2)
                d_ = g.integers(L, cfg.T + 1 - s - L)
                xy[b, s + d_:s + d_ + L] = xy[b, s:s + L]
        t0 = time.perf_counter()
        out.append(S.make_batch(xy, cfg, consts, lr))
        t_prep += time.perf_counter() - t0
    return out, t_prep / n


def _hbm():
    import jax
    rows = []
    for d in jax.local_devices():
        try:
            ms = d.memory_stats() or {}
        except Exception:                                        # CPU: memory_stats може бути відсутній
            ms = {}
        rows.append(dict(dev=d.id, peak=ms.get("peak_bytes_in_use"), inuse=ms.get("bytes_in_use"),
                         limit=ms.get("bytes_limit")))
    return rows


def main(argv=None):
    args = _args(argv)
    if args.cpu8:
        os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np
    import jax
    import jax.numpy as jnp
    from . import model, dist, instinct
    from .dist import NamedSharding, P

    compiles = []
    jax.monitoring.register_event_duration_secs_listener(
        lambda ev, d, **kw: compiles.append(d) if ev == "/jax/core/compile/backend_compile_duration" else None)

    plat = jax.devices()[0].platform
    if args.attn is None:
        args.attn = "splash" if plat == "tpu" else "dense"
    cfg = _cfg(args)
    W = args.chips or len(jax.devices())
    cfg = dataclasses.replace(cfg, W=W)
    mesh = dist.make_mesh(W)
    print(f"[bench] jax {jax.__version__} платформа {plat} W={W} пристрої {jax.devices()[0].device_kind}")
    print(f"[bench] cfg: d={cfg.d} E={cfg.E} heavy={cfg.heavy} V={cfg.V} B/чип={cfg.B} T={cfg.T} dtype={cfg.dtype} "
          f"attn={cfg.attn_impl} remat={args.remat}")
    consts, _, lt = _consts(cfg, args)
    # частина кортежу: True/False або будь-яка назва policy (раніше невідомі назви мовчки ставали False)
    remat = tuple({"True": True, "False": False}.get(x, x)
                  for x in args.remat.split(",")) if "," in args.remat else args.remat
    model.remat_parts(remat)                                       # перевірка назв до компіляції
    use_pallas = plat == "tpu" and not args.no_pallas and cfg.dtype == "bf16"
    icfg = instinct.InstinctCfg(lr_peak=args.lr, use_pallas=use_pallas, sr=cfg.dtype == "bf16",
                                m_bf16=cfg.dtype == "bf16")

    # --- ініціалізація ваг одразу шардованими (24B параметрів не проходять через хост)
    t0 = time.perf_counter()
    key = jax.random.PRNGKey(0)
    if args.host_init:
        params = model.init_params(key, cfg, lt)
    else:
        shapes = jax.eval_shape(lambda k: model.init_params(k, cfg, lt), key)
        shard = jax.tree_util.tree_map(lambda s: NamedSharding(mesh, s), dist.param_specs(shapes),
                                       is_leaf=lambda s: isinstance(s, P))
        params = jax.jit(lambda k: model.init_params(k, cfg, lt), out_shardings=shard)(key)
    n_par = sum(x.size for x in jax.tree_util.tree_leaves(params))
    spec = instinct.make_spec(params)
    ts, state, place = dist.make_step(cfg, consts, mesh, params, spec, icfg, remat=remat,
                                     offload_opt=args.offload_opt)
    del params
    jax.block_until_ready(state)
    t_init = time.perf_counter() - t0
    for pth, x in jax.tree_util.tree_flatten_with_path(state)[0]:   # контроль: великий листок без шардування
        if x.nbytes > 2**28 and x.sharding.is_fully_replicated:
            print(f"[bench] УВАГА репліковано {jax.tree_util.keystr(pth)} {x.shape} {x.nbytes / 2**30:.2f}GiB")
    print(f"[bench] параметрів {n_par / 1e9:.3f}B, init {t_init:.1f} с")

    host, t_prep = _batches(cfg, consts, W, min(4, max(1, args.steps)), args.lr)
    batches = [place(b) for b in host]
    print(f"[bench] хостова підготовка батчу (induct_raw): {t_prep * 1e3:.0f} мс")

    # --- компіляція окремо від виконання
    t0 = time.perf_counter()
    lowered = ts.lower(state, batches[0])
    t_lower = time.perf_counter() - t0
    t0 = time.perf_counter()
    compiled = lowered.compile()
    t_comp = time.perf_counter() - t0
    mem = None
    try:
        ma = compiled.memory_analysis()
        mem = {k: getattr(ma, k) for k in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
                                           "generated_code_size_in_bytes", "alias_size_in_bytes") if hasattr(ma, k)}
    except Exception as e:                                       # noqa: BLE001
        mem = {"error": repr(e)}
    xla_flops = None
    try:
        ca = compiled.cost_analysis()
        ca = ca[0] if isinstance(ca, (list, tuple)) else ca
        xla_flops = float(ca.get("flops", 0.0)) if ca else None
    except Exception:                                            # noqa: BLE001
        pass
    print(f"[bench] lower {t_lower:.1f} с, compile {t_comp:.1f} с; memory_analysis (на чип): "
          + ", ".join(f"{k}={v / 2**30:.2f}GiB" for k, v in mem.items() if isinstance(v, (int, float))))

    # --- кроки
    tok = W * cfg.B * cfg.T
    times, losses, n_comp0 = [], [], len(compiles)
    prof_on = False
    for i in range(args.steps):
        if args.profile and i == args.warmup and not prof_on:
            jax.profiler.start_trace(args.profile)
            prof_on = True
        t0 = time.perf_counter()
        try:
            state, m = compiled(state, batches[i % len(batches)])  # та сама програма, без повторної компіляції
        except TypeError as e:                                   # jax 0.11: hoisted const_args у AOT-виклику
            if i: raise
            print(f"[bench] AOT-виклик не пройшов ({e}); далі через jit-кеш")
            compiled = ts
            state, m = compiled(state, batches[i % len(batches)])
        loss = float(m["loss"])                                  # синхронізація з пристроєм
        dt = time.perf_counter() - t0
        times.append(dt)
        losses.append(loss)
        if prof_on and i == args.warmup + args.profile_steps - 1:
            jax.profiler.stop_trace()
            prof_on = False
            print(f"[bench] trace → {args.profile}")
        print(f"[bench] step {i:3d} {dt * 1e3:9.1f} ms  {tok / dt:12,.0f} tok/s  loss {loss:.4f}  "
              f"tau {float(m['tau']):.4f}", flush=True)
    if prof_on:
        jax.profiler.stop_trace()
    recompiles = len(compiles) - n_comp0
    steady = times[args.warmup:] or times
    med = float(np.median(steady))
    tps = tok / med
    mfu = args.step_flops * tps / (W * args.peak_tflops * 1e12)
    hbm = _hbm()
    peaks = [r["peak"] for r in hbm if r["peak"] is not None]
    print(f"[bench] медіана {med * 1e3:.1f} ms/крок (min {min(steady) * 1e3:.1f}), {tps:,.0f} tok/s на {W} чипів, "
          f"MFU {mfu * 100:.2f}% (STEP_FLOPS {args.step_flops:.3g}/tok, пік {args.peak_tflops} TF/чип)")
    if xla_flops:
        print(f"[bench] XLA cost_analysis: {xla_flops:.3e} FLOP/крок/чип → {xla_flops * W / tok:.3e} FLOP/tok")
    if peaks:
        print(f"[bench] пік HBM на чип: max {max(peaks) / 2**30:.2f} GiB, min {min(peaks) / 2**30:.2f} GiB, "
              f"ліміт {(hbm[0]['limit'] or 0) / 2**30:.2f} GiB")
    print(f"[bench] перекомпіляцій після кроку 1: {recompiles}")
    res = dict(model=args.model, W=W, batch=cfg.B, seq=cfg.T, remat=args.remat, dtype=cfg.dtype, attn=cfg.attn_impl,
               params_B=n_par / 1e9, init_s=t_init, lower_s=t_lower, compile_s=t_comp, step_ms=med * 1e3,
               step_ms_all=[t * 1e3 for t in times], tok_s=tps, mfu=mfu, hbm_peak_GiB=[p / 2**30 for p in peaks],
               mem_analysis=mem, xla_flops_per_chip=xla_flops, recompiles=recompiles, losses=losses,
               host_prep_ms=t_prep * 1e3, profile=args.profile)
    print("RESULT " + json.dumps(res))
    return res


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
