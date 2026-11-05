# memarch_jax on TPU

`memarch_jax` is a JAX port of the memarch training step (model `Fv2moeK40h14d2560`): pure functions over a parameter
pytree, distributed over all chips of one host with `shard_map` (data parallel for the light layers and memory,
expert parallel with all-to-all for the MoE layers). The CPU test suite checks the port against the PyTorch
reference in the repository root.

## Install on a TPU VM (v6e-8)

v6e-8 is a single host with 8 chips (`--accelerator-type v6e-8`, runtime image `v2-alpha-tpuv6e`). Python >= 3.11.

```bash
python3 -m venv ~/venv && . ~/venv/bin/activate
pip install -U pip
pip install "jax[tpu]==0.11.2" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install numpy orbax-checkpoint            # orbax is only needed by ckpt.py
python -c "import jax; print(jax.__version__, jax.devices())"   # expect 8 TPU devices
```

PyTorch is not needed on the VM. Without it, `--lock_trust` accepts only a `.npy` file
(`np.save('lock_trust.npy', torch.load('lock_trust.pt').float().numpy())` on a machine with torch).

## Benchmark

```bash
cd memarch_jax            # the folder that contains the memarch_jax/ package
python -m memarch_jax.bench_tpu --model Fv2moeK40h14d2560 --batch 1 --seq 8192 --remat heavy --steps 20 \
    --counts phr/counts_fineweb.npy --lock_trust phr/lock_trust.npy \
    --profile trace_heavy --profile_steps 3 2>&1 | tee bench.log
```

The script prints lower/compile time, per-step time and the median after `--warmup`, tokens/s over all chips, MFU,
peak HBM per chip, XLA memory and cost analysis, and the number of recompilations after step 1 (expected: 0).
The last line, `RESULT {json}`, is meant for scripts.

CPU smoke test without a TPU:

```bash
python -m memarch_jax.bench_tpu --model small --seq 256 --steps 5 --cpu8
```

### Main flags

| flag | meaning |
|---|---|
| `--model` | `Fv2moeK40h14d2560` (target), `small`, `tiny_k40` |
| `--batch` | rows **per chip**; tokens per step = chips x batch x seq |
| `--seq` | sequence length |
| `--remat` | rematerialisation policy: `none`, `heavy`, `all`, `dots_saveable`, `dots_nobatch`, `dots_nobatch_all`, `moe_save*` variants, or a triple `light,heavy,mem` where each part is `False`, `True` or a policy name (e.g. `dots_nobatch,True,True`) |
| `--offload_opt` | keep optimizer state in host memory |
| `--counts`, `--lock_trust` | phrasebook statistics; without them synthetic Zipf counts are used (speed is unaffected) |
| `--attn` | `splash` (default on TPU) or `dense` |
| `--no_pallas` | Instinct optimizer without the Pallas kernels (pure jnp) |
| `--moe_chunks`, `--moe_impl` | MoE all-to-all chunking and implementation (`a2a` or `ppermute`) |
| `--hoop`, `--hoop_budget` | enable the experimental Uroboros component and set the fraction of tokens it sends to the heavy layers |
| `--profile DIR`, `--profile_steps N` | write a `jax.profiler` trace |
| `--cpu8` | emulate 8 devices on CPU |

### Environment variables

| variable | meaning |
|---|---|
| `MOE_UNROLL=1` | unroll the heavy-layer stack as a Python loop instead of `lax.scan` (longer compile) |
| `MOE_UP_MID=w\|e` | layout of the expert intermediate: `w` = (W, E_local, c, h), `e` = expert-major |
| `STEP_FLOPS`, `PEAK_TFLOPS` | constants used for the MFU estimate |
| `MEMARCH_PHR` | default folder for `counts_fineweb.npy` / `lock_trust.pt` |
| `MEMARCH_DATA_ROOT` | default corpus root for `data.Corpus` (local folder or `gs://...`) |

## Tests

From this folder, on CPU with 8 emulated devices:

```bash
export XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONIOENCODING=utf-8
python tests/test_dist.py        # any tests/test_*.py file runs as a script or under pytest
```

Parity tests (`test_light`, `test_moe`, `test_memory`, `test_reread`, `test_heads`, `test_model`, `test_instinct`,
`test_combined`) compare against the PyTorch code in the repository root and need `torch` installed.

## Package layout

| module | contents |
|---|---|
| `model.py`, `light.py`, `attn.py`, `moe.py`, `memory.py`, `reread.py`, `anchors.py`, `induct.py`, `ce.py` | model components |
| `step.py`, `dist.py` | training step and 8-chip `shard_map` distribution |
| `instinct.py`, `pallas_instinct.py`, `pallas_hs1.py` | Instinct optimizer and its Pallas kernels |
| `hoop.py` (alias `uroboros.py`) | experimental Uroboros component |
| `data.py`, `ckpt.py`, `convert.py`, `config.py` | data loader, checkpoints, torch <-> JAX conversion, configuration |
| `bench_tpu.py` | TPU benchmark |
