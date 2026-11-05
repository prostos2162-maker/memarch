# memarch: a memory-inspired language model architecture

memarch is an experimental language-model architecture built from scratch by one independent developer. It is not a standard transformer and not an SSM. The goal is to **train faster than a transformer of the same size at equal quality**, measured as wall-clock time on the same GPU to reach the same validation loss.

The design starts from how human reading and memory work. Most words are read quickly, and heavy "thinking" happens only on surprising words. Common phrases are known from birth. Words already seen in the text are recognised and copied, not re-derived.

## Core ideas

- **Thinking on surprises (anchors).** Light sliding-window layers process every token. Heavy layers (a mixture of experts, 2 experts active per token) run only on the ~20% of tokens the model finds most surprising. The selection is causal: a running threshold plus a cap of 25% of tokens per text.
- **Born with a phrasebook.** The input/output word tables start from corpus statistics: an SVD of the bigram PPMI matrix plus a word-frequency bias. A separate table maps two-word contexts to the next word. The model starts training at loss 7.2 instead of 11.0.
- **Surprise from the phrasebook.** An anchor is a word that breaks a familiar phrase. This replaces a second full-vocabulary pass for measuring surprise.
- **Keys and locks (recognition reflex).** A hard-wired bookmark to the last occurrence of the current token and bigram proposes the word that followed it as an output candidate. Trust in each lock is calibrated on data before training, and a mixture head lets the pointer "interrupt" the normal prediction.
- **Independent memory.** Fixed-size fresh and dim memory with surprise-based writing, entity cards with bookmarks into the raw text, and re-reading. It is a separate system with its own signal, not tied to anchors.
- **Own optimizer ("instinct").** It keeps one momentum state per weight (half the optimizer state of AdamW). The step size follows prediction surprise, and a "homeostasis" normalisation balances each neuron's input and output updates.
- **Wave training of experts.** Experts that stopped changing move to a frozen bf16 bank, so per-step cost does not grow with the number of experts (6× parameters cost +2% step time).

## Results

All runs use the same FineWeb-Edu data (GPT-2 tokenizer) and the same validation windows. Each configuration was run once.

### Scale test, ~1B parameters (RTX PRO 6000 Blackwell, ~25 min per model)

| | params | ms/step (32k tok) | MFU | final val loss | time | time to 4.078 |
|---|---|---|---|---|---|---|
| Transformer, 16 × 2048 | 1.28B | 768 | 77% | 4.078 | 25.5 min | 25.5 min |
| **memarch** `Fv2moeK8h8d2048` | 2.25B (MoE) | **448** | 68% (useful) | **3.858** | 24.9 min | **~16.5 min (−35%)** |

memarch leads from the start: 4.84 vs 6.09 at minute 2. It reaches the transformer's final loss 35% sooner and finishes 0.22 lower.

**Caveats:**
- **Baseline optimizer.** Both models use our optimizer, which outperforms AdamW and Muon on the same transformer (see below). The 1B baseline is therefore a strengthened transformer, not an AdamW one.
- **Memory.** Memory was disabled in this test (texts are 1024 tokens, so the window already sees the whole text).

### Against a transformer trained with AdamW, ~300M parameters

| | final val | time to 3.922 |
|---|---|---|
| Transformer 16 × 1024, AdamW | 3.922 | 18.0 min |
| memarch (surprise anchors, cap) | 3.902 | ~14.7 min (−15%) |
| memarch (dense v2 with memory) | 3.901 | ~15.7 min (−13%) |

### Optimizer, same transformer (512 × 16, ~10M tokens)

| optimizer | val loss |
|---|---|
| AdamW | 5.818 |
| Muon | 5.786 |
| **instinct + homeostasis (ours)** | **5.555** |

### Ablations of our own ideas (short screens, ~10M tokens)

| idea | Δ val loss |
|---|---|
| homeostasis normalisation in the optimizer | −0.075 |
| phrasebook from birth (PPMI SVD init + frequency bias) | **−0.49** |
| two-word phrase table | −0.066 |
| keys and locks (recognition reflex) | −0.086 |
| mixture "interrupting" pointer head | −0.07 |

### Against modded-nanogpt (honest comparison)

[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) is the world speed record for this kind of training. We ran it untouched except for hardware ports (`race/`), on the same data and validation windows:

| setting | modded-nanogpt | memarch |
|---|---|---|
| GPT-2 Medium size, RTX PRO 6000 | 3.634 in 14.7 min | 3.901 in 12.0 min |
| d=512, RTX 3060 | 4.083 in 16.3 min | 4.433 in 14.9 min |

memarch finishes earlier and leads for the first ~4 minutes, but its final loss is still behind. Closing this gap using only our own ideas is ongoing work.

### Engineering

- **No host syncs.** The training step runs entirely on the GPU, with no CPU synchronisation (`sync_audit.py`) and weight updates fused into the backward pass.
- **Custom kernels.** Triton kernels cover the fused output + cross-entropy step (including the pointer mixture), SwiGLU, and the optimizer step.
- **Data pipeline.** About 300B tokens from permissively licensed sources: FineWeb-Edu, FineMath, The Stack v3 (permissive only), Nemotron-CC-Math, HPLT Ukrainian, FineWiki and others. The pipeline adds MinHash deduplication across all sources (6.97% of documents dropped), 13-gram benchmark decontamination (HumanEval, MBPP, GSM8K, MATH, MMLU, ARC, HellaSwag) and code-quality scoring. See `data/`.

## TPU / JAX port

`memarch_jax/` is a JAX port of the full training step (model, memory, MoE, optimizer) written for TPU v6e.
It is a set of pure functions over a parameter pytree and runs on all chips of one host with `shard_map`:
data parallelism for the light layers and memory, expert parallelism with all-to-all for the MoE layers, and
Pallas kernels for the optimizer update. A CPU test suite checks it against the PyTorch reference in this
repository (forward, loss, gradients and optimizer steps), including 8-device runs on emulated CPU devices.

```bash
cd memarch_jax
export XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONIOENCODING=utf-8
python tests/test_dist_step.py                                           # 8-device training step on CPU
python -m memarch_jax.bench_tpu --model small --seq 256 --steps 5 --cpu8   # benchmark smoke test
```

On a TPU v6e-8 VM, `python -m memarch_jax.bench_tpu --model Fv2moeK40h14d2560 --batch 1 --seq 8192 --remat heavy`
reports step time, tokens/s, MFU and peak HBM. Installation, flags and environment variables are described in
[`memarch_jax/docs/README_TPU.md`](memarch_jax/docs/README_TPU.md). The PyTorch code also runs on TPU through
`DEVICE=tpu` / `DEVICE=xla` (`dev.py`).

Measured throughput (Nov 6, 2026): the full ~12B model trains at **507k tokens/s on 8× TPU v6e**
(517 ms per step, 4×8192 tokens per chip, memory, experts and the Instinct optimizer on).

## Instinct optimizer

Instinct (`exp_opt.InstinctOptB`, JAX version in `memarch_jax/memarch_jax/instinct.py`) is the project's own
optimizer. It keeps a single momentum state per weight, scales the step by prediction surprise, and applies a
"homeostasis" normalisation that balances each neuron's input and output updates. Weights can be kept in bf16
with stochastic rounding; on TPU the update is a fused Pallas kernel.

## Uroboros

Uroboros (`memarch_jax/memarch_jax/hoop.py`, flags `--hoop` / `--hoop_budget`) is a new experimental component
under development; details will be published after validation.

## Layout

| path | contents |
|---|---|
| `full2.py`, `full.py`, `mem2.py` | the model: anchors, light/heavy layers, MoE banks, memory |
| `exp_opt.py` | the Instinct optimizer |
| `fused_ops.py`, `int8mm.py` | Triton kernels; optional int8 matrix multiplication (`INT8_MM=1`) |
| `fastloop.py` | training loop (`speed` / `train`) |
| `dev.py` | device selection (`DEVICE=cuda\|cpu\|tpu\|xla`) |
| `lm.py` | transformer baseline |
| `memarch_jax/` | JAX/TPU port: package, tests, `docs/README_TPU.md` |
| `tools/phr/` | phrasebook, phrase tables, lock trust, per-token diagnostics |
| `data/` | tokenization, dedup, decontamination, code scoring, source licenses |
| `tok/` | Qwen3.6 tokenizer pruned to 131k |
| `race/` | modded-nanogpt harness (MIT, see `race/LICENSE.modded-nanogpt`) |
| `scripts/` | example race / scale-test launchers |

Code comments are in Ukrainian.

## Status and next steps

The architecture and data are ready for a main training run (~12B total / ~2B active, distilled from Qwen3.6-35B-A3B, target: a coding LLM with strong Ukrainian). The JAX/TPU port is in place and reaches the 500k tokens/s throughput target on 8 TPU v6e chips. The next steps are an AdamW 1B baseline, multiple seeds, downstream benchmarks, and long-context tests of the memory system.

## License

Apache-2.0. `race/` contains modified modded-nanogpt code under its MIT license.
