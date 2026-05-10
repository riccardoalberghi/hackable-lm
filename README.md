# hackable-lm

A minimal, hackable language-model pretraining repo for fast architecture and
training experiments on one NVIDIA L40 GPU.

The baseline is a decoder-only causal LM with byte-level BPE tokenization,
untied token embedding and LM head, RoPE, non-parametric RMSNorm,
QK norm, 512-token sliding-window causal self-attention with every fourth layer
full attention and the final layer always full attention, SwiGLU MLPs,
fused QKV and fused gate/up projection matrices, `bias=False` linear layers,
Muon for transformer matrix weights, and AdamW for embeddings, LM head, and
scalar/vector parameters.

## Quickstart

```bash
uv sync --locked

uv run python prepare_data.py all \
  --input data/raw/*.txt \
  --vocab-size 32768 \
  --output data/processed

uv run python train.py \
  --depth 12 \
  --data data/processed \
  --run-name d12

# MLflow logs are written locally by default:
# runs/mlruns/<experiment>/<run> plus the normal runs/d12 artifacts.
uv run mlflow ui --backend-store-uri file://$(pwd)/runs/mlruns

uv run python prepare_eval.py --source hf --output eval_data

uv run python eval.py \
  --checkpoint runs/d12/checkpoints/latest.pt \
  --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq \
  --limit-per-task 128
```

Dependencies are managed by `uv` through `pyproject.toml` and `uv.lock`. The
lockfile targets the CUDA Linux x86_64 training path. Tokenization uses the
local Rust backend in `rustbpe/`, so `cargo` must be available when the native
extension needs to be built or rebuilt.

Normal scaling uses one knob:

```bash
uv run python train.py --depth 20 --data data/processed --run-name d20
```

Depth is only the default shape shortcut. For follow-up comparisons, the same
resolver can hold a different quantity fixed:

```bash
# Approximately match a previous run's scaling-parameter count.
uv run python train.py --target-params 150000000 --comparison-mode same_params --data data/processed --run-name p150m

# Keep the same token, byte, or FLOP budget across a candidate.
uv run python train.py --depth 12 --target-tokens 2000000000 --comparison-mode same_tokens --data data/processed --run-name d12_tokens
uv run python train.py --depth 12 --target-bytes 8000000000 --comparison-mode same_bytes --data data/processed --run-name d12_bytes
uv run python train.py --depth 12 --target-flops 3e19 --comparison-mode same_flops --data data/processed --run-name d12_flops

# Fill the relevant fields from an existing manifest.
uv run python train.py \
  --match-run runs/d12/manifest.json \
  --comparison-mode same_tokens \
  --depth 12 \
  --data data/processed \
  --run-name candidate_same_tokens
```

`train.py` is a strict fast-path script. If CUDA, FlashAttention 2, or requested
Liger kernels are missing, it fails with an install-oriented error instead of
silently falling back.
For a quick overfit sanity check, pass `--overfit-first-batch` to repeatedly
train on the first sampled training batch.

The CUDA training path keeps model parameters and Muon/AdamW optimizer state in
bf16. It does not maintain separate FP32 master weights.
Muon updates fused QKV and gate/up matrices as separate row-slice virtual
matrices, matching the optimizer behavior of unfused projections.

`train.py` also logs each run to MLflow by default. The default tracking URI is
a local file store under `--runs-dir/mlruns`, so training does not need network
access. Use `--mlflow-tracking-uri` for a different local or remote MLflow
store, `--mlflow-experiment` to group runs, `--mlflow-run-name` to override the
display name, or `--no-mlflow` for environments without the `mlflow` package.

## Scaling Policy

The default policy is `depth_simple` and is implemented in `config.py`. For a
given `--depth` shape policy:

```text
n_head = ceil(depth / 2)
n_embd = 128 * n_head
n_layer = depth
mlp_hidden = align(ceil(8 * n_embd / 3), 256)
```

Scaling parameters include transformer matrix weights and the LM head. Token
embeddings, RoPE buffers, optimizer-only state, and tiny scalar/vector
parameters are excluded. The token budget, batch tokens, learning rates, weight
decay, and iteration count use simple formulas unless an explicit budget flag is
supplied:

```text
target_tokens = target_param_data_ratio * scaling_params  # default ratio: 60
sequence_len = 2048
attention_window = 512  # use --attention-window 0 for full causal attention
attention_full_every = 4  # three local layers, then one full layer
global_batch_tokens ~= reference_batch_tokens * sqrt(depth / reference_depth)
lr_scale = sqrt(global_batch_tokens / reference_batch_tokens)
lr_schedule = WSD
warmup_ratio = 0.05
weight_decay = constant
steps = target_tokens // global_batch_tokens
```

Every run manifest records the resolved `shape_policy`, `budget_policy`,
`comparison_mode`, scheduled tokens, and train FLOPs budget.

Training logs report `mfu` on the standard BF16-peak denominator.

## Baseline Contract

Primary metric: validation loss on the processed pretraining validation memmap.

Secondary metrics: local loglikelihood/continuation tasks from `eval.py`,
including HellaSwag, PIQA, ARC-Easy, ARC-Challenge, OpenBookQA, Winogrande, and
BoolQ.

For canonical benchmark methodology, use the EleutherAI harness adapter:

```bash
uv run python run_lm_eval.py \
  --checkpoint runs/<run_id>/checkpoints/latest.pt \
  --tokenizer data/<dataset>/tokenizer.json \
  --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq \
  --device cuda \
  --batch-size 8 \
  --output runs/<run_id>/eval/lm_eval_results.json
```

The adapter lives in `lm_eval_simple_lm.py`, and `run_lm_eval.py` passes it to
`lm_eval.simple_evaluate`. This keeps this repo's local `eval.py` as the quick
sanity evaluator while letting the external harness own task prompts, splits,
metrics, and reporting.

A valid baseline-vs-candidate comparison should match:

- depth
- data files and hashes
- tokenizer hash
- train/val split seed
- sequence length
- attention window
- full attention interval
- global batch tokens
- token budget or number of tokens seen
- optimizer grouping unless the optimizer is the intervention
- LR schedule, momentum, and weight decay unless one is the intervention
- precision and kernels unless those are the intervention
- scaling policy
- seed, unless running an explicit multi-seed comparison
- eval datasets and hashes

Candidate changes should be a small code diff plus a manifest label passed with
`--candidate-label`.

Use `--comparison-mode` to say what the comparison is trying to hold fixed:
`same_depth`, `same_params`, `same_tokens`, `same_bytes`, `same_flops`, or
`same_time`. `same_depth` is the fastest first-pass ablation. Use `same_params`
when the intervention changes capacity, `same_bytes` for tokenizer or
preprocessing changes, and `same_flops` or `same_time` for efficiency claims.

## Paper-Ready Manual Comparison

This repo intentionally keeps comparisons CLI-controlled instead of wrapping
them in an experiment runner. A defensible manual flow is:

```bash
uv run python train.py \
  --depth 12 \
  --data data/processed \
  --run-name base_d12

uv run python train.py \
  --match-run runs/base_d12/manifest.json \
  --comparison-mode same_tokens \
  --data data/processed \
  --candidate-label idea_x \
  --run-name idea_x_same_tokens

uv run python repro.py compare \
  runs/base_d12/manifest.json \
  runs/idea_x_same_tokens/manifest.json \
  --fail-on-warning

uv run python eval.py \
  --checkpoint runs/base_d12/checkpoints/latest.pt \
  --tasks validation_loss,hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq \
  --data data/processed \
  --eval-data eval_data

uv run python eval.py \
  --checkpoint runs/idea_x_same_tokens/checkpoints/latest.pt \
  --tasks validation_loss,hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq \
  --data data/processed \
  --eval-data eval_data

uv run python run_lm_eval.py \
  --checkpoint runs/base_d12/checkpoints/latest.pt \
  --tokenizer data/processed/tokenizer.json \
  --output runs/base_d12/eval/lm_eval_results.json

uv run python run_lm_eval.py \
  --checkpoint runs/idea_x_same_tokens/checkpoints/latest.pt \
  --tokenizer data/processed/tokenizer.json \
  --output runs/idea_x_same_tokens/eval/lm_eval_results.json
```

For multi-seed claims, run the same commands with explicit `--seed` values and
keep one run directory per seed. `repro.py compare` is intentionally only an
audit: it checks whether manifests are compatible, but it does not decide which
metric matters or aggregate results for you.

When resuming a paper run, `train.py --resume` fails if the checkpoint manifest
does not match the requested data, seed, precision, kernels, optimizer grouping,
schedule, or comparison-critical config. Use `--allow-resume-mismatch` only for
intentional non-paper debugging.

## Files

- `config.py`: depth-derived model and training config
- `model.py`: editable model components
- `optim.py`: Muon plus AdamW grouping and schedules
- `tokenizer.py`: tokenizer training/loading/encoding boundary
- `prepare_data.py`: raw text/jsonl to token memmaps
- `data.py`: static-shape packed-token memmap batches
- `kernels.py`: bf16 precision policy, FlashAttention 2, standard Liger linear CE, compile, and backend resolution
- `train.py`: pretraining loop, logging, validation, checkpointing
- `prepare_eval.py`, `eval_tasks.py`, `eval.py`: local inspectable eval suite
- `repro.py`: seeds, hashes, environment, manifests, comparison warnings
- `tests/`: pytest smoke tests

## Offline Operation

The training path is offline after dependencies and data are present. It reads
local raw files during `prepare_data.py`, local memmaps and tokenizer manifests
during `train.py`, and local checkpoints during resume/eval. MLflow tracking is
local by default and records config, manifest fields, metrics, JSONL logs, final
metadata, and the latest checkpoint into `runs/mlruns`.

The commands that can require internet are explicit data acquisition paths:
`uv run python prepare_eval.py --source hf ...` downloads validation splits through
Hugging Face `datasets`, and `run_lm_eval.py` may need cached or downloadable
benchmark data from the external harness. For sealed environments, prepare
`data/processed` and `eval_data` ahead of time, then copy those directories and
the `uv` wheel/cache dependencies into the runtime environment.

## Tests

```bash
uv run pytest
```

On a non-CUDA machine, run the portable smoke tests only:

```bash
uv run pytest -m "not cuda"
```

Tests use explicit CPU/Torch smoke paths and are not a training fallback.
