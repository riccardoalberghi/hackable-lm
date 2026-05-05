# Minimal LM Research Repo Spec

Build a minimal, hackable language-model pretraining repo for fast architecture
and training experiments on one NVIDIA L40 GPU.

This repo is intended to be a base point for coding agents: a future agent
should be able to implement one model or training idea, run pretraining, and get
a useful signal with minimal unrelated infrastructure.

The goal is to implement a modern small-LM baseline with a clean experimental
surface. Scaling rules should be simple, local, and easy to reason about rather
than copied from a larger external training recipe.

## Scope

Implement only:

- model code
- optimizer code
- tokenizer and data preprocessing
- training loop
- run reproducibility and provenance
- simple but useful evaluation suite
- small tests

Do not build:

- experiment harness
- dashboard
- sweep system
- trainer framework
- callback framework
- plugin system
- mandatory distributed training
- instruction-tuning or chat harness

## Primary Goals

1. Easy for future coding agents to modify architecture and training ideas.
2. Strong modern small-LM baseline, suitable for ablations.
3. High MFU on NVIDIA L40; A100/H100 are secondary targets.
4. Fast path first: CUDA + bf16 + Muon + FlashAttention 2 + Liger kernels + `torch.compile`.
5. One main scale knob: `--depth`.
6. Simple depth-derived scaling policy, without copying an external recipe.
7. Project-native preprocessing from raw text to token memmaps.
8. Validation loss as the primary fast signal, plus small pretrained-LM evals.
9. Maximum practical reproducibility for fair baseline-vs-candidate comparison,
   excluding unavoidable GPU/kernel nondeterminism.
10. Liger fused kernels enabled by default where available.
11. A first-time user should be able to preprocess, train, and evaluate with a
    small number of obvious commands.
12. Paper-seed readiness: the repo should be a credible starting point for
    paper-level LM research. That means a strong baseline, fair comparisons,
    real evals, clean intervention points, and enough provenance to reproduce
    claims. It does not mean building a sweep system, plotting/reporting stack,
    or paper-specific diagnostics into the base repo.

## Fast Path Only

The normal runtime target is:

```text
device: CUDA
precision: bf16
optimizer: muon, with AdamW groups for parameters where Muon is inappropriate
norm kernels: torch.compile RMSNorm
lm-head + cross entropy: compile-visible Triton wrapper around Liger CE
MLP nonlinear kernels: torch.compile SwiGLU
compile: true
```

Do not provide first-class alternate-precision/AdamW/Torch-kernel training
flows. Small CPU or Torch-only unit tests are allowed for correctness, but they
are not a training fallback and should not appear as a golden path.

If a fast-path dependency is missing, fail clearly with an actionable install
message. Do not silently fall back to slower kernels or alternate optimizers.

## Repo Structure

Use a flat structure:

```text
model.py
train.py
data.py
tokenizer.py
prepare_data.py
optim.py
config.py
kernels.py
repro.py
eval.py
eval_tasks.py
prepare_eval.py
tests/
AGENTS.md
pyproject.toml
uv.lock
README.md
```

Avoid deep packages. Avoid registries. Avoid framework-style abstractions.

## Launch Ergonomics

The repo must have a compact golden path. A new user should not need to
understand every subsystem before getting a real run.

The first successful flow should be:

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

uv run python eval.py \
  --checkpoint runs/d12/checkpoints/latest.pt \
  --tasks lambada_openai,hellaswag \
  --limit-per-task 128
```

Normal model scaling should be:

```bash
uv run python train.py --depth 20 --data data/processed --run-name d20
```

Advanced flags should be optional and limited to:

```text
depth
data path
run name
seed
sequence length
target param:data ratio
target scaling params
target tokens
target bytes
global batch tokens
device microbatch size
number of iterations
target FLOPs
target seconds with measured tokens/sec
comparison mode
match-run manifest path
eval tasks
checkpoint path
small expert config overrides
```

Keep command-line parsing simple: `argparse` is enough. Do not introduce Hydra,
Click command trees, YAML config stacks, or launcher frameworks.

## Depth-Derived Configs

The main scale knob is:

```text
depth
```

Default scaling policy:

```text
scaling_policy = "depth_simple"
```

This policy should be explicitly documented and implemented in `config.py`.
The formulas should be simple enough to inspect directly.

Default constants:

```text
head_dim = 128
layers_per_head = 2
sequence_len = 2048
attention_window = 512
attention_full_every = 4
target_param_data_ratio = 60
reference_depth = 12
reference_batch_tokens = 2**19
embedding_lr_ref = 0.1
unembedding_lr_ref = 0.01
matrix_lr_ref = 0.02
scalar_lr_ref = 0.1
weight_decay = 0.1
lr_scheduler = "wsd"
warmup_ratio = 0.05
warmdown_ratio = 0.3
final_lr_frac = 0.1
```

Architecture dimensions:

```python
n_head = ceil(depth / layers_per_head)
n_embd = head_dim * n_head
n_layer = depth
mlp_hidden = align(ceil(8 * n_embd / 3), 256)
```

Scaling parameter count:

```python
scaling_params = transformer_matrices + lm_head
```

Do not include token embeddings, RoPE buffers, optimizer-only state, or tiny
scalar/vector parameters in `scaling_params`.

Training horizon:

```python
target_tokens = target_param_data_ratio * scaling_params
```

Batch size:

```python
predicted_batch_tokens = reference_batch_tokens * sqrt(depth / reference_depth)
grad_accum_steps = global_batch_tokens // (device_batch_size * sequence_len)
```

If divisibility is imperfect, round accumulation up and record the actual global
batch tokens in the manifest.

Learning rate scaling:

```python
batch_lr_scale = sqrt(global_batch_tokens / reference_batch_tokens)

embedding_lr = embedding_lr_ref * batch_lr_scale
unembedding_lr = unembedding_lr_ref * batch_lr_scale
matrix_lr = matrix_lr_ref * batch_lr_scale
scalar_lr = scalar_lr_ref * batch_lr_scale
```

LR schedule:

```python
lr_scheduler = "wsd"
warm up linearly, hold stable, then cosine decay to final_lr_frac
```

Weight decay:

```python
resolved_weight_decay = weight_decay
```

Iteration count:

```python
if num_iterations is provided:
    steps = num_iterations
elif target_flops is provided:
    steps = round(target_flops / (flops_per_token * global_batch_tokens))
else:
    steps = target_tokens // global_batch_tokens
```

The resolved config must record all derived values:

```text
depth
n_layer
n_embd
n_head
sequence length
scaling params
target tokens
reference depth
reference batch tokens
global batch tokens
device batch size
gradient accumulation steps
batch LR scale
all LR groups
weight decay
num iterations
estimated FLOPs/token
```

## Complexity Budget

The implementation should feel like a compact research repo, not an internal
platform.

Prefer:

- one obvious function per script: `main()`
- dataclass configs
- explicit backend switches
- JSON/JSONL manifests and logs
- short, readable modules
- local component switch points in `model.py`, not registries

Avoid:

- registries
- plugin architectures
- nested packages
- callback systems
- YAML config stacks
- launcher scripts that hide what is happening
- mandatory cloud/logging services

If a feature makes the golden path harder to understand, it must be moved behind
an optional flag, documented in one short README section, or deferred.

## Research Intervention Surface

The repo should make candidate ideas easy to isolate. A future agent should be
able to modify one subsystem, run the same comparison protocol, and know whether
the intended subsystem was the only meaningful change.

Common intervention points must remain local and hackable:

```text
tokenizer training and token segmentation
raw data filtering / normalization / splitting
packed-token data sampling
model residual path
attention block
MLP block
normalization
positional encoding
initialization
loss/logit handling
optimizer grouping
optimizer update rule
LR schedule
momentum and weight decay constants
precision conversion policy
kernel backend choice
validation and eval task definitions
```

Hackable does not mean abstract. Prefer small functions, dataclasses, and
explicit conditionals over registries or plugin systems.

The README must document a minimal comparison protocol:

```text
baseline and candidate use the same depth
same data files and hashes
same tokenizer hash
same train/val split
same sequence length
same global batch tokens
same token budget or number of tokens seen
same optimizer grouping unless optimizer is the intervention
same LR/momentum/weight-decay schedule unless schedule is the intervention
same precision and kernels unless those are the intervention
same seed, unless running an explicit multi-seed comparison
same eval datasets and hashes
```

Every run manifest must make it possible to check this protocol.

The README must also define the baseline contract:

```text
baseline architecture
baseline tokenizer and preprocessing assumptions
baseline optimizer grouping
baseline scaling policy
primary metric
secondary metrics
what fields must match for a valid baseline-vs-candidate comparison
```

Candidate changes should be expressible as a small code diff plus a manifest
label. The base repo should not include a sweep runner, but it should not make
paper-level run matrices hard to launch manually or from a short external shell
script.

Temporary instrumentation should be easy to add without changing the training
loop architecture. Examples include hidden-state statistics, gradient norms,
attention/residual weights, data-domain counters, tokenizer statistics, or
optimizer update norms. These diagnostics are paper-specific and should not be
required in the base repo.

## Architecture

Implement one coherent modern decoder-only causal LM baseline with explicit,
inspectable interactions between components.

The baseline architecture must be stated explicitly in README and in comments
near `ModelConfig`. A reader should be able to identify which architectural
choice changed in a candidate run without reverse-engineering the whole model.

Recommended baseline features:

- token embeddings
- untied LM head by default
- RoPE positional encoding
- pre-norm Transformer blocks
- non-param RMSNorm or clearly documented param RMSNorm
- QK norm
- 512-token sliding-window causal self-attention through the selected fast attention backend,
  with every fourth layer and the final layer using full causal attention
- GQA support if it stays simple
- fused QKV projection
- SwiGLU MLP
- fused SwiGLU gate/up projection
- `bias=False` linear layers
- tensor-core-friendly dimensions
- initialization documented in `model.py`

Keep architecture code flat and explicit in `model.py`.

Main editable components:

```text
ModelConfig
norm / RMSNorm
RotaryEmbedding / apply_rope
CausalSelfAttention
MLP
Block
LanguageModel
```

The code should make common ablations easy without a registry:

```text
attention variant
MLP variant
normalization variant
positional encoding
residual path
initialization
logit/loss handling
```

Small local config fields are fine. Do not build a general component framework.

For residual-path research such as attention residuals, the residual update
should be easy to replace in `model.py` without rewriting attention, MLP, data,
optimizer, or training code.

## Precision

Default precision:

```python
precision = "bf16"
```

Keep model parameters in bf16 for the CUDA training hot path, without separate
FP32 master weights. Let `torch.compile` handle RMSNorm and SwiGLU, and use a
compile-visible Triton wrapper around Liger's CE kernel for the fused loss path:

```text
LM head plus cross entropy
```

Keep these in BF16/FP32 as appropriate:

```text
RoPE math
attention softmax and masks
small scalar/control math
```

Muon momentum and AdamW moment buffers should be bf16.

If a requested Liger backend is unavailable, fail clearly with an actionable
error.

## Optimizer

Muon is the optimizer mode. AdamW is used inside the optimizer for parameters
where Muon is inappropriate.

Implement `optim.py` with:

```python
def create_optimizer(model, config):
    ...
```

Parameter grouping must be explicit:

- Muon for appropriate matrix weights:
  - fused QKV projection weights
  - attention output projection weights
  - fused MLP gate/up projection weights
  - MLP down projection weights
- Muon must update fused QKV and gate/up projections as row-slice virtual
  matrices so optimizer behavior matches unfused projections.
- AdamW for:
  - token embeddings
  - LM head
  - norms if parameterized
  - biases, if any
  - scalar/vector parameters
  - any parameter where Muon is inappropriate

Use explicit LR groups:

```text
embedding_lr
unembedding_lr
matrix_lr
scalar_lr
```

Support:

- fixed weight decay
- fixed Muon momentum
- gradient clipping
- checkpoint save/resume including optimizer state

The optimizer code should make optimizer research easy without a framework:

```text
parameter grouping is explicit and inspectable
the update rule is localized in optim.py
schedules are localized in train.py or a small schedule helper
group-level hyperparameters are manifested
baseline grouping can be compared against candidate grouping
```

Print or manifest optimizer grouping:

```text
number of Muon parameter tensors
number of AdamW parameter tensors
parameter counts per group
group LR
group weight decay
excluded parameter names/counts if any
```

## Training

`train.py` should support:

- fixed-length packed-token batches
- token memmap files
- gradient accumulation from derived global batch tokens
- Muon optimizer with AdamW subgroups
- WSD LR schedule
- fixed Muon momentum
- fixed weight decay
- gradient clipping
- checkpoint save/resume
- validation loss
- periodic stdout JSON logging
- run manifest creation

Learning rate schedule:

```python
warmup_steps = round(warmup_ratio * num_iterations)
if step < warmup_steps:
    lr_mult = (step + 1) / warmup_steps
elif step < decay_start:
    lr_mult = 1.0
else:
    progress = (step - decay_start) / decay_span
    lr_mult = final_lr_frac + (1 - final_lr_frac) * cosine_decay(progress)
```

Muon momentum:

```python
momentum = 0.95
```

Weight decay:

```python
weight_decay_t = weight_decay
```

Checkpoint state must include:

```text
model
optimizer
scheduler/scaling state
RNG state
config
run manifest
data manifest
tokenizer manifest
step
tokens seen
```

Logging should include:

```text
step
loss
validation loss when run
lr multiplier
matrix LR
Muon momentum
weight decay
tokens/sec
MFU
BF16 MFU
peak memory
precision mode
kernel backends actually used
run id
seed
data hash
code hash
```

No wandb dependency by default. Plain stdout JSONL logging is acceptable.

## Reproducibility And Fair Comparison

Reproducibility is a first-class feature. The repo should make it hard to
accidentally compare a candidate against a baseline with different data,
tokenizer, seed schedule, precision, kernels, optimizer grouping, scaling
policy, or eval versions.

Implement `repro.py` with:

```python
def seed_everything(seed: int) -> None:
    ...

def collect_environment() -> dict:
    ...

def write_run_manifest(run_dir, config, model, optimizer, data_info, tokenizer_info) -> None:
    ...

def hash_file(path) -> str:
    ...

def hash_directory(path, include_globs=None, exclude_globs=None) -> str:
    ...
```

Every run must create a run directory:

```text
runs/<run_id>/
  manifest.json
  config.json
  train_log.jsonl
  checkpoints/
  eval/
```

The manifest must record:

```text
run id
experiment name
baseline/candidate label if provided
scaling policy
full resolved config
command line argv
seed
all RNG seeds
model parameter count
scaling parameter count
optimizer parameter grouping summary
precision mode
kernel backends requested
kernel backends actually used
torch.compile settings
attention backend
attention window
full attention interval
global batch tokens
microbatch size
gradient accumulation steps
LR schedule
token budget
data files, sizes, and hashes
tokenizer files and hashes
vocab size
train/val split seed
eval dataset versions and hashes when eval is run
git commit if available
git diff hash if dirty
Python version
PyTorch version
Liger Kernel version
CUDA version
driver version
GPU name
GPU memory
hostname
important environment variables
```

Important environment variables include:

```text
CUBLAS_WORKSPACE_CONFIG
CUDA_VISIBLE_DEVICES
PYTHONHASHSEED
NCCL_* if DDP is used
```

Seed all relevant sources:

```text
Python random
NumPy
torch CPU RNG
torch CUDA RNG
data sampling RNG
data split RNG
eval subsampling RNG
```

Do not promise bitwise-identical GPU results. Runs should be seeded and
auditable, but the training path prioritizes fast CUDA kernels.

Fair comparison rules:

- A candidate run should be compared against a baseline with the same:
  - data files and data hashes
  - tokenizer hash
  - train/val split
  - token budget or number of tokens seen
  - sequence length
  - batch token schedule
  - LR schedule
  - optimizer grouping
  - precision mode
  - kernel backends
  - scaling policy
  - seed, unless explicitly running a multi-seed comparison
  - eval dataset versions
- If a run differs on any of those fields, `eval.py` or a small comparison
  helper should print a warning before comparing metrics.
- Do not silently resume from a checkpoint with mismatched data, tokenizer,
  model config, optimizer grouping, precision, or kernel backend.
- Checkpoint resume should restore RNG state and continue token accounting.
- Validation/eval subsets created with `--limit` must be deterministic and
  manifested.
- If a run changes tokenizer, data preparation, optimizer, schedule, precision,
  kernels, or eval task versions, the comparison helper should identify that
  change explicitly rather than hiding it under a generic incompatibility.

## Data Preprocessing

Data preprocessing must live inside the project.

Implement:

```text
tokenizer.py
prepare_data.py
```

Use a fast tokenizer backend suitable for the fast path, wrapped behind
`tokenizer.py` so the rest of the repo is insulated from API details.

Tokenizer and data-preparation research should be possible without touching
`train.py` or `model.py`. Keep these boundaries explicit:

```text
tokenizer.py owns tokenizer training/loading/encoding/decoding
prepare_data.py owns raw text reading, filtering, splitting, tokenization, and manifests
data.py owns runtime token sampling from processed artifacts
train.py consumes only processed token artifacts and manifests
```

Candidate tokenizer or preprocessing changes must be recorded in the processed
data manifest and in the training run manifest.

`prepare_data.py` should support:

1. training a tokenizer from raw text
2. encoding raw text into token memmaps
3. writing metadata needed by `train.py`

Expected raw formats:

```text
.txt
.jsonl with configurable text field, default "text"
```

Golden command:

```bash
uv run python prepare_data.py all \
  --input data/raw/*.txt \
  --vocab-size 32768 \
  --output data/processed
```

Tokenized outputs:

```text
data/processed/train.bin
data/processed/val.bin
data/processed/manifest.json
data/processed/tokenizer.json
```

Use `uint16` if `vocab_size <= 65535`, otherwise `uint32`.

Use deterministic splitting with a seed. Use packed contiguous token streams.
Do not tokenize inside the training loop.

The data manifest must record:

```text
raw input paths
raw input file sizes
raw input SHA256 hashes
tokenizer hash
vocab size
special tokens
split seed
val fraction
number of train tokens
number of val tokens
dtype
prepare_data.py code hash if available
```

## Runtime Data Loader

`data.py` should implement a simple memmap dataloader.

For sampled position `i`:

```python
x = tokens[i : i + block_size]
y = tokens[i + 1 : i + block_size + 1]
```

Batches must have static shape:

```text
[batch_size, block_size]
```

No padding masks in the training hot path.

Sampling policy changes should be localized to `data.py` and manifested. This
includes random span sampling, sequential sampling, domain-balanced sampling, or
multi-memmap mixing. The default should remain simple packed-token sampling.

## Kernels

Default performance features:

- `torch.compile` behind a config flag, default true for train
- fast attention backend where available
- bf16 model compute by default
- torch.compile RMSNorm/SwiGLU and standard Liger linear CE by default
- fixed static sequence length
- tensor-core-friendly dimensions

Implement `kernels.py` as the only place where model code chooses fused or
optional kernel backends. `model.py` should not import Liger directly.

Backend resolution rules:

- requested backend is read from config
- compatibility is checked at startup
- actual backend is recorded in the manifest
- unsupported backend combinations fail clearly
- fallback is not allowed by default

Pay special attention to interactions among:

```text
bf16 precision
torch.compile
fast attention
Liger kernels
Muon optimizer
```

If two features are incompatible, fail early and explain the incompatible
combination.

## Liger Kernels

Use Liger kernels through `kernels.py` for operations where this repo has a
direct compatible call site.

The Liger-backed fused linear CE loss should be behind an explicit config value:

```python
loss_backend = "liger"
```

Each Liger-backed operation must include:

- a plain Torch reference implementation for tests only
- a correctness test against the reference
- a backward correctness test when applicable
- a microbenchmark against the Torch reference
- dtype coverage notes
- shape constraints
- layout constraints
- deterministic/nondeterministic notes

## Training Performance Metrics

`train.py` should report realistic training performance metrics from the main
training loop:

```text
tokens/sec
MFU
peak memory
precision mode
kernel backends actually used
```

Use:

```python
L40_BF16_DENSE_PEAK = 181e12
```

MFU estimate:

```python
model_flops_sec = flops_per_token * tokens_per_second
mfu = model_flops_sec / L40_BF16_DENSE_PEAK
bf16_mfu = mfu
```

## Evaluation

Add a simple but serious evaluation suite. It should be separate from `train.py`
and should not become a full external harness.

Add:

```text
eval.py
eval_tasks.py
prepare_eval.py
```

Goals:

- validation loss/perplexity on the pretraining validation memmap
- optional per-domain validation loss if processed data has domain shards
- a small set of common academic benchmarks for base pretrained models
- loglikelihood / continuation scoring
- deterministic and inspectable evaluation

Do not use:

- instruction templates
- chat prompts
- chain-of-thought
- sampling
- generation-based judging
- external API dependencies

Tiny built-in fixtures are allowed only as smoke tests and must be labeled as
such. Real eval signal should come from local JSONL files prepared by
`prepare_eval.py`.

Evaluation must be strong enough to support early paper decisions, while staying
small and inspectable:

```text
validation loss is the primary metric
common pretrained-LM benchmarks are secondary metrics
all eval data is local and hashed
task scoring code is explicit in eval_tasks.py
eval subsets are deterministic
tiny built-ins are never presented as research evidence
```

Eval task additions should be local to `prepare_eval.py` and `eval_tasks.py`.
Do not introduce a broad external eval harness in the base repo.

`prepare_eval.py` must write:

```text
eval_data/eval_manifest.json
```

The eval manifest should record:

```text
task names
task versions
source dataset identifiers
download or conversion time
file hashes
number of examples
license notes if available
```

`eval.py` should write eval outputs under:

```text
runs/<run_id>/eval/
```

## Tests

Keep tests small and useful.

Test categories:

- config derivation from depth
- tokenizer roundtrip / preprocessing manifest
- memmap batch shapes
- model forward shape and loss when fast dependencies are installed
- optimizer grouping
- schedule functions
- manifest compatibility checks
- custom kernel correctness if custom kernels are added

Tests may include tiny pure-Python or Torch references, but those references are
for correctness testing only. They are not training backends.

The basic test command should be:

```bash
uv run pytest
```

On machines without CUDA, use `uv run pytest -m "not cuda"` for portable smoke
tests.

## Implementation Priority

Build in this order:

```text
P0: fast-path runnable repo
  config.py with depth-derived simple scaling policy
  coherent baseline model in model.py
  data.py
  tokenizer.py
  prepare_data.py
  optim.py with Muon plus AdamW subgroups
  train.py with fast-path training loop
  tests/
  README quickstart

P1: fair comparison and speed defaults
  repro.py manifests
  kernels.py with bf16/attention/Liger backend resolution and no silent fallback
  train.py with MFU
  checkpoint resume integrity

P2: useful evaluation
  prepare_eval.py
  eval_tasks.py
  eval.py
  eval manifests
  baseline-vs-candidate compatibility warnings

P3: extensibility
  kernel microbenchmarks
  optional DDP
  optional extra eval tasks
```

P0 must remain runnable on the intended fast-path environment. Later phases
cannot break the quickstart.
