#!/usr/bin/env bash
#SBATCH --job-name=hackable-lm-climbmix
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/3147347/hackable-lm/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/3147347/hackable-lm/slurm_logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/hackable-lm}"
cd "$REPO_DIR"

DEPTH="${1:-${DEPTH:-12}}"
case "$DEPTH" in
  6)
    DEFAULT_CLIMBMIX_DOCS=2500000
    ;;
  12)
    DEFAULT_CLIMBMIX_DOCS=12000000
    ;;
  18)
    DEFAULT_CLIMBMIX_DOCS=36000000
    ;;
  *)
    DEFAULT_CLIMBMIX_DOCS=12000000
    ;;
esac

CLIMBMIX_DATASET="${CLIMBMIX_DATASET:-karpathy/climbmix-400b-shuffle}"
CLIMBMIX_CONFIG="${CLIMBMIX_CONFIG:-}"
CLIMBMIX_DATASET_SLUG="${CLIMBMIX_DATASET_SLUG:-climbmix_400b_shuffle}"
RUN_NAME="${RUN_NAME:-d${DEPTH}_${CLIMBMIX_DATASET_SLUG}_tpp20_fa3}"
CLIMBMIX_SPLIT="${CLIMBMIX_SPLIT:-train}"
CLIMBMIX_DOCS="${CLIMBMIX_DOCS:-$DEFAULT_CLIMBMIX_DOCS}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
MIN_FREQUENCY="${MIN_FREQUENCY:-2}"
PREPARE_BATCH_SIZE="${PREPARE_BATCH_SIZE:-2048}"
JSONL_TEXT_FIELD="${JSONL_TEXT_FIELD:-text}"
VAL_FRACTION="${VAL_FRACTION:-0.0909090909}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER/hackable-lm}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH_ROOT/data}"
RUNS_DIR="${RUNS_DIR:-$SCRATCH_ROOT/runs}"
FLASH_ATTN_REPO_DIR="${FLASH_ATTN_REPO_DIR:-$SCRATCH_ROOT/src/flash-attention}"
DATA_DIR="${DATA_DIR:-$DATA_ROOT/processed/${CLIMBMIX_DATASET_SLUG}_${CLIMBMIX_DOCS}_d${DEPTH}}"
RAW_FILE="${RAW_FILE:-$DATA_ROOT/raw/${CLIMBMIX_DATASET_SLUG}_${CLIMBMIX_DOCS}.jsonl}"

export RAW_FILE CLIMBMIX_DOCS CLIMBMIX_DATASET CLIMBMIX_DATASET_SLUG CLIMBMIX_CONFIG CLIMBMIX_SPLIT
export DATA_DIR VOCAB_SIZE MIN_FREQUENCY PREPARE_BATCH_SIZE JSONL_TEXT_FIELD VAL_FRACTION RUNS_DIR
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH_ROOT/triton-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_ROOT/uv-cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

mkdir -p "$DATA_ROOT/raw" "$DATA_ROOT/processed" "$RUNS_DIR" "$SCRATCH_ROOT/slurm_logs" "$(dirname "$FLASH_ATTN_REPO_DIR")"

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load cuda/12.8
fi
if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
fi

if ! command -v uv >/dev/null 2>&1; then
  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip uv
fi

echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Scratch root: $SCRATCH_ROOT"
echo "Dataset: $CLIMBMIX_DATASET"
echo "Documents: $CLIMBMIX_DOCS"
echo "Raw file: $RAW_FILE"
echo "Prepared data: $DATA_DIR"
echo "Runs dir: $RUNS_DIR"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

PATH="$HOME/.cargo/bin:$PATH" cargo build --release --manifest-path hackablebpe/Cargo.toml --bin hackablebpe_train
PATH="$HOME/.cargo/bin:$PATH" cargo build --release --manifest-path hackablebpe/Cargo.toml --lib --features extension-module
uv sync --locked --no-install-package flash-attn
uv pip install --python .venv/bin/python setuptools wheel packaging ninja

uv run --no-sync python - <<'PY'
from tokenizer import TOKENIZER_BACKEND, TOKENIZER_FORMAT

print(f"Tokenizer backend: {TOKENIZER_BACKEND}")
print(f"Tokenizer format: {TOKENIZER_FORMAT}")
PY

if ! uv run --no-sync python - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("flash_attn_interface") is not None else 1)
PY
then
  if [[ ! -d "$FLASH_ATTN_REPO_DIR/.git" ]]; then
    git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git "$FLASH_ATTN_REPO_DIR"
  fi
  uv pip install --python .venv/bin/python --no-build-isolation "$FLASH_ATTN_REPO_DIR/hopper"
fi

uv run --no-sync python - <<'PY'
import flash_attn_interface

print(f"FlashAttention 3 interface: {flash_attn_interface.__file__}")
PY

raw_complete=0
if [[ -s "$RAW_FILE" ]]; then
  if uv run --no-sync python - <<'PY'
import os
import sys
from pathlib import Path

path = Path(os.environ["RAW_FILE"])
limit = int(os.environ["CLIMBMIX_DOCS"])
if not path.is_file() or path.stat().st_size == 0:
    sys.exit(1)
if limit <= 0:
    print(f"Using existing {os.environ['CLIMBMIX_DATASET']} file {path}")
    sys.exit(0)

count = 0
with path.open("rb") as f:
    for count, _ in enumerate(f, 1):
        if count >= limit:
            break
if count >= limit:
    print(f"Using existing {os.environ['CLIMBMIX_DATASET']} file {path} with at least {limit} documents")
    sys.exit(0)
print(
    f"Existing {os.environ['CLIMBMIX_DATASET']} file {path} has {count} documents; expected {limit}. "
    "Re-downloading.",
    file=sys.stderr,
)
sys.exit(1)
PY
  then
    raw_complete=1
  fi
fi

if [[ "$raw_complete" -ne 1 ]]; then
  uv run --no-sync python - <<'PY' || {
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm

out = os.environ["RAW_FILE"]
tmp = f"{out}.tmp"
limit = int(os.environ["CLIMBMIX_DOCS"])
dataset_kwargs = {
    "path": os.environ["CLIMBMIX_DATASET"],
    "split": os.environ["CLIMBMIX_SPLIT"],
    "streaming": True,
}
config = os.environ.get("CLIMBMIX_CONFIG")
if config:
    dataset_kwargs["name"] = config
ds = load_dataset(**dataset_kwargs)
n = 0
progress = tqdm(total=limit if limit > 0 else None, unit="docs", desc=f"Downloading {os.environ['CLIMBMIX_DATASET']}")
with open(tmp, "w", encoding="utf-8") as f:
    for row in ds:
        if limit > 0 and n >= limit:
            break
        text = row.get("text")
        if text:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1
            progress.update(1)
progress.close()
if limit > 0 and n < limit:
    raise RuntimeError(f"downloaded {n} documents, expected {limit}")
Path(tmp).replace(out)
print(json.dumps({"output": out, "documents": n}))
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
PY
    if uv run --no-sync python - <<'PY'
import os
import sys
from pathlib import Path

path = Path(os.environ["RAW_FILE"])
limit = int(os.environ["CLIMBMIX_DOCS"])
if not path.is_file() or path.stat().st_size == 0:
    sys.exit(1)
if limit <= 0:
    sys.exit(0)
count = 0
with path.open("rb") as f:
    for count, _ in enumerate(f, 1):
        if count >= limit:
            break
sys.exit(0 if count >= limit else 1)
PY
    then
      echo "$CLIMBMIX_DATASET download process exited nonzero after completing $RAW_FILE; continuing with the completed file."
    else
      exit 1
    fi
  }
fi

if ! uv run --no-sync python - <<'PY'
import json
import math
import os
import sys
from pathlib import Path
from tokenizer import TOKENIZER_BACKEND, TOKENIZER_FORMAT, tokenizer_impl_hash

manifest_path = Path(os.environ["DATA_DIR"]) / "manifest.json"
if not manifest_path.is_file():
    print(f"Preparing data: missing {manifest_path}", file=sys.stderr)
    sys.exit(1)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {
    "tokenizer_backend": TOKENIZER_BACKEND,
    "tokenizer_format": TOKENIZER_FORMAT,
    "tokenizer_impl_hash": tokenizer_impl_hash(),
    "tokenizer_training_split": "train",
    "requested_vocab_size": int(os.environ["VOCAB_SIZE"]),
    "min_frequency": int(os.environ["MIN_FREQUENCY"]),
    "tokenize_batch_size": int(os.environ["PREPARE_BATCH_SIZE"]),
    "jsonl_text_field": os.environ["JSONL_TEXT_FIELD"],
}
mismatches = [
    f"{key}: expected {value!r}, found {manifest.get(key)!r}"
    for key, value in expected.items()
    if manifest.get(key) != value
]

raw_file = str(Path(os.environ["RAW_FILE"]).resolve())
if manifest.get("raw_input_paths") != [raw_file]:
    mismatches.append(f"raw_input_paths: expected {[raw_file]!r}, found {manifest.get('raw_input_paths')!r}")
expected_sizes = {raw_file: Path(raw_file).stat().st_size}
if manifest.get("raw_input_file_sizes") != expected_sizes:
    mismatches.append(
        f"raw_input_file_sizes: expected {expected_sizes!r}, found {manifest.get('raw_input_file_sizes')!r}"
    )

val_fraction = float(os.environ["VAL_FRACTION"])
if not math.isclose(float(manifest.get("val_fraction", -1.0)), val_fraction, rel_tol=0.0, abs_tol=1e-12):
    mismatches.append(f"val_fraction: expected {val_fraction!r}, found {manifest.get('val_fraction')!r}")

for name in (
    "tokenizer.json",
    "train.jsonl",
    "val.jsonl",
    "train.bin",
    "val.bin",
    "train_offsets.npy",
    "val_offsets.npy",
):
    if not (manifest_path.parent / name).is_file():
        mismatches.append(f"missing {name}")

if mismatches:
    print("Preparing data: existing manifest is incompatible:", file=sys.stderr)
    for mismatch in mismatches:
        print(f"  - {mismatch}", file=sys.stderr)
    sys.exit(1)
PY
then
  echo "Preparing data in $DATA_DIR with tokenizer training over the train split only."
  uv run --no-sync python prepare_data.py all \
    --input "$RAW_FILE" \
    --vocab-size "$VOCAB_SIZE" \
    --output "$DATA_DIR" \
    --jsonl-text-field "$JSONL_TEXT_FIELD" \
    --val-fraction "$VAL_FRACTION" \
    --min-frequency "$MIN_FREQUENCY" \
    --batch-size "$PREPARE_BATCH_SIZE"
fi

train_args=(
  --depth "$DEPTH"
  --target-param-data-ratio 20
  --data "$DATA_DIR"
  --run-name "$RUN_NAME"
  --runs-dir "$RUNS_DIR"
  --attention-backend flash_attn_3
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
  --mlflow-experiment "${CLIMBMIX_DATASET_SLUG}-d${DEPTH}-validation"
)

uv run --no-sync python train.py "${train_args[@]}"

echo "MLflow: uv run --no-sync mlflow ui --backend-store-uri file://$RUNS_DIR/mlruns"
