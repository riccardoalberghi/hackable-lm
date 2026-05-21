#!/usr/bin/env bash
set -euo pipefail

DEPTH="${1:-${DEPTH:-12}}"
case "$DEPTH" in
  6)
    DEFAULT_FINEWEB_DOCS=1500000
    ;;
  12)
    DEFAULT_FINEWEB_DOCS=7000000
    ;;
  18)
    DEFAULT_FINEWEB_DOCS=21000000
    ;;
  *)
    DEFAULT_FINEWEB_DOCS=7000000
    ;;
esac

RUN_NAME="${RUN_NAME:-d${DEPTH}_fineweb_tpp20}"
FINEWEB_DATASET="${FINEWEB_DATASET:-HuggingFaceFW/fineweb}"
FINEWEB_CONFIG="${FINEWEB_CONFIG:-sample-10BT}"
FINEWEB_SPLIT="${FINEWEB_SPLIT:-train}"
FINEWEB_DOCS="${FINEWEB_DOCS:-$DEFAULT_FINEWEB_DOCS}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
MIN_FREQUENCY="${MIN_FREQUENCY:-2}"
PREPARE_BATCH_SIZE="${PREPARE_BATCH_SIZE:-2048}"
JSONL_TEXT_FIELD="${JSONL_TEXT_FIELD:-text}"
VAL_FRACTION="${VAL_FRACTION:-0.0909090909}"
DATA_DIR="${DATA_DIR:-data/processed/fineweb_${FINEWEB_CONFIG}_${FINEWEB_DOCS}_d${DEPTH}}"
RAW_FILE="${RAW_FILE:-data/raw/fineweb_${FINEWEB_CONFIG}_${FINEWEB_DOCS}.jsonl}"
EVAL_DATA="${EVAL_DATA:-eval_data}"
TASKS="${TASKS:-hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,validation_loss}"
EVAL_LIMIT_PER_TASK="${EVAL_LIMIT_PER_TASK:-128}"

export RAW_FILE FINEWEB_DOCS FINEWEB_DATASET FINEWEB_CONFIG FINEWEB_SPLIT
export DATA_DIR VOCAB_SIZE MIN_FREQUENCY PREPARE_BATCH_SIZE JSONL_TEXT_FIELD VAL_FRACTION

if ! command -v uv >/dev/null 2>&1; then
  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip uv
fi

make setup

uv run python - <<'PY'
from tokenizer import TOKENIZER_BACKEND, TOKENIZER_FORMAT

print(f"Tokenizer backend: {TOKENIZER_BACKEND}")
print(f"Tokenizer format: {TOKENIZER_FORMAT}")
PY

mkdir -p "$(dirname "$RAW_FILE")"
raw_complete=0
if [[ -s "$RAW_FILE" ]]; then
  if uv run python - <<'PY'
import os
import sys
from pathlib import Path

path = Path(os.environ["RAW_FILE"])
limit = int(os.environ["FINEWEB_DOCS"])
if not path.is_file() or path.stat().st_size == 0:
    sys.exit(1)
if limit <= 0:
    print(f"Using existing FineWeb file {path}")
    sys.exit(0)

count = 0
with path.open("rb") as f:
    for count, _ in enumerate(f, 1):
        if count >= limit:
            break
if count >= limit:
    print(f"Using existing FineWeb file {path} with at least {limit} documents")
    sys.exit(0)
print(f"Existing FineWeb file {path} has {count} documents; expected {limit}. Re-downloading.", file=sys.stderr)
sys.exit(1)
PY
  then
    raw_complete=1
  fi
fi

if [[ "$raw_complete" -ne 1 ]]; then
  uv run python - <<'PY' || {
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm

out = os.environ["RAW_FILE"]
tmp = f"{out}.tmp"
limit = int(os.environ["FINEWEB_DOCS"])
ds = load_dataset(
    os.environ["FINEWEB_DATASET"],
    name=os.environ["FINEWEB_CONFIG"],
    split=os.environ["FINEWEB_SPLIT"],
    streaming=True,
)
n = 0
progress = tqdm(total=limit if limit > 0 else None, unit="docs", desc="Downloading FineWeb")
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
    if uv run python - <<'PY'
import os
import sys
from pathlib import Path

path = Path(os.environ["RAW_FILE"])
limit = int(os.environ["FINEWEB_DOCS"])
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
      echo "FineWeb download process exited nonzero after completing $RAW_FILE; continuing with the completed file."
    else
      exit 1
    fi
  }
fi

if ! uv run python - <<'PY'
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
  uv run python prepare_data.py all \
    --input "$RAW_FILE" \
    --vocab-size "$VOCAB_SIZE" \
    --output "$DATA_DIR" \
    --jsonl-text-field "$JSONL_TEXT_FIELD" \
    --val-fraction "$VAL_FRACTION" \
    --min-frequency "$MIN_FREQUENCY" \
    --batch-size "$PREPARE_BATCH_SIZE"
fi

uv run python train.py \
  --depth "$DEPTH" \
  --target-param-data-ratio 20 \
  --data "$DATA_DIR" \
  --run-name "$RUN_NAME" \
  --mlflow-experiment "fineweb-d${DEPTH}-validation"

uv run python prepare_eval.py \
  --source hf \
  --output "$EVAL_DATA" \
  --limit-per-task "$EVAL_LIMIT_PER_TASK"

uv run python eval.py \
  --checkpoint "runs/$RUN_NAME/checkpoints/latest.pt" \
  --data "$DATA_DIR" \
  --eval-data "$EVAL_DATA" \
  --tasks "$TASKS" \
  --limit-per-task "$EVAL_LIMIT_PER_TASK"

echo "MLflow: uv run mlflow ui --backend-store-uri file://$(pwd)/runs/mlruns"
