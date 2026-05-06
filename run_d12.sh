#!/usr/bin/env bash
set -euo pipefail

DEPTH=12
RUN_NAME="${RUN_NAME:-d12_fineweb_tpp20}"
FINEWEB_DATASET="${FINEWEB_DATASET:-HuggingFaceFW/fineweb}"
FINEWEB_CONFIG="${FINEWEB_CONFIG:-sample-10BT}"
FINEWEB_SPLIT="${FINEWEB_SPLIT:-train}"
FINEWEB_DOCS="${FINEWEB_DOCS:-7000000}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
PREPARE_BATCH_SIZE="${PREPARE_BATCH_SIZE:-2048}"
VAL_FRACTION="${VAL_FRACTION:-0.0909090909}"
DATA_DIR="${DATA_DIR:-data/processed/fineweb_${FINEWEB_CONFIG}_${FINEWEB_DOCS}_d${DEPTH}}"
RAW_FILE="${RAW_FILE:-data/raw/fineweb_${FINEWEB_CONFIG}_${FINEWEB_DOCS}.jsonl}"
EVAL_DATA="${EVAL_DATA:-eval_data}"
TASKS="${TASKS:-hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,validation_loss}"
EVAL_LIMIT_PER_TASK="${EVAL_LIMIT_PER_TASK:-128}"
export RAW_FILE FINEWEB_DOCS FINEWEB_DATASET FINEWEB_CONFIG FINEWEB_SPLIT

if ! command -v uv >/dev/null 2>&1; then
  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip uv
fi

uv venv
uv sync --locked

mkdir -p "$(dirname "$RAW_FILE")"
if [[ ! -s "$RAW_FILE" ]]; then
  uv run python -c 'exec("import json, os\nfrom datasets import load_dataset\nfrom tqdm.auto import tqdm\nout = os.environ[\"RAW_FILE\"]\nlimit = int(os.environ[\"FINEWEB_DOCS\"])\nds = load_dataset(os.environ[\"FINEWEB_DATASET\"], name=os.environ[\"FINEWEB_CONFIG\"], split=os.environ[\"FINEWEB_SPLIT\"], streaming=True)\nn = 0\nprogress = tqdm(total=limit if limit > 0 else None, unit=\"docs\", desc=\"Downloading FineWeb\")\nwith open(out, \"w\", encoding=\"utf-8\") as f:\n    for row in ds:\n        if limit > 0 and n >= limit:\n            break\n        text = row.get(\"text\")\n        if text:\n            f.write(json.dumps({\"text\": text}, ensure_ascii=False) + \"\\n\")\n            n += 1\n            progress.update(1)\nprogress.close()\nprint(json.dumps({\"output\": out, \"documents\": n}))")' || {
    if [[ -s "$RAW_FILE" ]]; then
      echo "FineWeb download process exited nonzero after writing $RAW_FILE; continuing with the completed file."
    else
      exit 1
    fi
  }
fi

if [[ ! -s "$DATA_DIR/manifest.json" ]]; then
  uv run python prepare_data.py all \
    --input "$RAW_FILE" \
    --vocab-size "$VOCAB_SIZE" \
    --output "$DATA_DIR" \
    --val-fraction "$VAL_FRACTION" \
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
