from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from config import config_from_dict
from data import MemmapDataLoader
from eval_tasks import evaluate_task, load_jsonl
from kernels import apply_precision_policy
from model import LanguageModel
from repro import hash_file, load_trusted_checkpoint
from tokenizer import load_tokenizer


def infer_bytes_per_token(data_manifest: dict) -> float | None:
    val_bytes = data_manifest.get("val_text_bytes")
    val_tokens = data_manifest.get("val_tokens")
    if val_bytes and val_tokens:
        return val_bytes / val_tokens
    sizes = data_manifest.get("raw_input_file_sizes") or {}
    tokens = data_manifest.get("train_tokens", 0) + data_manifest.get("val_tokens", 0)
    raw_bytes = sum(sizes.values())
    if not tokens or not raw_bytes:
        return None
    return raw_bytes / tokens


def estimate_bits_per_byte(loss_nats: float, data_manifest: dict) -> float | None:
    bytes_per_token = infer_bytes_per_token(data_manifest)
    if bytes_per_token is None or bytes_per_token <= 0:
        return None
    return loss_nats / math.log(2.0) / bytes_per_token


def load_eval_manifest(eval_data: str | Path) -> dict:
    path = Path(eval_data) / "eval_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def task_file_provenance(eval_data: str | Path, tasks: list[str], eval_manifest: dict) -> dict:
    out = {}
    manifest_tasks = eval_manifest.get("tasks", {}) if isinstance(eval_manifest, dict) else {}
    for task in tasks:
        if task == "validation_loss":
            continue
        path = Path(eval_data) / f"{task}.jsonl"
        entry = dict(manifest_tasks.get(task, {}))
        if path.exists():
            entry.setdefault("file", str(path))
            entry["sha256"] = hash_file(path)
            entry["examples"] = sum(1 for _ in path.open("r", encoding="utf-8"))
        out[task] = entry
    return out


@torch.no_grad()
def validation_loss(model, loader: MemmapDataLoader, device: torch.device, batches: int, batch_size: int) -> float:
    model.eval()
    vals = []
    for _ in range(batches):
        x, y = loader.get_batch("val", batch_size, device)
        _, loss = model(x, y)
        vals.append(float(loss.item()))
    return sum(vals) / len(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a pretrained checkpoint on local explicit tasks.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", default="hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq")
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--eval-data", default="eval_data")
    parser.add_argument("--data", help="processed data dir for validation_loss task")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    device = torch.device(args.device)
    ckpt = load_trusted_checkpoint(args.checkpoint, map_location="cpu")
    config = config_from_dict(ckpt["config"])
    model = LanguageModel(config.model).to(device)
    model = apply_precision_policy(model, config.precision)
    model.load_state_dict(ckpt["model"])
    model.eval()
    manifest = ckpt.get("run_manifest", {})
    tokenizer_path = Path(manifest.get("data", {}).get("data_dir", args.data or ".")).joinpath("tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path)
    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    eval_manifest = load_eval_manifest(args.eval_data)
    results = []
    for task in task_names:
        if task == "validation_loss":
            if not args.data:
                raise ValueError("--data is required for validation_loss")
            loader = MemmapDataLoader(args.data, config.sequence_len, seed=args.seed)
            value = validation_loss(model, loader, device, batches=16, batch_size=min(4, config.device_batch_size))
            data_manifest = loader.manifest
            details = {}
            val_bpb = estimate_bits_per_byte(value, data_manifest)
            if val_bpb is not None:
                details["bits_per_byte"] = val_bpb
                details["bytes_per_token"] = infer_bytes_per_token(data_manifest)
            results.append({"task": task, "metric": "loss", "value": value, "n": 16, "details": details})
            continue
        path = Path(args.eval_data) / f"{task}.jsonl"
        examples = load_jsonl(path)
        res = evaluate_task(task, model, tokenizer, examples, device, args.limit_per_task, args.seed)
        results.append(res.__dict__)
    run_dir = Path(args.checkpoint).resolve().parents[1]
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_tokens_seen": ckpt.get("tokens_seen"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "seed": args.seed,
        "limit_per_task": args.limit_per_task,
        "tasks": task_names,
        "task_provenance": task_file_provenance(args.eval_data, task_names, eval_manifest),
        "eval_manifest": eval_manifest,
        "run_manifest_summary": {
            "run_id": manifest.get("run_id"),
            "label": manifest.get("label"),
            "seed": manifest.get("seed"),
            "comparison": manifest.get("comparison"),
            "data": {
                "tokenizer_hash": manifest.get("data", {}).get("manifest", {}).get("tokenizer_hash"),
                "raw_input_sha256": manifest.get("data", {}).get("manifest", {}).get("raw_input_sha256"),
                "split_seed": manifest.get("data", {}).get("manifest", {}).get("split_seed"),
            },
            "tokenizer": manifest.get("tokenizer"),
            "config": {
                "depth": manifest.get("config", {}).get("depth"),
                "sequence_len": manifest.get("config", {}).get("sequence_len"),
                "global_batch_tokens": manifest.get("config", {}).get("global_batch_tokens"),
                "precision": manifest.get("config", {}).get("precision"),
                "scaling_policy": manifest.get("config", {}).get("scaling_policy"),
            },
        },
        "results": results,
    }
    out_path = out_dir / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
