from __future__ import annotations

import argparse
import json
from pathlib import Path

import lm_eval
from lm_eval.utils import handle_non_serializable

from lm_eval_simple_lm import SimpleLMHarness
from repro import load_trusted_checkpoint
from tokenizer import tokenizer_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EleutherAI lm-evaluation-harness on a hackable-lm checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--tasks", default="hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="lm_eval_results.json")
    args = parser.parse_args()

    model = SimpleLMHarness(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
    ckpt = load_trusted_checkpoint(args.checkpoint, map_location="cpu")
    run_manifest = ckpt.get("run_manifest", {})
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=[task.strip() for task in args.tasks.split(",") if task.strip()],
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )
    results["hackable_lm_metadata"] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_tokens_seen": ckpt.get("tokens_seen"),
        "tokenizer": tokenizer_manifest(model.tokenizer_path),
        "device": args.device,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "tasks": [task.strip() for task in args.tasks.split(",") if task.strip()],
        "run_manifest_summary": {
            "run_id": run_manifest.get("run_id"),
            "label": run_manifest.get("label"),
            "seed": run_manifest.get("seed"),
            "comparison": run_manifest.get("comparison"),
            "data": {
                "tokenizer_hash": run_manifest.get("data", {}).get("manifest", {}).get("tokenizer_hash"),
                "raw_input_sha256": run_manifest.get("data", {}).get("manifest", {}).get("raw_input_sha256"),
                "split_seed": run_manifest.get("data", {}).get("manifest", {}).get("split_seed"),
            },
            "config": {
                "depth": run_manifest.get("config", {}).get("depth"),
                "sequence_len": run_manifest.get("config", {}).get("sequence_len"),
                "global_batch_tokens": run_manifest.get("config", {}).get("global_batch_tokens"),
                "precision": run_manifest.get("config", {}).get("precision"),
                "scaling_policy": run_manifest.get("config", {}).get("scaling_policy"),
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, default=handle_non_serializable, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": args.output, "tasks": args.tasks, "limit": args.limit}, sort_keys=True))


if __name__ == "__main__":
    main()
