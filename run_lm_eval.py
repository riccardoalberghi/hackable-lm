from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from repro import collect_environment, hash_file, load_trusted_checkpoint
from tokenizer import tokenizer_manifest


COMMONSENSE_TASKS = [
    "hellaswag",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "openbookqa",
    "boolq",
]
LAMBADA_TASKS = ["lambada_openai"]
MMLU_TASKS = ["mmlu"]


@dataclass(frozen=True)
class EvalSuite:
    name: str
    tasks: list[str]
    num_fewshot: int


SUITES = {
    "commonsense": [EvalSuite("commonsense_0shot", COMMONSENSE_TASKS, 0)],
    "lambada": [EvalSuite("lambada_0shot", LAMBADA_TASKS, 0)],
    "mmlu": [EvalSuite("mmlu_5shot", MMLU_TASKS, 5)],
    "standard": [
        EvalSuite("commonsense_0shot", COMMONSENSE_TASKS, 0),
        EvalSuite("lambada_0shot", LAMBADA_TASKS, 0),
        EvalSuite("mmlu_5shot", MMLU_TASKS, 5),
    ],
}


def parse_task_list(value: str) -> list[str]:
    tasks = [task.strip() for task in value.split(",") if task.strip()]
    if not tasks:
        raise ValueError("task list is empty")
    return tasks


def manifest_output_path(output: Path) -> Path:
    if output.suffix:
        return output.with_name(f"{output.stem}.manifest{output.suffix}")
    return output.with_name(f"{output.name}.manifest.json")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def data_manifest_hash(tokenizer_path: Path) -> str | None:
    manifest_path = tokenizer_path.parent / "manifest.json"
    return hash_file(manifest_path) if manifest_path.exists() else None


def run_suite(model: Any, suite: EvalSuite, limit: int | None) -> dict[str, Any]:
    import lm_eval

    return lm_eval.simple_evaluate(
        model=model,
        tasks=suite.tasks,
        num_fewshot=suite.num_fewshot,
        limit=limit,
    )


def build_metadata(args, ckpt: dict[str, Any], tokenizer_path: Path, suites: list[EvalSuite]) -> dict[str, Any]:
    run_manifest = ckpt.get("run_manifest", {})
    token_info = tokenizer_manifest(tokenizer_path)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluator": "EleutherAI lm-evaluation-harness",
        "lm_eval_version": package_version("lm-eval"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": hash_file(args.checkpoint),
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_tokens_seen": ckpt.get("tokens_seen"),
        "tokenizer": token_info,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "data_manifest_sha256": data_manifest_hash(tokenizer_path),
        "device": args.device,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "limit": args.limit,
        "sample_scope": "full" if args.limit is None else f"limit_{args.limit}",
        "suite": args.suite if args.tasks is None else "custom",
        "suites": [
            {
                "name": suite.name,
                "tasks": suite.tasks,
                "num_fewshot": suite.num_fewshot,
                "limit": args.limit,
                "sample_scope": "full" if args.limit is None else f"limit_{args.limit}",
            }
            for suite in suites
        ],
        "run_manifest_summary": {
            "run_id": run_manifest.get("run_id"),
            "label": run_manifest.get("label"),
            "seed": run_manifest.get("seed"),
            "comparison": run_manifest.get("comparison"),
            "data": {
                "tokenizer_hash": run_manifest.get("data", {}).get("manifest", {}).get("tokenizer_hash"),
                "raw_input_sha256": run_manifest.get("data", {}).get("manifest", {}).get("raw_input_sha256"),
                "split_seed": run_manifest.get("data", {}).get("manifest", {}).get("split_seed"),
                "data_shuffle_seed": run_manifest.get("data", {}).get("data_shuffle_seed"),
            },
            "config": {
                "depth": run_manifest.get("config", {}).get("depth"),
                "sequence_len": run_manifest.get("config", {}).get("sequence_len"),
                "global_batch_tokens": run_manifest.get("config", {}).get("global_batch_tokens"),
                "precision": run_manifest.get("config", {}).get("precision"),
                "scaling_policy": run_manifest.get("config", {}).get("scaling_policy"),
                "train_flops_budget": run_manifest.get("config", {}).get("train_flops_budget"),
                "scheduled_tokens": run_manifest.get("config", {}).get("scheduled_tokens"),
            },
        },
        "environment": collect_environment(Path.cwd()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EleutherAI lm-evaluation-harness on a hackable-lm checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--suite", default="standard", choices=sorted(SUITES), help="standard suites with fixed few-shot settings")
    parser.add_argument("--tasks", help="comma-separated custom task list; overrides --suite")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-fewshot", type=int, help="few-shot count for --tasks custom runs")
    parser.add_argument("--limit", type=int, help="optional sample cap; omitted means full benchmark")
    parser.add_argument("--output", default="lm_eval_results.json")
    args = parser.parse_args()
    if args.tasks is None and args.num_fewshot is not None:
        parser.error("--num-fewshot is only valid with --tasks; built-in suites fix this per benchmark group")
    if args.tasks is not None:
        try:
            suites = [EvalSuite("custom", parse_task_list(args.tasks), args.num_fewshot or 0)]
        except ValueError as exc:
            parser.error(str(exc))
    else:
        suites = SUITES[args.suite]

    from lm_eval.utils import handle_non_serializable
    from lm_eval_hackable_lm import SimpleLMHarness

    model = SimpleLMHarness(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
    ckpt = load_trusted_checkpoint(args.checkpoint, map_location="cpu")
    metadata = build_metadata(args, ckpt, model.tokenizer_path, suites)
    results = {
        "hackable_lm_metadata": metadata,
        "suites": {
            suite.name: run_suite(model, suite, args.limit)
            for suite in suites
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, default=handle_non_serializable, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = manifest_output_path(output)
    manifest_path.write_text(json.dumps(metadata, default=handle_non_serializable, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest": str(manifest_path),
                "suite": metadata["suite"],
                "suites": [suite.name for suite in suites],
                "limit": args.limit,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
