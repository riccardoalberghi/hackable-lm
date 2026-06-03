from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from config import ATTENTION_BACKENDS, COMPARISON_MODES, DEFAULTS, MODULE_BACKEND_FIELDS, MODULE_BACKENDS, resolve_config
from data import MemmapDataLoader, load_manifest
from kernels import apply_precision_policy, compile_training_model, mark_compiled_step_begin, resolve_kernel_backends
from model import LanguageModel
from optim import create_optimizer, lr_multiplier
from repro import load_trusted_checkpoint, seed_everything, write_json, write_run_manifest
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


@dataclass(frozen=True)
class EvalSuite:
    name: str
    tasks: list[str]
    num_fewshot: int


STANDARD_EVAL_SUITES = [
    EvalSuite("commonsense_0shot", COMMONSENSE_TASKS, 0),
    EvalSuite("lambada_0shot", LAMBADA_TASKS, 0),
]


def cuda_required() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the training fast path. Use tests.py for CPU correctness checks.")
    return torch.device("cuda")


def gpu_memory_gib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    props = torch.cuda.get_device_properties(device)
    return props.total_memory / 1024**3


def ddp_info() -> dict[str, int | bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {
        "enabled": world_size > 1,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
    }


def setup_ddp(info: dict[str, int | bool]) -> torch.device:
    if info["enabled"]:
        torch.cuda.set_device(int(info["local_rank"]))
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", int(info["local_rank"]))
    return cuda_required()


def cleanup_ddp(info: dict[str, int | bool]) -> None:
    if info["enabled"] and dist.is_initialized():
        dist.destroy_process_group()


def validate(model, loader: MemmapDataLoader, config, device: torch.device, batches: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = loader.get_batch("val", config.device_batch_size, device)
            mark_compiled_step_begin(config.compile)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_run_manifest_arg(path: str) -> dict:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    return json.loads(manifest_path.read_text())


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
    return loss_nats / np.log(2.0) / bytes_per_token


def _flatten_for_mlflow(obj: Any, prefix: str = "") -> dict[str, Any]:
    flat = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_for_mlflow(value, child))
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            child = f"{prefix}.{idx}" if prefix else str(idx)
            flat.update(_flatten_for_mlflow(value, child))
    else:
        flat[prefix] = obj
    return flat


def _mlflow_value(value: Any) -> str | int | float | bool:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _mlflow_param_name(name: str) -> str:
    return name[:250]


def _log_mlflow_params(mlflow, values: dict[str, Any], prefix: str = "") -> None:
    params = {}
    for key, value in _flatten_for_mlflow(values, prefix).items():
        if key:
            params[_mlflow_param_name(key)] = _mlflow_value(value)
    # MLflow rejects overwriting a param with a different value, so log once at run start.
    for start in range(0, len(params), 100):
        mlflow.log_params(dict(list(params.items())[start : start + 100]))


def _log_mlflow_metrics(mlflow, record: dict[str, Any], *, step: int, prefix: str = "train") -> None:
    metrics = {}
    skipped = {"step", "run_id", "precision", "kernel_backends", "seed", "data_hash", "peak_flops"}
    for key, value in record.items():
        if key in skipped or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[f"{prefix}.{key}"] = float(value)
    if metrics:
        mlflow.log_metrics(metrics, step=step)


def format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "0m"
    minutes = int(round(seconds / 60))
    days, rem_minutes = divmod(minutes, 24 * 60)
    hours, mins = divmod(rem_minutes, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def format_train_record(record: dict[str, Any], total_steps: int) -> str:
    step = int(record["step"])
    loss = float(record["loss"])
    tokens_seen = int(record["tokens_seen"])
    tokens_sec = float(record["tokens_sec"])
    lr_mult = float(record["lr_multiplier"])
    grad_norm = float(record["grad_norm"])
    eta = format_duration((total_steps - step - 1) * tokens_seen / (step + 1) / tokens_sec) if tokens_sec > 0 else "?"
    parts = [
        f"step {step:>5}/{total_steps - 1}",
        f"loss {loss:7.4f}",
        f"lr {lr_mult:5.3f}",
        f"gn {grad_norm:6.3f}",
        f"{tokens_sec / 1000:>4.0f}k tok/s",
    ]
    if "mfu" in record:
        mfu = float(record["mfu"])
        parts.append(f"mfu {mfu * 100:5.1f}%")
    parts.append(f"eta {eta:>7}")
    if "val_loss" in record:
        parts.append(f"val {float(record['val_loss']):7.4f}")
    if "val_bpb" in record:
        parts.append(f"bpb {float(record['val_bpb']):5.3f}")
    return " | ".join(parts)


def training_data_info(loader: MemmapDataLoader, *, overfit_first_batch: bool) -> dict[str, Any]:
    info = loader.info()
    info["overfit_first_batch"] = overfit_first_batch
    return info


def checkpoint_evaluation_steps(
    start_step: int,
    num_iterations: int,
    checkpoint_interval: int,
    *,
    eval_final_only: bool = False,
) -> list[int]:
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be > 0")
    final_step = num_iterations - 1
    if final_step < start_step:
        return []
    if eval_final_only:
        return [final_step]
    steps = []
    for step in range(start_step, num_iterations):
        if should_checkpoint_step(step, start_step, num_iterations, checkpoint_interval):
            steps.append(step)
    return steps


def should_checkpoint_step(step: int, start_step: int, num_iterations: int, checkpoint_interval: int) -> bool:
    final_step = num_iterations - 1
    return step == final_step or (step % checkpoint_interval == 0 and step != start_step)


def should_evaluate_checkpoint_step(step: int, num_iterations: int, *, eval_final_only: bool) -> bool:
    return step == num_iterations - 1 or not eval_final_only


def validation_batch_count(
    start_step: int,
    num_iterations: int,
    checkpoint_interval: int,
    val_batches: int,
    *,
    eval_enabled: bool = True,
    eval_final_only: bool = False,
) -> int:
    if not eval_enabled:
        return 0
    return len(
        checkpoint_evaluation_steps(
            start_step,
            num_iterations,
            checkpoint_interval,
            eval_final_only=eval_final_only,
        )
    ) * val_batches


def _sanitize_mlflow_metric_component(value: Any) -> str:
    text = str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)
    return cleaned.strip("_") or "metric"


def _mlflow_metric_name(name: str) -> str:
    return name[:250]


def benchmark_mlflow_metrics(results_by_suite: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    for suite_name, suite_result in results_by_suite.items():
        task_results = suite_result.get("results", {}) if isinstance(suite_result, dict) else {}
        if not isinstance(task_results, dict):
            continue
        for task_name, task_metrics in task_results.items():
            if not isinstance(task_metrics, dict):
                continue
            for metric_name, value in task_metrics.items():
                if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                    continue
                value = float(value)
                if not math.isfinite(value):
                    continue
                key = ".".join(
                    [
                        "benchmark",
                        _sanitize_mlflow_metric_component(suite_name),
                        _sanitize_mlflow_metric_component(task_name),
                        _sanitize_mlflow_metric_component(metric_name),
                    ]
                )
                metrics[_mlflow_metric_name(key)] = value
    return metrics


def _log_mlflow_benchmark_metrics(mlflow, results_by_suite: dict[str, Any], *, step: int) -> None:
    metrics = benchmark_mlflow_metrics(results_by_suite)
    if metrics:
        mlflow.log_metrics(metrics, step=step)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return str(value)


def write_benchmark_results(run_dir: Path, step: int, tokens_seen: int, results_by_suite: dict[str, Any]) -> Path:
    path = run_dir / "eval" / f"step_{step}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "tokens_seen": tokens_seen,
        "suites": results_by_suite,
    }
    path.write_text(json.dumps(payload, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_benchmark_suites(
    raw_model,
    config,
    tokenizer_path: Path,
    device: torch.device,
    *,
    batch_size: int = 8,
) -> dict[str, Any]:
    from lm_eval_hackable_lm import SimpleLMHarness

    import lm_eval

    was_training = raw_model.training
    try:
        harness = SimpleLMHarness(
            model=raw_model,
            config=config,
            tokenizer=str(tokenizer_path),
            device=str(device),
            batch_size=batch_size,
            dtype="bfloat16",
        )
        return {
            suite.name: lm_eval.simple_evaluate(
                model=harness,
                tasks=suite.tasks,
                num_fewshot=suite.num_fewshot,
                bootstrap_iters=0,
            )
            for suite in STANDARD_EVAL_SUITES
        }
    finally:
        raw_model.train(was_training)


def run_checkpoint_evaluation(
    raw_model,
    loader: MemmapDataLoader,
    config,
    device: torch.device,
    args,
    data_manifest: dict[str, Any],
    run_dir: Path,
    step: int,
    tokens_seen: int,
    mlflow,
) -> dict[str, Any]:
    if args.disable_eval:
        return {}

    record: dict[str, Any] = {
        "val_loss": validate(raw_model, loader, config, device, args.val_batches),
    }
    val_bpb = estimate_bits_per_byte(record["val_loss"], data_manifest)
    if val_bpb is not None:
        record["val_bpb"] = val_bpb

    if not args.disable_benchmarks:
        benchmark_results = run_benchmark_suites(
            raw_model,
            config,
            Path(args.data) / "tokenizer.json",
            device,
        )
        benchmark_path = write_benchmark_results(run_dir, step, tokens_seen, benchmark_results)
        record["benchmark_results"] = str(benchmark_path)
        record["benchmark_suites"] = sorted(benchmark_results)
        if mlflow is not None:
            _log_mlflow_benchmark_metrics(mlflow, benchmark_results, step=step)
            mlflow.log_artifact(str(benchmark_path), artifact_path="eval")

    return record


def require_scheduled_data(loader: MemmapDataLoader, config, args, start_step: int, *, validate_rank: bool) -> None:
    remaining_steps = max(0, config.num_iterations - start_step)
    train_batches = 1 if args.overfit_first_batch and remaining_steps > 0 else remaining_steps * config.gradient_accumulation_steps
    loader.require_batches("train", train_batches, config.device_batch_size)
    if validate_rank:
        val_batches = validation_batch_count(
            start_step,
            config.num_iterations,
            args.checkpoint_interval,
            args.val_batches,
            eval_enabled=not args.disable_eval,
            eval_final_only=args.eval_final_only,
        )
        loader.require_batches("val", val_batches, config.device_batch_size)


def next_train_batch(prefetcher, fixed_batch, *, prepare_next: bool = True):
    if fixed_batch is not None:
        return fixed_batch
    return prefetcher.next(prepare_next=prepare_next)


def should_prepare_next_train_batch(
    step: int,
    micro_step: int,
    num_iterations: int,
    gradient_accumulation_steps: int,
    checkpoint_this_step: bool,
) -> bool:
    more_train_batches = step != num_iterations - 1 or micro_step != gradient_accumulation_steps - 1
    if not more_train_batches:
        return False
    last_micro_step = micro_step == gradient_accumulation_steps - 1
    return not (checkpoint_this_step and last_micro_step)


def has_warmdown_resume_target(args) -> bool:
    return args.warmdown_to_target_tpp is not None or args.warmdown_to_target_steps is not None


def warmdown_target_steps_from_tpp(tokens_per_param: float, config) -> tuple[int, int]:
    if tokens_per_param <= 0:
        raise ValueError("--warmdown-to-target-tpp must be > 0")
    target_tokens = math.ceil(tokens_per_param * config.scaling_params)
    target_steps = math.ceil(target_tokens / config.global_batch_tokens)
    return target_steps, target_tokens


def warmup_steps_for_num_iterations(num_iterations: int, warmup_ratio: float) -> int:
    return max(1, round(warmup_ratio * num_iterations)) if num_iterations > 0 else 0


def wsd_decay_start_step(num_iterations: int, warmup_steps: int, warmdown_ratio: float, scheduler: str) -> int:
    if scheduler != "wsd":
        raise ValueError(f"warmdown resume requires the WSD scheduler, got {scheduler!r}")
    warmup_steps = min(warmup_steps, num_iterations)
    decay_iters = int(num_iterations * warmdown_ratio)
    return max(warmup_steps, num_iterations - decay_iters)


def apply_warmdown_resume_target(config, start_step: int, args) -> int | None:
    if not has_warmdown_resume_target(args):
        return None
    if args.warmdown_to_target_tpp is not None and args.warmdown_to_target_steps is not None:
        raise ValueError("choose only one warmdown target: --warmdown-to-target-tpp or --warmdown-to-target-steps")
    if config.lr_scheduler != "wsd":
        raise ValueError(f"warmdown resume requires the WSD scheduler, got {config.lr_scheduler!r}")

    if args.warmdown_to_target_steps is not None:
        target_steps = int(args.warmdown_to_target_steps)
        if target_steps <= 0:
            raise ValueError("--warmdown-to-target-steps must be > 0")
        requested_target_tokens = target_steps * config.global_batch_tokens
        target_kind = "steps"
        target_value = target_steps
    else:
        target_steps, requested_target_tokens = warmdown_target_steps_from_tpp(args.warmdown_to_target_tpp, config)
        target_kind = "tokens_per_param"
        target_value = args.warmdown_to_target_tpp

    if target_steps <= start_step:
        raise ValueError(
            "warmdown target must be after the resumed checkpoint: "
            f"checkpoint resumes at step {start_step}, target_steps={target_steps}"
        )

    original_warmup_steps = config.warmup_steps
    target_warmup_steps = warmup_steps_for_num_iterations(target_steps, config.warmup_ratio)
    target_decay_start = wsd_decay_start_step(
        target_steps,
        target_warmup_steps,
        config.warmdown_ratio,
        config.lr_scheduler,
    )
    if target_decay_start < start_step:
        raise ValueError(
            "warmdown for the target budget would start before the resumed checkpoint: "
            f"checkpoint resumes at step {start_step}, target decay starts at step {target_decay_start}"
        )

    config.num_iterations = target_steps
    config.warmup_steps = target_warmup_steps
    config.target_tokens = int(requested_target_tokens)
    config.scheduled_tokens = target_steps * config.global_batch_tokens
    config.train_flops_budget = float(config.estimated_flops_per_token * config.scheduled_tokens)
    config.budget_policy = f"warmdown_target_{target_kind}"
    config.extra["warmdown_resume"] = {
        "enabled": True,
        "checkpoint_start_step": start_step,
        "decay_start_step": target_decay_start,
        "original_warmup_steps": original_warmup_steps,
        "target_kind": target_kind,
        "target_value": target_value,
        "target_steps": target_steps,
        "requested_target_tokens": int(requested_target_tokens),
    }
    return target_decay_start


@torch.no_grad()
def grad_global_norm(parameters: list[torch.nn.Parameter], norm_type: float = 2.0) -> torch.Tensor:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    return torch.nn.utils.get_total_norm(grads, norm_type=norm_type, foreach=True)


def _start_mlflow(args, run_dir: Path, manifest: dict):
    if args.no_mlflow:
        return None
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow logging is enabled but mlflow is not installed. Run `uv sync --locked` or pass --no-mlflow.") from exc

    tracking_uri = args.mlflow_tracking_uri or (run_dir.parent / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment or "hackable-lm")
    mlflow.start_run(run_name=args.mlflow_run_name or args.run_name)
    mlflow.set_tags(
        {
            "hackable_lm.run_name": args.run_name,
            "hackable_lm.candidate_label": args.candidate_label or "",
            "hackable_lm.comparison_mode": manifest["comparison"]["mode"],
            "hackable_lm.precision": manifest["precision_mode"],
            "hackable_lm.data_hash": manifest["data"]["manifest"].get("tokenizer_hash", ""),
            "hackable_lm.offline_tracking": str(tracking_uri.startswith("file:")).lower(),
        }
    )
    _log_mlflow_params(
        mlflow,
        {
            "args": vars(args),
            "manifest": manifest,
        },
    )
    mlflow.log_artifact(str(run_dir / "manifest.json"))
    mlflow.log_artifact(str(run_dir / "config.json"))
    print(json.dumps({"mlflow_tracking_uri": tracking_uri, "mlflow_run_id": mlflow.active_run().info.run_id}), flush=True)
    return mlflow


def _nested(obj: dict, *keys: str):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def total_train_flops_from_manifest(manifest: dict) -> float | None:
    cfg = manifest.get("config", {})
    if cfg.get("train_flops_budget") is not None:
        return cfg["train_flops_budget"]
    flops = cfg.get("estimated_flops_per_token")
    batch = cfg.get("global_batch_tokens")
    steps = cfg.get("num_iterations")
    if flops is None or batch is None or steps is None:
        return None
    return float(flops * batch * steps)


def apply_match_run_defaults(args, manifest: dict) -> None:
    cfg = manifest.get("config", {})
    if args.sequence_len is None:
        args.sequence_len = cfg.get("sequence_len")
    if args.attention_window is None:
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        args.attention_window = (model_cfg.get("attention_window") or 0) if "attention_window" in model_cfg else 0
    if args.attention_full_every is None:
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        args.attention_full_every = (model_cfg.get("attention_full_every") or 0) if "attention_full_every" in model_cfg else 0
    if args.attention_backend is None:
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        args.attention_backend = model_cfg.get("attention_backend")
    if args.global_batch_tokens is None:
        args.global_batch_tokens = cfg.get("global_batch_tokens")
    if args.device_batch_size is None:
        args.device_batch_size = cfg.get("device_batch_size")
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    for name in MODULE_BACKEND_FIELDS:
        if getattr(args, name) is None:
            setattr(args, name, model_cfg.get(name))
    if args.loss_chunk_size is None:
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        args.loss_chunk_size = model_cfg.get("loss_chunk_size")
    if args.optimizer is None:
        args.optimizer = cfg.get("optimizer")
    if args.seed is None:
        args.seed = manifest.get("seed")
    if args.data_shuffle_seed is None:
        args.data_shuffle_seed = _nested(manifest, "data", "data_shuffle_seed")
    if args.target_param_data_ratio is None:
        args.target_param_data_ratio = cfg.get("target_param_data_ratio")
        if args.target_param_data_ratio is None and cfg.get("scaling_params"):
            args.target_param_data_ratio = max(1, round(cfg.get("target_tokens", 0) / cfg["scaling_params"]))
    if args.comparison_mode == "same_depth" and args.depth is None:
        args.depth = cfg.get("depth")
    elif args.comparison_mode == "same_params" and args.target_params is None:
        args.target_params = cfg.get("scaling_params")
    elif args.comparison_mode == "same_tokens" and args.target_tokens is None:
        args.target_tokens = cfg.get("target_tokens")
    elif args.comparison_mode == "same_bytes" and args.target_bytes is None:
        args.target_bytes = cfg.get("target_bytes")
        if args.target_bytes is None:
            matched_bpt = cfg.get("bytes_per_token") or infer_bytes_per_token(_nested(manifest, "data", "manifest") or {})
            if matched_bpt and cfg.get("target_tokens"):
                args.target_bytes = int(cfg["target_tokens"] * matched_bpt)
    elif args.comparison_mode == "same_flops" and args.target_flops is None:
        args.target_flops = total_train_flops_from_manifest(manifest)
    elif args.comparison_mode == "same_time":
        if args.target_seconds is None:
            args.target_seconds = cfg.get("target_time_seconds")
        if args.tokens_per_second is None:
            args.tokens_per_second = cfg.get("tokens_per_second")


def checkpoint_data_loader_state(loader: MemmapDataLoader, ddp: dict[str, int | bool]) -> dict[str, Any] | None:
    local_state = loader.state_dict()
    if not ddp["enabled"]:
        return {
            "version": 1,
            "world_size": 1,
            "rank_states": [local_state],
        }
    rank = int(ddp["rank"])
    world_size = int(ddp["world_size"])
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(local_state, gathered, dst=0)
    if rank != 0:
        return None
    return {
        "version": 1,
        "world_size": world_size,
        "rank_states": gathered,
    }


def restore_data_loader_state(loader: MemmapDataLoader, ckpt: dict[str, Any], ddp: dict[str, int | bool]) -> None:
    state = ckpt.get("data_loader_state")
    if state is None:
        raise RuntimeError("checkpoint is missing data_loader_state; old checkpoints cannot restore data order exactly")
    if state.get("version") != 1:
        raise RuntimeError(f"unsupported checkpoint data_loader_state version: {state.get('version')!r}")
    world_size = int(ddp["world_size"])
    if state.get("world_size") != world_size:
        raise RuntimeError(
            f"checkpoint was saved with world_size={state.get('world_size')}, "
            f"but this run has world_size={world_size}"
        )
    rank_states = state.get("rank_states")
    rank = int(ddp["rank"])
    if not isinstance(rank_states, list) or rank >= len(rank_states):
        raise RuntimeError("checkpoint data_loader_state does not contain this rank")
    loader.load_state_dict(rank_states[rank])


def save_checkpoint(
    path: Path,
    raw_model,
    optimizer,
    config,
    manifest,
    step: int,
    tokens_seen: int,
    data_loader_state: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "run_manifest": manifest,
            "step": step,
            "tokens_seen": tokens_seen,
            "rng_state": rng_state(),
            "data_loader_state": data_loader_state,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast-path small LM pretraining.")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-shuffle-seed", type=int)
    parser.add_argument("--sequence-len", type=int)
    parser.add_argument("--attention-window", type=int, help="local causal attention window; use 0 for full attention")
    parser.add_argument("--attention-full-every", type=int, help="make every Nth layer full attention after local layers; use 0 for no periodic full layers")
    parser.add_argument("--attention-backend", choices=sorted(ATTENTION_BACKENDS))
    parser.add_argument("--target-param-data-ratio", type=int)
    parser.add_argument("--target-params", type=int)
    parser.add_argument("--target-tokens", type=int)
    parser.add_argument("--target-bytes", type=int)
    parser.add_argument("--bytes-per-token", type=float)
    parser.add_argument("--global-batch-tokens", type=int)
    parser.add_argument("--device-batch-size", type=int)
    parser.add_argument("--num-iterations", type=int)
    parser.add_argument("--target-flops", type=float)
    parser.add_argument("--target-seconds", type=float)
    parser.add_argument("--tokens-per-second", type=float)
    parser.add_argument("--peak-flops", type=float, help="peak bf16 FLOP/s of the GPU (e.g. 181e12 for L40); sets the MFU denominator. If omitted, MFU is not logged.")
    parser.add_argument("--comparison-mode", default="same_depth", choices=sorted(COMPARISON_MODES))
    parser.add_argument("--match-run")
    parser.add_argument("--candidate-label")
    parser.add_argument("--resume")
    parser.add_argument(
        "--warmdown-to-target-tpp",
        "--warmdown-to-target-tokens-per-param",
        dest="warmdown_to_target_tpp",
        type=float,
        help="with --resume, continue to this total tokens-per-scaling-param budget using that target run's WSD warmdown boundary",
    )
    parser.add_argument(
        "--warmdown-to-target-steps",
        type=int,
        help="with --resume, continue to this total training-step budget using that target run's WSD warmdown boundary",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--disable-eval", action="store_true", help="disable checkpoint validation and benchmark evaluation")
    parser.add_argument("--disable-benchmarks", action="store_true", help="disable lm-eval benchmarks while keeping validation bpb")
    parser.add_argument("--eval-final-only", action="store_true", help="run checkpoint validation and benchmarks only for the final checkpoint")
    parser.add_argument("--max-grad-norm", type=float, default=0.0, help="clip gradients to this norm; set <= 0 to disable clipping")
    parser.add_argument("--precision", default="bf16", choices=["bf16"])
    for name in MODULE_BACKEND_FIELDS:
        parser.add_argument(f"--{name.replace('_', '-')}", choices=sorted(MODULE_BACKENDS))
    parser.add_argument("--loss-chunk-size", type=int, help="token rows per linear CE chunk; use 0 for the built-in heuristic")
    parser.add_argument("--optimizer", choices=["muon_adamw", "adamw"])
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--compile-mode", default=DEFAULTS["compile_mode"], choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    parser.add_argument("--no-compile-capture-scalar-outputs", action="store_true")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow tracking")
    parser.add_argument("--mlflow-tracking-uri", help="MLflow tracking URI; defaults to local file store under --runs-dir/mlruns")
    parser.add_argument("--mlflow-experiment", default="hackable-lm")
    parser.add_argument("--mlflow-run-name", help="override the MLflow run name; defaults to --run-name")
    parser.add_argument(
        "--overfit-first-batch",
        action="store_true",
        help="debug sanity check: repeatedly train on the first sampled training batch",
    )
    args = parser.parse_args()
    if has_warmdown_resume_target(args) and not args.resume:
        parser.error("--warmdown-to-target-tpp/steps require --resume")
    if args.warmdown_to_target_tpp is not None and args.warmdown_to_target_steps is not None:
        parser.error("choose only one warmdown target: --warmdown-to-target-tpp or --warmdown-to-target-steps")

    matched_manifest = load_run_manifest_arg(args.match_run) if args.match_run else None
    if matched_manifest:
        apply_match_run_defaults(args, matched_manifest)
    args.seed = args.seed if args.seed is not None else 1337
    args.data_shuffle_seed = args.data_shuffle_seed if args.data_shuffle_seed is not None else 1337
    args.sequence_len = args.sequence_len if args.sequence_len is not None else DEFAULTS["sequence_len"]
    args.attention_window = args.attention_window if args.attention_window is not None else DEFAULTS["attention_window"]
    args.attention_full_every = args.attention_full_every if args.attention_full_every is not None else DEFAULTS["attention_full_every"]
    args.attention_backend = args.attention_backend if args.attention_backend is not None else DEFAULTS["attention_backend"]
    args.target_param_data_ratio = args.target_param_data_ratio if args.target_param_data_ratio is not None else DEFAULTS["target_param_data_ratio"]
    for name in MODULE_BACKEND_FIELDS:
        if getattr(args, name) is None:
            setattr(args, name, DEFAULTS[name])
    args.loss_chunk_size = args.loss_chunk_size if args.loss_chunk_size is not None else DEFAULTS["loss_chunk_size"]
    if args.loss_chunk_size < 0:
        parser.error("--loss-chunk-size must be >= 0")
    args.optimizer = args.optimizer if args.optimizer is not None else DEFAULTS["optimizer"]
    if args.peak_flops is not None and args.peak_flops <= 0:
        parser.error("--peak-flops must be > 0")
    if args.log_interval <= 0:
        parser.error("--log-interval must be > 0")
    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint-interval must be > 0")
    if args.val_batches <= 0:
        parser.error("--val-batches must be > 0")
    if args.depth is None and args.target_params is None:
        parser.error("--depth is required unless --target-params is provided or --match-run fills it")

    ddp = ddp_info()
    device = setup_ddp(ddp)
    is_main = int(ddp["rank"]) == 0
    seed_everything(args.seed)
    data_manifest = load_manifest(args.data)
    bytes_per_token = args.bytes_per_token
    if args.target_bytes is not None and bytes_per_token is None:
        bytes_per_token = infer_bytes_per_token(data_manifest)
    config = resolve_config(
        depth=args.depth,
        vocab_size=data_manifest["vocab_size"],
        sequence_len=args.sequence_len,
        attention_window=args.attention_window,
        attention_full_every=args.attention_full_every,
        attention_backend=args.attention_backend,
        target_param_data_ratio=args.target_param_data_ratio,
        target_params=args.target_params,
        target_tokens=args.target_tokens,
        target_bytes=args.target_bytes,
        bytes_per_token=bytes_per_token,
        global_batch_tokens=args.global_batch_tokens,
        device_batch_size=args.device_batch_size,
        gpu_memory_gib=gpu_memory_gib(device),
        num_iterations=args.num_iterations,
        target_flops=args.target_flops,
        target_time_seconds=args.target_seconds,
        tokens_per_second=args.tokens_per_second,
        precision=args.precision,
        compile_model=not args.no_compile,
        compile_mode=args.compile_mode,
        compile_capture_scalar_outputs=not args.no_compile_capture_scalar_outputs,
        kernel_backend="torch",
        loss_chunk_size=args.loss_chunk_size,
        optimizer=args.optimizer,
        comparison_mode=args.comparison_mode,
        **{name: getattr(args, name) for name in MODULE_BACKEND_FIELDS},
    )
    if ddp["enabled"]:
        total_grad_accum = config.gradient_accumulation_steps
        world_size = int(ddp["world_size"])
        if total_grad_accum <= 1:
            cleanup_ddp(ddp)
            raise RuntimeError("DDP is only enabled when gradient_accumulation_steps > 1.")
        if total_grad_accum % world_size != 0:
            cleanup_ddp(ddp)
            raise RuntimeError(
                "DDP requires gradient_accumulation_steps to divide evenly by WORLD_SIZE "
                f"to preserve global_batch_tokens: got gradient_accumulation_steps={total_grad_accum}, WORLD_SIZE={world_size}."
            )
        config.total_gradient_accumulation_steps = total_grad_accum
        config.gradient_accumulation_steps = total_grad_accum // world_size
        config.world_size = world_size
    kernel_info = resolve_kernel_backends(
        requested=config.kernel_backend,
        precision=config.precision,
        compile_model=config.compile,
        compile_mode=config.compile_mode,
        compile_capture_scalar_outputs=config.compile_capture_scalar_outputs,
        attention_backend=config.model.attention_backend,
        **{name: getattr(config.model, name) for name in MODULE_BACKEND_FIELDS},
    ).to_dict()
    loader = MemmapDataLoader(
        args.data,
        config.sequence_len,
        data_shuffle_seed=args.data_shuffle_seed,
        rank=int(ddp["rank"]),
        world_size=int(ddp["world_size"]),
    )
    start_step = 0
    tokens_seen = 0
    resume_ckpt = None
    warmdown_decay_start_step = None
    checkpoint_path = args.resume
    if checkpoint_path:
        resume_ckpt = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
        if resume_ckpt["config"]["model"] != config.to_dict()["model"]:
            raise RuntimeError("checkpoint model config does not match requested config")
        start_step = resume_ckpt["step"] + 1
        tokens_seen = resume_ckpt["tokens_seen"]
        try:
            warmdown_decay_start_step = apply_warmdown_resume_target(config, start_step, args)
        except ValueError as exc:
            parser.error(str(exc))
        if warmdown_decay_start_step is not None and tokens_seen != start_step * config.global_batch_tokens:
            raise RuntimeError(
                "warmdown resume checkpoint token count does not match this run's global batch size: "
                f"tokens_seen={tokens_seen}, expected {start_step * config.global_batch_tokens}"
            )
        restore_data_loader_state(loader, resume_ckpt, ddp)
    require_scheduled_data(loader, config, args, start_step, validate_rank=is_main)

    raw_model = LanguageModel(config.model).to(device)
    raw_model.prepare_compile_cache(config.sequence_len, device, batch_size=config.device_batch_size)
    raw_model = apply_precision_policy(raw_model, config.precision)
    optimizer = create_optimizer(raw_model, config)
    clip_params = [p for p in raw_model.parameters() if p.requires_grad]

    run_dir = Path(args.runs_dir) / args.run_name
    manifest = None
    if is_main:
        manifest = write_run_manifest(
            run_dir,
            config,
            raw_model,
            optimizer,
            training_data_info(loader, overfit_first_batch=args.overfit_first_batch),
            tokenizer_manifest(Path(args.data) / "tokenizer.json"),
            run_id=args.run_name,
            argv=sys.argv,
            seed=args.seed,
            kernel_info=kernel_info,
            label=args.candidate_label,
            max_grad_norm=args.max_grad_norm,
            peak_flops=args.peak_flops,
        )
    if ddp["enabled"]:
        manifest_box = [manifest]
        dist.broadcast_object_list(manifest_box, src=0)
        manifest = manifest_box[0]
    mlflow = _start_mlflow(args, run_dir, manifest) if is_main else None
    if checkpoint_path:
        ckpt = resume_ckpt
        if ckpt is None:
            raise RuntimeError("resume checkpoint was not loaded")
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        restore_rng_state(ckpt["rng_state"])

    model = compile_training_model(raw_model, config.compile, config.compile_mode, config.compile_capture_scalar_outputs)
    if ddp["enabled"]:
        model = DistributedDataParallel(model, device_ids=[int(ddp["local_rank"])], output_device=int(ddp["local_rank"]))
    train_prefetcher = loader.cuda_prefetcher("train", config.device_batch_size, device)
    fixed_train_batch = train_prefetcher.next(prepare_next=False) if args.overfit_first_batch else None
    train_log = (run_dir / "train_log.jsonl").open("a", encoding="utf-8") if is_main else None
    torch.cuda.reset_peak_memory_stats()
    train_start_time = time.perf_counter()
    last_time = time.perf_counter()
    last_logged_tokens_seen = tokens_seen
    last_train_record = None
    last_val_loss = None
    tokens_sec_samples = []
    try:
        for step in range(start_step, config.num_iterations):
            checkpoint_this_step = should_checkpoint_step(step, start_step, config.num_iterations, args.checkpoint_interval)
            log_this_step = is_main and (step % args.log_interval == 0 or checkpoint_this_step)
            mark_compiled_step_begin(config.compile)
            optimizer.zero_grad(set_to_none=True)
            total_loss = None
            for micro_step in range(config.gradient_accumulation_steps):
                should_sync = micro_step == config.gradient_accumulation_steps - 1
                sync_context = (
                    contextlib.nullcontext()
                    if should_sync or not ddp["enabled"]
                    else model.no_sync()
                )
                with sync_context:
                    prepare_next = should_prepare_next_train_batch(
                        step,
                        micro_step,
                        config.num_iterations,
                        config.gradient_accumulation_steps,
                        checkpoint_this_step,
                    )
                    x, y = next_train_batch(train_prefetcher, fixed_train_batch, prepare_next=prepare_next)
                    _, loss = model(x, y)
                    (loss / config.gradient_accumulation_steps).backward()
                    if log_this_step:
                        detached_loss = loss.detach().clone()
                        total_loss = detached_loss if total_loss is None else total_loss + detached_loss
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, args.max_grad_norm, foreach=True)
            elif log_this_step:
                grad_norm = grad_global_norm(clip_params)
            lr_mult = lr_multiplier(
                step,
                config.num_iterations,
                config.warmup_steps,
                config.warmdown_ratio,
                config.final_lr_frac,
                config.lr_scheduler,
                decay_start_step=warmdown_decay_start_step,
            )
            optimizer.set_lr_multiplier(lr_mult)
            optimizer.step()
            tokens_seen += config.global_batch_tokens

            record = None
            if log_this_step:
                now = time.perf_counter()
                elapsed = now - last_time
                logged_tokens = tokens_seen - last_logged_tokens_seen
                tokens_sec = logged_tokens / elapsed if step != start_step and logged_tokens > 0 else 0.0
                last_time = now
                last_logged_tokens_seen = tokens_seen
                train_loss = float((total_loss / config.gradient_accumulation_steps).item())
                model_flops_sec = config.estimated_flops_per_token * tokens_sec if tokens_sec else 0.0
                record = {
                    "step": step,
                    "loss": train_loss,
                    "lr_multiplier": lr_mult,
                    "lr_scheduler": config.lr_scheduler,
                    "embedding_lr": config.embedding_lr * lr_mult,
                    "unembedding_lr": config.unembedding_lr * lr_mult,
                    "matrix_lr": config.matrix_lr * lr_mult,
                    "scalar_lr": config.scalar_lr * lr_mult,
                    "optimizer": config.optimizer,
                    "muon_momentum": optimizer.muon_momentum,
                    "weight_decay": config.weight_decay,
                    "grad_norm": float(grad_norm),
                    "tokens_seen": tokens_seen,
                    "tokens_sec": tokens_sec,
                    "model_flops_sec": model_flops_sec,
                    "peak_memory": torch.cuda.max_memory_allocated(),
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "reserved_memory": torch.cuda.max_memory_reserved(),
                    "reserved_memory_gib": torch.cuda.max_memory_reserved() / 1024**3,
                    "precision": config.precision,
                    "kernel_backends": kernel_info,
                    "run_id": args.run_name,
                    "seed": args.seed,
                    "data_shuffle_seed": args.data_shuffle_seed,
                    "data_hash": data_manifest["tokenizer_hash"],
                    "overfit_first_batch": args.overfit_first_batch,
                }
                if args.peak_flops is not None:
                    record["mfu"] = model_flops_sec / args.peak_flops if model_flops_sec else 0.0
                    record["peak_flops"] = args.peak_flops
            if checkpoint_this_step:
                data_loader_state = checkpoint_data_loader_state(loader, ddp)
                if is_main:
                    if data_loader_state is None:
                        raise RuntimeError("main rank did not receive checkpoint data loader state")
                    save_checkpoint(
                        run_dir / "checkpoints" / "latest.pt",
                        raw_model,
                        optimizer,
                        config,
                        manifest,
                        step,
                        tokens_seen,
                        data_loader_state,
                    )
                    if mlflow is not None:
                        mlflow.log_artifact(str(run_dir / "checkpoints" / "latest.pt"), artifact_path="checkpoints")
                    eval_record = (
                        run_checkpoint_evaluation(
                            raw_model,
                            loader,
                            config,
                            device,
                            args,
                            data_manifest,
                            run_dir,
                            step,
                            tokens_seen,
                            mlflow,
                        )
                        if should_evaluate_checkpoint_step(
                            step,
                            config.num_iterations,
                            eval_final_only=args.eval_final_only,
                        )
                        else {}
                    )
                    if record is not None:
                        record.update(eval_record)
                    if "val_loss" in eval_record:
                        last_val_loss = eval_record["val_loss"]
                if ddp["enabled"]:
                    dist.barrier()
                if fixed_train_batch is None and step != config.num_iterations - 1:
                    train_prefetcher.prepare_next()
            if record is not None:
                print(format_train_record(record, config.num_iterations), flush=True)
                train_log.write(json.dumps(record) + "\n")
                train_log.flush()
                last_train_record = record
                if record["tokens_sec"]:
                    tokens_sec_samples.append(record["tokens_sec"])
                if mlflow is not None:
                    _log_mlflow_metrics(mlflow, record, step=step)

        if ddp["enabled"]:
            dist.barrier()
        if not is_main:
            return
        elapsed = time.perf_counter() - train_start_time
        avg_tokens_sec = sum(tokens_sec_samples) / len(tokens_sec_samples) if tokens_sec_samples else None
        final = {
            "tokens_seen": tokens_seen,
            "scheduled_tokens": config.scheduled_tokens,
            "last_step": config.num_iterations - 1,
            "elapsed_seconds": elapsed,
            "average_logged_tokens_sec": avg_tokens_sec,
            "peak_memory": torch.cuda.max_memory_allocated(),
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "last_train_loss": last_train_record.get("loss") if last_train_record else None,
            "last_val_loss": last_val_loss,
            "last_log_record": last_train_record,
        }
        write_json(run_dir / "final.json", final)
        if mlflow is not None:
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in {
                        "final.tokens_seen": tokens_seen,
                        "final.last_step": config.num_iterations - 1,
                        "final.elapsed_seconds": elapsed,
                        "final.average_logged_tokens_sec": avg_tokens_sec,
                        "final.last_train_loss": final["last_train_loss"],
                        "final.last_val_loss": last_val_loss,
                    }.items()
                    if value is not None
                }
            )
            mlflow.log_artifact(str(run_dir / "train_log.jsonl"))
            mlflow.log_artifact(str(run_dir / "final.json"))
            mlflow.log_artifact(str(run_dir / "checkpoints" / "latest.pt"), artifact_path="checkpoints")
    finally:
        if train_log is not None:
            train_log.close()
        if mlflow is not None:
            mlflow.end_run()
        cleanup_ddp(ddp)


if __name__ == "__main__":
    main()
