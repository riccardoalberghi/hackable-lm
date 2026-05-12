from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from config import COMPARISON_MODES, DEFAULTS, L40_BF16_DENSE_PEAK, resolve_config
from data import MemmapDataLoader, load_manifest
from kernels import apply_precision_policy, compile_training_model, mark_compiled_step_begin, resolve_kernel_backends
from model import LanguageModel
from optim import create_optimizer, lr_multiplier
from repro import compatibility_warnings, load_trusted_checkpoint, seed_everything, write_json, write_run_manifest
from tokenizer import tokenizer_manifest


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
    skipped = {"step", "run_id", "precision", "kernel_backends", "seed", "data_hash"}
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
    mfu = float(record["mfu"])
    lr_mult = float(record["lr_multiplier"])
    grad_norm = float(record["grad_norm"])
    eta = format_duration((total_steps - step - 1) * tokens_seen / (step + 1) / tokens_sec) if tokens_sec > 0 else "?"
    parts = [
        f"step {step:>5}/{total_steps - 1}",
        f"loss {loss:7.4f}",
        f"lr {lr_mult:5.3f}",
        f"gn {grad_norm:6.3f}",
        f"{tokens_sec / 1000:>4.0f}k tok/s",
        f"mfu {mfu * 100:5.1f}%",
        f"eta {eta:>7}",
    ]
    if "val_loss" in record:
        parts.append(f"val {float(record['val_loss']):7.4f}")
    if "val_bpb" in record:
        parts.append(f"bpb {float(record['val_bpb']):5.3f}")
    return " | ".join(parts)


TIMING_PHASES = (
    "zero_grad",
    "data_wait",
    "forward_backward",
    "prefetch",
    "grad_norm",
    "lr_update",
    "optimizer",
)


class CudaStepTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.phase_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {phase: [] for phase in TIMING_PHASES}
        self.phase_cpu_ms: dict[str, float] = {phase: 0.0 for phase in TIMING_PHASES}
        self.step_start = torch.cuda.Event(enable_timing=True)
        self.step_end = torch.cuda.Event(enable_timing=True)
        self.wall_start = 0.0

    def start_step(self) -> None:
        self.wall_start = time.perf_counter()
        self.step_start.record(torch.cuda.current_stream(self.device))

    @contextlib.contextmanager
    def measure(self, phase: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(self.device)
        start.record(stream)
        cpu_start = time.perf_counter()
        try:
            yield
        finally:
            self.phase_cpu_ms[phase] = self.phase_cpu_ms.get(phase, 0.0) + (time.perf_counter() - cpu_start) * 1000.0
            end.record(stream)
            self.phase_events.setdefault(phase, []).append((start, end))

    def finish(self, step: int) -> dict[str, Any]:
        self.step_end.record(torch.cuda.current_stream(self.device))
        self.step_end.synchronize()
        wall_ms = (time.perf_counter() - self.wall_start) * 1000.0
        total_ms = self.step_start.elapsed_time(self.step_end)
        phase_ms = {
            phase: sum(start.elapsed_time(end) for start, end in events)
            for phase, events in self.phase_events.items()
        }
        accounted_ms = sum(phase_ms.values())
        return {
            "step": step,
            "total_ms": total_ms,
            "wall_ms": wall_ms,
            "phase_ms": phase_ms,
            "phase_cpu_ms": dict(self.phase_cpu_ms),
            "unaccounted_ms": max(0.0, total_ms - accounted_ms),
        }


@contextlib.contextmanager
def timed_phase(timer: CudaStepTimer | None, phase: str):
    if timer is None:
        yield
    else:
        with timer.measure(phase):
            yield


def should_time_step(step: int, args) -> bool:
    return args.timing_interval > 0 and step >= args.timing_warmup and step % args.timing_interval == 0


def format_timing_record(timing: dict[str, Any]) -> str:
    total_ms = float(timing["total_ms"])
    phases = timing["phase_ms"]

    def phase(label: str, name: str) -> str:
        ms = float(phases.get(name, 0.0))
        pct = 100.0 * ms / total_ms if total_ms > 0 else 0.0
        return f"{label} {ms:6.1f}ms {pct:4.1f}%"

    other_ms = float(timing.get("unaccounted_ms", 0.0))
    other_pct = 100.0 * other_ms / total_ms if total_ms > 0 else 0.0
    parts = [
        f"timing step {int(timing['step']):>5}",
        f"total {total_ms:7.1f}ms",
        phase("zero", "zero_grad"),
        phase("data", "data_wait"),
        phase("fb", "forward_backward"),
        phase("pref", "prefetch"),
        phase("norm", "grad_norm"),
        phase("lr", "lr_update"),
        phase("opt", "optimizer"),
        f"other {other_ms:6.1f}ms {other_pct:4.1f}%",
    ]
    return " | ".join(parts)


def training_data_info(loader: MemmapDataLoader, *, overfit_first_batch: bool) -> dict[str, Any]:
    info = loader.info()
    info["overfit_first_batch"] = overfit_first_batch
    if overfit_first_batch:
        info["sampling_policy"] = "repeat_first_random_packed_spans_batch"
    return info


def next_train_batch(prefetcher, fixed_batch):
    if fixed_batch is not None:
        return fixed_batch
    return prefetcher.next()


@torch.no_grad()
def grad_global_norm(parameters: list[torch.nn.Parameter], norm_type: float = 2.0) -> torch.Tensor:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    get_total_norm = getattr(torch.nn.utils, "get_total_norm", None)
    if get_total_norm is not None:
        return get_total_norm(grads, norm_type=norm_type, foreach=True)
    device = grads[0].device
    norms = torch.stack([torch.linalg.vector_norm(grad.detach(), ord=norm_type).to(device) for grad in grads])
    return torch.linalg.vector_norm(norms, ord=norm_type)


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
    if args.global_batch_tokens is None:
        args.global_batch_tokens = cfg.get("global_batch_tokens")
    if args.device_batch_size is None:
        args.device_batch_size = cfg.get("device_batch_size")
    if args.loss_backend is None:
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        args.loss_backend = model_cfg.get("loss_backend")
    if args.loss_chunk_size is None:
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        args.loss_chunk_size = model_cfg.get("loss_chunk_size")
    if args.rope_backend is None:
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        args.rope_backend = model_cfg.get("rope_backend")
    if args.seed is None:
        args.seed = manifest.get("seed")
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


def save_checkpoint(path: Path, raw_model, optimizer, config, manifest, step: int, tokens_seen: int) -> None:
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
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast-path small LM pretraining.")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sequence-len", type=int)
    parser.add_argument("--attention-window", type=int, help="local causal attention window; use 0 for full attention")
    parser.add_argument("--attention-full-every", type=int, help="make every Nth layer full attention after local layers; use 0 for no periodic full layers")
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
    parser.add_argument("--comparison-mode", default="same_depth", choices=sorted(COMPARISON_MODES))
    parser.add_argument("--match-run")
    parser.add_argument("--candidate-label")
    parser.add_argument("--resume")
    parser.add_argument("--allow-resume-mismatch", action="store_true", help="resume even if checkpoint provenance differs from requested run settings")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--timing-interval", type=int, default=0, help="log CUDA event phase timing every N optimizer steps; set 0 to disable")
    parser.add_argument("--timing-warmup", type=int, default=5, help="skip this many optimizer steps before timing")
    parser.add_argument("--val-interval", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=0.0, help="clip gradients to this norm; set <= 0 to disable clipping")
    parser.add_argument("--precision", default="bf16", choices=["bf16"])
    parser.add_argument("--loss-backend", choices=["torch", "triton"])
    parser.add_argument("--loss-chunk-size", type=int, help="token rows per linear CE chunk; use 0 for the built-in heuristic")
    parser.add_argument("--rope-backend", choices=["torch", "triton"])
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
    if args.timing_interval < 0:
        parser.error("--timing-interval must be >= 0")
    if args.timing_warmup < 0:
        parser.error("--timing-warmup must be >= 0")

    matched_manifest = load_run_manifest_arg(args.match_run) if args.match_run else None
    if matched_manifest:
        apply_match_run_defaults(args, matched_manifest)
    args.seed = args.seed if args.seed is not None else 1337
    args.sequence_len = args.sequence_len if args.sequence_len is not None else DEFAULTS["sequence_len"]
    args.attention_window = args.attention_window if args.attention_window is not None else DEFAULTS["attention_window"]
    args.attention_full_every = args.attention_full_every if args.attention_full_every is not None else DEFAULTS["attention_full_every"]
    args.target_param_data_ratio = args.target_param_data_ratio if args.target_param_data_ratio is not None else DEFAULTS["target_param_data_ratio"]
    args.loss_backend = args.loss_backend if args.loss_backend is not None else DEFAULTS["loss_backend"]
    args.loss_chunk_size = args.loss_chunk_size if args.loss_chunk_size is not None else DEFAULTS["loss_chunk_size"]
    if args.loss_chunk_size < 0:
        parser.error("--loss-chunk-size must be >= 0")
    args.rope_backend = args.rope_backend if args.rope_backend is not None else DEFAULTS["rope_backend"]
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
        loss_backend=args.loss_backend,
        loss_chunk_size=args.loss_chunk_size,
        rope_backend=args.rope_backend,
        comparison_mode=args.comparison_mode,
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
        config.kernel_backend,
        config.precision,
        config.compile,
        config.compile_mode,
        config.compile_capture_scalar_outputs,
        config.model.loss_backend,
        config.model.rope_backend,
    ).to_dict()
    loader = MemmapDataLoader(args.data, config.sequence_len, seed=args.seed + int(ddp["rank"]))
    raw_model = LanguageModel(config.model).to(device)
    raw_model.prepare_compile_cache(config.sequence_len, device)
    raw_model = apply_precision_policy(raw_model, config.precision)
    optimizer = create_optimizer(raw_model, config)
    clip_params = [p for p in raw_model.parameters() if p.requires_grad]

    start_step = 0
    tokens_seen = 0
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
        )
    if ddp["enabled"]:
        manifest_box = [manifest]
        dist.broadcast_object_list(manifest_box, src=0)
        manifest = manifest_box[0]
    mlflow = _start_mlflow(args, run_dir, manifest) if is_main else None
    if args.resume:
        ckpt = load_trusted_checkpoint(args.resume, map_location="cpu")
        if ckpt["config"]["model"] != config.to_dict()["model"]:
            raise RuntimeError("checkpoint model config does not match requested config")
        resume_manifest = ckpt.get("run_manifest")
        if resume_manifest:
            resume_warnings = compatibility_warnings(resume_manifest, manifest)
            if resume_warnings and not args.allow_resume_mismatch:
                formatted = "\n".join(f"- {warning}" for warning in resume_warnings)
                raise RuntimeError(
                    "checkpoint provenance does not match requested resume settings:\n"
                    f"{formatted}\n"
                    "Pass --allow-resume-mismatch only for an intentional non-paper resume."
                )
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        restore_rng_state(ckpt["rng_state"])
        start_step = ckpt["step"] + 1
        tokens_seen = ckpt["tokens_seen"]

    model = compile_training_model(raw_model, config.compile, config.compile_mode, config.compile_capture_scalar_outputs)
    if ddp["enabled"]:
        model = DistributedDataParallel(model, device_ids=[int(ddp["local_rank"])], output_device=int(ddp["local_rank"]))
    train_prefetcher = loader.cuda_prefetcher("train", config.device_batch_size, device)
    fixed_train_batch = train_prefetcher.next() if args.overfit_first_batch else None
    train_log = (run_dir / "train_log.jsonl").open("a", encoding="utf-8") if is_main else None
    timing_log = (run_dir / "timing_log.jsonl").open("a", encoding="utf-8") if is_main and args.timing_interval > 0 else None
    torch.cuda.reset_peak_memory_stats()
    train_start_time = time.perf_counter()
    last_time = time.perf_counter()
    last_train_record = None
    last_val_loss = None
    tokens_sec_samples = []
    try:
        for step in range(start_step, config.num_iterations):
            log_this_step = is_main and (step % args.log_interval == 0 or step == config.num_iterations - 1)
            timer = CudaStepTimer(device) if is_main and should_time_step(step, args) else None
            if timer is not None:
                timer.start_step()
            mark_compiled_step_begin(config.compile)
            with timed_phase(timer, "zero_grad"):
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
                    with timed_phase(timer, "data_wait"):
                        x, y = next_train_batch(train_prefetcher, fixed_train_batch)
                    with timed_phase(timer, "forward_backward"):
                        _, loss = model(x, y)
                        (loss / config.gradient_accumulation_steps).backward()
                    if fixed_train_batch is None:
                        with timed_phase(timer, "prefetch"):
                            train_prefetcher.preload()
                    if log_this_step:
                        detached_loss = loss.detach().clone()
                        total_loss = detached_loss if total_loss is None else total_loss + detached_loss
            with timed_phase(timer, "grad_norm"):
                if args.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, args.max_grad_norm, foreach=True)
                else:
                    grad_norm = grad_global_norm(clip_params)
            with timed_phase(timer, "lr_update"):
                lr_mult = lr_multiplier(step, config.num_iterations, config.warmup_steps, config.warmdown_ratio, config.final_lr_frac, config.lr_scheduler)
                optimizer.set_lr_multiplier(lr_mult)
            with timed_phase(timer, "optimizer"):
                optimizer.step()
            timing_record = timer.finish(step) if timer is not None else None
            tokens_seen += config.global_batch_tokens

            if timing_record is not None:
                print(format_timing_record(timing_record), flush=True)
                if timing_log is not None:
                    timing_log.write(json.dumps(timing_record) + "\n")
                    timing_log.flush()
                if mlflow is not None:
                    _log_mlflow_metrics(
                        mlflow,
                        {
                            "timing_total_ms": timing_record["total_ms"],
                            "timing_unaccounted_ms": timing_record["unaccounted_ms"],
                            **{f"timing_{name}_ms": value for name, value in timing_record["phase_ms"].items()},
                        },
                        step=step,
                    )

            if log_this_step:
                now = time.perf_counter()
                elapsed = now - last_time
                tokens_sec = config.global_batch_tokens * args.log_interval / elapsed if step != start_step else 0.0
                last_time = now
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
                    "muon_momentum": optimizer.muon_momentum,
                    "weight_decay": config.weight_decay,
                    "grad_norm": float(grad_norm),
                    "tokens_seen": tokens_seen,
                    "tokens_sec": tokens_sec,
                    "model_flops_sec": model_flops_sec,
                    "mfu": model_flops_sec / L40_BF16_DENSE_PEAK if model_flops_sec else 0.0,
                    "bf16_mfu": model_flops_sec / L40_BF16_DENSE_PEAK if model_flops_sec else 0.0,
                    "peak_memory": torch.cuda.max_memory_allocated(),
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "reserved_memory": torch.cuda.max_memory_reserved(),
                    "reserved_memory_gib": torch.cuda.max_memory_reserved() / 1024**3,
                    "precision": config.precision,
                    "kernel_backends": kernel_info,
                    "run_id": args.run_name,
                    "seed": args.seed,
                    "data_hash": data_manifest["tokenizer_hash"],
                    "overfit_first_batch": args.overfit_first_batch,
                }
                if timing_record is not None:
                    record["timing"] = timing_record
                if step % args.val_interval == 0 and step != start_step:
                    record["val_loss"] = validate(raw_model, loader, config, device, args.val_batches)
                    val_bpb = estimate_bits_per_byte(record["val_loss"], data_manifest)
                    if val_bpb is not None:
                        record["val_bpb"] = val_bpb
                    last_val_loss = record["val_loss"]
                print(format_train_record(record, config.num_iterations), flush=True)
                train_log.write(json.dumps(record) + "\n")
                train_log.flush()
                last_train_record = record
                if tokens_sec:
                    tokens_sec_samples.append(tokens_sec)
                if mlflow is not None:
                    _log_mlflow_metrics(mlflow, record, step=step)
            if is_main and step % args.checkpoint_interval == 0 and step != start_step:
                save_checkpoint(run_dir / "checkpoints" / "latest.pt", raw_model, optimizer, config, manifest, step, tokens_seen)
                if mlflow is not None:
                    mlflow.log_artifact(str(run_dir / "checkpoints" / "latest.pt"), artifact_path="checkpoints")

        if ddp["enabled"]:
            dist.barrier()
        if not is_main:
            return
        save_checkpoint(run_dir / "checkpoints" / "latest.pt", raw_model, optimizer, config, manifest, config.num_iterations - 1, tokens_seen)
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
            if timing_log is not None:
                mlflow.log_artifact(str(run_dir / "timing_log.jsonl"))
            mlflow.log_artifact(str(run_dir / "final.json"))
            mlflow.log_artifact(str(run_dir / "checkpoints" / "latest.pt"), artifact_path="checkpoints")
    finally:
        if train_log is not None:
            train_log.close()
        if timing_log is not None:
            timing_log.close()
        if mlflow is not None:
            mlflow.end_run()
        cleanup_ddp(ddp)


if __name__ == "__main__":
    main()
