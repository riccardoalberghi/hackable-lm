from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
import os
import platform
import random
import socket
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


def seed_everything(seed: int) -> None:
    if torch is None:
        raise RuntimeError("seed_everything requires torch to be installed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _match_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def hash_directory(
    path: str | Path,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
) -> str:
    root = Path(path)
    include_globs = include_globs or ["*.py", "*.rs", "*.sh", "*.md", "*.txt", "*.toml", "Cargo.lock", "uv.lock"]
    exclude_globs = exclude_globs or ["runs/*", "data/*", "eval_data/*", "*/target/*", "__pycache__/*", "*.pyc"]
    h = hashlib.sha256()
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = file.relative_to(root).as_posix()
        if _match_any(rel, exclude_globs) or not _match_any(rel, include_globs):
            continue
        h.update(rel.encode())
        h.update(hash_file(file).encode())
    return h.hexdigest()


def _version(module_name: str) -> str | None:
    if importlib.util.find_spec(module_name) is None:
        return None
    module = __import__(module_name)
    return getattr(module, "__version__", None) or "installed"


def _git_info(cwd: str | Path) -> dict[str, Any]:
    cwd = Path(cwd)
    commit = subprocess.check_output(["git", "-C", str(cwd), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    diff = subprocess.check_output(["git", "-C", str(cwd), "diff", "--", "."], text=True, stderr=subprocess.DEVNULL)
    return {
        "commit": commit,
        "dirty": bool(diff.strip()),
        "diff_hash": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def collect_environment(cwd: str | Path = ".") -> dict[str, Any]:
    if torch is None:
        cuda = None
        torch_version = None
        gpu = {}
    else:
        cuda = torch.version.cuda
        torch_version = torch.__version__
        gpu = {}
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            gpu = {"name": props.name, "memory_bytes": props.total_memory, "capability": [props.major, props.minor]}
    env_keys = ["CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED"]
    env_keys += sorted(k for k in os.environ if k.startswith("NCCL_"))
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch_version,
        "cuda": cuda,
        "gpu": gpu,
        "env": {k: os.environ.get(k) for k in env_keys if k in os.environ},
        "git": _git_info(cwd),
        "code_hash": hash_directory(cwd),
    }


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if torch is not None and isinstance(obj, torch.dtype):
        return str(obj)
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=_jsonable))


def load_trusted_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # Run checkpoints are local artifacts containing optimizer and RNG metadata,
    # so they require PyTorch's full trusted deserializer.
    if torch is None:
        raise RuntimeError("checkpoint loading requires torch to be installed")
    return torch.load(path, map_location=map_location, weights_only=False)


def write_run_manifest(
    run_dir: str | Path,
    config: Any,
    model: torch.nn.Module,
    optimizer: Any,
    data_info: dict[str, Any],
    tokenizer_info: dict[str, Any],
    *,
    run_id: str,
    argv: list[str],
    seed: int,
    kernel_info: dict[str, Any],
    label: str | None = None,
    max_grad_norm: float = 0.0,
    peak_flops: float | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "experiment_name": run_id,
        "label": label,
        "scaling_policy": config.scaling_policy,
        "config": config.to_dict(),
        "argv": argv,
        "seed": seed,
        "rng_seeds": {"python": seed, "numpy": seed, "torch": seed, "cuda": seed},
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "scaling_parameter_count": config.scaling_params,
        "optimizer_grouping": optimizer.summary(),
        "precision_mode": config.precision,
        "kernel_backends": kernel_info,
        "torch_compile": config.compile,
        "torch_compile_mode": config.compile_mode,
        "torch_compile_capture_scalar_outputs": config.compile_capture_scalar_outputs,
        "global_batch_tokens": config.global_batch_tokens,
        "microbatch_size": config.device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "total_gradient_accumulation_steps": config.total_gradient_accumulation_steps,
        "gradient_clipping": {
            "enabled": max_grad_norm > 0,
            "max_norm": max_grad_norm,
        },
        "world_size": config.world_size,
        "scheduled_tokens": config.scheduled_tokens,
        "train_flops_budget": config.train_flops_budget,
        "comparison": {
            "mode": config.comparison_mode,
            "shape_policy": config.shape_policy,
            "budget_policy": config.budget_policy,
        },
        "lr_schedule": {
            "scheduler": config.lr_scheduler,
            "warmup_ratio": config.warmup_ratio,
            "warmup_steps": config.warmup_steps,
            "warmdown_ratio": config.warmdown_ratio,
            "final_lr_frac": config.final_lr_frac,
        },
        "token_budget": config.target_tokens,
        "peak_flops": peak_flops,
        "data": data_info,
        "tokenizer": tokenizer_info,
        "environment": collect_environment(Path.cwd()),
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "config.json", config.to_dict())
    return manifest
