from __future__ import annotations

import importlib.util
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


COMPUTE_DTYPE_ENV = "SIMPLE_LM_COMPUTE_DTYPE"
COMPUTE_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


@dataclass
class KernelInfo:
    requested_backend: str
    actual_attention_backend: str
    actual_norm_backend: str
    actual_loss_backend: str
    actual_rope_backend: str
    torch_compile: bool
    torch_compile_mode: str
    torch_compile_capture_scalar_outputs: bool
    precision: str
    flash_attn_available: bool
    fp8_available: bool
    liger_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FLASH_ATTN = None
_LIGER_FUSED_LINEAR_CE = None


def compute_dtype_for_device(device: torch.device | str) -> torch.dtype:
    override = os.environ.get(COMPUTE_DTYPE_ENV)
    if override is not None:
        try:
            return COMPUTE_DTYPES[override.lower()]
        except KeyError as exc:
            raise RuntimeError(
                f"{COMPUTE_DTYPE_ENV} must be one of {sorted(COMPUTE_DTYPES)}, got {override!r}"
            ) from exc
    device = torch.device(device)
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def fp8_available() -> bool:
    try:
        from fp8 import fp8_cuda_supported
    except Exception:
        return False
    return fp8_cuda_supported()


def liger_available() -> bool:
    return importlib.util.find_spec("liger_kernel") is not None


def _liger_fused_linear_ce():
    global _LIGER_FUSED_LINEAR_CE
    if _LIGER_FUSED_LINEAR_CE is None:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        _LIGER_FUSED_LINEAR_CE = LigerFusedLinearCrossEntropyLoss(reduction="mean")
    return _LIGER_FUSED_LINEAR_CE


def _import_flash_attn():
    global _FLASH_ATTN
    if _FLASH_ATTN is not None:
        return None if _FLASH_ATTN is False else _FLASH_ATTN
    if importlib.util.find_spec("flash_attn") is None:
        _FLASH_ATTN = False
        return None
    from flash_attn import flash_attn_func

    _FLASH_ATTN = flash_attn_func
    return flash_attn_func


def configure_torch_compile(capture_scalar_outputs: bool) -> None:
    import torch._dynamo

    torch._dynamo.config.capture_scalar_outputs = capture_scalar_outputs


def compile_training_model(
    model: torch.nn.Module,
    enabled: bool,
    mode: str = "default",
    capture_scalar_outputs: bool = True,
) -> torch.nn.Module:
    if not enabled:
        return model
    configure_torch_compile(capture_scalar_outputs)
    return torch.compile(model, mode=None if mode == "default" else mode)


def mark_compiled_step_begin(enabled: bool) -> None:
    if not enabled:
        return
    torch.compiler.cudagraph_mark_step_begin()


def resolve_kernel_backends(
    requested: str,
    precision: str,
    compile_model: bool,
    compile_mode: str = "default",
    compile_capture_scalar_outputs: bool = True,
    norm_backend: str = "torch",
    loss_backend: str = "torch",
    allow_torch_backend: bool = False,
) -> KernelInfo:
    requested = requested.lower()
    if requested != "torch":
        raise ValueError(f"unknown kernel backend {requested!r}")
    norm_backend = norm_backend.lower()
    if norm_backend != "torch":
        raise ValueError(f"unknown norm backend {norm_backend!r}")
    loss_backend = loss_backend.lower()
    if loss_backend not in {"torch", "liger"}:
        raise ValueError(f"unknown loss backend {loss_backend!r}")
    flash_attn = _import_flash_attn()
    has_fp8 = fp8_available()
    has_liger = liger_available()
    if not flash_attn and not allow_torch_backend:
        raise RuntimeError(
            "FlashAttention 2 is required for the training fast path. "
            "Install it with `pip install flash-attn --no-build-isolation`."
        )
    if precision == "fp8" and not has_fp8 and not allow_torch_backend:
        raise RuntimeError(
            "FP8 training was requested, but this CUDA device/PyTorch build does not "
            "support torch._scaled_mm with float8 dtypes. Use an Ada/Hopper-or-newer "
            "CUDA GPU and a recent PyTorch build."
        )
    if loss_backend == "liger" and not has_liger and not allow_torch_backend:
        raise RuntimeError(
            "loss_backend=liger was requested but liger-kernel is not importable. "
            "Install with `pip install liger-kernel` on a CUDA PyTorch environment."
        )
    return KernelInfo(
        requested_backend=requested,
        actual_attention_backend="flash_attn_2" if flash_attn else "torch_sdpa",
        actual_norm_backend="torch",
        actual_loss_backend="liger_fused_linear_ce" if loss_backend == "liger" and has_liger else "torch",
        actual_rope_backend="torch",
        torch_compile=compile_model,
        torch_compile_mode=compile_mode,
        torch_compile_capture_scalar_outputs=compile_capture_scalar_outputs,
        precision=precision,
        flash_attn_available=bool(flash_attn),
        fp8_available=has_fp8,
        liger_available=has_liger,
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float, backend: str = "torch") -> torch.Tensor:
    if backend != "torch":
        raise RuntimeError(f"unsupported RMSNorm backend {backend!r}")
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return y if weight is None else y * weight


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    is_causal: bool = True,
    attn_mask: torch.Tensor | None = None,
    window_size: int | None = None,
    backend: str = "flash_attn_2",
) -> torch.Tensor:
    if backend == "torch":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    if backend != "flash_attn_2":
        raise RuntimeError(f"unsupported attention backend {backend!r}; the training fast path requires flash_attn_2")
    if attn_mask is not None:
        raise RuntimeError("flash_attn_2 path expects window_size instead of an explicit attention mask")
    flash_attn_func = _import_flash_attn()
    if flash_attn_func is None:
        raise RuntimeError(
            "FlashAttention 2 is required for attention. "
            "Install it with `pip install flash-attn --no-build-isolation`."
        )
    q = q.bfloat16()
    k = k.bfloat16()
    v = v.bfloat16()
    flash_window = (-1, -1) if window_size is None else (window_size - 1, 0)
    return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=is_causal, window_size=flash_window)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, backend: str = "torch") -> torch.Tensor:
    if backend != "torch":
        raise RuntimeError(f"unsupported cross entropy backend {backend!r}")
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def chunked_linear_cross_entropy(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    targets: torch.Tensor,
    chunk_size: int,
    backend: str = "torch",
) -> torch.Tensor:
    if backend == "liger":
        if importlib.util.find_spec("liger_kernel") is None:
            raise RuntimeError(
                "loss_backend=liger requires liger-kernel. Install with `pip install liger-kernel`."
            )
        hidden_flat = hidden.reshape(-1, hidden.size(-1))
        targets_flat = targets.reshape(-1).contiguous()
        loss_fn = _liger_fused_linear_ce()
        weight = lm_head.weight
        bias = lm_head.bias
        if weight.dtype != hidden_flat.dtype:
            weight = weight.to(dtype=hidden_flat.dtype)
        if bias is not None and bias.dtype != hidden_flat.dtype:
            bias = bias.to(dtype=hidden_flat.dtype)
        return loss_fn(weight, hidden_flat, targets_flat, bias=bias)
    if backend != "torch":
        raise RuntimeError(f"unsupported chunked linear cross entropy backend {backend!r}")
    if chunk_size <= 0:
        raise RuntimeError(f"loss_chunk_size must be positive, got {chunk_size}")
    hidden_flat = hidden.reshape(-1, hidden.size(-1))
    targets_flat = targets.reshape(-1)
    n_tokens = targets_flat.numel()
    loss_sum = hidden_flat.new_zeros((), dtype=torch.float32)
    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        logits = lm_head(hidden_flat[start:end])
        loss_sum = loss_sum + F.cross_entropy(logits.float(), targets_flat[start:end], reduction="sum")
    return loss_sum / n_tokens


def apply_precision_policy(model: torch.nn.Module, precision: str, allow_torch_backend: bool = False) -> torch.nn.Module:
    if precision == "fp8":
        from fp8 import convert_to_float8_training, fp8_cuda_supported

        if not fp8_cuda_supported():
            if allow_torch_backend:
                return model
            raise RuntimeError(
                "FP8 training requires torch._scaled_mm float8 support on an "
                "Ada/Hopper-or-newer CUDA GPU."
            )

        def module_filter(module: torch.nn.Module, fqn: str) -> bool:
            if not isinstance(module, torch.nn.Linear):
                return False
            if fqn == "lm_head":
                return False
            if module.in_features % 16 != 0 or module.out_features % 16 != 0:
                return False
            return min(module.in_features, module.out_features) >= 128

        return convert_to_float8_training(model, module_filter_fn=module_filter)
    if precision in {"fp32_test", "bf16_test"} and allow_torch_backend:
        return model
    raise RuntimeError(f"unsupported precision mode {precision!r}; the training fast path uses fp8")
