from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch._dynamo
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


@dataclass
class KernelInfo:
    requested_backend: str
    actual_attention_backend: str
    actual_norm_backend: str
    actual_mlp_backend: str
    actual_loss_backend: str
    actual_rope_backend: str
    torch_compile: bool
    torch_compile_mode: str
    torch_compile_capture_scalar_outputs: bool
    precision: str
    flash_attn_available: bool
    liger_available: bool
    triton_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FLASH_ATTN = None
_LIGER_FUSED_LINEAR_CE = None


def liger_available() -> bool:
    return importlib.util.find_spec("liger_kernel") is not None


def triton_available() -> bool:
    return triton is not None


def _require_liger(backend: str) -> None:
    if not liger_available():
        raise RuntimeError(
            f"{backend} requires liger-kernel. Add it with `uv sync --locked` "
            "in the CUDA environment."
        )


def _require_triton(backend: str) -> None:
    if not triton_available():
        raise RuntimeError(
            f"{backend} requires Triton. Add it with `uv sync --locked` "
            "in the CUDA environment."
        )


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
    loss_backend: str = "torch",
    rope_backend: str = "torch",
    allow_torch_backend: bool = False,
) -> KernelInfo:
    requested = requested.lower()
    if requested != "torch":
        raise ValueError(f"unknown kernel backend {requested!r}")
    loss_backend = loss_backend.lower()
    if loss_backend not in {"torch", "liger"}:
        raise ValueError(f"unknown loss backend {loss_backend!r}")
    rope_backend = rope_backend.lower()
    if rope_backend not in {"torch", "triton"}:
        raise ValueError(f"unknown RoPE backend {rope_backend!r}")
    flash_attn = _import_flash_attn()
    has_liger = liger_available()
    has_triton = triton_available()
    if not flash_attn and not allow_torch_backend:
        raise RuntimeError(
            "FlashAttention 2 is required for the training fast path. "
            "Run `uv sync --locked` in the CUDA environment."
        )
    if precision != "bf16" and not allow_torch_backend:
        raise RuntimeError(
            f"unsupported precision mode {precision!r}; the training fast path uses bf16"
        )
    if loss_backend == "liger" and not has_liger and not allow_torch_backend:
        raise RuntimeError(
            "A Liger backend was requested but liger-kernel is not importable. "
            "Run `uv sync --locked` in the CUDA environment."
        )
    if rope_backend == "triton" and not has_triton and not allow_torch_backend:
        raise RuntimeError(
            "rope_backend=triton requires Triton. "
            "Run `uv sync --locked` in the CUDA environment."
        )
    return KernelInfo(
        requested_backend=requested,
        actual_attention_backend="flash_attn_2" if flash_attn else "torch_sdpa",
        actual_norm_backend="torch",
        actual_mlp_backend="torch",
        actual_loss_backend="liger_fused_linear_ce" if loss_backend == "liger" and has_liger else "torch",
        actual_rope_backend="triton" if rope_backend == "triton" and has_triton else "torch",
        torch_compile=compile_model,
        torch_compile_mode=compile_mode,
        torch_compile_capture_scalar_outputs=compile_capture_scalar_outputs,
        precision=precision,
        flash_attn_available=bool(flash_attn),
        liger_available=has_liger,
        triton_available=has_triton,
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return y if weight is None else y * weight


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


if triton is not None:
    _QK_NORM_ROPE_CONFIGS = [
        triton.Config({"BLOCK_M": 1}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 2}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 4}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 4}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 8}, num_warps=8, num_stages=3),
    ]

    @triton.autotune(
        configs=_QK_NORM_ROPE_CONFIGS,
        key=["SEQ_LEN", "Q_HEADS", "K_HEADS", "HEAD_DIM", "ROTARY_DIM"],
    )
    @triton.jit
    def _qk_norm_rope_kernel(
        q,
        k,
        cos,
        sin,
        q_out,
        k_out,
        Q_HEADS: tl.constexpr,
        K_HEADS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        EPS: tl.constexpr,
        Q_STRIDE_B: tl.constexpr,
        Q_STRIDE_T: tl.constexpr,
        Q_STRIDE_H: tl.constexpr,
        Q_STRIDE_D: tl.constexpr,
        K_STRIDE_B: tl.constexpr,
        K_STRIDE_T: tl.constexpr,
        K_STRIDE_H: tl.constexpr,
        K_STRIDE_D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)
        offs_t = pid_t * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask = (offs_t[:, None] < SEQ_LEN) & (offs_d[None, :] < HEAD_DIM)
        half: tl.constexpr = ROTARY_DIM // 2

        is_q = pid_h < Q_HEADS
        head = tl.where(is_q, pid_h, pid_h - Q_HEADS)
        q_ptr = q + pid_b * Q_STRIDE_B + offs_t[:, None] * Q_STRIDE_T + head * Q_STRIDE_H + offs_d[None, :] * Q_STRIDE_D
        k_ptr = k + pid_b * K_STRIDE_B + offs_t[:, None] * K_STRIDE_T + head * K_STRIDE_H + offs_d[None, :] * K_STRIDE_D
        x_ptr = tl.where(is_q, q_ptr, k_ptr)
        x_vals = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x_vals * x_vals, axis=1) / HEAD_DIM + EPS)
        x_norm = x_vals * inv[:, None]

        pass_mask = offs_d >= ROTARY_DIM
        pair = offs_d % half
        rot_mask = mask & (offs_d[None, :] < ROTARY_DIM)
        c = tl.load(cos + offs_t[:, None] * ROTARY_DIM + pair[None, :], mask=rot_mask, other=1.0).to(tl.float32)
        s = tl.load(sin + offs_t[:, None] * ROTARY_DIM + pair[None, :], mask=rot_mask, other=0.0).to(tl.float32)
        pair_delta = tl.where(offs_d[None, :] < half, half, -half)
        q_pair_ptr = q + pid_b * Q_STRIDE_B + offs_t[:, None] * Q_STRIDE_T + head * Q_STRIDE_H + (offs_d[None, :] + pair_delta) * Q_STRIDE_D
        k_pair_ptr = k + pid_b * K_STRIDE_B + offs_t[:, None] * K_STRIDE_T + head * K_STRIDE_H + (offs_d[None, :] + pair_delta) * K_STRIDE_D
        pair_ptr = tl.where(is_q, q_pair_ptr, k_pair_ptr)
        x_pair = tl.load(pair_ptr, mask=rot_mask, other=0.0).to(tl.float32) * inv[:, None]
        sign = tl.where(offs_d < half, -1.0, 1.0)
        x_rot = tl.where(pass_mask[None, :], x_norm, x_norm * c + sign[None, :] * x_pair * s)

        q_out_ptr = q_out + ((pid_b * SEQ_LEN + offs_t[:, None]) * Q_HEADS + head) * HEAD_DIM + offs_d[None, :]
        k_out_ptr = k_out + ((pid_b * SEQ_LEN + offs_t[:, None]) * K_HEADS + head) * HEAD_DIM + offs_d[None, :]
        out_ptr = tl.where(is_q, q_out_ptr, k_out_ptr)
        tl.store(out_ptr, x_rot, mask=mask)

    @torch.library.triton_op("simple_lm::qk_norm_rope", mutates_args={})
    def _triton_qk_norm_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
        k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
        head_dim = q.shape[-1]
        rotary_dim = cos.shape[-1]
        seq_len = q.shape[1]
        cos_2d = cos.reshape(seq_len, rotary_dim)
        sin_2d = sin.reshape(seq_len, rotary_dim)
        block_d = triton.next_power_of_2(head_dim)
        grid = lambda meta: (triton.cdiv(seq_len, meta["BLOCK_M"]), q.shape[0], q.shape[2] + k.shape[2])
        torch.library.wrap_triton(_qk_norm_rope_kernel)[grid](
            q,
            k,
            cos_2d,
            sin_2d,
            q_out,
            k_out,
            q.shape[2],
            k.shape[2],
            seq_len,
            head_dim,
            rotary_dim,
            eps,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            BLOCK_D=block_d,
        )
        return q_out, k_out

    @triton.autotune(
        configs=_QK_NORM_ROPE_CONFIGS,
        key=["SEQ_LEN", "Q_HEADS", "K_HEADS", "HEAD_DIM", "ROTARY_DIM"],
    )
    @triton.jit
    def _qk_norm_rope_backward_kernel(
        q,
        k,
        grad_q,
        grad_k,
        cos,
        sin,
        grad_q_out,
        grad_k_out,
        Q_HEADS: tl.constexpr,
        K_HEADS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        EPS: tl.constexpr,
        Q_STRIDE_B: tl.constexpr,
        Q_STRIDE_T: tl.constexpr,
        Q_STRIDE_H: tl.constexpr,
        Q_STRIDE_D: tl.constexpr,
        K_STRIDE_B: tl.constexpr,
        K_STRIDE_T: tl.constexpr,
        K_STRIDE_H: tl.constexpr,
        K_STRIDE_D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)
        offs_t = pid_t * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask = (offs_t[:, None] < SEQ_LEN) & (offs_d[None, :] < HEAD_DIM)
        half: tl.constexpr = ROTARY_DIM // 2

        is_q = pid_h < Q_HEADS
        head = tl.where(is_q, pid_h, pid_h - Q_HEADS)
        q_ptr = q + pid_b * Q_STRIDE_B + offs_t[:, None] * Q_STRIDE_T + head * Q_STRIDE_H + offs_d[None, :] * Q_STRIDE_D
        k_ptr = k + pid_b * K_STRIDE_B + offs_t[:, None] * K_STRIDE_T + head * K_STRIDE_H + offs_d[None, :] * K_STRIDE_D
        x_ptr = tl.where(is_q, q_ptr, k_ptr)
        x_vals = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)

        q_grad_ptr = grad_q + ((pid_b * SEQ_LEN + offs_t[:, None]) * Q_HEADS + head) * HEAD_DIM + offs_d[None, :]
        k_grad_ptr = grad_k + ((pid_b * SEQ_LEN + offs_t[:, None]) * K_HEADS + head) * HEAD_DIM + offs_d[None, :]
        grad_ptr = tl.where(is_q, q_grad_ptr, k_grad_ptr)
        grad_vals = tl.load(grad_ptr, mask=mask, other=0.0).to(tl.float32)

        inv = tl.rsqrt(tl.sum(x_vals * x_vals, axis=1) / HEAD_DIM + EPS)
        pass_mask = offs_d >= ROTARY_DIM
        pair = offs_d % half
        rot_mask = mask & (offs_d[None, :] < ROTARY_DIM)
        c = tl.load(cos + offs_t[:, None] * ROTARY_DIM + pair[None, :], mask=rot_mask, other=1.0).to(tl.float32)
        s = tl.load(sin + offs_t[:, None] * ROTARY_DIM + pair[None, :], mask=rot_mask, other=0.0).to(tl.float32)
        pair_delta = tl.where(offs_d[None, :] < half, half, -half)
        q_grad_pair_ptr = grad_q + ((pid_b * SEQ_LEN + offs_t[:, None]) * Q_HEADS + head) * HEAD_DIM + offs_d[None, :] + pair_delta
        k_grad_pair_ptr = grad_k + ((pid_b * SEQ_LEN + offs_t[:, None]) * K_HEADS + head) * HEAD_DIM + offs_d[None, :] + pair_delta
        grad_pair_ptr = tl.where(is_q, q_grad_pair_ptr, k_grad_pair_ptr)
        grad_pair = tl.load(grad_pair_ptr, mask=rot_mask, other=0.0).to(tl.float32)
        sign = tl.where(offs_d < half, 1.0, -1.0)
        grad_norm = tl.where(pass_mask[None, :], grad_vals, grad_vals * c + sign[None, :] * grad_pair * s)

        z = x_vals * inv[:, None]
        dot = tl.sum(grad_norm * z, axis=1) / HEAD_DIM
        grad_x = inv[:, None] * (grad_norm - z * dot[:, None])
        q_grad_out_ptr = grad_q_out + ((pid_b * SEQ_LEN + offs_t[:, None]) * Q_HEADS + head) * HEAD_DIM + offs_d[None, :]
        k_grad_out_ptr = grad_k_out + ((pid_b * SEQ_LEN + offs_t[:, None]) * K_HEADS + head) * HEAD_DIM + offs_d[None, :]
        grad_out_ptr = tl.where(is_q, q_grad_out_ptr, k_grad_out_ptr)
        tl.store(grad_out_ptr, grad_x, mask=mask)

    @torch.library.triton_op("simple_lm::qk_norm_rope_backward", mutates_args={})
    def _triton_qk_norm_rope_backward_op(
        q: torch.Tensor,
        k: torch.Tensor,
        grad_q: torch.Tensor,
        grad_k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grad_q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
        grad_k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
        head_dim = q.shape[-1]
        rotary_dim = cos.shape[-1]
        seq_len = q.shape[1]
        cos_2d = cos.reshape(seq_len, rotary_dim)
        sin_2d = sin.reshape(seq_len, rotary_dim)
        block_d = triton.next_power_of_2(head_dim)
        grid = lambda meta: (triton.cdiv(seq_len, meta["BLOCK_M"]), q.shape[0], q.shape[2] + k.shape[2])
        torch.library.wrap_triton(_qk_norm_rope_backward_kernel)[grid](
            q,
            k,
            grad_q,
            grad_k,
            cos_2d,
            sin_2d,
            grad_q_out,
            grad_k_out,
            q.shape[2],
            k.shape[2],
            seq_len,
            head_dim,
            rotary_dim,
            eps,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            BLOCK_D=block_d,
        )
        return grad_q_out, grad_k_out
else:
    _triton_qk_norm_rope = None
    _triton_qk_norm_rope_backward_op = None


def _apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    half = rotary_dim // 2
    cos_half = cos[..., :half]
    sin_half = sin[..., :half]
    x1, x2, x_pass = x[..., :half], x[..., half:rotary_dim], x[..., rotary_dim:]
    return torch.cat((x1 * cos_half - x2 * sin_half, x2 * cos_half + x1 * sin_half, x_pass), dim=-1)


def _qk_norm_rope_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.ndim == 4 and cos.shape[1] != q.shape[1] and cos.shape[2] == q.shape[1]:
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)
    q_normed = rms_norm(q, None, eps)
    k_normed = rms_norm(k, None, eps)
    return _apply_partial_rope(q_normed, cos, sin), _apply_partial_rope(k_normed, cos, sin)


def _qk_norm_rope_setup(ctx, inputs, output) -> None:
    q, k, cos, sin, eps = inputs
    ctx.save_for_backward(q, k, cos, sin)
    ctx.eps = eps


def _norm_backward_no_weight(x: torch.Tensor, grad: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    grad_f = grad.float()
    inv = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    dot = (grad_f * x_f).mean(dim=-1, keepdim=True)
    return ((grad_f * inv) - (x_f * inv.pow(3) * dot)).to(x.dtype)


def _qk_norm_rope_backward(
    ctx,
    grad_q: torch.Tensor | None,
    grad_k: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
    q, k, cos, sin = ctx.saved_tensors
    if grad_q is None:
        grad_q = torch.zeros_like(q, memory_format=torch.contiguous_format)
    if grad_k is None:
        grad_k = torch.zeros_like(k, memory_format=torch.contiguous_format)
    if q.is_cuda:
        grad_q_in, grad_k_in = _triton_qk_norm_rope_backward_op(
            q,
            k,
            grad_q.contiguous(),
            grad_k.contiguous(),
            cos,
            sin,
            ctx.eps,
        )
    else:
        grad_q_norm, grad_k_norm = _apply_partial_rope(grad_q, cos, -sin), _apply_partial_rope(grad_k, cos, -sin)
        grad_q_in = _norm_backward_no_weight(q, grad_q_norm, ctx.eps)
        grad_k_in = _norm_backward_no_weight(k, grad_k_norm, ctx.eps)
    return grad_q_in, grad_k_in, None, None, None


if triton is not None:
    torch.library.register_autograd("simple_lm::qk_norm_rope", _qk_norm_rope_backward, setup_context=_qk_norm_rope_setup)


def qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    backend: str = "torch",
) -> tuple[torch.Tensor, torch.Tensor]:
    if backend == "torch":
        return _qk_norm_rope_torch(q, k, cos, sin, eps)
    if backend == "triton":
        if q.is_cuda:
            _require_triton("rope_backend=triton")
            return _triton_qk_norm_rope(q, k, cos, sin, eps)
        return _qk_norm_rope_torch(q, k, cos, sin, eps)
    raise RuntimeError(f"unsupported QK norm RoPE backend {backend!r}")


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
            "Run `uv sync --locked` in the CUDA environment."
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


@torch._dynamo.disable
def _liger_fused_linear_cross_entropy(
    weight: torch.Tensor,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return _liger_fused_linear_ce()(weight, hidden, targets, bias=bias)


def chunked_linear_cross_entropy(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    targets: torch.Tensor,
    chunk_size: int,
    backend: str = "torch",
) -> torch.Tensor:
    if backend == "liger":
        _require_liger("loss_backend=liger")
        hidden_flat = hidden.reshape(-1, hidden.size(-1))
        targets_flat = targets.reshape(-1).contiguous()
        weight = lm_head.weight
        bias = lm_head.bias
        return _liger_fused_linear_cross_entropy(weight, hidden_flat, targets_flat, bias)
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
    if precision == "bf16":
        for param in model.parameters():
            if param.is_floating_point():
                param.data = param.data.to(dtype=torch.bfloat16)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(dtype=torch.bfloat16)
        return model
    if precision in {"fp32_test", "bf16_test"} and allow_torch_backend:
        return model
    raise RuntimeError(f"unsupported precision mode {precision!r}; the training fast path uses bf16")
