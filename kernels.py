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
_LIGER_RMS_NORM = None
_LIGER_SWIGLU = None


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


def _liger_rms_norm():
    global _LIGER_RMS_NORM
    if _LIGER_RMS_NORM is None:
        from liger_kernel.transformers.functional import liger_rms_norm

        _LIGER_RMS_NORM = liger_rms_norm
    return _LIGER_RMS_NORM


def _liger_swiglu():
    global _LIGER_SWIGLU
    if _LIGER_SWIGLU is None:
        from liger_kernel.transformers.functional import liger_swiglu

        _LIGER_SWIGLU = liger_swiglu
    return _LIGER_SWIGLU


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
    mlp_backend: str = "torch",
    loss_backend: str = "torch",
    rope_backend: str = "torch",
    allow_torch_backend: bool = False,
) -> KernelInfo:
    requested = requested.lower()
    if requested != "torch":
        raise ValueError(f"unknown kernel backend {requested!r}")
    norm_backend = norm_backend.lower()
    if norm_backend not in {"torch", "liger"}:
        raise ValueError(f"unknown norm backend {norm_backend!r}")
    mlp_backend = mlp_backend.lower()
    if mlp_backend not in {"torch", "liger"}:
        raise ValueError(f"unknown MLP backend {mlp_backend!r}")
    loss_backend = loss_backend.lower()
    if loss_backend not in {"torch", "liger"}:
        raise ValueError(f"unknown loss backend {loss_backend!r}")
    rope_backend = rope_backend.lower()
    if rope_backend not in {"torch", "triton_qk_norm_rope"}:
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
    if (
        norm_backend == "liger" or mlp_backend == "liger" or loss_backend == "liger"
    ) and not has_liger and not allow_torch_backend:
        raise RuntimeError(
            "A Liger backend was requested but liger-kernel is not importable. "
            "Run `uv sync --locked` in the CUDA environment."
        )
    if rope_backend == "triton_qk_norm_rope" and not has_triton and not allow_torch_backend:
        raise RuntimeError(
            "rope_backend=triton_qk_norm_rope requires Triton. "
            "Run `uv sync --locked` in the CUDA environment."
        )
    return KernelInfo(
        requested_backend=requested,
        actual_attention_backend="flash_attn_2" if flash_attn else "torch_sdpa",
        actual_norm_backend="liger_rms_norm" if norm_backend == "liger" and has_liger else "torch",
        actual_mlp_backend="liger_swiglu" if mlp_backend == "liger" and has_liger else "torch",
        actual_loss_backend="liger_fused_linear_ce" if loss_backend == "liger" and has_liger else "torch",
        actual_rope_backend="triton_qk_norm_rope" if rope_backend == "triton_qk_norm_rope" and has_triton else "torch",
        torch_compile=compile_model,
        torch_compile_mode=compile_mode,
        torch_compile_capture_scalar_outputs=compile_capture_scalar_outputs,
        precision=precision,
        flash_attn_available=bool(flash_attn),
        liger_available=has_liger,
        triton_available=has_triton,
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float, backend: str = "torch") -> torch.Tensor:
    if backend == "liger":
        _require_liger("norm_backend=liger")
        return _liger_rms_norm()(x, weight, eps, in_place=False)
    if backend != "torch":
        raise RuntimeError(f"unsupported RMSNorm backend {backend!r}")
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return y if weight is None else y * weight


def swiglu(gate: torch.Tensor, up: torch.Tensor, backend: str = "torch") -> torch.Tensor:
    if backend == "liger":
        _require_liger("mlp_backend=liger")
        return _liger_swiglu()(gate, up)
    if backend != "torch":
        raise RuntimeError(f"unsupported SwiGLU backend {backend!r}")
    return F.silu(gate) * up


if triton is not None:
    _QK_NORM_ROPE_CONFIGS = [
        triton.Config({"BLOCK_M": 1}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 2}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 4}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 4}, num_warps=8, num_stages=3),
    ]

    @triton.autotune(
        configs=_QK_NORM_ROPE_CONFIGS,
        key=["T", "N_Q", "N_K", "HEAD_DIM", "ROTARY_DIM"],
    )
    @triton.jit
    def _qk_norm_rope_fwd_kernel(
        Q,
        K,
        Q_OUT,
        K_OUT,
        INV_Q,
        INV_K,
        COS,
        SIN,
        Q_SB: tl.constexpr,
        Q_ST: tl.constexpr,
        Q_SH: tl.constexpr,
        Q_SD: tl.constexpr,
        K_SB: tl.constexpr,
        K_ST: tl.constexpr,
        K_SH: tl.constexpr,
        K_SD: tl.constexpr,
        QO_SB: tl.constexpr,
        QO_ST: tl.constexpr,
        QO_SH: tl.constexpr,
        QO_SD: tl.constexpr,
        KO_SB: tl.constexpr,
        KO_ST: tl.constexpr,
        KO_SH: tl.constexpr,
        KO_SD: tl.constexpr,
        EPS: tl.constexpr,
        T: tl.constexpr,
        N_Q: tl.constexpr,
        N_K: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ) -> None:
        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_hk = tl.program_id(2)
        offs_t = pid_t * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        half: tl.constexpr = ROTARY_DIM // 2

        is_q = pid_hk < N_Q
        head = tl.where(is_q, pid_hk, pid_hk - N_Q)
        q_ptr = Q + pid_b * Q_SB + offs_t[:, None] * Q_ST + head * Q_SH + offs_d[None, :] * Q_SD
        k_ptr = K + pid_b * K_SB + offs_t[:, None] * K_ST + head * K_SH + offs_d[None, :] * K_SD
        x_ptr = tl.where(is_q, q_ptr, k_ptr)
        mask = (offs_t[:, None] < T) & (offs_d[None, :] < HEAD_DIM)

        x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, axis=1) / HEAD_DIM + EPS)
        z = x * inv[:, None]

        pair_d = tl.where(offs_d < half, offs_d + half, offs_d - half)
        q_pair_ptr = Q + pid_b * Q_SB + offs_t[:, None] * Q_ST + head * Q_SH + pair_d[None, :] * Q_SD
        k_pair_ptr = K + pid_b * K_SB + offs_t[:, None] * K_ST + head * K_SH + pair_d[None, :] * K_SD
        pair_ptr = tl.where(is_q, q_pair_ptr, k_pair_ptr)
        pair = tl.load(pair_ptr, mask=mask & (offs_d[None, :] < ROTARY_DIM), other=0.0).to(tl.float32)
        pair = pair * inv[:, None]

        rot_i = tl.where(offs_d < half, offs_d, offs_d - half)
        rot_mask = (offs_t[:, None] < T) & (offs_d[None, :] < ROTARY_DIM)
        c = tl.load(COS + offs_t[:, None] * half + rot_i[None, :], mask=rot_mask, other=1.0).to(tl.float32)
        s = tl.load(SIN + offs_t[:, None] * half + rot_i[None, :], mask=rot_mask, other=0.0).to(tl.float32)

        y_lo = z * c - pair * s
        y_hi = z * c + pair * s
        y = tl.where(offs_d[None, :] < half, y_lo, tl.where(offs_d[None, :] < ROTARY_DIM, y_hi, z))

        q_out_ptr = Q_OUT + pid_b * QO_SB + offs_t[:, None] * QO_ST + head * QO_SH + offs_d[None, :] * QO_SD
        k_out_ptr = K_OUT + pid_b * KO_SB + offs_t[:, None] * KO_ST + head * KO_SH + offs_d[None, :] * KO_SD
        out_ptr = tl.where(is_q, q_out_ptr, k_out_ptr)
        tl.store(out_ptr, y, mask=mask)

        inv_q_ptr = INV_Q + (pid_b * T + offs_t) * N_Q + head
        inv_k_ptr = INV_K + (pid_b * T + offs_t) * N_K + head
        inv_ptr = tl.where(is_q, inv_q_ptr, inv_k_ptr)
        tl.store(inv_ptr, inv, mask=offs_t < T)

    @triton.autotune(
        configs=_QK_NORM_ROPE_CONFIGS,
        key=["T", "N_Q", "N_K", "HEAD_DIM", "ROTARY_DIM"],
    )
    @triton.jit
    def _qk_norm_rope_bwd_kernel(
        DQ_OUT,
        DK_OUT,
        Q,
        K,
        DQ,
        DK,
        INV_Q,
        INV_K,
        COS,
        SIN,
        DQO_SB: tl.constexpr,
        DQO_ST: tl.constexpr,
        DQO_SH: tl.constexpr,
        DQO_SD: tl.constexpr,
        DKO_SB: tl.constexpr,
        DKO_ST: tl.constexpr,
        DKO_SH: tl.constexpr,
        DKO_SD: tl.constexpr,
        Q_SB: tl.constexpr,
        Q_ST: tl.constexpr,
        Q_SH: tl.constexpr,
        Q_SD: tl.constexpr,
        K_SB: tl.constexpr,
        K_ST: tl.constexpr,
        K_SH: tl.constexpr,
        K_SD: tl.constexpr,
        DQ_SB: tl.constexpr,
        DQ_ST: tl.constexpr,
        DQ_SH: tl.constexpr,
        DQ_SD: tl.constexpr,
        DK_SB: tl.constexpr,
        DK_ST: tl.constexpr,
        DK_SH: tl.constexpr,
        DK_SD: tl.constexpr,
        T: tl.constexpr,
        N_Q: tl.constexpr,
        N_K: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ) -> None:
        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_hk = tl.program_id(2)
        offs_t = pid_t * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        half: tl.constexpr = ROTARY_DIM // 2

        is_q = pid_hk < N_Q
        head = tl.where(is_q, pid_hk, pid_hk - N_Q)
        mask = (offs_t[:, None] < T) & (offs_d[None, :] < HEAD_DIM)

        dqo_ptr = DQ_OUT + pid_b * DQO_SB + offs_t[:, None] * DQO_ST + head * DQO_SH + offs_d[None, :] * DQO_SD
        dko_ptr = DK_OUT + pid_b * DKO_SB + offs_t[:, None] * DKO_ST + head * DKO_SH + offs_d[None, :] * DKO_SD
        q_ptr = Q + pid_b * Q_SB + offs_t[:, None] * Q_ST + head * Q_SH + offs_d[None, :] * Q_SD
        k_ptr = K + pid_b * K_SB + offs_t[:, None] * K_ST + head * K_SH + offs_d[None, :] * K_SD
        dy_ptr = tl.where(is_q, dqo_ptr, dko_ptr)
        x_ptr = tl.where(is_q, q_ptr, k_ptr)

        dy = tl.load(dy_ptr, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)

        pair_d = tl.where(offs_d < half, offs_d + half, offs_d - half)
        dqo_pair = DQ_OUT + pid_b * DQO_SB + offs_t[:, None] * DQO_ST + head * DQO_SH + pair_d[None, :] * DQO_SD
        dko_pair = DK_OUT + pid_b * DKO_SB + offs_t[:, None] * DKO_ST + head * DKO_SH + pair_d[None, :] * DKO_SD
        dy_pair = tl.load(tl.where(is_q, dqo_pair, dko_pair), mask=mask & (offs_d[None, :] < ROTARY_DIM), other=0.0).to(tl.float32)

        rot_i = tl.where(offs_d < half, offs_d, offs_d - half)
        rot_mask = (offs_t[:, None] < T) & (offs_d[None, :] < ROTARY_DIM)
        c = tl.load(COS + offs_t[:, None] * half + rot_i[None, :], mask=rot_mask, other=1.0).to(tl.float32)
        s = tl.load(SIN + offs_t[:, None] * half + rot_i[None, :], mask=rot_mask, other=0.0).to(tl.float32)

        g_lo = dy * c + dy_pair * s
        g_hi = dy * c - dy_pair * s
        g = tl.where(offs_d[None, :] < half, g_lo, tl.where(offs_d[None, :] < ROTARY_DIM, g_hi, dy))

        inv_q_ptr = INV_Q + (pid_b * T + offs_t) * N_Q + head
        inv_k_ptr = INV_K + (pid_b * T + offs_t) * N_K + head
        inv = tl.load(tl.where(is_q, inv_q_ptr, inv_k_ptr), mask=offs_t < T, other=0.0).to(tl.float32)
        z = x * inv[:, None]
        dot = tl.sum(g * z, axis=1) / HEAD_DIM
        dx = inv[:, None] * (g - z * dot[:, None])

        dq_ptr = DQ + pid_b * DQ_SB + offs_t[:, None] * DQ_ST + head * DQ_SH + offs_d[None, :] * DQ_SD
        dk_ptr = DK + pid_b * DK_SB + offs_t[:, None] * DK_ST + head * DK_SH + offs_d[None, :] * DK_SD
        tl.store(tl.where(is_q, dq_ptr, dk_ptr), dx, mask=mask)
else:
    _qk_norm_rope_fwd_kernel = None
    _qk_norm_rope_bwd_kernel = None


def _rope_half_tables(cos: torch.Tensor, sin: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.ndim == 4:
        if cos.shape[1] >= seq_len:
            cos = cos[0, :seq_len, 0]
            sin = sin[0, :seq_len, 0]
        elif cos.shape[2] >= seq_len:
            cos = cos[0, 0, :seq_len]
            sin = sin[0, 0, :seq_len]
        else:
            raise RuntimeError(f"RoPE cache is shorter than sequence length {seq_len}")
    elif cos.ndim != 2:
        raise RuntimeError(f"unsupported RoPE table shape {tuple(cos.shape)}")
    rotary_dim = cos.shape[-1]
    if rotary_dim % 2 != 0:
        raise RuntimeError(f"RoPE rotary_dim must be even, got {rotary_dim}")
    half = rotary_dim // 2
    return cos[:, :half].contiguous(), sin[:, :half].contiguous()


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
    q_normed = rms_norm(q, None, eps, backend="torch")
    k_normed = rms_norm(k, None, eps, backend="torch")
    return _apply_partial_rope(q_normed, cos, sin), _apply_partial_rope(k_normed, cos, sin)


class _QKNormRoPE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        cos_half: torch.Tensor,
        sin_half: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_triton("rope_backend=triton_qk_norm_rope")
        if not q.is_cuda or not k.is_cuda:
            raise RuntimeError("rope_backend=triton_qk_norm_rope requires CUDA tensors")
        if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[-1] != k.shape[-1]:
            raise RuntimeError(f"Q/K shape mismatch: q={tuple(q.shape)}, k={tuple(k.shape)}")
        batch, seq_len, n_q, head_dim = q.shape
        n_k = k.shape[2]
        rotary_dim = cos_half.shape[-1] * 2
        if head_dim & (head_dim - 1):
            raise RuntimeError(f"Triton QK norm RoPE requires power-of-two head_dim, got {head_dim}")
        if rotary_dim > head_dim:
            raise RuntimeError(f"rotary_dim={rotary_dim} exceeds head_dim={head_dim}")

        q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
        k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
        inv_q = torch.empty((batch, seq_len, n_q), device=q.device, dtype=torch.float32)
        inv_k = torch.empty((batch, seq_len, n_k), device=k.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(seq_len, meta["BLOCK_M"]), batch, n_q + n_k)
        _qk_norm_rope_fwd_kernel[grid](
            q,
            k,
            q_out,
            k_out,
            inv_q,
            inv_k,
            cos_half,
            sin_half,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            q_out.stride(0),
            q_out.stride(1),
            q_out.stride(2),
            q_out.stride(3),
            k_out.stride(0),
            k_out.stride(1),
            k_out.stride(2),
            k_out.stride(3),
            float(eps),
            seq_len,
            n_q,
            n_k,
            HEAD_DIM=head_dim,
            ROTARY_DIM=rotary_dim,
            BLOCK_D=triton.next_power_of_2(head_dim),
        )
        ctx.save_for_backward(q, k, inv_q, inv_k, cos_half, sin_half)
        ctx.meta = (seq_len, n_q, n_k, head_dim, rotary_dim)
        return q_out, k_out

    @staticmethod
    def backward(ctx: Any, dq_out: torch.Tensor | None, dk_out: torch.Tensor | None) -> tuple[Any, ...]:
        q, k, inv_q, inv_k, cos_half, sin_half = ctx.saved_tensors
        seq_len, n_q, n_k, head_dim, rotary_dim = ctx.meta
        if dq_out is None:
            dq_out = torch.zeros_like(q, memory_format=torch.contiguous_format)
        if dk_out is None:
            dk_out = torch.zeros_like(k, memory_format=torch.contiguous_format)
        dq = torch.empty_like(q, memory_format=torch.contiguous_format)
        dk = torch.empty_like(k, memory_format=torch.contiguous_format)
        batch = q.shape[0]
        grid = lambda meta: (triton.cdiv(seq_len, meta["BLOCK_M"]), batch, n_q + n_k)
        _qk_norm_rope_bwd_kernel[grid](
            dq_out,
            dk_out,
            q,
            k,
            dq,
            dk,
            inv_q,
            inv_k,
            cos_half,
            sin_half,
            dq_out.stride(0),
            dq_out.stride(1),
            dq_out.stride(2),
            dq_out.stride(3),
            dk_out.stride(0),
            dk_out.stride(1),
            dk_out.stride(2),
            dk_out.stride(3),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            seq_len,
            n_q,
            n_k,
            HEAD_DIM=head_dim,
            ROTARY_DIM=rotary_dim,
            BLOCK_D=triton.next_power_of_2(head_dim),
        )
        return dq, dk, None, None, None


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
    if backend != "triton_qk_norm_rope":
        raise RuntimeError(f"unsupported QK norm RoPE backend {backend!r}")
    cos_half, sin_half = _rope_half_tables(cos, sin, q.shape[1])
    return _QKNormRoPE.apply(q, k, cos_half, sin_half, eps)


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
