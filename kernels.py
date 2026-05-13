from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from typing import Any

import torch
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
    actual_mlp_backend: str
    actual_loss_backend: str
    actual_rope_backend: str
    torch_compile: bool
    torch_compile_mode: str
    torch_compile_capture_scalar_outputs: bool
    precision: str
    flash_attn_available: bool
    triton_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FLASH_ATTN = None


def triton_available() -> bool:
    return triton is not None


def _require_triton(backend: str) -> None:
    if not triton_available():
        raise RuntimeError(
            f"{backend} requires Triton. Add it with `uv sync --locked` "
            "in the CUDA environment."
        )


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
    *,
    mlp_backend: str = "torch",
) -> KernelInfo:
    requested = requested.lower()
    if requested != "torch":
        raise ValueError(f"unknown kernel backend {requested!r}")
    mlp_backend = mlp_backend.lower()
    if mlp_backend not in {"torch", "triton"}:
        raise ValueError(f"unknown MLP backend {mlp_backend!r}")
    loss_backend = loss_backend.lower()
    if loss_backend not in {"torch", "triton"}:
        raise ValueError(f"unknown loss backend {loss_backend!r}")
    rope_backend = rope_backend.lower()
    if rope_backend not in {"torch", "triton"}:
        raise ValueError(f"unknown RoPE backend {rope_backend!r}")
    flash_attn = _import_flash_attn()
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
    if mlp_backend == "triton" and not has_triton and not allow_torch_backend:
        raise RuntimeError(
            "mlp_backend=triton requires Triton. "
            "Run `uv sync --locked` in the CUDA environment."
        )
    if loss_backend == "triton" and not has_triton and not allow_torch_backend:
        raise RuntimeError(
            "loss_backend=triton requires Triton. "
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
        actual_mlp_backend="triton_swiglu" if mlp_backend == "triton" and has_triton else "torch",
        actual_loss_backend="triton_fused_linear_ce" if loss_backend == "triton" and has_triton else "torch",
        actual_rope_backend="triton" if rope_backend == "triton" and has_triton else "torch",
        torch_compile=compile_model,
        torch_compile_mode=compile_mode,
        torch_compile_capture_scalar_outputs=compile_capture_scalar_outputs,
        precision=precision,
        flash_attn_available=bool(flash_attn),
        triton_available=has_triton,
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return y if weight is None else y * weight


def _swiglu_torch(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


if triton is not None:
    _LINEAR_CE_MAX_BLOCK_SIZE = 32768
    _LINEAR_CE_MIN_CHUNK_SIZE = 2048
    _SWIGLU_MAX_BLOCK_SIZE = 8192
    _SWIGLU_CONFIGS = [
        triton.Config({"BLOCK_M": block_m}, num_warps=num_warps, num_stages=num_stages)
        for block_m in (1, 2, 4)
        for num_warps in (4, 8, 16)
        for num_stages in (3, 4)
    ]
    _LINEAR_CE_CONFIGS = [
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in (2, 4, 8, 16, 32)
        for num_stages in (3, 4, 5)
    ]

    # Autotune replays the kernel; this kernel overwrites logits with gradients.
    @triton.autotune(
        configs=_LINEAR_CE_CONFIGS,
        key=["N_ROWS", "n_cols", "BLOCK_SIZE"],
        restore_value=["logits"],
    )
    @triton.jit
    def _linear_ce_kernel(
        logits,
        logits_stride: tl.constexpr,
        targets,
        loss_out,
        n_cols: tl.constexpr,
        n_rows: tl.constexpr,
        N_ROWS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        target = tl.load(targets + row)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        row_ptr = logits + row * logits_stride
        values = tl.load(row_ptr + offsets, mask=mask, other=float("-inf"))
        m = tl.max(values, axis=0)
        exp_values = tl.exp(values - m)
        d = tl.sum(exp_values, axis=0)
        lse = m + tl.log(d)
        target_logit = tl.load(row_ptr + target).to(tl.float32)
        tl.store(loss_out + row, (lse - target_logit) / n_rows)

        inv_d = 1.0 / d
        grad = exp_values * inv_d
        grad = tl.where(offsets == target, grad - 1.0, grad) / n_rows
        tl.store(row_ptr + offsets, grad, mask=mask)

    @torch.library.triton_op("hackable_lm::linear_ce", mutates_args={"logits"})
    def _triton_linear_ce(logits: torch.Tensor, targets: torch.Tensor, n_tokens: int) -> torch.Tensor:
        loss_1d = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)
        vocab_size = logits.shape[1]
        block_size = min(_LINEAR_CE_MAX_BLOCK_SIZE, triton.next_power_of_2(vocab_size))
        grid = (logits.shape[0],)
        torch.library.wrap_triton(_linear_ce_kernel)[grid](
            logits,
            logits.stride(0),
            targets,
            loss_1d,
            vocab_size,
            n_tokens,
            N_ROWS=logits.shape[0],
            BLOCK_SIZE=block_size,
        )
        return loss_1d

    @triton.autotune(
        configs=_SWIGLU_CONFIGS,
        key=["N_ROWS", "N_COLS", "BLOCK_N"],
    )
    @triton.jit
    def _swiglu_kernel(
        gate_up,
        out,
        N_ROWS: tl.constexpr,
        N_COLS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < N_ROWS) & (cols[None, :] < N_COLS)
        row_offsets = rows[:, None] * (2 * N_COLS)
        gate = tl.load(gate_up + row_offsets + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        up = tl.load(gate_up + row_offsets + N_COLS + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        tl.store(out + rows[:, None] * N_COLS + cols[None, :], gate * sigmoid * up, mask=mask)

    @triton.autotune(
        configs=_SWIGLU_CONFIGS,
        key=["N_ROWS", "N_COLS", "BLOCK_N"],
    )
    @triton.jit
    def _swiglu_backward_kernel(
        gate_up,
        grad_out,
        grad_gate_up,
        N_ROWS: tl.constexpr,
        N_COLS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < N_ROWS) & (cols[None, :] < N_COLS)
        gate_up_offsets = rows[:, None] * (2 * N_COLS)
        out_offsets = rows[:, None] * N_COLS + cols[None, :]
        gate = tl.load(gate_up + gate_up_offsets + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        up = tl.load(gate_up + gate_up_offsets + N_COLS + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_out + out_offsets, mask=mask, other=0.0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        silu = gate * sigmoid
        dsilu = sigmoid * (1.0 + gate * (1.0 - sigmoid))
        tl.store(grad_gate_up + gate_up_offsets + cols[None, :], grad * up * dsilu, mask=mask)
        tl.store(grad_gate_up + gate_up_offsets + N_COLS + cols[None, :], grad * silu, mask=mask)

    @torch.library.triton_op("hackable_lm::swiglu", mutates_args={})
    def _triton_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
        hidden = gate_up.shape[-1] // 2
        rows = gate_up.numel() // (2 * hidden)
        out = torch.empty((*gate_up.shape[:-1], hidden), dtype=gate_up.dtype, device=gate_up.device)
        block_n = min(_SWIGLU_MAX_BLOCK_SIZE, triton.next_power_of_2(hidden))
        grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]),)
        torch.library.wrap_triton(_swiglu_kernel)[grid](
            gate_up,
            out,
            rows,
            hidden,
            BLOCK_N=block_n,
        )
        return out

    @torch.library.triton_op("hackable_lm::swiglu_backward", mutates_args={})
    def _triton_swiglu_backward_op(gate_up: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
        hidden = grad_out.shape[-1]
        rows = grad_out.numel() // hidden
        grad_gate_up = torch.empty_like(gate_up, memory_format=torch.contiguous_format)
        block_n = min(_SWIGLU_MAX_BLOCK_SIZE, triton.next_power_of_2(hidden))
        grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]),)
        torch.library.wrap_triton(_swiglu_backward_kernel)[grid](
            gate_up,
            grad_out,
            grad_gate_up,
            rows,
            hidden,
            BLOCK_N=block_n,
        )
        return grad_gate_up


    _QK_NORM_ROPE_CONFIGS = [
        triton.Config({"BLOCK_M": block_m}, num_warps=num_warps, num_stages=num_stages)
        for block_m in (1, 2, 4, 8, 16)
        for num_warps in (1, 2, 4, 8)
        for num_stages in (3, 4)
        if not (block_m == 1 and num_warps == 8)
        if not (block_m >= 8 and num_warps == 1)
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

    @torch.library.triton_op("hackable_lm::qk_norm_rope", mutates_args={})
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

    @torch.library.triton_op("hackable_lm::qk_norm_rope_backward", mutates_args={})
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
    _LINEAR_CE_MAX_BLOCK_SIZE = 32768
    _LINEAR_CE_MIN_CHUNK_SIZE = 2048
    _SWIGLU_MAX_BLOCK_SIZE = 8192
    _triton_swiglu = None
    _triton_swiglu_backward_op = None
    _triton_linear_ce = None
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


def _swiglu_setup(ctx, inputs, output) -> None:
    (gate_up,) = inputs
    ctx.save_for_backward(gate_up)


def _swiglu_backward(ctx, grad_out: torch.Tensor | None) -> tuple[torch.Tensor | None]:
    if grad_out is None:
        return (None,)
    (gate_up,) = ctx.saved_tensors
    if gate_up.is_cuda:
        return (_triton_swiglu_backward_op(gate_up, grad_out.contiguous()),)
    gate, up = gate_up.chunk(2, dim=-1)
    sigmoid = torch.sigmoid(gate)
    silu = gate * sigmoid
    dsilu = sigmoid * (1.0 + gate * (1.0 - sigmoid))
    return (torch.cat((grad_out * up * dsilu, grad_out * silu), dim=-1),)


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
    torch.library.register_autograd("hackable_lm::swiglu", _swiglu_backward, setup_context=_swiglu_setup)
    torch.library.register_autograd("hackable_lm::qk_norm_rope", _qk_norm_rope_backward, setup_context=_qk_norm_rope_setup)


def swiglu_triton(gate_up: torch.Tensor) -> torch.Tensor:
    if gate_up.shape[-1] % 2:
        raise RuntimeError(f"SwiGLU input last dimension must be even, got {gate_up.shape[-1]}")
    hidden = gate_up.shape[-1] // 2
    if hidden > _SWIGLU_MAX_BLOCK_SIZE:
        raise RuntimeError(f"SwiGLU hidden dimension {hidden} exceeds Triton kernel limit {_SWIGLU_MAX_BLOCK_SIZE}")
    if gate_up.is_cuda:
        _require_triton("mlp_backend=triton")
        return _triton_swiglu(gate_up.contiguous())
    return _swiglu_torch(gate_up)


def qk_norm_rope_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.is_cuda:
        _require_triton("rope_backend=triton")
        return _triton_qk_norm_rope(q, k, cos, sin, eps)
    return _qk_norm_rope_torch(q, k, cos, sin, eps)


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


def _scale_saved_grad(grad: torch.Tensor | None, scale: torch.Tensor) -> torch.Tensor | None:
    if grad is None:
        return None
    return grad * scale.to(dtype=grad.dtype)


def _linear_ce_heuristic_chunk_size(n_tokens: int, hidden_size: int, vocab_size: int) -> int:
    inc_factor = triton.cdiv(vocab_size, hidden_size)
    chunk_size = triton.next_power_of_2(triton.cdiv(n_tokens, inc_factor))
    return min(n_tokens, max(chunk_size, _LINEAR_CE_MIN_CHUNK_SIZE))


class _TritonFusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        bias: torch.Tensor | None,
        requested_chunk_size: int | None,
    ) -> torch.Tensor:
        _require_triton("loss_backend=triton")
        if requested_chunk_size is not None and requested_chunk_size < 0:
            raise RuntimeError(f"loss_chunk_size must be nonnegative, got {requested_chunk_size}")
        hidden = hidden.contiguous()
        weight = weight.contiguous()
        targets = targets.contiguous()
        n_tokens, hidden_size = hidden.shape
        vocab_size = weight.shape[0]
        if requested_chunk_size:
            chunk_size = min(n_tokens, max(requested_chunk_size, _LINEAR_CE_MIN_CHUNK_SIZE))
        else:
            chunk_size = _linear_ce_heuristic_chunk_size(n_tokens, hidden_size, vocab_size)
        num_chunks = triton.cdiv(n_tokens, chunk_size)

        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros_like(weight) if weight.requires_grad else None
        grad_bias = torch.zeros_like(bias) if bias is not None and bias.requires_grad else None
        loss_1d = torch.empty(n_tokens, dtype=torch.float32, device=hidden.device)

        for chunk_id in range(num_chunks):
            start = chunk_id * chunk_size
            end = min(start + chunk_size, n_tokens)
            hidden_chunk = hidden[start:end]
            logits = hidden_chunk @ weight.t()
            if bias is not None:
                logits = logits + bias
            logits = logits.contiguous()
            loss_1d[start:end] = _triton_linear_ce(logits, targets[start:end], n_tokens)
            grad_logits = logits
            if hidden.requires_grad:
                grad_hidden[start:end] = grad_logits @ weight
            if grad_weight is not None:
                grad_weight += torch.mm(grad_logits.t(), hidden_chunk).to(grad_weight.dtype)
            if grad_bias is not None:
                grad_bias += grad_logits.sum(dim=0).to(grad_bias.dtype)

        ctx.save_for_backward(
            grad_hidden.detach() if hidden.requires_grad else None,
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if grad_bias is not None else None,
        )
        return loss_1d.sum()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None, torch.Tensor | None, None]:
        grad_hidden, grad_weight, grad_bias = ctx.saved_tensors
        return (
            _scale_saved_grad(grad_hidden, grad_output) if ctx.needs_input_grad[0] else None,
            _scale_saved_grad(grad_weight, grad_output) if ctx.needs_input_grad[1] else None,
            None,
            _scale_saved_grad(grad_bias, grad_output) if ctx.needs_input_grad[3] else None,
            None,
        )


def _triton_fused_linear_cross_entropy(
    weight: torch.Tensor,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    chunk_size: int | None,
) -> torch.Tensor:
    if not hidden.is_cuda:
        return F.cross_entropy(F.linear(hidden, weight, bias).float(), targets)
    return _TritonFusedLinearCrossEntropy.apply(hidden, weight, targets, bias, chunk_size)


def fused_linear_cross_entropy_with_weight(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    targets: torch.Tensor,
    backend: str,
    chunk_size: int | None = None,
) -> torch.Tensor:
    hidden_flat = hidden.reshape(-1, hidden.size(-1))
    targets_flat = targets.reshape(-1).contiguous()
    if backend == "triton":
        return _triton_fused_linear_cross_entropy(weight, hidden_flat, targets_flat, bias, chunk_size)
    raise RuntimeError(f"unsupported fused linear cross entropy backend {backend!r}")


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
