from __future__ import annotations

from pathlib import Path

import pytest

from conftest import requires_cuda, requires_torch, torch


@requires_torch
def test_linear_cross_entropy_torch_path_matches_dense() -> None:
    import torch.nn.functional as F

    from config import resolve_config
    from model import LinearCrossEntropyLoss

    torch.manual_seed(0)
    cfg = resolve_config(depth=2, vocab_size=33, sequence_len=7, precision="fp32_test", compile_model=False, loss_backend="torch")
    loss_fn = LinearCrossEntropyLoss(cfg.model)
    weight = torch.randn(33, 16, requires_grad=True)
    hidden = torch.randn(3, 7, 16, requires_grad=True)
    targets = torch.randint(0, 33, (3, 7))
    dense = F.cross_entropy(F.linear(hidden, weight).float().reshape(-1, 33), targets.reshape(-1))
    chunked = loss_fn(hidden, weight, None, targets)
    assert torch.allclose(chunked, dense, atol=1e-6)


@requires_torch
def test_bf16_precision_policy_leaves_modules_unchanged() -> None:
    import kernels

    model = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False))
    assert model[0].weight.dtype == torch.float32
    assert kernels.apply_precision_policy(model, "bf16") is model
    assert model[0].weight.dtype == torch.bfloat16


@requires_torch
def test_swiglu_torch_fallback_matches_baseline_and_backward() -> None:
    import torch.nn.functional as F

    import kernels

    torch.manual_seed(0)
    gate_up = torch.randn(2, 5, 64, requires_grad=True)
    ref_gate_up = gate_up.detach().clone().requires_grad_(True)
    grad = torch.randn(2, 5, 32)

    out = kernels.swiglu_triton(gate_up)
    gate, up = ref_gate_up.chunk(2, dim=-1)
    ref = F.silu(gate) * up
    out.backward(grad)
    ref.backward(grad)

    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.allclose(gate_up.grad, ref_gate_up.grad, atol=1e-6)


@pytest.mark.cuda
@requires_torch
@requires_cuda
def test_swiglu_triton_matches_baseline_and_backward() -> None:
    import torch.nn.functional as F

    import kernels

    torch.manual_seed(0)
    device = torch.device("cuda")
    fp32_gate_up = torch.randn(17, 130, device=device, dtype=torch.float32, requires_grad=True)
    ref_fp32_gate_up = fp32_gate_up.detach().clone().requires_grad_(True)
    fp32_grad = torch.randn(17, 65, device=device)

    fp32_out = kernels.swiglu_triton(fp32_gate_up)
    fp32_gate, fp32_up = ref_fp32_gate_up.chunk(2, dim=-1)
    fp32_ref = F.silu(fp32_gate) * fp32_up
    fp32_out.backward(fp32_grad)
    fp32_ref.backward(fp32_grad)

    assert torch.allclose(fp32_out, fp32_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(fp32_gate_up.grad, ref_fp32_gate_up.grad, atol=1e-5, rtol=1e-5)

    gate_up = torch.randn(128, 4096, device=device, dtype=torch.bfloat16, requires_grad=True)
    ref_gate_up = gate_up.detach().clone().requires_grad_(True)
    grad = torch.randn(128, 2048, device=device, dtype=torch.bfloat16)

    out = kernels.swiglu_triton(gate_up)
    gate, up = ref_gate_up.chunk(2, dim=-1)
    ref = F.silu(gate) * up
    out.backward(grad)
    ref.backward(grad)
    torch.cuda.synchronize()

    assert torch.allclose(out.float(), ref.float(), atol=8e-2, rtol=8e-2)
    assert torch.allclose(gate_up.grad.float(), ref_gate_up.grad.float(), atol=8e-2, rtol=8e-2)


@requires_torch
def test_flash_attention_casts_qkv_to_bfloat16() -> None:
    import kernels

    calls = []

    def fake_flash_attn(q, k, v, dropout_p, causal, window_size):
        calls.append((q.dtype, k.dtype, v.dtype, dropout_p, causal, window_size))
        return q

    original_flash_attn = kernels._FLASH_ATTN
    kernels._FLASH_ATTN = fake_flash_attn
    try:
        q = torch.randn(2, 4, 3, 8, dtype=torch.float32)
        k = torch.randn(2, 4, 1, 8, dtype=torch.float32)
        v = torch.randn(2, 4, 1, 8, dtype=torch.float32)
        out = kernels.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
            window_size=3,
            backend="flash_attn_2",
        )
    finally:
        kernels._FLASH_ATTN = original_flash_attn

    assert out.dtype == torch.bfloat16
    assert calls == [(torch.bfloat16, torch.bfloat16, torch.bfloat16, 0.0, True, (2, 0))]


@requires_torch
def test_qk_norm_rope_torch_matches_reference_and_backward() -> None:
    import kernels
    from config import resolve_config
    from model import QKNormRoPE

    torch.manual_seed(0)
    cfg = resolve_config(depth=2, vocab_size=128, precision="fp32_test", compile_model=False, rope_backend="torch")
    qk_norm_rope = QKNormRoPE(cfg.model)
    q = torch.randn(2, 5, 3, 8, requires_grad=True)
    k = torch.randn(2, 5, 2, 8, requires_grad=True)
    freqs = torch.randn(5, 2)
    cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)[None, :, None, :]
    sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)[None, :, None, :]

    q_out, k_out = qk_norm_rope(q, k, cos.transpose(1, 2), sin.transpose(1, 2))

    def ref(x: torch.Tensor) -> torch.Tensor:
        x_norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        x1, x2, x_pass = x_norm[..., :2], x_norm[..., 2:4], x_norm[..., 4:]
        c = cos[..., :2]
        s = sin[..., :2]
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s, x_pass), dim=-1)

    q_ref = ref(q)
    k_ref = ref(k)
    assert torch.allclose(q_out, q_ref, atol=1e-6)
    assert torch.allclose(k_out, k_ref, atol=1e-6)
    q_triton_cpu, k_triton_cpu = kernels.qk_norm_rope_triton(q, k, cos, sin, 1e-6)
    assert torch.allclose(q_triton_cpu, q_ref, atol=1e-6)
    assert torch.allclose(k_triton_cpu, k_ref, atol=1e-6)

    dq = torch.randn_like(q_out)
    dk = torch.randn_like(k_out)
    loss = (q_out * dq).sum() + (k_out * dk).sum()
    ref_loss = (q_ref * dq).sum() + (k_ref * dk).sum()
    loss.backward()
    q_grad, k_grad = q.grad.clone(), k.grad.clone()
    q.grad = None
    k.grad = None
    ref_loss.backward()
    assert torch.allclose(q_grad, q.grad, atol=1e-6)
    assert torch.allclose(k_grad, k.grad, atol=1e-6)


@requires_torch
def test_kernel_resolution_and_hashing() -> None:
    from kernels import resolve_kernel_backends
    from repro import hash_directory

    info = resolve_kernel_backends(
        "torch",
        "fp32_test",
        False,
        loss_backend="triton",
        allow_torch_backend=True,
    )
    assert info.actual_attention_backend in {"flash_attn_2", "torch_sdpa"}
    assert info.actual_mlp_backend == "torch"
    assert info.actual_loss_backend in {"triton_fused_linear_ce", "torch"}
    assert info.actual_rope_backend == "torch"
    assert isinstance(info.triton_available, bool)
    rope_info = resolve_kernel_backends(
        "torch",
        "fp32_test",
        False,
        loss_backend="triton",
        rope_backend="triton",
        allow_torch_backend=True,
    )
    assert rope_info.actual_loss_backend in {"triton_fused_linear_ce", "torch"}
    assert rope_info.actual_rope_backend in {"triton", "torch"}
    mlp_info = resolve_kernel_backends(
        "torch",
        "fp32_test",
        False,
        mlp_backend="triton",
        allow_torch_backend=True,
    )
    assert mlp_info.actual_mlp_backend in {"triton_swiglu", "torch"}
    assert len(hash_directory(Path.cwd())) == 64
