from __future__ import annotations

from pathlib import Path

from conftest import requires_torch, torch


@requires_torch
def test_chunked_linear_cross_entropy_matches_dense() -> None:
    import torch.nn.functional as F

    import kernels

    torch.manual_seed(0)
    head = torch.nn.Linear(16, 33, bias=False)
    hidden = torch.randn(3, 7, 16, requires_grad=True)
    targets = torch.randint(0, 33, (3, 7))
    dense = F.cross_entropy(head(hidden).float().reshape(-1, 33), targets.reshape(-1))
    chunked = kernels.chunked_linear_cross_entropy(hidden, head, targets, chunk_size=5)
    assert torch.allclose(chunked, dense, atol=1e-6)


@requires_torch
def test_bf16_precision_policy_leaves_modules_unchanged() -> None:
    import kernels

    model = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False))
    assert model[0].weight.dtype == torch.float32
    assert kernels.apply_precision_policy(model, "bf16") is model
    assert model[0].weight.dtype == torch.bfloat16


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

    torch.manual_seed(0)
    q = torch.randn(2, 5, 3, 8, requires_grad=True)
    k = torch.randn(2, 5, 2, 8, requires_grad=True)
    freqs = torch.randn(5, 2)
    cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)[None, :, None, :]
    sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)[None, :, None, :]

    q_out, k_out = kernels.qk_norm_rope(q, k, cos, sin, 1e-6, backend="torch")

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
    q_triton_cpu, k_triton_cpu = kernels.qk_norm_rope(q, k, cos, sin, 1e-6, backend="triton")
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
        loss_backend="liger",
        allow_torch_backend=True,
    )
    assert info.actual_attention_backend in {"flash_attn_2", "torch_sdpa"}
    assert info.actual_norm_backend == "torch"
    assert info.actual_mlp_backend == "torch"
    assert info.actual_loss_backend in {"liger_fused_linear_ce", "torch"}
    assert info.actual_rope_backend == "torch"
    assert isinstance(info.liger_available, bool)
    assert isinstance(info.triton_available, bool)
    rope_info = resolve_kernel_backends(
        "torch",
        "fp32_test",
        False,
        loss_backend="liger",
        rope_backend="triton",
        allow_torch_backend=True,
    )
    assert rope_info.actual_loss_backend in {"liger_fused_linear_ce", "torch"}
    assert rope_info.actual_rope_backend in {"triton", "torch"}
    assert len(hash_directory(Path.cwd())) == 64
