from __future__ import annotations

from pathlib import Path

from config import resolve_config
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
def test_fp8_policy_keeps_lm_head_bf16_linear() -> None:
    import fp8
    import kernels
    from model import LanguageModel

    cfg = resolve_config(
        depth=2,
        vocab_size=128,
        sequence_len=8,
        device_batch_size=2,
        precision="fp8",
        compile_model=False,
        norm_backend="torch",
        loss_backend="torch",
    )
    model = LanguageModel(cfg.model)
    original_supported = fp8.fp8_cuda_supported
    fp8.fp8_cuda_supported = lambda: True
    try:
        model = kernels.apply_precision_policy(model, "fp8")
    finally:
        fp8.fp8_cuda_supported = original_supported

    assert isinstance(model.blocks[0].attn.qkv_proj, fp8.Float8Linear)
    assert not isinstance(model.lm_head, fp8.Float8Linear)


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
def test_kernel_resolution_and_hashing() -> None:
    from kernels import resolve_kernel_backends
    from repro import hash_directory

    info = resolve_kernel_backends("torch", "fp32_test", False, norm_backend="torch", loss_backend="liger", allow_torch_backend=True)
    assert info.actual_attention_backend in {"flash_attn_2", "torch_sdpa"}
    assert info.actual_norm_backend == "torch"
    assert info.actual_loss_backend in {"liger_fused_linear_ce", "torch"}
    assert isinstance(info.liger_available, bool)
    assert len(hash_directory(Path.cwd())) == 64
