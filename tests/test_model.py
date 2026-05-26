from __future__ import annotations

import pytest

from config import MODULE_BACKEND_FIELDS, resolve_config
from conftest import requires_torch, torch


TORCH_BACKENDS = {name: "torch" for name in MODULE_BACKEND_FIELDS}


@requires_torch
def test_model_forward_and_optimizer_grouping() -> None:
    from model import (
        AcceleratedModule,
        AttentionCore,
        GateUpProjection,
        LanguageModel,
        LMHeadProjection,
        OutputProjection,
        QKVProjection,
        DownProjection,
    )
    from optim import create_optimizer

    cfg = resolve_config(
        depth=5,
        vocab_size=128,
        sequence_len=8,
        attention_window=4,
        device_batch_size=2,
        precision="fp32_test",
        compile_model=False,
        **TORCH_BACKENDS,
    )
    model = LanguageModel(cfg.model)
    assert isinstance(model.blocks[0].attn.qkv, QKVProjection)
    assert isinstance(model.blocks[0].attn.attention, AttentionCore)
    assert isinstance(model.blocks[0].attn.out, OutputProjection)
    assert isinstance(model.blocks[0].mlp.gate_up, GateUpProjection)
    assert isinstance(model.blocks[0].mlp.down, DownProjection)
    assert isinstance(model.lm_head, LMHeadProjection)
    assert all(
        isinstance(module, AcceleratedModule)
        for module in (
            model.blocks[0].attn.qkv,
            model.blocks[0].attn.attention,
            model.blocks[0].attn.out,
            model.blocks[0].mlp.gate_up,
            model.blocks[0].mlp.down,
            model.lm_head,
        )
    )
    assert cfg.model.rope_fraction == 0.25
    assert model.blocks[0].attn.qkv.rope.rotary_dim == cfg.model.head_dim // 4
    mask = model.blocks[0].attn.attention._sliding_window_mask(cfg.sequence_len, torch.device("cpu"))
    assert model.blocks[3].attn.attention.attention_window is None
    assert model.blocks[-1].attn.attention.attention_window is None
    assert mask is not None
    assert bool(mask[0, 0, 7, 7])
    assert bool(mask[0, 0, 7, 4])
    assert not bool(mask[0, 0, 7, 3])
    x = torch.randint(0, cfg.model.vocab_size, (2, cfg.sequence_len))
    logits, loss = model(x, x)
    assert logits is None
    assert loss is not None and torch.isfinite(loss)
    logits, loss = model(x)
    assert logits.shape == (2, cfg.sequence_len, cfg.model.vocab_size)
    assert loss is None
    opt = create_optimizer(model, cfg)
    summary = opt.summary()
    assert summary["optimizer_kind"] == "muon_adamw"
    assert summary["muon_tensors"] == 7 * cfg.model.n_layer
    assert summary["adamw_tensors"] > 0
    split_groups = [group for group in opt.param_groups if group["kind"] == "muon" and group.get("split_sizes")]
    assert sorted(group["split_sizes"] for group in split_groups) == [
        (
            cfg.model.n_head * cfg.model.head_dim,
            cfg.model.n_kv_head * cfg.model.head_dim,
            cfg.model.n_kv_head * cfg.model.head_dim,
        ),
        (cfg.model.mlp_hidden, cfg.model.mlp_hidden),
    ]


@requires_torch
def test_model_rejects_invalid_kv_head_count() -> None:
    from model import LanguageModel

    cfg = resolve_config(
        depth=5,
        vocab_size=128,
        sequence_len=8,
        device_batch_size=2,
        precision="fp32_test",
        compile_model=False,
        **TORCH_BACKENDS,
    )
    cfg.model.n_kv_head = 2

    with pytest.raises(AssertionError):
        LanguageModel(cfg.model)


@requires_torch
def test_adamw_optimizer_mode_groups_all_trainable_params() -> None:
    from model import LanguageModel
    from optim import create_optimizer

    cfg = resolve_config(
        depth=2,
        vocab_size=128,
        sequence_len=8,
        attention_window=4,
        device_batch_size=2,
        precision="fp32_test",
        compile_model=False,
        optimizer="adamw",
        **TORCH_BACKENDS,
    )
    model = LanguageModel(cfg.model)
    opt = create_optimizer(model, cfg)
    summary = opt.summary()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert summary["optimizer_kind"] == "adamw"
    assert summary["muon_tensors"] == 0
    assert summary["muon_params"] == 0
    assert summary["adamw_params"] == trainable_params
    assert summary["matrix_params"] > 0
    assert opt.muon_lr is None
    assert opt.muon_momentum is None
    assert all(group["kind"] == "adamw" for group in opt.param_groups)

    matrix_param = dict(model.named_parameters())["blocks.0.attn.qkv.qkv_proj.weight"]
    before = matrix_param.detach().clone()
    matrix_param.grad = torch.ones_like(matrix_param)
    opt.step()
    assert not torch.allclose(matrix_param, before)


@requires_torch
def test_model_initialization_is_gpt_style() -> None:
    from model import LanguageModel

    torch.manual_seed(0)
    cfg = resolve_config(
        depth=6,
        vocab_size=512,
        sequence_len=8,
        device_batch_size=2,
        precision="fp32_test",
        compile_model=False,
        **TORCH_BACKENDS,
    )
    model = LanguageModel(cfg.model)
    base_std = 0.02
    residual_std = base_std / (2 * cfg.model.n_layer) ** 0.5

    def assert_std_close(param: torch.Tensor, expected: float) -> None:
        actual = float(param.detach().float().std(unbiased=False))
        assert abs(actual - expected) / expected < 0.12, (actual, expected)

    assert_std_close(model.tok_emb.weight, base_std)
    assert_std_close(model.lm_head.lm_head.weight, base_std)
    assert_std_close(model.blocks[0].attn.qkv.qkv_proj.weight, base_std)
    assert_std_close(model.blocks[0].mlp.gate_up.gate_up_proj.weight, base_std)
    assert_std_close(model.blocks[0].attn.out.o_proj.weight, residual_std)
    assert_std_close(model.blocks[0].mlp.down.down_proj.weight, residual_std)
