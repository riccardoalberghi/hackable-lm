from __future__ import annotations

from config import DEFAULTS, config_from_dict, layer_attention_window, resolve_config, scaling_params_for_depth


def test_config_derivation() -> None:
    cfg = resolve_config(depth=12, vocab_size=32768, sequence_len=128, device_batch_size=2, precision="fp32_test", compile_model=False)
    assert cfg.scaling_policy == "depth_simple"
    assert cfg.model.n_layer == 12
    assert cfg.model.n_head == 6
    assert cfg.model.n_embd == 768
    assert cfg.model.n_embd % DEFAULTS["head_dim"] == 0
    assert 3 * cfg.model.mlp_hidden == 8 * cfg.model.n_embd
    assert cfg.model.rope_fraction == 0.25
    assert cfg.scaling_params == scaling_params_for_depth(12, 32768)
    assert cfg.target_param_data_ratio == DEFAULTS["target_param_data_ratio"]
    assert cfg.target_param_data_ratio == 60
    assert DEFAULTS["sequence_len"] == 2048
    assert DEFAULTS["attention_window"] == 512
    assert DEFAULTS["attention_full_every"] == 4
    assert DEFAULTS["norm_backend"] == "torch"
    assert DEFAULTS["loss_backend"] == "liger"
    default_cfg = resolve_config(depth=2, vocab_size=128, precision="fp32_test", compile_model=False)
    assert default_cfg.sequence_len == 2048
    assert default_cfg.model.attention_window == 512
    assert default_cfg.model.attention_full_every == 4
    assert default_cfg.model.norm_backend == "torch"
    assert default_cfg.model.loss_backend == "liger"
    pattern = [layer_attention_window(i, 8, 512, 4) for i in range(8)]
    assert pattern == [512, 512, 512, None, 512, 512, 512, None]
    short_pattern = [layer_attention_window(i, 3, 512, 4) for i in range(3)]
    assert short_pattern == [512, 512, None]
    assert cfg.lr_scheduler == "wsd"
    assert cfg.warmup_ratio == DEFAULTS["warmup_ratio"]
    assert cfg.warmup_steps == max(1, round(DEFAULTS["warmup_ratio"] * cfg.num_iterations))
    assert cfg.shape_policy == "depth"
    assert cfg.budget_policy == "param_data_ratio"
    assert cfg.gradient_accumulation_steps >= 1
    assert config_from_dict(cfg.to_dict()).model.n_embd == cfg.model.n_embd
    old_style = cfg.to_dict()
    old_style["model"].pop("attention_window")
    assert config_from_dict(old_style).model.attention_window is None
    hybrid_style = cfg.to_dict()
    hybrid_style["model"].pop("attention_full_every")
    assert config_from_dict(hybrid_style).model.attention_full_every is None
    legacy_style = cfg.to_dict()
    legacy_style["model"].pop("loss_backend")
    assert config_from_dict(legacy_style).model.loss_backend == DEFAULTS["loss_backend"]
    stale_mlp_style = cfg.to_dict()
    stale_mlp_style["model"]["mlp_activation"] = "legacy"
    assert not hasattr(config_from_dict(stale_mlp_style).model, "mlp_activation")


def test_config_shape_and_budget_controls() -> None:
    base = resolve_config(depth=6, vocab_size=32768, sequence_len=128, device_batch_size=2, precision="fp32_test", compile_model=False)
    matched_params = resolve_config(depth=None, target_params=base.scaling_params, vocab_size=32768, sequence_len=128, device_batch_size=2, precision="fp32_test", compile_model=False)
    assert matched_params.shape_policy == "target_params"
    assert matched_params.depth == base.depth
    fixed_tokens = resolve_config(depth=6, vocab_size=32768, sequence_len=128, device_batch_size=2, target_tokens=1_000_000, precision="fp32_test", compile_model=False, comparison_mode="same_tokens")
    assert fixed_tokens.budget_policy == "fixed_tokens"
    assert fixed_tokens.target_tokens == 1_000_000
    fixed_bytes = resolve_config(depth=6, vocab_size=32768, sequence_len=128, device_batch_size=2, target_bytes=2_000_000, bytes_per_token=2.0, precision="fp32_test", compile_model=False, comparison_mode="same_bytes")
    assert fixed_bytes.budget_policy == "fixed_bytes"
    assert fixed_bytes.target_tokens == 1_000_000
    try:
        resolve_config(depth=6, vocab_size=32768, target_tokens=1, target_flops=1.0)
    except ValueError as exc:
        assert "choose only one budget override" in str(exc)
    else:
        raise AssertionError("conflicting budget overrides should fail")
