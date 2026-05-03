from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from config import DEFAULTS, config_from_dict, layer_attention_window, resolve_config, scaling_params_for_depth

try:
    import torch
except ModuleNotFoundError:
    torch = None


def require_torch(test_name: str) -> bool:
    if torch is None:
        print(f"skip {test_name}: torch is not installed")
        return False
    return True


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


def test_model_forward_and_optimizer_grouping() -> None:
    if not require_torch("model/optimizer test"):
        return
    from model import LanguageModel
    from optim import create_optimizer

    cfg = resolve_config(
        depth=5,
        vocab_size=128,
        sequence_len=8,
        attention_window=4,
        device_batch_size=2,
        precision="fp32_test",
        compile_model=False,
        norm_backend="torch",
        loss_backend="torch",
    )
    model = LanguageModel(cfg.model)
    assert hasattr(model.blocks[0].mlp, "up_proj")
    assert hasattr(model.blocks[0].mlp, "gate_proj")
    assert cfg.model.rope_fraction == 0.25
    assert model.blocks[0].attn.rope.rotary_dim == cfg.model.head_dim // 4
    mask = model.blocks[0].attn._sliding_window_mask(cfg.sequence_len, torch.device("cpu"))
    assert model.blocks[3].attn.attention_window is None
    assert model.blocks[-1].attn.attention_window is None
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
    assert summary["muon_tensors"] > 0
    assert summary["adamw_tensors"] > 0


def test_model_initialization_is_gpt_style() -> None:
    if not require_torch("model initialization test"):
        return
    from model import LanguageModel

    torch.manual_seed(0)
    cfg = resolve_config(
        depth=6,
        vocab_size=512,
        sequence_len=8,
        device_batch_size=2,
        precision="fp32_test",
        compile_model=False,
        norm_backend="torch",
        loss_backend="torch",
    )
    model = LanguageModel(cfg.model)
    base_std = 0.02
    residual_std = base_std / (2 * cfg.model.n_layer) ** 0.5

    def assert_std_close(param: torch.Tensor, expected: float) -> None:
        actual = float(param.detach().float().std(unbiased=False))
        assert abs(actual - expected) / expected < 0.12, (actual, expected)

    assert_std_close(model.tok_emb.weight, base_std)
    assert_std_close(model.lm_head.weight, base_std)
    assert_std_close(model.blocks[0].attn.qkv_proj.weight, base_std)
    assert_std_close(model.blocks[0].mlp.gate_proj.weight, base_std)
    assert_std_close(model.blocks[0].mlp.up_proj.weight, base_std)
    assert_std_close(model.blocks[0].attn.o_proj.weight, residual_std)
    assert_std_close(model.blocks[0].mlp.down_proj.weight, residual_std)


def test_chunked_linear_cross_entropy_matches_dense() -> None:
    if not require_torch("chunked linear CE test"):
        return
    import torch.nn.functional as F

    import kernels

    torch.manual_seed(0)
    head = torch.nn.Linear(16, 33, bias=False)
    hidden = torch.randn(3, 7, 16, requires_grad=True)
    targets = torch.randint(0, 33, (3, 7))
    dense = F.cross_entropy(head(hidden).float().reshape(-1, 33), targets.reshape(-1))
    chunked = kernels.chunked_linear_cross_entropy(hidden, head, targets, chunk_size=5)
    assert torch.allclose(chunked, dense, atol=1e-6)


def test_fp8_policy_keeps_lm_head_bf16_linear() -> None:
    if not require_torch("fp8 policy test"):
        return
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


def test_flash_attention_casts_qkv_to_bfloat16() -> None:
    if not require_torch("flash attention dtype test"):
        return
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


def test_schedules() -> None:
    if not require_torch("optimizer schedule test"):
        return
    from optim import lr_multiplier

    assert 0 < lr_multiplier(0, 100, 10, 0.3, 0.1) <= 1
    assert lr_multiplier(20, 100, 10, 0.3, 0.1) == 1.0
    assert lr_multiplier(99, 100, 10, 0.3, 0.1) == 0.1


def test_preprocess_and_memmap() -> None:
    if not require_torch("preprocess/memmap test"):
        return
    from data import MemmapDataLoader
    from prepare_data import prepare_all

    try:
        import tokenizers  # noqa: F401
    except ImportError:
        print("skip tokenizer/preprocess test: tokenizers is not installed")
        return
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        raw = root / "raw.txt"
        raw.write_text(("hello world. this is a tiny language model corpus.\n" * 200), encoding="utf-8")
        out = root / "processed"
        manifest = prepare_all([str(raw)], out, vocab_size=128, val_fraction=0.2, min_frequency=1)
        assert (out / "train.bin").exists()
        assert manifest["train_tokens"] > 16
        loader = MemmapDataLoader(out, block_size=8, seed=123)
        x, y = loader.get_batch("train", 4, "cpu")
        assert x.shape == y.shape == (4, 8)
        assert x.dtype == torch.long
        assert x.is_contiguous()
        assert y.is_contiguous()
        assert torch.equal(x[:, 1:], y[:, :-1])


def test_prepare_eval_smoke_tasks() -> None:
    from prepare_eval import DEFAULT_TASKS, prepare_eval_data

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "eval_data"
        manifest = prepare_eval_data(out, tasks=DEFAULT_TASKS, source="smoke")
        assert sorted(manifest["tasks"]) == sorted(DEFAULT_TASKS)
        for task, entry in manifest["tasks"].items():
            path = Path(entry["file"])
            assert path.exists()
            assert entry["smoke_only"] is True
            assert entry["examples"] == 2
            assert len(entry["sha256"]) == 64


def test_kernel_resolution_and_hashing() -> None:
    if not require_torch("kernel/hash test"):
        return
    from kernels import resolve_kernel_backends
    from repro import hash_directory

    info = resolve_kernel_backends("torch", "fp32_test", False, norm_backend="torch", loss_backend="liger", allow_torch_backend=True)
    assert info.actual_attention_backend in {"flash_attn_2", "torch_sdpa"}
    assert info.actual_norm_backend == "torch"
    assert info.actual_loss_backend in {"liger_fused_linear_ce", "torch"}
    assert isinstance(info.liger_available, bool)
    assert len(hash_directory(Path.cwd())) == 64


def test_manifest_compatibility() -> None:
    if not require_torch("manifest compatibility test"):
        return
    from repro import compatibility_warnings

    base = {"config": {"sequence_len": 8, "global_batch_tokens": 16, "scaling_policy": "p", "precision": "fp8"}, "seed": 1, "data": {"manifest": {"tokenizer_hash": "a", "raw_input_sha256": {"x": "1"}}}}
    other = json.loads(json.dumps(base))
    assert compatibility_warnings(base, other) == []
    other["seed"] = 2
    assert compatibility_warnings(base, other)


def test_repro_compare_cli() -> None:
    base = {"config": {"sequence_len": 8, "global_batch_tokens": 16, "scaling_policy": "p", "precision": "fp8"}, "seed": 1, "data": {"manifest": {"tokenizer_hash": "a", "raw_input_sha256": {"x": "1"}, "split_seed": 1}}}
    same = json.loads(json.dumps(base))
    different = json.loads(json.dumps(base))
    different["seed"] = 2
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        left = root / "left.json"
        right = root / "right.json"
        left.write_text(json.dumps(base), encoding="utf-8")
        right.write_text(json.dumps(same), encoding="utf-8")
        ok = subprocess.run([sys.executable, "repro.py", "compare", str(left), str(right), "--fail-on-warning"], cwd=Path.cwd(), text=True, capture_output=True)
        assert ok.returncode == 0
        assert '"compatible": true' in ok.stdout
        right.write_text(json.dumps(different), encoding="utf-8")
        bad = subprocess.run([sys.executable, "repro.py", "compare", str(left), str(right), "--fail-on-warning"], cwd=Path.cwd(), text=True, capture_output=True)
        assert bad.returncode == 1
        assert "seed differs" in bad.stdout


def test_eval_task_file_provenance() -> None:
    if not require_torch("eval provenance test"):
        return
    from eval import task_file_provenance
    from prepare_eval import prepare_eval_data

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "eval_data"
        manifest = prepare_eval_data(out, tasks=["boolq"], source="smoke")
        provenance = task_file_provenance(out, ["boolq", "validation_loss"], manifest)
        assert sorted(provenance) == ["boolq"]
        assert provenance["boolq"]["sha256"] == manifest["tasks"]["boolq"]["sha256"]
        assert provenance["boolq"]["examples"] == 2


def test_trusted_checkpoint_load() -> None:
    if not require_torch("checkpoint load test"):
        return
    from repro import load_trusted_checkpoint

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "checkpoint.pt"
        payload = {"torch_version": torch.torch_version.TorchVersion(torch.__version__), "tensor": torch.ones(1)}
        torch.save(payload, path)
        loaded = load_trusted_checkpoint(path)
        assert str(loaded["torch_version"]) == str(payload["torch_version"])
        assert torch.equal(loaded["tensor"], payload["tensor"])


def test_mlflow_flattening_helpers() -> None:
    if not require_torch("mlflow helper test"):
        return
    from train import _flatten_for_mlflow, _mlflow_value

    flat = _flatten_for_mlflow({"config": {"depth": 2, "layers": [1, None]}, "ok": True})
    assert flat["config.depth"] == 2
    assert flat["config.layers.0"] == 1
    assert flat["config.layers.1"] is None
    assert flat["ok"] is True
    assert _mlflow_value(None) == "null"
    assert _mlflow_value({"a": 1}) == '{"a": 1}'


def test_optimizer_resume_muon_state_device() -> None:
    if not require_torch("optimizer resume device test"):
        return
    from model import LanguageModel
    from optim import create_optimizer

    if not torch.cuda.is_available():
        print("skip optimizer resume device test: CUDA is not available")
        return
    cfg = resolve_config(
        depth=2,
        vocab_size=128,
        sequence_len=8,
        device_batch_size=2,
        precision="fp32_test",
        compile_model=False,
        norm_backend="torch",
        loss_backend="torch",
    )
    device = torch.device("cuda")
    model = LanguageModel(cfg.model).to(device)
    opt = create_optimizer(model, cfg)
    x = torch.randint(0, cfg.model.vocab_size, (2, cfg.sequence_len), device=device)
    _, loss = model(x, x)
    loss.backward()
    opt.step()
    state = opt.state_dict()
    moved_state = {"muon_state": {}, **{k: v for k, v in state.items() if k != "muon_state"}}
    for name, param_state in state["muon_state"].items():
        moved_state["muon_state"][name] = {
            key: value.cpu() if torch.is_tensor(value) else value for key, value in param_state.items()
        }
    resumed_model = LanguageModel(cfg.model).to(device)
    resumed_opt = create_optimizer(resumed_model, cfg)
    resumed_opt.load_state_dict(moved_state)
    assert all(param_state["momentum_buffer"].device.type == device.type for param_state in resumed_opt.muon_state.values())
    resumed_opt.zero_grad(set_to_none=True)
    _, resumed_loss = resumed_model(x, x)
    resumed_loss.backward()
    resumed_opt.step()


def main() -> None:
    tests = [
        test_config_derivation,
        test_config_shape_and_budget_controls,
        test_model_forward_and_optimizer_grouping,
        test_model_initialization_is_gpt_style,
        test_chunked_linear_cross_entropy_matches_dense,
        test_fp8_policy_keeps_lm_head_bf16_linear,
        test_flash_attention_casts_qkv_to_bfloat16,
        test_schedules,
        test_preprocess_and_memmap,
        test_prepare_eval_smoke_tasks,
        test_kernel_resolution_and_hashing,
        test_manifest_compatibility,
        test_repro_compare_cli,
        test_eval_task_file_provenance,
        test_trusted_checkpoint_load,
        test_mlflow_flattening_helpers,
        test_optimizer_resume_muon_state_device,
    ]
    for test in tests:
        test()
        print(f"ok {test.__name__}")


if __name__ == "__main__":
    main()
