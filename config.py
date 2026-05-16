from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


SCALING_POLICY = "depth_simple"
COMPARISON_MODES = {"same_depth", "same_params", "same_tokens", "same_bytes", "same_flops", "same_time"}

DEFAULTS = {
    "head_dim": 128,
    "layers_per_head": 2,
    "sequence_len": 2048,
    "attention_window": 512,
    "attention_full_every": 4,
    "target_param_data_ratio": 60,
    "device_batch_size": 16,
    "auto_device_batch_memory_fraction": 0.68,
    "auto_device_batch_max": 128,
    "global_batch_tokens": 2**20,
    "lr_depth_stability_reference": 6,
    "embedding_lr_ref": 0.1,
    "unembedding_lr_ref": 0.01,
    "matrix_lr_ref": 0.02,
    "scalar_lr_ref": 0.1,
    "weight_decay": 0.1,
    "optimizer": "muon_adamw",
    "lr_scheduler": "wsd",
    "warmup_ratio": 0.05,
    "warmdown_ratio": 0.3,
    "final_lr_frac": 0.1,
    "compile_mode": "max-autotune",
    "compile_capture_scalar_outputs": True,
    "mlp_backend": "triton",
    "loss_backend": "triton",
    "rope_backend": "triton",
    "loss_chunk_size": 4096,
}

BF16_BYTES = 2
ACTIVATION_MEMORY_SAFETY = 1.20
ACTIVATION_EMBD_STREAMS = 10
ACTIVATION_MLP_STREAMS = 3


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def round_up(x: int, multiple: int) -> int:
    return ceil_div(x, multiple) * multiple


def _pow2_floor(x: float) -> int:
    if x <= 1:
        return 1
    return 2 ** int(math.floor(math.log2(x)))


def depth_dimensions(depth: int, head_dim: int = DEFAULTS["head_dim"], layers_per_head: int = DEFAULTS["layers_per_head"]) -> tuple[int, int]:
    n_head = ceil_div(depth, layers_per_head)
    n_embd = n_head * head_dim
    return n_embd, n_head


def estimate_activation_gib_per_sample(depth: int, n_embd: int, sequence_len: int) -> float:
    hidden = mlp_hidden_dim(n_embd)
    # FlashAttention keeps attention memory linear in sequence length; the safety
    # multiplier covers compiler workspaces, temporaries, and allocator effects.
    activation_elements_per_token_layer = (
        ACTIVATION_EMBD_STREAMS * n_embd
        + ACTIVATION_MLP_STREAMS * hidden
    )
    return (
        ACTIVATION_MEMORY_SAFETY
        * BF16_BYTES
        * depth
        * sequence_len
        * activation_elements_per_token_layer
        / 1024**3
    )


def auto_device_batch_size(
    depth: int,
    n_embd: int,
    sequence_len: int,
    gpu_memory_gib: float | None = None,
    *,
    vocab_size: int,
    max_batch_size: int = DEFAULTS["auto_device_batch_max"],
    memory_fraction: float = DEFAULTS["auto_device_batch_memory_fraction"],
) -> int:
    if gpu_memory_gib is None:
        return DEFAULTS["device_batch_size"]

    # Memory consumed by non-activation tensors (parameters, gradients, optimizer states)
    # must be subtracted from the budget before allocating for activations.
    # Conservative estimate: all params use AdamW = 8 bytes/param
    # (2 param + 2 grad + 2 exp_avg + 2 exp_avg_sq, all BF16).
    # This overestimates Muon params (6 bytes/param), giving a small safety margin.
    total_params = (
        scaling_params_for_depth(depth, vocab_size)
        + vocab_size * n_embd           # embedding table (not in scaling_params)
    )
    static_gib = total_params * 8 / 1024**3

    # Fixed overhead: CUDA context, compilation cache, pinned buffers, fragmentation.
    cuda_overhead_gib = 2.0

    budget_gib = max(1.0, gpu_memory_gib * memory_fraction)
    available_gib = max(0.0, budget_gib - static_gib - cuda_overhead_gib)

    per_sample_gib = estimate_activation_gib_per_sample(depth, n_embd, sequence_len)
    if per_sample_gib <= 0:
        return max(1, min(max_batch_size, 1))

    estimated_batch = available_gib / per_sample_gib
    memory_cap = _pow2_floor(estimated_batch)

    return max(1, min(max_batch_size, memory_cap))


def mlp_hidden_dim(n_embd: int) -> int:
    return round_up(ceil_div(8 * n_embd, 3), 256)


def scaling_params_for_depth(
    depth: int,
    vocab_size: int,
    head_dim: int = DEFAULTS["head_dim"],
    layers_per_head: int = DEFAULTS["layers_per_head"],
) -> int:
    n_embd, _ = depth_dimensions(depth, head_dim, layers_per_head)
    hidden = mlp_hidden_dim(n_embd)
    attn_mats = 4 * n_embd * n_embd
    mlp_mats = 3 * n_embd * hidden
    lm_head = n_embd * vocab_size
    return depth * (attn_mats + mlp_mats) + lm_head


def depth_for_target_params(target_params: int, vocab_size: int) -> int:
    hi = 1
    while scaling_params_for_depth(hi, vocab_size) < target_params:
        hi *= 2
    lo = max(1, hi // 2)
    return min(range(lo, hi + 1), key=lambda d: abs(scaling_params_for_depth(d, vocab_size) - target_params))


def normalize_attention_window(attention_window: int | None) -> int | None:
    if attention_window is None or attention_window == 0:
        return None
    return int(attention_window)


def normalize_attention_full_every(attention_full_every: int | None) -> int | None:
    if attention_full_every is None or attention_full_every == 0:
        return None
    return int(attention_full_every)


def layer_attention_window(layer_idx: int, n_layer: int, attention_window: int | None, attention_full_every: int | None) -> int | None:
    if attention_window is None:
        return None
    if layer_idx == n_layer - 1:
        return None
    if attention_full_every is not None and (layer_idx + 1) % attention_full_every == 0:
        return None
    return attention_window


def estimate_flops_per_token(
    depth: int,
    n_embd: int,
    seq_len: int,
    scaling_params: int,
    attention_window: int | None = None,
    attention_full_every: int | None = None,
) -> int:
    attn_flops = 0
    for layer_idx in range(depth):
        layer_window = layer_attention_window(layer_idx, depth, attention_window, attention_full_every)
        effective_context = seq_len if layer_window is None else min(seq_len, layer_window)
        attn_flops += 12 * effective_context * n_embd
    return int(6 * scaling_params + attn_flops)


@dataclass
class ModelConfig:
    """Decoder-only baseline: RoPE, pre-norm RMSNorm, QK norm, SwiGLU MLP, untied head."""

    vocab_size: int
    block_size: int
    n_layer: int
    n_embd: int
    n_head: int
    n_kv_head: int
    head_dim: int = DEFAULTS["head_dim"]
    mlp_hidden: int = 0
    rope_theta: float = 1_000_000.0
    rope_fraction: float = 0.25
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = False
    qk_norm: bool = True
    attention_window: int | None = DEFAULTS["attention_window"]
    attention_full_every: int | None = DEFAULTS["attention_full_every"]
    attention_backend: str = "flash_attn_2"
    mlp_backend: str = DEFAULTS["mlp_backend"]
    loss_backend: str = DEFAULTS["loss_backend"]
    loss_chunk_size: int | None = DEFAULTS["loss_chunk_size"]
    rope_backend: str = DEFAULTS["rope_backend"]


@dataclass
class ResolvedConfig:
    depth: int
    scaling_policy: str
    model: ModelConfig
    sequence_len: int
    scaling_params: int
    target_tokens: int
    global_batch_tokens: int
    requested_global_batch_tokens: int | None
    device_batch_size: int
    gradient_accumulation_steps: int
    total_gradient_accumulation_steps: int
    world_size: int
    batch_lr_scale: float
    embedding_lr: float
    unembedding_lr: float
    matrix_lr: float
    scalar_lr: float
    weight_decay: float
    optimizer: str
    warmup_ratio: float
    warmup_steps: int
    warmdown_ratio: float
    final_lr_frac: float
    num_iterations: int
    estimated_flops_per_token: int
    lr_scheduler: str = DEFAULTS["lr_scheduler"]
    target_flops: float | None = None
    precision: str = "bf16"
    compile: bool = True
    compile_mode: str = DEFAULTS["compile_mode"]
    compile_capture_scalar_outputs: bool = DEFAULTS["compile_capture_scalar_outputs"]
    kernel_backend: str = "torch"
    shape_policy: str = "depth"
    budget_policy: str = "param_data_ratio"
    comparison_mode: str = "same_depth"
    requested_depth: int | None = None
    requested_target_params: int | None = None
    target_bytes: int | None = None
    bytes_per_token: float | None = None
    target_time_seconds: float | None = None
    tokens_per_second: float | None = None
    target_param_data_ratio: int = DEFAULTS["target_param_data_ratio"]
    scheduled_tokens: int = 0
    train_flops_budget: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_config(
    *,
    depth: int | None,
    vocab_size: int = 32768,
    sequence_len: int = DEFAULTS["sequence_len"],
    attention_window: int | None = DEFAULTS["attention_window"],
    attention_full_every: int | None = DEFAULTS["attention_full_every"],
    target_param_data_ratio: int = DEFAULTS["target_param_data_ratio"],
    target_params: int | None = None,
    target_tokens: int | None = None,
    target_bytes: int | None = None,
    bytes_per_token: float | None = None,
    global_batch_tokens: int | None = None,
    device_batch_size: int | None = None,
    gpu_memory_gib: float | None = None,
    num_iterations: int | None = None,
    target_flops: float | None = None,
    target_time_seconds: float | None = None,
    tokens_per_second: float | None = None,
    precision: str = "bf16",
    compile_model: bool = True,
    compile_mode: str = DEFAULTS["compile_mode"],
    compile_capture_scalar_outputs: bool = DEFAULTS["compile_capture_scalar_outputs"],
    kernel_backend: str = "torch",
    mlp_backend: str = DEFAULTS["mlp_backend"],
    loss_backend: str = DEFAULTS["loss_backend"],
    loss_chunk_size: int | None = DEFAULTS["loss_chunk_size"],
    rope_backend: str = DEFAULTS["rope_backend"],
    optimizer: str = DEFAULTS["optimizer"],
    comparison_mode: str = "same_depth",
) -> ResolvedConfig:
    if comparison_mode not in COMPARISON_MODES:
        raise ValueError(f"unknown comparison mode {comparison_mode!r}")
    attention_window = normalize_attention_window(attention_window)
    attention_full_every = normalize_attention_full_every(attention_full_every)
    if compile_mode not in {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}:
        raise ValueError(f"unknown compile mode {compile_mode!r}")
    if kernel_backend != "torch":
        raise ValueError(f"unknown kernel backend {kernel_backend!r}")
    if mlp_backend not in {"torch", "triton"}:
        raise ValueError(f"unknown MLP backend {mlp_backend!r}")
    if loss_backend not in {"torch", "triton"}:
        raise ValueError(f"unknown loss backend {loss_backend!r}")
    if loss_chunk_size is not None and loss_chunk_size < 0:
        raise ValueError(f"loss_chunk_size must be nonnegative, got {loss_chunk_size}")
    if rope_backend not in {"torch", "triton"}:
        raise ValueError(f"unknown RoPE backend {rope_backend!r}")
    if optimizer not in {"muon_adamw", "adamw"}:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    budget_overrides = [num_iterations, target_tokens, target_bytes, target_flops, target_time_seconds]
    if sum(value is not None for value in budget_overrides) > 1:
        raise ValueError("choose only one budget override: num_iterations, target_tokens, target_bytes, target_flops, or target_time_seconds")

    requested_depth = depth
    if target_params is not None:
        depth = depth_for_target_params(target_params, vocab_size)
        shape_policy = "target_params"
    elif depth is None:
        raise ValueError("depth is required unless target_params is provided")
    else:
        shape_policy = "depth"

    n_embd, n_head = depth_dimensions(depth)
    requested_device_batch_size = device_batch_size
    if device_batch_size is None or device_batch_size <= 0:
        device_batch_size = auto_device_batch_size(
            depth,
            n_embd,
            sequence_len,
            gpu_memory_gib,
            vocab_size=vocab_size,
        )
    hidden = mlp_hidden_dim(n_embd)
    scaling_params = scaling_params_for_depth(depth, vocab_size)
    param_ratio_target_tokens = int(target_param_data_ratio * scaling_params)
    resolved_target_tokens = param_ratio_target_tokens
    budget_policy = "param_data_ratio"
    if target_tokens is not None:
        resolved_target_tokens = int(target_tokens)
        budget_policy = "fixed_tokens"
    elif target_bytes is not None:
        resolved_target_tokens = int(target_bytes / bytes_per_token)
        budget_policy = "fixed_bytes"
    elif target_time_seconds is not None:
        resolved_target_tokens = int(target_time_seconds * tokens_per_second)
        budget_policy = "fixed_time_estimate"
    elif target_flops is not None:
        budget_policy = "fixed_flops"
    elif num_iterations is not None:
        budget_policy = "fixed_steps"

    if global_batch_tokens is not None and global_batch_tokens <= 0:
        raise ValueError(f"global_batch_tokens must be positive, got {global_batch_tokens}")
    requested_global = global_batch_tokens
    nominal_global = global_batch_tokens if global_batch_tokens is not None else DEFAULTS["global_batch_tokens"]
    micro_tokens = device_batch_size * sequence_len
    grad_accum = max(1, ceil_div(nominal_global, micro_tokens))
    actual_global = grad_accum * micro_tokens

    batch_lr_scale = math.sqrt(actual_global / DEFAULTS["global_batch_tokens"])
    depth_lr_cap = math.sqrt(DEFAULTS["lr_depth_stability_reference"] / depth)
    lr_scale = min(batch_lr_scale, depth_lr_cap)
    embedding_lr = DEFAULTS["embedding_lr_ref"] * lr_scale
    unembedding_lr = DEFAULTS["unembedding_lr_ref"] * lr_scale
    matrix_lr = DEFAULTS["matrix_lr_ref"] * lr_scale
    scalar_lr = DEFAULTS["scalar_lr_ref"] * lr_scale
    weight_decay = DEFAULTS["weight_decay"]

    flops_per_token = estimate_flops_per_token(depth, n_embd, sequence_len, scaling_params, attention_window, attention_full_every)
    if num_iterations is not None:
        steps = int(num_iterations)
    elif target_flops is not None:
        steps = round(target_flops / (flops_per_token * actual_global))
    else:
        steps = resolved_target_tokens // actual_global
    warmup_steps = max(1, round(DEFAULTS["warmup_ratio"] * steps)) if steps > 0 else 0
    scheduled_tokens = steps * actual_global
    train_flops_budget = float(flops_per_token * scheduled_tokens)

    model = ModelConfig(
        vocab_size=vocab_size,
        block_size=sequence_len,
        n_layer=depth,
        n_embd=n_embd,
        n_head=n_head,
        n_kv_head=n_head,
        mlp_hidden=hidden,
        attention_window=attention_window,
        attention_full_every=attention_full_every,
        attention_backend="flash_attn_2",
        mlp_backend=mlp_backend,
        loss_backend=loss_backend,
        rope_backend=rope_backend,
        loss_chunk_size=loss_chunk_size,
    )

    return ResolvedConfig(
        depth=depth,
        scaling_policy=SCALING_POLICY,
        model=model,
        sequence_len=sequence_len,
        scaling_params=scaling_params,
        target_tokens=resolved_target_tokens,
        global_batch_tokens=actual_global,
        requested_global_batch_tokens=requested_global,
        device_batch_size=device_batch_size,
        gradient_accumulation_steps=grad_accum,
        total_gradient_accumulation_steps=grad_accum,
        world_size=1,
        batch_lr_scale=lr_scale,
        embedding_lr=embedding_lr,
        unembedding_lr=unembedding_lr,
        matrix_lr=matrix_lr,
        scalar_lr=scalar_lr,
        weight_decay=weight_decay,
        optimizer=optimizer,
        lr_scheduler=DEFAULTS["lr_scheduler"],
        warmup_ratio=DEFAULTS["warmup_ratio"],
        warmup_steps=warmup_steps,
        warmdown_ratio=DEFAULTS["warmdown_ratio"],
        final_lr_frac=DEFAULTS["final_lr_frac"],
        num_iterations=steps,
        estimated_flops_per_token=flops_per_token,
        target_flops=target_flops,
        precision=precision,
        compile=compile_model,
        compile_mode=compile_mode,
        compile_capture_scalar_outputs=compile_capture_scalar_outputs,
        kernel_backend=kernel_backend,
        shape_policy=shape_policy,
        budget_policy=budget_policy,
        comparison_mode=comparison_mode,
        requested_depth=requested_depth,
        requested_target_params=target_params,
        target_bytes=target_bytes,
        bytes_per_token=bytes_per_token,
        target_time_seconds=target_time_seconds,
        tokens_per_second=tokens_per_second,
        target_param_data_ratio=target_param_data_ratio,
        scheduled_tokens=scheduled_tokens,
        train_flops_budget=train_flops_budget,
        extra={
            "requested_device_batch_size": requested_device_batch_size,
            "gpu_memory_gib": gpu_memory_gib,
            "auto_device_batch_memory_fraction": DEFAULTS["auto_device_batch_memory_fraction"],
            "uncapped_batch_lr_scale": batch_lr_scale,
            "depth_lr_cap": depth_lr_cap,
            "lr_depth_stability_reference": DEFAULTS["lr_depth_stability_reference"],
        },
    )


def config_from_dict(obj: dict[str, Any]) -> ResolvedConfig:
    data = dict(obj)
    data.pop("D_REF", None)
    data.pop("B_REF", None)
    data.pop("reference_depth", None)
    data.pop("reference_batch_tokens", None)
    data.pop("predicted_batch_tokens", None)
    data.pop("dmodel_lr_scale", None)
    data.setdefault("lr_scheduler", DEFAULTS["lr_scheduler"])
    data.setdefault("optimizer", DEFAULTS["optimizer"])
    data.setdefault("total_gradient_accumulation_steps", data["gradient_accumulation_steps"])
    data.setdefault("world_size", 1)
    model_data = dict(data["model"])
    if "attention_window" not in model_data:
        model_data["attention_window"] = None
    if "attention_full_every" not in model_data:
        model_data["attention_full_every"] = None
    model_data.setdefault("rope_fraction", 0.25)
    model_data.pop("mlp_activation", None)
    model_data.setdefault("mlp_backend", DEFAULTS["mlp_backend"])
    if model_data.get("mlp_backend") not in {"torch", "triton"}:
        model_data["mlp_backend"] = DEFAULTS["mlp_backend"]
    model_data.setdefault("loss_backend", DEFAULTS["loss_backend"])
    if model_data.get("loss_backend") not in {"torch", "triton"}:
        model_data["loss_backend"] = DEFAULTS["loss_backend"]
    if model_data.get("rope_backend") not in {"torch", "triton"}:
        model_data["rope_backend"] = DEFAULTS["rope_backend"]
    model_data.setdefault("loss_chunk_size", DEFAULTS["loss_chunk_size"])
    model_data["attention_window"] = normalize_attention_window(model_data["attention_window"])
    model_data["attention_full_every"] = normalize_attention_full_every(model_data["attention_full_every"])
    data["kernel_backend"] = "torch"
    data["model"] = ModelConfig(**model_data)
    return ResolvedConfig(**data)
