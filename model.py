from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

import kernels
from config import ModelConfig, layer_attention_window


def rms_norm_torch(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return y if weight is None else y * weight


class AcceleratedModule(nn.Module):
    backend_attr: str | None = None

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def selected_backend(self) -> str:
        if self.backend_attr is None:
            return "torch"
        return getattr(self.config, self.backend_attr)

    def forward(self, *args, **kwargs):
        backend = self.selected_backend()
        if backend == "torch":
            return self.fwd_torch(*args, **kwargs)
        fwd_backend = getattr(self, f"fwd_{backend}", None)
        if fwd_backend is None:
            raise RuntimeError(f"{type(self).__name__} does not support backend {backend!r}")
        return fwd_backend(*args, **kwargs)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, param: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if param else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm_torch(x, self.weight, self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 1_000_000.0, fraction: float = 0.25) -> None:
        super().__init__()
        rotary_dim = int(dim * fraction)
        rotary_dim -= rotary_dim % 2
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cos_cached is None or seq_len > self._seq_len_cached or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached[:, :, :seq_len].to(dtype=dtype), self._sin_cached[:, :, :seq_len].to(dtype=dtype)


def residual_rms_norm(residual: torch.Tensor, branch: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    out = residual + branch
    return out, rms_norm_torch(out, None, eps)


class QKVProjection(AcceleratedModule):
    backend_attr = "qkv_backend"

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__(config)
        del layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.q_dim = config.n_head * config.head_dim
        self.kv_dim = config.n_kv_head * config.head_dim
        self.qkv_proj = nn.Linear(config.n_embd, self.q_dim + 2 * self.kv_dim, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta, config.rope_fraction)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv_proj(x).split((self.q_dim, self.kv_dim, self.kv_dim), dim=-1)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim)
        k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim)
        v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim)
        cos, sin = self.rope(seq_len, x.device, q.dtype)
        return q, k, v, cos, sin

    def fwd_torch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v, cos, sin = self._project(x)
        q = rms_norm_torch(q, None, self.config.norm_eps)
        k = rms_norm_torch(k, None, self.config.norm_eps)
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)
        dim = cos.shape[-1]
        half = dim // 2
        cos_half = cos[..., :half]
        sin_half = sin[..., :half]
        q1, q2, q_pass = q[..., :half], q[..., half:dim], q[..., dim:]
        k1, k2, k_pass = k[..., :half], k[..., half:dim], k[..., dim:]
        q = torch.cat((q1 * cos_half - q2 * sin_half, q2 * cos_half + q1 * sin_half, q_pass), dim=-1)
        k = torch.cat((k1 * cos_half - k2 * sin_half, k2 * cos_half + k1 * sin_half, k_pass), dim=-1)
        return q, k, v

    def fwd_triton(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v, cos, sin = self._project(x)
        q, k = kernels.qk_norm_rope_triton(
            q,
            k,
            cos.transpose(1, 2),
            sin.transpose(1, 2),
            self.config.norm_eps,
        )
        return q, k, v


class AttentionCore(AcceleratedModule):
    backend_attr = "attention_backend"

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__(config)
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.attention_window = layer_attention_window(layer_idx, config.n_layer, config.attention_window, config.attention_full_every)
        self.dropout = config.dropout
        self._mask_cache: torch.Tensor | None = None
        self._mask_cache_seq_len = 0
        self._flex_mask_cache = None
        self._flex_mask_cache_key = None

    def _sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
        window = self.attention_window
        if window is None or window >= seq_len:
            return None
        if self._mask_cache is None or self._mask_cache_seq_len != seq_len or self._mask_cache.device != device:
            pos = torch.arange(seq_len, device=device)
            distance = pos[:, None] - pos[None, :]
            mask = (distance >= 0) & (distance < window)
            self._mask_cache = mask[None, None, :, :]
            self._mask_cache_seq_len = seq_len
        return self._mask_cache

    def _attention_window_size(self, seq_len: int) -> int | None:
        window = self.attention_window
        if window is None or window >= seq_len:
            return None
        return window

    def _flex_block_mask(self, batch_size: int, n_head: int, seq_len: int, device: torch.device):
        window_size = self._attention_window_size(seq_len)
        key = (batch_size, n_head, seq_len, device, window_size)
        if self._flex_mask_cache is None or self._flex_mask_cache_key != key:
            self._flex_mask_cache = kernels.flex_attention_block_mask(batch_size, n_head, seq_len, device, window_size)
            self._flex_mask_cache_key = key
        return self._flex_mask_cache

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        backend = "torch" if q.device.type == "cpu" else self.selected_backend()
        fwd_backend = getattr(self, f"fwd_{backend}", None)
        if fwd_backend is None:
            raise RuntimeError(f"{type(self).__name__} does not support backend {backend!r}")
        return fwd_backend(q, k, v)

    def fwd_torch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        seq_len = q.shape[1]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        attn_mask = self._sliding_window_mask(seq_len, q.device)
        y = kernels.scaled_dot_product_attention(
            q,
            k,
            v,
            self.dropout if self.training else 0.0,
            is_causal=attn_mask is None,
            attn_mask=attn_mask,
            window_size=self._attention_window_size(seq_len),
            backend="torch",
        )
        return y.transpose(1, 2)

    def fwd_flex_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        seq_len = q.shape[1]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        block_mask = self._flex_block_mask(q.shape[0], q.shape[1], seq_len, q.device)
        y = kernels.scaled_dot_product_attention(
            q,
            k,
            v,
            self.dropout if self.training else 0.0,
            is_causal=True,
            block_mask=block_mask,
            window_size=self._attention_window_size(seq_len),
            backend="flex_attention",
        )
        return y.transpose(1, 2)

    def fwd_flash_attn_2(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return kernels.scaled_dot_product_attention(
            q,
            k,
            v,
            self.dropout if self.training else 0.0,
            is_causal=True,
            attn_mask=None,
            window_size=self._attention_window_size(q.shape[1]),
            backend="flash_attn_2",
        )

    def fwd_flash_attn_3(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return kernels.scaled_dot_product_attention(
            q,
            k,
            v,
            self.dropout if self.training else 0.0,
            is_causal=True,
            attn_mask=None,
            window_size=self._attention_window_size(q.shape[1]),
            backend="flash_attn_3",
        )

    def fwd_flash_attn_4(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return kernels.scaled_dot_product_attention(
            q,
            k,
            v,
            self.dropout if self.training else 0.0,
            is_causal=True,
            attn_mask=None,
            window_size=self._attention_window_size(q.shape[1]),
            backend="flash_attn_4",
        )


class OutputProjection(AcceleratedModule):
    backend_attr = "output_backend"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.norm_eps = config.norm_eps
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, bias=False)

    def fwd_torch(self, y: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, n_head, head_dim = y.shape
        y = y.reshape(bsz, seq_len, n_head * head_dim)
        return residual_rms_norm(residual, self.o_proj(y), self.norm_eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.qkv = QKVProjection(config, layer_idx)
        self.attention = AttentionCore(config, layer_idx)
        self.out = OutputProjection(config)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self.qkv(x)
        y = self.attention(q, k, v)
        return self.out(y, residual)


class GateUpProjection(AcceleratedModule):
    backend_attr = "gate_up_backend"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.gate_up_proj = nn.Linear(config.n_embd, 2 * config.mlp_hidden, bias=False)

    def fwd_torch(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return F.silu(gate) * up

    def fwd_triton(self, x: torch.Tensor) -> torch.Tensor:
        return kernels.swiglu_triton(self.gate_up_proj(x))


class DownProjection(AcceleratedModule):
    backend_attr = "down_backend"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.norm_eps = config.norm_eps
        self.down_proj = nn.Linear(config.mlp_hidden, config.n_embd, bias=False)

    def fwd_torch(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return residual_rms_norm(residual, self.down_proj(x), self.norm_eps)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_up = GateUpProjection(config)
        self.down = DownProjection(config)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.down(self.gate_up(x), residual)


class LMHeadProjection(AcceleratedModule):
    backend_attr = "lm_head_backend"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.chunk_size = config.loss_chunk_size

    def fwd_torch(self, hidden: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if targets is None:
            return self.lm_head(hidden), None
        hidden_flat = hidden.reshape(-1, hidden.size(-1))
        targets_flat = targets.reshape(-1)
        n_tokens = targets_flat.numel()
        chunk_size = self.chunk_size if self.chunk_size else n_tokens
        if chunk_size < 0:
            raise RuntimeError(f"loss_chunk_size must be nonnegative, got {self.chunk_size}")
        loss_sum = hidden_flat.new_zeros((), dtype=torch.float32)
        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            logits = F.linear(hidden_flat[start:end], self.lm_head.weight, self.lm_head.bias)
            loss_sum = loss_sum + F.cross_entropy(logits.float(), targets_flat[start:end], reduction="sum")
        return None, loss_sum / n_tokens

    def fwd_triton(self, hidden: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if targets is None:
            return self.lm_head(hidden), None
        loss = kernels.fused_linear_cross_entropy_with_weight(
            hidden,
            self.lm_head.weight,
            self.lm_head.bias,
            targets,
            backend="triton",
            chunk_size=self.chunk_size,
        )
        return None, loss


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, x_norm = self.attn(x_norm, x)
        x, x_norm = self.mlp(x_norm, x)
        return x, x_norm


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.input_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.blocks = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])
        self.lm_head = LMHeadProjection(config)
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x = self.tok_emb(idx)
        x_norm = self.input_norm(x)
        for block in self.blocks:
            x, x_norm = block(x, x_norm)
        return self.lm_head(x_norm, targets)

    def prepare_compile_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 1,
    ) -> None:
        for block in self.blocks:
            block.attn.qkv.rope(seq_len, device, dtype)
            block.attn.attention._sliding_window_mask(seq_len, device)
            if self.config.attention_backend == "flex_attention" and device.type != "cpu":
                block.attn.attention._flex_block_mask(batch_size, self.config.n_head, seq_len, device)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict:
        return asdict(self.config)
