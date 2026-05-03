from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

import kernels
from config import ModelConfig, layer_attention_window


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, param: bool = False, backend: str = "torch") -> None:
        super().__init__()
        self.eps = eps
        self.backend = backend
        self.weight = nn.Parameter(torch.ones(dim)) if param else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kernels.rms_norm(x, self.weight, self.eps, backend=self.backend)


class Linear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)


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


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dim = cos.shape[-1]
    half = dim // 2
    cos_half = cos[..., :half]
    sin_half = sin[..., :half]
    q1, q2, q_pass = q[..., :half], q[..., half:dim], q[..., dim:]
    k1, k2, k_pass = k[..., :half], k[..., half:dim], k[..., dim:]
    q_rot = torch.cat((q1 * cos_half - q2 * sin_half, q2 * cos_half + q1 * sin_half, q_pass), dim=-1)
    k_rot = torch.cat((k1 * cos_half - k2 * sin_half, k2 * cos_half + k1 * sin_half, k_pass), dim=-1)
    return q_rot, k_rot


def apply_rope_flash_layout(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.transpose(1, 2)
    sin = sin.transpose(1, 2)
    return apply_rope(q, k, cos, sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.attention_window = layer_attention_window(layer_idx, config.n_layer, config.attention_window, config.attention_full_every)
        self.dropout = config.dropout
        self.qkv_proj = Linear(config.n_embd, (config.n_head + 2 * config.n_kv_head) * config.head_dim, bias=False)
        self.o_proj = Linear(config.n_head * config.head_dim, config.n_embd, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.norm_eps) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(config.head_dim, config.norm_eps) if config.qk_norm else nn.Identity()
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta, config.rope_fraction)
        self._mask_cache: torch.Tensor | None = None
        self._mask_cache_seq_len = 0

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q_size = self.n_head * self.head_dim
        kv_size = self.n_kv_head * self.head_dim
        q, k, v = self.qkv_proj(x).split((q_size, kv_size, kv_size), dim=-1)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim)
        k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim)
        v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = self.rope(seq_len, x.device, q.dtype)
        q, k = apply_rope_flash_layout(q, k, cos, sin)
        attn_mask = None
        if self.config.attention_backend == "torch":
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if self.n_kv_head != self.n_head:
                repeat = self.n_head // self.n_kv_head
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            attn_mask = self._sliding_window_mask(seq_len, x.device)
        y = kernels.scaled_dot_product_attention(
            q,
            k,
            v,
            self.dropout if self.training else 0.0,
            is_causal=attn_mask is None,
            attn_mask=attn_mask,
            window_size=self._attention_window_size(seq_len),
            backend=self.config.attention_backend,
        )
        if self.config.attention_backend == "torch":
            y = y.transpose(1, 2)
        y = y.reshape(bsz, seq_len, self.n_head * self.head_dim)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = Linear(config.n_embd, config.mlp_hidden, bias=False)
        self.up_proj = Linear(config.n_embd, config.mlp_hidden, bias=False)
        self.down_proj = Linear(config.mlp_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, config.norm_eps, backend=config.norm_backend)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.norm_2 = RMSNorm(config.n_embd, config.norm_eps, backend=config.norm_backend)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd, config.norm_eps, backend=config.norm_backend)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
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
        x = self.tok_emb(idx).to(dtype=kernels.compute_dtype_for_device(idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        loss = None
        if targets is not None:
            loss = kernels.chunked_linear_cross_entropy(
                x,
                self.lm_head,
                targets,
                self.config.loss_chunk_size,
                backend=self.config.loss_backend,
            )
            return None, loss
        logits = self.lm_head(x)
        return logits, loss

    def prepare_compile_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> None:
        for block in self.blocks:
            block.attn.rope(seq_len, device, dtype)
            block.attn._sliding_window_mask(seq_len, device)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict:
        return asdict(self.config)
