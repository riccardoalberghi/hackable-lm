from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch


MUON_NAME_MARKERS = (
    "qkv_proj.weight",
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)
MUON_SCALAR_PLACEMENTS = {"cpu", "cuda"}
MUON_STATE_DTYPE = torch.bfloat16


@dataclass
class OptimizerSummary:
    muon_tensors: int
    adamw_tensors: int
    muon_params: int
    embedding_params: int
    unembedding_params: int
    scalar_params: int
    excluded: list[str]
    groups: list[dict[str, Any]]


@dataclass
class MuonParam:
    name: str
    param: torch.nn.Parameter
    start: int = 0
    end: int | None = None

    def tensor(self) -> torch.Tensor:
        return self.param if self.end is None else self.param[self.start : self.end]

    def grad(self) -> torch.Tensor | None:
        if self.param.grad is None:
            return None
        return self.param.grad if self.end is None else self.param.grad[self.start : self.end]


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    if g.ndim != 2:
        return g
    orig_dtype = g.dtype
    x = g.float()
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True
    else:
        transposed = False
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x
    if transposed:
        x = x.T
    return x.to(orig_dtype)


@torch.compile(dynamic=False, fullgraph=True)
def muon_update_fused_current(
    stacked_grads: torch.Tensor,
    momentum_buffer: torch.Tensor,
    momentum_t: torch.Tensor,
) -> torch.Tensor:
    momentum = momentum_t.to(momentum_buffer.dtype)
    momentum_buffer.lerp_(stacked_grads, 1.0 - momentum)

    x = momentum_buffer.bfloat16()
    if x.size(-2) > x.size(-1):
        x = x.mT
        transposed = True
    else:
        transposed = False
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        xx_t = x @ x.mT
        poly = torch.baddbmm(xx_t, xx_t, xx_t, beta=b, alpha=c)
        x = torch.baddbmm(x, poly, x, beta=a)
    if transposed:
        x = x.mT

    return x.to(momentum_buffer.dtype)


class HybridMuonAdamW:
    def __init__(
        self,
        named_params: list[tuple[str, torch.nn.Parameter]],
        *,
        embedding_lr: float,
        unembedding_lr: float,
        matrix_lr: float,
        scalar_lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        muon_momentum: float = 0.95,
        qkv_split_sizes: tuple[int, int, int] | None = None,
    ) -> None:
        self.muon: list[MuonParam] = []
        self.adam_embedding: list[torch.nn.Parameter] = []
        self.adam_unembedding: list[torch.nn.Parameter] = []
        self.adam_scalar: list[torch.nn.Parameter] = []
        self.excluded: list[str] = []
        for name, param in named_params:
            if not param.requires_grad:
                self.excluded.append(name)
                continue
            if param.ndim == 2 and name.endswith("qkv_proj.weight"):
                if qkv_split_sizes is None:
                    split = param.shape[0] // 3
                    qkv_split_sizes = (split, split, param.shape[0] - 2 * split)
                q_size, k_size, v_size = qkv_split_sizes
                if q_size + k_size + v_size != param.shape[0]:
                    raise ValueError(f"qkv split sizes {qkv_split_sizes} do not match {name} shape {tuple(param.shape)}")
                self.muon.extend(
                    [
                        MuonParam(f"{name}.q", param, 0, q_size),
                        MuonParam(f"{name}.k", param, q_size, q_size + k_size),
                        MuonParam(f"{name}.v", param, q_size + k_size, param.shape[0]),
                    ]
                )
            elif param.ndim == 2 and any(marker in name for marker in MUON_NAME_MARKERS):
                self.muon.append(MuonParam(name, param))
            elif "tok_emb" in name:
                self.adam_embedding.append(param)
            elif "lm_head" in name:
                self.adam_unembedding.append(param)
            else:
                self.adam_scalar.append(param)

        self.muon_lr = matrix_lr
        self.muon_base_lr = matrix_lr
        self.muon_weight_decay = weight_decay
        self.muon_base_weight_decay = weight_decay
        self.muon_momentum = muon_momentum
        self.muon_state: dict[str, dict[str, torch.Tensor]] = {}
        self.muon_names = [entry.name for entry in self.muon]
        self.muon_params = []
        seen_muon_params = set()
        for entry in self.muon:
            if id(entry.param) not in seen_muon_params:
                self.muon_params.append(entry.param)
                seen_muon_params.add(id(entry.param))
        self._muon_groups: list[dict[str, Any]] = []
        for shape in sorted({tuple(entry.tensor().shape) for entry in self.muon}):
            indices = [i for i, entry in enumerate(self.muon) if tuple(entry.tensor().shape) == shape]
            self._muon_groups.append(
                {
                    "shape": shape,
                    "indices": indices,
                    "names": [self.muon_names[i] for i in indices],
                    "entries": [self.muon[i] for i in indices],
                    "params": [self.muon[i].tensor() for i in indices],
                    "momentum_buffer": None,
                    "grad_buffer": None,
                }
            )
        self._muon_scalar_placement = os.environ.get("SIMPLE_LM_MUON_SCALAR_DEVICE", "cuda").lower()
        if self._muon_scalar_placement not in MUON_SCALAR_PLACEMENTS:
            raise ValueError(
                "SIMPLE_LM_MUON_SCALAR_DEVICE must be one of "
                f"{sorted(MUON_SCALAR_PLACEMENTS)}, got {self._muon_scalar_placement!r}"
            )
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        adam_groups = [
            {"params": self.adam_embedding, "lr": embedding_lr, "base_lr": embedding_lr, "weight_decay": weight_decay, "name": "embedding"},
            {"params": self.adam_unembedding, "lr": unembedding_lr, "base_lr": unembedding_lr, "weight_decay": weight_decay, "name": "unembedding"},
            {"params": self.adam_scalar, "lr": scalar_lr, "base_lr": scalar_lr, "weight_decay": 0.0, "name": "scalar_vector"},
        ]
        use_fused_adamw = any(p.is_cuda for group in adam_groups for p in group["params"])
        self.adam = torch.optim.AdamW(adam_groups, betas=betas, eps=eps, fused=use_fused_adamw)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [{"params": self.muon_params, "lr": self.muon_lr, "weight_decay": self.muon_weight_decay, "name": "muon_matrix"}] + self.adam.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.muon_params:
            p.grad = None if set_to_none else torch.zeros_like(p)
        self.adam.zero_grad(set_to_none=set_to_none)

    def _sync_muon_scalar(self, device: torch.device) -> None:
        scalar_device = torch.device("cpu") if self._muon_scalar_placement == "cpu" else device
        if self._muon_momentum_t.device != scalar_device:
            self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device=scalar_device)
        self._muon_momentum_t.fill_(self.muon_momentum)

    def _ensure_group_buffers(self, group: dict[str, Any], ref_param: torch.Tensor, ref_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        momentum_buffer = group["momentum_buffer"]
        if momentum_buffer is None:
            momentum_buffer = torch.zeros(
                len(group["entries"]),
                *group["shape"],
                dtype=MUON_STATE_DTYPE,
                device=ref_param.device,
            )
            group["momentum_buffer"] = momentum_buffer
            for name, buf in zip(group["names"], momentum_buffer.unbind(0)):
                self.muon_state[name] = {"momentum_buffer": buf}
        elif momentum_buffer.device != ref_param.device or momentum_buffer.dtype != MUON_STATE_DTYPE:
            momentum_buffer = momentum_buffer.to(device=ref_param.device, dtype=MUON_STATE_DTYPE)
            group["momentum_buffer"] = momentum_buffer
            for name, buf in zip(group["names"], momentum_buffer.unbind(0)):
                self.muon_state[name] = {"momentum_buffer": buf}

        grad_buffer = group["grad_buffer"]
        if grad_buffer is None or grad_buffer.device != ref_grad.device or grad_buffer.dtype != MUON_STATE_DTYPE:
            grad_buffer = torch.empty(
                len(group["entries"]),
                *group["shape"],
                dtype=MUON_STATE_DTYPE,
                device=ref_grad.device,
            )
            group["grad_buffer"] = grad_buffer
        return momentum_buffer, grad_buffer

    def _stack_muon_grads(self, grads: list[torch.Tensor], out: torch.Tensor | None = None) -> torch.Tensor:
        dtype = MUON_STATE_DTYPE if out is None else out.dtype
        casted_grads = [grad if grad.dtype == dtype else grad.to(dtype=dtype) for grad in grads]
        return torch.stack(casted_grads, out=out) if out is not None else torch.stack(casted_grads)

    def _apply_muon_updates(self, params: list[torch.Tensor], updates: torch.Tensor) -> None:
        if updates.dtype != params[0].dtype:
            updates = updates.to(dtype=params[0].dtype)
        update_views = list(updates.unbind(0))
        decay = 1.0 - self.muon_lr * self.muon_weight_decay
        if params[0].is_cuda:
            if decay != 1.0:
                torch._foreach_mul_(params, decay)
            torch._foreach_add_(params, update_views, alpha=-self.muon_lr)
            return
        for param, update in zip(params, update_views):
            if decay != 1.0:
                param.mul_(decay)
            param.add_(update, alpha=-self.muon_lr)

    @torch.no_grad()
    def step(self) -> None:
        synced_muon_scalar = False
        for group in self._muon_groups:
            entries = group["entries"]
            grads: list[torch.Tensor | None] = [entry.grad() for entry in entries]
            active_indices = [i for i, grad in enumerate(grads) if grad is not None]
            if not active_indices:
                continue
            params = group["params"]
            ref_param = params[active_indices[0]]
            ref_grad = grads[active_indices[0]]
            assert ref_grad is not None
            full_buffer, grad_buffer = self._ensure_group_buffers(group, ref_param, ref_grad)
            if not synced_muon_scalar:
                self._sync_muon_scalar(ref_param.device)
                synced_muon_scalar = True

            if len(active_indices) == len(entries):
                active_params = params
                active_buffer = full_buffer
                active_grads = [grad for grad in grads if grad is not None]
                stacked_grads = self._stack_muon_grads(active_grads, grad_buffer)
                scatter_buffer = False
            else:
                active_params = [params[i] for i in active_indices]
                active_grads = [grads[i] for i in active_indices]
                active_buffer = full_buffer[active_indices]
                stacked_grads = self._stack_muon_grads(active_grads)
                scatter_buffer = True

            updates = muon_update_fused_current(stacked_grads, active_buffer, self._muon_momentum_t)
            if scatter_buffer:
                full_buffer[active_indices] = active_buffer
            self._apply_muon_updates(active_params, updates)
        self.adam.step()

    def set_lr_multiplier(self, mult: float) -> None:
        self.muon_lr = self.muon_base_lr * mult
        for group in self.adam.param_groups:
            group["lr"] = group["base_lr"] * mult

    def state_dict(self) -> dict[str, Any]:
        muon_state = {}
        for group in self._muon_groups:
            buffer = group["momentum_buffer"]
            if buffer is None:
                continue
            for name, buf in zip(group["names"], buffer.unbind(0)):
                muon_state[name] = {"momentum_buffer": buf}
        return {
            "muon_state": muon_state,
            "muon_lr": self.muon_lr,
            "muon_base_lr": self.muon_base_lr,
            "muon_weight_decay": self.muon_weight_decay,
            "muon_base_weight_decay": self.muon_base_weight_decay,
            "muon_momentum": self.muon_momentum,
            "adam": self.adam.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        params_by_name = {entry.name: entry for entry in self.muon}
        self.muon_state = {}
        for group in self._muon_groups:
            group["momentum_buffer"] = None
        for name, raw_param_state in state["muon_state"].items():
            entry = params_by_name.get(name)
            if entry is None:
                self.muon_state[name] = raw_param_state
                continue
            buffer = raw_param_state.get("momentum_buffer")
            for group in self._muon_groups:
                if name not in group["names"]:
                    continue
                if group["momentum_buffer"] is None:
                    group["momentum_buffer"] = torch.zeros(
                        len(group["entries"]),
                        *group["shape"],
                        dtype=MUON_STATE_DTYPE,
                        device=entry.param.device,
                    )
                idx = group["names"].index(name)
                group["momentum_buffer"][idx].copy_(buffer.to(device=entry.param.device, dtype=MUON_STATE_DTYPE))
                break
        for group in self._muon_groups:
            buffer = group["momentum_buffer"]
            if buffer is not None:
                for name, buf in zip(group["names"], buffer.unbind(0)):
                    self.muon_state[name] = {"momentum_buffer": buf}
        self.muon_lr = state["muon_lr"]
        self.muon_base_lr = state["muon_base_lr"]
        self.muon_weight_decay = state["muon_weight_decay"]
        self.muon_base_weight_decay = state["muon_base_weight_decay"]
        self.muon_momentum = state["muon_momentum"]
        self.adam.load_state_dict(state["adam"])

    def summary(self) -> dict[str, Any]:
        groups = [
            {
                "name": "muon_matrix",
                "tensors": len(self.muon),
                "params": sum(entry.tensor().numel() for entry in self.muon),
                "lr": self.muon_base_lr,
                "weight_decay": self.muon_base_weight_decay,
                "scalar_placement": self._muon_scalar_placement,
            }
        ]
        for group in self.adam.param_groups:
            groups.append(
                {
                    "name": group.get("name"),
                    "tensors": len(group["params"]),
                    "params": sum(p.numel() for p in group["params"]),
                    "lr": group["base_lr"],
                    "weight_decay": group["weight_decay"],
                }
            )
        return {
            "muon_tensors": len(self.muon),
            "adamw_tensors": sum(len(g["params"]) for g in self.adam.param_groups),
            "muon_params": sum(entry.tensor().numel() for entry in self.muon),
            "embedding_params": sum(p.numel() for p in self.adam_embedding),
            "unembedding_params": sum(p.numel() for p in self.adam_unembedding),
            "scalar_params": sum(p.numel() for p in self.adam_scalar),
            "muon_scalar_placement": self._muon_scalar_placement,
            "excluded": self.excluded,
            "groups": groups,
        }


def create_optimizer(model: torch.nn.Module, config: Any) -> HybridMuonAdamW:
    model_cfg = getattr(config, "model", None)
    qkv_split_sizes = None
    if model_cfg is not None:
        qkv_split_sizes = (
            model_cfg.n_head * model_cfg.head_dim,
            model_cfg.n_kv_head * model_cfg.head_dim,
            model_cfg.n_kv_head * model_cfg.head_dim,
        )
    return HybridMuonAdamW(
        list(model.named_parameters()),
        embedding_lr=config.embedding_lr,
        unembedding_lr=config.unembedding_lr,
        matrix_lr=config.matrix_lr,
        scalar_lr=config.scalar_lr,
        weight_decay=config.weight_decay,
        qkv_split_sizes=qkv_split_sizes,
    )


def lr_multiplier(
    step: int,
    num_iterations: int,
    warmup_steps: int,
    warmdown_ratio: float,
    final_lr_frac: float,
    scheduler: str = "wsd",
) -> float:
    if scheduler != "wsd":
        raise ValueError(f"unsupported LR scheduler {scheduler!r}")
    warmup_steps = min(warmup_steps, num_iterations)
    decay_iters = int(num_iterations * warmdown_ratio)
    decay_start = max(warmup_steps, num_iterations - decay_iters)
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step < decay_start:
        return 1.0
    decay_span = num_iterations - decay_start - 1
    progress = (step - decay_start) / decay_span if decay_span else 1.0
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr_frac + (1.0 - final_lr_frac) * cosine
