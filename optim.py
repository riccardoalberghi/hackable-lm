from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


MUON_NAME_MARKERS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)
MUON_STATE_DTYPE = torch.bfloat16


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step_t: Tensor,
    lr_t: Tensor,
    beta1_t: Tensor,
    beta2_t: Tensor,
    eps_t: Tensor,
    weight_decay_t: Tensor,
) -> None:
    param.mul_(1.0 - lr_t * weight_decay_t)
    exp_avg.lerp_(grad, 1.0 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1.0 - beta2_t)

    bias_correction1 = 1.0 - beta1_t**step_t
    bias_correction2 = 1.0 - beta2_t**step_t
    denom = (exp_avg_sq / bias_correction2).sqrt() + eps_t
    step_size = (lr_t / bias_correction1).to(dtype=exp_avg.dtype)
    param.add_((exp_avg / denom) * step_size, alpha=-1.0)


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    weight_decay_t: Tensor,
) -> None:
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

    update = x.to(stacked_params.dtype)
    lr = lr_t.to(stacked_params.dtype)
    weight_decay = weight_decay_t.to(stacked_params.dtype)
    stacked_params.mul_(1.0 - lr * weight_decay)
    stacked_params.add_(update * lr, alpha=-1.0)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: compiled AdamW for non-matrices, compiled Muon for matrix groups."""

    def __init__(self, param_groups: list[dict[str, Any]], *, excluded: list[str] | None = None) -> None:
        super().__init__(param_groups, defaults={})
        self.excluded = [] if excluded is None else excluded
        muon_group = self._first_muon_group()
        self.muon_momentum = muon_group.get("momentum", 0.95)
        self.muon_lr = muon_group.get("lr", 0.0)

        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_weight_decay_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_weight_decay_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _first_muon_group(self) -> dict[str, Any]:
        for group in self.param_groups:
            if group["kind"] == "muon":
                return group
        raise RuntimeError("MuonAdamW needs at least one Muon parameter group")

    def _step_adamw(self, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        for param in group["params"]:
            if param.grad is None:
                continue
            state = self.state[param]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
            state["step"] += 1

            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(beta1)
            self._adamw_beta2_t.fill_(beta2)
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_weight_decay_t.fill_(group["weight_decay"])
            adamw_step_fused(
                param,
                param.grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_weight_decay_t,
            )

    def _step_muon(self, group: dict[str, Any]) -> None:
        params = group["params"]
        if not params:
            return
        active = [(i, p) for i, p in enumerate(params) if p.grad is not None]
        if not active:
            return

        state = self.state[params[0]]
        shape = tuple(params[0].shape)
        expected_buffer_shape = (len(params), *shape)
        momentum_buffer = state.get("momentum_buffer")
        if momentum_buffer is None or tuple(momentum_buffer.shape) != expected_buffer_shape:
            momentum_buffer = torch.zeros(
                len(params),
                *shape,
                dtype=MUON_STATE_DTYPE,
                device=params[0].device,
            )
        elif momentum_buffer.device != params[0].device or momentum_buffer.dtype != MUON_STATE_DTYPE:
            momentum_buffer = momentum_buffer.to(device=params[0].device, dtype=MUON_STATE_DTYPE)
        state["momentum_buffer"] = momentum_buffer

        active_indices = [i for i, _ in active]
        active_params = [p for _, p in active]
        active_grads = [p.grad.to(dtype=MUON_STATE_DTYPE) for p in active_params]
        stacked_grads = torch.stack(active_grads)
        stacked_params = torch.stack(active_params)
        active_buffer = momentum_buffer if len(active) == len(params) else momentum_buffer[active_indices]

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_lr_t.fill_(group["lr"])
        self._muon_weight_decay_t.fill_(group["weight_decay"])
        muon_step_fused(
            stacked_grads,
            stacked_params,
            active_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_weight_decay_t,
        )
        if len(active) != len(params):
            momentum_buffer[active_indices] = active_buffer

        if active_params[0].is_cuda:
            torch._foreach_copy_(active_params, list(stacked_params.unbind(0)))
        else:
            for param, updated in zip(active_params, stacked_params.unbind(0)):
                param.copy_(updated)

        self.muon_lr = group["lr"]
        self.muon_momentum = group["momentum"]

    @torch.no_grad()
    def step(self, closure=None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)
            else:
                raise ValueError(f"unknown optimizer kind {group['kind']!r}")
        return loss

    def set_lr_multiplier(self, mult: float) -> None:
        for group in self.param_groups:
            group["lr"] = group["base_lr"] * mult
        muon_group = self._first_muon_group()
        self.muon_lr = muon_group["lr"]

    def summary(self) -> dict[str, Any]:
        groups = [
            {
                "name": group["name"],
                "kind": group["kind"],
                "tensors": len(group["params"]),
                "params": sum(p.numel() for p in group["params"]),
                "lr": group["base_lr"],
                "weight_decay": group["weight_decay"],
            }
            for group in self.param_groups
        ]
        muon_groups = [group for group in self.param_groups if group["kind"] == "muon"]
        adamw_groups = [group for group in self.param_groups if group["kind"] == "adamw"]
        return {
            "muon_tensors": sum(len(group["params"]) for group in muon_groups),
            "adamw_tensors": sum(len(group["params"]) for group in adamw_groups),
            "muon_params": sum(p.numel() for group in muon_groups for p in group["params"]),
            "embedding_params": _group_param_count(self.param_groups, "embedding"),
            "unembedding_params": _group_param_count(self.param_groups, "unembedding"),
            "scalar_params": _group_param_count(self.param_groups, "scalar_vector"),
            "excluded": self.excluded,
            "groups": groups,
        }


def _group_param_count(param_groups: list[dict[str, Any]], name: str) -> int:
    return sum(p.numel() for group in param_groups if group["name"] == name for p in group["params"])


def _requires_muon(name: str, param: torch.nn.Parameter) -> bool:
    return param.ndim == 2 and any(marker in name for marker in MUON_NAME_MARKERS)


def _adamw_group(name: str, params: list[torch.nn.Parameter], lr: float, weight_decay: float) -> dict[str, Any]:
    return {
        "kind": "adamw",
        "params": params,
        "lr": lr,
        "base_lr": lr,
        "weight_decay": weight_decay,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "name": name,
    }


def create_optimizer(model: torch.nn.Module, config: Any) -> MuonAdamW:
    muon_params: list[torch.nn.Parameter] = []
    embedding_params: list[torch.nn.Parameter] = []
    unembedding_params: list[torch.nn.Parameter] = []
    scalar_params: list[torch.nn.Parameter] = []
    excluded: list[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            excluded.append(name)
        elif _requires_muon(name, param):
            muon_params.append(param)
        elif "tok_emb" in name:
            embedding_params.append(param)
        elif "lm_head" in name:
            unembedding_params.append(param)
        else:
            scalar_params.append(param)

    param_groups = [
        _adamw_group("embedding", embedding_params, config.embedding_lr, config.weight_decay),
        _adamw_group("unembedding", unembedding_params, config.unembedding_lr, config.weight_decay),
        _adamw_group("scalar_vector", scalar_params, config.scalar_lr, 0.0),
    ]
    for shape in sorted({tuple(param.shape) for param in muon_params}):
        shape_params = [param for param in muon_params if tuple(param.shape) == shape]
        param_groups.append(
            {
                "kind": "muon",
                "params": shape_params,
                "lr": config.matrix_lr,
                "base_lr": config.matrix_lr,
                "weight_decay": config.weight_decay,
                "momentum": 0.95,
                "name": f"muon_matrix_{shape[0]}x{shape[1]}",
            }
        )

    return MuonAdamW(param_groups, excluded=excluded)


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
