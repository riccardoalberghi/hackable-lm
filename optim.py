from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


MUON_NAME_MARKERS = (
    "qkv_proj.weight",
    "o_proj.weight",
    "gate_up_proj.weight",
    "down_proj.weight",
)
MUON_STATE_DTYPE = torch.bfloat16
OPTIMIZER_SCALAR_DTYPE = torch.float32


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
    grad = grad.to(dtype=exp_avg.dtype)
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

    x = stacked_grads.to(momentum_buffer.dtype).lerp(momentum_buffer, momentum).bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    if x.size(-2) > x.size(-1):
        for _ in range(5):
            x_t_x = x.mT @ x
            poly = torch.baddbmm(x_t_x, x_t_x, x_t_x, beta=b, alpha=c)
            x = torch.baddbmm(x, x, poly, beta=a)
    else:
        for _ in range(5):
            xx_t = x @ x.mT
            poly = torch.baddbmm(xx_t, xx_t, xx_t, beta=b, alpha=c)
            x = torch.baddbmm(x, poly, x, beta=a)

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
        self.muon_momentum = muon_group.get("momentum", 0.95) if muon_group is not None else None
        self.muon_lr = muon_group.get("lr", 0.0) if muon_group is not None else None

        self._adamw_step_t = torch.tensor(0.0, dtype=OPTIMIZER_SCALAR_DTYPE, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=OPTIMIZER_SCALAR_DTYPE, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=OPTIMIZER_SCALAR_DTYPE, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=OPTIMIZER_SCALAR_DTYPE, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=OPTIMIZER_SCALAR_DTYPE, device="cpu")
        self._adamw_weight_decay_t = torch.tensor(0.0, dtype=OPTIMIZER_SCALAR_DTYPE, device="cpu")

        self._muon_momentum_t = torch.tensor(0.0, dtype=OPTIMIZER_SCALAR_DTYPE, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=OPTIMIZER_SCALAR_DTYPE, device="cpu")
        self._muon_weight_decay_t = torch.tensor(0.0, dtype=OPTIMIZER_SCALAR_DTYPE, device="cpu")

    def _first_muon_group(self) -> dict[str, Any] | None:
        for group in self.param_groups:
            if group["kind"] == "muon":
                return group
        return None

    def _step_adamw(self, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        for param in group["params"]:
            if param.grad is None:
                continue
            state = self.state[param]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param, dtype=torch.bfloat16)
                state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.bfloat16)
            else:
                for key in ("exp_avg", "exp_avg_sq"):
                    value = state[key]
                    if value.device != param.device or value.dtype != torch.bfloat16:
                        state[key] = value.to(device=param.device, dtype=torch.bfloat16)
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

    def _muon_momentum_buffer(self, state: dict[str, Any], params: list[torch.nn.Parameter]) -> Tensor:
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
        return momentum_buffer

    def _set_muon_scalars(self, group: dict[str, Any]) -> None:
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_lr_t.fill_(group["lr"])
        self._muon_weight_decay_t.fill_(group["weight_decay"])

    def _copy_views(self, dest: list[Tensor], src: Tensor) -> None:
        src_parts = list(src.unbind(0))
        if dest and dest[0].is_cuda:
            torch._foreach_copy_(dest, src_parts)
        else:
            for dest_part, src_part in zip(dest, src_parts):
                dest_part.copy_(src_part)

    def _step_muon_split(self, group: dict[str, Any], split_sizes: tuple[int, ...]) -> None:
        params = group["params"]
        if not params:
            return
        active = [(i, p) for i, p in enumerate(params) if p.grad is not None]
        if not active:
            return

        rows, cols = params[0].shape
        if sum(split_sizes) != rows:
            raise RuntimeError(f"Muon split sizes {split_sizes} do not match parameter rows {rows}")
        offsets: list[tuple[int, int]] = []
        start = 0
        for split_rows in split_sizes:
            offsets.append((start, split_rows))
            start += split_rows

        state = self.state[params[0]]
        momentum_buffer = self._muon_momentum_buffer(state, params)
        self._set_muon_scalars(group)

        equal_splits = len(set(split_sizes)) == 1
        if equal_splits and len(active) == len(params):
            split_rows = split_sizes[0]
            n_splits = len(split_sizes)
            if len(params) == 1:
                param = params[0]
                muon_step_fused(
                    param.grad.view(n_splits, split_rows, cols).to(dtype=MUON_STATE_DTYPE),
                    param.view(n_splits, split_rows, cols),
                    momentum_buffer.view(n_splits, split_rows, cols),
                    self._muon_momentum_t,
                    self._muon_lr_t,
                    self._muon_weight_decay_t,
                )
            else:
                param_views = [param.view(n_splits, split_rows, cols) for param in params]
                stacked_params = torch.stack(param_views).reshape(-1, split_rows, cols)
                stacked_grads = torch.stack(
                    [param.grad.view(n_splits, split_rows, cols).to(dtype=MUON_STATE_DTYPE) for param in params]
                ).reshape(-1, split_rows, cols)
                active_buffer = momentum_buffer.view(len(params), n_splits, split_rows, cols).reshape(-1, split_rows, cols)
                muon_step_fused(
                    stacked_grads,
                    stacked_params,
                    active_buffer,
                    self._muon_momentum_t,
                    self._muon_lr_t,
                    self._muon_weight_decay_t,
                )
                self._copy_views(param_views, stacked_params.view(len(params), n_splits, split_rows, cols))
            self.muon_lr = group["lr"]
            self.muon_momentum = group["momentum"]
            return

        buckets: dict[int, list[tuple[int, torch.nn.Parameter, int]]] = {}
        for param_idx, param in active:
            for start, split_rows in offsets:
                buckets.setdefault(split_rows, []).append((param_idx, param, start))

        for split_rows, entries in buckets.items():
            if len(entries) == 1:
                param_idx, param, start = entries[0]
                muon_step_fused(
                    param.grad.narrow(0, start, split_rows).to(dtype=MUON_STATE_DTYPE).unsqueeze(0),
                    param.narrow(0, start, split_rows).unsqueeze(0),
                    momentum_buffer[param_idx, start : start + split_rows].unsqueeze(0),
                    self._muon_momentum_t,
                    self._muon_lr_t,
                    self._muon_weight_decay_t,
                )
                continue

            param_views = [param.narrow(0, start, split_rows) for _, param, start in entries]
            buffer_views = [momentum_buffer[param_idx, start : start + split_rows] for param_idx, _, start in entries]
            stacked_params = torch.stack(param_views)
            stacked_grads = torch.stack(
                [param.grad.narrow(0, start, split_rows).to(dtype=MUON_STATE_DTYPE) for _, param, start in entries]
            )
            stacked_buffer = torch.stack(buffer_views)
            muon_step_fused(
                stacked_grads,
                stacked_params,
                stacked_buffer,
                self._muon_momentum_t,
                self._muon_lr_t,
                self._muon_weight_decay_t,
            )
            self._copy_views(param_views, stacked_params)
            self._copy_views(buffer_views, stacked_buffer)

        self.muon_lr = group["lr"]
        self.muon_momentum = group["momentum"]

    def _step_muon(self, group: dict[str, Any]) -> None:
        params = group["params"]
        if not params:
            return
        active = [(i, p) for i, p in enumerate(params) if p.grad is not None]
        if not active:
            return

        state = self.state[params[0]]
        momentum_buffer = self._muon_momentum_buffer(state, params)

        active_indices = [i for i, _ in active]
        active_params = [p for _, p in active]
        if len(active_params) == 1:
            stacked_grads = active_params[0].grad.to(dtype=MUON_STATE_DTYPE).unsqueeze(0)
            stacked_params = active_params[0].unsqueeze(0)
            idx = active_indices[0]
            active_buffer = momentum_buffer[idx : idx + 1]
            copy_back = False
        else:
            active_grads = [p.grad.to(dtype=MUON_STATE_DTYPE) for p in active_params]
            stacked_grads = torch.stack(active_grads)
            stacked_params = torch.stack(active_params)
            active_buffer = momentum_buffer if len(active) == len(params) else momentum_buffer[active_indices]
            copy_back = True

        self._set_muon_scalars(group)
        muon_step_fused(
            stacked_grads,
            stacked_params,
            active_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_weight_decay_t,
        )
        if copy_back and len(active) != len(params):
            momentum_buffer[active_indices] = active_buffer

        if copy_back:
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
                split_sizes = group.get("split_sizes")
                if split_sizes is None:
                    self._step_muon(group)
                else:
                    self._step_muon_split(group, tuple(split_sizes))
            else:
                raise ValueError(f"unknown optimizer kind {group['kind']!r}")
        return loss

    def set_lr_multiplier(self, mult: float) -> None:
        for group in self.param_groups:
            group["lr"] = group["base_lr"] * mult
        muon_group = self._first_muon_group()
        self.muon_lr = muon_group["lr"] if muon_group is not None else None

    def summary(self) -> dict[str, Any]:
        groups = [
            {
                "name": group["name"],
                "kind": group["kind"],
                "tensors": len(group["params"]) * len(group.get("split_sizes") or (None,)),
                "params": sum(p.numel() for p in group["params"]),
                "lr": group["base_lr"],
                "weight_decay": group["weight_decay"],
            }
            for group in self.param_groups
        ]
        muon_groups = [group for group in self.param_groups if group["kind"] == "muon"]
        adamw_groups = [group for group in self.param_groups if group["kind"] == "adamw"]
        return {
            "optimizer_kind": "muon_adamw" if muon_groups else "adamw",
            "muon_tensors": sum(len(group["params"]) * len(group.get("split_sizes") or (None,)) for group in muon_groups),
            "adamw_tensors": sum(len(group["params"]) for group in adamw_groups),
            "muon_params": sum(p.numel() for group in muon_groups for p in group["params"]),
            "adamw_params": sum(p.numel() for group in adamw_groups for p in group["params"]),
            "embedding_params": _group_param_count(self.param_groups, "embedding"),
            "unembedding_params": _group_param_count(self.param_groups, "unembedding"),
            "matrix_params": _group_param_count(self.param_groups, "matrix"),
            "scalar_params": _group_param_count(self.param_groups, "scalar_vector"),
            "excluded": self.excluded,
            "groups": groups,
        }


def _group_param_count(param_groups: list[dict[str, Any]], name: str) -> int:
    return sum(p.numel() for group in param_groups if group["name"] == name for p in group["params"])


def _requires_muon(name: str, param: torch.nn.Parameter) -> bool:
    return param.ndim == 2 and any(marker in name for marker in MUON_NAME_MARKERS)


def _muon_split_sizes(name: str, param: torch.nn.Parameter, config: Any) -> tuple[int, ...] | None:
    if name.endswith("qkv_proj.weight"):
        model_config = config.model
        q_dim = model_config.n_head * model_config.head_dim
        kv_dim = model_config.n_kv_head * model_config.head_dim
        split_sizes = (q_dim, kv_dim, kv_dim)
    elif name.endswith("gate_up_proj.weight"):
        split_rows = param.shape[0] // 2
        split_sizes = (split_rows, split_rows)
    else:
        return None
    if sum(split_sizes) != param.shape[0]:
        raise RuntimeError(f"Muon split sizes {split_sizes} do not match {name} shape {tuple(param.shape)}")
    return split_sizes


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
    optimizer_kind = config.optimizer
    if optimizer_kind not in {"muon_adamw", "adamw"}:
        raise ValueError(f"unknown optimizer {optimizer_kind!r}")

    muon_params: list[tuple[str, torch.nn.Parameter, tuple[int, ...] | None]] = []
    embedding_params: list[torch.nn.Parameter] = []
    unembedding_params: list[torch.nn.Parameter] = []
    scalar_params: list[torch.nn.Parameter] = []
    excluded: list[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            excluded.append(name)
        elif _requires_muon(name, param):
            muon_params.append((name, param, _muon_split_sizes(name, param, config)))
        elif "tok_emb" in name:
            embedding_params.append(param)
        elif "lm_head" in name:
            unembedding_params.append(param)
        else:
            scalar_params.append(param)

    param_groups = [
        _adamw_group("embedding", embedding_params, config.embedding_lr, config.weight_decay),
        _adamw_group("unembedding", unembedding_params, config.unembedding_lr, config.weight_decay),
    ]
    if optimizer_kind == "adamw":
        matrix_params = [param for _, param, _ in muon_params]
        param_groups.extend(
            [
                _adamw_group("matrix", matrix_params, config.matrix_lr, config.weight_decay),
                _adamw_group("scalar_vector", scalar_params, config.scalar_lr, 0.0),
            ]
        )
        return MuonAdamW(param_groups, excluded=excluded)

    param_groups.append(_adamw_group("scalar_vector", scalar_params, config.scalar_lr, 0.0))
    group_keys = sorted({(tuple(param.shape), split_sizes) for _, param, split_sizes in muon_params}, key=str)
    for shape, split_sizes in group_keys:
        shape_params = [param for _, param, param_split_sizes in muon_params if tuple(param.shape) == shape and param_split_sizes == split_sizes]
        name = f"muon_matrix_{shape[0]}x{shape[1]}"
        if split_sizes is not None:
            name = f"{name}_split_{'x'.join(str(size) for size in split_sizes)}"
        param_groups.append(
            {
                "kind": "muon",
                "params": shape_params,
                "lr": config.matrix_lr,
                "base_lr": config.matrix_lr,
                "weight_decay": config.weight_decay,
                "momentum": 0.95,
                "name": name,
                "split_sizes": split_sizes,
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
    decay_start_step: int | None = None,
) -> float:
    if scheduler != "wsd":
        raise ValueError(f"unsupported LR scheduler {scheduler!r}")
    warmup_steps = min(warmup_steps, num_iterations)
    if decay_start_step is None:
        decay_iters = int(num_iterations * warmdown_ratio)
        decay_start = max(warmup_steps, num_iterations - decay_iters)
    else:
        decay_start = int(decay_start_step)
        if decay_start < warmup_steps:
            raise ValueError(f"decay_start_step={decay_start} must be >= warmup_steps={warmup_steps}")
        if decay_start > num_iterations:
            raise ValueError(f"decay_start_step={decay_start} must be <= num_iterations={num_iterations}")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step < decay_start:
        return 1.0
    decay_span = num_iterations - decay_start - 1
    progress = (step - decay_start) / decay_span if decay_span else 1.0
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr_frac + (1.0 - final_lr_frac) * cosine
