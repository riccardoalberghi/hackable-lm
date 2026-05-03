from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import compute_dtype_for_device


def fp8_primitives_available() -> bool:
    return all(
        hasattr(torch, name)
        for name in ("_scaled_mm", "float8_e4m3fn", "float8_e5m2")
    )


def fp8_cuda_supported() -> bool:
    if not torch.cuda.is_available() or not fp8_primitives_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 9)


def _to_fp8(x: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.float().abs().max()
    scale = (fp8_max / amax.double().clamp(min=1e-12)).float()
    x_fp8 = (x.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, scale.reciprocal()


def _to_col_major(x: torch.Tensor) -> torch.Tensor:
    return x.t().contiguous().t()


@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)
        return torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv,
            scale_b=weight_inv,
            out_dtype=input_2d.dtype,
            use_fast_accum=True,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_fp8, input_inv, weight_fp8, weight_inv = ctx.saved_tensors
        grad_output_fp8, grad_output_inv = _to_fp8(grad_output, torch.float8_e5m2)

        grad_input = torch._scaled_mm(
            grad_output_fp8,
            _to_col_major(weight_fp8),
            scale_a=grad_output_inv,
            scale_b=weight_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )
        grad_weight = torch._scaled_mm(
            grad_output_fp8.t().contiguous(),
            _to_col_major(input_fp8),
            scale_a=grad_output_inv,
            scale_b=input_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )
        return grad_input, grad_weight


class Float8Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_cuda:
            weight = self.weight.to(dtype=input.dtype)
            bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
            return F.linear(input, weight, bias)
        input = input.to(compute_dtype_for_device(input.device))
        orig_shape = input.shape
        output = _Float8Matmul.apply(input.reshape(-1, orig_shape[-1]), self.weight)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(dtype=output.dtype)
        return output

    @classmethod
    def from_float(cls, module: nn.Linear) -> "Float8Linear":
        with torch.device("meta"):
            converted = cls(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                dtype=module.weight.dtype,
            )
        converted.weight = module.weight
        converted.bias = module.bias
        return converted


def convert_to_float8_training(
    module: nn.Module,
    *,
    module_filter_fn=None,
) -> nn.Module:
    def _convert(parent: nn.Module, prefix: str = "") -> None:
        for name, child in parent.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(parent, name, Float8Linear.from_float(child))

    _convert(module)
    return module
