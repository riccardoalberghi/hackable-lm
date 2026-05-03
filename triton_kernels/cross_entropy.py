from __future__ import annotations

import torch
import torch.nn.functional as F


def torch_reference(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

