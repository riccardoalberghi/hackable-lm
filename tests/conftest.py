from __future__ import annotations

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None


requires_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")
requires_cuda = pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="CUDA is not available")
