from __future__ import annotations

from conftest import requires_torch


@requires_torch
def test_schedules() -> None:
    from optim import lr_multiplier

    assert 0 < lr_multiplier(0, 100, 10, 0.3, 0.1) <= 1
    assert lr_multiplier(20, 100, 10, 0.3, 0.1) == 1.0
    assert lr_multiplier(99, 100, 10, 0.3, 0.1) == 0.1
