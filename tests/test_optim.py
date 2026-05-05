from __future__ import annotations

from conftest import requires_torch, torch


def _muon_orthogonalize(update):
    update = update.bfloat16()
    update = update / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    if update.size(-2) > update.size(-1):
        for _ in range(5):
            x_t_x = update.mT @ update
            poly = torch.baddbmm(x_t_x, x_t_x, x_t_x, beta=b, alpha=c)
            update = torch.baddbmm(update, update, poly, beta=a)
    else:
        for _ in range(5):
            xx_t = update @ update.mT
            poly = torch.baddbmm(xx_t, xx_t, xx_t, beta=b, alpha=c)
            update = torch.baddbmm(update, poly, update, beta=a)
    return update


@requires_torch
def test_schedules() -> None:
    from optim import lr_multiplier

    assert 0 < lr_multiplier(0, 100, 10, 0.3, 0.1) <= 1
    assert lr_multiplier(20, 100, 10, 0.3, 0.1) == 1.0
    assert lr_multiplier(99, 100, 10, 0.3, 0.1) == 0.1


@requires_torch
def test_muon_step_uses_nesterov_momentum() -> None:
    from optim import MUON_STATE_DTYPE, muon_step_fused

    grads = torch.tensor(
        [[[0.25, -0.50], [1.25, 0.75], [-0.25, 1.50]]],
        dtype=MUON_STATE_DTYPE,
    )
    params = torch.ones(1, 3, 2, dtype=torch.float32)
    momentum_buffer = torch.tensor(
        [[[0.50, 0.25], [-0.75, 0.50], [1.00, -0.25]]],
        dtype=MUON_STATE_DTYPE,
    )
    expected_buffer = momentum_buffer.clone()
    momentum = torch.tensor(0.95)

    expected_buffer.lerp_(grads, 1.0 - momentum.item())
    nesterov_input = grads.lerp(expected_buffer, momentum.item()).bfloat16()
    legacy_input = expected_buffer.bfloat16()
    expected_params = params - _muon_orthogonalize(nesterov_input).to(params.dtype) * 0.1
    legacy_params = params - _muon_orthogonalize(legacy_input).to(params.dtype) * 0.1

    muon_step_fused(grads, params, momentum_buffer, momentum, torch.tensor(0.1), torch.tensor(0.0))

    assert torch.allclose(momentum_buffer, expected_buffer)
    assert torch.allclose(params, expected_params, atol=1e-3, rtol=1e-3)
    assert not torch.allclose(params, legacy_params, atol=1e-3, rtol=1e-3)


@requires_torch
def test_muon_tall_matrix_formula_matches_transpose_formula() -> None:
    from optim import MUON_STATE_DTYPE, muon_step_fused

    grads = torch.tensor(
        [[[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0], [0.125, -0.5]]],
        dtype=MUON_STATE_DTYPE,
    )
    params = torch.ones(1, 4, 2, dtype=torch.float32)
    momentum_buffer = torch.zeros_like(grads)
    momentum = torch.tensor(0.95)

    expected_buffer = momentum_buffer.clone()
    expected_buffer.lerp_(grads, 1.0 - momentum.item())
    expected_update = grads.lerp(expected_buffer, momentum.item()).bfloat16().mT
    expected_update = _muon_orthogonalize(expected_update).mT.to(params.dtype)
    expected_params = params - expected_update * 0.1

    muon_step_fused(grads, params, momentum_buffer, momentum, torch.tensor(0.1), torch.tensor(0.0))

    assert torch.allclose(params, expected_params, atol=1e-3, rtol=1e-3)
