from __future__ import annotations

import pytest

from config import resolve_config
from conftest import requires_cuda, requires_torch, torch


@pytest.mark.cuda
@requires_torch
@requires_cuda
def test_optimizer_resume_muon_state_device() -> None:
    from model import LanguageModel
    from optim import create_optimizer

    cfg = resolve_config(
        depth=2,
        vocab_size=128,
        sequence_len=8,
        device_batch_size=2,
        precision="fp32_test",
        compile_model=False,
        norm_backend="torch",
        loss_backend="torch",
    )
    device = torch.device("cuda")
    model = LanguageModel(cfg.model).to(device)
    opt = create_optimizer(model, cfg)
    x = torch.randint(0, cfg.model.vocab_size, (2, cfg.sequence_len), device=device)
    _, loss = model(x, x)
    loss.backward()
    opt.step()
    state = opt.state_dict()
    moved_state = {"muon_state": {}, **{k: v for k, v in state.items() if k != "muon_state"}}
    for name, param_state in state["muon_state"].items():
        moved_state["muon_state"][name] = {
            key: value.cpu() if torch.is_tensor(value) else value for key, value in param_state.items()
        }
    resumed_model = LanguageModel(cfg.model).to(device)
    resumed_opt = create_optimizer(resumed_model, cfg)
    resumed_opt.load_state_dict(moved_state)
    assert all(param_state["momentum_buffer"].device.type == device.type for param_state in resumed_opt.muon_state.values())
    resumed_opt.zero_grad(set_to_none=True)
    _, resumed_loss = resumed_model(x, x)
    resumed_loss.backward()
    resumed_opt.step()
