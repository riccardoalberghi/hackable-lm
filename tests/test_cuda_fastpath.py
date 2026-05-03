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
    moved_state = {
        "state": {
            key: {state_key: value.cpu() if torch.is_tensor(value) else value for state_key, value in param_state.items()}
            for key, param_state in state["state"].items()
        },
        "param_groups": state["param_groups"],
    }
    resumed_model = LanguageModel(cfg.model).to(device)
    resumed_opt = create_optimizer(resumed_model, cfg)
    resumed_opt.load_state_dict(moved_state)
    muon_states = [
        resumed_opt.state[group["params"][0]]
        for group in resumed_opt.param_groups
        if group["kind"] == "muon" and group["params"] and "momentum_buffer" in resumed_opt.state[group["params"][0]]
    ]
    assert muon_states
    assert all(param_state["momentum_buffer"].device.type == device.type for param_state in muon_states)
    resumed_opt.zero_grad(set_to_none=True)
    _, resumed_loss = resumed_model(x, x)
    resumed_loss.backward()
    resumed_opt.step()
