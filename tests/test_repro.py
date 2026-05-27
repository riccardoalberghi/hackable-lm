from __future__ import annotations

from pathlib import Path

from conftest import requires_torch, torch


@requires_torch
def test_trusted_checkpoint_load(tmp_path: Path) -> None:
    from repro import load_trusted_checkpoint

    path = tmp_path / "checkpoint.pt"
    payload = {"torch_version": torch.torch_version.TorchVersion(torch.__version__), "tensor": torch.ones(1)}
    torch.save(payload, path)
    loaded = load_trusted_checkpoint(path)
    assert str(loaded["torch_version"]) == str(payload["torch_version"])
    assert torch.equal(loaded["tensor"], payload["tensor"])
