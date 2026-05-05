from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import requires_torch, torch


@requires_torch
def test_manifest_compatibility() -> None:
    from repro import compatibility_warnings

    base = {"config": {"sequence_len": 8, "global_batch_tokens": 16, "scaling_policy": "p", "precision": "bf16"}, "seed": 1, "data": {"manifest": {"tokenizer_hash": "a", "raw_input_sha256": {"x": "1"}}}}
    other = json.loads(json.dumps(base))
    assert compatibility_warnings(base, other) == []
    other["seed"] = 2
    assert compatibility_warnings(base, other)


@requires_torch
def test_repro_compare_cli(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    base = {"config": {"sequence_len": 8, "global_batch_tokens": 16, "scaling_policy": "p", "precision": "bf16"}, "seed": 1, "data": {"manifest": {"tokenizer_hash": "a", "raw_input_sha256": {"x": "1"}, "split_seed": 1}}}
    same = json.loads(json.dumps(base))
    different = json.loads(json.dumps(base))
    different["seed"] = 2
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps(base), encoding="utf-8")
    right.write_text(json.dumps(same), encoding="utf-8")
    ok = subprocess.run([sys.executable, "repro.py", "compare", str(left), str(right), "--fail-on-warning"], cwd=Path.cwd(), text=True, capture_output=True)
    assert ok.returncode == 0
    assert '"compatible": true' in ok.stdout
    right.write_text(json.dumps(different), encoding="utf-8")
    bad = subprocess.run([sys.executable, "repro.py", "compare", str(left), str(right), "--fail-on-warning"], cwd=Path.cwd(), text=True, capture_output=True)
    assert bad.returncode == 1
    assert "seed differs" in bad.stdout


@requires_torch
def test_trusted_checkpoint_load(tmp_path: Path) -> None:
    from repro import load_trusted_checkpoint

    path = tmp_path / "checkpoint.pt"
    payload = {"torch_version": torch.torch_version.TorchVersion(torch.__version__), "tensor": torch.ones(1)}
    torch.save(payload, path)
    loaded = load_trusted_checkpoint(path)
    assert str(loaded["torch_version"]) == str(payload["torch_version"])
    assert torch.equal(loaded["tensor"], payload["tensor"])
