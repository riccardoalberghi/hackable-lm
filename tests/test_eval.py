from __future__ import annotations

from pathlib import Path

from conftest import requires_torch


def test_prepare_eval_smoke_tasks(tmp_path: Path) -> None:
    from prepare_eval import DEFAULT_TASKS, prepare_eval_data

    out = tmp_path / "eval_data"
    manifest = prepare_eval_data(out, tasks=DEFAULT_TASKS, source="smoke")
    assert sorted(manifest["tasks"]) == sorted(DEFAULT_TASKS)
    for task, entry in manifest["tasks"].items():
        path = Path(entry["file"])
        assert path.exists()
        assert entry["smoke_only"] is True
        assert entry["examples"] == 2
        assert len(entry["sha256"]) == 64


@requires_torch
def test_eval_task_file_provenance(tmp_path: Path) -> None:
    from eval import task_file_provenance
    from prepare_eval import prepare_eval_data

    out = tmp_path / "eval_data"
    manifest = prepare_eval_data(out, tasks=["boolq"], source="smoke")
    provenance = task_file_provenance(out, ["boolq", "validation_loss"], manifest)
    assert sorted(provenance) == ["boolq"]
    assert provenance["boolq"]["sha256"] == manifest["tasks"]["boolq"]["sha256"]
    assert provenance["boolq"]["examples"] == 2
