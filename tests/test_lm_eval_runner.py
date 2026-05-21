from __future__ import annotations

from pathlib import Path


def test_standard_suite_uses_group_specific_fewshot_settings() -> None:
    from run_lm_eval import SUITES

    suites = {suite.name: suite for suite in SUITES["standard"]}
    assert suites["commonsense_0shot"].num_fewshot == 0
    assert suites["lambada_0shot"].num_fewshot == 0
    assert suites["mmlu_5shot"].num_fewshot == 5
    assert "openbookqa" in suites["commonsense_0shot"].tasks


def test_parse_task_list_rejects_empty_lists() -> None:
    import pytest

    from run_lm_eval import parse_task_list

    assert parse_task_list("hellaswag, piqa") == ["hellaswag", "piqa"]
    with pytest.raises(ValueError, match="empty"):
        parse_task_list(" , ")


def test_manifest_output_path_sits_next_to_results() -> None:
    from run_lm_eval import manifest_output_path

    assert manifest_output_path(Path("runs/d12/eval/lm_eval_results.json")) == Path(
        "runs/d12/eval/lm_eval_results.manifest.json"
    )
