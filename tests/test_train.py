from __future__ import annotations

import pytest

from conftest import requires_torch


@requires_torch
def test_mlflow_flattening_helpers() -> None:
    from train import _flatten_for_mlflow, _mlflow_value

    flat = _flatten_for_mlflow({"config": {"depth": 2, "layers": [1, None]}, "ok": True})
    assert flat["config.depth"] == 2
    assert flat["config.layers.0"] == 1
    assert flat["config.layers.1"] is None
    assert flat["ok"] is True
    assert _mlflow_value(None) == "null"
    assert _mlflow_value({"a": 1}) == '{"a": 1}'


@requires_torch
def test_training_data_info_marks_first_batch_overfit() -> None:
    from train import training_data_info

    class Loader:
        def info(self) -> dict:
            return {"data_shuffle_seed": 123, "block_size": 8}

    normal = training_data_info(Loader(), overfit_first_batch=False)
    assert normal["data_shuffle_seed"] == 123
    assert normal["overfit_first_batch"] is False

    overfit = training_data_info(Loader(), overfit_first_batch=True)
    assert overfit["data_shuffle_seed"] == 123
    assert overfit["overfit_first_batch"] is True


@requires_torch
def test_checkpoint_evaluation_steps_match_checkpoint_schedule() -> None:
    from train import checkpoint_evaluation_steps, validation_batch_count

    assert checkpoint_evaluation_steps(start_step=0, num_iterations=11, checkpoint_interval=5) == [5, 10]
    assert checkpoint_evaluation_steps(start_step=5, num_iterations=11, checkpoint_interval=5) == [10]
    assert checkpoint_evaluation_steps(start_step=0, num_iterations=11, checkpoint_interval=20) == [10]
    assert checkpoint_evaluation_steps(
        start_step=0,
        num_iterations=11,
        checkpoint_interval=5,
        eval_final_only=True,
    ) == [10]
    assert validation_batch_count(start_step=0, num_iterations=11, checkpoint_interval=5, val_batches=3) == 6
    assert (
        validation_batch_count(
            start_step=0,
            num_iterations=11,
            checkpoint_interval=5,
            val_batches=3,
            eval_final_only=True,
        )
        == 3
    )
    assert (
        validation_batch_count(
            start_step=0,
            num_iterations=11,
            checkpoint_interval=5,
            val_batches=3,
            eval_enabled=False,
        )
        == 0
    )


@requires_torch
def test_standard_benchmark_suites_use_expected_tasks() -> None:
    from train import STANDARD_EVAL_SUITES

    suites = {suite.name: suite for suite in STANDARD_EVAL_SUITES}
    assert suites["commonsense_0shot"].num_fewshot == 0
    assert suites["lambada_0shot"].num_fewshot == 0
    assert "openbookqa" in suites["commonsense_0shot"].tasks


@requires_torch
def test_benchmark_mlflow_metrics_extracts_numeric_task_results() -> None:
    from train import benchmark_mlflow_metrics

    metrics = benchmark_mlflow_metrics(
        {
            "commonsense_0shot": {
                "results": {
                    "hellaswag": {
                        "acc,none": 0.25,
                        "alias": "HellaSwag",
                    },
                    "boolq": {
                        "acc_norm,none": 0.5,
                        "ignored": True,
                    },
                },
            },
        }
    )

    assert metrics == {
        "benchmark.commonsense_0shot.hellaswag.acc_none": 0.25,
        "benchmark.commonsense_0shot.boolq.acc_norm_none": 0.5,
    }


@requires_torch
def test_format_train_record_keeps_common_columns_aligned() -> None:
    from train import format_train_record

    base = {
        "loss": 5.4991,
        "tokens_seen": 50_000_000,
        "tokens_sec": 101_000,
        "mfu": 0.422,
        "lr_multiplier": 0.481,
        "grad_norm": 1.078,
    }
    train_only = format_train_record({"step": 90, **base}, 4200)
    with_val = format_train_record({"step": 100, **base, "val_loss": 5.5012, "val_bpb": 1.8}, 4200)

    assert [idx for idx, char in enumerate(train_only) if char == "|"] == [16, 31, 42, 54, 67, 80]
    assert [idx for idx, char in enumerate(with_val) if char == "|"][:6] == [16, 31, 42, 54, 67, 80]
    assert train_only.split(" | ")[1:6] == with_val.split(" | ")[1:6]
    assert with_val.endswith(" | val  5.5012 | bpb 1.800")


@requires_torch
def test_next_train_batch_reuses_fixed_batch_without_advancing_prefetcher() -> None:
    from train import next_train_batch

    class Prefetcher:
        def __init__(self) -> None:
            self.calls = 0

        def next(self, prepare_next: bool = True):
            self.calls += 1
            return ("fresh", self.calls, prepare_next)

    prefetcher = Prefetcher()
    fixed = ("fixed", 0)

    assert next_train_batch(prefetcher, fixed) == fixed
    assert next_train_batch(prefetcher, fixed) == fixed
    assert prefetcher.calls == 0
    assert next_train_batch(prefetcher, None, prepare_next=False) == ("fresh", 1, False)


@requires_torch
def test_checkpoint_steps_stop_prefetch_after_last_microbatch() -> None:
    from train import should_prepare_next_train_batch

    assert should_prepare_next_train_batch(2, 0, 5, 2, checkpoint_this_step=True) is True
    assert should_prepare_next_train_batch(2, 1, 5, 2, checkpoint_this_step=True) is False
    assert should_prepare_next_train_batch(2, 1, 5, 2, checkpoint_this_step=False) is True
    assert should_prepare_next_train_batch(4, 1, 5, 2, checkpoint_this_step=False) is False


@requires_torch
def test_apply_warmdown_resume_target_updates_total_budget() -> None:
    from train import apply_warmdown_resume_target, warmdown_target_steps_from_tpp

    class Config:
        def __init__(self) -> None:
            self.num_iterations = 100
            self.target_tokens = 0
            self.scheduled_tokens = 0
            self.train_flops_budget = 0.0
            self.budget_policy = "fixed_steps"
            self.warmup_ratio = 0.05
            self.warmup_steps = 5
            self.scaling_params = 100
            self.global_batch_tokens = 64
            self.estimated_flops_per_token = 10
            self.lr_scheduler = "wsd"
            self.extra = {}

    class Args:
        warmdown_to_target_tpp = 20.0
        warmdown_to_target_steps = None

    target_steps, target_tokens = warmdown_target_steps_from_tpp(20.0, Config())
    assert (target_steps, target_tokens) == (32, 2000)

    config = Config()
    assert apply_warmdown_resume_target(config, start_step=20, args=Args) == 23
    assert config.num_iterations == 32
    assert config.warmup_steps == 2
    assert config.target_tokens == 2000
    assert config.scheduled_tokens == 32 * 64
    assert config.train_flops_budget == 32 * 64 * 10
    assert config.budget_policy == "warmdown_target_tokens_per_param"
    assert config.extra["warmdown_resume"]["checkpoint_start_step"] == 20
    assert config.extra["warmdown_resume"]["decay_start_step"] == 23

    class StepArgs:
        warmdown_to_target_tpp = None
        warmdown_to_target_steps = 40

    step_config = Config()
    assert apply_warmdown_resume_target(step_config, start_step=20, args=StepArgs) == 28
    assert step_config.num_iterations == 40
    assert step_config.warmup_steps == 2
    assert step_config.target_tokens == 40 * 64

    with pytest.raises(ValueError, match="would start before the resumed checkpoint"):
        apply_warmdown_resume_target(Config(), start_step=24, args=Args)

    with pytest.raises(ValueError, match="after the resumed checkpoint"):
        apply_warmdown_resume_target(Config(), start_step=40, args=StepArgs)


@requires_torch
def test_grad_global_norm_reports_without_clipping() -> None:
    import torch

    from train import grad_global_norm

    p1 = torch.nn.Parameter(torch.zeros(2))
    p2 = torch.nn.Parameter(torch.zeros(2))
    p1.grad = torch.tensor([3.0, 4.0])
    p2.grad = torch.tensor([0.0, 12.0])

    assert torch.isclose(grad_global_norm([p1, p2]), torch.tensor(13.0))
