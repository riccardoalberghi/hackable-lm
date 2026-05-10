from __future__ import annotations

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
            return {"sampling_policy": "random_packed_spans", "block_size": 8}

    normal = training_data_info(Loader(), overfit_first_batch=False)
    assert normal["sampling_policy"] == "random_packed_spans"
    assert normal["overfit_first_batch"] is False

    overfit = training_data_info(Loader(), overfit_first_batch=True)
    assert overfit["sampling_policy"] == "repeat_first_random_packed_spans_batch"
    assert overfit["overfit_first_batch"] is True


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
def test_format_timing_record_summarizes_phase_ms() -> None:
    from train import format_timing_record, should_time_step

    class Args:
        timing_interval = 5
        timing_warmup = 10

    assert should_time_step(9, Args()) is False
    assert should_time_step(10, Args()) is True
    assert should_time_step(11, Args()) is False
    timing = {
        "step": 10,
        "total_ms": 100.0,
        "phase_ms": {
            "zero_grad": 1.0,
            "data_wait": 2.0,
            "forward_backward": 80.0,
            "prefetch": 3.0,
            "grad_norm": 4.0,
            "lr_update": 0.5,
            "optimizer": 6.0,
        },
        "unaccounted_ms": 3.5,
    }

    line = format_timing_record(timing)

    assert line.startswith("timing step    10 | total   100.0ms")
    assert "fb   80.0ms 80.0%" in line
    assert "opt    6.0ms  6.0%" in line
    assert line.endswith("other    3.5ms  3.5%")


@requires_torch
def test_next_train_batch_reuses_fixed_batch_without_advancing_prefetcher() -> None:
    from train import next_train_batch

    class Prefetcher:
        def __init__(self) -> None:
            self.calls = 0

        def next(self):
            self.calls += 1
            return ("fresh", self.calls)

    prefetcher = Prefetcher()
    fixed = ("fixed", 0)

    assert next_train_batch(prefetcher, fixed) == fixed
    assert next_train_batch(prefetcher, fixed) == fixed
    assert prefetcher.calls == 0
    assert next_train_batch(prefetcher, None) == ("fresh", 1)


@requires_torch
def test_grad_global_norm_reports_without_clipping() -> None:
    import torch

    from train import grad_global_norm

    p1 = torch.nn.Parameter(torch.zeros(2))
    p2 = torch.nn.Parameter(torch.zeros(2))
    p1.grad = torch.tensor([3.0, 4.0])
    p2.grad = torch.tensor([0.0, 12.0])

    assert torch.isclose(grad_global_norm([p1, p2]), torch.tensor(13.0))
