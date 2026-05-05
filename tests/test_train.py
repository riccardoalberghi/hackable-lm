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
