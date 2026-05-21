from __future__ import annotations

from types import SimpleNamespace

import pytest

from conftest import requires_torch


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        table = {
            "Answer:": [10],
            "Answer: ": [10, 11],
            "Answer: yes": [10, 20],
            "yes": [21],
            " yes": [20],
        }
        return table[text]

    def decode(self, ids: list[int]) -> str:
        return str(ids)


@requires_torch
def test_harness_encode_pair_moves_trailing_context_space_to_continuation() -> None:
    pytest.importorskip("lm_eval")
    from lm_eval_hackable_lm import SimpleLMHarness

    harness = SimpleLMHarness.__new__(SimpleLMHarness)
    harness.tokenizer = FakeTokenizer()
    harness._eot_token_id = 0
    harness._config = SimpleNamespace(model=SimpleNamespace(block_size=16))

    scored = harness._encode_pair("Answer: ", "yes")

    assert scored.tokens == [10, 20]
    assert scored.cont_start == 1
    assert scored.cont_len == 1
