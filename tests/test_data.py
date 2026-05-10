from __future__ import annotations

from pathlib import Path

from conftest import requires_torch, torch
from tokenizer import TOKENIZER_BACKEND


@requires_torch
def test_preprocess_and_memmap(tmp_path: Path) -> None:
    from data import MemmapDataLoader
    from prepare_data import prepare_all

    raw = tmp_path / "raw.txt"
    raw.write_text(("hello world. this is a tiny language model corpus.\n" * 200), encoding="utf-8")
    out = tmp_path / "processed"
    manifest = prepare_all([str(raw)], out, vocab_size=300, val_fraction=0.2, min_frequency=1)
    assert (out / "train.bin").exists()
    assert manifest["train_tokens"] > 16
    assert manifest["train_text_bytes"] > 0
    assert manifest["val_text_bytes"] > 0
    assert manifest["tokenizer_backend"] == TOKENIZER_BACKEND
    assert manifest["preprocessing"] == f"{TOKENIZER_BACKEND}_packed_contiguous"
    assert manifest["min_frequency"] == 1
    loader = MemmapDataLoader(out, block_size=8, seed=123)
    x, y = loader.get_batch("train", 4, "cpu")
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == torch.long
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert torch.equal(x[:, 1:], y[:, :-1])
