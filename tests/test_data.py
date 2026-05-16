from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import requires_torch, torch
from tokenizer import BOS_TOKEN, TOKENIZER_BACKEND, tokenizer_impl_hash, load_tokenizer


@requires_torch
def test_preprocess_and_memmap(tmp_path: Path) -> None:
    from data import MemmapDataLoader
    from prepare_data import prepare_all

    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(json.dumps({"text": f"document {i} has unique text {i * 17}"}) for i in range(200)),
        encoding="utf-8",
    )
    out = tmp_path / "processed"
    manifest = prepare_all([str(raw)], out, vocab_size=300, val_fraction=0.2, min_frequency=1)
    assert (out / "train.bin").exists()
    assert (out / "train_offsets.npy").exists()
    assert (out / "val_offsets.npy").exists()
    assert manifest["train_tokens"] > 16
    assert manifest["train_text_bytes"] > 0
    assert manifest["val_text_bytes"] > 0
    assert manifest["tokenizer_backend"] == TOKENIZER_BACKEND
    assert manifest["tokenizer_format"] == "hackablebpe"
    assert manifest["tokenizer_impl_hash"] == tokenizer_impl_hash()
    assert manifest["min_frequency"] == 1
    capacity_loader = MemmapDataLoader(out, block_size=8, data_shuffle_seed=123)
    available = capacity_loader.available_spans("train")
    assert available >= 4
    capacity_loader.require_batches("train", 1, 4)
    with pytest.raises(RuntimeError, match="requires"):
        capacity_loader.require_batches("train", available + 1, 1)

    loader = MemmapDataLoader(out, block_size=8, data_shuffle_seed=123)
    x, y = loader.get_batch("train", 4, "cpu")
    bos_id = load_tokenizer(out / "tokenizer.json").token_to_id(BOS_TOKEN)
    assert bos_id is not None
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == torch.long
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert torch.equal(x[:, 0], torch.full((4,), bos_id, dtype=torch.long))
    assert torch.equal(x[:, 1:], y[:, :-1])

    same_loader = MemmapDataLoader(out, block_size=8, data_shuffle_seed=123)
    same_x, same_y = same_loader.get_batch("train", 4, "cpu")
    assert torch.equal(x, same_x)
    assert torch.equal(y, same_y)

    different_loader = MemmapDataLoader(out, block_size=8, data_shuffle_seed=456)
    different_x, different_y = different_loader.get_batch("train", 4, "cpu")
    assert not (torch.equal(x, different_x) and torch.equal(y, different_y))
