from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenizer import (
    SPECIAL_TOKENS,
    TOKENIZER_BACKEND,
    decode,
    encode,
    load_tokenizer,
    tokenizer_manifest,
    train_tokenizer,
)


def _train(tmp_path: Path, text: str, vocab_size: int = 300):
    raw = tmp_path / "raw.txt"
    raw.write_text(text, encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    train_tokenizer([raw], tokenizer_path, vocab_size=vocab_size, min_frequency=1)
    return load_tokenizer(tokenizer_path), tokenizer_path


def test_rust_tokenizer_round_trips_utf8(tmp_path: Path) -> None:
    tok, _ = _train(tmp_path, "hello hello\ncafe \u00e9 \U0001f680\n")
    text = "hello cafe \u00e9 \U0001f680"
    assert decode(tok, encode(tok, text, add_eos=False)) == text


def test_rust_tokenizer_exposes_existing_api_shape(tmp_path: Path) -> None:
    tok, _ = _train(tmp_path, "abababab\n")
    one = tok.encode("abab").ids
    batch = [encoding.ids for encoding in tok.encode_batch(["ab", "abab"])]
    assert one
    assert batch[0]
    assert batch[1] == one
    assert tok.token_to_id(SPECIAL_TOKENS[0]) == 0
    assert tok.get_vocab_size() > 256


def test_encode_appends_eos_and_decode_skips_specials(tmp_path: Path) -> None:
    tok, _ = _train(tmp_path, "hello\n")
    ids = encode(tok, "hello", add_eos=True)
    assert ids[-1] == tok.token_to_id(SPECIAL_TOKENS[0])
    assert decode(tok, ids) == "hello"


def test_training_is_deterministic(tmp_path: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("abc abc abd\n" * 20, encoding="utf-8")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    train_tokenizer([raw], first, vocab_size=280, min_frequency=1)
    train_tokenizer([raw], second, vocab_size=280, min_frequency=1)
    assert json.loads(first.read_text(encoding="utf-8")) == json.loads(second.read_text(encoding="utf-8"))


def test_vocab_size_must_include_byte_alphabet(tmp_path: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("tiny", encoding="utf-8")
    with pytest.raises(ValueError, match="vocab_size must be at least"):
        train_tokenizer([raw], tmp_path / "tokenizer.json", vocab_size=128, min_frequency=1)


def test_manifest_records_rust_backend(tmp_path: Path) -> None:
    _, tokenizer_path = _train(tmp_path, "hello world\n")
    manifest = tokenizer_manifest(tokenizer_path)
    assert manifest["backend"] == TOKENIZER_BACKEND
    assert manifest["special_tokens"] == SPECIAL_TOKENS


def test_prepare_all_uses_rust_tokenizer_without_torch(tmp_path: Path) -> None:
    from prepare_data import prepare_all

    raw = tmp_path / "raw.txt"
    raw.write_text("hello world\n" * 40, encoding="utf-8")
    out = tmp_path / "processed"
    manifest = prepare_all([str(raw)], out, vocab_size=300, val_fraction=0.2, min_frequency=1)
    assert (out / "tokenizer.json").exists()
    assert (out / "train.bin").exists()
    assert (out / "val.bin").exists()
    assert manifest["tokenizer_backend"] == TOKENIZER_BACKEND
    assert manifest["preprocessing"] == f"{TOKENIZER_BACKEND}_packed_contiguous"
    assert manifest["min_frequency"] == 1
    assert manifest["vocab_size"] <= 300
    assert manifest["train_tokens"] > 0
    assert manifest["val_tokens"] > 0
