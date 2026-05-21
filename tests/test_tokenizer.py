from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    SPECIAL_TOKENS,
    TOKENIZER_BACKEND,
    decode,
    encode,
    load_tokenizer,
    tokenizer_manifest,
    tokenizer_impl_hash,
    train_tokenizer,
)


def _train(tmp_path: Path, text: str, vocab_size: int = 300):
    raw = tmp_path / "raw.txt"
    raw.write_text(text, encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    train_tokenizer([raw], tokenizer_path, vocab_size=vocab_size, min_frequency=1)
    return load_tokenizer(tokenizer_path), tokenizer_path


def test_rust_trained_tokenizer_round_trips_utf8(tmp_path: Path) -> None:
    tok, _ = _train(tmp_path, "hello hello\ncafe \u00e9 \U0001f680\n")
    text = "hello cafe \u00e9 \U0001f680"
    assert decode(tok, encode(tok, text, add_eos=False)) == text


def test_rust_trained_tokenizer_exposes_existing_api_shape(tmp_path: Path) -> None:
    tok, _ = _train(tmp_path, "abababab\n")
    one = tok.encode("abab")
    batch = tok.encode_batch(["ab", "abab"])
    assert one
    assert batch[0]
    assert batch[1] == one
    assert tok.token_to_id(EOS_TOKEN) == 0
    assert tok.token_to_id(BOS_TOKEN) == 1
    assert len(SPECIAL_TOKENS) + 256 < tok.get_vocab_size() <= 300


def test_rust_trainer_learns_byte_pair_merges(tmp_path: Path) -> None:
    tok, tokenizer_path = _train(tmp_path, "aaaa\n", vocab_size=len(SPECIAL_TOKENS) + 257)
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    a = len(SPECIAL_TOKENS) + ord("a")
    merged = len(SPECIAL_TOKENS) + 256
    assert payload["format"] == "hackablebpe"
    assert payload["merges"][0] == {"id": merged, "left": a, "right": a}
    assert tok.encode("aa") == [merged]
    assert tok.encode_batch(["aa", "aaaa"])[0] == [merged]
    assert tok.decode([merged]) == "aa"


def test_rust_runtime_encodes_special_tokens(tmp_path: Path) -> None:
    tok, _ = _train(tmp_path, "hello\n")
    ids = tok.encode(f"a{EOS_TOKEN}b")
    assert tok.token_to_id(EOS_TOKEN) in ids
    assert tok.decode(ids) == "ab"


def test_encode_appends_eos_and_decode_skips_specials(tmp_path: Path) -> None:
    tok, _ = _train(tmp_path, "hello\n")
    ids = encode(tok, "hello", add_eos=True)
    assert ids[-1] == tok.token_to_id(EOS_TOKEN)
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
    assert manifest["format"] == "hackablebpe"
    assert manifest["tokenizer_impl_hash"] == tokenizer_impl_hash()
    assert manifest["special_tokens"] == SPECIAL_TOKENS


def test_prepare_all_uses_rust_tokenizer_without_torch(tmp_path: Path) -> None:
    from prepare_data import prepare_all

    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(json.dumps({"text": f"hello world document {i}"}) for i in range(40)),
        encoding="utf-8",
    )
    out = tmp_path / "processed"
    manifest = prepare_all([str(raw)], out, vocab_size=300, val_fraction=0.2, min_frequency=1)
    assert (out / "tokenizer.json").exists()
    assert (out / "train.bin").exists()
    assert (out / "val.bin").exists()
    assert (out / "train.jsonl").exists()
    assert (out / "val.jsonl").exists()
    assert (out / "train_offsets.npy").exists()
    assert (out / "val_offsets.npy").exists()
    assert manifest["tokenizer_backend"] == TOKENIZER_BACKEND
    assert manifest["tokenizer_format"] == "hackablebpe"
    assert manifest["tokenizer_impl_hash"] == tokenizer_impl_hash()
    assert manifest["min_frequency"] == 1
    assert manifest["vocab_size"] <= 300
    assert manifest["train_tokens"] > 0
    assert manifest["val_tokens"] > 0
    assert manifest["split_raw_paths"] == {
        "train": str(out / "train.jsonl"),
        "val": str(out / "val.jsonl"),
    }
    assert manifest["split_raw_documents"]["train"] > 0
    assert manifest["split_raw_documents"]["val"] > 0


def test_prepare_all_trains_tokenizer_on_train_split_only(tmp_path: Path) -> None:
    from prepare_data import prepare_all

    train_text = "ab" * 20
    val_text = "z" * 200
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps({"text": train_text}),
                json.dumps({"text": val_text}),
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "processed"
    manifest = prepare_all(
        [str(raw)],
        out,
        vocab_size=len(SPECIAL_TOKENS) + 257,
        val_fraction=0.5,
        split_seed=0,
        min_frequency=1,
    )

    payload = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    z_id = len(SPECIAL_TOKENS) + ord("z")
    assert all((merge["left"], merge["right"]) != (z_id, z_id) for merge in payload["merges"])
    assert manifest["tokenizer_training_split"] == "train"
    assert manifest["tokenizer_training_documents"] == 1
    assert manifest["tokenizer_training_text_bytes"] == len(train_text.encode("utf-8"))
    assert manifest["train_text_bytes"] == len(train_text.encode("utf-8"))
    assert manifest["val_text_bytes"] == len(val_text.encode("utf-8"))
    assert (out / "train.jsonl").read_text(encoding="utf-8") == json.dumps({"text": train_text}) + "\n"
    assert (out / "val.jsonl").read_text(encoding="utf-8") == json.dumps({"text": val_text}) + "\n"
