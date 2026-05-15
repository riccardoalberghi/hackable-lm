from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from _rustbpe import Tokenizer, load_from_file, train_from_iterator

EOS_TOKEN = "<|endoftext|>"
BOS_TOKEN = "<|beginofsequence|>"
SPECIAL_TOKENS = [EOS_TOKEN, BOS_TOKEN]
TOKENIZER_BACKEND = "hackable_lm_rustbpe_bytelevel"


def iter_texts(paths: Iterable[str | Path], jsonl_text_field: str = "text") -> Iterable[str]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    text = obj.get(jsonl_text_field)
                    if isinstance(text, str) and text.strip():
                        yield text
        else:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                yield text


def train_tokenizer(
    input_paths: list[str | Path],
    output_path: str | Path,
    vocab_size: int,
    jsonl_text_field: str = "text",
    min_frequency: int = 2,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_from_iterator(
        iter_texts(input_paths, jsonl_text_field),
        str(output_path),
        vocab_size,
        min_frequency,
        SPECIAL_TOKENS,
    )


def load_tokenizer(path: str | Path) -> Tokenizer:
    return load_from_file(str(path))


def encode(tokenizer, text: str, add_eos: bool = True) -> list[int]:
    ids = tokenizer.encode(text).ids
    if add_eos:
        eos_id = tokenizer.token_to_id(EOS_TOKEN)
        if eos_id is not None:
            ids.append(eos_id)
    return ids


def decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)


def tokenizer_manifest(path: str | Path) -> dict:
    from repro import hash_file

    tok = load_tokenizer(path)
    return {
        "tokenizer_path": str(path),
        "tokenizer_hash": hash_file(path),
        "vocab_size": tok.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "backend": TOKENIZER_BACKEND,
    }
