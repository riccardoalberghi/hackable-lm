from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from repro import hash_file
from tokenizer import (
    SPECIAL_TOKENS,
    TOKENIZER_BACKEND,
    iter_texts,
    load_tokenizer,
    tokenizer_manifest,
    train_tokenizer,
)


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern)]
        paths.extend(matches or [Path(pattern)])
    paths = sorted({p.resolve() for p in paths})
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"input files not found: {missing}")
    return paths


def batched(iterable: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def append_ids(path: Path, ids: list[int], dtype: np.dtype) -> int:
    arr = np.asarray(ids, dtype=dtype)
    with path.open("ab") as fh:
        arr.tofile(fh)
    return int(arr.size)


def rebalance_empty_split(output: Path, dtype: np.dtype, val_fraction: float, train_bytes: int, val_bytes: int) -> tuple[int, int, int, int]:
    train_path = output / "train.bin"
    val_path = output / "val.bin"
    train = np.fromfile(train_path, dtype=dtype) if train_path.exists() else np.asarray([], dtype=dtype)
    val = np.fromfile(val_path, dtype=dtype) if val_path.exists() else np.asarray([], dtype=dtype)
    all_ids = np.concatenate([train, val])
    if all_ids.size < 2:
        raise RuntimeError("need at least two tokens to create non-empty train and validation splits")
    val_count = min(all_ids.size - 1, max(1, int(round(all_ids.size * val_fraction))))
    train_ids = all_ids[:-val_count]
    val_ids = all_ids[-val_count:]
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    total_bytes = train_bytes + val_bytes
    if total_bytes > 0:
        val_byte_count = min(total_bytes - 1, max(1, int(round(total_bytes * val_fraction))))
        train_byte_count = total_bytes - val_byte_count
    else:
        train_byte_count = 0
        val_byte_count = 0
    return int(train_ids.size), int(val_ids.size), train_byte_count, val_byte_count


def encode_and_write_splits(
    input_paths: list[Path],
    tokenizer_path: Path,
    output: Path,
    vocab_size: int,
    jsonl_text_field: str,
    val_fraction: float,
    split_seed: int,
    batch_size: int,
) -> tuple[int, int, int, int]:
    tok = load_tokenizer(tokenizer_path)
    eos_id = tok.token_to_id(SPECIAL_TOKENS[0])
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    train_path = output / "train.bin"
    val_path = output / "val.bin"
    train_path.unlink(missing_ok=True)
    val_path.unlink(missing_ok=True)
    rng = np.random.default_rng(split_seed)
    train_tokens = 0
    val_tokens = 0
    train_bytes = 0
    val_bytes = 0
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None
    progress = tqdm(desc="Tokenizing corpus", unit="docs") if tqdm is not None else None
    for texts in batched(iter_texts(input_paths, jsonl_text_field), batch_size):
        encodings = tok.encode_batch(texts)
        train_ids: list[int] = []
        val_ids: list[int] = []
        for text, encoding in zip(texts, encodings):
            ids = encoding.ids
            if eos_id is not None:
                ids.append(eos_id)
            text_bytes = len(text.encode("utf-8"))
            if rng.random() < val_fraction:
                val_ids.extend(ids)
                val_bytes += text_bytes
            else:
                train_ids.extend(ids)
                train_bytes += text_bytes
        if train_ids:
            train_tokens += append_ids(train_path, train_ids, dtype)
        if val_ids:
            val_tokens += append_ids(val_path, val_ids, dtype)
        if progress is not None:
            progress.update(len(texts))
    if progress is not None:
        progress.close()
    if val_tokens == 0 or train_tokens == 0:
        return rebalance_empty_split(output, dtype, val_fraction, train_bytes, val_bytes)
    return train_tokens, val_tokens, train_bytes, val_bytes


def prepare_all(
    input_patterns: list[str],
    output: str | Path,
    vocab_size: int,
    jsonl_text_field: str = "text",
    val_fraction: float = 0.01,
    split_seed: int = 1337,
    min_frequency: int = 2,
    batch_size: int = 2048,
) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    input_paths = expand_inputs(input_patterns)
    tokenizer_path = output / "tokenizer.json"
    train_tokenizer(
        input_paths,
        tokenizer_path,
        vocab_size,
        jsonl_text_field,
        min_frequency,
    )
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    dtype_name = "uint16" if dtype == np.uint16 else "uint32"
    train_tokens, val_tokens, train_bytes, val_bytes = encode_and_write_splits(
        input_paths,
        tokenizer_path,
        output,
        vocab_size,
        jsonl_text_field,
        val_fraction,
        split_seed,
        batch_size,
    )
    tok_info = tokenizer_manifest(tokenizer_path)
    manifest = {
        "raw_input_paths": [str(p) for p in input_paths],
        "raw_input_file_sizes": {str(p): p.stat().st_size for p in input_paths},
        "raw_input_sha256": {str(p): hash_file(p) for p in input_paths},
        "tokenizer_hash": tok_info["tokenizer_hash"],
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_backend": tok_info["backend"],
        "vocab_size": tok_info["vocab_size"],
        "requested_vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": SPECIAL_TOKENS,
        "split_seed": split_seed,
        "split_policy": "streaming_document_bernoulli",
        "val_fraction": val_fraction,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_text_bytes": train_bytes,
        "val_text_bytes": val_bytes,
        "dtype": dtype_name,
        "jsonl_text_field": jsonl_text_field,
        "tokenize_batch_size": batch_size,
        "preprocessing": f"{TOKENIZER_BACKEND}_packed_contiguous",
        "prepare_data_code_hash": hash_file(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tokenizer and build packed token memmaps.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    all_p = sub.add_parser("all")
    all_p.add_argument("--input", nargs="+", required=True)
    all_p.add_argument("--vocab-size", type=int, required=True)
    all_p.add_argument("--output", required=True)
    all_p.add_argument("--jsonl-text-field", default="text")
    all_p.add_argument("--val-fraction", type=float, default=0.01)
    all_p.add_argument("--split-seed", type=int, default=1337)
    all_p.add_argument("--min-frequency", type=int, default=2)
    all_p.add_argument("--batch-size", type=int, default=2048, help="documents per batched tokenizer encode call")
    args = parser.parse_args()
    if args.cmd == "all":
        manifest = prepare_all(
            args.input,
            args.output,
            args.vocab_size,
            args.jsonl_text_field,
            args.val_fraction,
            args.split_seed,
            args.min_frequency,
            args.batch_size,
        )
        print(json.dumps({"output": args.output, "train_tokens": manifest["train_tokens"], "val_tokens": manifest["val_tokens"]}))


if __name__ == "__main__":
    main()
