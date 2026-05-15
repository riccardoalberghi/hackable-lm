from __future__ import annotations

import argparse
from array import array
import glob
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from repro import hash_file
from tokenizer import (
    EOS_TOKEN,
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


def save_document_offsets(output: Path, split: str, offsets: array) -> None:
    np.save(output / f"{split}_offsets.npy", np.asarray(offsets, dtype=np.uint64))


def split_stats(offsets: array, doc_text_bytes: array) -> dict[str, int]:
    return {
        "tokens": int(offsets[-1]),
        "text_bytes": int(sum(doc_text_bytes)),
    }


def encode_and_write_splits(
    input_paths: list[Path],
    tokenizer_path: Path,
    output: Path,
    vocab_size: int,
    jsonl_text_field: str,
    val_fraction: float,
    split_seed: int,
    batch_size: int,
) -> dict[str, int]:
    tok = load_tokenizer(tokenizer_path)
    eos_id = tok.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise RuntimeError(f"tokenizer is missing required special token {EOS_TOKEN!r}")
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    train_path = output / "train.bin"
    val_path = output / "val.bin"
    for path in (
        train_path,
        val_path,
        output / "train_offsets.npy",
        output / "val_offsets.npy",
    ):
        path.unlink(missing_ok=True)
    train_path.touch()
    val_path.touch()
    rng = np.random.default_rng(split_seed)
    offsets = {"train": array("Q", [0]), "val": array("Q", [0])}
    doc_text_bytes = {"train": array("Q"), "val": array("Q")}
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None
    progress = tqdm(desc="Tokenizing corpus", unit="docs") if tqdm is not None else None
    for texts in batched(iter_texts(input_paths, jsonl_text_field), batch_size):
        encodings = tok.encode_batch(texts)
        pending_ids = {"train": [], "val": []}
        pending_lengths: dict[str, list[int]] = {"train": [], "val": []}
        pending_text_bytes: dict[str, list[int]] = {"train": [], "val": []}
        for text, encoding in zip(texts, encodings):
            ids = encoding.ids
            ids.append(eos_id)
            text_bytes = len(text.encode("utf-8"))
            split = "val" if rng.random() < val_fraction else "train"
            pending_ids[split].extend(ids)
            pending_lengths[split].append(len(ids))
            pending_text_bytes[split].append(text_bytes)
        for split, path in (("train", train_path), ("val", val_path)):
            if not pending_ids[split]:
                continue
            append_ids(path, pending_ids[split], dtype)
            for length, text_bytes in zip(pending_lengths[split], pending_text_bytes[split]):
                offsets[split].append(offsets[split][-1] + length)
                doc_text_bytes[split].append(text_bytes)
        if progress is not None:
            progress.update(len(texts))
    if progress is not None:
        progress.close()
    if len(offsets["train"]) == 1 and len(offsets["val"]) == 1:
        raise RuntimeError("no documents were found in the input data")
    if len(offsets["train"]) == 1 or len(offsets["val"]) == 1:
        raise RuntimeError("document split produced an empty train or validation split")
    save_document_offsets(output, "train", offsets["train"])
    save_document_offsets(output, "val", offsets["val"])
    train_stats = split_stats(offsets["train"], doc_text_bytes["train"])
    val_stats = split_stats(offsets["val"], doc_text_bytes["val"])
    return {
        "train_tokens": train_stats["tokens"],
        "val_tokens": val_stats["tokens"],
        "train_text_bytes": train_stats["text_bytes"],
        "val_text_bytes": val_stats["text_bytes"],
    }


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
    split_stats = encode_and_write_splits(
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
        "train_tokens": split_stats["train_tokens"],
        "val_tokens": split_stats["val_tokens"],
        "train_text_bytes": split_stats["train_text_bytes"],
        "val_text_bytes": split_stats["val_text_bytes"],
        "dtype": dtype_name,
        "jsonl_text_field": jsonl_text_field,
        "tokenize_batch_size": batch_size,
        "prepare_data_code_hash": hash_file(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tokenizer and build document-offset token memmaps.")
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
        print(
            json.dumps(
                {
                    "output": args.output,
                    "train_tokens": manifest["train_tokens"],
                    "val_tokens": manifest["val_tokens"],
                }
            )
        )


if __name__ == "__main__":
    main()
