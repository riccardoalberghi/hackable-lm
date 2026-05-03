from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from repro import hash_file
from tokenizer import SPECIAL_TOKENS, encode, iter_texts, load_tokenizer, train_tokenizer, tokenizer_manifest


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


def write_memmap(path: Path, ids: list[int], dtype: np.dtype) -> None:
    arr = np.asarray(ids, dtype=dtype)
    mmap = np.memmap(path, dtype=dtype, mode="w+", shape=arr.shape)
    mmap[:] = arr[:]
    mmap.flush()


def prepare_all(
    input_patterns: list[str],
    output: str | Path,
    vocab_size: int,
    jsonl_text_field: str = "text",
    val_fraction: float = 0.01,
    split_seed: int = 1337,
    min_frequency: int = 2,
) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    input_paths = expand_inputs(input_patterns)
    tokenizer_path = output / "tokenizer.json"
    train_tokenizer(input_paths, tokenizer_path, vocab_size, jsonl_text_field, min_frequency)
    tok = load_tokenizer(tokenizer_path)
    docs = [encode(tok, text, add_eos=True) for text in iter_texts(input_paths, jsonl_text_field)]
    rng = np.random.default_rng(split_seed)
    rng.shuffle(docs)
    if len(docs) > 1:
        val_docs = max(1, int(round(len(docs) * val_fraction)))
        val_ids = [tid for doc in docs[:val_docs] for tid in doc]
        train_ids = [tid for doc in docs[val_docs:] for tid in doc]
    else:
        ids = docs[0]
        cut = max(1, int(len(ids) * (1 - val_fraction)))
        train_ids, val_ids = ids[:cut], ids[cut:]
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    dtype_name = "uint16" if dtype == np.uint16 else "uint32"
    write_memmap(output / "train.bin", train_ids, dtype)
    write_memmap(output / "val.bin", val_ids, dtype)
    tok_info = tokenizer_manifest(tokenizer_path)
    manifest = {
        "raw_input_paths": [str(p) for p in input_paths],
        "raw_input_file_sizes": {str(p): p.stat().st_size for p in input_paths},
        "raw_input_sha256": {str(p): hash_file(p) for p in input_paths},
        "tokenizer_hash": tok_info["tokenizer_hash"],
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": tok.get_vocab_size(),
        "requested_vocab_size": vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "dtype": dtype_name,
        "jsonl_text_field": jsonl_text_field,
        "preprocessing": "bytelevel_bpe_packed_contiguous",
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
        )
        print(json.dumps({"output": args.output, "train_tokens": manifest["train_tokens"], "val_tokens": manifest["val_tokens"]}))


if __name__ == "__main__":
    main()
