from __future__ import annotations

import json
import os
import importlib
import importlib.machinery
import importlib.util
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

EOS_TOKEN = "<|endoftext|>"
BOS_TOKEN = "<|beginofsequence|>"
SPECIAL_TOKENS = [EOS_TOKEN, BOS_TOKEN]
TOKENIZER_BACKEND = "hackablebpe_bytelevel"
TOKENIZER_FORMAT = "hackablebpe"
TOKENIZER_VERSION = 1
_MODULE_NAME = "_hackablebpe"
_TRAINER_BIN = "hackablebpe_train.exe" if os.name == "nt" else "hackablebpe_train"
_ROOT = Path(__file__).resolve().parent
_CRATE = _ROOT / "hackablebpe"


class Tokenizer:
    """Python wrapper for the local Rust byte-level BPE tokenizer."""

    def __init__(self, native, backend: str, format: str):
        self._native = native
        self.backend = backend
        self.format = format

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        fmt = payload.get("format")
        if fmt != TOKENIZER_FORMAT:
            raise ValueError(f"unsupported tokenizer format {fmt!r}; expected {TOKENIZER_FORMAT!r}")
        if payload.get("version") != TOKENIZER_VERSION:
            raise ValueError(f"unsupported tokenizer version {payload.get('version')!r}; expected {TOKENIZER_VERSION}")
        native = _load_native().load_from_file(str(path))
        return cls(
            native=native,
            backend=payload.get("backend", TOKENIZER_BACKEND),
            format=fmt,
        )

    def encode(self, text: str) -> list[int]:
        return self._native.encode(text)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self._native.encode_batch(texts)

    def decode(self, ids: list[int]) -> str:
        return self._native.decode(ids)

    def token_to_id(self, token: str) -> int | None:
        return self._native.token_to_id(token)

    def get_vocab_size(self) -> int:
        return self._native.get_vocab_size()


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


def _rust_sources() -> list[Path]:
    return [_CRATE / "Cargo.toml", *(_CRATE / "src").glob("*.rs")]


def _trainer_output() -> Path:
    return _CRATE / "target" / "release" / _TRAINER_BIN


def _extension_candidates() -> list[Path]:
    return [_ROOT / f"{_MODULE_NAME}{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES]


def _native_outputs() -> list[Path]:
    release = _CRATE / "target" / "release"
    return [
        release / "libhackablebpe.so",
        release / "libhackablebpe.dylib",
        release / "hackablebpe.dll",
    ]


def _needs_rebuild(output: Path) -> bool:
    if not output.exists():
        return True
    built_at = output.stat().st_mtime
    return any(path.exists() and path.stat().st_mtime > built_at for path in _rust_sources())


def _load_extension(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Rust tokenizer extension from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _copy_native_output() -> Path:
    for output in _native_outputs():
        if output.exists():
            target = _extension_candidates()[0]
            shutil.copy2(output, target)
            return target
    expected = ", ".join(str(path) for path in _native_outputs())
    raise ImportError(f"cargo build finished but no tokenizer extension was found; expected one of: {expected}")


def _build_extension() -> Path:
    cargo = shutil.which("cargo")
    if cargo is None:
        raise ImportError("cargo is required to build the local Rust tokenizer extension")
    env = os.environ.copy()
    if sys.platform == "darwin":
        extra = "-C link-arg=-undefined -C link-arg=dynamic_lookup"
        env["RUSTFLAGS"] = f"{env.get('RUSTFLAGS', '')} {extra}".strip()
    proc = subprocess.run(
        [
            cargo,
            "build",
            "--release",
            "--manifest-path",
            str(_CRATE / "Cargo.toml"),
            "--lib",
            "--features",
            "extension-module",
        ],
        cwd=_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise ImportError(f"failed to build local Rust tokenizer extension:\n{proc.stderr}") from None
    return _copy_native_output()


def _load_native() -> ModuleType:
    try:
        return importlib.import_module(_MODULE_NAME)
    except ImportError:
        pass
    for candidate in _extension_candidates():
        if candidate.exists() and not _needs_rebuild(candidate):
            try:
                return _load_extension(candidate)
            except ImportError:
                candidate.unlink(missing_ok=True)
    return _load_extension(_build_extension())


def _build_trainer() -> Path:
    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError("cargo is required to build the local Rust tokenizer trainer")
    binary = _trainer_output()
    if not _needs_rebuild(binary):
        return binary
    proc = subprocess.run(
        [
            cargo,
            "build",
            "--release",
            "--manifest-path",
            str(_CRATE / "Cargo.toml"),
            "--bin",
            "hackablebpe_train",
        ],
        cwd=_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to build local Rust tokenizer trainer:\n{proc.stderr}") from None
    if not binary.exists():
        raise RuntimeError(f"cargo build finished but tokenizer trainer was not found at {binary}")
    return binary


def _trainer_binary() -> Path:
    override = os.environ.get("HACKABLE_LM_TOKENIZER_TRAINER")
    if override:
        path = Path(override)
        if not path.exists():
            raise RuntimeError(f"HACKABLE_LM_TOKENIZER_TRAINER does not exist: {path}")
        return path
    return _build_trainer()


def train_tokenizer(
    input_paths: list[str | Path],
    output_path: str | Path,
    vocab_size: int,
    jsonl_text_field: str = "text",
    min_frequency: int = 2,
) -> None:
    min_vocab_size = len(SPECIAL_TOKENS) + 256
    if vocab_size < min_vocab_size:
        raise ValueError(f"vocab_size must be at least {min_vocab_size} for byte-level BPE")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        str(_trainer_binary()),
        "train",
        "--output",
        str(output_path),
        "--vocab-size",
        str(vocab_size),
        "--jsonl-text-field",
        jsonl_text_field,
        "--min-frequency",
        str(min_frequency),
    ]
    for token in SPECIAL_TOKENS:
        args.extend(["--special-token", token])
    args.extend(str(Path(path)) for path in input_paths)
    proc = subprocess.run(args, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Rust tokenizer trainer failed with exit code {proc.returncode}") from None


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.load(path)


def encode(tokenizer, text: str, add_eos: bool = True) -> list[int]:
    ids = tokenizer.encode(text)
    if add_eos:
        eos_id = tokenizer.token_to_id(EOS_TOKEN)
        if eos_id is not None:
            ids.append(eos_id)
    return ids


def decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)


def tokenizer_impl_hash() -> str:
    hasher = hashlib.sha256()
    paths = [Path(__file__).resolve(), *_rust_sources()]
    for path in sorted({p.resolve() for p in paths if p.exists()}):
        try:
            name = path.relative_to(_ROOT)
        except ValueError:
            name = path
        hasher.update(str(name).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def tokenizer_manifest(path: str | Path) -> dict:
    from repro import hash_file

    tok = load_tokenizer(path)
    return {
        "tokenizer_path": str(path),
        "tokenizer_hash": hash_file(path),
        "tokenizer_impl_hash": tokenizer_impl_hash(),
        "vocab_size": tok.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "backend": tok.backend,
        "format": tok.format,
    }
