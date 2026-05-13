from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType


_MODULE_NAME = "_hackable_lm_tokenizer"
_ROOT = Path(__file__).resolve().parent
_CRATE = _ROOT / "rustbpe"


def _extension_candidates() -> list[Path]:
    return [_ROOT / f"{_MODULE_NAME}{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES]


def _native_outputs() -> list[Path]:
    release = _CRATE / "target" / "release"
    return [
        release / f"lib{_MODULE_NAME}.so",
        release / f"lib{_MODULE_NAME}.dylib",
        release / f"{_MODULE_NAME}.dll",
    ]


def _rust_sources() -> list[Path]:
    return [_CRATE / "Cargo.toml", *(_CRATE / "src").glob("*.rs")]


def _needs_rebuild(extension: Path) -> bool:
    if not extension.exists():
        return True
    built_at = extension.stat().st_mtime
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
    proc = subprocess.run(
        [cargo, "build", "--release", "--manifest-path", str(_CRATE / "Cargo.toml")],
        cwd=_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise ImportError(f"failed to build local Rust tokenizer extension:\n{proc.stderr}") from None
    return _copy_native_output()


def load_native() -> ModuleType:
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


native = load_native()

Encoding = native.Encoding
Tokenizer = native.Tokenizer
load_from_file = native.load_from_file
train_from_iterator = native.train_from_iterator
train_from_texts = native.train_from_texts
