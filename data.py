from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tokenizer import BOS_TOKEN, load_tokenizer


DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


def load_manifest(data_dir: str | Path) -> dict[str, Any]:
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"processed data manifest not found: {path}")
    return json.loads(path.read_text())


class MemmapDataLoader:
    def __init__(
        self,
        data_dir: str | Path,
        block_size: int,
        data_shuffle_seed: int = 1337,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if data_shuffle_seed < 0:
            raise ValueError("data_shuffle_seed must be non-negative")
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if rank >= world_size:
            raise ValueError("rank must be less than world_size")
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.data_shuffle_seed = data_shuffle_seed
        self.rank = rank
        self.world_size = world_size
        self.manifest = load_manifest(self.data_dir)
        dtype = DTYPES[self.manifest["dtype"]]

        def load_tokens(path: Path) -> np.ndarray:
            if not path.exists() or path.stat().st_size == 0:
                return np.asarray([], dtype=dtype)
            return np.memmap(path, dtype=dtype, mode="r")

        self.tokens = {
            "train": load_tokens(self.data_dir / "train.bin"),
            "val": load_tokens(self.data_dir / "val.bin"),
        }
        self.offsets = {
            "train": np.load(self.data_dir / "train_offsets.npy", mmap_mode="r"),
            "val": np.load(self.data_dir / "val_offsets.npy", mmap_mode="r"),
        }
        tok = load_tokenizer(self.data_dir / "tokenizer.json")
        self.bos_id = tok.token_to_id(BOS_TOKEN)
        if self.bos_id is None:
            raise RuntimeError(f"tokenizer is missing required special token {BOS_TOKEN!r}")
        self.doc_orders = {}
        self.doc_positions = {}
        for split, salt in (("train", 0), ("val", 1)):
            rng = np.random.default_rng(np.random.SeedSequence([data_shuffle_seed, salt]))
            self.doc_orders[split] = rng.permutation(self._doc_count(split))
            self.doc_positions[split] = rank

    def _doc_count(self, split: str) -> int:
        return max(0, int(self.offsets[split].shape[0]) - 1)

    def _next_doc_index(self, split: str) -> int:
        order = self.doc_orders[split]
        position = self.doc_positions[split]
        if len(order) == 0:
            raise RuntimeError("cannot sample from an empty document split")
        if position >= len(order):
            raise StopIteration("document split exhausted")
        self.doc_positions[split] += self.world_size
        return int(order[position])

    def available_spans(self, split: str) -> int:
        if split not in self.tokens:
            raise ValueError(f"unknown split {split!r}")
        offsets = self.offsets[split]
        remaining = self.block_size
        spans = 0
        order = self.doc_orders[split]
        for position in range(self.doc_positions[split], len(order), self.world_size):
            doc_idx = int(order[position])
            doc_len = int(offsets[doc_idx + 1] - offsets[doc_idx])
            if doc_len <= 0:
                continue
            if doc_len >= remaining:
                spans += 1
                remaining = self.block_size
            else:
                remaining -= doc_len
        return spans

    def require_batches(self, split: str, batches: int, batch_size: int) -> None:
        required_spans = batches * batch_size
        available_spans = self.available_spans(split)
        if available_spans < required_spans:
            available_batches = available_spans // batch_size
            raise RuntimeError(
                f"{split} split has {available_spans} complete one-pass spans "
                f"({available_batches} full batches of size {batch_size}) for rank {self.rank}/{self.world_size}, "
                f"but the run requires {required_spans} spans ({batches} batches)."
            )

    def _new_batch_buffer(self, batch_size: int, pin_memory: bool) -> torch.Tensor:
        return torch.empty((2, batch_size, self.block_size), dtype=torch.long, pin_memory=pin_memory)

    def _next_span(self, split: str) -> np.ndarray:
        tokens = self.tokens[split]
        offsets = self.offsets[split]
        span = np.empty(self.block_size + 1, dtype=np.int64)
        span[0] = int(self.bos_id)
        out_pos = 1
        while out_pos < span.size:
            try:
                doc_idx = self._next_doc_index(split)
            except StopIteration as exc:
                raise RuntimeError(
                    f"{split} split exhausted before filling a {self.block_size + 1}-token training span"
                ) from exc
            start = int(offsets[doc_idx])
            end = int(offsets[doc_idx + 1])
            if end <= start:
                continue
            take = min(end - start, span.size - out_pos)
            span[out_pos : out_pos + take] = tokens[start : start + take]
            out_pos += take
        return span

    def _fill_batch_buffer(
        self,
        split: str,
        buffer: torch.Tensor,
        numpy_views: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        if split not in self.tokens:
            raise ValueError(f"unknown split {split!r}")
        x_np, y_np = numpy_views if numpy_views is not None else (buffer[0].numpy(), buffer[1].numpy())
        for row in range(buffer.size(1)):
            span = self._next_span(split)
            x_np[row] = span[:-1]
            y_np[row] = span[1:]

    def get_batch_cpu(self, split: str, batch_size: int, pin_memory: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        buffer = self._new_batch_buffer(batch_size, pin_memory)
        self._fill_batch_buffer(split, buffer)
        return buffer[0], buffer[1]

    def get_batch(self, split: str, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device)
        pin_memory = device.type == "cuda"
        buffer = self._new_batch_buffer(batch_size, pin_memory)
        self._fill_batch_buffer(split, buffer)
        if device.type == "cpu":
            return buffer[0], buffer[1]
        gpu_buffer = buffer.to(device, non_blocking=pin_memory)
        return gpu_buffer[0], gpu_buffer[1]

    def cuda_prefetcher(self, split: str, batch_size: int, device: torch.device | str) -> "CudaBatchPrefetcher":
        return CudaBatchPrefetcher(self, split, batch_size, torch.device(device))

    def info(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "manifest": self.manifest,
            "data_shuffle_seed": self.data_shuffle_seed,
            "block_size": self.block_size,
        }


class CudaBatchPrefetcher:
    def __init__(self, loader: MemmapDataLoader, split: str, batch_size: int, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError("CudaBatchPrefetcher requires a CUDA device")
        self.loader = loader
        self.split = split
        self.batch_size = batch_size
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        # Two slots let H2D for one batch overlap with kernels consuming the other.
        self.cpu_buffers = [
            loader._new_batch_buffer(batch_size, pin_memory=True),
            loader._new_batch_buffer(batch_size, pin_memory=True),
        ]
        self.cpu_numpy_views = [(buffer[0].numpy(), buffer[1].numpy()) for buffer in self.cpu_buffers]
        self.gpu_buffers = [
            torch.empty_like(self.cpu_buffers[0], device=device),
            torch.empty_like(self.cpu_buffers[1], device=device),
        ]
        self.ready_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.ready_recorded = [False, False]
        self.copy_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.copy_recorded = [False, False]
        self.pending_idx: int | None = None
        self.current_idx: int | None = None
        self.fill_requests: queue.Queue[int | None] = queue.Queue(maxsize=1)
        self.ready_slots: queue.Queue[tuple[int, BaseException | None]] = queue.Queue(maxsize=1)
        self.producer = threading.Thread(target=self._producer_loop, daemon=True)
        self.producer.start()
        self._prepare_slot(0)
        self.ready_slots.put((0, None))

    def _fill_cpu_buffer(self, idx: int) -> None:
        self.loader._fill_batch_buffer(self.split, self.cpu_buffers[idx], self.cpu_numpy_views[idx])

    def _prepare_slot(self, idx: int) -> None:
        self._fill_cpu_buffer(idx)
        with torch.cuda.stream(self.stream):
            if self.ready_recorded[idx]:
                self.stream.wait_event(self.ready_events[idx])
            self.gpu_buffers[idx].copy_(self.cpu_buffers[idx], non_blocking=True)
            self.copy_events[idx].record(self.stream)
            self.copy_recorded[idx] = True

    def _producer_loop(self) -> None:
        while True:
            idx = self.fill_requests.get()
            if idx is None:
                return
            try:
                self._prepare_slot(idx)
            except BaseException as exc:
                self.ready_slots.put((idx, exc))
            else:
                self.ready_slots.put((idx, None))

    def _start_cpu_fill(self, idx: int) -> None:
        self.fill_requests.put(idx)

    def _receive_ready_slot(self) -> None:
        filled_idx, error = self.ready_slots.get()
        if error is not None:
            raise RuntimeError(f"failed to prepare next {self.split} batch") from error
        self.pending_idx = filled_idx

    def next(self, prepare_next: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pending_idx is None:
            self._receive_ready_slot()
        current_stream = torch.cuda.current_stream(self.device)
        if self.pending_idx is None:
            raise RuntimeError("CUDA prefetcher has no pending batch")
        current_stream.wait_event(self.copy_events[self.pending_idx])
        released_idx = self.current_idx
        if released_idx is not None:
            self.ready_events[released_idx].record(current_stream)
            self.ready_recorded[released_idx] = True
        buffer = self.gpu_buffers[self.pending_idx]
        self.current_idx = self.pending_idx
        self.pending_idx = None
        if prepare_next:
            target_idx = released_idx if released_idx is not None else 1 - self.current_idx
            self._start_cpu_fill(target_idx)
        batch = (buffer[0], buffer[1])
        for tensor in batch:
            tensor.record_stream(current_stream)
        return batch
