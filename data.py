from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


def load_manifest(data_dir: str | Path) -> dict[str, Any]:
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"processed data manifest not found: {path}")
    return json.loads(path.read_text())


class MemmapDataLoader:
    def __init__(self, data_dir: str | Path, block_size: int, seed: int = 0) -> None:
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.manifest = load_manifest(self.data_dir)
        dtype = DTYPES[self.manifest["dtype"]]
        self.train = np.memmap(self.data_dir / "train.bin", dtype=dtype, mode="r")
        self.val = np.memmap(self.data_dir / "val.bin", dtype=dtype, mode="r")
        self.rng = np.random.default_rng(seed)

    def _new_batch_buffer(self, batch_size: int, pin_memory: bool) -> torch.Tensor:
        return torch.empty((2, batch_size, self.block_size), dtype=torch.long, pin_memory=pin_memory)

    def _fill_batch_buffer(
        self,
        split: str,
        buffer: torch.Tensor,
        numpy_views: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        tokens = self.train if split == "train" else self.val
        max_start = len(tokens) - self.block_size - 1
        starts = self.rng.integers(0, max_start, size=buffer.size(1))
        x_np, y_np = numpy_views if numpy_views is not None else (buffer[0].numpy(), buffer[1].numpy())
        for row, start in enumerate(starts):
            start = int(start)
            span = tokens[start : start + self.block_size + 1]
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
            "sampling_policy": "random_packed_spans",
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
        self.next_idx = 0
        self.preload()

    def preload(self) -> None:
        if self.pending_idx is not None:
            return
        current_stream = torch.cuda.current_stream(self.device)
        if self.current_idx is None:
            target_idx = self.next_idx
        else:
            released_idx = self.current_idx
            self.ready_events[released_idx].record(current_stream)
            self.ready_recorded[released_idx] = True
            target_idx = 1 - released_idx
            self.current_idx = None

        # The pinned CPU source cannot be refilled until its prior async copy is done.
        if self.copy_recorded[target_idx]:
            self.copy_events[target_idx].synchronize()
        self.loader._fill_batch_buffer(self.split, self.cpu_buffers[target_idx], self.cpu_numpy_views[target_idx])
        with torch.cuda.stream(self.stream):
            # The GPU destination cannot be overwritten until its prior consumer is done.
            if self.ready_recorded[target_idx]:
                self.stream.wait_event(self.ready_events[target_idx])
            self.gpu_buffers[target_idx].copy_(self.cpu_buffers[target_idx], non_blocking=True)
            self.copy_events[target_idx].record(self.stream)
            self.copy_recorded[target_idx] = True
        self.pending_idx = target_idx
        self.next_idx = 1 - target_idx

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pending_idx is None:
            self.preload()
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_stream(self.stream)
        if self.pending_idx is None:
            raise RuntimeError("CUDA prefetcher has no pending batch")
        buffer = self.gpu_buffers[self.pending_idx]
        self.current_idx = self.pending_idx
        self.pending_idx = None
        batch = (buffer[0], buffer[1])
        for tensor in batch:
            tensor.record_stream(current_stream)
        return batch
