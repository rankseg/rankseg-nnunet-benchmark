from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from rankseg import RankSEG

from .config import DatasetConfig


@dataclass(frozen=True)
class DecodeResult:
    prediction: np.ndarray
    milliseconds: float
    peak_memory_bytes: int | None
    device: str


class PairedDecoders:
    def __init__(self, config: DatasetConfig):
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("runtime.device is CUDA but torch.cuda.is_available() is false")
        self._rankseg = RankSEG(
            metric=config.rankseg_metric,
            solver=config.rankseg_solver,
            output_mode=config.rankseg_output_mode,
            pruning_prob=config.pruning_prob,
            smooth=config.smooth,
            unassigned_policy=config.unassigned_policy,
        )

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _reset_peak(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _peak(self) -> int | None:
        return int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else None

    def move_probabilities(self, probabilities: np.ndarray) -> torch.Tensor:
        contiguous = np.ascontiguousarray(probabilities.astype(np.float32, copy=False))
        return torch.from_numpy(contiguous).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def decode_argmax(self, probabilities: torch.Tensor) -> DecodeResult:
        self._reset_peak()
        self._synchronize()
        start = perf_counter()
        prediction = torch.argmax(probabilities, dim=1)
        self._synchronize()
        elapsed = (perf_counter() - start) * 1000.0
        peak = self._peak()
        return DecodeResult(prediction.squeeze(0).cpu().numpy(), elapsed, peak, str(self.device))

    @torch.inference_mode()
    def decode_rankseg(self, probabilities: torch.Tensor) -> DecodeResult:
        self._reset_peak()
        self._synchronize()
        start = perf_counter()
        prediction = self._rankseg(probabilities)
        self._synchronize()
        elapsed = (perf_counter() - start) * 1000.0
        peak = self._peak()
        return DecodeResult(prediction.squeeze(0).cpu().numpy(), elapsed, peak, str(self.device))

    @torch.inference_mode()
    def decode_rankseg_cpu(self, probabilities: np.ndarray) -> DecodeResult:
        """Retry an otherwise identical decode on CPU when a volume exceeds GPU memory."""
        contiguous = np.ascontiguousarray(probabilities.astype(np.float32, copy=False))
        tensor = torch.from_numpy(contiguous).unsqueeze(0)
        start = perf_counter()
        prediction = self._rankseg(tensor)
        elapsed = (perf_counter() - start) * 1000.0
        return DecodeResult(
            prediction.squeeze(0).numpy(),
            elapsed,
            None,
            "cpu_fallback_after_cuda_oom",
        )
