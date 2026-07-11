"""Raw latency benchmarking with explicit warm-up and percentile reporting."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


@dataclass
class LatencyObservation:
    iteration: int
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def benchmark(
    call: Callable[[], object], *, warmup: int = 3, repeats: int = 30
) -> tuple[list[LatencyObservation], dict]:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")
    for _ in range(warmup):
        call()
    observations: list[LatencyObservation] = []
    started_all = time.perf_counter()
    for index in range(repeats):
        started = time.perf_counter()
        call()
        observations.append(LatencyObservation(index, time.perf_counter() - started))
    wall = time.perf_counter() - started_all
    values = np.array([row.elapsed_seconds for row in observations])
    summary = {
        "warmup": warmup,
        "repeats": repeats,
        "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.percentile(values, 95)),
        "mean_seconds": float(values.mean()),
        "sequential_throughput_per_minute": float(60 * repeats / wall),
    }
    return observations, summary
