"""Benchmarking utilities for AutoCast."""

from autocast.benchmarking.inference import (
    benchmark_model,
    make_synthetic_batch,
    measure_flops,
)

__all__ = [
    "benchmark_model",
    "make_synthetic_batch",
    "measure_flops",
]
