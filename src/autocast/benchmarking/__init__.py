"""Benchmarking utilities for AutoCast."""

from autocast.benchmarking.inference import (
    benchmark_model,
    benchmark_rollout,
    make_synthetic_batch,
    measure_flops,
)

__all__ = [
    "benchmark_model",
    "benchmark_rollout",
    "make_synthetic_batch",
    "measure_flops",
]
