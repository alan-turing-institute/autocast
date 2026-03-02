"""Reusable inference benchmarking utilities."""

from __future__ import annotations

from time import perf_counter

import torch

from autocast.types.batch import Batch

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:
    FlopCounterMode = None


def _slice_optional_first(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim == 0:
        return value
    return value[:1]


def make_synthetic_batch(example_batch: Batch, batch_size: int) -> Batch:
    """Create a synthetic batch with matching tensor shapes for benchmarking."""
    single = Batch(
        input_fields=example_batch.input_fields[:1],
        output_fields=example_batch.output_fields[:1],
        constant_scalars=_slice_optional_first(example_batch.constant_scalars),
        constant_fields=_slice_optional_first(example_batch.constant_fields),
        boundary_conditions=_slice_optional_first(example_batch.boundary_conditions),
    )
    return single.repeat(batch_size) if batch_size > 1 else single


def _predict_once(model: torch.nn.Module, batch: Batch) -> None:
    with torch.no_grad():
        try:
            model.predict_step(batch, 0)
        except Exception:
            model(batch)


def measure_flops(
    model: torch.nn.Module,
    example_batch: Batch,
) -> dict[str, float]:
    """Measure FLOPs for one forward pass and report GFLOPs/sample."""
    if FlopCounterMode is None:
        return {}

    device = next(model.parameters()).device
    single = make_synthetic_batch(example_batch, batch_size=1).to(device)

    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as flop_counter:
        _predict_once(model, single)

    flops_per_sample = float(flop_counter.get_total_flops())
    return {
        "gflops_per_sample": round(flops_per_sample / 1e9, 4),
    }


def benchmark_model(
    model: torch.nn.Module,
    example_batch: Batch,
    *,
    n_warmup: int,
    n_benchmark: int,
    batch_size: int,
) -> dict[str, float]:
    """Benchmark model throughput and latency using synthetic batches."""
    if n_benchmark <= 0:
        msg = "n_benchmark must be > 0"
        raise ValueError(msg)

    device = next(model.parameters()).device
    synthetic_batch = make_synthetic_batch(example_batch, batch_size=batch_size).to(
        device
    )

    model.eval()
    batch_times_s: list[float] = []

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    total_steps = n_warmup + n_benchmark
    for step in range(total_steps):
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        start_s = perf_counter()
        _predict_once(model, synthetic_batch)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elapsed_s = perf_counter() - start_s
        if step >= n_warmup:
            batch_times_s.append(elapsed_s)

    total_s = sum(batch_times_s)
    if total_s <= 0:
        msg = "Measured benchmark duration is zero; cannot compute throughput."
        raise RuntimeError(msg)

    n_measured = len(batch_times_s)
    latency_batch_ms = (total_s / n_measured) * 1000.0

    metrics: dict[str, float] = {
        "throughput_samples_per_sec": (n_measured * batch_size) / total_s,
        "latency_ms_per_batch": latency_batch_ms,
        "latency_ms_per_sample": latency_batch_ms / batch_size,
    }

    if device.type == "cuda" and torch.cuda.is_available():
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        metrics["peak_gpu_memory_mb"] = round(peak_bytes / 1024**2, 1)

    metrics.update(measure_flops(model, example_batch))
    return metrics
