"""Measure lazy Zarr DataLoader throughput with reproducible window sampling."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler

from autocast.data.zarr_dataset import ZarrDataset
from autocast.types.collation import collate_batches


def benchmark_loader(
    data_path: str | Path,
    *,
    n_steps_input: int = 2,
    n_steps_output: int = 1,
    batch_size: int = 2,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    warmup_batches: int = 5,
    measured_batches: int = 50,
    seed: int = 42,
    use_normalization: bool = True,
    pin_memory: bool = False,
) -> dict[str, Any]:
    """Time CPU batches; batch size is per process, not the global GPU batch."""
    for name, value in (
        ("batch_size", batch_size),
        ("prefetch_factor", prefetch_factor),
        ("warmup_batches", warmup_batches),
        ("measured_batches", measured_batches),
    ):
        if value < 1:
            msg = f"{name} must be positive."
            raise ValueError(msg)
    if num_workers < 0:
        msg = "num_workers must be non-negative."
        raise ValueError(msg)
    start = time.perf_counter()
    dataset = ZarrDataset(
        data_path,
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        use_normalization=use_normalization,
    )
    initialization_seconds = time.perf_counter() - start
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=(warmup_batches + measured_batches) * batch_size,
        generator=torch.Generator().manual_seed(seed),
    )
    options: dict[str, Any] = {}
    if num_workers:
        options.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batches,
        **options,
    )
    start = time.perf_counter()
    iterator = iter(loader)
    next(iterator)
    first_batch_seconds = time.perf_counter() - start
    for _ in range(warmup_batches - 1):
        next(iterator)

    latencies = []
    returned_bytes = 0
    measured_start = time.perf_counter()
    for _ in range(measured_batches):
        start = time.perf_counter()
        batch = next(iterator)
        latencies.append(time.perf_counter() - start)
        returned_bytes += sum(
            field.numel() * field.element_size()
            for field in (batch.input_fields, batch.output_fields)
        )
        del batch
    elapsed = time.perf_counter() - measured_start
    array = dataset._data_array()
    result = {
        "data_path": str(data_path),
        "array_shape": list(array.shape),
        "chunks": list(array.chunks),
        "shards": list(array.shards) if array.shards is not None else None,
        "n_windows": len(dataset),
        "n_steps_input": n_steps_input,
        "n_steps_output": n_steps_output,
        "batch_size_per_process": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor if num_workers else None,
        "warmup_batches": warmup_batches,
        "measured_batches": measured_batches,
        "seed": seed,
        "normalization": use_normalization,
        "pin_memory": pin_memory,
        "torch_threads": torch.get_num_threads(),
        "initialization_seconds": initialization_seconds,
        "first_batch_seconds": first_batch_seconds,
        "measured_seconds": elapsed,
        "samples_per_second": measured_batches * batch_size / elapsed,
        "effective_mib_per_second": returned_bytes / elapsed / 1024**2,
        "batch_wait_median_seconds": statistics.median(latencies),
        "batch_wait_max_seconds": max(latencies),
        "measurement": (
            "Loader-only CPU throughput with uncontrolled filesystem cache. "
            "Returned bytes are not physical disk bytes; no GPU work is timed."
        ),
    }
    del iterator, loader
    return result


def main() -> None:
    """Benchmark existing layouts using the same sampled window indices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_paths", nargs="+", type=Path)
    parser.add_argument("--input-steps", type=int, default=2)
    parser.add_argument("--output-steps", type=int, default=1)
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Per-process batch size"
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--no-normalization", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    results = []
    for path in args.data_paths:
        result = benchmark_loader(
            path,
            n_steps_input=args.input_steps,
            n_steps_output=args.output_steps,
            batch_size=args.batch_size,
            num_workers=args.workers,
            prefetch_factor=args.prefetch_factor,
            warmup_batches=args.warmup_batches,
            measured_batches=args.batches,
            seed=args.seed,
            use_normalization=not args.no_normalization,
            pin_memory=args.pin_memory,
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
