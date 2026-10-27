"""Measure bounded, single-GPU ensemble training on a lazy Zarr dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler

import autocast
from autocast.scripts.data import batch_to_device
from autocast.scripts.setup import setup_datamodule, setup_epd_model
from autocast.scripts.utils import get_default_config_path


def _prepare(args):
    with initialize_config_dir(version_base=None, config_dir=get_default_config_path()):
        config = compose(
            config_name="encoder_processor_decoder",
            overrides=[
                f"experiment={args.experiment}",
                f"datamodule.data_path={args.dataset.resolve()}",
                f"model.n_members={args.members}",
            ],
        )
    if config.optimizer.optimizer != "adamw" or config.optimizer.scheduler is not None:
        msg = "The benchmark supports AdamW without a learning-rate scheduler."
        raise ValueError(msg)
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision(config.float32_matmul_precision or "highest")
    workers = config.datamodule.num_workers
    # Shape inference consumes only one batch; do not start workers for it.
    config.datamodule.num_workers = 0
    datamodule, config, stats = setup_datamodule(config)
    with torch.device("meta" if args.validate else "cpu"):
        model = setup_epd_model(config, stats, datamodule=datamodule)
    datamodule.num_workers = config.datamodule.num_workers = workers
    OmegaConf.save(config, args.output / "resolved_config.yaml", resolve=True)
    source = Path(autocast.__file__).resolve()
    result = subprocess.run(
        ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "dataset": str(args.dataset.resolve()),
        "autocast_import": str(source),
        "git_sha": result.stdout.strip() if result.returncode == 0 else None,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "parameters_trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "input_shape": list(stats["input_shape"]),
        "output_shape": list(stats["output_shape"]),
        "batch_size": config.datamodule.batch_size,
        "ensemble_members": args.members,
        "world_size": 1,
        "gradient_accumulation": 1,
        "seed": config.seed,
        "dtype": "float32",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "checkpoint": config.autoencoder_checkpoint,
        "warmup_updates": 2 if args.smoke else 20,
        "timed_updates": 3 if args.smoke else 100,
        "memory_smoke_only": args.smoke,
        "cuda_allocator_limit_gib": args.cuda_memory_limit_gib,
        "workers": workers,
        "prefetch_factor": config.datamodule.prefetch_factor,
        "torch_threads": torch.get_num_threads(),
    }
    patch = getattr(model.processor, "patch_size", None)
    if patch is not None:
        spatial = config.model.processor.spatial_resolution
        report["patch"] = list(patch)
        report["spatial_tokens"] = math.prod(
            n // p for n, p in zip(spatial, patch, strict=True)
        )
    return config, datamodule, model, report


def _measure(model, iterator, optimizer, device, report):
    def update(batch):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.loss(batch)
        loss.backward()
        optimizer.step()
        return loss.detach()

    loss = None
    for _ in range(report["warmup_updates"]):
        loss = update(batch_to_device(next(iterator), device))
    torch.cuda.synchronize(device)
    assert loss is not None, "Warmup did not produce a loss."
    assert bool(torch.isfinite(loss)), "Non-finite warmup loss."
    probe_name, probe = next(
        (name, parameter)
        for name, parameter in sorted(
            model.named_parameters(), key=lambda item: item[1].numel()
        )
        if parameter.requires_grad
        and parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool(parameter.grad.any())
    )
    report["update_probe"] = probe_name
    before = probe.detach().clone()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(report["timed_updates"])
    ]
    losses, loader_waits = [], []
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    report["timed_start_unix"] = time.time()
    start = time.perf_counter()
    for begin, end in events:
        waiting = time.perf_counter()
        cpu_batch = next(iterator)
        loader_waits.append(time.perf_counter() - waiting)
        batch = batch_to_device(cpu_batch, device)
        begin.record()
        losses.append(update(batch))
        end.record()
        del cpu_batch, batch
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    report["timed_end_unix"] = time.time()
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)

    # Worker teardown, diagnostics and serialization are outside the timed block.
    assert next(iterator, None) is None, "Unexpected extra training batch."
    assert bool(torch.isfinite(torch.stack(losses)).all()), "Non-finite timed loss."
    assert float((probe.detach() - before).abs().max()) > 0, "Weights did not update."
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters()), (
        "Non-finite weights."
    )
    gpu_times = [begin.elapsed_time(end) / 1000 for begin, end in events]
    report.update(
        {
            "status": "smoke_passed" if report["memory_smoke_only"] else "passed",
            "device": torch.cuda.get_device_name(device),
            "device_capability": torch.cuda.get_device_capability(device),
            "elapsed_seconds": elapsed,
            "mean_end_to_end_seconds_per_update": elapsed / report["timed_updates"],
            "training_windows_per_second": report["timed_updates"]
            * report["batch_size"]
            / elapsed,
            "median_gpu_update_seconds": statistics.median(gpu_times),
            "p90_gpu_update_seconds": statistics.quantiles(
                gpu_times, n=10, method="inclusive"
            )[8],
            "mean_loader_wait_seconds": statistics.mean(loader_waits),
            "gpu_update_seconds": gpu_times,
            "loader_wait_seconds": loader_waits,
            "losses": [float(value) for value in losses],
            "measurement": (
                "End-to-end wall time includes loading, transfer, forward/loss, "
                "backward "
                "and AdamW. CUDA events span model/optimizer work after transfer. "
                "Setup, warmup, diagnostics and teardown are excluded. No validation, "
                "checkpointing or distributed communication. Filesystem cache "
                "is uncontrolled. "
                "CUDA allocator peaks exclude CPU memory/cache. Smoke timings "
                "are diagnostic only."
            ),
        }
    )


def _run(config, datamodule, model, report):
    if not torch.cuda.is_available():
        msg = "A working CUDA GPU is required; use --validate for CPU/meta preflight."
        raise RuntimeError(msg)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if report["cuda_allocator_limit_gib"] is not None:
        capacity = torch.cuda.get_device_properties(device).total_memory
        fraction = report["cuda_allocator_limit_gib"] * 1024**3 / capacity
        if not 0 < fraction <= 1:
            msg = "Requested CUDA allocator limit exceeds device capacity."
            raise ValueError(msg)
        # This bounds PyTorch allocations, not other CUDA/driver allocations.
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.optimizer.learning_rate,
        betas=tuple(config.optimizer.betas),
        weight_decay=config.optimizer.weight_decay,
    )
    loader = datamodule.train_dataloader()
    # A finite sampler lets workers finish normally without aborting prefetches.
    loader.batch_sampler.sampler = RandomSampler(
        loader.dataset,
        replacement=True,
        num_samples=(report["warmup_updates"] + report["timed_updates"])
        * report["batch_size"],
        generator=torch.Generator().manual_seed(config.seed),
    )
    _measure(model, iter(loader), optimizer, device, report)


def main() -> None:
    """Run a short timing or memory check with the normal model setup path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path, help="New result directory")
    parser.add_argument("--experiment", default="benchmark_synthetic_vit")
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument(
        "--validate", action="store_true", help="CPU/meta preflight only"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="2 warmup + 3 checked updates"
    )
    parser.add_argument("--cuda-memory-limit-gib", type=float)
    args = parser.parse_args()
    if any(int(os.environ.get(key, "1")) > 1 for key in ("WORLD_SIZE", "SLURM_NTASKS")):
        parser.error("Only one process/GPU is supported; this is not a DDP benchmark.")
    if args.members < 2:
        parser.error("--members must be at least 2 for ensemble CRPS")
    limit = args.cuda_memory_limit_gib
    if limit is not None and (not math.isfinite(limit) or limit <= 0):
        parser.error("--cuda-memory-limit-gib must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config, datamodule, model, report = _prepare(args)
    if args.validate:
        report["status"] = "configuration_validated_on_meta"
    else:
        _run(config, datamodule, model, report)
    filename = "preflight.json" if args.validate else "timing.json"
    (args.output / filename).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
