"""Measure bounded ensemble training and DDP throughput on lazy Zarr data."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import subprocess
import time
from datetime import timedelta
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

import autocast
from autocast.scripts.data import batch_to_device
from autocast.scripts.setup import setup_datamodule, setup_epd_model
from autocast.scripts.utils import get_default_config_path


class _TrainingLoss(nn.Module):
    """Route loss through DDP.forward so gradient synchronization runs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model.loss(batch)[0]


def _training_indices(dataset, world_size, rank, seed, count):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    )
    indices = list(islice(sampler, count))
    if len(indices) != count:
        msg = "The dataset has too few windows for the requested benchmark."
        raise ValueError(msg)
    return indices


def _aggregate_reports(reports):
    summary = dict(reports[0])
    elapsed = max(row["elapsed_seconds"] for row in reports)
    summary.update(
        elapsed_seconds=elapsed,
        mean_end_to_end_seconds_per_update=elapsed / summary["timed_updates"],
        training_windows_per_second=(
            summary["timed_updates"] * summary["global_batch_size"] / elapsed
        ),
        peak_cuda_allocated_bytes=max(
            row["peak_cuda_allocated_bytes"] for row in reports
        ),
        peak_cuda_reserved_bytes=max(
            row["peak_cuda_reserved_bytes"] for row in reports
        ),
        ranks=[
            {
                key: row[key]
                for key in (
                    "rank",
                    "hostname",
                    "elapsed_seconds",
                    "training_windows_per_second",
                    "peak_cuda_allocated_bytes",
                    "peak_cuda_reserved_bytes",
                )
            }
            for row in reports
        ],
    )
    return summary


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
    if args.rank == 0:
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
        "world_size": args.world_size,
        "rank": args.rank,
        "local_rank": args.local_rank,
        "hostname": socket.gethostname(),
        "global_batch_size": config.datamodule.batch_size * args.world_size,
        "scaling_mode": "fixed_batch_per_gpu",
        "ddp_find_unused_parameters": args.world_size > 1,
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
        loss = model(batch)
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
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])
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
            "device_uuid": str(
                getattr(torch.cuda.get_device_properties(device), "uuid", "")
            ),
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
                "Setup, warmup, diagnostics, teardown, validation and checkpointing "
                "are excluded. DDP gradient communication is included when enabled. "
                "Filesystem cache is uncontrolled. "
                "CUDA allocator peaks exclude CPU memory/cache. Smoke timings "
                "are diagnostic only."
            ),
        }
    )


def _run(config, datamodule, model, report):
    if not torch.cuda.is_available():
        msg = "A working CUDA GPU is required; use --validate for CPU/meta preflight."
        raise RuntimeError(msg)
    # Slurm may expose one assigned GPU, or all GPUs with a rank-local index.
    index = 0 if torch.cuda.device_count() == 1 else report["local_rank"]
    device = torch.device("cuda", index)
    torch.cuda.set_device(device)
    if report["world_size"] > 1:
        dist.init_process_group(
            "nccl",
            rank=report["rank"],
            world_size=report["world_size"],
            timeout=timedelta(minutes=3),
            device_id=device,
        )
    if report["cuda_allocator_limit_gib"] is not None:
        capacity = torch.cuda.get_device_properties(device).total_memory
        fraction = report["cuda_allocator_limit_gib"] * 1024**3 / capacity
        if not 0 < fraction <= 1:
            msg = "Requested CUDA allocator limit exceeds device capacity."
            raise ValueError(msg)
        # This bounds PyTorch allocations, not other CUDA/driver allocations.
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    model.to(device).train()
    model = _TrainingLoss(model)
    if report["world_size"] > 1:
        model = DistributedDataParallel(
            model, device_ids=[device.index], find_unused_parameters=True
        )
    torch.manual_seed(config.seed + report["rank"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.optimizer.learning_rate,
        betas=tuple(config.optimizer.betas),
        weight_decay=config.optimizer.weight_decay,
    )
    loader = datamodule.train_dataloader()
    # A finite sampler lets workers finish normally without aborting prefetches.
    loader.batch_sampler.sampler = _training_indices(
        loader.dataset,
        report["world_size"],
        report["rank"],
        config.seed,
        (report["warmup_updates"] + report["timed_updates"]) * report["batch_size"],
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
    args.world_size = int(
        os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
    )
    args.rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    args.local_rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
    )
    if args.validate and args.world_size > 1:
        parser.error("Run CPU/meta preflight with one process.")
    if not 0 <= args.rank < args.world_size or args.local_rank < 0:
        parser.error("Invalid distributed rank configuration.")
    if args.members < 2:
        parser.error("--members must be at least 2 for ensemble CRPS")
    limit = args.cuda_memory_limit_gib
    if limit is not None and (not math.isfinite(limit) or limit <= 0):
        parser.error("--cuda-memory-limit-gib must be finite and positive")
    if args.rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config, datamodule, model, report = _prepare(args)
    try:
        if args.validate:
            report["status"] = "configuration_validated_on_meta"
        else:
            _run(config, datamodule, model, report)
        if args.world_size > 1:
            (args.output / f"timing-rank-{args.rank}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            reports = [None] * args.world_size
            dist.all_gather_object(reports, report)
            report = _aggregate_reports(reports)
        if args.rank == 0:
            filename = "preflight.json" if args.validate else "timing.json"
            (args.output / filename).write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
