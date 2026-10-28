"""Distributed timing must synchronize updates and count distinct windows."""

import json
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel

from autocast.scripts.benchmark_training import (
    _aggregate_reports,
    _training_indices,
    _TrainingLoss,
)


class _Forecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.unused = nn.Parameter(torch.ones(1))

    def loss(self, inputs):
        return (self.weight * inputs - 1).square().mean(), None


def _ddp_worker(rank, init_path, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        forecast = _Forecast()
        model = DistributedDataParallel(
            _TrainingLoss(forecast), find_unused_parameters=True
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(torch.tensor([float(rank + 1)])).backward()
            optimizer.step()
        (output / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "weight": forecast.weight.item(),
                    "unused": forecast.unused.item(),
                }
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
def test_ddp_loss_averages_gradients_across_distinct_rank_inputs(tmp_path):
    spawn(_ddp_worker, args=(tmp_path / "rendezvous", tmp_path), nprocs=2)
    # Averaged gradients give w <- 0.5*w + 0.3; independent ranks diverge.
    for rank in range(2):
        result = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert result["weight"] == pytest.approx(0.525)
        assert result["unused"] == 1


def test_training_windows_are_finite_and_partitioned_by_rank():
    windows = range(32)
    first = _training_indices(windows, 2, 0, 42, 10)
    second = _training_indices(windows, 2, 1, 42, 10)
    assert len(first) == len(second) == 10
    assert set(first).isdisjoint(second)
    with pytest.raises(ValueError, match="too few windows"):
        _training_indices(windows, 2, 0, 42, 17)


def test_global_throughput_uses_slowest_rank_and_excludes_member_multiplier():
    reports = [
        {
            "rank": rank,
            "hostname": f"node-{rank}",
            "world_size": 2,
            "batch_size": 2,
            "global_batch_size": 4,
            "ensemble_members": 8,
            "timed_updates": 100,
            "elapsed_seconds": elapsed,
            "training_windows_per_second": 200 / elapsed,
            "peak_cuda_allocated_bytes": 100 + rank,
            "peak_cuda_reserved_bytes": 200 + rank,
        }
        for rank, elapsed in enumerate((1.0, 2.0))
    ]
    summary = _aggregate_reports(reports)
    assert summary["elapsed_seconds"] == 2.0
    assert summary["training_windows_per_second"] == 200.0
    assert summary["mean_end_to_end_seconds_per_update"] == 0.02
    assert summary["peak_cuda_allocated_bytes"] == 101
    assert summary["peak_cuda_reserved_bytes"] == 201
