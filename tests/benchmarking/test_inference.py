"""Unit tests for autocast.benchmarking.inference."""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

from autocast.benchmarking.inference import (
    _tqdm_disabled,
    benchmark_rollout,
)
from autocast.types import Batch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    batch_size: int = 1,
    t: int = 2,
    w: int = 2,
    h: int = 2,
    c: int = 1,
    const_c: int = 1,
    scalar_c: int = 1,
) -> Batch:
    fields = torch.zeros(batch_size, t, w, h, c)
    constant_fields = torch.ones(batch_size, w, h, const_c)
    constant_scalars = torch.full((batch_size, scalar_c), 0.0)
    return Batch(
        input_fields=fields,
        output_fields=fields,
        constant_scalars=constant_scalars,
        constant_fields=constant_fields,
    )


class _FakeRolloutModel(nn.Module):
    """Minimal model with a rollout() method for benchmarking tests."""

    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))

    def rollout(
        self,
        batch,
        stride: int,
        max_rollout_steps: int = 10,
        free_running_only: bool = True,  # noqa: ARG002
        **kwargs,
    ):
        b = batch.input_fields.shape[0]
        w = batch.input_fields.shape[2]
        h = batch.input_fields.shape[3]
        c = batch.input_fields.shape[4]
        preds = torch.zeros(b, max_rollout_steps * stride, w, h, c)
        trues = torch.zeros(b, max_rollout_steps * stride, w, h, c)
        return preds, trues


# ---------------------------------------------------------------------------
# _tqdm_disabled
# ---------------------------------------------------------------------------


def test_tqdm_disabled_sets_and_restores_env_when_absent():
    os.environ.pop("TQDM_DISABLE", None)
    with _tqdm_disabled():
        assert os.environ.get("TQDM_DISABLE") == "1"
    assert "TQDM_DISABLE" not in os.environ


def test_tqdm_disabled_restores_previous_value():
    os.environ["TQDM_DISABLE"] = "0"
    with _tqdm_disabled():
        assert os.environ.get("TQDM_DISABLE") == "1"
    assert os.environ.get("TQDM_DISABLE") == "0"
    del os.environ["TQDM_DISABLE"]


# ---------------------------------------------------------------------------
# benchmark_rollout
# ---------------------------------------------------------------------------


def test_benchmark_rollout_metrics():
    model = _FakeRolloutModel()
    batch = _make_batch()
    metrics = benchmark_rollout(
        model,
        batch,
        stride=1,
        max_rollout_steps=3,
        n_warmup=1,
        n_benchmark=2,
        batch_size=1,
    )
    assert "throughput_samples_per_sec" in metrics
    assert "latency_ms_per_batch" in metrics
    assert "latency_ms_per_sample" in metrics
    assert "latency_ms_per_step" in metrics
    assert metrics["throughput_samples_per_sec"] > 0
    assert metrics["latency_ms_per_batch"] > 0
    assert metrics["latency_ms_per_sample"] > 0
    assert metrics["latency_ms_per_step"] > 0


def test_benchmark_rollout_latency_per_step_consistent_with_per_rollout():
    model = _FakeRolloutModel()
    batch = _make_batch()
    max_rollout_steps = 4
    metrics = benchmark_rollout(
        model,
        batch,
        stride=1,
        max_rollout_steps=max_rollout_steps,
        n_warmup=1,
        n_benchmark=2,
        batch_size=1,
    )
    expected = metrics["latency_ms_per_batch"] / max_rollout_steps
    assert metrics["latency_ms_per_step"] == pytest.approx(expected)


def test_benchmark_rollout_raises_on_zero_n_benchmark():
    model = _FakeRolloutModel()
    batch = _make_batch()
    with pytest.raises(ValueError, match="n_benchmark must be > 0"):
        benchmark_rollout(
            model,
            batch,
            stride=1,
            max_rollout_steps=3,
            n_warmup=1,
            n_benchmark=0,
            batch_size=1,
        )
