"""Unit tests for evaluation batch-limit resolution."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from omegaconf import OmegaConf

from autocast.scripts.eval.encoder_processor_decoder import (
    _render_rollouts,
    _resolve_rollout_batch_limit,
    _resolve_rollout_channel_names,
    _resolve_rollout_timestep_limit,
    _split_metric_and_metadata_rows,
    _training_runtime_rows,
)


def test_resolve_rollout_batch_limit_falls_back_to_test_limit_when_null():
    eval_cfg = OmegaConf.create(
        {
            "max_test_batches": 2,
            "max_rollout_batches": None,
        }
    )

    assert _resolve_rollout_batch_limit(eval_cfg) == 2


def test_resolve_rollout_batch_limit_prefers_explicit_rollout_limit():
    eval_cfg = OmegaConf.create(
        {
            "max_test_batches": 2,
            "max_rollout_batches": 5,
        }
    )

    assert _resolve_rollout_batch_limit(eval_cfg) == 5


def test_resolve_rollout_timestep_limit_multiplies_by_stride():
    assert (
        _resolve_rollout_timestep_limit(max_rollout_steps=50, rollout_stride=4) == 200
    )


def test_resolve_rollout_timestep_limit_returns_none_for_invalid_inputs():
    assert (
        _resolve_rollout_timestep_limit(max_rollout_steps=None, rollout_stride=4)
        is None
    )
    assert (
        _resolve_rollout_timestep_limit(max_rollout_steps=0, rollout_stride=4) is None
    )
    assert (
        _resolve_rollout_timestep_limit(max_rollout_steps=10, rollout_stride=0) is None
    )


def test_split_metric_and_metadata_rows_separates_meta_rows():
    rows = [
        {"window": "all", "batch_idx": "all", "mse": 0.1},
        {
            "window": "meta",
            "batch_idx": "all",
            "category": "runtime_eval",
            "metric": "total_s",
            "value": 1.2,
        },
        {"window": "0-1", "batch_idx": 0, "rmse": 0.2},
    ]

    metric_rows, metadata_rows = _split_metric_and_metadata_rows(rows)

    assert metric_rows == [
        {"window": "all", "batch_idx": "all", "mse": 0.1},
        {"window": "0-1", "batch_idx": 0, "rmse": 0.2},
    ]
    assert metadata_rows == [
        {
            "window": "meta",
            "batch_idx": "all",
            "category": "runtime_eval",
            "metric": "total_s",
            "value": 1.2,
        }
    ]


def test_resolve_rollout_channel_names_from_norm_with_output_selection():
    dataset = SimpleNamespace(
        norm=SimpleNamespace(core_field_names=["u", "v", "p"]),
        output_channel_idxs=(2, 0),
    )

    assert _resolve_rollout_channel_names(dataset) == ["p", "u"]


def test_resolve_rollout_channel_names_returns_none_without_norm_names():
    dataset = SimpleNamespace(
        norm=None,
        metadata=SimpleNamespace(field_names={0: ["velocity_x", "velocity_y"]}),
        output_channel_idxs=None,
    )

    assert _resolve_rollout_channel_names(dataset) is None


def test_resolve_rollout_channel_names_returns_none_on_invalid_output_indices():
    dataset = SimpleNamespace(
        norm=SimpleNamespace(core_field_names=["u", "v"]),
        output_channel_idxs=(0, 3),
    )

    assert _resolve_rollout_channel_names(dataset) is None


# --- _training_runtime_rows with actual epoch times ---


def _make_timed_payload(
    epoch: int,
    global_step: int,
    total_s: float,
    epoch_times: list[float],
) -> dict:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "callbacks": {
            "TrainingTimerCallback": {
                "training_runtime_total_s": total_s,
                "training_runtime_elapsed_s": total_s,
                "mean_epoch_s": sum(epoch_times) / len(epoch_times),
                "min_epoch_s": min(epoch_times),
                "max_epoch_s": max(epoch_times),
                "epoch_times_s": epoch_times,
            }
        },
    }


def test_training_runtime_rows_emit_min_mean_max_when_epoch_times_available():
    payload = _make_timed_payload(
        epoch=2, global_step=300, total_s=30.0, epoch_times=[8.0, 10.0, 12.0]
    )
    rows = _training_runtime_rows(payload)
    by_metric = {r["metric"]: r["value"] for r in rows}

    assert by_metric["total_s"] == pytest.approx(30.0)
    assert by_metric["mean_epoch_s"] == pytest.approx(10.0)
    assert by_metric["min_epoch_s"] == pytest.approx(8.0)
    assert by_metric["max_epoch_s"] == pytest.approx(12.0)


def test_training_runtime_rows_fall_back_to_elapsed_runtime_when_total_missing():
    payload = {
        "callbacks": {
            "TrainingTimerCallback": {
                "training_runtime_total_s": None,
                "training_runtime_elapsed_s": 12.5,
            }
        }
    }
    rows = _training_runtime_rows(payload)
    by_metric = {r["metric"]: r["value"] for r in rows}
    assert by_metric["total_s"] == pytest.approx(12.5)


def test_training_runtime_rows_fall_back_to_average_without_epoch_times():
    payload = {"epoch": 3, "global_step": 400, "training_runtime_total_s": 40.0}
    rows = _training_runtime_rows(payload)

    assert rows == []


def test_render_rollouts_resolves_indices_within_batched_samples(tmp_path, monkeypatch):
    class DummyModel:
        def rollout(self, *_args, **_kwargs):
            preds = torch.randn(4, 3, 2, 2, 1)
            trues = torch.randn(4, 3, 2, 2, 1)
            return preds, trues

    captured_paths: list[str] = []

    def _fake_plot_spatiotemporal_video(**kwargs):
        captured_paths.append(kwargs["save_path"])

    monkeypatch.setattr(
        "autocast.scripts.eval.encoder_processor_decoder.plot_spatiotemporal_video",
        _fake_plot_spatiotemporal_video,
    )

    out_paths = _render_rollouts(
        model=cast(Any, DummyModel()),
        dataloader=[object()],
        batch_indices=[0, 1, 2, 3],
        video_dir=tmp_path,
        sample_index=0,
        fmt="mp4",
        fps=5,
        stride=1,
        max_rollout_steps=2,
        free_running_only=True,
        n_members=None,
    )

    assert len(out_paths) == 4
    assert len(captured_paths) == 4
    for idx in range(4):
        assert any(f"batch_{idx}_sample_{idx}.mp4" in p for p in captured_paths)
