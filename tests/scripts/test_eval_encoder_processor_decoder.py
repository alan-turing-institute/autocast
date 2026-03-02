"""Unit tests for evaluation batch-limit resolution."""

import pytest
from omegaconf import OmegaConf

from autocast.scripts.eval.encoder_processor_decoder import (
    _benchmark_metadata_rows,
    _extract_epoch_times_from_checkpoint,
    _extract_training_runtime_total_s,
    _resolve_rollout_batch_limit,
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


def test_benchmark_metadata_rows_include_config_and_metrics():
    rows = _benchmark_metadata_rows(
        {
            "throughput_samples_per_sec": 123.4,
            "latency_ms_per_batch": 10.0,
        },
        batch_size=2,
        n_warmup=5,
        n_benchmark=20,
    )

    metrics = {(row["category"], row["metric"]) for row in rows}
    assert ("benchmark", "batch_size") in metrics
    assert ("benchmark", "n_warmup") in metrics
    assert ("benchmark", "n_benchmark") in metrics
    assert ("benchmark", "throughput_samples_per_sec") in metrics
    assert ("benchmark", "latency_ms_per_batch") in metrics


# --- _extract_training_runtime_total_s ---


def test_extract_training_runtime_reads_top_level_key():
    payload = {"training_runtime_total_s": 42.0}
    assert _extract_training_runtime_total_s(payload) == pytest.approx(42.0)


def test_extract_training_runtime_reads_callback_state_dict():
    payload = {
        "callbacks": {
            "autocast.scripts.training.TrainingTimerCallback": {
                "training_runtime_total_s": 99.5,
                "epoch_times_s": [10.0, 11.0],
            }
        }
    }
    assert _extract_training_runtime_total_s(payload) == pytest.approx(99.5)


def test_extract_training_runtime_returns_none_when_absent():
    assert _extract_training_runtime_total_s({}) is None
    assert _extract_training_runtime_total_s({"callbacks": {}}) is None


# --- _extract_epoch_times_from_checkpoint ---


def test_extract_epoch_times_returns_list_from_callback():
    payload = {
        "callbacks": {
            "TrainingTimerCallback": {
                "epoch_times_s": [1.0, 2.0, 3.0],
            }
        }
    }
    assert _extract_epoch_times_from_checkpoint(payload) == pytest.approx(
        [1.0, 2.0, 3.0]
    )


def test_extract_epoch_times_returns_empty_when_absent():
    assert _extract_epoch_times_from_checkpoint({}) == []
    assert _extract_epoch_times_from_checkpoint({"callbacks": {}}) == []


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

    assert by_metric["epochs_completed"] == 3
    assert by_metric["total_s"] == pytest.approx(30.0)
    assert by_metric["mean_epoch_s"] == pytest.approx(10.0)
    assert by_metric["min_epoch_s"] == pytest.approx(8.0)
    assert by_metric["max_epoch_s"] == pytest.approx(12.0)
    # per_epoch_s (average fallback) should NOT be emitted when actual times exist
    assert "per_epoch_s" not in by_metric


def test_training_runtime_rows_fall_back_to_average_without_epoch_times():
    payload = {
        "epoch": 3,
        "global_step": 400,
        "training_runtime_total_s": 40.0,
    }
    rows = _training_runtime_rows(payload)
    by_metric = {r["metric"]: r["value"] for r in rows}

    assert by_metric["mean_epoch_s"] == pytest.approx(10.0)  # 40 / 4 epochs
    assert "min_epoch_s" not in by_metric
    assert "max_epoch_s" not in by_metric
