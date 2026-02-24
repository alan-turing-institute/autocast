"""Unit tests for evaluation batch-limit resolution."""

from omegaconf import OmegaConf

from autocast.scripts.eval.encoder_processor_decoder import (
    _resolve_rollout_batch_limit,
    _split_metric_and_metadata_rows,
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
