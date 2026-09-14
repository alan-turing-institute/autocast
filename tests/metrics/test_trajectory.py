import math
from types import SimpleNamespace

import pytest
import torch

from autocast.metrics import MAE, VRMSE, EnergyScore, MultiCoverage
from autocast.metrics.trajectory import (
    TrajectoryMetricAccumulator,
    TrajectoryTask,
    build_trajectory_metadata,
)
from autocast.utils.plots import compute_metrics_from_dataloader


def _metric_fns():
    return {
        "mae": MAE,
        "vrmse": VRMSE,
        "energy": EnergyScore,
        "coverage": lambda: MultiCoverage(coverage_levels=[0.5, 0.9]),
    }


def _two_window_example() -> tuple[torch.Tensor, torch.Tensor]:
    trues = torch.tensor(
        [
            [[[[-1.0], [1.0]]]],
            [[[[-1.0], [1.0]]]],
        ]
    ).squeeze(2)
    preds = torch.zeros((*trues.shape, 2))
    preds[1, ..., 0] = trues[1] - 2.0
    preds[1, ..., 1] = trues[1] + 2.0
    return preds, trues


def test_accumulator_groups_overlapping_windows_by_trajectory():
    preds, trues = _two_window_example()
    accumulator = TrajectoryMetricAccumulator(
        sample_trajectory_indices=[0, 0],
        sample_window_indices=[0, 1],
        windows=[None],
        metric_fns=_metric_fns(),
    )

    accumulator.update(preds[:1], trues[:1])
    accumulator.update(preds[1:], trues[1:])
    accumulator.validate_complete()
    rows = accumulator.window_rows(task=TrajectoryTask.SINGLE_STEP)

    assert len(rows) == 1
    row = rows[0]
    assert row["trajectory_idx"] == 0
    assert row["trajectory_id"] == "test_0000"
    assert row["window"] == "all"
    assert row["n_windows"] == 2
    assert row["n_timesteps"] == 2
    assert row["n_vrmse_components"] == 2
    assert row["n_coverage_components"] == 4
    assert row["mae"] == pytest.approx(0.5)
    assert row["vrmse"] == pytest.approx(1.0 / (2.0 * math.sqrt(2.0)))
    assert row["coverage_0.50"] == pytest.approx(0.5)
    assert row["coverage_0.90"] == pytest.approx(0.5)
    assert row["coverage_mae"] == pytest.approx(0.2)


def test_rollout_window_and_lead_rows_share_the_same_predictions():
    preds, trues = _two_window_example()
    preds = preds.movedim(0, 1)
    trues = trues.movedim(0, 1)
    accumulator = TrajectoryMetricAccumulator(
        sample_trajectory_indices=[0],
        sample_window_indices=[0],
        windows=[(0, 2)],
        metric_fns=_metric_fns(),
        include_per_timestep=True,
    )

    accumulator.update(preds, trues)
    window_rows = accumulator.window_rows(task=TrajectoryTask.ROLLOUT_WINDOW)
    lead_rows = accumulator.per_timestep_rows()

    assert len(window_rows) == 1
    assert len(lead_rows) == 2
    assert window_rows[0]["vrmse"] == pytest.approx(
        sum(float(row["vrmse"]) for row in lead_rows) / 2.0
    )
    assert window_rows[0]["coverage_0.50"] == pytest.approx(
        sum(float(row["coverage_0.50"]) for row in lead_rows) / 2.0
    )
    assert [row["lead_time"] for row in lead_rows] == [0, 1]
    for lead_time, row in enumerate(lead_rows):
        metric = EnergyScore()
        metric.update(
            preds[:, lead_time : lead_time + 1],
            trues[:, lead_time : lead_time + 1],
        )
        assert row["energy"] == pytest.approx(metric.compute().item())


def test_stream_callback_uses_the_same_inference_batch():
    preds, trues = _two_window_example()
    accumulator = TrajectoryMetricAccumulator(
        sample_trajectory_indices=[0, 0],
        sample_window_indices=[0, 1],
        windows=[None],
        metric_fns={"vrmse": VRMSE},
    )
    predict_calls = 0

    def predict_fn(_batch):
        nonlocal predict_calls
        predict_calls += 1
        return preds, trues

    aggregate, _, _ = compute_metrics_from_dataloader(
        dataloader=[object()],
        metric_fns={"vrmse": VRMSE},
        predict_fn=predict_fn,
        batch_result_callback=accumulator.update,
    )

    accumulator.validate_complete()
    row = accumulator.window_rows(task=TrajectoryTask.SINGLE_STEP)[0]
    assert predict_calls == 1
    assert row["vrmse"] == pytest.approx(aggregate[None]["vrmse"].compute().item())


def test_accumulator_rejects_incomplete_metadata_consumption():
    preds, trues = _two_window_example()
    accumulator = TrajectoryMetricAccumulator(
        sample_trajectory_indices=[0, 0],
        sample_window_indices=[0, 1],
        windows=[None],
        metric_fns=_metric_fns(),
    )

    accumulator.update(preds[:1], trues[:1])

    with pytest.raises(RuntimeError, match="consumed 1 of 2"):
        accumulator.validate_complete()


def test_trajectory_metadata_exports_generic_constant_scalar_columns():
    values = torch.tensor([[0.030, 0.062, 2.0e-5], [0.014, 0.054, 2.0e-5]])
    dataset = SimpleNamespace(
        constant_scalars=values,
        data_path="/datasets/gray_scott_68b0669/test/data.pt",
    )

    metadata = build_trajectory_metadata(dataset)

    assert metadata == {
        0: {
            "cs0": pytest.approx(0.030),
            "cs1": pytest.approx(0.062),
            "cs2": pytest.approx(2.0e-5),
        },
        1: {
            "cs0": pytest.approx(0.014),
            "cs1": pytest.approx(0.054),
            "cs2": pytest.approx(2.0e-5),
        },
    }
