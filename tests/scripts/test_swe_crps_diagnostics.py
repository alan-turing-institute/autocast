"""Tests for shallow-water CRPS diagnostic metrics."""

import pytest
import torch

from autocast.scripts.eval.swe_crps_diagnostics import (
    compute_fit_metrics,
    compute_swe_structure_metrics,
)


def test_compute_fit_metrics_uses_ensemble_mean_and_member_axis():
    truth = torch.zeros(1, 2, 2, 2, 3)
    prediction = torch.stack((truth + 1.0, truth - 1.0), dim=-1)
    persistence = truth + 2.0

    metrics = compute_fit_metrics(prediction, truth, persistence)

    assert metrics["ensemble_mean_mae"] == [0.0, 0.0, 0.0]
    assert metrics["expected_member_mae"] == [1.0, 1.0, 1.0]
    assert metrics["persistence_mae"] == [2.0, 2.0, 2.0]
    assert metrics["ensemble_mean_over_persistence"] == [0.0, 0.0, 0.0]


def test_swe_structure_metrics_identify_coherent_member_anomalies():
    prediction = torch.zeros(1, 2, 8, 8, 3, 2)
    prediction[..., 1:, 0] = 1.0
    prediction[..., 1:, 1] = -1.0

    metrics = compute_swe_structure_metrics(prediction, high_k_cutoff=3.0)

    assert metrics["high_k_fraction"]["mean"] == pytest.approx(0.0, abs=1.0e-7)
    assert metrics["neighbor_corr_x"]["mean"] == pytest.approx(1.0)
    assert metrics["neighbor_corr_y"]["mean"] == pytest.approx(1.0)


def test_swe_structure_metrics_identify_checkerboard_member_anomalies():
    x_index = torch.arange(8)[:, None]
    y_index = torch.arange(8)[None, :]
    checkerboard = (1 - 2 * ((x_index + y_index) % 2)).float()
    prediction = torch.zeros(1, 1, 8, 8, 3, 2)
    prediction[0, 0, ..., 1:, 0] = checkerboard[..., None]
    prediction[0, 0, ..., 1:, 1] = -checkerboard[..., None]

    metrics = compute_swe_structure_metrics(prediction, high_k_cutoff=3.0)

    assert metrics["high_k_fraction"]["mean"] == pytest.approx(1.0)
    assert metrics["absolute_high_k_energy"]["mean"] > 0.0
    assert metrics["neighbor_corr_x"]["mean"] == pytest.approx(-1.0)
    assert metrics["neighbor_corr_y"]["mean"] == pytest.approx(-1.0)
