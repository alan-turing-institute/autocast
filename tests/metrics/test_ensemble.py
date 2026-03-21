import pytest
import torch

from autocast.metrics import ALL_ENSEMBLE_METRICS
from autocast.metrics.base import BaseMetric
from autocast.metrics.coverage import Coverage
from autocast.metrics.ensemble import SpreadSkillRatio
from autocast.types import TensorBTSC
from autocast.types.types import TensorBTC

ENSEMBLE_BASE_METRICS = tuple(
    metric_cls
    for metric_cls in ALL_ENSEMBLE_METRICS
    if issubclass(metric_cls, BaseMetric)
)

ENSEMBLE_ERROR_METRICS = tuple(
    m for m in ENSEMBLE_BASE_METRICS if m not in [Coverage, SpreadSkillRatio]
)


@pytest.mark.parametrize("MetricCls", ENSEMBLE_ERROR_METRICS)
def test_ensemble_metrics_same(MetricCls):
    # (B, T, S1, S2, C, M) with n_spatial_dims = 2
    y_pred: TensorBTSC = torch.ones((2, 3, 4, 4, 5, 6))
    y_true: TensorBTSC = torch.ones((2, 3, 4, 4, 5))

    # instantiate the metric with n_spatial_dims
    metric = MetricCls()

    # score computes the metric over the spatial dims, returning (B, T, C)
    error = metric.score(y_pred, y_true)

    # for identical tensors, all errors must be zero
    assert torch.allclose(error.nansum(), torch.tensor(0.0))


@pytest.mark.parametrize("MetricCls", ENSEMBLE_BASE_METRICS)
def test_ensemble_metrics_wrong_shape(MetricCls):
    # (B, T, S1, S2, C, M) with n_spatial_dims = 2
    y_pred: TensorBTSC = torch.ones((2, 3, 4, 4, 5, 6))
    y_true: TensorBTSC = torch.ones((2, 3, 4, 10, 5))

    # instantiate the metric with n_spatial_dims
    metric = MetricCls()

    with pytest.raises(ValueError):  # noqa: PT011
        # score computes the metric over the spatial dims, returning (B, T, C)
        metric.score(y_pred, y_true)


@pytest.mark.parametrize("MetricCls", ENSEMBLE_ERROR_METRICS)
def test_ensemble_metrics_diff(MetricCls):
    # (B, T, S1, S2, C, M) with n_spatial_dims = 2
    y_pred: TensorBTSC = torch.ones((2, 3, 4, 4, 5, 6))
    y_true: TensorBTSC = torch.ones((2, 3, 4, 4, 5))
    y_true[:, 0, ...] += 1.0

    # instantiate the metric with n_spatial_dims
    metric = MetricCls()

    # score computes the metric over the spatial dims, returning (B, T, C)
    error: TensorBTC = metric.score(y_pred, y_true)

    assert error[:, 0, :].sum() != torch.tensor(0.0)
    assert error[:, 1, :].sum() == torch.tensor(0.0)
    assert error[:, 2, :].sum() == torch.tensor(0.0)


@pytest.mark.parametrize("MetricCls", ENSEMBLE_ERROR_METRICS)
def test_ensemble_metrics_stateful(MetricCls):
    y_pred = torch.ones((2, 3, 4, 4, 5, 6))
    y_true = torch.ones((2, 3, 4, 4, 5))

    metric = MetricCls()
    metric.update(y_pred, y_true)
    value = metric.compute()

    assert torch.allclose(value, torch.tensor(0.0))


def test_spread_skill_ratio_matches_reference_formula():
    # Shape: (B=1, T=1, S=1, C=1, M=2)
    # Members: [0, 2], truth: [0]
    # ensemble_mean = 1 -> skill = sqrt((1 - 0)^2) = 1
    # unbiased ensemble variance = ((0-1)^2 + (2-1)^2) / (2-1) = 2
    # spread = sqrt(2)
    # correction = sqrt((M+1)/M) = sqrt(3/2)
    # corrected SSR = sqrt(2) * sqrt(3/2) = sqrt(3)
    y_pred = torch.tensor([[[[[0.0, 2.0]]]]])
    y_true = torch.tensor([[[[0.0]]]])

    value = SpreadSkillRatio(eps=1e-12)(y_pred, y_true)
    assert torch.allclose(value, torch.tensor(3.0**0.5), atol=1e-6)


def test_spread_skill_ratio_requires_multiple_ensemble_members():
    y_pred = torch.ones((1, 1, 1, 1, 1))
    y_true = torch.ones((1, 1, 1, 1))

    with pytest.raises(ValueError, match="at least 2 ensemble members"):
        SpreadSkillRatio()(y_pred, y_true)


@pytest.mark.parametrize("MetricCls", ALL_ENSEMBLE_METRICS)
def test_ensemble_metrics_device_consistency(MetricCls):
    target_device: torch.device | None = None
    if torch.cuda.is_available():
        target_device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        target_device = torch.device("mps")

    if target_device is None:
        pytest.skip("No GPU/MPS available for device test")

    torch.manual_seed(0)
    # Give coverage a realistic level (some default, it's fine)
    # Ensure SpreadSkillRatio gets > 1 ensemble members and avoids exactly 0 skill
    # by making sure predictions and true differ slightly.
    y_pred = torch.randn((1, 1, 8, 8, 1, 3), device=target_device)
    y_true = torch.zeros((1, 1, 8, 8, 1), device=target_device)

    # Some metrics (like AlphaFairCRPS) might need an explicit kwarg if tested later,
    # but the defaults work for all ALL_ENSEMBLE_METRICS right now.
    metric = MetricCls()

    # Evaluation should succeed without device mismatch errors
    value = metric(y_pred, y_true)

    assert value.device.type == target_device.type
