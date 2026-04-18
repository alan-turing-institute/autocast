import pytest
import torch
from einops import rearrange, repeat

from autocast.metrics import ALL_ENSEMBLE_METRICS
from autocast.metrics.base import BaseMetric
from autocast.metrics.coverage import Coverage
from autocast.metrics.ensemble import (
    EnergyScore,
    SpreadSkillRatio,
    VariogramScore,
    WinklerScore,
    _alpha_fair_crps_score,
    _common_crps_score,
)
from autocast.types import TensorBTSC
from autocast.types.types import TensorBTC

ENSEMBLE_BASE_METRICS = tuple(
    metric_cls
    for metric_cls in ALL_ENSEMBLE_METRICS
    if issubclass(metric_cls, BaseMetric)
)

ENSEMBLE_ERROR_METRICS = tuple(
    m
    for m in ENSEMBLE_BASE_METRICS
    if m not in [Coverage, VariogramScore, SpreadSkillRatio]
)


def _make_metric(MetricCls):
    if MetricCls in (EnergyScore, VariogramScore):
        return MetricCls(vector_dims="spatial")
    return MetricCls()


@pytest.mark.parametrize("MetricCls", ENSEMBLE_ERROR_METRICS)
def test_ensemble_metrics_same(MetricCls):
    # (B, T, S1, S2, C, M) with n_spatial_dims = 2
    y_pred: TensorBTSC = torch.ones((2, 3, 4, 4, 5, 6))
    y_true: TensorBTSC = torch.ones((2, 3, 4, 4, 5))

    # instantiate the metric with n_spatial_dims
    metric = _make_metric(MetricCls)

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
    metric = _make_metric(MetricCls)

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
    metric = _make_metric(MetricCls)

    # score computes the metric over the spatial dims, returning (B, T, C)
    error: TensorBTC = metric.score(y_pred, y_true)

    assert error[:, 0, :].sum() != torch.tensor(0.0)
    assert error[:, 1, :].sum() == torch.tensor(0.0)
    assert error[:, 2, :].sum() == torch.tensor(0.0)


@pytest.mark.parametrize("MetricCls", ENSEMBLE_ERROR_METRICS)
def test_ensemble_metrics_stateful(MetricCls):
    y_pred = torch.ones((2, 3, 4, 4, 5, 6))
    y_true = torch.ones((2, 3, 4, 4, 5))

    metric = _make_metric(MetricCls)
    metric.update(y_pred, y_true)
    value = metric.compute()

    assert torch.allclose(value, torch.tensor(0.0))


def test_energy_score_manual_value():
    # One sample, one timestep, one spatial point, two channels, two members.
    y_pred = torch.tensor([[[[[0.0, 2.0], [0.0, 0.0]]]]])  # (1, 1, 1, 2, 2)
    y_true = torch.tensor([[[[1.0, 0.0]]]])  # (1, 1, 1, 2)

    metric = EnergyScore(
        alpha=1.0,
        vector_dims="spatial_temporal_channels",
        reduce_all=False,
    )
    score = metric.score(y_pred, y_true)

    # term1 = mean(||x_m - y||) = (1 + 1) / 2 = 1
    # term2 = 0.5 * mean(||x_m - x_j||) = 0.5 * (0 + 2 + 2 + 0)/4 = 0.5
    # ES = 1 - 0.5 = 0.5
    expected = torch.tensor([[[0.5]]])
    assert torch.allclose(score, expected)


def test_variogram_score_manual_value():
    # One sample, one timestep, one spatial point, two channels, two members.
    y_pred = torch.tensor([[[[[0.0, 2.0], [0.0, 0.0]]]]])  # (1, 1, 1, 2, 2)
    y_true = torch.tensor([[[[0.0, 0.0]]]])  # (1, 1, 1, 2)

    metric = VariogramScore(
        p=1.0,
        vector_dims="spatial_temporal_channels",
        reduce_all=False,
    )
    score = metric.score(y_pred, y_true)

    # E|X1-X2| = (0 + 2) / 2 = 1, |y1-y2| = 0
    # Off-diagonal terms are 1^2 each, diagonal terms are zero -> total = 2
    expected = torch.tensor([[[2.0]]])
    assert torch.allclose(score, expected)


def test_energy_score_invalid_alpha():
    with pytest.raises(ValueError):  # noqa: PT011
        EnergyScore(alpha=0.0)

    with pytest.raises(ValueError):  # noqa: PT011
        EnergyScore(alpha=2.0)


def test_variogram_score_invalid_parameters():
    with pytest.raises(ValueError):  # noqa: PT011
        VariogramScore(p=0.0)

    bad_weights = torch.ones((3, 3))
    metric = VariogramScore(weights=bad_weights)

    y_pred = torch.ones((1, 1, 1, 2, 2))
    y_true = torch.ones((1, 1, 1, 2))

    with pytest.raises(ValueError):  # noqa: PT011
        metric.score(y_pred, y_true)


@pytest.mark.parametrize("MetricCls", [EnergyScore, VariogramScore])
def test_multivariate_scores_default_vector_dims(MetricCls):
    metric = MetricCls()
    assert metric.vector_dims == "spatial_temporal"


@pytest.mark.parametrize("MetricCls", [EnergyScore, VariogramScore])
@pytest.mark.parametrize(
    ("vector_dims", "expected_shape"),
    [
        ("spatial", (2, 3, 1, 5)),
        ("temporal", (2, 1, 4, 5)),
        ("spatial_temporal", (2, 1, 1, 5)),
        ("spatial_temporal_channels", (2, 1, 1, 1)),
    ],
)
def test_multivariate_scores_vector_dims_shapes(MetricCls, vector_dims, expected_shape):
    y_pred = torch.ones((2, 3, 4, 5, 6))
    y_true = torch.ones((2, 3, 4, 5))

    metric = MetricCls(vector_dims=vector_dims, score_dims=None, reduce_all=False)
    score = metric.score(y_pred, y_true)

    assert score.shape == expected_shape


@pytest.mark.parametrize("MetricCls", [EnergyScore, VariogramScore])
def test_multivariate_scores_vector_dims_typos_invalid(MetricCls):
    with pytest.raises(ValueError):  # noqa: PT011
        MetricCls(vector_dims="spatials")

    with pytest.raises(ValueError):  # noqa: PT011
        MetricCls(vector_dims="temproal")


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


def test_winkler_score_manual_value():
    # Shape: (B=1, T=1, S=2, C=1, M=5)
    # Ensemble members: [0, 1, 2, 3, 4], alpha=0.2
    # Interval bounds: q0.1=0.4, q0.9=3.6
    # S=0: y=2.0 in interval -> score = width = 3.2
    # S=1: y=4.6 above upper by 1.0 -> + (2/0.2)*1.0 = +10.0
    # Mean over spatial points = (3.2 + 13.2) / 2 = 8.2
    members = torch.arange(5.0)
    y_pred = members.view(1, 1, 1, 1, 5).expand(1, 1, 2, 1, 5)
    y_true = torch.tensor([[[[2.0], [4.6]]]])

    value = WinklerScore(alpha=0.2)(y_pred, y_true)
    assert torch.allclose(value, torch.tensor(8.2), atol=1e-6)


def test_winkler_score_invalid_alpha():
    with pytest.raises(ValueError):  # noqa: PT011
        WinklerScore(alpha=0.0)

    with pytest.raises(ValueError):  # noqa: PT011
        WinklerScore(alpha=1.0)


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


def _reference_common_crps_score(
    y_pred: torch.Tensor, y_true: torch.Tensor, adjustment_factor: float
) -> torch.Tensor:
    """Reference O(M^2) implementation kept for test purposes only."""
    n_ensemble = y_pred.shape[-1]
    y_true_expanded = repeat(y_true, "... -> ... m", m=n_ensemble)
    term1 = torch.mean(torch.abs(y_pred - y_true_expanded), dim=-1)
    term2 = (
        0.5
        * torch.mean(
            torch.abs(
                rearrange(y_pred, "... m -> ... 1 m")
                - rearrange(y_pred, "... m -> ... m 1")
            ),
            dim=(-2, -1),
        )
        * adjustment_factor
    )
    return term1 - term2


def _reference_alpha_fair_crps_score(
    y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Reference O(M^2) implementation kept for test purposes only."""
    n_ensemble = y_pred.shape[-1]
    y_true_m = repeat(y_true, "... -> ... m", m=n_ensemble)
    eps = (1.0 - alpha) / n_ensemble

    abs_diff_ens = torch.abs(
        rearrange(y_pred, "... m -> ... 1 m") - rearrange(y_pred, "... m -> ... m 1")
    )
    abs_diff_truth = torch.abs(y_pred - y_true_m)

    mask = ~torch.eye(n_ensemble, dtype=torch.bool, device=y_pred.device)
    mask = mask.view(*([1] * (abs_diff_ens.ndim - 2)), n_ensemble, n_ensemble)

    term_pair = (
        rearrange(abs_diff_truth, "... m -> ... m 1")
        + rearrange(abs_diff_truth, "... m -> ... 1 m")
        - (1.0 - eps) * abs_diff_ens
    )
    term_pair = term_pair.masked_fill(~mask, 0.0)
    sum_pair = term_pair.sum(dim=(-1, -2))
    norm = 2.0 * n_ensemble * (n_ensemble - 1)
    return sum_pair / norm


@pytest.mark.parametrize("adjustment_factor", [1.0, 11.0 / 10.0])
@pytest.mark.parametrize("n_ensemble", [2, 5, 11])
def test_common_crps_matches_reference(adjustment_factor, n_ensemble):
    torch.manual_seed(0)
    y_pred = torch.randn((2, 3, 4, 5, n_ensemble), dtype=torch.float64)
    y_true = torch.randn((2, 3, 4, 5), dtype=torch.float64)

    fast = _common_crps_score(y_pred, y_true, adjustment_factor=adjustment_factor)
    ref = _reference_common_crps_score(
        y_pred, y_true, adjustment_factor=adjustment_factor
    )

    assert torch.allclose(fast, ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("alpha", [0.25, 0.75, 0.95, 1.0])
@pytest.mark.parametrize("n_ensemble", [2, 5, 11])
def test_alpha_fair_crps_matches_reference(alpha, n_ensemble):
    torch.manual_seed(0)
    y_pred = torch.randn((2, 3, 4, 5, n_ensemble), dtype=torch.float64)
    y_true = torch.randn((2, 3, 4, 5), dtype=torch.float64)

    fast = _alpha_fair_crps_score(y_pred, y_true, alpha=alpha)
    ref = _reference_alpha_fair_crps_score(y_pred, y_true, alpha=alpha)

    assert torch.allclose(fast, ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("adjustment_factor", [1.0, 4.0 / 3.0])
def test_common_crps_gradients_match_reference(adjustment_factor):
    torch.manual_seed(0)
    y_pred = torch.randn((2, 2, 3, 2, 4), dtype=torch.float64, requires_grad=True)
    y_true = torch.randn((2, 2, 3, 2), dtype=torch.float64)

    fast = _common_crps_score(y_pred, y_true, adjustment_factor=adjustment_factor).sum()
    (grad_fast,) = torch.autograd.grad(fast, y_pred)

    y_pred_ref = y_pred.detach().clone().requires_grad_(True)
    ref = _reference_common_crps_score(
        y_pred_ref, y_true, adjustment_factor=adjustment_factor
    ).sum()
    (grad_ref,) = torch.autograd.grad(ref, y_pred_ref)

    assert torch.allclose(grad_fast, grad_ref, atol=1e-10, rtol=1e-10)


def test_alpha_fair_crps_gradients_match_reference():
    torch.manual_seed(0)
    y_pred = torch.randn((2, 2, 3, 2, 4), dtype=torch.float64, requires_grad=True)
    y_true = torch.randn((2, 2, 3, 2), dtype=torch.float64)

    fast = _alpha_fair_crps_score(y_pred, y_true, alpha=0.95).sum()
    (grad_fast,) = torch.autograd.grad(fast, y_pred)

    y_pred_ref = y_pred.detach().clone().requires_grad_(True)
    ref = _reference_alpha_fair_crps_score(y_pred_ref, y_true, alpha=0.95).sum()
    (grad_ref,) = torch.autograd.grad(ref, y_pred_ref)

    assert torch.allclose(grad_fast, grad_ref, atol=1e-10, rtol=1e-10)
