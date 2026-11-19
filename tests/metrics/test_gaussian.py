import math

import pytest
import torch

from autocast.metrics import CRPS, GaussianCoverage, GaussianCRPS, GaussianWinkler


def _pack(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return torch.cat([mu, log_var], dim=-1)


def test_gaussian_crps_standard_normal_at_mean():
    # mu=0, sigma=1, y=0 -> CRPS = sqrt(2/pi) - 1/sqrt(pi)
    mu = torch.zeros(1, 1, 1, 1)
    log_var = torch.zeros(1, 1, 1, 1)
    target = torch.zeros(1, 1, 1, 1)

    metric = GaussianCRPS()
    metric.update(_pack(mu, log_var), target)
    value = metric.compute().item()

    expected = math.sqrt(2.0 / math.pi) - 1.0 / math.sqrt(math.pi)
    assert value == pytest.approx(expected, abs=1e-6)


def test_gaussian_crps_standard_normal_at_z_one():
    # mu=0, sigma=1, y=1 -> closed-form value
    mu = torch.zeros(1, 1, 1, 1)
    log_var = torch.zeros(1, 1, 1, 1)
    target = torch.ones(1, 1, 1, 1)

    metric = GaussianCRPS()
    metric.update(_pack(mu, log_var), target)
    value = metric.compute().item()

    z = 1.0
    cdf_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    phi_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    expected = 1.0 * (z * (2.0 * cdf_z - 1.0) + 2.0 * phi_z - 1.0 / math.sqrt(math.pi))
    assert value == pytest.approx(expected, abs=1e-6)


def test_gaussian_crps_scales_linearly_with_sigma():
    # Standardised CRPS at fixed z should scale linearly with sigma.
    log_var = math.log(4.0)  # sigma = 2
    mu = torch.zeros(1, 1, 1, 1)
    target = torch.full_like(mu, 2.0)  # z = 1
    metric = GaussianCRPS()
    metric.update(_pack(mu, torch.full_like(mu, log_var)), target)
    value = metric.compute().item()

    # Same z=1 with sigma=1 reference:
    metric_ref = GaussianCRPS()
    metric_ref.update(
        _pack(torch.zeros_like(mu), torch.zeros_like(mu)), torch.ones_like(mu)
    )
    ref_value = metric_ref.compute().item()
    assert value == pytest.approx(2.0 * ref_value, abs=1e-6)


def test_gaussian_crps_matches_sample_crps_in_large_sample_limit():
    torch.manual_seed(0)
    mu_val = 2.0
    sigma_val = 1.5
    target_val = 1.0
    log_var_val = 2.0 * math.log(sigma_val)

    # Analytic Gaussian CRPS on a single point
    mu = torch.full((1, 1, 1, 1), mu_val)
    log_var = torch.full((1, 1, 1, 1), log_var_val)
    target = torch.full((1, 1, 1, 1), target_val)
    gcrps = GaussianCRPS()
    gcrps.update(_pack(mu, log_var), target)
    analytic = gcrps.compute().item()

    # Sample-based CRPS using the existing ensemble metric.
    # Shape (B, T, S, C, M) where M = n_members.
    n_members = 4000
    samples = torch.randn(1, 1, 1, 1, n_members) * sigma_val + mu_val
    truth = torch.full((1, 1, 1, 1), target_val)
    sample_metric = CRPS(score_dims=None, reduce_all=True)
    sample_metric.update(samples, truth)
    sample_value = sample_metric.compute().item()

    # 1/sqrt(M) scaled tolerance with a small slack
    tolerance = 5.0 / math.sqrt(n_members)
    assert abs(sample_value - analytic) < tolerance


def test_gaussian_coverage_constructor_validates_level():
    with pytest.raises(ValueError, match="coverage_level must be in"):
        GaussianCoverage(coverage_level=0.0)
    with pytest.raises(ValueError, match="coverage_level must be in"):
        GaussianCoverage(coverage_level=1.0)


def test_gaussian_coverage_calibrated_under_correct_predictive():
    torch.manual_seed(42)
    n = 20000
    target = torch.randn(1, 1, n, 1)
    mu = torch.zeros_like(target)
    log_var = torch.zeros_like(target)  # sigma = 1

    metric = GaussianCoverage(coverage_level=0.9)
    metric.update(_pack(mu, log_var), target)
    fraction = metric.compute().item()

    # 1/sqrt(n) ~ 0.007, plus a small slack
    assert fraction == pytest.approx(0.9, abs=0.02)


def test_gaussian_coverage_value_at_known_points():
    # mu=0, sigma=1, coverage=0.9 -> half-width = 1.6449
    z_half = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.95)).item()
    target = torch.tensor(
        [[[[0.0], [z_half - 1e-3], [z_half + 1e-3], [-(z_half + 1.0)]]]]
    )
    mu = torch.zeros_like(target)
    log_var = torch.zeros_like(target)

    metric = GaussianCoverage(coverage_level=0.9)
    metric.update(_pack(mu, log_var), target)
    fraction = metric.compute().item()

    # 2 of 4 are inside, 2 are outside -> 0.5
    assert fraction == pytest.approx(0.5, abs=1e-6)


def test_gaussian_winkler_constructor_validates_level():
    with pytest.raises(ValueError, match="coverage_level must be in"):
        GaussianWinkler(coverage_level=0.0)


def test_gaussian_winkler_value_inside_interval_equals_width():
    z_half = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.95)).item()
    mu = torch.zeros(1, 1, 1, 1)
    log_var = torch.zeros(1, 1, 1, 1)
    target = torch.zeros(1, 1, 1, 1)  # inside the interval

    metric = GaussianWinkler(coverage_level=0.9)
    metric.update(_pack(mu, log_var), target)
    value = metric.compute().item()

    expected_width = 2.0 * z_half  # sigma = 1
    assert value == pytest.approx(expected_width, abs=1e-6)


def test_gaussian_winkler_value_above_interval_includes_penalty():
    z_half = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.95)).item()
    mu = torch.zeros(1, 1, 1, 1)
    log_var = torch.zeros(1, 1, 1, 1)
    target = torch.full((1, 1, 1, 1), z_half + 1.0)  # above upper bound by 1

    metric = GaussianWinkler(coverage_level=0.9)
    metric.update(_pack(mu, log_var), target)
    value = metric.compute().item()

    alpha = 0.1
    expected = 2.0 * z_half + (2.0 / alpha) * 1.0
    assert value == pytest.approx(expected, abs=1e-6)


def test_gaussian_metric_rejects_ndim_mismatch():
    y_pred = torch.zeros(1, 1, 1, 1, 2)
    target = torch.zeros(1, 1, 1, 1)
    with pytest.raises(ValueError, match="pack"):
        GaussianCRPS().update(y_pred, target)


def test_gaussian_metric_rejects_non_packed_channels():
    # y_pred.shape[-1] != 2 * target.shape[-1]
    y_pred = torch.zeros(1, 1, 1, 3)
    target = torch.zeros(1, 1, 1, 1)
    with pytest.raises(ValueError, match="pack"):
        GaussianCRPS().update(y_pred, target)


def test_gaussian_metric_compute_before_update_raises():
    with pytest.raises(RuntimeError, match="No samples"):
        GaussianCRPS().compute()


def test_gaussian_metric_accumulates_across_updates():
    mu = torch.zeros(1, 1, 1, 1)
    log_var = torch.zeros(1, 1, 1, 1)
    target_a = torch.zeros(1, 1, 1, 1)
    target_b = torch.ones(1, 1, 1, 1)

    metric = GaussianCRPS()
    metric.update(_pack(mu, log_var), target_a)
    metric.update(_pack(mu, log_var), target_b)
    combined = metric.compute().item()

    # Same answer as a single update of two stacked elements
    metric_ref = GaussianCRPS()
    metric_ref.update(
        _pack(torch.zeros(1, 1, 2, 1), torch.zeros(1, 1, 2, 1)),
        torch.cat([target_a, target_b], dim=2),
    )
    expected = metric_ref.compute().item()
    assert combined == pytest.approx(expected, abs=1e-6)


def test_gaussian_metric_per_axis_reduction_returns_tensor():
    # reduce_all=False keeps the per-axis (T, ..., C) shape after dividing by
    # the batch count.
    n_t, n_s, n_c = 2, 3, 2
    mu = torch.zeros(4, n_t, n_s, n_c)
    log_var = torch.zeros(4, n_t, n_s, n_c)
    target = torch.zeros(4, n_t, n_s, n_c)

    metric = GaussianCRPS(reduce_all=False)
    metric.update(_pack(mu, log_var), target)
    result = metric.compute()

    assert result.shape == (n_t, n_s, n_c)
    # At z=0 every element is the standard-normal-at-mean CRPS
    expected = math.sqrt(2.0 / math.pi) - 1.0 / math.sqrt(math.pi)
    assert torch.allclose(result, torch.full_like(result, expected), atol=1e-6)
