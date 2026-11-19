"""Tests that DSS's low-rank Gaussian terms equal the dense covariance path.

``_shrunk_covariance`` returns ``Sigma = (1-gamma) X_c X_c^T / M + (gamma*nu +
eps) I``, which is diagonal-plus-rank-M. :func:`_shrunk_gaussian_terms` uses that
structure to get ``log det Sigma`` and the Mahalanobis term from an ``(M, M)``
factorisation instead of a ``(D, D)`` one -- O(D M^2 + M^3) rather than O(D^3),
and without ever materialising the covariance.

The change is only admissible if it is *exact*, so these tests compare it against
the dense path it replaces (``_dense_gaussian_terms``, still reachable as the
zero-floor fallback) and require agreement at float64 round-off, for both
shrinkage modes and across the rank-deficient regime D >> M that production runs
in.
"""

import math

import pytest
import torch

from autocast.metrics.ensemble import (
    DawidSebastianiScore,
    _shrunk_covariance,
    _shrunk_gaussian_terms,
)

M = 8
JITTER = 1e-6
# Relative tolerance for "the same number in float64". The dense path does a
# D-dimensional Cholesky, so its own rounding grows with D; 1e-9 is many orders
# tighter than any difference a wrong formula would produce.
RTOL = 1e-9


def _dense_terms(members, obs, method, intensity, jitter=JITTER):
    """log-det and Mahalanobis exactly as the pre-2026-08-19 DSS computed them."""
    cov, mean, _gamma = _shrunk_covariance(members, method, intensity, jitter)
    chol = torch.linalg.cholesky(cov)
    log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    delta = (obs - mean).unsqueeze(-1)
    whitened = torch.linalg.solve_triangular(chol, delta, upper=False)
    return log_det, (whitened**2).sum(dim=(-2, -1))


@pytest.mark.parametrize("dim", [16, 128, 512])
@pytest.mark.parametrize(
    ("method", "intensity"), [("ledoit_wolf", None), ("fixed", 0.1)]
)
def test_lowrank_terms_match_dense(dim, method, intensity):
    """Both terms agree with the dense covariance path to float64 round-off."""
    torch.manual_seed(0)
    members = torch.randn(3, dim, M, dtype=torch.float64)
    obs = torch.randn(3, dim, dtype=torch.float64)

    ld_dense, mh_dense = _dense_terms(members, obs, method, intensity)
    ld_low, mh_low = _shrunk_gaussian_terms(members, obs, method, intensity, JITTER)

    torch.testing.assert_close(ld_low, ld_dense, rtol=RTOL, atol=0.0)
    torch.testing.assert_close(mh_low, mh_dense, rtol=RTOL, atol=0.0)


def test_lowrank_terms_do_not_materialise_the_covariance():
    """A D where a dense (D, D) float64 covariance would be ~8 GB still runs.

    32768^2 float64 is 8.6 GB per frame; the low-rank path holds (D, M) and
    (M, M) only, so this completes in milliseconds on CPU. If someone
    reintroduces the dense form this test does not fail subtly -- it OOMs.
    """
    torch.manual_seed(0)
    dim = 32768
    members = torch.randn(1, dim, M, dtype=torch.float64)
    obs = torch.randn(1, dim, dtype=torch.float64)

    log_det, mahalanobis = _shrunk_gaussian_terms(
        members, obs, "ledoit_wolf", None, JITTER
    )
    assert torch.isfinite(log_det).all()
    assert torch.isfinite(mahalanobis).all()
    assert (mahalanobis > 0).all()


def test_zero_floor_raises_when_sigma_is_genuinely_singular():
    """A zero floor with M - 1 < D leaves Sigma singular; say so, don't emit inf."""
    torch.manual_seed(0)
    members = torch.randn(2, 32, M, dtype=torch.float64)
    obs = torch.randn(2, 32, dtype=torch.float64)
    with pytest.raises(ValueError, match="singular"):
        _shrunk_gaussian_terms(members, obs, "fixed", 0.0, 0.0)


def test_zero_floor_falls_back_to_dense_when_sigma_is_invertible():
    """M - 1 >= D: Sigma = B B^T is full rank, so the zero floor is legitimate.

    Woodbury cannot be used (it divides by the floor), so the dense path takes
    over. This is the regime ``test_dss_fixed_shrinkage_reduces_to_sample_cov``
    exercises -- 50 members over 6 sites -- and it must keep working.
    """
    torch.manual_seed(0)
    dim, many = 6, 50
    members = torch.randn(2, dim, many, dtype=torch.float64)
    obs = torch.randn(2, dim, dtype=torch.float64)

    got = _shrunk_gaussian_terms(members, obs, "fixed", 0.0, 0.0)
    expected = _dense_terms(members, obs, "fixed", 0.0, jitter=0.0)
    torch.testing.assert_close(got[0], expected[0], rtol=RTOL, atol=0.0)
    torch.testing.assert_close(got[1], expected[1], rtol=RTOL, atol=0.0)


def test_metric_end_to_end_matches_hand_computed_score():
    """The public metric still returns 0.5*(D log 2pi + logdet + mahalanobis)."""
    torch.manual_seed(0)
    b, t, h, w, c = 2, 1, 4, 4, 1
    y_pred = torch.randn(b, t, h, w, c, M, dtype=torch.float64)
    y_true = torch.randn(b, t, h, w, c, dtype=torch.float64)

    metric = DawidSebastianiScore(reduce_all=True)
    metric.update(y_pred, y_true)
    got = metric.compute()

    dim = h * w
    members = y_pred.reshape(b * t, dim, M)
    obs = y_true.reshape(b * t, dim)
    log_det, mahalanobis = _shrunk_gaussian_terms(
        members, obs, metric.shrinkage, metric.shrinkage_intensity, metric.jitter
    )
    expected = (0.5 * (dim * math.log(2.0 * math.pi) + log_det + mahalanobis)).mean()

    torch.testing.assert_close(got.to(torch.float64), expected, rtol=1e-9, atol=0.0)
