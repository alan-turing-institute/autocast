"""Self-tests for the planted-miscalibration generators (Phase 4.1).

The UQ assessment battery (Phase 4.2) is only meaningful if each generator has the
statistical property it claims. These tests verify, from the *raw* draws, that the
calibrated ensembles reproduce their oracle covariance and that every planted variant
is wrong in exactly its own column while leaving the others exact.
"""

import pytest
import torch

from ._planted_ensembles import (
    _TAIL_NU,
    PlantedSystem,
    ou_system,
    ring_system,
)


def _structured(y_pred: torch.Tensor, system: PlantedSystem) -> torch.Tensor:
    """Metric y_pred (B,T,S,C,M) -> pooled draws (B*M, n) over the structured axis."""
    if system.axis == "spatial":  # noqa: SIM108 (ternary would exceed the line limit)
        draws = y_pred[:, 0, :, 0, :]  # (B, D, M)
    else:
        draws = y_pred[:, :, 0, 0, :]  # (B, T, M)
    b, n, m = draws.shape
    return draws.permute(0, 2, 1).reshape(b * m, n)  # (B*M, n)


def _empirical_cov(samples: torch.Tensor) -> torch.Tensor:
    centered = samples - samples.mean(dim=0, keepdim=True)
    return centered.T @ centered / (samples.shape[0] - 1)


def _corr_from_cov(cov: torch.Tensor) -> torch.Tensor:
    inv_std = cov.diagonal().rsqrt()
    return cov * inv_std[:, None] * inv_std[None, :]


SYSTEMS = {"ring": ring_system(), "ou": ou_system()}


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_calibrated_ensemble_reproduces_oracle_covariance(name):
    system = SYSTEMS[name]
    y_pred, y_true = system.draw("calibrated", batch=400, members=64, seed=0)
    emp = _empirical_cov(_structured(y_pred, system).double())
    # Whole covariance matrix within sampling error of the oracle.
    assert torch.allclose(emp, system.cov, atol=0.05)
    # Truth carries the same law (coarse matched-marginal-variance sanity check;
    # only `batch` truth samples, so the tight validation is the ensemble cov above).
    truth = y_true[:, 0, :, 0] if system.axis == "spatial" else y_true[:, :, 0, 0]
    assert torch.allclose(truth.double().var(dim=0), system.marginal_var, atol=0.2)


@pytest.mark.parametrize("name", list(SYSTEMS))
@pytest.mark.parametrize(
    ("failure", "scale"), [("scale_1.5", 1.5), ("scale_0.67", 0.67)]
)
def test_scale_failure_is_marginal_only(name, failure, scale):
    """Ensemble variance is scaled by s^2; the correlation matrix is unchanged."""
    system = SYSTEMS[name]
    y_pred, _ = system.draw(failure, batch=400, members=64, seed=1)
    emp = _empirical_cov(_structured(y_pred, system).double())
    assert torch.allclose(emp.diagonal(), scale**2 * system.marginal_var, atol=0.1)
    # Correlation (dependence) is scale-invariant -> unchanged.
    assert torch.allclose(_corr_from_cov(emp), _corr_from_cov(system.cov), atol=0.05)


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_whiten_failure_is_dependence_only(name):
    """Exact per-site marginals, but the off-diagonal correlation is removed."""
    system = SYSTEMS[name]
    y_pred, _ = system.draw("whiten", batch=400, members=64, seed=2)
    emp = _empirical_cov(_structured(y_pred, system).double())
    # Marginals exact.
    assert torch.allclose(emp.diagonal(), system.marginal_var, atol=0.1)
    # Off-diagonal correlation ~ 0 while the oracle has real structure.
    off = ~torch.eye(system.n, dtype=torch.bool)
    assert _corr_from_cov(emp)[off].abs().max() < 0.06
    assert _corr_from_cov(system.cov)[off].abs().max() > 0.3


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_double_failure_is_dependence_only(name):
    """Exact per-site marginals, but a longer correlation length (over-correlated)."""
    system = SYSTEMS[name]
    y_pred, _ = system.draw("double", batch=400, members=64, seed=3)
    emp = _empirical_cov(_structured(y_pred, system).double())
    assert torch.allclose(emp.diagonal(), system.marginal_var, atol=0.1)
    # Correlation at a representative near-neighbour lag exceeds the oracle's.
    emp_corr = _corr_from_cov(emp)
    oracle_corr = _corr_from_cov(system.cov)
    i, j = 0, 2
    assert emp_corr[i, j] > oracle_corr[i, j] + 0.05


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_tail_failure_is_marginal_tail_only(name):
    """Truth has matched variance but heavy tails; the ensemble stays Gaussian."""
    system = SYSTEMS[name]
    y_pred, y_true = system.draw("tail_t", batch=4000, members=32, seed=4)
    truth = y_true[:, 0, :, 0] if system.axis == "spatial" else y_true[:, :, 0, 0]
    truth = truth.double()
    # Matched variance (dependence/variance column stays clean).
    assert torch.allclose(truth.var(dim=0), system.marginal_var, atol=0.15)
    # Heavier-than-Gaussian tails: standardized 3-sigma exceedance well above 0.0027.
    std = truth.std(dim=0, keepdim=True)
    exceed = (truth.abs() / std > 3.0).double().mean().item()
    assert exceed > 2 * 0.0027
    # Ensemble is Gaussian-calibrated -> its own 3-sigma exceedance ~ Gaussian.
    ens = _structured(y_pred, system).double()
    ens_std = ens.std(dim=0, keepdim=True)
    ens_exceed = (ens.abs() / ens_std > 3.0).double().mean().item()
    assert ens_exceed < 3 * 0.0027


def test_metric_shapes_match_convention():
    ring = ring_system(n_sites=16)
    yp, yt = ring.draw("calibrated", batch=5, members=7, seed=0)
    assert yp.shape == (5, 1, 16, 1, 7)
    assert yt.shape == (5, 1, 16, 1)
    ou = ou_system(n_steps=24)
    yp, yt = ou.draw("calibrated", batch=5, members=7, seed=0)
    assert yp.shape == (5, 24, 1, 1, 7)
    assert yt.shape == (5, 24, 1, 1)


def test_tail_dof_is_moderate_and_finite_kurtosis():
    assert _TAIL_NU > 4  # finite 4th moment for a stable "matched variance" check
