"""Closed-form planted-miscalibration ensembles for the UQ assessment battery.

Two closed-form Gaussian systems with a *known* covariance over one structured axis:

* **ring** (spatial): ``D`` sites on a circle with a circulant covariance
  (``sigma2 * exp(-d^2 / 2 corr_len^2)`` in circular distance ``d``), embedded as the
  spatial axis ``S`` of a metric tensor.
* **OU / AR(1)** (temporal): ``T`` lead steps with the Toeplitz covariance
  ``var * phi^|t-s|``, embedded as the lead axis ``T``.

From each system we draw a **calibrated** ensemble (truth and members share the exact
law) and a battery of **planted** ensembles, each wrong in *exactly one* of the three
assessment columns (plan sec. 2), keeping the other two exactly right:

* column 1 (marginal): ``scale_1.5`` / ``scale_0.67`` -- ensemble spread inflated /
  deflated, correlation structure intact; ``tail_t`` -- correct variance, Student-t
  (nu) *truth* against a Gaussian ensemble (heavy-tail marginal, matched dependence).
* column 2/3 (dependence): ``whiten`` -- exact per-site marginals but the cross-axis
  correlation removed (under-correlated); ``double`` -- exact marginals but a longer
  correlation length (over-correlated).

Every draw returns ``(y_pred, y_true)`` in the metric convention
``y_pred: (B, T, S, C, M)``, ``y_true: (B, T, S, C)`` with ``C = 1``. ``B`` independent
realisations let the coverage / rank diagnostics pool to a stable estimate.

These are TEST FIXTURES, not shipped code; correctness is asserted by
``test_planted_ensembles.py`` (each system reproduces its oracle; each planted variant
is wrong only in its own column).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Failure labels grouped by the column they perturb (the battery keys off these).
MARGINAL_FAILURES = ("scale_1.5", "scale_0.67", "tail_t")
DEPENDENCE_FAILURES = ("whiten", "double")
ALL_FAILURES = ("calibrated", *MARGINAL_FAILURES, *DEPENDENCE_FAILURES)

_TAIL_NU = 5  # Student-t dof for the marginal-tail truth: heavy but finite kurtosis.


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _psd_covariance_from_correlation(corr: torch.Tensor, sigma2: float) -> torch.Tensor:
    """Nearest-PSD covariance with an *exact* ``sigma2`` diagonal.

    Clips tiny negative eigenvalues (a wrapped kernel on a ring need not be exactly
    PSD) then rescales rows/cols so every marginal variance is exactly ``sigma2`` --
    the planted "marginals exact" guarantee depends on this diagonal being exact.
    """
    evals, evecs = torch.linalg.eigh(corr)
    corr = (evecs * evals.clamp_min(1e-10)) @ evecs.T
    inv_std = corr.diagonal().rsqrt()
    corr = corr * inv_std[:, None] * inv_std[None, :]  # unit diagonal
    return sigma2 * corr


def _ring_covariance(n_sites: int, corr_len: float, sigma2: float) -> torch.Tensor:
    idx = torch.arange(n_sites)
    dist = (idx[:, None] - idx[None, :]).abs()
    dist = torch.minimum(dist, n_sites - dist).to(torch.float64)
    corr = torch.exp(-0.5 * (dist / corr_len) ** 2)
    return _psd_covariance_from_correlation(corr, sigma2)


def _ou_covariance(n_steps: int, phi: float, var: float) -> torch.Tensor:
    idx = torch.arange(n_steps)
    lag = (idx[:, None] - idx[None, :]).abs().to(torch.float64)
    return var * (phi**lag)


def _sample_gaussian(
    cov: torch.Tensor, batch: int, members: int, generator: torch.Generator
) -> torch.Tensor:
    """``(batch, members, n)`` zero-mean draws from ``N(0, cov)``."""
    n = cov.shape[0]
    chol = torch.linalg.cholesky(cov)
    z = torch.randn(batch, members, n, dtype=cov.dtype, generator=generator)
    return z @ chol.T


def _sample_student_t(
    cov: torch.Tensor, nu: int, batch: int, members: int, generator: torch.Generator
) -> torch.Tensor:
    """Multivariate-t draws with covariance exactly ``cov`` (shape = (nu-2)/nu cov).

    Heavy-tailed marginals (variance ``diag(cov)``) sharing ``cov``'s correlation, so
    only the tail SHAPE differs from a Gaussian of the same covariance.
    """
    shape_cov = (nu - 2) / nu * cov
    z = _sample_gaussian(shape_cov, batch, members, generator)  # (B, M, n)
    # chi2_nu / nu as the mean of nu squared standard normals (integer nu).
    chi = torch.randn(batch, members, nu, dtype=cov.dtype, generator=generator)
    scale = (chi.pow(2).mean(dim=-1, keepdim=True)).rsqrt()  # (B, M, 1)
    return z * scale


@dataclass(frozen=True)
class PlantedSystem:
    """A closed-form system + its planted draws over one structured axis."""

    cov: torch.Tensor  # (n, n) oracle covariance over the structured axis
    cov_long: torch.Tensor  # (n, n) longer-correlation cov, same diagonal ("double")
    axis: str  # "spatial" -> structured dim is S; "temporal" -> T

    @property
    def n(self) -> int:
        return self.cov.shape[0]

    @property
    def marginal_var(self) -> torch.Tensor:
        return self.cov.diagonal()

    def _to_metric_shape(
        self, ensemble: torch.Tensor, truth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, M, n), (B, n) -> metric-shaped y_pred (B,T,S,C,M), y_true (B,T,S,C)."""
        ensemble = ensemble.to(torch.float32)
        truth = truth.to(torch.float32)
        if self.axis == "spatial":
            # structured dim -> S; T = C = 1
            y_pred = ensemble.permute(0, 2, 1)[:, None, :, None, :]  # (B,1,D,1,M)
            y_true = truth[:, None, :, None]  # (B,1,D,1)
        else:  # temporal: structured dim -> T; S = C = 1
            y_pred = ensemble.permute(0, 2, 1)[:, :, None, None, :]  # (B,T,1,1,M)
            y_true = truth[:, :, None, None]  # (B,T,1,1)
        return y_pred, y_true

    def draw(
        self, failure: str, batch: int, members: int, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(y_pred, y_true)`` for one planted (or calibrated) ensemble."""
        g = _gen(seed)
        # Truth is the calibrated law for every failure except the tail one, where the
        # *truth* carries the heavy tail and the ensemble stays Gaussian-calibrated.
        if failure == "tail_t":
            truth = _sample_student_t(self.cov, _TAIL_NU, batch, 1, g)[:, 0, :]
            ensemble = _sample_gaussian(self.cov, batch, members, g)
        else:
            truth = _sample_gaussian(self.cov, batch, 1, g)[:, 0, :]
            ensemble = self._draw_ensemble(failure, batch, members, g)
        return self._to_metric_shape(ensemble, truth)

    def _draw_ensemble(
        self, failure: str, batch: int, members: int, g: torch.Generator
    ) -> torch.Tensor:
        if failure in ("calibrated",):
            return _sample_gaussian(self.cov, batch, members, g)
        if failure == "scale_1.5":
            return 1.5 * _sample_gaussian(self.cov, batch, members, g)
        if failure == "scale_0.67":
            return 0.67 * _sample_gaussian(self.cov, batch, members, g)
        if failure == "whiten":
            # Exact per-site marginals, cross-axis correlation removed.
            std = self.marginal_var.sqrt()
            z = torch.randn(batch, members, self.n, dtype=self.cov.dtype, generator=g)
            return z * std
        if failure == "double":
            # Exact per-site marginals, longer correlation length (over-correlated).
            return _sample_gaussian(self.cov_long, batch, members, g)
        msg = f"unknown failure {failure!r}; known: {ALL_FAILURES}"
        raise ValueError(msg)


def ring_system(
    n_sites: int = 16, corr_len: float = 2.5, sigma2: float = 1.0
) -> PlantedSystem:
    """Spatial ring system (structured dim -> S). ``cov_long`` doubles corr_len."""
    cov = _ring_covariance(n_sites, corr_len, sigma2)
    cov_long = _ring_covariance(n_sites, 2.0 * corr_len, sigma2)
    return PlantedSystem(cov=cov, cov_long=cov_long, axis="spatial")


def ou_system(n_steps: int = 24, phi: float = 0.6, var: float = 1.0) -> PlantedSystem:
    """Temporal OU/AR(1) system (structured dim -> T); ``cov_long`` lengthens phi."""
    cov = _ou_covariance(n_steps, phi, var)
    # A longer correlation time: phi -> phi**0.5 pushes the lag-1 toward 1 (smoother).
    cov_long = _ou_covariance(n_steps, phi**0.5, var)
    return PlantedSystem(cov=cov, cov_long=cov_long, axis="temporal")
