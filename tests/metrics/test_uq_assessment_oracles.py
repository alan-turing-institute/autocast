"""Closed-form oracle checks for the UQ assessment suite (plan sec. 4.3).

Where the column-exclusivity battery (``test_uq_assessment_battery.py``) proves the
metrics FLAG the right failures, these tests prove the complementary property: on a
CALIBRATED ensemble every column-1/3 diagnostic sits at its *closed-form* reference,
so the suite does not raise false alarms and its calibrated readings agree with the
known law of the ring / OU systems (not merely with a self-consistent null).

The oracles used, all analytic for the Gaussian ring / OU systems of
``_planted_ensembles``:

* coverage of a calibrated ensemble equals the nominal level (M -> inf);
* the per-site / per-lead ensemble variance equals the oracle covariance diagonal;
* the member-pooled temporal autocorrelation recovers ``phi**k`` (OU);
* the ensemble pairwise variogram ``E|X_i - X_j|**p`` equals the Gaussian value
  ``v_ij**(p/2) * 2**(p/2) * Gamma((p+1)/2) / sqrt(pi)`` with
  ``v_ij = Sigma_ii + Sigma_jj - 2 Sigma_ij``;
* the rank histograms of a calibrated ensemble are uniform (reliability index Delta
  near its finite-sample floor, mean rank centred).

Larger ensembles than the battery (``M_ORACLE``) tighten these to closed-form
tolerances.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from einops import rearrange

from autocast.metrics.coverage import Coverage
from autocast.metrics.rank_histogram import (
    DependenceRankHistogram,
    MultivariateRankHistogram,
)

from ._planted_ensembles import ou_system, ring_system

M_ORACLE = 96  # large ensemble -> closed-form tolerances
B_ORACLE = 256
SEED = 4242


def _mean_rank(metric) -> float:
    counts = metric.counts.cpu().numpy()
    ranks = np.arange(1, len(counts) + 1)
    return float((counts * ranks).sum() / counts.sum() / len(counts))


@pytest.fixture(scope="module")
def ring():
    return ring_system(n_sites=32, corr_len=2.5)


@pytest.fixture(scope="module")
def ou():
    return ou_system()


# ---------------------------------------------------------------------------
# Column 1 oracle: calibrated coverage matches the nominal level across a curve.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("system_name", ["ring", "ou"])
@pytest.mark.parametrize("level", [0.5, 0.7, 0.9])
def test_calibrated_coverage_matches_nominal(system_name, level, ring, ou):
    system = ring if system_name == "ring" else ou
    y_pred, y_true = system.draw(
        "calibrated", batch=B_ORACLE, members=M_ORACLE, seed=SEED
    )
    cov = Coverage(coverage_level=level)
    cov.update(y_pred, y_true)
    observed = float(cov.compute().item())
    assert observed == pytest.approx(level, abs=0.03), (
        f"{system_name}: coverage at level {level} was {observed:.3f}"
    )


# NB: the corrected spread-skill ratio of a calibrated ensemble is NOT 1 on this
# system (empirically ~1.11): SSR reduces the per-realisation spread/skill as a
# ratio of square-roots of spatial means, and over a *correlated* field that
# ratio carries a Jensen bias (few effective independent sites). SSR calibration
# is read from its null band in the battery, not against a closed-form 1 -- so the
# clean spread oracle here is the marginal-variance check below, not SSR.


# ---------------------------------------------------------------------------
# Marginal-variance oracle: ensemble per-site / per-lead variance == diag(Sigma).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("system_name", ["ring", "ou"])
def test_calibrated_marginal_variance_matches_oracle(system_name, ring, ou):
    system = ring if system_name == "ring" else ou
    y_pred, _ = system.draw("calibrated", batch=B_ORACLE, members=M_ORACLE, seed=SEED)
    # (B, T, S, C, M) -> (B, n, M) draws along the structured axis (S or T).
    draws = y_pred[:, 0, :, 0, :] if system.axis == "spatial" else y_pred[:, :, 0, 0, :]
    b, n, m = draws.shape
    pooled = draws.permute(0, 2, 1).reshape(b * m, n).double()  # (B*M, n)
    emp_var = pooled.var(dim=0)
    assert torch.allclose(emp_var, system.marginal_var, atol=0.05), (
        f"{system_name}: emp var {emp_var[:4]} vs oracle {system.marginal_var[:4]}"
    )


# ---------------------------------------------------------------------------
# Column 3 oracle: member-pooled temporal autocorrelation recovers phi**k (OU).
# ---------------------------------------------------------------------------
def test_calibrated_temporal_autocorr_recovers_phi_k(ou):
    y_pred, _ = ou.draw("calibrated", batch=B_ORACLE, members=M_ORACLE, seed=SEED)
    phi = float(ou.cov[0, 1] / ou.cov[0, 0])  # AR(1) coefficient from the oracle cov
    x = rearrange(y_pred[:, :, 0, 0, :], "b t m -> b m t").double()
    xc = x - x.mean(dim=(1, 2), keepdim=True)
    den = (xc * xc).sum(dim=(1, 2))
    for k in (1, 2, 3):
        num = (xc[..., :-k] * xc[..., k:]).sum(dim=(1, 2))
        acf_k = float((num / den).mean().item())
        assert acf_k == pytest.approx(phi**k, abs=0.05), (
            f"lag {k}: pooled autocorr {acf_k:.3f} vs phi**k {phi**k:.3f}"
        )


# ---------------------------------------------------------------------------
# Variogram oracle: the ensemble pairwise variogram E|X_i - X_j|**p matches the
# Gaussian closed form for representative pairs (validates the quantity the
# VariogramScore is built on against the oracle covariance).
# ---------------------------------------------------------------------------
def test_calibrated_pairwise_variogram_matches_gaussian_oracle(ring):
    p = 0.5
    y_pred, _ = ring.draw("calibrated", batch=B_ORACLE, members=M_ORACLE, seed=SEED)
    draws = y_pred[:, 0, :, 0, :]  # (B, D, M)
    b, d, m = draws.shape
    pooled = draws.permute(0, 2, 1).reshape(b * m, d).double()  # (B*M, D)
    sigma = ring.cov  # (D, D) oracle covariance
    # E|Z|**p for Z ~ N(0, v) = v**(p/2) * 2**(p/2) * Gamma((p+1)/2) / sqrt(pi).
    const = 2.0 ** (p / 2.0) * math.gamma((p + 1) / 2.0) / math.sqrt(math.pi)
    for i, j in [(0, 1), (0, 2), (0, 8)]:
        emp = (pooled[:, i] - pooled[:, j]).abs().pow(p).mean().item()
        v_ij = float(sigma[i, i] + sigma[j, j] - 2 * sigma[i, j])
        oracle = const * v_ij ** (p / 2.0)
        assert emp == pytest.approx(oracle, abs=0.03), (
            f"pair ({i},{j}): empirical {emp:.3f} vs Gaussian oracle {oracle:.3f}"
        )


# ---------------------------------------------------------------------------
# Rank-histogram oracle: calibrated ensembles give (near-)uniform histograms.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [MultivariateRankHistogram, DependenceRankHistogram])
def test_calibrated_rank_histograms_are_uniform(cls, ring):
    torch.manual_seed(0)
    # Fewer members -> fewer bins -> the finite-sample Delta floor is small enough
    # to make "uniform" a meaningful bound; accumulate several batches for stability.
    m_hist = 24
    metric = cls(n_members=m_hist)
    for s in range(8):
        y_pred, y_true = ring.draw(
            "calibrated", batch=B_ORACLE, members=m_hist, seed=SEED + s
        )
        metric.update(y_pred, y_true)
    # Uniform: reliability index Delta near its finite-sample floor, rank centred.
    assert metric.reliability_index() < 0.15
    assert _mean_rank(metric) == pytest.approx(0.5, abs=0.05)
