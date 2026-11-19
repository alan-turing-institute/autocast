"""Column-exclusivity battery for the UQ assessment suite (plan sec. 4.2).

The suite's acceptance principle (plan sec. 2): *each column's metrics flag a
failure planted in their column and stay clean on failures planted in the other
two.* This module wires every suite metric against the closed-form planted
ensembles of :mod:`_planted_ensembles` (ring = spatial dependence, OU = temporal
dependence) and asserts, per pre-registered thresholds, which metrics flag and
which stay clean on each planted failure.

Methodology is validation-by-planted-perturbation, the field's standard
(Thorarinsdottir, Scheuerer & Heinz 2016; Allen, Ziegel & Ginsbourger 2024;
Pinson & Girard 2012). Thresholds are pre-registered from the CLEAN (calibrated)
null distribution over ``N_NULL`` seeds: a metric *flags* a failure when the
failure's mean statistic exceeds ``null_mean + K * null_std`` and stays *clean*
when it does not. ``K`` is set generously (Bonferroni over the ~40 assertions,
alpha = 0.05 / 40 => z ~ 3); failures are evaluated as the mean over ``N_FAIL``
seeds so the sampling error is far below the margin. The empirical matrix and the
provenance of every non-obvious choice (n_sites=32 for the dependence-pre-rank
direction; winkler-99 as the tail read; autocorr_match max_lag; variogram as a
magnitude-not-exclusive detector) live in
``notes/design/uq_evaluation_metrics/uq_assessment/results_phase4_2026-07-04.md``.

These tests are intentionally Monte Carlo (seeded, deterministic) and heavier than
a unit test -- they are the suite's proof, gated by Fable at A1.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from einops import rearrange

from autocast.metrics.collection_coverage import MultiCollectionCoverage
from autocast.metrics.coverage import Coverage
from autocast.metrics.ensemble import (
    CRPS,
    DawidSebastianiScore,
    EnergyScore,
    LogScore,
    SpatialVariogramScore,
    SpreadSkillRatio,
    TemporalAutocorrelationMatch,
    TemporalVariogramScore,
    WinklerScore,
)
from autocast.metrics.rank_histogram import (
    DependenceRankHistogram,
    MultivariateRankHistogram,
)

from ._planted_ensembles import (
    DEPENDENCE_FAILURES,
    MARGINAL_FAILURES,
    ou_system,
    ring_system,
)

# --- battery configuration (provenance: results_phase4 note) ----------------
N_SITES = 32  # dependence-pre-rank direction breaks at the generator default 16
M = 24  # ensemble members
B = 128  # batch (independent realisations) per draw
N_NULL = 20  # plan floor for the calibrated null band
N_FAIL = 10  # seeds per failure (mean is the asserted statistic)
AUTOCORR_MAX_LAG = 2  # default T-1 dilutes the low-lag temporal signal
K_FLAG = 3.0  # Bonferroni-safe flag margin in null sigmas
K_CLEAN = 3.0  # clean = within this many null sigmas above the null mean
DIR_HIGH = 0.70  # dep_rank normalised mean rank: under-correlated (whiten)
DIR_LOW = 0.35  # dep_rank normalised mean rank: over-correlated (double)

_BASE_SEED = 20_000


# --- statistic helpers ------------------------------------------------------
def _scalar(metric, y_pred, y_true) -> float:
    metric.reset()
    metric.update(y_pred, y_true)
    return float(metric.compute().item())


def _coverage_error(level: float, y_pred, y_true) -> float:
    cov = Coverage(coverage_level=level)
    cov.update(y_pred, y_true)
    return abs(float(cov.compute().item()) - level)


def _hist_delta(metric, y_pred, y_true) -> float:
    metric.reset()
    metric.update(y_pred, y_true)
    return metric.reliability_index()


def _hist_mean_rank(metric, y_pred, y_true) -> float:
    """Normalised mean observation rank in [0, 1]; 0.5 = flat/centred."""
    metric.reset()
    metric.update(y_pred, y_true)
    counts = metric.counts.cpu().numpy()
    ranks = np.arange(1, len(counts) + 1)
    return float((counts * ranks).sum() / counts.sum() / len(counts))


def _spatial_stats(y_pred, y_true) -> dict[str, float]:
    """Every spatial-system statistic in one pass over a single draw."""
    return {
        "cov_err_0.5": _coverage_error(0.5, y_pred, y_true),
        "cov_err_0.9": _coverage_error(0.9, y_pred, y_true),
        "cov_err_0.95": _coverage_error(0.95, y_pred, y_true),
        "cov_err_0.99": _coverage_error(0.99, y_pred, y_true),
        "ssr_dev": abs(_scalar(SpreadSkillRatio(), y_pred, y_true) - 1.0),
        "crps": _scalar(CRPS(), y_pred, y_true),
        "log_score": _scalar(LogScore(), y_pred, y_true),
        "winkler_99": _scalar(WinklerScore(alpha=0.01), y_pred, y_true),
        "variogram_spatial": _scalar(SpatialVariogramScore(), y_pred, y_true),
        "dss": _scalar(DawidSebastianiScore(vector_dims="spatial"), y_pred, y_true),
        "energy_spatial": _scalar(EnergyScore(vector_dims="spatial"), y_pred, y_true),
        "collcov_err": _scalar(
            MultiCollectionCoverage(coverage_level=0.90), y_pred, y_true
        ),
        "mv_rank_delta": _hist_delta(
            MultivariateRankHistogram(n_members=M), y_pred, y_true
        ),
        "dep_rank_delta": _hist_delta(
            DependenceRankHistogram(n_members=M), y_pred, y_true
        ),
        "dep_rank_meanrank": _hist_mean_rank(
            DependenceRankHistogram(n_members=M), y_pred, y_true
        ),
    }


def _temporal_stats(y_pred, y_true) -> dict[str, float]:
    """Every temporal-system statistic in one pass over a single draw.

    SSR reduces over the spatial axis; on the OU system S=1 it degenerates
    (unstable), so it is asserted on the ring only and omitted here.
    """
    return {
        "cov_err_0.5": _coverage_error(0.5, y_pred, y_true),
        "cov_err_0.9": _coverage_error(0.9, y_pred, y_true),
        "log_score": _scalar(LogScore(), y_pred, y_true),
        "winkler_99": _scalar(WinklerScore(alpha=0.01), y_pred, y_true),
        "variogram_temporal": _scalar(TemporalVariogramScore(), y_pred, y_true),
        "autocorr_match": _scalar(
            TemporalAutocorrelationMatch(max_lag=AUTOCORR_MAX_LAG), y_pred, y_true
        ),
    }


def _accumulate(system, failure, stat_fn, n_seeds: int) -> dict[str, np.ndarray]:
    """Per-seed statistic arrays for one (system, failure) over ``n_seeds`` seeds."""
    per_seed: dict[str, list[float]] = {}
    for s in range(n_seeds):
        y_pred, y_true = system.draw(failure, batch=B, members=M, seed=_BASE_SEED + s)
        for key, value in stat_fn(y_pred, y_true).items():
            per_seed.setdefault(key, []).append(value)
    return {k: np.asarray(v) for k, v in per_seed.items()}


class _Battery:
    """Null band + per-failure means for one system (computed once)."""

    def __init__(self, system, stat_fn):
        self.null = _accumulate(system, "calibrated", stat_fn, N_NULL)
        self.fail = {
            f: _accumulate(system, f, stat_fn, N_FAIL)
            for f in (*MARGINAL_FAILURES, *DEPENDENCE_FAILURES)
        }

    def null_band(self, metric: str) -> tuple[float, float]:
        arr = self.null[metric]
        return float(arr.mean()), float(arr.std())

    def value(self, metric: str, failure: str) -> float:
        return float(self.fail[failure][metric].mean())

    def flags(self, metric: str, failure: str, k: float = K_FLAG) -> bool:
        mean, std = self.null_band(metric)
        return self.value(metric, failure) > mean + k * std

    def clean(self, metric: str, failure: str, k: float = K_CLEAN) -> bool:
        mean, std = self.null_band(metric)
        return self.value(metric, failure) <= mean + k * std


@pytest.fixture(scope="module")
def ring_battery() -> _Battery:
    torch.manual_seed(0)  # deterministic rank-histogram tie-break jitter
    return _Battery(ring_system(n_sites=N_SITES, corr_len=2.5), _spatial_stats)


@pytest.fixture(scope="module")
def ou_battery() -> _Battery:
    torch.manual_seed(0)
    return _Battery(ou_system(), _temporal_stats)


# ---------------------------------------------------------------------------
# CORE column-exclusivity: marginal metrics stay CLEAN on dependence failures.
# (Per-site marginals are exact under whiten/double -- this is the assertion
# that carries the doctrine.)
# ---------------------------------------------------------------------------
# winkler_99 is included: it is the column-1 TAIL read (see the tail test below), and
# its staying clean on the dependence failures -- tail_t has matched variance AND
# matched correlation, whiten/double have matched per-site marginals -- is itself a
# doctrine-carrying cell (the tail detector must not fire on a pure dependence error).
_MARGINAL_METRICS = (
    "cov_err_0.5",
    "cov_err_0.9",
    "ssr_dev",
    "log_score",
    "crps",
    "winkler_99",
)


@pytest.mark.parametrize("metric", _MARGINAL_METRICS)
@pytest.mark.parametrize("failure", DEPENDENCE_FAILURES)
def test_marginal_metrics_clean_on_spatial_dependence(ring_battery, metric, failure):
    assert ring_battery.clean(metric, failure), (
        f"{metric} should stay clean on the spatial-dependence failure {failure!r}; "
        f"value={ring_battery.value(metric, failure):.4f} "
        f"null={ring_battery.null_band(metric)}"
    )


@pytest.mark.parametrize("metric", ["cov_err_0.5", "cov_err_0.9", "log_score"])
@pytest.mark.parametrize("failure", DEPENDENCE_FAILURES)
def test_marginal_metrics_clean_on_temporal_dependence(ou_battery, metric, failure):
    assert ou_battery.clean(metric, failure), (
        f"{metric} should stay clean on the temporal-dependence failure {failure!r}; "
        f"value={ou_battery.value(metric, failure):.4f} "
        f"null={ou_battery.null_band(metric)}"
    )


# ---------------------------------------------------------------------------
# Marginal metrics FLAG their own (marginal) failures.
# ---------------------------------------------------------------------------
def test_scale_failures_flagged_by_marginal_metrics(ring_battery):
    # An inflated ensemble over-covers the central interval; a deflated one
    # under-covers everywhere. SSR moves in both directions.
    assert ring_battery.flags("cov_err_0.5", "scale_1.5")
    assert ring_battery.flags("cov_err_0.9", "scale_0.67")
    assert ring_battery.flags("ssr_dev", "scale_1.5")
    assert ring_battery.flags("ssr_dev", "scale_0.67")
    assert ring_battery.flags("log_score", "scale_1.5")
    assert ring_battery.flags("log_score", "scale_0.67")


# ---------------------------------------------------------------------------
# Tail read (plan column 1, the 0.95/0.99 levels). Finding: coverage FREQUENCY
# does not separate the tail failure (matched-variance Student-t exceeds the
# Gaussian interval only slightly more OFTEN); winkler-99 (excursion MAGNITUDE)
# is the tail detector, while central coverage and the dependence columns stay
# clean.
# ---------------------------------------------------------------------------
def test_tail_failure_flagged_by_winkler_not_central_coverage(ring_battery):
    # winkler-99 is heavy-tailed under tail_t; assert on the 20/10-seed mean with
    # a slightly relaxed margin (documented) -- it is still many mean-sigmas clear.
    assert ring_battery.flags("winkler_99", "tail_t", k=2.5), (
        f"winkler_99 tail_t={ring_battery.value('winkler_99', 'tail_t'):.3f} "
        f"null={ring_battery.null_band('winkler_99')}"
    )
    # Central coverage (<=0.9) stays clean on the tail failure.
    assert ring_battery.clean("cov_err_0.5", "tail_t")
    assert ring_battery.clean("cov_err_0.9", "tail_t")
    # Documented weakness: the tail-FREQUENCY read does not flag it.
    assert ring_battery.clean("cov_err_0.95", "tail_t")
    assert ring_battery.clean("cov_err_0.99", "tail_t")


def test_dependence_metrics_clean_on_tail_failure(ring_battery, ou_battery):
    # tail_t has matched variance AND matched correlation -> the dependence
    # columns must not fire.
    assert ring_battery.clean("dss", "tail_t")
    assert ring_battery.clean("dep_rank_delta", "tail_t")
    assert ou_battery.clean("autocorr_match", "tail_t")


# ---------------------------------------------------------------------------
# Column 2 (spatial dependence): dependence metrics FLAG whiten & double.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("failure", DEPENDENCE_FAILURES)
def test_dss_flags_spatial_dependence(ring_battery, failure):
    assert ring_battery.flags("dss", failure), (
        f"dss should flag {failure!r}; "
        f"value={ring_battery.value('dss', failure):.3f} "
        f"null={ring_battery.null_band('dss')}"
    )


@pytest.mark.parametrize("failure", DEPENDENCE_FAILURES)
def test_dependence_prerank_flags_spatial_dependence(ring_battery, failure):
    assert ring_battery.flags("dep_rank_delta", failure)


def test_collection_coverage_flags_under_correlation(ring_battery):
    # Directional: regional sums of a whitened ensemble are too narrow -> flags
    # whiten strongly. Over-correlation (double) slightly over-covers and is a
    # documented miss for this one-sided probe (not asserted as a flag).
    assert ring_battery.flags("collcov_err", "whiten")


def test_mv_rank_flags_whiten_via_joint_dispersion(ring_battery):
    # The average-rank construction catches whiten as a joint under-dispersion,
    # only weakly the correlation direction (doctrine: pair it with variogram /
    # dep_rank for direction).
    assert ring_battery.flags("mv_rank_delta", "whiten")


# ---------------------------------------------------------------------------
# The dependence pre-rank histogram gets the DIRECTION right (plan sec. 4.2's
# explicit requirement) AND is scale/tail-invariant.
# ---------------------------------------------------------------------------
def test_dependence_prerank_direction(ring_battery):
    # Under-correlated (whiten): truth is the smooth outlier -> HIGH ranks.
    assert ring_battery.value("dep_rank_meanrank", "whiten") > DIR_HIGH
    # Over-correlated (double): truth is the rough outlier -> LOW ranks.
    assert ring_battery.value("dep_rank_meanrank", "double") < DIR_LOW


@pytest.mark.parametrize("failure", MARGINAL_FAILURES)
def test_dependence_prerank_blind_to_marginal_failures(ring_battery, failure):
    # Designed scale/tail invariance: Delta pinned at the null and the histogram
    # stays centred for every marginal failure.
    assert ring_battery.clean("dep_rank_delta", failure)
    mean_rank = ring_battery.value("dep_rank_meanrank", failure)
    assert DIR_LOW < mean_rank < DIR_HIGH, (
        f"dep_rank must stay centred on the marginal failure {failure!r}; "
        f"mean_rank={mean_rank:.3f}"
    )


# ---------------------------------------------------------------------------
# Energy score is NOT the dependence judge -- its documented blind spot on the
# spatial correlation failure is itself an assertion (plan sec. 4.2 / sec. 9).
# ---------------------------------------------------------------------------
def test_energy_score_near_clean_on_whiten(ring_battery):
    assert ring_battery.clean("energy_spatial", "whiten"), (
        "energy score is expected to be a weak (near-clean) dependence detector; "
        f"value={ring_battery.value('energy_spatial', 'whiten'):.3f} "
        f"null={ring_battery.null_band('energy_spatial')}"
    )


# ---------------------------------------------------------------------------
# Column 3 (temporal dependence): autocorr_match FLAGS whiten & double and is
# clean on every marginal failure (scale/tail invariant).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("failure", DEPENDENCE_FAILURES)
def test_autocorr_match_flags_temporal_dependence(ou_battery, failure):
    assert ou_battery.flags("autocorr_match", failure), (
        f"autocorr_match should flag {failure!r}; "
        f"value={ou_battery.value('autocorr_match', failure):.4f} "
        f"null={ou_battery.null_band('autocorr_match')}"
    )


@pytest.mark.parametrize("failure", MARGINAL_FAILURES)
def test_autocorr_match_clean_on_marginal_failures(ou_battery, failure):
    assert ou_battery.clean("autocorr_match", failure)


# ---------------------------------------------------------------------------
# Documented non-exclusivity: the variogram is a MAGNITUDE detector (not
# scale-invariant) -- it flags marginal failures too, so it is only interpreted
# after the layer-0 marginals gate. Asserting this pins the known behaviour.
# ---------------------------------------------------------------------------
def test_variogram_spatial_is_a_magnitude_detector(ring_battery):
    # Flags the spatial dependence failure (whiten, strong) ...
    assert ring_battery.flags("variogram_spatial", "whiten")
    # ... but ALSO the marginal scale failure (it is not column-exclusive: it is
    # a magnitude detector, interpreted only after the layer-0 marginals gate).
    assert ring_battery.flags("variogram_spatial", "scale_1.5")


# ---------------------------------------------------------------------------
# Temporal DIRECTION is not provided by any shipped metric (autocorr_match is
# magnitude-only, dep_rank is spatial-only). This diagnostic proves the temporal
# failures ARE directionally distinct via a signed member-pooled lag-1
# autocorrelation, documenting the shipped-metric gap for Gate A1.
# ---------------------------------------------------------------------------
def _signed_pooled_acf1(system, failure: str) -> float:
    """Mean over seeds of (ensemble - truth) member-pooled lag-1 autocorr."""
    diffs = []
    for s in range(N_FAIL):
        y_pred, y_true = system.draw(failure, batch=B, members=M, seed=_BASE_SEED + s)
        ens = rearrange(y_pred[:, :, 0, 0, :], "b t m -> b m t").double()
        tru = y_true[:, :, 0, 0].unsqueeze(1).double()  # (B, 1, T)

        def acf1(x):  # x: (B, M, T) member-pooled biased lag-1 autocorr
            xc = x - x.mean(dim=(1, 2), keepdim=True)
            num = (xc[..., :-1] * xc[..., 1:]).sum(dim=(1, 2))
            den = (xc * xc).sum(dim=(1, 2))
            return num / den

        diffs.append(float((acf1(ens) - acf1(tru)).mean().item()))
    return float(np.mean(diffs))


def test_temporal_failures_are_directionally_distinguishable():
    ou = ou_system()
    calibrated = _signed_pooled_acf1(ou, "calibrated")
    whiten = _signed_pooled_acf1(ou, "whiten")
    double = _signed_pooled_acf1(ou, "double")
    # whiten under-correlates (ensemble acf below truth), double over-correlates,
    # both relative to the finite-T calibrated baseline.
    assert whiten < calibrated - 0.1
    assert double > calibrated + 0.1
