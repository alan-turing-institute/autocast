"""Score raw, EMOS, and conformal calibrated forecasts.

Which score comes from where
-----------------------------
- **Coverage and Winkler** (both raw/EMOS/conformal) are computed from
  precomputed ``(lower, upper)`` interval bands with :func:`band_coverage` /
  :func:`band_winkler`, two module-local functions that reimplement, formula
  for formula, ``autocast.metrics.coverage.Coverage._score`` and
  ``autocast.metrics.ensemble.WinklerScore._score``. This is deliberate, not
  a duplication of convenience: conformal calibration only ever produces
  bands, never ensemble members, so there is no member axis to feed the
  quantile-based classes with. Using the same band-based path for raw and
  EMOS too (rather than the classes for those and bands only for conformal)
  means all three methods run through one code path and are directly
  comparable. ``test_scoring.py`` proves the two are numerically identical:
  feeding the raw ensemble's own empirical quantile interval into
  :func:`band_coverage`/:func:`band_winkler` reproduces
  ``Coverage``/``WinklerScore`` exactly.
- Intervals come from each calibrator's ``.predict()`` (raw: empirical
  ensemble quantiles via :func:`raw_interval_multi`; EMOS/conformal: the
  ``autouq`` calibrator's own ``.predict()``), evaluated once on the full
  test set at the full :data:`LEVELS` grid, then sliced per window/frame --
  never refit per window.
- **CRPS, spread-skill ratio (SSR), ensemble spread, ensemble skill, NRMSE,
  VRMSE** need actual ensemble members, so they reuse
  ``autocast.metrics.ensemble.{CRPS,SpreadSkillRatio,EnsembleSpread,
  EnsembleSkill}`` and ``autocast.metrics.deterministic.{NRMSE,VRMSE}``
  directly. ``window_row`` feeds a fresh instance per window through
  ``.update()``/``.compute()`` (matching how the eval script itself computes
  ``rollout_metrics.csv``); the per-frame and bootstrap paths instead call
  ``.score()`` directly (bypassing the stateful batch-pooling accumulator),
  which is the per-trajectory, per-frame value both need and lets both be
  computed in one broadcast call with no Python loop over frames or draws.
  Conformal has no members, so none of this is ever computed for it.
- **Rank histograms and excess kurtosis** (raw/EMOS only) port
  ``field_autouq.py``'s ``rank_hist``/``excess_kurtosis`` math, vectorized
  across every frame/bootstrap draw at once (see below).
- **Dependence** (spatial-mean spread-skill of raw / EMOS-indep / EMOS+ECC)
  generalizes ``field_autouq.py``'s ``spatial_mean_ssr`` (there a single
  pooled scalar) to a per-channel vector.

Window-reconstruction note (why ``per_frame_ingredients.csv`` works)
---------------------------------------------------------------------
Every metric above (excluding the linear coverage/Winkler pair) reduces
spatial dimensions -- and, where present, the ensemble-member axis -- down to
a per-(batch, frame, channel) value *before* applying its nonlinearity
(square root or ratio); batch, frame, and channel are then pooled by a plain
arithmetic mean outside that nonlinearity (`autocast.metrics.base.BaseMetric
.compute`, `autocast.metrics.ensemble.BTSCMMetric.score`). Because the batch
count is identical at every lead time, a window spanning several frames is
therefore *exactly* the unweighted mean of the single-frame values -- never
the nonlinear function of pooled raw sums (e.g. ``sqrt(sum_sq_err /
sum_true_sq)`` over the whole window), which would NOT match. This was
verified empirically (not just derived) against the real metric classes
before committing to the design. ``per_frame_ingredients.csv`` therefore
stores, per frame, ``sum_<metric> = value_at_frame * n`` and ``n`` (the
trajectory count, constant across frames); a window's value is
``sum(sum_<metric> for frames in window) / sum(n for frames in window)``.
Coverage is linear (a proportion), so its ingredients are plain covered/total
counts per frame and channel. Its calibration error is not: as
`autocast.metrics.coverage.MultiCoverage`, the absolute error is taken per
frame, channel and level before averaging, so a window's value is rebuilt
frame by frame, never from counts summed over the window. The same
per-(batch, frame) linearity is
what makes the vectorized bootstrap below exact: resampling trajectories and
averaging their (batch, frame) values is identical to resampling then
recomputing the metric from scratch.

Vectorization and device notes
-------------------------------
No Python loop scales with frame count, pixel count, channel count, coverage
level count, or bootstrap draw count anywhere in this module -- every such
axis is a broadcast dimension in one tensor op (`per_frame_ingredients` and
`rank_histogram_per_frame` still loop over *blocks* of frames, sized by
``frame_block_size``, to bound peak memory when a per-level or per-member
broadcast would otherwise be large; each iteration still vectorizes many
frames at once, not one). Every score reduction is computed in float32 (the
input dtype); accumulation over O(10^7) elements -- the per-frame ingredient
sums, the excess-kurtosis moment sums -- is done in float64 to avoid
precision loss, per the same convention `torch.std`/`torch.var` already use
internally. Tensors are used on whatever device the caller already put them
on (this module never moves data between devices); random draws
(:func:`seeded_generator`) require a generator on that same device.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
import torch

from autocast.metrics.base import BaseMetric
from autocast.metrics.deterministic import NRMSE, VRMSE
from autocast.metrics.ensemble import (
    CRPS,
    EnsembleSkill,
    EnsembleSpread,
    SpreadSkillRatio,
)
from autocast.types import TensorBTSC, TensorBTSCM, TensorC

#: Headline miscoverage level: 90% central intervals.
NOMINAL_ALPHA = 0.1
NOMINAL_LEVEL = round(1.0 - NOMINAL_ALPHA, 2)

#: Reliability grid for `rollout_coverage_window_<w>.csv` and the "coverage"
#: (average absolute calibration error) column of `rollout_metrics.csv`.
LEVELS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))

#: Rollout windows: the paper's own (`encoder_processor_decoder.yaml`'s
#: `metric_windows_rollout`), sliced ``[start, end)`` as the eval does, plus two
#: extra windows that split its ``[31, 99)`` tail exactly (D9, D30).
WINDOWS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 4),
    (6, 12),
    (13, 30),
    (31, 99),
    (31, 65),
    (65, 99),
)

#: Default frame-block size for `per_frame_ingredients`/`rank_histogram_per_frame`:
#: bounds the peak size of the per-block `(B, block, H, W, C, len(LEVELS))`
#: coverage tensor (the one broadcast in this module large enough, at the
#: real H=W=64 scale, to be worth chunking -- see the module docstring).
DEFAULT_FRAME_BLOCK_SIZE = 20

_Z_STD_FLOOR = 1e-8
_SSR_SKILL_FLOOR = 1e-8
_EXCESS_KURTOSIS_NORMAL = 3.0


class Method(StrEnum):
    """Calibration method; values double as output subdirectory names."""

    RAW = "raw"
    EMOS = "EMOS"
    CONFORMAL = "conformal"


@dataclass(frozen=True)
class FittedMethod:
    """One method's fitted intervals (+ members, if any) on the full test set.

    Attributes
    ----------
    method
        Which method this is.
    lower, upper
        Central interval bounds at every level in :data:`LEVELS`, shape
        ``(B, T, H, W, C, len(LEVELS))``.
    samples
        Ensemble members for sample-based scores, shape ``(B, T, H, W, C, M)``,
        or ``None`` for conformal (bands only, no members).
    """

    method: Method
    lower: TensorBTSCM
    upper: TensorBTSCM
    samples: TensorBTSCM | None


def seeded_generator(seed: int, device: torch.device | str) -> torch.Generator:
    """Build a :class:`torch.Generator` seeded for reproducible draws on ``device``.

    CUDA random ops (``torch.randn``/``torch.randint`` with a ``generator``)
    require the generator's device to match the output tensor's device, so
    every stochastic draw in this package goes through this one helper
    rather than the bare ``torch.Generator()`` (CPU-only) constructor.
    """
    return torch.Generator(device=device).manual_seed(seed)


def _linear_order_statistic(sorted_pred: torch.Tensor, quantile: float) -> torch.Tensor:
    """One quantile from an already-sorted ``(..., M)`` tensor's last axis.

    Reimplements ``torch.quantile``'s default ``interpolation="linear"``
    formula by hand (linear interpolation between the two bracketing order
    statistics) rather than calling ``torch.quantile`` -- see
    :func:`raw_interval_multi`'s docstring for why. ``quantile`` is a plain
    Python float (not a tensor), so every operation here is a cheap
    ``(..., 1)``-sized slice/lerp, independent of how many quantiles the
    caller needs in total.
    """
    n_members = sorted_pred.shape[-1]
    position = quantile * (n_members - 1)
    lower_index = min(int(position), n_members - 1)
    fraction = position - lower_index
    lower_value = sorted_pred[..., lower_index]
    if fraction == 0.0 or lower_index >= n_members - 1:
        return lower_value
    upper_value = sorted_pred[..., lower_index + 1]
    return lower_value + (upper_value - lower_value) * fraction


def raw_interval_multi(
    pred: TensorBTSCM, levels: Sequence[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Empirical ensemble-quantile interval at multiple coverage levels at once.

    Sorts ``pred`` along the member axis exactly once and reads every
    requested quantile off that one sorted copy (:func:`_linear_order_statistic`)
    instead of calling ``torch.quantile(pred, quantiles, dim=-1)`` with a
    batched ``quantiles`` tensor. That batched call is *not* the O(input-size)
    operation it looks like: measured directly (this machine, CUDA, a
    ``(10, 100, 64, 64, 3, 10)`` input), passing a 38-element ``q`` tensor
    peaks at ~14.8 GB -- ~32x the ~0.46 GB input -- and peak memory scales
    linearly in the *number* of quantiles requested, not just the input size
    (1 quantile: ~1.8 GB; 38 quantiles: ~14.8 GB), i.e. PyTorch's CUDA
    multi-quantile kernel does not share one sort across quantiles the way
    this function does. The one-sort-many-reads version above measured ~5.4
    GB for the same 38-quantile call (bit-identical results, float32
    round-off only) -- this was the single largest contributor to this
    package's excess GPU memory (:func:`fit_raw` calls this with the full
    19-level :data:`LEVELS` grid, i.e. 38 quantiles, as the very first step
    of every calibration combination).

    Parameters
    ----------
    pred
        Ensemble forecast, shape ``(..., M)``.
    levels
        Central coverage levels (e.g. 0.9 for a 90% interval).

    Returns
    -------
    tuple of Tensor
        ``(lower, upper)``, each shape ``(..., len(levels))``.
    """
    alphas = [1.0 - level for level in levels]
    sorted_pred, _ = torch.sort(pred, dim=-1)
    out_shape = (*pred.shape[:-1], len(levels))
    lower = torch.empty(out_shape, dtype=pred.dtype, device=pred.device)
    upper = torch.empty(out_shape, dtype=pred.dtype, device=pred.device)
    for index, alpha in enumerate(alphas):
        lower[..., index] = _linear_order_statistic(sorted_pred, alpha / 2)
        upper[..., index] = _linear_order_statistic(sorted_pred, 1 - alpha / 2)
    return lower, upper


def band_coverage(true: TensorBTSC, lower: TensorBTSC, upper: TensorBTSC) -> TensorBTSC:
    """Pointwise coverage indicator from a precomputed interval.

    Identical formula to ``autocast.metrics.coverage.Coverage._score``.
    """
    return ((true >= lower) & (true <= upper)).float()


def band_coverage_multi(
    true: TensorBTSC, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    """Pointwise coverage indicator at every level in one broadcast.

    ``lower``/``upper`` carry a trailing levels axis (shape matching
    ``true`` everywhere else, plus ``(..., n_levels)``); returns that same
    shape. This is what lets :func:`coverage_calibration_error`,
    :func:`coverage_reliability_table`, and :func:`per_frame_ingredients`
    score every level in one call instead of looping over `LEVELS`.
    """
    return ((true.unsqueeze(-1) >= lower) & (true.unsqueeze(-1) <= upper)).float()


def band_winkler(
    true: TensorBTSC, lower: TensorBTSC, upper: TensorBTSC, alpha: float
) -> TensorBTSC:
    """Pointwise Winkler interval score from a precomputed interval.

    Identical formula to ``autocast.metrics.ensemble.WinklerScore._score``.
    """
    width = upper - lower
    below_penalty = (2.0 / alpha) * torch.clamp(lower - true, min=0.0)
    above_penalty = (2.0 / alpha) * torch.clamp(true - upper, min=0.0)
    return width + below_penalty + above_penalty


def per_lead_coverage(
    true: TensorBTSC, lower: TensorBTSC, upper: TensorBTSC
) -> np.ndarray:
    """Coverage at each lead time, pooled over batch, spatial dims, and channel.

    Ported from ``field_autouq.py::cov_per_lead``; used by the sufficiency
    sweep, which needs a per-lead curve rather than the single scalar
    :func:`coverage_calibration_error` produces.
    """
    return band_coverage(true, lower, upper).mean(dim=(0, 2, 3, 4)).cpu().numpy()


def per_lead_winkler(
    true: TensorBTSC, lower: TensorBTSC, upper: TensorBTSC, alpha: float
) -> np.ndarray:
    """Winkler score at each lead time, pooled over batch, spatial dims, and channel.

    Ported from ``field_autouq.py::winkler_per_lead``.
    """
    return band_winkler(true, lower, upper, alpha).mean(dim=(0, 2, 3, 4)).cpu().numpy()


def coverage_map_slice(
    true: TensorBTSC, lower: TensorBTSC, upper: TensorBTSC
) -> torch.Tensor:
    """Per-pixel, per-channel coverage at one level, pooled over batch and time.

    Deliberately takes single-level ``(lower, upper)`` slices (not a
    :class:`FittedMethod`'s full :data:`LEVELS`-grid tensor): callers should
    extract this small ``(H, W, ..., C)`` result and drop the much larger
    grid tensor immediately afterward -- see
    :func:`autocast.scripts.conformal.calibrate.run_combination`'s docstring
    for why that ordering is load-bearing for peak memory.
    """
    return band_coverage(true, lower, upper).mean(dim=(0, 1)).cpu()


def coverage_map_by_window(
    true: TensorBTSC, lower: TensorBTSC, upper: TensorBTSC
) -> dict[tuple[int, int], torch.Tensor]:
    """:func:`coverage_map_slice` within each of :func:`windows_within`'s windows.

    Each value is ``(H, W, ..., C)``, pooled over the batch and that window's
    frames only, so the maps show how the spatial pattern of coverage changes
    with lead time.
    """
    return {
        window: coverage_map_slice(
            _window_slice(true, window),
            _window_slice(lower, window),
            _window_slice(upper, window),
        )
        for window in windows_within(true.shape[1])
    }


def coverage_calibration_error(
    true: TensorBTSC, lower: TensorBTSC, upper: TensorBTSC, levels: Sequence[float]
) -> float:
    """Average absolute calibration error across frames, channels and ``levels``.

    As `autocast.metrics.coverage.MultiCoverage`: coverage is pooled over
    batch and space only, the absolute error is taken per frame, channel and
    level, then averaged -- i.e. the mean over the slice's frames of
    :func:`per_frame_coverage_calibration_error`. Pooling the frames before
    the absolute error would understate any window whose per-frame coverage
    crosses or scatters around nominal.

    Parameters
    ----------
    true
        Truth slice, shape ``(B, T, H, W, C)``.
    lower, upper
        Interval bounds at every level, shape ``(B, T, H, W, C, len(levels))``.
    levels
        Coverage levels matching the trailing axis of ``lower``/``upper``.
    """
    observed = per_frame_observed_coverage(true, lower, upper)
    return float(per_frame_coverage_calibration_error(observed, levels).mean())


def coverage_reliability_table(
    true: TensorBTSC, lower: TensorBTSC, upper: TensorBTSC, levels: Sequence[float]
) -> tuple[list[float], list[float], np.ndarray]:
    """Reliability table matching ``rollout_coverage_window_<w>.csv``'s columns.

    Computed for every level in one broadcast via :func:`band_coverage_multi`
    -- no Python loop over ``levels``.

    Returns
    -------
    tuple
        ``(levels, observed_mean_per_level, observed_per_channel)`` where
        ``observed_per_channel`` has shape ``(len(levels), C)``.
    """
    covered = band_coverage_multi(true, lower, upper)  # (..., C, len(levels))
    reduce_dims = tuple(
        range(true.ndim - 1)
    )  # batch, time, spatial -- not channel/levels
    per_channel = covered.mean(dim=reduce_dims)  # (C, len(levels))
    per_channel_np = per_channel.movedim(-1, 0).cpu().numpy()  # (len(levels), C)
    observed_means = per_channel_np.mean(axis=1).tolist()
    return list(levels), observed_means, per_channel_np


def windows_within(n_frames: int) -> list[tuple[int, int]]:
    """`WINDOWS` clipped to `[0, n_frames)`, dropping any that become empty.

    Production dumps always have T=100 frames, so every configured window
    fits. Test dumps are often shorter; mirrors
    ``autocast.metrics.trajectory.TrajectoryMetricAccumulator.update``'s own
    windowing convention (clip ``end``, skip when ``start >= end``) rather
    than silently scoring an empty slice.
    """
    clipped = ((start, min(end, n_frames)) for start, end in WINDOWS)
    return [(start, end) for start, end in clipped if start < end]


def _window_slice(tensor: torch.Tensor, window: tuple[int, int]) -> torch.Tensor:
    start, end = window
    end = min(end, tensor.shape[1])
    return tensor[:, start:end]


def _member_metric_factories() -> list[tuple[str, Callable[[], BaseMetric]]]:
    """Factories for the member-based metrics, in column order.

    Factories (not instances): every caller needs a *fresh* metric each time
    it scores a different window/frame slice (a `torchmetrics.Metric`
    accumulates state across `.update()` calls).
    """
    return [
        ("nrmse", NRMSE),
        ("vrmse", VRMSE),
        ("crps", CRPS),
        ("ssr", SpreadSkillRatio),
        ("spread", EnsembleSpread),
        ("skill", EnsembleSkill),
    ]


def _per_trajectory_frame_metric(
    factory: Callable[[], BaseMetric], samples: TensorBTSCM, true: TensorBTSC
) -> torch.Tensor:
    """``(B, T)`` per-trajectory, per-frame, channel-averaged metric value.

    Calls the metric's ``.score()`` directly rather than
    ``.update()``/``.compute()``: ``.score()`` is pure (no accumulator state)
    and returns ``(B, T, C)`` *without* pooling over batch, which is exactly
    the per-trajectory granularity :func:`bootstrap_summary` and
    :func:`per_frame_ingredients` need -- computed in one call, independent
    of ``T``.
    """
    metric = factory()
    raw = metric.score(samples, true)  # (B, T, C)
    return raw.double().mean(dim=-1)  # (B, T), float64


def per_frame_member_metric_values(
    fitted: FittedMethod, true: TensorBTSC
) -> dict[str, np.ndarray]:
    """``{name: (T,) array}`` for every member-based metric, one call each.

    Empty for conformal (``fitted.samples is None``). Used by
    ``rollout_metrics_per_timestep_channel_all.csv``; no Python loop over
    frames.
    """
    if fitted.samples is None:
        return {}
    return {
        name: _per_trajectory_frame_metric(factory, fitted.samples, true)
        .mean(dim=0)
        .cpu()
        .numpy()
        for name, factory in _member_metric_factories()
    }


def per_frame_observed_coverage(
    true: TensorBTSC,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    frame_block_size: int = DEFAULT_FRAME_BLOCK_SIZE,
) -> np.ndarray:
    """Observed coverage per frame, channel and level, pooled over batch and space.

    No loop over levels. When run on the full ``T``-frame test set -- like
    :func:`per_frame_ingredients` -- it processes ``frame_block_size`` frames
    at a time via :func:`band_coverage_multi` rather than broadcasting the
    full ``(B, T, H, W, C, n_levels)`` tensor at once; measured
    empirically (see this package's memory investigation) to be one of this
    module's largest peak-memory contributors when left unblocked at the
    real H=W=64 scale, since it duplicates a tensor the same size as the
    already-held ``fitted.lower``/``fitted.upper``.

    Returns
    -------
    numpy.ndarray
        Shape ``(T, C, n_levels)``, float64; ``n_levels`` is the trailing axis
        of ``lower``/``upper``.
    """
    n_frames = true.shape[1]
    observed_blocks: list[torch.Tensor] = []
    for start in range(0, n_frames, frame_block_size):
        end = min(start + frame_block_size, n_frames)
        block = (
            band_coverage_multi(
                true[:, start:end], lower[:, start:end], upper[:, start:end]
            )
            .double()
            .mean(dim=(0, 2, 3))
        )  # (block, C, len(levels))
        observed_blocks.append(block)
    return torch.cat(observed_blocks, dim=0).cpu().numpy()


def per_frame_coverage_calibration_error(
    observed: np.ndarray, levels: Sequence[float]
) -> np.ndarray:
    """Average absolute calibration error across channels and ``levels``, per frame.

    Parameters
    ----------
    observed
        Observed coverage, shape ``(T, C, len(levels))``: from
        :func:`per_frame_observed_coverage` or
        :func:`observed_coverage_from_ingredients`.
    levels
        Coverage levels matching the trailing axis of ``observed``.

    Returns
    -------
    numpy.ndarray
        Shape ``(T,)``. :func:`coverage_calibration_error` is its mean over a
        window's frames.
    """
    return np.abs(observed - np.asarray(levels)).mean(axis=(1, 2))


def per_frame_excess_kurtosis(true: TensorBTSC, samples: TensorBTSCM) -> np.ndarray:
    """Excess kurtosis of standardized residuals, at every frame, in one broadcast.

    The per-frame, vectorized form of ``field_autouq.py``'s
    ``excess_kurtosis`` of standardized residuals, kept as a single tensor
    reduction over ``(batch, spatial, channel)`` at every
    frame rather than a Python loop. Does not filter non-finite residuals
    (the ``_Z_STD_FLOOR`` clamp keeps them finite in practice) -- see
    :func:`_z_score_moment_sums`'s docstring for the same caveat.
    """
    mean = samples.mean(dim=-1)
    std = samples.std(dim=-1).clamp_min(_Z_STD_FLOOR)
    z = ((true - mean) / std).double()  # (B, T, H, W, C)
    reduce_dims = (0, 2, 3, 4)
    z_mean = z.mean(dim=reduce_dims, keepdim=True)
    fourth_central = ((z - z_mean) ** 4).mean(dim=reduce_dims)
    variance = z.var(dim=reduce_dims, unbiased=False)
    return (fourth_central / variance**2 - _EXCESS_KURTOSIS_NORMAL).cpu().numpy()


def window_row(
    fitted: FittedMethod, true: TensorBTSC, window: tuple[int, int]
) -> dict[str, float]:
    """One ``rollout_metrics.csv`` row's metric columns for one window.

    Reuses the repo's own metric classes (fresh instance per window) for the
    member-based scores, and :func:`band_winkler`/:func:`coverage_calibration_error`
    for winkler/coverage. Conformal (``fitted.samples is None``) only gets
    winkler and coverage.
    """
    true_w = _window_slice(true, window)
    lower_w = _window_slice(fitted.lower, window)
    upper_w = _window_slice(fitted.upper, window)
    level_index = LEVELS.index(NOMINAL_LEVEL)

    row: dict[str, float] = {}
    if fitted.samples is not None:
        samples_w = _window_slice(fitted.samples, window)
        for name, factory in _member_metric_factories():
            metric = factory()
            metric.update(samples_w, true_w)
            row[name] = float(metric.compute())

    row["winkler"] = float(
        band_winkler(
            true_w, lower_w[..., level_index], upper_w[..., level_index], NOMINAL_ALPHA
        ).mean()
    )
    row["coverage"] = coverage_calibration_error(true_w, lower_w, upper_w, LEVELS)
    return row


def per_frame_ingredients(
    fitted: FittedMethod,
    true: TensorBTSC,
    *,
    frame_block_size: int = DEFAULT_FRAME_BLOCK_SIZE,
) -> pd.DataFrame:
    """Per-frame additive ingredients so any frame window can be rebuilt exactly.

    See the module docstring's "Window-reconstruction note" for why a plain
    per-frame value (times the trajectory count) is the correct additive unit
    for this codebase's metrics, and why coverage's ingredient is a plain
    covered/total count.

    Every member-based metric and Winkler are computed for all ``T`` frames
    in one call each (no Python loop over frames). Coverage is computed
    ``frame_block_size`` frames at a time -- the one broadcast here
    (``(B, block, H, W, C, len(LEVELS))``) that is worth bounding in memory
    at the real H=W=64 scale; each block still covers many frames per
    iteration, not one. All sums are accumulated in float64.

    Returns
    -------
    pandas.DataFrame
        One row per frame, columns ``frame``, ``n``, ``sum_<metric>`` for
        each metric available to ``fitted`` (all seven for raw/EMOS; only
        ``sum_winkler`` for conformal), ``coverage_total_count`` and
        ``covered_<level>`` for each level in :data:`LEVELS` (pooled over
        channels), and ``coverage_count_per_channel`` and
        ``covered_<level>_channel_<c>`` (per channel, which the coverage
        calibration error needs).
    """
    n_frames = true.shape[1]
    n_trajectories = true.shape[0]
    level_index = LEVELS.index(NOMINAL_LEVEL)

    data: dict[str, np.ndarray] = {
        "frame": np.arange(n_frames),
        "n": np.full(n_frames, n_trajectories),
    }

    if fitted.samples is not None:
        for name, factory in _member_metric_factories():
            per_traj_frame = _per_trajectory_frame_metric(factory, fitted.samples, true)
            data[f"sum_{name}"] = per_traj_frame.sum(dim=0).cpu().numpy()  # (T,)

    winkler_vals = band_winkler(
        true,
        fitted.lower[..., level_index],
        fitted.upper[..., level_index],
        NOMINAL_ALPHA,
    )  # (B, T, H, W, C)
    winkler_sum_per_frame = winkler_vals.double().mean(dim=(2, 3, 4)).sum(dim=0)  # (T,)
    data["sum_winkler"] = winkler_sum_per_frame.cpu().numpy()

    n_channels = true.shape[-1]
    data["coverage_total_count"] = np.full(n_frames, int(true[:, 0].numel()))
    data["coverage_count_per_channel"] = np.full(
        n_frames, int(true[:, 0].numel()) // n_channels
    )
    covered_blocks: list[torch.Tensor] = []
    for start in range(0, n_frames, frame_block_size):
        end = min(start + frame_block_size, n_frames)
        covered = band_coverage_multi(
            true[:, start:end], fitted.lower[:, start:end], fitted.upper[:, start:end]
        )  # (B, block, H, W, C, len(LEVELS))
        covered_blocks.append(
            covered.double().sum(dim=(0, 2, 3))
        )  # (block, C, len(LEVELS))
    covered_by_frame = torch.cat(covered_blocks, dim=0).cpu().numpy()  # (T, C, L)
    for i, level in enumerate(LEVELS):
        data[f"covered_{level:.2f}"] = covered_by_frame[:, :, i].sum(axis=1)
    for i, level in enumerate(LEVELS):
        for channel in range(n_channels):
            data[f"covered_{level:.2f}_channel_{channel}"] = covered_by_frame[
                :, channel, i
            ]

    return pd.DataFrame(data)


def reconstruct_window_metric(
    ingredients: pd.DataFrame, window: tuple[int, int], metric: str
) -> float:
    """Rebuild one ``sum_<metric>``-backed window value from the ingredients."""
    start, end = window
    subset = ingredients[(ingredients["frame"] >= start) & (ingredients["frame"] < end)]
    return float(subset[f"sum_{metric}"].sum() / subset["n"].sum())


def reconstruct_window_coverage(
    ingredients: pd.DataFrame,
    window: tuple[int, int],
    levels: Sequence[float] = LEVELS,
) -> float:
    """Rebuild a window's coverage calibration error from the ingredients.

    Per frame and channel, as :func:`coverage_calibration_error`: each
    frame's covered count becomes a coverage fraction before the absolute
    error is taken, so the counts are never pooled over the window's frames.
    """
    start, end = window
    subset = ingredients.loc[
        (ingredients["frame"] >= start) & (ingredients["frame"] < end)
    ]
    observed = observed_coverage_from_ingredients(subset, levels)
    return float(per_frame_coverage_calibration_error(observed, levels).mean())


def observed_coverage_from_ingredients(
    ingredients: pd.DataFrame, levels: Sequence[float] = LEVELS
) -> np.ndarray:
    """Observed coverage per frame, channel and level from the ingredient counts.

    The same array :func:`per_frame_observed_coverage` computes from the
    bands, for the frames (rows) in ``ingredients``.

    Returns
    -------
    numpy.ndarray
        Shape ``(frames, C, len(levels))``, float64.
    """
    prefix = f"covered_{levels[0]:.2f}_channel_"
    n_channels = sum(column.startswith(prefix) for column in ingredients.columns)
    covered = np.stack(
        [
            ingredients[[f"covered_{level:.2f}_channel_{c}" for c in range(n_channels)]]
            for level in levels
        ],
        axis=-1,
    )  # (frames, C, len(levels))
    counts = np.asarray(ingredients["coverage_count_per_channel"], dtype=np.float64)
    return covered / counts[:, None, None]


def _pooled_excess_kurtosis(
    sum1: torch.Tensor,
    sum2: torch.Tensor,
    sum3: torch.Tensor,
    sum4: torch.Tensor,
    n: float,
) -> torch.Tensor:
    """Excess kurtosis of a pooled sample from its raw power sums.

    Raw power sums (``sum_k = sum(z**k)``) are additive across trajectories,
    so this reconstructs the *exact* pooled excess kurtosis of any resampled
    subset from per-trajectory sums via the standard moment-expansion
    identity for the 4th central moment, without re-touching per-pixel data.
    Works elementwise, so the same function scores the single point estimate
    and all bootstrap draws at once (``sum1``..``sum4`` may be scalars or a
    ``(n_boot,)`` batch).
    """
    mean = sum1 / n
    second = sum2 / n
    third = sum3 / n
    fourth = sum4 / n
    variance = second - mean**2
    fourth_central = (
        fourth - 4 * mean * third + 6 * mean**2 * second - 4 * mean**3 * mean + mean**4
    )
    return fourth_central / variance**2 - _EXCESS_KURTOSIS_NORMAL


def _z_score_moment_sums(
    true_frame: TensorBTSC, samples_frame: TensorBTSCM
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Per-trajectory raw power sums of standardized residuals at one frame.

    Enough to reconstruct the pooled excess kurtosis of ANY resampled subset
    of trajectories exactly (via :func:`_pooled_excess_kurtosis`), without
    re-touching per-pixel data. Like :func:`per_frame_excess_kurtosis`, it
    does not filter non-finite residuals (the ``_Z_STD_FLOOR`` clamp on
    ``std`` keeps ``z`` finite in practice).
    """
    mean = samples_frame.mean(dim=-1)
    std = samples_frame.std(dim=-1).clamp_min(_Z_STD_FLOOR)
    z = ((true_frame - mean) / std).double()  # (B, H, W, C)
    flat_dims = tuple(range(1, z.ndim))
    n_pixels = int(z[0].numel())
    return (
        z.sum(dim=flat_dims),
        (z**2).sum(dim=flat_dims),
        (z**3).sum(dim=flat_dims),
        (z**4).sum(dim=flat_dims),
        n_pixels,
    )


def bootstrap_summary(
    fitted: FittedMethod,
    true: TensorBTSC,
    *,
    n_boot: int = 200,
    seed: int,
) -> dict[str, float]:
    """Headline scores over the whole rollout, with bootstrap std over trajectories.

    Point estimate plus a bootstrap standard deviation from resampling test
    trajectories with replacement, holding the already-fitted calibrator
    fixed -- i.e. "if we'd happened to draw a different test set from the
    same population, how much would the reported number move" (ported from
    ``calibrate_full_cell.py::bootstrap_std``, generalized to also cover
    CRPS/NRMSE/VRMSE/SSR/excess kurtosis, not just coverage/Winkler).

    All ``n_boot`` draws are scored in one batched gather + reduction (a
    single ``[n_boot, B]`` index tensor drawn once, then advanced-indexed
    into the precomputed per-trajectory, per-frame values) -- no Python loop
    over draws. This is possible because every scored quantity here is
    linear in (trajectory, frame) once its own nonlinearity has been applied
    at the per-(trajectory, frame[, channel]) grain (see the module
    docstring's "Window-reconstruction note"): the mean of a resampled
    ``[B, T]`` (or the excess-kurtosis moment sums') subset is exactly what a
    from-scratch computation on that resampled subset would give.

    Parameters
    ----------
    fitted
        The method's fitted intervals (+ samples, if any) on the test set.
    true
        Test-set truth, shape ``(B, T, H, W, C)``.
    n_boot
        Bootstrap resamples.
    seed
        Generator seed; vary this across methods so their bootstrap draws
        are independent (as ``calibrate_full_cell.py`` does with
        ``seed + 1``, ``seed + 2``, ...).

    Returns
    -------
    dict
        ``coverage_90``, ``winkler_90``, and -- for raw/EMOS only --
        ``nrmse``, ``vrmse``, ``crps``, ``ssr``, ``exkurt_last_frame``, each
        paired with a ``<name>_boot_std`` key.
    """
    b_total = true.shape[0]
    device = true.device
    level_index = LEVELS.index(NOMINAL_LEVEL)
    lower_90 = fitted.lower[..., level_index]
    upper_90 = fitted.upper[..., level_index]

    per_traj_frame: dict[str, torch.Tensor] = {
        "coverage_90": band_coverage(true, lower_90, upper_90)
        .double()
        .mean(dim=(2, 3, 4)),
        "winkler_90": band_winkler(true, lower_90, upper_90, NOMINAL_ALPHA)
        .double()
        .mean(dim=(2, 3, 4)),
    }
    if fitted.samples is not None:
        for name, factory in _member_metric_factories():
            per_traj_frame[name] = _per_trajectory_frame_metric(
                factory, fitted.samples, true
            )

    generator = seeded_generator(seed, device)
    index = torch.randint(
        0, b_total, (n_boot, b_total), generator=generator, device=device
    )

    result: dict[str, float] = {}
    for key, values in per_traj_frame.items():
        result[key] = float(values.mean())
        draws = values[index].mean(dim=(1, 2))  # (n_boot,)
        result[f"{key}_boot_std"] = float(draws.std(unbiased=False))

    if fitted.samples is not None:
        sum1, sum2, sum3, sum4, n_pixels = _z_score_moment_sums(
            true[:, -1], fitted.samples[:, -1]
        )
        pooled_n = n_pixels * b_total
        point_kurt = _pooled_excess_kurtosis(
            sum1.sum(), sum2.sum(), sum3.sum(), sum4.sum(), pooled_n
        )
        draw_kurt = _pooled_excess_kurtosis(
            sum1[index].sum(dim=1),
            sum2[index].sum(dim=1),
            sum3[index].sum(dim=1),
            sum4[index].sum(dim=1),
            pooled_n,
        )
        result["exkurt_last_frame"] = float(point_kurt)
        result["exkurt_last_frame_boot_std"] = float(draw_kurt.std(unbiased=False))

    return result


def rank_histogram_per_frame(
    true: TensorBTSC,
    ensemble: TensorBTSCM,
    *,
    frame_block_size: int = DEFAULT_FRAME_BLOCK_SIZE,
) -> np.ndarray:
    """Talagrand rank histogram at every frame, in one linearized bincount.

    Each frame's rank is offset by ``frame_index * (M + 1)`` and everything
    is bin-counted in a single call, reshaped to ``(T, M + 1)`` -- no Python
    loop over frames, and (unlike a one-hot expansion) no extra ``M``-sized
    memory blowup. Frames are still processed in blocks of
    ``frame_block_size`` (each block one broadcast covering many frames) to
    bound peak memory at the real H=W=64 scale.
    """
    n_frames = true.shape[1]
    n_members = ensemble.shape[-1]
    n_bins = n_members + 1
    histograms: list[torch.Tensor] = []
    for start in range(0, n_frames, frame_block_size):
        end = min(start + frame_block_size, n_frames)
        block_frames = end - start
        ranks = (
            (ensemble[:, start:end] < true[:, start:end].unsqueeze(-1))
            .sum(dim=-1)
            .long()
        )  # (B, block, H, W, C)
        frame_offset = torch.arange(block_frames, device=ranks.device) * n_bins
        view_shape = (1, block_frames) + (1,) * (ranks.ndim - 2)
        combined = ranks + frame_offset.view(view_shape)
        counts = torch.bincount(combined.flatten(), minlength=block_frames * n_bins)
        histograms.append(counts.view(block_frames, n_bins))
    return torch.cat(histograms, dim=0).cpu().numpy()


def spatial_mean_spread_skill(ensemble: TensorBTSCM, true: TensorBTSC) -> TensorC:
    """Spread-skill ratio of the whole-field (spatial mean), per channel.

    Generalizes ``field_autouq.py::run_spatial``'s ``spatial_mean_ssr`` (there
    a single pooled scalar) to a per-channel vector, pooling over batch and
    time, in the same convention as the ``ssr`` column (variance and squared
    error averaged before the square roots). SSR near 1.0 means the
    whole-field mean's ensemble spread matches its actual error; independent
    per-site calibration collapses it towards 0 when the true field mean is
    nearly conserved, while ECC (restoring cross-site dependence) should
    recover it.

    Parameters
    ----------
    ensemble
        Ensemble forecast, shape ``(B, T, H, W, ..., C, M)``.
    true
        Truth, shape ``(B, T, H, W, ..., C)``.

    Returns
    -------
    TensorC
        Per-channel spread-skill ratio, shape ``(C,)``.
    """
    n_members = ensemble.shape[-1]
    spatial_dims = tuple(range(2, ensemble.ndim - 2))
    field_mean = ensemble.mean(dim=spatial_dims)  # (B, T, C, M)
    true_field_mean = true.mean(dim=tuple(range(2, true.ndim - 1)))  # (B, T, C)

    # Same convention as `autocast.metrics.ensemble.SpreadSkillRatio`: unbiased
    # member variance and squared error of the ensemble mean are averaged (here
    # over batch and time) before the square roots, then the sqrt((M+1)/M)
    # small-ensemble correction; a calibrated ensemble gives ~1.
    spread_var = field_mean.var(dim=-1, unbiased=True).mean(dim=(0, 1))  # (C,)
    skill_sq = (field_mean.mean(dim=-1) - true_field_mean).pow(2).mean(dim=(0, 1))
    correction = ((n_members + 1) / n_members) ** 0.5
    skill = skill_sq.sqrt().clamp_min(_SSR_SKILL_FLOOR)
    return correction * spread_var.sqrt() / skill
