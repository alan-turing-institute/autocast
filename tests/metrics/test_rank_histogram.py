"""Tests for the multivariate rank-histogram diagnostics (research piece sec. 6.6).

Average-rank and band-depth rank histograms map a multivariate ensemble forecast
to a single observation rank via a pre-rank function, then bin the rank. The
average-rank histogram reads like the familiar univariate one (U-shaped =
under-dispersed, dome = over-dispersed, uniform = calibrated); the band-depth
histogram instead piles up at high (central) ranks under over-dispersion.

Definitions follow Thorarinsdottir, Scheuerer & Heinz (2016), "Assessing the
calibration of high-dimensional ensemble forecasts using rank histograms",
J. Comput. Graph. Stat. 25(1).
"""

from typing import Any

import numpy as np
import pytest
import torch

from autocast.metrics.base import rows_per_pair_block
from autocast.metrics.rank_histogram import (
    AverageRankHistogram,
    BandDepthRankHistogram,
    DependenceRankHistogram,
    MultivariateRankHistogram,
    PooledMultivariateRankHistogram,
    TrajectoryBandDepthRankHistogram,
    _pool_ranks,
    _weighted_variogram,
    grid_inverse_distance_weights,
    ring_inverse_distance_weights,
)
from autocast.types import TensorBTSC, TensorBTSCM


def _multichannel_population(
    n: int, d: int, c: int, member_std, m: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """A (B=n, T=1, S=d, C=c[, M]) population: truth ~ N(0, I); members ~ N(0,
    member_std^2 I). ``member_std`` may be a per-channel sequence of length ``c``."""
    g = torch.Generator().manual_seed(seed)
    truth = torch.randn(n, 1, d, c, generator=g)
    std = torch.as_tensor(member_std, dtype=torch.float32).reshape(1, 1, 1, c, 1)
    members = std * torch.randn(n, 1, d, c, m, generator=g)
    return members, truth


def _gaussian_population(
    n: int, d: int, member_std: float, m: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-channel (C=1) population: truth ~ N(0, I_d); members ~ N(0,
    member_std^2 I_d). member_std=1 is calibrated, <1 under-dispersed, >1 over."""
    return _multichannel_population(n, d, 1, member_std, m, seed)


def _hist_shape(metric, members, truth) -> np.ndarray:
    metric.update(members, truth)
    return metric.compute_histogram()


def test_average_rank_calibrated_is_uniform():
    """A calibrated ensemble gives a flat average-rank histogram."""
    members, truth = _gaussian_population(4000, 8, member_std=1.0, m=20, seed=0)
    hist = _hist_shape(AverageRankHistogram(n_members=20), members, truth)
    assert hist.shape == (21,)  # M + 1 rank bins
    assert abs(hist.sum() - 1.0) < 1e-6
    # flat: max deviation from uniform is small
    uniform = 1.0 / 21
    assert np.max(np.abs(hist - uniform)) < 0.35 * uniform


def test_average_rank_underdispersed_is_u_shaped():
    """An under-dispersed ensemble piles mass in the end bins (U-shape)."""
    members, truth = _gaussian_population(4000, 8, member_std=0.5, m=20, seed=1)
    hist = _hist_shape(AverageRankHistogram(n_members=20), members, truth)
    ends = hist[0] + hist[-1]
    middle = hist[len(hist) // 2 - 2 : len(hist) // 2 + 3].mean()
    assert ends > 4.0 * middle  # heavy ends, light middle


def test_average_rank_overdispersed_is_dome():
    """An over-dispersed ensemble piles mass in the middle bins (inverted-U)."""
    members, truth = _gaussian_population(4000, 8, member_std=2.0, m=20, seed=2)
    hist = _hist_shape(AverageRankHistogram(n_members=20), members, truth)
    ends = hist[0] + hist[-1]
    middle = hist[len(hist) // 2 - 2 : len(hist) // 2 + 3].mean()
    assert middle > 3.0 * ends  # heavy middle, light ends


def test_band_depth_overdispersion_excess_high_ranks():
    """Band depth: over-dispersion pushes the observation to high (central) ranks."""
    m = 20
    members_o, truth_o = _gaussian_population(4000, 8, member_std=2.0, m=m, seed=3)
    members_c, truth_c = _gaussian_population(4000, 8, member_std=1.0, m=m, seed=4)
    hist_over = _hist_shape(BandDepthRankHistogram(n_members=m), members_o, truth_o)
    hist_cal = _hist_shape(BandDepthRankHistogram(n_members=m), members_c, truth_c)
    assert hist_over.shape == (m + 1,)
    top = m + 1 - 4  # the top few rank bins
    # over-dispersion accumulates excess mass in the top (most-central) ranks
    assert hist_over[top:].sum() > 1.6 * hist_cal[top:].sum()


def test_band_depth_underdispersion_excess_low_ranks():
    """Band depth: under-dispersion pushes the observation to low ranks."""
    m = 20
    members_u, truth_u = _gaussian_population(4000, 8, member_std=0.5, m=m, seed=5)
    members_c, truth_c = _gaussian_population(4000, 8, member_std=1.0, m=m, seed=6)
    hist_under = _hist_shape(BandDepthRankHistogram(n_members=m), members_u, truth_u)
    hist_cal = _hist_shape(BandDepthRankHistogram(n_members=m), members_c, truth_c)
    assert hist_under[:4].sum() > 1.6 * hist_cal[:4].sum()


def test_rank_histogram_compute_is_calibration_error():
    """compute() returns a non-negative scalar deviation-from-uniform score that is
    smaller for a calibrated ensemble than a miscalibrated one."""
    members_c, truth_c = _gaussian_population(4000, 8, member_std=1.0, m=20, seed=7)
    members_u, truth_u = _gaussian_population(4000, 8, member_std=0.4, m=20, seed=8)

    cal = AverageRankHistogram(n_members=20)
    cal.update(members_c, truth_c)
    bad = AverageRankHistogram(n_members=20)
    bad.update(members_u, truth_u)

    cal_err = float(cal.compute().item())
    bad_err = float(bad.compute().item())
    assert cal_err >= 0.0
    assert bad_err > cal_err


def test_init_rejects_too_few_members():
    """Fewer than two members leaves no rank structure; reject at construction."""
    with pytest.raises(ValueError, match="n_members >= 2"):
        AverageRankHistogram(n_members=1)


def test_update_rejects_non_tensor_input(validation_error):
    """Inputs must be tensors or numpy arrays, not arbitrary sequences."""
    metric = AverageRankHistogram(n_members=4)
    bad: Any = [1, 2, 3]  # deliberately wrong type to trip the runtime guard
    with validation_error(TypeError, match=r"Tensors or np\.ndarrays"):
        metric.update(bad, torch.randn(1, 1, 2, 1))


def test_update_rejects_low_rank_input(validation_error):
    """y_pred must carry the (B, T, S, C, M) axes; fewer than five dims is rejected."""
    metric = AverageRankHistogram(n_members=4)
    y_pred = torch.randn(2, 3, 4, 4)  # missing the ensemble axis
    y_true = torch.randn(2, 3, 4)
    with validation_error(ValueError, match="at least 5"):
        metric.update(y_pred, y_true)


def test_update_rejects_shape_mismatch():
    """y_pred must match y_true on every axis but the trailing ensemble axis."""
    metric = AverageRankHistogram(n_members=4)
    y_pred = torch.randn(2, 1, 8, 1, 4)
    y_true = torch.randn(2, 1, 6, 1)  # spatial size disagrees
    with pytest.raises(ValueError, match="match except for the ensemble axis"):
        metric.update(y_pred, y_true)


def test_update_rejects_wrong_member_count():
    """The ensemble axis must equal the declared n_members."""
    metric = AverageRankHistogram(n_members=4)
    y_pred = torch.randn(2, 1, 8, 1, 5)  # 5 members, but 4 were declared
    y_true = torch.randn(2, 1, 8, 1)
    with pytest.raises(ValueError, match="!= n_members"):
        metric.update(y_pred, y_true)


def test_compute_without_samples_raises():
    """Computing before any mass has accumulated is an explicit error, not a NaN."""
    metric = BandDepthRankHistogram(n_members=4)
    with pytest.raises(RuntimeError, match="No samples"):
        metric.compute_histogram()


# ---------------------------------------------------------------------------
# MultivariateRankHistogram (spatial-joint, per-channel; column 2 of the suite)
# ---------------------------------------------------------------------------


def test_mv_rank_calibrated_is_uniform():
    """A calibrated ensemble gives a flat multivariate rank histogram."""
    members, truth = _gaussian_population(4000, 8, member_std=1.0, m=20, seed=10)
    hist = _hist_shape(MultivariateRankHistogram(n_members=20), members, truth)
    assert hist.shape == (21,)
    assert abs(hist.sum() - 1.0) < 1e-6
    uniform = 1.0 / 21
    assert np.max(np.abs(hist - uniform)) < 0.35 * uniform


def test_mv_rank_underdispersed_is_u_shaped():
    """An under-dispersed ensemble piles mass in the end bins (U-shape)."""
    members, truth = _gaussian_population(4000, 8, member_std=0.5, m=20, seed=11)
    hist = _hist_shape(MultivariateRankHistogram(n_members=20), members, truth)
    ends = hist[0] + hist[-1]
    middle = hist[len(hist) // 2 - 2 : len(hist) // 2 + 3].mean()
    assert ends > 4.0 * middle


def test_mv_rank_matches_average_rank_for_single_channel():
    """For a single channel the per-channel spatial-joint MV-rank pools the field
    identically to the spatial+channel-joint AverageRankHistogram, so with the same
    tie-break RNG they produce the same histogram. (Average-rank pre-ranks collide
    often, so the randomized tie-break is load-bearing -- hence the matched seed.)"""
    members, truth = _gaussian_population(2000, 6, member_std=1.0, m=16, seed=12)
    torch.manual_seed(0)
    mv_metric = MultivariateRankHistogram(n_members=16)
    mv_metric.update(members, truth)
    torch.manual_seed(0)
    avg_metric = AverageRankHistogram(n_members=16)
    avg_metric.update(members, truth)
    assert np.allclose(mv_metric.compute_histogram(), avg_metric.compute_histogram())


def test_mv_rank_is_per_channel_not_channel_joint():
    """With two channels of different dispersion, ranking each channel's spatial
    field separately (MV-rank) differs from folding both channels into one joint
    vector (AverageRankHistogram) -- the two histograms disagree."""
    members, truth = _multichannel_population(
        4000, 8, c=2, member_std=[0.5, 2.0], m=20, seed=13
    )
    mv = _hist_shape(MultivariateRankHistogram(n_members=20), members, truth)
    avg = _hist_shape(AverageRankHistogram(n_members=20), members, truth)
    assert not np.allclose(mv, avg, atol=0.01)


def test_mv_rank_pools_opposing_channels_into_one_histogram():
    """Per-channel ranks are aggregated into a single shared histogram, so an
    under-dispersed channel (U-shape) and an over-dispersed channel (dome) partially
    cancel: the pooled two-channel histogram is flatter (smaller compute() deviation)
    than either channel scored alone."""
    m = 20
    both = _multichannel_population(4000, 8, c=2, member_std=[0.5, 2.0], m=m, seed=20)
    under = _multichannel_population(4000, 8, c=1, member_std=[0.5], m=m, seed=20)
    over = _multichannel_population(4000, 8, c=1, member_std=[2.0], m=m, seed=21)

    def deviation(pop):
        metric = MultivariateRankHistogram(n_members=m)
        metric.update(*pop)
        return float(metric.compute().item())

    pooled = deviation(both)
    assert pooled < deviation(under)
    assert pooled < deviation(over)


def test_mv_rank_accepts_integer_fields():
    """Integer-valued fields must not crash (regression for af3c299e): the working
    base casts the pool to float64 before ranking, so torch.mean never sees an
    integer dtype. Output is finite."""
    torch.manual_seed(0)
    y_pred = torch.randint(0, 5, (16, 1, 32, 1, 8))
    y_true = torch.randint(0, 5, (16, 1, 32, 1))
    metric = MultivariateRankHistogram(n_members=8)
    metric.update(y_pred, y_true)
    assert np.isfinite(metric.compute_histogram()).all()


def test_mv_rank_rejects_non_spatial_vector_dims():
    """Only vector_dims='spatial' is implemented; other values fail at construction."""
    with pytest.raises(NotImplementedError, match="spatial"):
        MultivariateRankHistogram(n_members=8, vector_dims="temporal")


# ---------------------------------------------------------------------------
# DependenceRankHistogram -- the variogram (correlation-structure) pre-rank.
# Its verification uses spatially *correlated* Gaussian fields (unlike the iid
# populations above): a smooth truth against whitened members reads as an
# under-correlated ensemble, and so on.
# ---------------------------------------------------------------------------


def _corr_matrix(d: int, ell: float) -> torch.Tensor:
    """Exponential correlation matrix on a length-``d`` line: Sigma_ij =
    exp(-|i-j|/ell).

    Larger ``ell`` = longer correlation length = smoother fields. Unit diagonal, so
    every site is marginally N(0, 1) whatever ``ell`` is -- correlation varies, the
    per-site marginal does not.
    """
    idx = torch.arange(d, dtype=torch.float64)
    dist = (idx[:, None] - idx[None, :]).abs()
    return torch.exp(-dist / ell)


def _draw_correlated(
    lead: tuple[int, ...], d: int, ell: float | None, g: torch.Generator
) -> torch.Tensor:
    """Draw fields ~ N(0, Sigma(ell)), shape ``(*lead, d)``; ``ell=None`` => white."""
    z = torch.randn(*lead, d, generator=g, dtype=torch.float64)
    if ell is None:
        return z
    sigma = _corr_matrix(d, ell) + 1e-8 * torch.eye(d, dtype=torch.float64)
    chol = torch.linalg.cholesky(sigma)
    return z @ chol.transpose(-1, -2)  # cov(row) = chol chol^T = Sigma


def _correlated_population(
    n: int,
    d: int,
    truth_ell: float | None,
    member_ell: float | None,
    m: int,
    seed: int,
    member_scale: float = 1.0,
    member_bias: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(B=n, T=1, S=d, C=1[, M]) population of Gaussian fields with a prescribed spatial
    correlation length. Truth ~ N(0, Sigma(truth_ell)); each member ~
    ``member_bias + member_scale * N(0, Sigma(member_ell))`` (``member_ell=None`` =>
    white). ``member_scale``/``member_bias`` inject pure marginal (variance/mean)
    errors while leaving the correlation structure intact.
    """
    g = torch.Generator().manual_seed(seed)
    truth = _draw_correlated((n, 1), d, truth_ell, g)  # (n, 1, d)
    members = _draw_correlated((n, 1, m), d, member_ell, g)  # (n, 1, m, d)
    members = member_bias + member_scale * members
    truth = truth.unsqueeze(-1).to(torch.float32)  # (n, 1, d, C=1)
    members = members.movedim(2, 3).unsqueeze(3).to(torch.float32)  # -> (n,1,d,1,m)
    return members, truth


def _mean_rank(hist: np.ndarray) -> float:
    """Mean observation rank implied by a rank histogram (uniform => (M+2)/2)."""
    ranks = np.arange(1, len(hist) + 1)
    return float((ranks * hist).sum())


def test_dependence_pre_rank_ranks_smooth_above_rough():
    """Sign anchor: the variogram pre-rank scores a *smooth* field above a *rough* one.

    ``[0, 1, 2, 3]`` (smooth ramp) and ``[0, 2, 1, 3]`` (rough) are permutations of
    the same values -- identical marginals, different spatial arrangement -- so any
    difference is pure correlation structure. rho(smooth) > rho(rough): a smoother
    observation ranks higher, which is why an under-correlated ensemble (members
    rougher than truth) drives the observation toward high ranks.
    """
    metric = DependenceRankHistogram(n_members=2)
    smooth = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    rough = torch.tensor([0.0, 2.0, 1.0, 3.0], dtype=torch.float64)
    obs = torch.zeros(4, dtype=torch.float64)
    pool = torch.stack([smooth, rough, obs], dim=-1).unsqueeze(0)  # (F=1, D=4, K=3)
    pre = metric._pre_rank(pool)  # (1, 3)
    assert pre[0, 0] > pre[0, 1]


def test_rows_per_pair_block_respects_the_cap_and_never_returns_zero():
    """The block size keeps ``rows * D * D * (M+1)`` under the cap, but is >= 1.

    The second half is the load-bearing part: a cap smaller than a single row's
    block must degrade to row-by-row, not to a zero-length block (which would
    loop forever or silently score nothing).
    """
    d, pool_size = 64, 11
    per_row = d * d * pool_size
    for cap in (per_row, 10 * per_row, 2**27):
        rows = rows_per_pair_block(per_row, cap)
        assert rows >= 1
        assert rows * per_row <= cap
    assert rows_per_pair_block(per_row, 1) == 1  # cap below one row


def test_dependence_pre_rank_is_invariant_to_chunking():
    """Chunking the pairwise block changes peak memory, never the pre-ranks.

    Frames are independent, so a cap that forces many passes must reproduce the
    single-pass result. This is what lets a whole enlarged test split be scored
    in one ``update`` call on a machine that cannot hold the full block.
    """
    torch.manual_seed(7)
    d, m = 12, 8
    pool = torch.randn(40, d, m + 1, dtype=torch.float64)

    unchunked = DependenceRankHistogram(n_members=m)._pre_rank(pool)
    # The budget is in BYTES, and one byte is below a single row, so the
    # implementation floors to one row and one frame per pass -- the most
    # chunked schedule available, against the unchunked default.
    chunked = DependenceRankHistogram(n_members=m, block_bytes=1)._pre_rank(pool)

    assert chunked.shape == unchunked.shape
    torch.testing.assert_close(chunked, unchunked, rtol=1e-13, atol=1e-13)


def test_dependence_rank_histogram_is_invariant_to_chunking():
    """End to end: the accumulated histogram is unchanged by the memory cap."""
    members, truth = _correlated_population(
        200, 12, truth_ell=4.0, member_ell=None, m=8, seed=41
    )
    torch.manual_seed(0)
    full = _hist_shape(DependenceRankHistogram(n_members=8), members, truth)
    torch.manual_seed(0)
    chunked = _hist_shape(
        DependenceRankHistogram(n_members=8, block_bytes=1),
        members,
        truth,
    )
    np.testing.assert_array_equal(chunked, full)


def test_dependence_rank_rejects_nonpositive_block_bytes():
    """A zero/negative cap is a caller error, not a silently unbounded run."""
    with pytest.raises(ValueError, match="block_bytes must be positive"):
        DependenceRankHistogram(n_members=4, block_bytes=0)


def test_dependence_rank_calibrated_is_uniform():
    """Members drawn from the truth's own correlated law give a flat histogram."""
    members, truth = _correlated_population(
        2500, 12, truth_ell=4.0, member_ell=4.0, m=16, seed=30
    )
    hist = _hist_shape(DependenceRankHistogram(n_members=16), members, truth)
    assert hist.shape == (17,)
    assert abs(hist.sum() - 1.0) < 1e-6
    uniform = 1.0 / 17
    assert np.max(np.abs(hist - uniform)) < 0.6 * uniform
    assert abs(_mean_rank(hist) - 9.0) < 0.8  # center = (M+2)/2 = 9


def test_dependence_rank_under_correlated_slopes_to_high_ranks():
    """Whitened members against a smooth (correlated) truth = an under-correlated
    ensemble: the truth is the smooth outlier and piles up at HIGH ranks."""
    members, truth = _correlated_population(
        2500, 12, truth_ell=5.0, member_ell=None, m=16, seed=31
    )
    hist = _hist_shape(DependenceRankHistogram(n_members=16), members, truth)
    assert _mean_rank(hist) > 13.0  # well above the uniform center (9)
    assert hist[-1] > hist[0]


def test_dependence_rank_over_correlated_slopes_to_low_ranks():
    """Members smoother (longer correlation length) than the truth = an over-correlated
    ensemble: the truth is the rough outlier and piles up at LOW ranks."""
    members, truth = _correlated_population(
        2500, 12, truth_ell=1.5, member_ell=12.0, m=16, seed=32
    )
    hist = _hist_shape(DependenceRankHistogram(n_members=16), members, truth)
    assert _mean_rank(hist) < 7.0  # clearly below the uniform center (9)
    assert hist[:4].sum() > hist[-4:].sum()  # mass concentrated in the low ranks


def test_dependence_rank_blind_to_variance_scaling():
    """Inflating the member variance (correct correlation) leaves the ranks unchanged:
    the /s(x)^p normalisation is scale-invariant, so the histogram is the same."""
    base = _correlated_population(2000, 12, 4.0, 4.0, m=16, seed=33, member_scale=1.0)
    infl = _correlated_population(2000, 12, 4.0, 4.0, m=16, seed=33, member_scale=1.5)
    h_base = _hist_shape(DependenceRankHistogram(n_members=16), *base)
    h_infl = _hist_shape(DependenceRankHistogram(n_members=16), *infl)
    assert np.allclose(h_base, h_infl, atol=1e-3)


def test_dependence_rank_blind_to_bias_shift():
    """Adding a constant to every member (correct correlation) leaves the ranks
    unchanged: the pairwise differences and the variance are both bias-invariant."""
    base = _correlated_population(2000, 12, 4.0, 4.0, m=16, seed=34, member_bias=0.0)
    shifted = _correlated_population(2000, 12, 4.0, 4.0, m=16, seed=34, member_bias=5.0)
    h_base = _hist_shape(DependenceRankHistogram(n_members=16), *base)
    h_shift = _hist_shape(DependenceRankHistogram(n_members=16), *shifted)
    assert np.allclose(h_base, h_shift, atol=1e-3)


def test_dependence_rank_rejects_bad_p():
    """The variogram power must be positive."""
    with pytest.raises(ValueError, match="p must be"):
        DependenceRankHistogram(n_members=8, p=0.0)


# ---------------------------------------------------------------------------
# Geometry-aware pair weights: grid_inverse_distance_weights (2-D, e.g. diff2d)
# and ring_inverse_distance_weights (1-D periodic, e.g. Lorenz-96 / l96c). Both
# fix DependenceRankHistogram's flattened-index default, w_ij = 1/|i-j|, which
# its own docstring flags as exact only for a non-wrapping 1-D field.
# ---------------------------------------------------------------------------

DIAGONAL_WEIGHT = 2.0**-0.5


def test_grid_weights_encode_grid_adjacency_not_flattened_adjacency():
    """The four nearest neighbours must all weigh 1, whichever axis they lie on.

    Under the flattened-order default a vertical neighbour is `width` apart and so
    is down-weighted by that factor -- the anisotropy this builder exists to fix.
    """
    height, width = 3, 4
    weights = grid_inverse_distance_weights(height, width)

    assert weights.shape == (height * width, height * width)
    assert torch.diagonal(weights).eq(0).all()
    assert torch.allclose(weights, weights.T)

    # Site 5 is (row 1, col 1). Horizontal neighbours 4 and 6, vertical 1 and 9.
    for neighbour in (4, 6, 1, 9):
        assert weights[5, neighbour] == 1.0, neighbour
    # Diagonal neighbours sit at sqrt(2).
    assert torch.isclose(
        weights[5, 0], torch.tensor(DIAGONAL_WEIGHT, dtype=torch.float64)
    )
    # The default the flattened index actually gives sees the vertical neighbour as
    # `width` away -- documented here so the anisotropy is not re-discovered.
    assert 1.0 / abs(5 - 1) == pytest.approx(1.0 / width)


def test_grid_weights_are_capped_at_one():
    """No pair may be up-weighted above adjacency: every off-diagonal weight <= 1."""
    weights = grid_inverse_distance_weights(5, 6)
    off_diag = weights[~torch.eye(30, dtype=torch.bool)]
    assert (off_diag <= 1.0).all()
    assert (off_diag >= 0.0).all()


def test_grid_weights_periodic_joins_opposite_edges():
    """Off by default (reflecting/Neumann fields have no wrap adjacency); when
    turned on, opposite edges become neighbours."""
    height = width = 4
    plain = grid_inverse_distance_weights(height, width)
    wrapped = grid_inverse_distance_weights(height, width, periodic=True)

    # Sites (0, 0) and (3, 0): three rows apart normally, adjacent when wrapped.
    assert plain[0, 12] == pytest.approx(1.0 / 3.0)
    assert wrapped[0, 12] == 1.0


def test_grid_weights_reject_a_mismatched_field_size():
    """height * width must match the field's site count D, checked via the metric
    that consumes the matrix -- the mismatch is a caller error, not silently ignored."""
    metric = DependenceRankHistogram(
        n_members=2, weights=grid_inverse_distance_weights(4, 4)
    )
    y_true: TensorBTSC = torch.randn(1, 1, 8, 8, 1, dtype=torch.float64)
    y_pred: TensorBTSCM = y_true.unsqueeze(-1).repeat(1, 1, 1, 1, 1, 2)

    with pytest.raises(ValueError, match="matching the spatial field size"):
        metric.update(y_pred, y_true)


def test_grid_weights_reject_nonpositive_extent():
    with pytest.raises(ValueError, match="height and width must be positive"):
        grid_inverse_distance_weights(0, 4)


def test_ring_weights_wrap_around_joins_the_two_ends():
    """A periodic ring's wrap-around pair (site 0, site N-1) must weigh 1, the
    same as any other nearest-neighbour pair -- the case the flattened default
    gets most wrong, scoring it as the single MOST distant pair instead."""
    n_sites = 40
    weights = ring_inverse_distance_weights(n_sites)

    assert weights.shape == (n_sites, n_sites)
    assert torch.diagonal(weights).eq(0).all()
    assert torch.allclose(weights, weights.T)
    assert (weights >= 0.0).all()
    assert (weights <= 1.0).all()

    assert weights[0, n_sites - 1] == 1.0
    assert weights[0, 1] == 1.0
    # Two sites straight across the ring are the most distant pair (n/2 apart).
    assert weights[0, n_sites // 2] == pytest.approx(1.0 / (n_sites // 2))

    # The flattened default this replaces treats the wrap-around pair as the
    # single most distant one instead of neighbours -- documented so the
    # regression is not re-discovered.
    flattened_default_weight = 1.0 / (n_sites - 1)
    assert flattened_default_weight < weights[0, n_sites - 1]


def test_ring_weights_do_not_wrap_when_built_as_a_plain_line():
    """Sanity check on the distance formula: without the ring's wrap term, the two
    ends of a line are the most distant pair, not neighbours -- i.e. the wrap
    term in ring_inverse_distance_weights is load-bearing, not a no-op."""
    n_sites = 6
    idx = torch.arange(n_sites, dtype=torch.float64)
    line_dist = (idx[:, None] - idx[None, :]).abs()
    line_weights = (1.0 / line_dist.clamp_min(1.0)) * (line_dist > 0)

    ring_weights = ring_inverse_distance_weights(n_sites)
    assert line_weights[0, n_sites - 1] == pytest.approx(1.0 / (n_sites - 1))
    assert ring_weights[0, n_sites - 1] == 1.0
    assert not torch.allclose(line_weights, ring_weights)


def test_ring_weights_reject_nonpositive_n_sites():
    with pytest.raises(ValueError, match="n_sites must be positive"):
        ring_inverse_distance_weights(0)


def test_dependence_rank_histogram_default_weighting_is_unchanged():
    """Passing no weights must still give the flattened-index default -- the
    geometry matrices are opt-in, so existing (non-grid, non-ring) callers keep
    identical numbers."""
    metric = DependenceRankHistogram(n_members=2)
    d = 6
    ref = metric._pair_weights(d, torch.zeros(1, dtype=torch.float64))
    idx = torch.arange(d, dtype=torch.float64)
    dist = (idx[:, None] - idx[None, :]).abs()
    expected = (1.0 / dist.clamp_min(1.0)) * (dist > 0)
    assert torch.allclose(ref, expected)


# ---------------------------------------------------------------------------
# Reliability index Delta (Candille & Talagrand 2005) -- a shared helper on the
# histogram base: the SUMMED absolute deviation from uniformity (compute() is the
# MEAN version). Tested on hand-built histograms via direct `counts` assignment.
# ---------------------------------------------------------------------------


def test_reliability_index_zero_for_flat_histogram():
    """A perfectly uniform histogram has reliability index Delta = 0."""
    metric = AverageRankHistogram(n_members=20)
    metric.counts = torch.ones(21, dtype=torch.float64)  # uniform after normalisation
    assert metric.reliability_index() == pytest.approx(0.0, abs=1e-12)


def test_reliability_index_one_bin_spike_known_value():
    """All mass in one bin gives Delta = 2M/(M+1) (magnitude cancels in the
    normalisation, so only the *shape* matters)."""
    m = 20
    metric = AverageRankHistogram(n_members=m)
    counts = torch.zeros(m + 1, dtype=torch.float64)
    counts[0] = 7.0
    metric.counts = counts
    # |1 - 1/(M+1)| + M * |0 - 1/(M+1)| = 2M/(M+1)
    assert metric.reliability_index() == pytest.approx(2 * m / (m + 1))


def test_reliability_index_is_the_sum_behind_the_mean_compute():
    """Delta (the sum) equals (M+1) * compute() (the mean absolute deviation) on the
    same histogram -- the two are the L1 deviation under different normalisations. Use a
    real (under-dispersed, non-uniform) update so both read the same populated state."""
    m = 16
    members, truth = _gaussian_population(2000, 8, member_std=0.5, m=m, seed=40)
    metric = AverageRankHistogram(n_members=m)
    metric.update(members, truth)
    delta = metric.reliability_index()
    assert delta == pytest.approx((m + 1) * float(metric.compute().item()))


def test_reliability_index_exposed_by_every_histogram():
    """The Delta helper lives on the base, so every histogram subclass inherits it."""
    for cls in (
        AverageRankHistogram,
        BandDepthRankHistogram,
        MultivariateRankHistogram,
        DependenceRankHistogram,
    ):
        metric = cls(n_members=8)
        metric.counts = torch.ones(9, dtype=torch.float64)
        assert metric.reliability_index() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Grid geometry and the dependence pre-rank: that the pair weights encode grid
# adjacency rather than flattened-array adjacency, that the direction reading
# survives them, and that the dependence histogram sees a permutation the
# pooled average-rank one is blind to.
# ---------------------------------------------------------------------------

P = 0.5
CALIBRATED_MEAN_RANK = 0.5
DIAGONAL_WEIGHT = 2.0**-0.5


def _naive_weighted_variogram(fields, weights, p):
    """The one-expression form: materialises the whole (N, D, D) tensor."""
    pairwise = (fields.unsqueeze(2) - fields.unsqueeze(1)).abs().pow(p)
    return (pairwise * weights[None]).sum(dim=(1, 2))


def _smooth(x, passes):
    """Average each site with its four neighbours, repeatedly (spatial dims 2, 3)."""
    for _ in range(passes):
        x = (
            x
            + x.roll(1, dims=2)
            + x.roll(-1, dims=2)
            + x.roll(1, dims=3)
            + x.roll(-1, dims=3)
        ) / 5.0
    return x


def _mean_rank_normalised(metric):
    hist = torch.as_tensor(metric.compute_histogram(), dtype=torch.float64)
    centres = (torch.arange(len(hist), dtype=torch.float64) + 0.5) / len(hist)
    return float((hist * centres).sum())


@pytest.mark.parametrize("block_bytes", [1, 1 << 10, 1 << 30])
def test_weighted_variogram_matches_unchunked(block_bytes):
    """Chunking is a memory strategy, not a different statistic."""
    torch.manual_seed(0)
    fields = torch.randn(5, 24, dtype=torch.float64)
    weights = torch.rand(24, 24, dtype=torch.float64)

    got = _weighted_variogram(fields, weights, P, block_bytes)
    expected = _naive_weighted_variogram(fields, weights, P)

    assert torch.allclose(got, expected, rtol=0, atol=1e-9)


def test_dependence_histogram_invariant_to_block_size():
    """The block budget must not move a single count."""
    torch.manual_seed(0)
    y_true: TensorBTSC = torch.randn(2, 3, 6, 6, 2, dtype=torch.float64)
    y_pred: TensorBTSCM = y_true.unsqueeze(-1) + 0.5 * torch.randn(
        2, 3, 6, 6, 2, 4, dtype=torch.float64
    )

    counts = []
    for block_bytes in (1, 1 << 30):
        torch.manual_seed(7)  # the rank is randomized on ties
        metric = DependenceRankHistogram(n_members=4, block_bytes=block_bytes)
        metric.update(y_pred, y_true)
        counts.append(metric.counts.clone())

    assert torch.equal(counts[0], counts[1])


def test_dependence_direction_is_not_inverted():
    """Members rougher than the truth (under-correlated) must rank the truth HIGH.

    The mean rank is the load-bearing output of this metric -- a sign error would
    invert the conclusion rather than merely degrade it.
    """
    torch.manual_seed(0)
    n_members, sites, passes = 8, 16, 8
    white_t = torch.randn(2, 3, sites, sites, 2, dtype=torch.float64)
    smooth_t = _smooth(white_t.clone(), passes)
    white_m = torch.randn(2, 3, sites, sites, 2, n_members, dtype=torch.float64)
    smooth_m = _smooth(white_m.clone(), passes)

    under = DependenceRankHistogram(n_members=n_members)
    under.update(white_m, smooth_t)  # members too rough
    over = DependenceRankHistogram(n_members=n_members)
    over.update(smooth_m, white_t)  # members too smooth

    assert (
        _mean_rank_normalised(under)
        > CALIBRATED_MEAN_RANK
        > _mean_rank_normalised(over)
    )


def test_dependence_runs_at_campaign_field_size():
    """A 64x64 field is the geometry the campaign runs at.

    Built as one tensor this pre-rank asks for 4.1 GiB per frame per pool element;
    the guard is that this completes at all, on CPU, with a modest block budget.
    """
    torch.manual_seed(0)
    n_members, side = 2, 64
    y_true: TensorBTSC = torch.randn(1, 1, side, side, 1, dtype=torch.float64)
    y_pred: TensorBTSCM = y_true.unsqueeze(-1) + torch.randn(
        1, 1, side, side, 1, n_members, dtype=torch.float64
    )

    metric = DependenceRankHistogram(n_members=n_members, block_bytes=1 << 24)
    metric.update(y_pred, y_true)

    assert float(metric.counts.sum()) == 1.0


def test_recorded_ranks_match_the_histogram():
    """Opt-in per-frame ranks must be the same ranks the histogram binned."""
    torch.manual_seed(0)
    y_true: TensorBTSC = torch.randn(2, 3, 5, 5, 2, dtype=torch.float64)
    y_pred: TensorBTSCM = y_true.unsqueeze(-1) + 0.5 * torch.randn(
        2, 3, 5, 5, 2, 4, dtype=torch.float64
    )

    metric = DependenceRankHistogram(n_members=4, record_ranks=True)
    metric.update(y_pred, y_true)
    ranks = metric.recorded_ranks()

    # one rank per (batch, time, channel) verification instance
    assert ranks.shape == (2 * 3 * 2,)
    binned = np.bincount(ranks.numpy() - 1, minlength=5)
    assert np.array_equal(binned, metric.counts.numpy().astype(int))


def test_recorded_ranks_are_off_by_default():
    """The default must not silently accumulate a per-frame buffer."""
    metric = AverageRankHistogram(n_members=4)
    with pytest.raises(RuntimeError, match="record_ranks"):
        metric.recorded_ranks()


def test_pooled_and_dependence_disagree_on_a_permuted_field():
    """The dependence pre-rank sees spatial arrangement; the pooled one does not."""
    torch.manual_seed(0)
    n_members = 6
    y_true: TensorBTSC = _smooth(torch.randn(2, 4, 12, 12, 1, dtype=torch.float64), 6)
    # Members share the truth's value multiset exactly, but scrambled in space.
    flat = y_true.reshape(-1)
    members = torch.stack(
        [
            flat[torch.randperm(flat.numel())].reshape(y_true.shape)
            for _ in range(n_members)
        ],
        dim=-1,
    )

    dependence = DependenceRankHistogram(n_members=n_members)
    dependence.update(members, y_true)
    pooled = PooledMultivariateRankHistogram(n_members=n_members)
    pooled.update(members, y_true)

    # Truth smooth, members scrambled -> under-correlated -> truth ranks high.
    assert _mean_rank_normalised(dependence) > _mean_rank_normalised(pooled)


def test_grid_weights_remove_the_flattened_orders_anisotropy():
    """A row-smooth field and its transpose must score identically on a square grid.

    The flattened default cannot: it reads along rows far more strongly than down
    columns, so it rates the two orientations differently though they are the same
    field observed through a relabelling of the axes.
    """
    side = 8
    generator = torch.Generator().manual_seed(0)
    field = torch.randn(side, side, generator=generator, dtype=torch.float64).cumsum(
        dim=1
    )
    pair = torch.stack([field.reshape(-1), field.T.reshape(-1)])

    index = torch.arange(side * side, dtype=torch.float64)
    flat_dist = (index[:, None] - index[None, :]).abs()
    flat_weights = (1.0 / flat_dist.clamp_min(1.0)) * (flat_dist > 0)

    flat = _weighted_variogram(pair, flat_weights, P, 1 << 30)
    grid = _weighted_variogram(
        pair, grid_inverse_distance_weights(side, side), P, 1 << 30
    )

    assert not torch.isclose(flat[0], flat[1], rtol=0.05)
    assert torch.allclose(grid[0], grid[1])


def test_grid_weights_preserve_the_direction_reading():
    """Swapping in grid weights must not invert the metric's sign convention."""
    torch.manual_seed(0)
    n_members, side, passes = 8, 16, 8
    weights = grid_inverse_distance_weights(side, side)
    white_t = torch.randn(2, 3, side, side, 2, dtype=torch.float64)
    smooth_t = _smooth(white_t.clone(), passes)
    white_m = torch.randn(2, 3, side, side, 2, n_members, dtype=torch.float64)
    smooth_m = _smooth(white_m.clone(), passes)

    under = DependenceRankHistogram(n_members=n_members, weights=weights)
    under.update(white_m, smooth_t)
    over = DependenceRankHistogram(n_members=n_members, weights=weights)
    over.update(smooth_m, white_t)

    assert (
        _mean_rank_normalised(under)
        > CALIBRATED_MEAN_RANK
        > _mean_rank_normalised(over)
    )


def test_a_tied_observation_ranks_the_same_at_either_end_of_the_pool():
    """A COLLAPSED ensemble is nothing but ties, and diagnosing that collapse is
    what a rank histogram is for -- so the tie convention has to be right exactly
    where it is least often exercised.

    ``_pool_ranks`` used argsort-of-argsort, which resolves ties BY POSITION. The
    two multivariate histogram classes stack the observation at OPPOSITE ends of
    the pool, so for one perfect forecast they reported opposite ends of the
    histogram -- both of them reading array order rather than the forecast. Under
    the average-rank convention the position cannot be read at all.

    Mutation this catches: restoring ``pool.argsort(-1).argsort(-1) + 1`` returns
    5.0 for the observation stacked last and 2.0 for the same observation stacked
    first.
    """
    members = torch.tensor([1.0, 2.0, 2.0, 2.0], dtype=torch.float64)
    observation = torch.tensor([2.0], dtype=torch.float64)

    stacked_last = torch.cat([members, observation]).reshape(1, 1, 5)
    stacked_first = torch.cat([observation, members]).reshape(1, 1, 5)

    rank_last = _pool_ranks(stacked_last)[0, 0, -1]
    rank_first = _pool_ranks(stacked_first)[0, 0, 0]

    assert rank_last == rank_first
    # 1 strictly below, 4 equal: 1 + 1 + (4 - 1) / 2.
    assert rank_last.item() == pytest.approx(3.5)


def test_average_ranks_are_identical_to_argsort_ranks_on_continuous_input():
    """The tie fix is INERT where there are no ties, which is why no histogram
    computed under the previous convention moves. Pinned rather than remeasured:
    this is the claim that lets earlier results stand unrescored.
    """
    torch.manual_seed(0)
    pool = torch.randn(8, 64, 9, dtype=torch.float64)

    argsort_ranks = pool.argsort(dim=-1).argsort(dim=-1).to(pool.dtype) + 1.0

    assert torch.equal(_pool_ranks(pool), argsort_ranks)


def _coherent_and_incoherent(
    n_traj: int, n_leads: int, n_sites: int, n_members: int, seed: int
) -> tuple[TensorBTSCM, TensorBTSCM, TensorBTSC]:
    """Two ensembles with the SAME per-frame marginals, different coherence.

    The coherent ensemble gives each MEMBER one offset held for its whole
    trajectory; the incoherent one reshuffles which member holds which offset
    independently at every lead. Frame by frame the two are the same set of
    values, so any per-frame statistic must agree; only a statistic that reads
    the trajectory as one object can tell them apart.
    """
    generator = torch.Generator().manual_seed(seed)
    offsets = torch.randn(n_traj, 1, n_sites, 1, n_members, generator=generator)
    coherent = offsets.expand(n_traj, n_leads, n_sites, 1, n_members).clone()
    incoherent = torch.stack(
        [
            coherent[:, lead][..., torch.randperm(n_members, generator=generator)]
            for lead in range(n_leads)
        ],
        dim=1,
    )
    truth = torch.zeros(n_traj, n_leads, n_sites, 1)
    return coherent, incoherent, truth


def test_trajectory_band_depth_sees_coherence_the_frame_wise_one_is_blind_to():
    """⛔ Why the trajectory grouping had to be written rather than reused.

    Plan §4.2 asks for a trajectory-SHAPE calibration read -- "do the frames
    hang together". The stock ``BandDepthRankHistogram`` groups one ranked
    vector per ``(batch, lead)`` frame, so it cannot answer that: this test
    builds two ensembles that are frame-for-frame the same values and differ
    only in which member holds which offset across leads, and shows the
    frame-wise reliability index does not move while the trajectory-grouped one
    does. Using the stock class for §4.2 would have reported a number that was
    blind to the very thing it was quoted for.
    """
    members = 8
    coherent, incoherent, truth = _coherent_and_incoherent(
        n_traj=64, n_leads=8, n_sites=5, n_members=members, seed=20260907
    )

    # ⚠ The base class breaks rank TIES with a draw from the GLOBAL RNG, so two
    # instances fed identical pre-ranks can still land in different bins. Seeded
    # identically before each update, the blindness below is exact rather than
    # approximate -- measured: without this the frame-wise gap is 0.0117 (six
    # frames of 512), which is tie-breaking noise and not coherence.
    def _updated(metric: Any, ensemble: TensorBTSCM) -> Any:
        torch.manual_seed(4242)
        metric.update(ensemble, truth)
        return metric

    frame_coherent = _updated(BandDepthRankHistogram(n_members=members), coherent)
    frame_incoherent = _updated(BandDepthRankHistogram(n_members=members), incoherent)
    traj_coherent = _updated(
        TrajectoryBandDepthRankHistogram(n_members=members), coherent
    )
    traj_incoherent = _updated(
        TrajectoryBandDepthRankHistogram(n_members=members), incoherent
    )

    frame_gap = abs(
        frame_coherent.reliability_index() - frame_incoherent.reliability_index()
    )
    traj_gap = abs(
        traj_coherent.reliability_index() - traj_incoherent.reliability_index()
    )
    assert frame_gap == pytest.approx(0.0, abs=1e-12), (
        "the frame-wise histogram must be EXACTLY blind here: the two ensembles "
        "are permutations of one another within each frame, and its pre-rank is "
        "computed per frame"
    )
    assert traj_gap > 0.1, (
        f"the trajectory grouping must SEE the difference; gap was {traj_gap}"
    )


def test_trajectory_band_depth_ranks_one_instance_per_trajectory():
    """``F = B``, not ``B * T`` -- the grouping change, pinned by count."""
    members = 8
    coherent, _, truth = _coherent_and_incoherent(
        n_traj=16, n_leads=8, n_sites=5, n_members=members, seed=11
    )
    metric = TrajectoryBandDepthRankHistogram(n_members=members)
    metric.update(coherent, truth)
    assert int(metric.counts.sum().item()) == 16
    frame_wise = BandDepthRankHistogram(n_members=members)
    frame_wise.update(coherent, truth)
    assert int(frame_wise.counts.sum().item()) == 16 * 8


# The randomized rank that breaks TIES draws from ``torch.rand``. Without a
# generator that is the GLOBAL RNG, so the same inputs can give different
# answers depending on what was drawn earlier in the process -- measured
# 2026-09-14 on the Task 14 block sweep, where two byte-identical runs moved
# ``trajectory_shape_reliability`` on 2 of 12 arms while every other column
# stayed bit-identical. These pin the fix (an optional ``generator``) and, just
# as importantly, pin that the DEFAULT is unchanged, so no number already
# published from this metric moves underneath its write-up.
def _fully_tied_case(
    n_traj: int = 64, n_leads: int = 8, n_sites: int = 5, n_members: int = 8
) -> tuple[TensorBTSCM, TensorBTSC]:
    """An ensemble whose members all equal the observation: every rank is a tie.

    With ``below = 0`` and ``tied = M`` the rank is decided ENTIRELY by the
    jitter, so this isolates the draw the fix is about. On continuous input
    there are no ties and the jitter cannot move anything -- which is why a
    naive determinism check on random data passes even with the bug present.

    Returns:
        ``(ensemble, truth)`` -- identically zero, so every pre-rank ties.
    """
    ensemble = torch.zeros(n_traj, n_leads, n_sites, 1, n_members)
    truth = torch.zeros(n_traj, n_leads, n_sites, 1)
    return ensemble, truth


def test_randomized_rank_reproduces_under_a_seeded_generator():
    """Same seed, same counts -- the property the Task 14 artefacts needed."""
    ensemble, truth = _fully_tied_case()
    counts = []
    for _ in range(2):
        metric = TrajectoryBandDepthRankHistogram(
            n_members=8, generator=torch.Generator().manual_seed(43)
        )
        metric.update(ensemble, truth)
        counts.append(metric.counts.clone())
    assert torch.equal(counts[0], counts[1]), (
        "two metrics given the same seeded generator must accumulate identical "
        f"rank counts; got {counts[0].tolist()} and {counts[1].tolist()}"
    )


def test_randomized_rank_actually_consults_the_generator():
    """Different seeds must move the histogram, or the tie-break is inert.

    ⚠ **Checked, and it does NOT catch dropping ``generator=self._generator``**
    -- under that mutation two draws off the global RNG still differ, so this
    test stays green and only
    :func:`test_randomized_rank_reproduces_under_a_seeded_generator` goes red.
    What it DOES catch is the tie-break being made constant (a fixed jitter, or
    the seed ignored), which would make every seed agree. Stated as measured
    rather than as intended, because a test credited with a mutation it does not
    catch is worse than no test.
    """
    ensemble, truth = _fully_tied_case()
    counts = []
    for seed in (43, 44):
        metric = TrajectoryBandDepthRankHistogram(
            n_members=8, generator=torch.Generator().manual_seed(seed)
        )
        metric.update(ensemble, truth)
        counts.append(metric.counts.clone())
    assert not torch.equal(counts[0], counts[1]), (
        "two different seeds produced identical tie-break counts over 64 fully "
        "tied trajectories, which means the generator is not being consulted"
    )


def test_randomized_rank_default_still_follows_the_global_rng():
    """The default is UNCHANGED, deliberately (D123's precedent).

    Every number already published from this metric was produced on the global
    RNG. Seeding by default would silently move them, so ``generator=None``
    must keep taking the global stream -- pinned by seeding the global RNG and
    getting a reproducible answer without passing a generator at all.
    """
    ensemble, truth = _fully_tied_case()
    counts = []
    for _ in range(2):
        torch.manual_seed(7)
        metric = TrajectoryBandDepthRankHistogram(n_members=8)
        metric.update(ensemble, truth)
        counts.append(metric.counts.clone())
    assert torch.equal(counts[0], counts[1])
