import itertools
import math

import pytest
import torch
from einops import rearrange, repeat

from autocast.metrics import ALL_ENSEMBLE_METRICS
from autocast.metrics.base import (
    DEFAULT_MAX_PAIR_ELEMENTS,
    BaseMetric,
    rows_per_pair_block,
)
from autocast.metrics.coverage import Coverage
from autocast.metrics.deterministic import RMSE
from autocast.metrics.ensemble import (
    CRPS,
    AlmostFairEnergyScore,
    AlphaFairCRPS,
    AlphaFairCRPSMAETerm,
    AlphaFairCRPSSpreadTerm,
    CRPSMAETerm,
    CRPSSpreadTerm,
    DawidSebastianiScore,
    EnergyScore,
    EnsembleSkill,
    EnsembleSpread,
    FairCRPSSpreadTerm,
    LogScore,
    MultiWinkler,
    PooledSpreadSkillRatio,
    SpatialVariogramScore,
    SpatiotemporalVariogramScore,
    SpreadSkillRatio,
    TemporalAutocorrelationMatch,
    TemporalVariogramScore,
    VariogramScore,
    WinklerScore,
    _alpha_fair_crps_score,
    _common_crps_score,
    _shrunk_covariance,
)
from autocast.processors.drifting import _energy_score
from autocast.types import TensorBTSC
from autocast.types.types import TensorBTC

# Metrics with no per-batch score() by construction: the generic parametrized
# tests below all call score(), which these cannot answer. PooledSpreadSkillRatio
# divides only after accumulating across batches, so a per-batch value would
# defeat its purpose; it is exercised through update()/compute() by its own
# dedicated tests instead.
SCORELESS_METRICS = (PooledSpreadSkillRatio,)

ENSEMBLE_BASE_METRICS = tuple(
    metric_cls
    for metric_cls in ALL_ENSEMBLE_METRICS
    if issubclass(metric_cls, BaseMetric) and metric_cls not in SCORELESS_METRICS
)

ENSEMBLE_ERROR_METRICS = tuple(
    m
    for m in ENSEMBLE_BASE_METRICS
    if m
    not in [
        Coverage,
        VariogramScore,
        SpreadSkillRatio,
        EnsembleSpread,
        CRPSSpreadTerm,
        FairCRPSSpreadTerm,
        AlphaFairCRPSSpreadTerm,
        # The Dawid-Sebastiani score is a log-score, not a distance-to-truth:
        # a zero-spread, exactly-correct forecast has *infinite* density, so it
        # does not vanish when y_pred == y_true. Tested separately below.
        DawidSebastianiScore,
        # LogScore is DawidSebastianiScore's marginal (per-site) counterpart --
        # same log-density construction, so it shares the same non-vanishing
        # behaviour at y_pred == y_true.
        LogScore,
    ]
)


def _make_metric(MetricCls):
    if MetricCls in (AlmostFairEnergyScore, EnergyScore, VariogramScore):
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


def test_variogram_score_is_invariant_to_pair_block_chunking():
    """Capping the all-pairs block changes peak memory, never the score.

    Batch entries are independent, so a cap that forces one block per entry must
    reproduce the single-block result exactly. This is what makes scoring a whole
    enlarged test split feasible: unchunked, a spatiotemporal vector over 400
    cases needs tens of GB and is simply refused by the allocator.
    """
    torch.manual_seed(3)
    y_pred = torch.randn(6, 2, 4, 1, 5, dtype=torch.float64)
    y_true = torch.randn(6, 2, 4, 1, dtype=torch.float64)

    kwargs = {"p": 0.5, "vector_dims": "spatial_temporal", "reduce_all": False}
    unchunked = VariogramScore(**kwargs).score(y_pred, y_true)
    # D = 2 leads * 4 sites = 8, M = 5 -> one batch entry per block.
    chunked = VariogramScore(max_pair_elements=8 * 8 * 5, **kwargs).score(
        y_pred, y_true
    )

    assert chunked.shape == unchunked.shape
    torch.testing.assert_close(chunked, unchunked, rtol=1e-13, atol=1e-13)


def test_variogram_score_blocks_below_the_batch_row():
    """Blocking must reach below one batch row -- across lead-time and channel.

    For ``vector_dims="spatial"`` the leading (independent) axes are batch,
    lead-time *and* channel, since only the spatial sites are vectorized into
    D. A cap that forces one (batch, lead, channel) frame per block -- finer
    than one batch row -- must still reproduce the unchunked score exactly:
    this is the fix for the OOM (a whole-batch-row block was 16x too big to
    shrink further at the reported 64x64/T4/C4/M8 eval shape).
    """
    torch.manual_seed(4)
    # (B=2, T=3, S=5, C=4): D = 5 sites, M = 6 members. Every leading extent is
    # DISTINCT and the leading shape (2, 3, 4) is deliberately NOT palindromic:
    # with a symmetric shape a helper that reshapes the blocks back onto the
    # leading axes in the WRONG ORDER still round-trips, and this assertion
    # cannot see it. Verified by mutation -- reversing the reshape order fails
    # this test only because the extents differ.
    y_pred = torch.randn(2, 3, 5, 4, 6, dtype=torch.float64)
    y_true = torch.randn(2, 3, 5, 4, dtype=torch.float64)

    kwargs = {"p": 0.5, "vector_dims": "spatial", "reduce_all": False}
    unchunked = VariogramScore(**kwargs).score(y_pred, y_true)
    # D=5, M=6 -> one (b, t, c) frame costs 5*5*6=150 elements; one batch row
    # (bundling T=3, C=4) would cost 1800. Capping at 150 forces one frame per
    # block -- splitting across T and C, not just B.
    chunked = VariogramScore(max_pair_elements=5 * 5 * 6, **kwargs).score(
        y_pred, y_true
    )

    assert chunked.shape == unchunked.shape
    torch.testing.assert_close(chunked, unchunked, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
    ("metric_cls", "kwargs", "tol"),
    [
        (VariogramScore, {"p": 0.5, "vector_dims": "spatial"}, 1e-12),
        (DawidSebastianiScore, {"vector_dims": "spatial", "jitter": 1e-6}, 1e-9),
    ],
)
def test_blocked_frames_land_on_the_right_leading_positions(metric_cls, kwargs, tol):
    """Each block's result must be reassembled at the frame it came FROM.

    The chunked-vs-unchunked assertions nearby cannot see this. Both of their
    branches run through ``block_pairwise_frames``, so a systematic
    reassembly bug -- reshaping onto the leading axes in the wrong ORDER, say
    -- corrupts both sides identically and the comparison still passes.
    Verified by mutation: reversing that reshape leaves those tests green.

    This one is absolute, not relative. It pins the output shape as a literal
    and checks every entry against a reference scored ONE FRAME AT A TIME,
    each frame sliced out and passed in on its own, so the helper plays no
    part in producing the expected value.
    """
    torch.manual_seed(11)
    # All leading extents distinct (B=2, T=3, C=4) so a transposed or reversed
    # reassembly cannot round-trip into the same shape.
    y_pred = torch.randn(2, 3, 5, 4, 6, dtype=torch.float64)
    y_true = torch.randn(2, 3, 5, 4, dtype=torch.float64)

    # One frame per block: D=5, M=6 -> variogram 150 elements, DSS 25.
    cap = 5 * 5 * 6 if metric_cls is VariogramScore else 5 * 5
    scored = metric_cls(max_pair_elements=cap, reduce_all=False, **kwargs).score(
        y_pred, y_true
    )

    assert tuple(scored.shape) == (2, 3, 4)

    for b, t, c in itertools.product(range(2), range(3), range(4)):
        frame = metric_cls(reduce_all=False, **kwargs).score(
            y_pred[b : b + 1, t : t + 1, :, c : c + 1, :],
            y_true[b : b + 1, t : t + 1, :, c : c + 1],
        )
        torch.testing.assert_close(
            scored[b, t, c], frame.reshape(()), rtol=tol, atol=tol
        )


def test_variogram_score_rejects_nonpositive_max_pair_elements():
    """A zero/negative cap is a caller error, not a silently unbounded run."""
    with pytest.raises(ValueError, match="max_pair_elements must be positive"):
        VariogramScore(max_pair_elements=0)


def test_variogram_score_p_half_full_frame_matches_reference():
    """VS-0.5 (the study's config: p=0.5, uniform weights, full frame).

    Pins the correlation-sensitive proper score the cal-study reports: order
    p=0.5, default uniform weights, vectorising the whole (spatial, channel)
    frame. Reference is the brute-force double sum over the frame's elements.
    """
    torch.manual_seed(0)
    # (B=2, T=1, S=3, C=2) frame -> D = 3 * 2 = 6 vector elements, M=5 members.
    y_pred = torch.randn(2, 1, 3, 2, 5, dtype=torch.float64)
    y_true = torch.randn(2, 1, 3, 2, dtype=torch.float64)

    metric = VariogramScore(
        p=0.5, vector_dims="spatial_temporal_channels", reduce_all=False
    )
    score = metric.score(y_pred, y_true).reshape(2)  # one value per batch element

    p = 0.5
    xv = y_pred.reshape(2, 6, 5)
    yv = y_true.reshape(2, 6)
    ref = torch.zeros(2, dtype=torch.float64)
    for b in range(2):
        total = 0.0
        for i in range(6):
            for j in range(6):
                ens_exp = (xv[b, i] - xv[b, j]).abs().pow(p).mean()
                obs = (yv[b, i] - yv[b, j]).abs().pow(p)
                total = total + (ens_exp - obs) ** 2  # uniform weight w_ij = 1
        ref[b] = total
    assert torch.allclose(score, ref, atol=1e-10)


def test_variogram_score_weights_default_is_uniform():
    """The default (weights=None) equals an explicit all-ones weight matrix."""
    torch.manual_seed(1)
    y_pred = torch.randn(1, 1, 4, 1, 3, dtype=torch.float64)  # D = 4
    y_true = torch.randn(1, 1, 4, 1, dtype=torch.float64)
    kw = {"p": 0.5, "vector_dims": "spatial_temporal_channels", "reduce_all": False}
    default = VariogramScore(**kw).score(y_pred, y_true)
    explicit = VariogramScore(weights=torch.ones(4, 4), **kw).score(y_pred, y_true)
    assert torch.allclose(default, explicit, atol=1e-12)


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


def test_crps_component_metrics_match_manual_decomposition():
    y_pred = torch.tensor([[[[[0.0, 2.0]]]]])  # (1, 1, 1, 1, 2)
    y_true = torch.tensor([[[[1.0]]]])  # (1, 1, 1, 1)

    crps = CRPS(reduce_all=False).score(y_pred, y_true)
    mae_term = CRPSMAETerm(reduce_all=False).score(y_pred, y_true)
    spread_term = CRPSSpreadTerm(reduce_all=False).score(y_pred, y_true)
    ensemble_spread = EnsembleSpread(corrected=False, reduce_all=False).score(
        y_pred, y_true
    )

    assert torch.allclose(mae_term, torch.tensor([[[1.0]]]))
    assert torch.allclose(spread_term, torch.tensor([[[0.5]]]))
    assert torch.allclose(crps, mae_term - spread_term)
    assert torch.allclose(crps, torch.tensor([[[0.5]]]))
    assert not torch.allclose(spread_term, ensemble_spread)


def test_afcrps_component_metrics_match_manual_decomposition():
    y_pred = torch.tensor([[[[[0.0, 2.0]]]]])  # (1, 1, 1, 1, 2)
    y_true = torch.tensor([[[[1.0]]]])  # (1, 1, 1, 1)

    afcrps = AlphaFairCRPS(alpha=0.75, reduce_all=False).score(y_pred, y_true)
    mae_term = AlphaFairCRPSMAETerm(alpha=0.75, reduce_all=False).score(y_pred, y_true)
    spread_term = AlphaFairCRPSSpreadTerm(alpha=0.75, reduce_all=False).score(
        y_pred, y_true
    )

    assert torch.allclose(mae_term, torch.tensor([[[1.0]]]))
    assert torch.allclose(spread_term, torch.tensor([[[0.875]]]))
    assert torch.allclose(afcrps, mae_term - spread_term)
    assert torch.allclose(afcrps, torch.tensor([[[0.125]]]))


def test_dss_penalizes_overdispersion_when_mean_is_correct():
    # When the ensemble mean already matches the truth, the (y-mean)^2/var term
    # is ~0 and the score is dominated by 0.5*log(2*pi*var): a tighter ensemble
    # is rewarded and an over-dispersed one penalised. This is exactly the
    # over-dispersion the low-rank head exhibits.
    torch.manual_seed(0)
    truth = torch.zeros((1, 1, 4096, 1))
    tight = torch.randn((1, 1, 4096, 1, 32)) * 0.1
    wide = torch.randn((1, 1, 4096, 1, 32)) * 1.0

    s_tight = DawidSebastianiScore().score(tight, truth).mean()
    s_wide = DawidSebastianiScore().score(wide, truth).mean()

    assert s_tight < s_wide


def test_multiwinkler_matches_mean_winkler_scores():
    y_pred = torch.randn((2, 3, 4, 4, 1, 16))
    y_true = torch.randn((2, 3, 4, 4, 1))
    coverage_levels = [0.5, 0.9]

    multiwinkler = MultiWinkler(coverage_levels=coverage_levels)
    multiwinkler.update(y_pred, y_true)

    expected_scores = []
    for coverage_level in coverage_levels:
        winkler = WinklerScore(alpha=1.0 - coverage_level)
        winkler.update(y_pred, y_true)
        expected_scores.append(winkler.compute())

    assert torch.allclose(multiwinkler.compute(), torch.stack(expected_scores).mean())


def test_multiwinkler_manual_interval_score_average():
    members = torch.arange(5.0)
    y_pred = members.view(1, 1, 1, 1, 5).expand(1, 2, 2, 1, 5)
    y_true = torch.tensor([[[[2.0], [4.5]], [[-0.5], [1.5]]]])

    metric = MultiWinkler(coverage_levels=[0.5, 0.8])
    score = metric.score(y_pred, y_true)
    metric.update(y_pred, y_true)

    expected_by_level = torch.tensor(
        [
            [[[5.0], [5.0]]],
            [[[7.7], [7.7]]],
        ]
    )
    expected_scalar = expected_by_level.mean()

    assert torch.allclose(score, expected_by_level)
    assert torch.allclose(metric.compute(), expected_scalar)


def test_multiwinkler_score_shape_includes_interval_levels():
    y_pred = torch.randn((2, 3, 4, 4, 5, 16))
    y_true = torch.randn((2, 3, 4, 4, 5))

    score = MultiWinkler(coverage_levels=[0.5, 0.9]).score(y_pred, y_true)

    assert score.shape == (2, 2, 3, 5)


def test_multiwinkler_plot_saves_png_and_csv(tmp_path):
    y_pred = torch.randn((2, 3, 4, 4, 1, 16))
    y_true = torch.randn((2, 3, 4, 4, 1))
    metric = MultiWinkler(coverage_levels=[0.5, 0.9])
    metric.update(y_pred, y_true)

    save_path = tmp_path / "multiwinkler.png"
    fig = metric.plot(save_path=str(save_path))

    assert fig is not None
    assert save_path.exists()
    assert save_path.with_suffix(".csv").exists()


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
def test_multivariate_scores_vector_dims_typos_invalid(validation_error, MetricCls):
    with validation_error(ValueError):
        MetricCls(vector_dims="spatials")

    with validation_error(ValueError):
        MetricCls(vector_dims="temproal")


@pytest.mark.parametrize(
    ("MetricCls", "expected_vector_dims", "expected_name"),
    [
        (SpatialVariogramScore, "spatial", "variogram_spatial"),
        (TemporalVariogramScore, "temporal", "variogram_temporal"),
        (
            SpatiotemporalVariogramScore,
            "spatial_temporal",
            "variogram_spatiotemporal",
        ),
    ],
)
def test_variogram_vector_dims_aliases_default(
    MetricCls, expected_vector_dims, expected_name
):
    """Each alias bakes in its column's default vector_dims and its own name."""
    metric = MetricCls()
    assert metric.vector_dims == expected_vector_dims
    assert metric.name == expected_name


def test_variogram_alias_default_stays_overridable():
    """The alias only changes the default -- vector_dims stays overridable."""
    metric = SpatialVariogramScore(vector_dims="temporal")
    assert metric.vector_dims == "temporal"


def test_variogram_alias_scores_identically_to_base():
    """An alias must score identically to VariogramScore at the same vector_dims."""
    torch.manual_seed(0)
    y_pred = torch.randn((2, 3, 4, 5, 6))
    y_true = torch.randn((2, 3, 4, 5))
    kw = {"p": 0.5, "score_dims": None, "reduce_all": False}
    alias = SpatialVariogramScore(**kw).score(y_pred, y_true)
    base = VariogramScore(vector_dims="spatial", **kw).score(y_pred, y_true)
    assert torch.allclose(alias, base)


# --- TemporalAutocorrelationMatch (Phase 3) ---


def _ar1_ensemble(rho, n_members, n_steps, *, seed):
    """Seeded AR(1) ensemble started from the stationary distribution.

    Returns a y_pred-shaped tensor ``(B=1, T=n_steps, S=1, C=1, M=n_members)``.
    """
    g = torch.Generator().manual_seed(seed)
    stationary_std = (1.0 / (1.0 - rho**2)) ** 0.5  # 1.0 at rho = 0
    x = torch.empty(n_members, n_steps)
    x[:, 0] = torch.randn(n_members, generator=g) * stationary_std
    for t in range(1, n_steps):
        x[:, t] = rho * x[:, t - 1] + torch.randn(n_members, generator=g)
    # (M, T) -> (B, T, S, C, M)
    return x.permute(1, 0)[None, :, None, None, :]


def test_autocorr_estimator_matches_hand_computed_biased_acf():
    """The biased ACF on a deterministic ramp matches the hand-computed values."""
    # x = [1, 2, 3, 4]; mean 2.5; var-sum 5.0.
    # lag1 = (1.25)/5 = 0.25; lag2 = (-1.5)/5 = -0.3; lag3 = (-2.25)/5 = -0.45.
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 4, 1, 1)
    acf = TemporalAutocorrelationMatch()._truth_autocorr(y_true, max_lag=3)
    expected = torch.tensor([0.25, -0.3, -0.45])
    assert torch.allclose(acf.reshape(3), expected, atol=1e-6)


def test_autocorr_is_scale_invariant_for_small_magnitude_fields():
    """Autocorrelation is scale-invariant; an absolute denom floor would shrink it."""
    torch.manual_seed(0)
    base = torch.randn(1, 64, 1, 1)  # (B, T, S, C)
    metric = TemporalAutocorrelationMatch()
    acf_small = metric._truth_autocorr(base * 1e-7, max_lag=3)
    acf_large = metric._truth_autocorr(base * 1e7, max_lag=3)
    assert torch.allclose(acf_small, acf_large, atol=1e-6)


def test_autocorr_match_recovers_ar1_rho_k():
    """Member-pooled autocorrelation of AR(1) paths recovers rho^k (analytic)."""
    rho = 0.7
    y_pred = _ar1_ensemble(rho, n_members=512, n_steps=256, seed=0)
    acf = (
        TemporalAutocorrelationMatch()
        ._member_pooled_autocorr(y_pred, max_lag=3)
        .reshape(3)
    )
    for k, recovered in enumerate(acf.tolist(), start=1):
        assert abs(recovered - rho**k) < 0.03


def test_autocorr_match_discrepancy_zero_when_ensemble_matches_truth():
    """Ensemble and observation from the same AR(1) -> near-zero discrepancy profile."""
    rho = 0.6
    y_pred = _ar1_ensemble(rho, n_members=512, n_steps=256, seed=1)
    y_true = _ar1_ensemble(rho, n_members=1, n_steps=256, seed=2)[..., 0]
    profile = TemporalAutocorrelationMatch(max_lag=5).discrepancy_profile(
        y_pred, y_true
    )
    assert profile.shape == (1, 1, 1, 5)
    assert profile.max().item() < 0.15


def test_autocorr_match_discrepancy_flags_wrong_autocorr():
    """An iid ensemble against a strongly autocorrelated truth is flagged at lag 1."""
    y_pred = _ar1_ensemble(0.0, n_members=512, n_steps=256, seed=3)  # iid
    y_true = _ar1_ensemble(0.9, n_members=1, n_steps=256, seed=4)[..., 0]  # rho=0.9
    profile = TemporalAutocorrelationMatch(max_lag=3).discrepancy_profile(
        y_pred, y_true
    )
    assert profile.reshape(3)[0].item() > 0.5


def test_autocorr_match_score_shape_and_scalar_reduction():
    """_score collapses the lead axis to a singleton; default reduce_all -> scalar."""
    y_pred = _ar1_ensemble(0.5, n_members=16, n_steps=32, seed=5)
    y_true = _ar1_ensemble(0.5, n_members=1, n_steps=32, seed=6)[..., 0]
    unreduced = TemporalAutocorrelationMatch(
        max_lag=4, score_dims=None, reduce_all=False
    ).score(y_pred, y_true)
    assert unreduced.shape == (1, 1, 1, 1)  # (B, 1, S, C)
    metric = TemporalAutocorrelationMatch(max_lag=4)
    metric.update(y_pred, y_true)
    assert metric.compute().ndim == 0


def test_autocorr_match_handles_multiple_spatial_dims():
    """The lead-axis reduction preserves multiple spatial dims -> (B, 1, S1, S2, C)."""
    torch.manual_seed(0)
    y_pred = torch.randn((2, 8, 3, 4, 5, 6))  # B, T, S1, S2, C, M
    y_true = torch.randn((2, 8, 3, 4, 5))
    score = TemporalAutocorrelationMatch(
        max_lag=3, score_dims=None, reduce_all=False
    ).score(y_pred, y_true)
    assert score.shape == (2, 1, 3, 4, 5)


def test_autocorr_match_invalid_parameters():
    with pytest.raises(ValueError):  # noqa: PT011
        TemporalAutocorrelationMatch(max_lag=0)
    # A single lead step cannot form a lag.
    y_pred = torch.randn((1, 1, 1, 1, 4))
    y_true = torch.randn((1, 1, 1, 1))
    with pytest.raises(ValueError):  # noqa: PT011
        TemporalAutocorrelationMatch().score(y_pred, y_true)


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


def _controlled_ssr_batch(
    sigma_t: torch.Tensor,
    bias: float | torch.Tensor,
    B: int,
    S1: int,
    S2: int,
    C: int,
    M: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an ensemble with known per-lead-time spread and skill.

    For each lead time t, ensemble members equal ``bias + sigma_t * z_m'`` where
    ``z_m'`` are ensemble-centered (i.e. sum to zero across ``M``) standard
    normals. This gives:
      - pointwise ensemble mean == bias (so skill == |bias| exactly),
      - pointwise unbiased ensemble variance with expectation ``sigma_t**2``.

    ``bias`` may be a scalar (constant skill across time) or a 1D tensor of
    length T (per-lead-time skill). Ensemble-centering removes sampling noise
    in the skill so per-t SSR values are predictable within a tight tolerance
    even with modest B/S/M.
    """
    T = sigma_t.shape[0]
    torch.manual_seed(seed)
    z = torch.randn((B, T, S1, S2, C, M))
    z = z - z.mean(dim=-1, keepdim=True)
    sigma = sigma_t.view(1, T, 1, 1, 1, 1)
    bias_bcast = bias.view(1, T, 1, 1, 1, 1) if isinstance(bias, torch.Tensor) else bias
    y_pred = bias_bcast + sigma * z
    y_true = torch.zeros((B, T, S1, S2, C))
    return y_pred, y_true


def test_spread_skill_ratio_over_time_tracks_growing_skill_when_spread_is_fixed():
    """Holding spread constant and growing skill with lead time, SSR should
    decrease monotonically across T (``spread / skill -> 0``). This emulates a
    rollout where ensemble fails to broaden as error grows -- the classic
    under-dispersion signature that should show up on the lead-time panel.
    """
    B, T, S1, S2, C, M = 8, 5, 16, 16, 1, 32
    skill_t = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    sigma_t = torch.ones(T)  # spread fixed at 1.0

    y_pred, y_true = _controlled_ssr_batch(
        sigma_t, bias=skill_t, B=B, S1=S1, S2=S2, C=C, M=M
    )

    ssr = SpreadSkillRatio().score(y_pred, y_true)  # (B, T, C)
    per_t = ssr.mean(dim=(0, 2))

    correction = float(((M + 1) / M) ** 0.5)
    expected = correction / skill_t
    assert torch.allclose(per_t, expected, rtol=5e-2, atol=5e-3), (
        per_t,
        expected,
    )
    # Explicit monotonic-decrease check: belt-and-braces against any future
    # change that silently re-orders time or swaps numerator/denominator.
    diffs = per_t[1:] - per_t[:-1]
    assert torch.all(diffs < 0), per_t


def test_spread_skill_ratio_over_time_calibrated_stays_near_one():
    """Well-dispersed ensemble: members and truth drawn iid from the same
    per-lead-time distribution, with independent noise for truth.

    The classical identities used here:
      - E[spread^2]  = sigma_t^2           (unbiased variance estimator)
      - E[skill^2]   = sigma_t^2 * (M+1)/M (ensemble mean vs independent truth)
      -> SSR_corrected = sqrt(M/(M+1)) * sqrt((M+1)/M) = 1.

    SCOPE (important, and narrower than it looks): those identities constrain the
    ratio of the two EXPECTATIONS. ``SpreadSkillRatio`` forms one ratio per case
    and averages the ratios, which is a different functional and equals 1 only
    when each case averages over many effectively independent sites. This test
    uses SPATIALLY INDEPENDENT noise over 256 sites, so it lives in exactly that
    regime and passes. It is therefore NOT evidence that the metric reads 1 on
    correlated fields -- it does not. See
    ``test_spread_skill_ratio_mean_of_ratios_is_biased_on_correlated_fields``
    for the measured departure, and ``PooledSpreadSkillRatio`` for the estimator
    that does hold at any correlation length.
    """
    B, S1, S2, C, M = 16, 16, 16, 1, 16
    sigma_t = torch.tensor([0.5, 1.0, 2.0, 4.0])
    T = sigma_t.shape[0]

    torch.manual_seed(0)
    sigma_bcast = sigma_t.view(1, T, 1, 1, 1, 1)
    y_pred = sigma_bcast * torch.randn((B, T, S1, S2, C, M))
    y_true = sigma_bcast.squeeze(-1) * torch.randn((B, T, S1, S2, C))

    ssr = SpreadSkillRatio().score(y_pred, y_true)  # (B, T, C)
    per_t = ssr.mean(dim=(0, 2))

    # Sampling noise at (B=16, S=256, M=16); 10% tolerance is comfortably safe.
    assert torch.all(torch.abs(per_t - 1.0) < 0.1), per_t


def _calibrated_correlated_draw(corr_len, n=16, batch=16, n_lead=2, members=16, seed=0):
    """Members and truth drawn from ONE Gaussian law with a given spatial
    correlation length, so the ensemble is calibrated by construction
    (exchangeable draws). corr_len=0 gives spatially independent noise.
    """
    generator = torch.Generator().manual_seed(seed)

    def draw():
        field = torch.randn((batch, n_lead, 1, n, n), generator=generator)
        if corr_len > 0:
            axis = torch.arange(n)
            dist = torch.minimum(axis, n - axis).float()
            scale = corr_len / math.sqrt(2.0)
            kernel = torch.exp(
                -0.5 * ((dist[:, None] ** 2 + dist[None, :] ** 2) / scale**2)
            )
            field = torch.fft.irfft2(
                torch.fft.rfft2(field) * torch.fft.rfft2(kernel), s=(n, n)
            )
            field = field / field.std()
        return field.permute(0, 1, 3, 4, 2)  # (B, T, S1, S2, C)

    y_pred = torch.stack([draw() for _ in range(members)], dim=-1)
    return y_pred, draw()


@pytest.mark.parametrize("corr_len", [0, 1, 2, 4, 8])
def test_pooled_spread_skill_ratio_reads_one_at_any_correlation_length(corr_len):
    """The property the metric exists to have: a provably calibrated ensemble
    scores 1, whatever the field's smoothness.

    This is the check the mean-of-ratios form cannot pass -- see the companion
    test below. Dividing only after pooling over cases and space keeps the
    estimator on the ratio-of-expectations that the calibration identity
    actually constrains.
    """
    metric = PooledSpreadSkillRatio()
    for update_idx in range(8):
        y_pred, y_true = _calibrated_correlated_draw(corr_len, seed=update_idx)
        metric.update(y_pred, y_true)

    assert abs(float(metric.compute()) - 1.0) < 0.05


def test_spread_skill_ratio_mean_of_ratios_is_biased_on_correlated_fields():
    """Pin the KNOWN, intentional bias of the mean-of-ratios form.

    Not a defect being tolerated silently: ``SpreadSkillRatio`` reproduces the
    aggregation used by the reference implementation it was ported from, and is
    kept for comparability. This test records what that choice costs, so the
    number is never mistaken for a calibration failure of the model under test.

    On a perfectly calibrated ensemble the value grows with the field's
    correlation length -- there is no single "calibrated" reference value for
    this estimator, which is why it must be read against a matched null and why
    ``PooledSpreadSkillRatio`` exists.
    """
    values = {}
    for corr_len in (0, 4, 8):
        metric = SpreadSkillRatio()
        for update_idx in range(8):
            y_pred, y_true = _calibrated_correlated_draw(corr_len, seed=update_idx)
            metric.update(y_pred, y_true)
        values[corr_len] = float(metric.compute())

    # Independent sites: the regime where mean-of-ratios happens to be fine.
    assert abs(values[0] - 1.0) < 0.05
    # Smooth fields: materially high, and monotonically worse with smoothness.
    assert values[4] > 1.1
    assert values[8] > values[4]

    # The pooled estimator stays at 1 on the very same draws.
    pooled = PooledSpreadSkillRatio()
    for update_idx in range(8):
        y_pred, y_true = _calibrated_correlated_draw(8, seed=update_idx)
        pooled.update(y_pred, y_true)
    assert abs(float(pooled.compute()) - 1.0) < 0.05


def test_spread_skill_ratio_stateful_returns_per_lead_time_and_is_mean_of_ratios():
    """Lock in the mean-of-ratios aggregation convention across update() calls.

    Two batches with deliberately different per-batch SSRs are streamed through
    ``update()``. The stateful ``compute()`` with ``reduce_all=False`` should:
      1. expose a per-lead-time vector (shape (T, C)), and
      2. equal the arithmetic mean of the per-batch SSRs (mean of per-sample
         ``spread/skill``) rather than the macroscopic ratio of pooled second
         moments.

    If somebody silently switches to a macroscopic ratio in the future, this
    test fails and the behaviour change is caught.
    """
    B, S1, S2, C, M = 4, 16, 16, 1, 32
    sigma_t = torch.tensor([1.0, 1.0, 1.0])  # spread fixed across t
    T = sigma_t.shape[0]
    correction = float(((M + 1) / M) ** 0.5)

    # Batch A: skill=1, spread=1   -> SSR ~= 1 * correction at every t
    pred_a, true_a = _controlled_ssr_batch(
        sigma_t, bias=1.0, B=B, S1=S1, S2=S2, C=C, M=M, seed=0
    )
    # Batch B: skill=0.1, spread=1 -> SSR ~= 10 * correction at every t
    pred_b, true_b = _controlled_ssr_batch(
        sigma_t, bias=0.1, B=B, S1=S1, S2=S2, C=C, M=M, seed=1
    )

    metric = SpreadSkillRatio(reduce_all=False)
    metric.update(pred_a, true_a)
    metric.update(pred_b, true_b)
    value = metric.compute()  # (T, C)

    assert value.shape == (T, C), value.shape

    expected_mean_of_ratios = 0.5 * (1.0 + 10.0) * correction  # ~5.5 * correction
    expected_macroscopic = correction * (
        ((sigma_t[0] ** 2 + sigma_t[0] ** 2) / 2).sqrt()
        / ((1.0**2 + 0.1**2) / 2) ** 0.5
    )  # ~= correction / sqrt(0.505) ~= 1.41 * correction

    # Must match mean-of-ratios, must NOT match macroscopic ratio.
    assert torch.allclose(
        value, torch.full_like(value, expected_mean_of_ratios), rtol=5e-2
    ), value
    assert not torch.allclose(
        value, torch.full_like(value, float(expected_macroscopic)), rtol=2e-1
    ), value


def test_ensemble_spread_matches_lola_correction():
    # Shape: (B=1, T=1, S=1, C=1, M=2)
    # Members: [0, 2]
    # unbiased var = 2, sqrt(var)=sqrt(2)
    # corrected spread = sqrt(2) * sqrt((M+1)/M) = sqrt(3)
    y_pred = torch.tensor([[[[[0.0, 2.0]]]]])
    y_true = torch.tensor([[[[0.0]]]])

    value = EnsembleSpread()(y_pred, y_true)
    assert torch.allclose(value, torch.tensor(3.0**0.5), atol=1e-6)


def test_ensemble_spread_can_be_uncorrected():
    y_pred = torch.tensor([[[[[0.0, 2.0]]]]])
    y_true = torch.tensor([[[[0.0]]]])

    value = EnsembleSpread(corrected=False)(y_pred, y_true)
    assert torch.allclose(value, torch.tensor(2.0**0.5), atol=1e-6)


def test_ensemble_spread_requires_multiple_ensemble_members():
    y_pred = torch.ones((1, 1, 1, 1, 1))
    y_true = torch.ones((1, 1, 1, 1))

    with pytest.raises(ValueError, match="at least 2 ensemble members"):
        EnsembleSpread()(y_pred, y_true)


def test_ensemble_spread_accepts_base_metric_kwargs():
    y_pred = torch.tensor([[[[[0.0, 2.0]]], [[[0.0, 4.0]]]]])
    y_true = torch.zeros((1, 2, 1, 1))

    value = EnsembleSpread(score_dims="temporal", reduce_all=False).score(
        y_pred, y_true
    )

    expected = torch.tensor([[[7.5**0.5]]])
    assert torch.allclose(value, expected, atol=1e-6)


def test_ensemble_skill_is_rmse_of_ensemble_mean():
    # Shape: (B=1, T=1, S=1, C=1, M=2)
    # Members: [0, 2], truth: [0]
    # mean=1 -> rmse = 1
    y_pred = torch.tensor([[[[[0.0, 2.0]]]]])
    y_true = torch.tensor([[[[0.0]]]])

    value = EnsembleSkill()(y_pred, y_true)
    assert torch.allclose(value, torch.tensor(1.0), atol=1e-6)


def test_ensemble_skill_matches_deterministic_rmse_on_ensemble_predictions():
    torch.manual_seed(0)
    y_pred = torch.randn((2, 3, 4, 5, 2, 7))
    y_true = torch.randn((2, 3, 4, 5, 2))

    skill = EnsembleSkill()(y_pred, y_true)
    rmse = RMSE()(y_pred, y_true)

    assert torch.allclose(skill, rmse, atol=1e-6)


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


def test_almost_fair_energy_identical_members_is_zero_spread():
    y_pred = torch.ones((2, 3, 4, 4, 5, 6))  # (B,T,S,S,C,M)
    y_true = torch.ones((2, 3, 4, 4, 5))
    metric = AlmostFairEnergyScore(a_f=0.95)
    metric.update(y_pred, y_true)
    # identical members & equal to truth => score 0
    torch.testing.assert_close(metric.compute(), torch.tensor(0.0), atol=1e-5, rtol=0)


def test_almost_fair_energy_naive_endpoint_matches_energyscore():
    """a_f->0 must equal the existing naive EnergyScore (joint vector_dims)."""
    torch.manual_seed(0)
    y_pred = torch.randn((2, 3, 4, 4, 5, 6))
    y_true = torch.randn((2, 3, 4, 4, 5))
    af0 = AlmostFairEnergyScore(a_f=1e-9, vector_dims="spatial_temporal_channels")
    naive = EnergyScore(alpha=1.0, vector_dims="spatial_temporal_channels")
    af0.update(y_pred, y_true)
    naive.update(y_pred, y_true)
    torch.testing.assert_close(af0.compute(), naive.compute(), atol=1e-4, rtol=1e-3)


def test_almost_fair_energy_m1_raises():
    y_pred = torch.ones((1, 1, 1, 1, 1, 1))  # M=1
    y_true = torch.ones((1, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="at least 2 ensemble members"):
        AlmostFairEnergyScore(a_f=0.95).score(y_pred, y_true)


@pytest.mark.parametrize("a_f", [0.95, 1.0])
def test_almost_fair_energy_metric_matches_loss_helper(a_f):
    """The validation metric must equal the training loss it selects lambda on.

    ``AlmostFairEnergyScore`` (the eval/val metric, BTSCM layout) and
    ``_energy_score`` (the DriftingProcessor loss term, (B, K, D) layout) are
    two views of the same object. With ``vector_dims="spatial_temporal_channels"``
    the metric vectorises the whole field, exactly as the loss flattens it, so
    the per-sample score — and its batch mean — must agree to machine precision.
    This pins the roadmap's "metric mirrors the loss" assumption.
    """
    torch.manual_seed(0)
    b, t, s, c, m = 3, 2, 4, 2, 6
    y_pred = torch.randn(b, t, s, s, c, m)
    y_true = torch.randn(b, t, s, s, c)

    # Loss-helper view: members -> K axis, the whole field flattened -> D.
    # Euclidean distance is permutation-invariant over D, so the exact flatten
    # order need not match the metric's internal vectorisation.
    d = t * s * s * c
    gen_b = y_pred.permute(0, 5, 1, 2, 3, 4).reshape(b, m, d)
    pos_b = y_true.reshape(b, 1, d)
    loss_val = _energy_score(gen_b, pos_b, a_f=a_f)

    metric = AlmostFairEnergyScore(a_f=a_f, vector_dims="spatial_temporal_channels")
    metric.update(y_pred, y_true)

    torch.testing.assert_close(metric.compute(), loss_val, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# LogScore -- per-site marginal logarithmic score (column 1 of the suite).
# The marginal (diagonal) Gaussian ignorance score: the univariate companion to
# the multivariate DawidSebastianiScore (column 2). Sensitive to the density at
# the realised truth per site, so it complements the CDF-oriented CRPS.
# ---------------------------------------------------------------------------


def test_log_score_manual_value():
    # One sample/timestep/point/channel; two members {0, 2}; truth = 1.
    y_pred = torch.tensor([[[[[0.0, 2.0]]]]])  # (1, 1, 1, 1, 2)
    y_true = torch.tensor([[[[1.0]]]])  # (1, 1, 1, 1)

    score = LogScore(reduce_all=False).score(y_pred, y_true)

    # mean = 1, unbiased var = ((0-1)^2 + (2-1)^2) / (2-1) = 2
    # log-score = 0.5*log(2*pi*var) + 0.5*(y-mean)^2/var = 0.5*log(4*pi) + 0
    expected = 0.5 * math.log(2.0 * math.pi * 2.0)
    assert torch.allclose(score, torch.tensor([[[expected]]]), atol=1e-6)


def test_log_score_penalizes_overdispersion_when_mean_is_correct():
    # When the mean already matches the truth the (y-mean)^2/var term is ~0 and
    # the score is dominated by 0.5*log(2*pi*var): a tighter ensemble is rewarded
    # and an over-dispersed one penalised.
    torch.manual_seed(0)
    truth = torch.zeros((1, 1, 4096, 1))
    tight = torch.randn((1, 1, 4096, 1, 32)) * 0.1
    wide = torch.randn((1, 1, 4096, 1, 32)) * 1.0

    s_tight = LogScore().score(tight, truth).mean()
    s_wide = LogScore().score(wide, truth).mean()

    assert s_tight < s_wide


def test_log_score_tail_miss_is_penalized():
    # A too-narrow predictive that misses a genuine tail observation is punished
    # hard by the density-at-truth term -- far more than a calibrated ensemble.
    # This is the column-1 tail read the log-score earns its place for.
    torch.manual_seed(0)
    m = 64
    truth = torch.full((1, 1, 2048, 1), 3.0)  # a tail value under N(0, 1)
    calibrated = torch.randn((1, 1, 2048, 1, m)) * 1.0
    narrow = torch.randn((1, 1, 2048, 1, m)) * 0.2

    s_cal = LogScore().score(calibrated, truth).mean()
    s_narrow = LogScore().score(narrow, truth).mean()

    assert s_narrow > s_cal


def test_log_score_var_floor_guards_collapsed_ensemble():
    # All members identical and equal to truth -> sample var is exactly 0.
    # The hard var_floor keeps the score finite (no -inf / nan).
    y_pred = torch.ones((1, 1, 4, 1, 8))
    y_true = torch.ones((1, 1, 4, 1))

    score = LogScore(var_floor=1e-6).score(y_pred, y_true)

    assert torch.isfinite(score).all()
    # (y - mean) = 0, so score = 0.5*log(2*pi*var_floor) for every point.
    expected = 0.5 * math.log(2.0 * math.pi * 1e-6)
    assert torch.allclose(score, torch.full_like(score, expected), atol=1e-5)


def test_log_score_rejects_nonpositive_var_floor():
    with pytest.raises(ValueError, match="var_floor must be positive"):
        LogScore(var_floor=0.0)


# ---------------------------------------------------------------------------
# DawidSebastianiScore -- multivariate (full-covariance) Gaussian ignorance
# score (column 2 of the suite). Correlation-sensitive via Sigma^-1; the
# covariance is Ledoit-Wolf shrinkage-regularised so it stays invertible when
# D > M-1.
# ---------------------------------------------------------------------------


def _gaussian_nll_reference(cov, members_dm, obs_vec, d):
    """Reference multivariate Gaussian NLL 0.5*(D*log2pi + logdet + mahalanobis)."""
    delta = obs_vec - members_dm.mean(dim=-1)
    _sign, logabsdet = torch.linalg.slogdet(cov)
    maha = float(delta @ torch.linalg.solve(cov, delta))
    return 0.5 * (d * math.log(2.0 * math.pi) + float(logabsdet) + maha)


def test_lw_shrinkage_matches_sklearn():
    # The batched Ledoit-Wolf helper must reproduce sklearn's reference exactly
    # (same 1/M empirical covariance, same optimal intensity, same shrunk cov).
    ledoit_wolf = pytest.importorskip("sklearn.covariance").ledoit_wolf

    torch.manual_seed(0)
    n_samples, d = 40, 12
    x = torch.randn(n_samples, d, dtype=torch.float64)  # (samples, features)
    sk_cov, sk_gamma = ledoit_wolf(x.numpy())

    members = x.transpose(0, 1).reshape(1, d, n_samples)  # (..., D, M)
    cov, _mean, gamma = _shrunk_covariance(
        members, method="ledoit_wolf", intensity=None, jitter=0.0
    )

    assert abs(float(gamma.reshape(()).item()) - float(sk_gamma)) < 1e-10
    torch.testing.assert_close(
        cov.reshape(d, d), torch.from_numpy(sk_cov), atol=1e-10, rtol=1e-10
    )


def test_shrunk_covariance_fixed_requires_intensity():
    # method="fixed" with no intensity is a caller bug the class-level validation
    # normally catches before this helper runs; this locks the helper's own guard.
    members = torch.randn(2, 3, 5, dtype=torch.float64)  # (..., D, M)
    with pytest.raises(ValueError, match="intensity"):
        _shrunk_covariance(members, method="fixed", intensity=None, jitter=0.0)


def test_memory_intensive_flag_on_variogram_and_dss_family():
    # Class-attribute flag, not a hand-maintained name list: every VariogramScore
    # subclass inherits it, DSS sets it directly, cheap metrics stay False.
    for cls in (
        VariogramScore,
        SpatialVariogramScore,
        TemporalVariogramScore,
        SpatiotemporalVariogramScore,
        DawidSebastianiScore,
    ):
        assert cls.memory_intensive is True
    for cls in (CRPS, LogScore, TemporalAutocorrelationMatch):
        assert cls.memory_intensive is False
    assert BaseMetric.memory_intensive is False


def test_dss_matches_gaussian_nll_reference():
    # The full metric must equal the multivariate Gaussian negative log-density
    # with Sigma the Ledoit-Wolf shrunk covariance.
    ledoit_wolf = pytest.importorskip("sklearn.covariance").ledoit_wolf

    torch.manual_seed(1)
    n_samples, d = 30, 5
    members = torch.randn(1, 1, d, 1, n_samples, dtype=torch.float64)  # B T S C M
    truth = torch.randn(1, 1, d, 1, dtype=torch.float64)  # B T S C

    x = members[0, 0, :, 0, :]  # (D, M)
    sk_cov = torch.from_numpy(ledoit_wolf(x.transpose(0, 1).numpy())[0])
    expected = _gaussian_nll_reference(sk_cov, x, truth[0, 0, :, 0], d)

    metric = DawidSebastianiScore(vector_dims="spatial", jitter=0.0, reduce_all=False)
    score = metric.score(members, truth)  # (B, T, C) = (1, 1, 1)
    assert torch.allclose(
        score, torch.tensor([[[expected]]], dtype=torch.float64), atol=1e-8
    )


def test_dss_flags_wrong_correlation_but_log_score_does_not():
    # Correct marginals, wrong correlation: DSS (column 2) flags it; LogScore
    # (column 1, marginal) is blind to it. This is the column-exclusivity the
    # suite is built on.
    torch.manual_seed(0)
    n_frames, d, m, rho = 256, 8, 64, 0.8

    corr = torch.full((d, d), rho, dtype=torch.float64)
    corr.fill_diagonal_(1.0)
    chol = torch.linalg.cholesky(corr)

    def correlated(n, k):
        z = torch.randn(n, k, d, dtype=torch.float64)
        return z @ chol.transpose(0, 1)  # (n, k, d), unit marginals, corr rho

    def to_pred(x):  # (n, m, d) -> (n, 1, d, 1, m)
        return x.permute(0, 2, 1).reshape(n_frames, 1, d, 1, m)

    truth = torch.randn(n_frames, d, dtype=torch.float64) @ chol.transpose(0, 1)
    y_true = truth.reshape(n_frames, 1, d, 1)
    correct = correlated(n_frames, m)  # right correlation
    whitened = torch.randn(n_frames, m, d, dtype=torch.float64)  # unit var, no corr

    s_correct = (
        DawidSebastianiScore(vector_dims="spatial")
        .score(to_pred(correct), y_true)
        .mean()
    )
    s_whitened = (
        DawidSebastianiScore(vector_dims="spatial")
        .score(to_pred(whitened), y_true)
        .mean()
    )
    # DSS rewards the correctly-correlated ensemble.
    assert s_correct < s_whitened

    ls_correct = LogScore().score(to_pred(correct), y_true).mean()
    ls_whitened = LogScore().score(to_pred(whitened), y_true).mean()
    # LogScore is marginal: the two unit-variance ensembles are ~indistinguishable,
    # far below the DSS separation.
    assert abs(float(ls_correct - ls_whitened)) < 0.1 * abs(
        float(s_whitened - s_correct)
    )


def test_dss_collapsed_ensemble_is_finite():
    # Members collapse to the truth -> sample covariance is exactly zero; the
    # diagonal jitter keeps the Cholesky/logdet finite.
    y_pred = torch.ones((1, 1, 4, 1, 8))
    y_true = torch.ones((1, 1, 4, 1))
    score = DawidSebastianiScore(vector_dims="spatial").score(y_pred, y_true)
    assert torch.isfinite(score).all()


def test_dss_accepts_integer_fields():
    # Integer-valued fields must not crash or truncate: the score is computed in
    # float64 and returned as a float (regression -- the previous cast to
    # y_true.dtype truncated the log-density to int / crashed the base reduction).
    torch.manual_seed(0)
    y_pred = torch.randint(0, 5, (4, 1, 6, 1, 16))
    y_true = torch.randint(0, 5, (4, 1, 6, 1))
    score = DawidSebastianiScore(vector_dims="spatial").score(y_pred, y_true)
    assert score.is_floating_point()
    assert torch.isfinite(score).all()


def test_dss_fixed_shrinkage_reduces_to_sample_cov():
    # shrinkage="fixed" with intensity 0 and no jitter == raw sample-covariance NLL.
    torch.manual_seed(2)
    n_samples, d = 50, 6
    members = torch.randn(1, 1, d, 1, n_samples, dtype=torch.float64)
    truth = torch.randn(1, 1, d, 1, dtype=torch.float64)

    x = members[0, 0, :, 0, :]
    xc = x - x.mean(-1, keepdim=True)
    cov = xc @ xc.transpose(0, 1) / n_samples
    expected = _gaussian_nll_reference(cov, x, truth[0, 0, :, 0], d)

    metric = DawidSebastianiScore(
        vector_dims="spatial",
        shrinkage="fixed",
        shrinkage_intensity=0.0,
        jitter=0.0,
        reduce_all=False,
    )
    score = metric.score(members, truth)
    assert torch.allclose(
        score, torch.tensor([[[expected]]], dtype=torch.float64), atol=1e-8
    )


def test_dss_rejects_bad_config(validation_error):
    with validation_error(ValueError, match="vector_dims"):
        DawidSebastianiScore(vector_dims="bogus")  # pyright: ignore[reportArgumentType]
    with validation_error(ValueError, match="shrinkage must be"):
        DawidSebastianiScore(shrinkage="bogus")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="shrinkage_intensity"):
        DawidSebastianiScore(shrinkage="fixed")  # intensity required
    with pytest.raises(ValueError, match="shrinkage_intensity"):
        DawidSebastianiScore(shrinkage="fixed", shrinkage_intensity=1.5)
    with pytest.raises(ValueError, match="jitter must be non-negative"):
        DawidSebastianiScore(jitter=-1.0)
    with pytest.raises(ValueError, match="max_pair_elements must be positive"):
        DawidSebastianiScore(max_pair_elements=0)


def test_dss_score_is_invariant_to_pair_block_chunking():
    """Capping the per-frame covariance block changes peak memory, never the
    score -- DSS's analogue of VariogramScore's chunking-invariance test.

    Unlike VariogramScore, DSS previously had no blocking at all: the full
    ``(B, T, C, D, D)`` covariance was built in one allocation, which is why it
    was the first metric to OOM in production. ``_shrunk_covariance``, the
    Cholesky factorisation and the triangular solve all broadcast over the
    leading (batch, lead-time, channel) axes independently -- no cross-frame
    coupling -- so blocking frame-by-frame is exact.
    """
    torch.manual_seed(5)
    # (B=2, T=3, S=5, C=4): D = 5 sites, M = 6 members. Leading extents are all
    # DISTINCT and (2, 3, 4) is not palindromic -- see the sibling variogram
    # test for why a symmetric shape hides block-reassembly order bugs.
    y_pred = torch.randn(2, 3, 5, 4, 6, dtype=torch.float64)
    y_true = torch.randn(2, 3, 5, 4, dtype=torch.float64)

    kwargs = {"vector_dims": "spatial", "jitter": 1e-6, "reduce_all": False}
    unchunked = DawidSebastianiScore(**kwargs).score(y_pred, y_true)
    # D=5 -> one (b, t, c) frame's covariance costs 5*5=25 elements; one batch
    # row (bundling T=3, C=4) would cost 300. Capping at 25 forces one frame per
    # block, splitting across T and C, not just B.
    chunked = DawidSebastianiScore(max_pair_elements=5 * 5, **kwargs).score(
        y_pred, y_true
    )

    assert chunked.shape == unchunked.shape
    torch.testing.assert_close(chunked, unchunked, rtol=1e-10, atol=1e-10)


def test_ad64_shape_variogram_and_dss_block_sizes():
    """Regression pin for the eval shape that OOM'd (jobs 6038001, 6042592):

    64x64 grid (S=4096 sites), T=4 lead steps, C=4 channels, M=8 members. Before
    this fix, the coarsest achievable block bundled T and C into the leading
    axis; after it, the per-frame (b, t, c) block is the coarsest unit. Pins
    the element counts so a future change to the blocking granularity is
    caught here rather than rediscovered as an OOM.
    """
    n_sites, n_leads, n_channels, n_members = 4096, 4, 4, 8

    # variogram_spatial: D = n_sites; per-frame block is exactly the default cap.
    variogram_spatial_frame = n_sites * n_sites * n_members
    variogram_spatial_old_row = n_leads * n_channels * variogram_spatial_frame
    assert variogram_spatial_frame == DEFAULT_MAX_PAIR_ELEMENTS
    assert variogram_spatial_old_row == 16 * DEFAULT_MAX_PAIR_ELEMENTS

    # dss (vector_dims="spatial", float64): D = n_sites; per-frame covariance is
    # comfortably under the cap even though the old, unblocked full allocation
    # (B * T * C * D * D) was not.
    dss_frame = n_sites * n_sites
    assert dss_frame < DEFAULT_MAX_PAIR_ELEMENTS

    # variogram_spatial's per-frame cost sits exactly at the cap, so blocking
    # floors at one frame -- it cannot shrink further without changing the
    # score's definition. DSS's per-frame covariance is 8x smaller, so several
    # frames share a block; either way, no block ever needs the old B-row size.
    assert rows_per_pair_block(variogram_spatial_frame, DEFAULT_MAX_PAIR_ELEMENTS) == 1
    dss_rows_per_block = rows_per_pair_block(dss_frame, DEFAULT_MAX_PAIR_ELEMENTS)
    assert dss_rows_per_block == DEFAULT_MAX_PAIR_ELEMENTS // dss_frame
    assert dss_rows_per_block * dss_frame <= DEFAULT_MAX_PAIR_ELEMENTS
