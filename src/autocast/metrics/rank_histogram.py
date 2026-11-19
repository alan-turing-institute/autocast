"""Multivariate rank-histogram diagnostics (research piece sec. 6.6).

The joint-structure companions to :class:`~autocast.metrics.coverage.Coverage`
and :class:`~autocast.metrics.collection_coverage.CollectionCoverage`: where
coverage asks whether the truth falls inside an interval, a rank histogram asks
*where in the ensemble the truth ranks* once the multivariate field is mapped to
a single number by a pre-rank function. Binning that rank over many forecast
cases gives a histogram whose shape reveals miscalibration.

Two pre-rank functions, following Thorarinsdottir, Scheuerer & Heinz (2016),
"Assessing the calibration of high-dimensional ensemble forecasts using rank
histograms", J. Comput. Graph. Stat. 25(1):

* **Average rank** -- pre-rank = the mean over components of the univariate rank
  of each pool member. Reads exactly like a univariate rank histogram: U-shaped =
  under-dispersed, dome (inverted-U) = over-dispersed, flat = calibrated.
* **Band depth** -- pre-rank = the (simplified, modified) functional band depth,
  a centrality score ``(1/D) sum_d (M+1 - r_{i,d})(r_{i,d} - 1)``. The truth's
  band-depth rank piles up at *low* ranks under under-dispersion (the truth is an
  outlier) and at *high* (central) ranks under over-dispersion (the truth looks
  unusually central). This is the over-dispersion tell the average-rank histogram
  blurs into a symmetric dome.

The observation rank uses the standard randomized rank (uniform under
calibration): ``rank = 1 + #(members below) + U{0, #(members tied)}``. Twin for
the wrapper/CSV pattern: :class:`~autocast.metrics.coverage.MultiCoverage`.
"""

from __future__ import annotations

import abc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torchmetrics import Metric

from autocast.types import ArrayLike, Tensor, TensorRealBTSC, TensorRealBTSCM

# Ceiling on the pairwise block the dependence pre-rank materialises at one time.
# Written naively the variogram sum builds the whole (F, D, D, M+1) tensor: at
# D = 4096 sites (a 64x64 field) with 32 members that is 4.1 GiB per FRAME, so a
# 64-frame batch asks for 264 GiB and dies on any device (measured, 2026-08-01).
# Peak use is about one block, since the weighting is applied in place.
PAIRWISE_BLOCK_BYTES: int = 1 << 30  # 1 GiB


def _weighted_variogram(
    fields: Tensor, weights: Tensor, p: float, block_bytes: int
) -> Tensor:
    r"""``sum_{ij} w_ij |x_i - x_j|^p`` for each field, in bounded memory.

    ``fields`` is ``(N, D)`` -- one row per field to be scored -- and ``weights``
    is the ``(D, D)`` pair-weight matrix. Returns ``(N,)``.

    The sum is taken over the full ``D x D`` matrix, but only a
    ``(n_chunk, r_chunk, D)`` slab of it exists at any moment, sized to
    ``block_bytes``. The arithmetic is otherwise identical to the unchunked form;
    only the summation order changes, which is immaterial in the float64 the
    caller works in.
    """
    n, d = fields.shape
    row_bytes = max(1, d * fields.element_size())
    rows = max(1, min(d, block_bytes // row_bytes))
    chunk = max(1, min(n, block_bytes // (rows * row_bytes)))

    out = fields.new_zeros(n)
    for start in range(0, n, chunk):
        block_fields = fields[start : start + chunk]  # (n_chunk, D)
        acc = fields.new_zeros(block_fields.shape[0])
        for row in range(0, d, rows):
            # (n_chunk, r_chunk, D); abs_/pow_/mul_ in place, the subtraction
            # already allocated a fresh tensor.
            pairwise = (
                block_fields[:, row : row + rows, None] - block_fields[:, None, :]
            )
            pairwise = pairwise.abs_().pow_(p)
            pairwise.mul_(weights[None, row : row + rows, :])
            acc = acc + pairwise.sum(dim=(1, 2))
        out[start : start + chunk] = acc
    return out


def grid_inverse_distance_weights(
    height: int,
    width: int,
    *,
    periodic: bool = False,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Inverse-distance pair weights for a 2-D grid flattened in row-major order.

    :class:`DependenceRankHistogram`'s default weighting is ``w_ij = 1/|i-j|`` on
    the FLATTENED index, which its own docstring flags as exact only for a 1-D
    field. On an ``H x W`` grid that default makes horizontally-adjacent sites
    neighbours (``|i-j| = 1``) while vertically-adjacent sites, which are ``W``
    apart in the flattened order, are down-weighted by a factor of ``W`` -- 8x on
    an 8x8 field. The probe then reads along rows far more strongly than down
    columns, so an anisotropic correlation error is mis-sized.

    This builds the geometry-aware matrix that removes the anisotropy:
    ``w_ij = 1 / d_ij`` with ``d_ij`` the Euclidean distance between sites on the
    grid, so the four nearest neighbours all weigh 1 and diagonal neighbours
    ``1/sqrt(2)``. As with the default, the diagonal is zero and the weight is
    capped at 1 so no pair is up-weighted above adjacency.

    Row-major is not a convention chosen here, it is the one already in force:
    :meth:`PooledMultivariateRankHistogram._group_frames` collapses the spatial axes
    with ``rearrange(y, "b t ... c m -> (b t c) (...) m")``, and einops flattens
    ``(...)`` with the last axis varying fastest, so site ``(row, col)`` lands at
    ``row * width + col``. Weights built for any other layout would be worse than
    the default, so a caller on a differently-packed field must not use this.

    Parameters
    ----------
    height, width
        Grid extent. ``height * width`` must equal the field's site count ``D``.
    periodic
        Wrap distances at the boundary. Left OFF by default: a field with
        reflecting (Neumann) boundaries has no wrap-around adjacency, and
        asserting one would mis-weight the boundary pairs. Use ``True`` only
        for a field whose generator is itself periodic on both grid axes; a
        1-D periodic *ring* (e.g. Lorenz-96) is not a 2-D grid and should use
        :func:`ring_inverse_distance_weights` instead.
    dtype
        Working dtype; the pre-rank casts to the pool's dtype on use.

    Returns:
    Tensor
        ``(D, D)`` non-negative weights with a zero diagonal, ``D = height*width``.
    """
    if height < 1 or width < 1:
        msg = f"height and width must be positive; got {height} x {width}"
        raise ValueError(msg)
    d = height * width
    index = torch.arange(d)
    rows = torch.div(index, width, rounding_mode="floor")
    cols = index % width
    d_row = (rows[:, None] - rows[None, :]).abs()
    d_col = (cols[:, None] - cols[None, :]).abs()
    if periodic:
        d_row = torch.minimum(d_row, height - d_row)
        d_col = torch.minimum(d_col, width - d_col)
    dist = (d_row.to(dtype).square() + d_col.to(dtype).square()).sqrt()
    return (1.0 / dist.clamp_min(1.0)) * (dist > 0)


def ring_inverse_distance_weights(
    n_sites: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Inverse-distance pair weights for a 1-D PERIODIC ring (cyclic lattice).

    :class:`DependenceRankHistogram`'s flattened-index default, ``w_ij =
    1/|i-j|``, is exact for a 1-D field only if the field's two ends are not
    themselves adjacent. A cyclic lattice -- Lorenz-96's 40 sites, wired site
    ``i`` to sites ``i-1`` and ``i+1`` mod ``n_sites`` by construction -- violates
    that: the flattened default treats site ``0`` and site ``n_sites - 1`` as the
    single MOST distant pair (weight ``1/(n_sites - 1)``), when physically they
    are neighbours.

    This builds the wrap-aware 1-D matrix that fixes it: ``w_ij = 1 / d_ij`` with
    ``d_ij = min(|i-j|, n_sites - |i-j|)`` the distance around the ring, zero
    diagonal, capped at 1 exactly as the flattened default and
    :func:`grid_inverse_distance_weights` are.

    Why this differs from :func:`grid_inverse_distance_weights`'s ``periodic``
    default
    -------------------------------------------------------------------------
    ``grid_inverse_distance_weights`` leaves wrapping OFF by default because its
    2-D fields carry reflecting (Neumann) boundaries -- there is no physical
    adjacency between opposite edges to assert. A periodic ring is the opposite
    case: the wrap-around IS the field's actual geometry, not an assumption
    layered on top of it, so leaving it off would be the wrong default here, not
    the safe one. The two functions do not disagree about when to wrap; they
    agree that the weighting must match the generator's real boundary condition,
    and Lorenz-96's is periodic where the 2-D campaign's is not.

    Parameters
    ----------
    n_sites
        Number of sites on the ring. Must be positive.
    dtype
        Working dtype; the pre-rank casts to the pool's dtype on use.

    Returns:
    Tensor
        ``(n_sites, n_sites)`` non-negative weights with a zero diagonal.
    """
    if n_sites < 1:
        msg = f"n_sites must be positive; got {n_sites}"
        raise ValueError(msg)
    index = torch.arange(n_sites)
    d_linear = (index[:, None] - index[None, :]).abs()
    dist = torch.minimum(d_linear, n_sites - d_linear).to(dtype)
    return (1.0 / dist.clamp_min(1.0)) * (dist > 0)


def _pool_ranks(pool: Tensor) -> Tensor:
    """Rank each pool element in ``1..M+1`` along the last axis, per ``(F, D)`` row.

    ``pool`` is ``(F, D, M+1)``; returns ranks ``(F, D, M+1)`` (ascending), using
    the AVERAGE-RANK convention: tied values share the mean of the positions they
    span, ``1 + #below + (#equal - 1) / 2``.

    ⛔ **This used argsort-of-argsort, which resolves ties BY POSITION**, on the
    stated grounds that "the inputs are continuous, so exact ties are
    measure-zero". That holds for a healthy model and fails for exactly the case
    a rank histogram is used to diagnose: a COLLAPSED ensemble, where members are
    bit-identical. Measured on a fully tied pool of five, argsort returns
    ``[1, 2, 3, 4, 5]`` -- pure array order -- so the observation's rank was
    decided by where it sat in the concatenation. That is why
    ``coverage.MultivariateRankHistogram`` (which builds the pool observation
    FIRST) and ``PooledMultivariateRankHistogram`` (observation LAST) reported
    opposite ends of the histogram for one perfect forecast: 0 members below
    versus all M below. Neither was right; both were reading array order.

    Average ranks are order-independent, so the two agree by construction.
    ⚠ On continuous input the two conventions are IDENTICAL -- verified, max
    absolute difference 0.0 over an ``(8, 512, 9)`` float64 draw -- so no result
    computed with the previous convention changes. The cost is 2.09 ms against
    0.86 ms at a realistic ``(32, 4096, 9)`` shape; paid deliberately, because a
    diagnostic that silently depends on argument order is worth more than 1.2 ms.
    """
    a = pool.unsqueeze(-1)
    b = pool.unsqueeze(-2)
    below = (b < a).sum(dim=-1).to(pool.dtype)
    equal = (b == a).sum(dim=-1).to(pool.dtype)
    return 1.0 + below + (equal - 1.0) / 2.0


class _RankHistogramBase(Metric):
    """Shared accumulation for multivariate rank histograms.

    Subclasses define :meth:`_pre_rank`, mapping a pool of ``M + 1`` vectors
    (the ensemble members plus the observation, stacked last) to one scalar
    pre-rank each. The base flattens ``(B, T, *spatial, C[, M])`` into one
    multivariate vector per ``(B, T)`` frame, computes the observation's
    randomized rank among the members by pre-rank, and accumulates the rank
    histogram over all frames and batches.
    """

    name: str = "rank_histogram"

    # A rank histogram needs at least two members for the observation's rank among
    # them to carry any information.
    MIN_MEMBERS: int = 2
    # Input is (B, T, S, C, M) -- batch, time, space, channel, member.
    ENSEMBLE_NDIM: int = 5

    # torchmetrics installs these via add_state, which a type checker cannot see.
    counts: Tensor
    rank_values: list[Tensor]

    def __init__(
        self,
        n_members: int,
        record_ranks: bool = False,
        dist_sync_on_step: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        if n_members < self.MIN_MEMBERS:
            msg = f"rank histogram requires n_members >= 2, got {n_members}"
            raise ValueError(msg)
        self.n_members = n_members
        self.record_ranks = record_ranks
        # ⛔ The randomized rank that breaks TIES draws from ``torch.rand``. With
        # no generator that is the GLOBAL RNG, so the reported index depends on
        # whatever was drawn earlier in the process and the same inputs can give
        # different answers -- measured 2026-09-14 on the Task 14 block sweep,
        # where two identical runs moved ``trajectory_shape_reliability`` on 2 of
        # 12 arms while every other column stayed bit-identical. This is D123's
        # defect in a second place.
        #
        # ⚠ The default stays ``None`` DELIBERATELY, so every number already
        # published from this metric keeps the behaviour it was produced under
        # (D123's own precedent for ``score_recovered.py``). Pass a seeded
        # generator to make a NEW measurement reproducible.
        self._generator = generator
        # One count per rank bin 1..M+1.
        self.add_state(
            "counts",
            default=torch.zeros(n_members + 1, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        # The pooled histogram alone cannot carry an error bar: frames within a
        # trajectory are dependent, so a margin has to come from a
        # trajectory-block bootstrap over the individual ranks. Opt in to keep
        # them (one int per verification instance, in loader order) -- the same
        # per-frame-plus-traj_id pattern joint_coverage.py uses for ``d2``.
        self.add_state("rank_values", default=[], dist_reduce_fx="cat")

    @abc.abstractmethod
    def _pre_rank(self, pool: Tensor) -> Tensor:
        """Map a pool ``(F, D, M+1)`` to pre-ranks ``(F, M+1)`` (last index = obs)."""

    def _check_input(
        self, y_pred: ArrayLike, y_true: ArrayLike
    ) -> tuple[Tensor, Tensor]:
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        if not isinstance(y_pred, Tensor) or not isinstance(y_true, Tensor):
            msg = "y_pred and y_true must be Tensors or np.ndarrays"
            raise TypeError(msg)
        if y_pred.ndim < self.ENSEMBLE_NDIM:
            msg = (
                f"y_pred has {y_pred.ndim} dimensions, should be at least 5 "
                "(B, T, S, C, M)"
            )
            raise ValueError(msg)
        if y_pred.shape[:-1] != y_true.shape:
            msg = (
                "y_pred and y_true must match except for the ensemble axis; got "
                f"{y_pred.shape} and {y_true.shape}"
            )
            raise ValueError(msg)
        if y_pred.shape[-1] != self.n_members:
            msg = (
                f"y_pred ensemble axis {y_pred.shape[-1]} != n_members {self.n_members}"
            )
            raise ValueError(msg)
        return y_pred, y_true

    def _group_frames(
        self, y_pred_t: Tensor, y_true_t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Group ``(B, T, *spatial, C, M)`` into frame-vectors.

        Returns ``(members, truth)`` with shapes ``(F, D, M)`` and ``(F, D)`` --
        the frame axis ``F`` and the ranked-vector axis ``D``. Default: one vector
        per ``(B, T)`` frame spanning every spatial point *and* channel together.
        Subclasses override this seam alone to change what forms the vector;
        :meth:`_pool` owns the shared float64 cast and observation-stacking.
        """
        b, t = y_true_t.shape[0], y_true_t.shape[1]
        m = self.n_members
        members = y_pred_t.reshape(b * t, -1, m)  # (F=B*T, D, M)
        truth = y_true_t.reshape(b * t, -1)  # (F, D)
        return members, truth

    def _pool(self, y_pred_t: Tensor, y_true_t: Tensor) -> Tensor:
        """Pool grouped frames into ``(F, D, M+1)`` with the observation stacked last.

        Cast to float64 up front so the rank arithmetic is exact and
        integer-valued fields never reach a mean over an integer dtype.
        """
        members, truth = self._group_frames(y_pred_t, y_true_t)
        members = members.to(torch.float64)
        truth = truth.to(torch.float64)
        return torch.cat([members, truth.unsqueeze(-1)], dim=-1)  # (F, D, M+1)

    def update(self, y_pred: TensorRealBTSCM, y_true: TensorRealBTSC) -> None:
        """Accumulate ranks from one batch ``(B, T, *spatial, C, M)``."""
        y_pred_t, y_true_t = self._check_input(y_pred, y_true)
        m = self.n_members

        pool = self._pool(y_pred_t, y_true_t)  # (F, D, M+1)
        n_frames = pool.shape[0]

        pre = self._pre_rank(pool)  # (F, M+1)
        pre_members = pre[:, :m]
        pre_obs = pre[:, m:]  # (F, 1)

        below = (pre_members < pre_obs).sum(dim=1)  # (F,)
        tied = (pre_members == pre_obs).sum(dim=1)  # (F,)
        # Randomized rank in 1..M+1: break ties uniformly (uniform under calibration).
        jitter = (
            torch.rand(n_frames, device=pre.device, generator=self._generator)
            * (tied + 1).double()
        ).floor()
        ranks = (below + 1 + jitter.long()).clamp(min=1, max=m + 1)

        binned = torch.bincount(ranks - 1, minlength=m + 1).to(self.counts.dtype)
        self.counts = self.counts + binned.to(self.counts.device)
        if self.record_ranks:
            self.rank_values.append(ranks.detach().cpu())

    def recorded_ranks(self) -> Tensor:
        """Every observation rank seen so far, ``(frames,)`` in loader order.

        Requires ``record_ranks=True``; the resampling unit for a margin on these
        is the trajectory, so the caller pairs them with a trajectory label per
        frame (:func:`~autocast.benchmarking.rank_scoring.build_traj_ids`).
        """
        if not self.record_ranks:
            msg = "ranks were not kept; construct the metric with record_ranks=True"
            raise RuntimeError(msg)
        if not self.rank_values:
            msg = "No samples were provided to the metric"
            raise RuntimeError(msg)
        return torch.cat(self.rank_values, dim=0)

    def compute_histogram(self) -> np.ndarray:
        """Return the normalized rank histogram, shape ``(M + 1,)``."""
        total = float(self.counts.sum().item())
        if total == 0:
            msg = "No samples were provided to the metric"
            raise RuntimeError(msg)
        return (self.counts / total).detach().cpu().numpy()

    def compute(self) -> Tensor:
        """Scalar miscalibration score: mean absolute deviation from uniform."""
        hist = self.compute_histogram()
        uniform = 1.0 / (self.n_members + 1)
        return torch.tensor(float(np.mean(np.abs(hist - uniform))))

    def reliability_index(self) -> float:
        r"""Candille-Talagrand reliability index :math:`\Delta`.

        The *summed* absolute deviation of the rank histogram from uniformity,

        .. math::
            \Delta = \sum_{k=1}^{M+1} \Bigl| p_k - \tfrac{1}{M+1} \Bigr|,

        with :math:`p_k` the observed frequency in bin :math:`k`. It is ``0`` for a
        perfectly flat (calibrated) histogram and grows with any departure from
        uniform, reaching :math:`2M/(M+1)` when all mass falls in a single bin. Unlike
        :meth:`compute` (the *mean* absolute deviation) this is the *sum*, matching the
        Candille & Talagrand (2005) definition; the two differ only by the ``M+1``
        bin-count factor (:math:`\Delta = (M+1)\,\times`\ :meth:`compute`). Every
        rank-histogram subclass inherits it.

        References:
        Candille, G. and Talagrand, O. (2005). Evaluation of probabilistic prediction
        systems for a scalar variable. Q. J. R. Meteorol. Soc., 131(609), 2131-2150.
        """
        hist = self.compute_histogram()
        uniform = 1.0 / (self.n_members + 1)
        return float(np.sum(np.abs(hist - uniform)))

    def plot(
        self,
        save_path: Path | str | None = None,
        title: str = "Rank histogram",
        save_csv: bool = True,
    ):
        """Plot the rank histogram against the uniform reference and write a CSV."""
        if plt.get_backend().lower() != "agg":
            plt.switch_backend("Agg")
        hist = self.compute_histogram()
        ranks = list(range(1, self.n_members + 2))

        if save_path is not None and save_csv:
            csv_path = Path(save_path).with_suffix(".csv")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"rank": ranks, "frequency": hist}).to_csv(
                csv_path, index=False
            )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(ranks, hist, width=1.0, edgecolor="k", alpha=0.7)
        ax.axhline(1.0 / (self.n_members + 1), color="r", ls="--", label="Uniform")
        ax.set_xlabel("observation rank")
        ax.set_ylabel("frequency")
        ax.set_title(title)
        ax.legend()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return fig


class AverageRankHistogram(_RankHistogramBase):
    """Average-rank multivariate rank histogram.

    Pre-rank of pool element ``i`` is the mean over components ``d`` of its
    univariate rank ``r_{i,d}`` within the pool. The histogram reads like a
    univariate rank histogram: U-shaped under-dispersed, dome over-dispersed,
    flat calibrated.
    """

    name: str = "average_rank_histogram"

    def _pre_rank(self, pool: Tensor) -> Tensor:
        # Per-component rank in 1..M+1 within the pool (over the M+1 axis),
        # averaged over the D components.
        r = _pool_ranks(pool)  # (F, D, M+1) ranks in 1..M+1
        return r.mean(dim=1)  # (F, M+1)


class BandDepthRankHistogram(_RankHistogramBase):
    """Band-depth multivariate rank histogram.

    Pre-rank is the simplified modified band depth (a centrality score),
    ``bd_i = (1/D) sum_d (M+1 - r_{i,d})(r_{i,d} - 1)`` with ``r_{i,d}`` the
    univariate rank in ``1..M+1``. A central observation scores high band depth.
    Under-dispersion (the truth is an outlier) crowds the *low* ranks; over-
    dispersion (the truth looks unusually central) crowds the *high* ranks --
    the directional over-dispersion tell the symmetric average-rank dome blurs.
    """

    name: str = "band_depth_rank_histogram"

    def _pre_rank(self, pool: Tensor) -> Tensor:
        r = _pool_ranks(pool)  # (F, D, M+1) ranks in 1..M+1
        size = pool.shape[-1]  # M+1
        # Centrality score: high when ranks sit near the middle of the pool.
        return ((size - r) * (r - 1.0)).mean(dim=1)  # (F, M+1)


class TrajectoryBandDepthRankHistogram(BandDepthRankHistogram):
    """Band depth over the WHOLE trajectory, one ranked vector per trajectory.

    ⛔ **Not a variant of :class:`BandDepthRankHistogram` for convenience --
    it answers a different question, and the default answers the wrong one for
    a rollout study.** The base class's ``_group_frames`` builds one vector per
    ``(B, T)`` FRAME spanning space, so its histogram asks "is each frame's
    spatial pattern calibrated?" and is blind, by construction, to whether the
    frames HANG TOGETHER: a forecast whose every frame is individually perfect
    but whose frames are mutually incoherent scores identically to one that
    gets the trajectory right.

    Here the ranked vector spans ``T`` and space together, so ``F = B`` and one
    instance IS one trajectory. That is what makes the result a statement about
    trajectory SHAPE -- the calibration counterpart to the lead-correlation
    distance, which measures the same coherence as a distance rather than as a
    rank.

    ⚠ It costs power: the default sees ``B * T`` instances and this sees ``B``,
    so the histogram is ``T`` times sparser at the same batch count and its
    reliability index is correspondingly noisier. Read it with a floor measured
    at the population size actually used, never against zero.

    Reduce with
    :meth:`~autocast.metrics.rank_histogram._RankHistogramBase.reliability_index`
    -- the Candille-Talagrand index. Do not invent a flatness statistic.
    """

    name: str = "trajectory_band_depth_rank_histogram"

    def _group_frames(
        self, y_pred_t: Tensor, y_true_t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Group ``(B, T, *spatial, C, M)`` into ONE vector per trajectory.

        Args:
            y_pred_t: Member trajectories, ``(B, T, *spatial, C, M)``.
            y_true_t: Observations, ``(B, T, *spatial, C)``.

        Returns:
            ``(members, truth)`` of shapes ``(B, T * D, M)`` and ``(B, T * D)``.
        """
        b = y_true_t.shape[0]
        m = self.n_members
        return y_pred_t.reshape(b, -1, m), y_true_t.reshape(b, -1)


class PooledMultivariateRankHistogram(AverageRankHistogram):
    r"""Average-rank multivariate rank histogram, spatial field, channels pooled.

    .. warning::
       **Not interchangeable with**
       :class:`autocast.metrics.coverage.MultivariateRankHistogram`, which carries
       the same name on this branch and is the class the eval registry exposes as
       ``mv_rank_histogram``. Both compute the identical per-instance pre-rank; they
       differ in ACCUMULATION. ``coverage.MultivariateRankHistogram`` keeps one
       histogram **per channel** and averages the total-variation distance across
       them, so per-channel miscalibration cannot cancel. This class pools every
       channel into one shared histogram, where opposing per-channel shapes
       partially cancel (see the note below). The per-channel form is the more
       conservative read and is the one every published rank-ladder
       ``mv_rank_histogram`` number was produced with; this pooled form was renamed
       on import (2026-08-01) so the two can coexist without a silent name
       collision. Upstream (``origin/add-rollout-accumulation-testbed``) calls this
       class ``MultivariateRankHistogram``.

    Shares :class:`AverageRankHistogram`'s pre-rank (the mean over components of the
    univariate pool rank) but changes the jointly ranked vector to the *spatial field
    of a single channel*: each ``(batch, time, channel)`` is a separate verification
    instance, so channels are ranked independently rather than folded into one
    vector. This makes it the per-channel spatial-calibration probe (column 2 of the
    assessment suite): a flat histogram is calibrated, ``\cup`` (U-shaped) is joint
    under-dispersion, a dome is joint over-dispersion. For a single channel it reduces
    exactly to :class:`AverageRankHistogram`.

    The average-rank construction is sensitive to multivariate bias and dispersion
    but only weakly to a mis-specified *correlation structure*; pair it with the
    correlation-sensitive
    :class:`~autocast.metrics.ensemble.VariogramScore` for the latter.

    Notes:
    Each channel's per-instance rank is computed independently, but all
    ``B * T * C`` ranks are then accumulated into the *single* shared histogram of
    :class:`_RankHistogramBase` (there is no per-channel histogram breakdown).
    Consequently opposing per-channel shapes partially cancel -- e.g. one
    under-dispersed and one over-dispersed channel pool toward a flatter aggregate --
    so a flat histogram here means *calibrated on average across channels*, not
    per-channel calibrated. Read it alongside the per-site marginal diagnostics.
    This diverges from the classic construction that keeps one histogram per channel.

    Ties are broken by the shared randomized rank of :class:`_RankHistogramBase`
    (observation stacked last, uniform jitter over the tie block), which is unbiased
    under calibration. This differs from the classic construction that places the
    observation deterministically lowest on a tie -- so this metric does not inherit
    that construction's bias toward bin 0 on heavily quantised or zero-inflated
    fields. Integer-valued inputs are supported (the pool is cast to float64 before
    ranking); the rank remains ordinal, so exact ties on quantised data are resolved
    only by the randomized jitter, not by mid-ranks.

    Only ``vector_dims="spatial"`` (rank each per-timestep spatial field) is
    implemented.

    References:
    Gneiting, T., Stanberry, L.I., Grimit, E.P., Held, L. and Johnson, N.A. (2008).
    Assessing probabilistic forecasts of multivariate quantities, with an
    application to ensemble predictions of surface winds. TEST, 17, 211-235.
    """

    name: str = "mv_rank_histogram"

    def __init__(
        self,
        n_members: int,
        vector_dims: str = "spatial",
        record_ranks: bool = False,
        dist_sync_on_step: bool = False,
    ) -> None:
        """Initialize the multivariate rank histogram.

        Parameters
        ----------
        n_members
            Number of ensemble members (the ensemble axis size).
        vector_dims
            Which dimensions form the jointly ranked vector. Only ``"spatial"``
            (the per-timestep spatial field, ranked per channel) is implemented.
        record_ranks
            Keep the per-frame observation ranks for a trajectory-block margin
            (see :meth:`~_RankHistogramBase.recorded_ranks`).
        dist_sync_on_step
            Passed through to :class:`torchmetrics.Metric`.
        """
        if vector_dims != "spatial":
            msg = (
                "PooledMultivariateRankHistogram currently supports only "
                f"vector_dims='spatial', got {vector_dims!r}"
            )
            raise NotImplementedError(msg)
        super().__init__(
            n_members=n_members,
            record_ranks=record_ranks,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.vector_dims = vector_dims

    def _group_frames(
        self, y_pred_t: Tensor, y_true_t: Tensor
    ) -> tuple[Tensor, Tensor]:
        # One vector per (batch, time, channel): the spatial field, with the channel
        # axis kept out of the ranked vector (unlike the base, which folds spatial
        # and channel together). ``(b t c)`` groups b-slowest / c-fastest.
        members = rearrange(y_pred_t, "b t ... c m -> (b t c) (...) m")
        truth = rearrange(y_true_t, "b t ... c -> (b t c) (...)")
        return members, truth


#: Former name of :class:`PooledMultivariateRankHistogram`, kept as a redirect
#: rather than removed. The class was renamed here to say plainly that it pools
#: the spatial axes before ranking; other branches of this repository still
#: import the old spelling, and a rename that deletes the old name turns a
#: mechanical merge into a silent breakage. Both names are the same object, so
#: `isinstance` checks and pickles are unaffected.
MultivariateRankHistogram = PooledMultivariateRankHistogram


class DependenceRankHistogram(PooledMultivariateRankHistogram):
    r"""Dependence (variogram) pre-rank histogram -- the correlation-structure probe.

    Where :class:`PooledMultivariateRankHistogram` pre-ranks by the average pool rank
    (sensitive to joint bias and dispersion but only weakly to correlation), this
    histogram pre-ranks each field by a *variogram-type dependence functional*
    (Allen, Ziegel & Ginsbourger 2024) that reads the ensemble's spatial
    *correlation structure* while staying blind to a homogeneous marginal error --
    a whole-field shift or rescaling of the field (see the normalisation below;
    it is not invariant to independent per-site perturbations, which also alter the
    correlation). It reuses that class's per-channel spatial vector formation -- each
    ``(batch, time, channel)`` is one verification instance -- and changes only the
    pre-rank.

    Pre-rank
    --------
    For a single field vector :math:`x \in \mathbb{R}^D` (one channel's spatial field),

    .. math::
        \rho(x) = -\frac{\sum_{i \ne j} w_{ij}\,|x_i - x_j|^p}{s(x)^p},

    with :math:`p = 0.5` by default and :math:`s(x)^2` the sample variance of
    :math:`x` over its :math:`D` sites. The numerator is a (weighted) variogram --
    large when the field is rough (neighbouring sites decorrelated), small when
    smooth (neighbours correlated). Two choices make it a *dependence* probe rather
    than a marginal one:

    * **Local weights (essential).** With uniform :math:`w_{ij}=1` the sum is
      invariant to any permutation of the sites, so it sees only the field's value
      *multiset* (its marginal shape) and is blind to spatial arrangement. The
      default weights are therefore inverse-distance in the flattened site order,
      :math:`w_{ij}=1/|i-j|`, which emphasise neighbouring pairs and so respond to
      local correlation. (Exact for a 1-D / line field; for a 2-D grid or a periodic
      ring, pass a geometry-aware ``(D, D)`` matrix, as
      :class:`~autocast.metrics.ensemble.VariogramScore` accepts.) A caller who
      *wants* the permutation-invariant marginal-shape probe can pass an all-ones
      matrix.
    * **Scale-invariant normalisation.** Dividing by :math:`s(x)^p` (equivalently:
      standardise :math:`x` to unit variance, then take the variogram) makes
      :math:`\rho` invariant to any whole-field affine rescaling
      :math:`x \mapsto a + c\,x`, so it is blind to a homogeneous bias or variance
      change -- the metric's *designed blindness*. Note this is :math:`s^p`, not the
      :math:`s^2` that only cancels the scaling at :math:`p=2`.

    The observation is then ranked among the members by :math:`\rho`; a high
    :math:`\rho` marks a *smooth* (highly correlated) field. (The implementation sums
    the full :math:`D \times D` matrix, i.e. :math:`\sum_{i \ne j}` with the zero
    diagonal dropped; that is twice :math:`\sum_{i<j}`, an overall constant that
    leaves every rank unchanged.)

    Sign convention (worked example)
    --------------------------------
    Take :math:`p=0.5`, default inverse-distance weights, and two 4-site fields that
    are permutations of the same values (identical marginals): a smooth ramp
    :math:`s=(0,1,2,3)` and a rough :math:`r=(0,2,1,3)`. The ramp's neighbouring
    differences are all :math:`1`; the rough field's are :math:`2, 1, 2`, so its
    local (heavily weighted) variogram is larger. Hence :math:`\rho(s) > \rho(r)` --
    the smooth field ranks *higher*. Consequently, if the ensemble is
    **under-correlated** (members rougher than the truth) the truth is the smooth
    outlier and ranks high, so the histogram slopes toward **high ranks**; if the
    ensemble is **over-correlated** (members too smooth) the truth is the rough
    outlier and the histogram slopes toward **low ranks**. A flat histogram is
    calibrated correlation, and by construction the shape is (near-)flat under pure
    marginal errors.

    Cost
    ----
    The variogram sum is over a :math:`D \times D` pairwise matrix, so the work is
    quadratic in the number of sites. Memory is **not**: the sum is accumulated over
    slabs bounded by ``block_bytes`` (:data:`PAIRWISE_BLOCK_BYTES`), never as one
    :math:`(F, D, D, M+1)` tensor. Building it whole is what killed the 2026-08-01
    cluster runs -- 264 GiB requested for a single 64-frame batch of 64x64 fields at
    32 members -- so do not "simplify" the chunked accumulation back into one
    expression. The work itself remains heavy on large fields; gate it with
    ``eval.compute_rank_histograms`` accordingly.

    Only ``vector_dims="spatial"`` is implemented.

    References:
    Allen, S., Ziegel, J. and Ginsbourger, D. (2024). Assessing the calibration of
    multivariate probabilistic forecasts. arXiv:2307.05846 (pre-rank functions for
    multivariate rank histograms).
    """

    name: str = "dependence_rank_histogram"

    def __init__(
        self,
        n_members: int,
        p: float = 0.5,
        weights: Tensor | np.ndarray | None = None,
        var_floor: float = 1e-12,
        vector_dims: str = "spatial",
        block_bytes: int = PAIRWISE_BLOCK_BYTES,
        record_ranks: bool = False,
        dist_sync_on_step: bool = False,
    ) -> None:
        """Initialize the dependence (variogram) pre-rank histogram.

        Parameters
        ----------
        n_members
            Number of ensemble members (the ensemble axis size).
        p
            Variogram power ``|x_i - x_j|^p``; must be positive (``0.5`` by default,
            matching :class:`~autocast.metrics.ensemble.VariogramScore`).
        weights
            Optional non-negative ``(D, D)`` pair-weight matrix. ``None`` (default) uses
            inverse-distance weights in the flattened site order, ``w_ij = 1/|i-j|`` --
            the local emphasis that makes the pre-rank correlation-sensitive.
        var_floor
            Positive floor on the per-field variance before the ``s(x)^p``
            normalisation, guarding a constant (zero-variance) field. Must be positive.
        vector_dims
            Only ``"spatial"`` (rank each per-timestep spatial field, per channel) is
            implemented.
        block_bytes
            Memory ceiling for the pairwise slab held at one time (see
            :data:`PAIRWISE_BLOCK_BYTES`). Lower it on a small device; it changes
            speed and peak memory only, never the value.
        record_ranks
            Keep the per-frame observation ranks for a trajectory-block margin
            (see :meth:`~_RankHistogramBase.recorded_ranks`).
        dist_sync_on_step
            Passed through to :class:`torchmetrics.Metric`.
        """
        super().__init__(
            n_members=n_members,
            vector_dims=vector_dims,
            record_ranks=record_ranks,
            dist_sync_on_step=dist_sync_on_step,
        )
        if p <= 0:
            msg = f"p must be > 0, got {p}"
            raise ValueError(msg)
        if var_floor <= 0:
            msg = f"var_floor must be positive, got {var_floor}"
            raise ValueError(msg)
        if weights is not None:
            if isinstance(weights, np.ndarray):
                weights = torch.from_numpy(weights)
            if not isinstance(weights, Tensor):
                msg = "weights must be a Tensor, np.ndarray, or None"
                raise TypeError(msg)
            if (weights < 0).any().item():
                msg = "weights must be non-negative (w_ij >= 0)"
                raise ValueError(msg)
        if block_bytes <= 0:
            msg = f"block_bytes must be positive, got {block_bytes}"
            raise ValueError(msg)
        self.p = p
        self.var_floor = var_floor
        self.weights = weights
        self.block_bytes = block_bytes

    def _pair_weights(self, d: int, ref: Tensor) -> Tensor:
        """Return the ``(D, D)`` pair weights on ``ref``'s device/dtype (zero diag)."""
        if self.weights is not None:
            if tuple(self.weights.shape) != (d, d):
                msg = (
                    "weights must have shape (D, D) matching the spatial field size; "
                    f"got {tuple(self.weights.shape)} with D={d}"
                )
                raise ValueError(msg)
            return self.weights.to(device=ref.device, dtype=ref.dtype)
        # Default: inverse flattened-index distance w_ij = 1/|i-j| (zero diagonal).
        idx = torch.arange(d, device=ref.device, dtype=ref.dtype)
        dist = (idx[:, None] - idx[None, :]).abs()
        return (1.0 / dist.clamp_min(1.0)) * (dist > 0)

    def _pre_rank(self, pool: Tensor) -> Tensor:
        # pool: (F, D, M+1), float64 (base casts), observation stacked last.
        n_frames, d, n_pool = pool.shape
        weights = self._pair_weights(d, pool)  # (D, D)
        # One row per field to be scored: (F*(M+1), D), frame-slowest.
        fields = pool.permute(0, 2, 1).reshape(n_frames * n_pool, d)
        variogram = _weighted_variogram(
            fields, weights, self.p, self.block_bytes
        ).reshape(n_frames, n_pool)  # (F, M+1)
        # Scale/bias-invariant normalisation s(x)^p = var^(p/2), per pool element.
        var = pool.var(dim=1, unbiased=False)  # (F, M+1)
        denom = var.clamp_min(self.var_floor).pow(self.p / 2.0)
        return -variogram / denom  # (F, M+1); high rho = smooth (correlated) field
