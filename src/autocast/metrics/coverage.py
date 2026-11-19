from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from torch.nn import ModuleList
from torchmetrics import Metric

from autocast.metrics.ensemble import BTSCMMetric
from autocast.metrics.intervals import interval_indicator
from autocast.types import (
    Tensor,
    TensorBTSC,
    TensorBTSCM,
    TensorRealBTSC,
    TensorRealBTSCM,
)


class Coverage(BTSCMMetric):
    """Coverage probability for a fixed coverage level.

    Calculates the proportion of true values that fall within the symmetric
    prediction interval defined by the coverage level.
    """

    name: str = "coverage"

    def __init__(self, coverage_level: float = 0.95, **kwargs):
        """Initialize Coverage metric.

        Args:
            coverage_level: nominal coverage probability (e.g. 0.95 for 95% interval).
                Must be between 0 and 1.
            **kwargs: Additional arguments passed to BaseMetric.
        """
        super().__init__(**kwargs)
        if not (0 < coverage_level < 1):
            raise ValueError(f"coverage_level must be in (0, 1), got {coverage_level}")
        self.coverage_level = coverage_level

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTSC:
        """Compute per-gridpoint coverage indicator.

        Args:
            y_pred: (B, T, S, C, M)
            y_true: (B, T, S, C)

        Returns:
            coverage: (B, T, S, C) — 1.0 where y_true falls inside the
                interval, else 0.0
        """
        # e.g. coverage_level=0.95 -> 0.025 and 0.975 quantiles
        q_low = 0.5 - self.coverage_level / 2
        q_high = 0.5 + self.coverage_level / 2

        q_tensor = torch.tensor(
            [q_low, q_high], device=y_pred.device, dtype=y_pred.dtype
        )
        quantiles = torch.quantile(y_pred, q_tensor, dim=-1)  # (2, B, T, S, C)

        return interval_indicator(quantiles[0], quantiles[1], y_true)


class MultiCoverage(Metric):
    """Computes coverage for multiple coverage levels at once.

    This is a wrapper around multiple Coverage metrics. It inherits from Metric
    to integrate with PyTorch Lightning and TorchMetrics.
    """

    def __init__(self, coverage_levels: list[float] | None = None):
        super().__init__()
        if coverage_levels is None:
            coverage_levels = [
                round(x, 2) for x in torch.linspace(0.05, 0.95, steps=19).tolist()
            ]

        self.coverage_levels = coverage_levels
        # List of Coverage metrics with reduce_all=False to allow per-channel analysis
        self.metrics = ModuleList(
            [Coverage(coverage_level=cl, reduce_all=False) for cl in coverage_levels]
        )

    def update(self, y_pred, y_true):
        for metric in self.metrics:
            assert isinstance(metric, Coverage)
            metric.update(y_pred, y_true)

    def compute(self) -> Tensor:
        """Compute the Average Calibration Error."""
        errors = []
        for cl, metric in zip(self.coverage_levels, self.metrics, strict=True):
            assert isinstance(metric, Coverage)
            # Calibration error: |observed - expected|
            errors.append(torch.abs(metric.compute() - cl))

        return torch.stack(errors).mean()

    def _compute_levels_and_values(self) -> tuple[list[float], list[float]]:
        """Get coverage levels and observed values for plotting."""
        levels, observed = [], []
        for cl, metric in zip(self.coverage_levels, self.metrics, strict=True):
            assert isinstance(metric, Coverage)
            levels.append(cl)
            observed.append(metric.compute().mean().item())
        return levels, observed

    def compute_detailed(self) -> dict[str, float]:
        """Return a dict of results, keys formatted as 'coverage_{coverage_level}'."""
        return {
            f"coverage_{level}": value
            for level, value in zip(*self._compute_levels_and_values(), strict=True)
        }

    def plot(
        self,
        save_path: Path | str | None = None,
        title: str = "Coverage Plot",
        cmap_str: str = "viridis",
        save_csv: bool = True,
    ):
        """Plot reliability diagram showing expected vs observed coverage.

        Args:
            save_path: Path to save the plot (PNG). If provided and save_csv=True,
                a CSV file with the same name will also be saved.
            title: Plot title.
            cmap_str: Color map string from matplotlib.
            save_csv: If True and save_path is provided, save plot data as CSV
                before creating the plot.

        Returns:
            matplotlib.figure.Figure
        """
        # Prepare data structure: levels -> [val_c1, val_c2, ...]
        levels = self.coverage_levels
        observed_means = []
        observed_channels = []  # shape (L, C)

        # Loop over metrics
        for metric in self.metrics:
            assert isinstance(metric, Coverage)
            val = metric.compute()
            # Ensure channel coverage is always represented as 1D, even if a
            # degenerate shape collapses to a scalar.
            val_c = np.atleast_1d(val.mean(dim=0).cpu().numpy())  # (C,)
            observed_channels.append(val_c)
            observed_means.append(val_c.mean().item())

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Optimal line (y=x)
        ax.plot([0, 1], [0, 1], "k:", label="Expected", linewidth=2)

        # Plot channels
        observed_arr = np.stack(observed_channels)  # (L, C)

        # Save CSV data if requested
        if save_path and save_csv:
            self._save_csv_data(
                save_path=save_path,
                levels=levels,
                observed_means=observed_means,
                observed_channels=observed_arr,
            )

        cmap = plt.get_cmap(cmap_str)  # cmap for each channel
        n_channels = observed_arr.shape[1]
        for c in range(n_channels):
            color = cmap(c / n_channels) if n_channels > 1 else "blue"
            label = f"Ch {c}" if n_channels <= 10 else None
            ax.plot(
                levels,
                observed_arr[:, c],
                color=color,
                alpha=0.3,
                linewidth=1,
                label=label,
            )

        # Plot mean coverage in bold
        ax.plot(levels, observed_means, "k-", linewidth=3, label="Mean")
        ax.set_xlabel(r"Coverage level, $\alpha$")
        ax.set_ylabel("Observed Coverage")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.6)

        # Only show legend if not cluttered
        if n_channels > 10:
            # Add manual legend for trace
            custom_lines = [
                Line2D([0], [0], color="k", lw=3),
                Line2D([0], [0], color="grey", lw=1, alpha=0.5),
            ]
            ax.legend(custom_lines, ["Mean", "Individual Channels"])
        else:
            ax.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.close(fig)
        return fig

    def _save_csv_data(
        self,
        save_path: Path | str,
        levels: list[float],
        observed_means: list[float],
        observed_channels: np.ndarray,
    ) -> None:
        """Save coverage plot data to CSV file.

        Args:
            save_path: Path for the PNG file. CSV will use the same path
                with .csv extension.
            levels: Coverage levels (expected coverage values).
            observed_means: Mean observed coverage across all channels for each level.
            observed_channels: Observed coverage per level per channel, shape (L, C).
        """
        # Generate CSV path from PNG path
        csv_path = Path(save_path).with_suffix(".csv")

        # Build DataFrame
        data = {
            "coverage_level": levels,
            "observed_mean": observed_means,
        }

        if observed_channels.ndim == 1:
            observed_channels = observed_channels[:, None]

        # Add per-channel columns
        n_channels = observed_channels.shape[1]
        for c in range(n_channels):
            data[f"channel_{c}"] = observed_channels[:, c].tolist()

        df = pd.DataFrame(data)

        # Save CSV
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    def reset(self):
        # Reset all sub-metrics
        for metric in self.metrics:
            assert isinstance(metric, Coverage)
            metric.reset()


class RankHistogram(Metric):
    r"""Verification-rank (Talagrand) histogram for ensemble calibration.

    This is the ensemble analogue of the probability integral transform (PIT).
    For each scalar grid point the *rank* of the observation among the
    :math:`M` ensemble members is computed (with random tie-breaking), giving an
    integer in :math:`\{0, \dots, M\}`. Pooling these ranks over all grid points
    yields a histogram with :math:`M + 1` bins. A perfectly calibrated ensemble
    produces a flat (uniform) histogram; a :math:`\cup`-shape indicates
    under-dispersion (over-confidence), a :math:`\cap`-shape (dome)
    over-dispersion, and a slope a systematic bias.

    Unlike symmetric-interval coverage, which only probes central intervals, the
    rank histogram probes the *whole* predictive distribution, so it can flag a
    model that has correct central coverage but the wrong tail/skew shape.

    The scalar :meth:`compute` returns the total-variation distance between the
    observed rank frequencies and the uniform distribution (lower = better,
    ``0`` = perfectly flat), averaged over channels.

    References:
    ----------
    Hamill, T.M. (2001). Interpretation of rank histograms for verifying
    ensemble forecasts. Mon. Wea. Rev., 129(3), 550-560.
    """

    name: str = "rank_histogram"

    def __init__(self, generator: torch.Generator | None = None):
        """Initialize the rank histogram.

        Args:
            generator: Optional ``torch.Generator`` for the random tie-breaking,
                for reproducibility. Must live on the same device as the inputs.
        """
        super().__init__()
        self.generator = generator
        # Per-channel counts, lazily shaped (C, M+1) on the first update once the
        # ensemble size M and channel count C are known; numel() == 0 marks it
        # uninitialised (torchmetrics restores this default on reset()).
        self.add_state("counts", default=torch.zeros(0), dist_reduce_fx="sum")

    def _check_shapes(self, y_pred: TensorRealBTSCM, y_true: TensorRealBTSC) -> None:
        """Validate that ``y_pred`` is ``y_true`` plus a trailing ensemble axis."""
        if y_pred.shape[:-1] != y_true.shape:
            raise ValueError(
                "y_pred and y_true must match except for the ensemble "
                f"dimension; got {tuple(y_pred.shape)} and {tuple(y_true.shape)}"
            )

    def update(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> None:
        """Accumulate verification ranks from a batch.

        Args:
            y_pred: Ensemble predictions of shape (B, T, S, C, M).
            y_true: Ground truth of shape (B, T, S, C).
        """
        self._check_shapes(y_pred, y_true)
        n_ensemble = y_pred.shape[-1]
        rank = self._ranks_with_tiebreak(
            below=(y_pred < y_true.unsqueeze(-1)).sum(dim=-1),
            equal=(y_pred == y_true.unsqueeze(-1)).sum(dim=-1),
            n_ensemble=n_ensemble,
        )  # (..., C) in [0, M]
        ranks_by_channel = rearrange(rank, "... c -> c (...)")  # (C, N)
        self._accumulate(ranks_by_channel, n_bins=n_ensemble + 1)

    def _ranks_with_tiebreak(
        self, below: Tensor, equal: Tensor, n_ensemble: int
    ) -> Tensor:
        """Resolve integer ranks in ``[0, M]`` with random tie-breaking.

        ``below`` counts members strictly below the observation and ``equal``
        counts ties (including the observation itself). With no ties this is
        simply ``below``; ties are spread uniformly over the ``equal + 1`` ranks
        they share.
        """
        u = torch.rand(below.shape, device=below.device, generator=self.generator)
        rank = below + (u * (equal.to(u.dtype) + 1.0)).floor().long()
        return rank.clamp_(0, n_ensemble)

    def _accumulate(self, ranks_by_channel: Tensor, n_bins: int) -> None:
        """Bin integer ranks of shape (C, N) into the per-channel histogram."""
        n_channels = ranks_by_channel.shape[0]
        if self.counts.numel() == 0:
            self.counts = torch.zeros(
                n_channels, n_bins, device=ranks_by_channel.device, dtype=torch.long
            )
        elif tuple(self.counts.shape) != (n_channels, n_bins):
            raise ValueError(
                "Inconsistent ensemble/channel size across batches: had "
                f"{tuple(self.counts.shape)} (channels, bins), got "
                f"({n_channels}, {n_bins})"
            )
        for c in range(n_channels):
            self.counts[c] += torch.bincount(ranks_by_channel[c], minlength=n_bins)

    def _frequencies(self) -> Tensor:
        """Return the per-channel normalised rank frequencies, shape (C, n_bins)."""
        if self.counts.numel() == 0:
            msg = "No samples were provided to RankHistogram"
            raise RuntimeError(msg)
        totals = self.counts.sum(dim=1, keepdim=True).clamp(min=1)
        return self.counts.float() / totals

    def compute(self) -> Tensor:
        """Total-variation distance from uniform, averaged over channels."""
        freqs = self._frequencies()
        uniform = 1.0 / freqs.shape[1]
        tv = 0.5 * (freqs - uniform).abs().sum(dim=1)  # (C,)
        return tv.mean()

    def compute_detailed(self) -> dict[str, list[int] | list[float]]:
        """Return the channel-averaged rank frequencies keyed for logging."""
        freqs = self._frequencies()
        return {
            "bin": list(range(freqs.shape[1])),
            "frequency_mean": freqs.mean(dim=0).cpu().tolist(),
        }

    def plot(
        self,
        save_path: Path | str | None = None,
        title: str = "Rank Histogram",
        save_csv: bool = True,
    ):
        """Plot the (channel-averaged) rank histogram against the uniform line.

        Parameters
        ----------
        save_path: Path or str, optional
            Path to save the PNG; if given and ``save_csv`` is True a CSV with the
            same stem is written first.
        title: str
            Plot title.
        save_csv: bool, default True
            Save the histogram frequencies to CSV alongside the PNG.

        Returns:
        -------
        matplotlib.figure.Figure
        """
        freqs = self._frequencies().cpu().numpy()  # (C, n_bins)
        n_bins = freqs.shape[1]
        mean_freq = freqs.mean(axis=0)
        uniform = 1.0 / n_bins

        if save_path and save_csv:
            self._save_csv_data(save_path, freqs)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            range(n_bins),
            mean_freq,
            width=1.0,
            edgecolor="k",
            color="steelblue",
            alpha=0.7,
            label="Observed",
        )
        ax.axhline(uniform, color="k", linestyle=":", linewidth=2, label="Uniform")
        ax.set_xlabel("Verification rank")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.set_xlim(-0.5, n_bins - 0.5)
        ax.set_ylim(bottom=0)
        ax.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.close(fig)
        return fig

    def _save_csv_data(self, save_path: Path | str, freqs: np.ndarray) -> None:
        """Write per-channel rank frequencies to a CSV next to the PNG."""
        csv_path = Path(save_path).with_suffix(".csv")
        data: dict[str, list] = {
            "bin": list(range(freqs.shape[1])),
            "frequency_mean": freqs.mean(axis=0).tolist(),
        }
        for c in range(freqs.shape[0]):
            data[f"channel_{c}"] = freqs[c].tolist()
        df = pd.DataFrame(data)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)


class MultivariateRankHistogram(RankHistogram):
    r"""Average-rank multivariate verification-rank histogram (joint calibration).

    This is the multivariate analogue of :class:`RankHistogram`: instead of
    ranking the observation one grid point at a time, it ranks the *entire
    spatial field* jointly, probing whether the ensemble captures the
    multivariate (cross-grid-point) predictive distribution -- not just each
    location marginally.

    For each verification instance (one field per batch/time/channel) the
    observation and the :math:`M` members are pooled, and each is given a
    *pre-rank* equal to the mean over grid points of its componentwise univariate
    rank within the pool. The multivariate rank of the observation is then its
    rank among the :math:`M + 1` pre-ranks (with random tie-breaking),
    histogrammed over instances. A calibrated ensemble gives a flat histogram; a
    dome indicates joint over-dispersion and a :math:`\cup` joint
    under-dispersion.

    Notes:
    -----
    The average-rank construction is sensitive to multivariate bias and
    dispersion but only weakly to a mis-specified *correlation structure*; pair
    it with the correlation-sensitive
    :class:`~autocast.metrics.ensemble.VariogramScore` for the latter.

    This assumes continuous fields. The componentwise ranks are ordinal
    (``argsort``-based), and because the observation sits at pool index 0 it is
    deterministically assigned the lowest rank at any grid point where it *exactly*
    ties a member -- biasing the histogram toward bin 0. For continuous forecasts
    such exact ties are measure-zero, but on heavily quantised or zero-inflated
    fields the result is biased (the standard fix is average/mid-ranks for ties).

    Only ``vector_dims="spatial"`` (rank each per-timestep spatial field) is
    currently implemented.

    References:
    ----------
    Gneiting, T., Stanberry, L.I., Grimit, E.P., Held, L. and Johnson, N.A.
    (2008). Assessing probabilistic forecasts of multivariate quantities, with an
    application to ensemble predictions of surface winds. TEST, 17, 211-235.
    """

    name: str = "mv_rank_histogram"

    def __init__(
        self,
        vector_dims: str = "spatial",
        generator: torch.Generator | None = None,
    ):
        """Initialize the multivariate rank histogram.

        Args:
            vector_dims: Which dimensions form the jointly-ranked vector. Only
                ``"spatial"`` (the per-timestep spatial field) is implemented.
            generator: Optional ``torch.Generator`` for random tie-breaking.
        """
        if vector_dims != "spatial":
            msg = (
                "MultivariateRankHistogram currently supports only "
                f"vector_dims='spatial', got {vector_dims!r}"
            )
            raise NotImplementedError(msg)
        super().__init__(generator=generator)
        self.vector_dims = vector_dims

    def update(self, y_pred: TensorRealBTSCM, y_true: TensorRealBTSC) -> None:
        """Accumulate multivariate verification ranks from a batch.

        Args:
            y_pred: Ensemble predictions of shape (B, T, S, C, M).
            y_true: Ground truth of shape (B, T, S, C).
        """
        self._check_shapes(y_pred, y_true)
        n_ensemble = y_pred.shape[-1]

        # One jointly-ranked vector per (batch, time, channel): the spatial field.
        obs = rearrange(y_true, "b t ... c -> c (b t) (...)")  # (C, N, d)
        members = rearrange(y_pred, "b t ... c m -> c (b t) (...) m")  # (C, N, d, M)

        # Pool obs (index 0) with the members, rank each grid point within the
        # pool, then average those componentwise ranks into a scalar pre-rank.
        # The integer ranks must be averaged in a floating dtype (preserve a
        # float64 input's precision, but fall back to float32 for integer fields
        # since torch.mean has no integer implementation).
        pooled = torch.cat([obs.unsqueeze(-1), members], dim=-1)  # (C, N, d, M+1)
        component_ranks = pooled.argsort(dim=-1).argsort(dim=-1)  # (C, N, d, M+1)
        rank_dtype = obs.dtype if obs.is_floating_point() else torch.float32
        pre_ranks = component_ranks.to(rank_dtype).mean(dim=2)  # (C, N, M+1)

        obs_pre = pre_ranks[..., 0:1]  # (C, N, 1)
        member_pre = pre_ranks[..., 1:]  # (C, N, M)
        mv_rank = self._ranks_with_tiebreak(
            below=(member_pre < obs_pre).sum(dim=-1),
            equal=(member_pre == obs_pre).sum(dim=-1),
            n_ensemble=n_ensemble,
        )  # (C, N) in [0, M]

        self._accumulate(mv_rank, n_bins=n_ensemble + 1)
