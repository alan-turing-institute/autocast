"""Collection (regional-sum) coverage for ensemble forecasts.

The correlation-structure diagnostic the per-site :class:`Coverage` is blind to:
for each spatial region (a contiguous arc in 1-D, a square tile in 2-D, or the
whole field), the ensemble members and the truth are *summed across the region's
sites* and the symmetric-quantile coverage of that regional sum is measured. A
spatially-white ensemble with a correct per-site marginal still under-covers the
sum (the members cancel rather than add coherently), so the regional-sum coverage
collapses while the per-site coverage stays nominal -- the headline tell of the
correlation-structure miss (research piece sec. 6.6).

Region masks and the symmetric-quantile-of-the-summed-members kernel mirror
``local_hydra/local_experiment/cal_study/_diff_shared.py`` (``build_masks`` and
``patch_coverage``) exactly, so this production metric agrees with the harness it
ports. Twins: :class:`~autocast.metrics.coverage.Coverage` (the symmetric-quantile
coverage indicator) and :class:`~autocast.metrics.ensemble.SpreadSkillRatio` (the
``score()`` override for a metric whose reduction changes the spatial shape).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torchmetrics import Metric

from autocast.metrics.ensemble import BTSCMMetric
from autocast.types import ArrayLike, Tensor, TensorBTC, TensorBTSCM


def _region_index(
    spatial: tuple[int, ...], n_channels: int, region_size: int
) -> Tensor:
    """Flat indices of every region, shape ``(n_regions, region_size)``.

    Same tiling and the same region order as :func:`_region_masks`, but built
    arithmetically instead of as a list of ``(d_total,)`` boolean arrays. That
    list was the metric's dominant cost: at ``region_size=1`` on a 64x64x4 field
    it allocated 16384 masks of 16384 bools (~268 MB) on *every* update, then did
    one ``np.where`` and one host-to-device copy per mask.

    Regions tile the field exactly (:func:`_region_masks` raises otherwise), so
    every region holds the same number of sites and the result is rectangular.
    """
    big = [d for d in spatial if d > 1]

    if len(big) <= 1:
        length = int(np.prod(spatial))
        _validate_1d_tiling(length, n_channels, region_size)
        return torch.arange(length, dtype=torch.long).reshape(-1, region_size)

    h, w = big[0], big[1]
    side = round(region_size**0.5)
    _validate_2d_tiling(h, w, side, region_size)
    # Flat position of (hh, ww, ch) in the row-major (*spatial, channel) frame.
    site = torch.arange(h * w, dtype=torch.long).reshape(h, w)
    flat = site.unsqueeze(-1) * n_channels + torch.arange(n_channels)  # (h, w, C)
    # Split each axis into tiles, then order regions (channel, h0, w0) and the
    # sites within a region (hh, ww) -- exactly _region_masks' loop nesting.
    tiled = flat.reshape(h // side, side, w // side, side, n_channels)
    return tiled.permute(4, 0, 2, 1, 3).reshape(-1, side * side)


def _validate_1d_tiling(length: int, n_channels: int, region_size: int) -> None:
    """Raise unless contiguous ``region_size`` arcs tile a 1-D single-channel field."""
    if n_channels > 1:
        msg = (
            "CollectionCoverage 1-D region masks support a single channel only; "
            f"got n_channels={n_channels}. The interleaved flat layout would "
            "leave channels >= 1 uncovered; pass explicit masks for multichannel "
            "1-D fields."
        )
        raise ValueError(msg)
    if region_size > length or length % region_size != 0:
        msg = (
            f"CollectionCoverage region_size={region_size} does not tile a "
            f"length-{length} field; pick a divisor of {length}."
        )
        raise ValueError(msg)


def _validate_2d_tiling(h: int, w: int, side: int, region_size: int) -> None:
    """Raise unless square ``side`` tiles divide an ``h`` x ``w`` field."""
    if side * side != region_size or side > h or h % side != 0 or w % side != 0:
        msg = (
            f"CollectionCoverage region_size={region_size} does not tile a "
            f"{h}x{w} field with a square tile; pick a perfect square that "
            f"divides the side."
        )
        raise ValueError(msg)


def _region_masks(
    spatial: tuple[int, ...], n_channels: int, region_size: int
) -> list[np.ndarray]:
    """Boolean masks over the flattened ``(*spatial, channel)`` frame for one size.

    Mirrors ``_diff_shared.py::build_masks`` exactly: in 1-D, contiguous arcs of
    ``region_size`` sites; in 2-D, square tiles of side ``sqrt(region_size)``. The
    flatten order is row-major over ``(*spatial, channel)`` (``reshape(..., -1)``),
    matching the harness ``patch_coverage`` input.
    """
    big = [d for d in spatial if d > 1]
    d_total = int(np.prod(spatial)) * n_channels
    masks: list[np.ndarray] = []

    if len(big) <= 1:
        length = int(np.prod(spatial))
        if n_channels > 1:
            msg = (
                "CollectionCoverage 1-D region masks support a single channel only; "
                f"got n_channels={n_channels}. The interleaved flat layout would "
                "leave channels >= 1 uncovered; pass explicit masks for multichannel "
                "1-D fields."
            )
            raise ValueError(msg)
        if region_size > length or length % region_size != 0:
            msg = (
                f"CollectionCoverage region_size={region_size} does not tile a "
                f"length-{length} field; pick a divisor of {length}."
            )
            raise ValueError(msg)
        for start in range(0, length, region_size):
            m = np.zeros(d_total, bool)
            m[start : start + region_size] = True
            masks.append(m)
        return masks

    h, w = big[0], big[1]
    side = round(region_size**0.5)
    if side * side != region_size or side > h or h % side != 0 or w % side != 0:
        msg = (
            f"CollectionCoverage region_size={region_size} does not tile a "
            f"{h}x{w} field with a square tile; pick a perfect square that "
            f"divides the side."
        )
        raise ValueError(msg)
    for ch in range(n_channels):
        for h0 in range(0, h, side):
            for w0 in range(0, w, side):
                m = np.zeros(d_total, bool)
                for hh in range(h0, h0 + side):
                    for ww in range(w0, w0 + side):
                        m[(hh * w + ww) * n_channels + ch] = True
                masks.append(m)
    return masks


def default_region_sizes(spatial: tuple[int, ...]) -> list[int]:
    """Region sizes that tile a field of shape ``spatial`` (mirrors build_masks keys).

    1-D: divisors of the length from a fixed sweep, always including the per-site
    size (1) and the whole field. 2-D: square tiles whose side divides the grid,
    drawn from the sweep ``(1, 2, 4, 8)`` -- so the whole field is included only
    when its side is at most 8 (an 8x8 grid yields the full 64, but a 16x16 grid
    stops at 64-site tiles and omits the 256-site whole field). Pass
    ``region_sizes`` explicitly to probe the whole field on larger 2-D grids.
    """
    big = [d for d in spatial if d > 1]
    if len(big) <= 1:
        length = int(np.prod(spatial))
        sizes = [
            k for k in (1, 2, 4, 5, 8, 10, 20, 40) if k <= length and length % k == 0
        ]
        if length not in sizes:
            sizes.append(length)
        return sorted(set(sizes))
    h, w = big[0], big[1]
    sizes = [s * s for s in (1, 2, 4, 8) if s <= h and h % s == 0 and w % s == 0]
    return sorted(set(sizes))


class CollectionCoverage(BTSCMMetric):
    """Symmetric-quantile coverage of regional (collection) sums.

    For a single ``region_size`` (number of sites per region; a perfect square in
    2-D), the field is tiled into regions, the ensemble members and the truth are
    summed across each region's sites, and the symmetric ``coverage_level``
    interval is checked on the summed members. The per-region indicators are
    averaged over regions (a spatial-style reduction), giving ``(B, T, C)`` so the
    base-class batch accumulation and ``reduce_all`` behave like the other
    ensemble metrics.

    Parameters
    ----------
    coverage_level: float, default=0.90
        Nominal coverage of the symmetric interval (``q = 0.5 +/- level/2``),
        matching :class:`~autocast.metrics.coverage.Coverage`.
    region_size: int, default=1
        Sites per region. ``1`` recovers per-site coverage; the whole-field sum
        is ``region_size = prod(spatial)``. Must tile the field.
    """

    name: str = "collection_coverage"

    def __init__(
        self,
        coverage_level: float = 0.90,
        region_size: int = 1,
        *,
        score_dims: Literal["spatial", "temporal"] | None = "spatial",
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(
            score_dims=score_dims,
            reduce_all=reduce_all,
            dist_sync_on_step=dist_sync_on_step,
        )
        if not (0 < coverage_level < 1):
            msg = f"coverage_level must be in (0, 1), got {coverage_level}"
            raise ValueError(msg)
        if region_size < 1:
            msg = f"region_size must be >= 1, got {region_size}"
            raise ValueError(msg)
        self.coverage_level = coverage_level
        self.region_size = region_size
        # The tiling depends only on (spatial, n_channels, device), all fixed
        # after the first update, so it is built once rather than per update.
        self._index_key: tuple | None = None
        self._index: Tensor | None = None

    def _regions(
        self, spatial: tuple[int, ...], n_channels: int, device: torch.device
    ) -> Tensor:
        """Cached ``(n_regions, region_size)`` flat index for this field shape."""
        key = (spatial, n_channels, device)
        if self._index_key != key or self._index is None:
            self._index = _region_index(spatial, n_channels, self.region_size).to(
                device
            )
            self._index_key = key
        return self._index

    def _score(self, y_pred: TensorBTSCM, y_true: Tensor) -> Tensor:
        """Not used directly; ``score()`` is overridden (regional reduction)."""
        msg = "CollectionCoverage overrides score() directly."
        raise NotImplementedError(msg)

    def score(self, y_pred: ArrayLike, y_true: ArrayLike) -> TensorBTC:
        """Regional-sum coverage averaged over regions, shape ``(B, T, C)``.

        Args:
            y_pred: ensemble predictions ``(B, T, *spatial, C, M)``.
            y_true: ground truth ``(B, T, *spatial, C)``.
        """
        y_pred_t, y_true_t = self._check_input(y_pred, y_true)

        b, t = y_true_t.shape[0], y_true_t.shape[1]
        n_channels = y_true_t.shape[-1]
        spatial = tuple(y_true_t.shape[2:-1])
        m = y_pred_t.shape[-1]

        region_idx = self._regions(spatial, n_channels, y_pred_t.device)

        # Flatten to (B, T, D, [M]) row-major over (*spatial, C) -- the harness layout.
        d_total = int(np.prod(spatial)) * n_channels
        pred_flat = y_pred_t.reshape(b, t, d_total, m)
        true_flat = y_true_t.reshape(b, t, d_total)

        a = (1.0 - self.coverage_level) / 2.0
        q = torch.tensor([a, 1.0 - a], device=y_pred_t.device, dtype=y_pred_t.dtype)

        # Every region at once: gather (B, T, R, L, [M]) and sum the L sites.
        # The old per-region Python loop ran ~21760 iterations per update across
        # the default size sweep, each launching its own gather/sum/quantile.
        ens_sum = pred_flat[:, :, region_idx, :].sum(dim=3)  # (B, T, R, M)
        truth_sum = true_flat[:, :, region_idx].sum(dim=3)  # (B, T, R)
        quantiles = torch.quantile(ens_sum, q, dim=-1)  # (2, B, T, R)
        covered = (truth_sum >= quantiles[0]) & (truth_sum <= quantiles[1])

        # Average over regions, then expose a singleton channel axis -> (B, T, C).
        coverage_bt = covered.float().mean(dim=-1)  # (B, T)
        return coverage_bt.unsqueeze(-1).expand(b, t, n_channels)


class MultiCollectionCoverage(Metric):
    """Collection coverage across multiple region sizes (the size sweep).

    The collection-coverage analogue of
    :class:`~autocast.metrics.coverage.MultiCoverage`: wraps one
    :class:`CollectionCoverage` per region size and reports / plots the observed
    coverage as a function of region size, writing a per-region-size CSV.
    """

    name: str = "multi_collection_coverage"

    def __init__(
        self,
        coverage_level: float = 0.90,
        region_sizes: list[int] | None = None,
        score_dims: Literal["spatial", "temporal"] | None = "spatial",
    ) -> None:
        super().__init__()
        self.coverage_level = coverage_level
        self.score_dims: Literal["spatial", "temporal"] | None = score_dims
        # region_sizes=None defers to the field shape on the first update (the
        # divisor sweep from default_region_sizes), so the eval wiring need not
        # know the toy's spatial shape in advance.
        self.region_sizes = region_sizes if region_sizes is not None else []
        self.metrics = torch.nn.ModuleList(
            [self._make(size) for size in self.region_sizes]
        )

    def _make(self, size: int) -> CollectionCoverage:
        return CollectionCoverage(
            coverage_level=self.coverage_level,
            region_size=size,
            score_dims=self.score_dims,
            reduce_all=False,
        )

    def _resolve_region_sizes(self, y_true: ArrayLike) -> None:
        """Lazily fix the region-size sweep from the first batch's field shape."""
        if self.region_sizes:
            return
        true_t = y_true if isinstance(y_true, Tensor) else torch.as_tensor(y_true)
        spatial = tuple(int(d) for d in true_t.shape[2:-1])
        self.region_sizes = default_region_sizes(spatial)
        self.metrics = torch.nn.ModuleList(
            [self._make(size) for size in self.region_sizes]
        ).to(true_t.device)

    def update(self, y_pred: ArrayLike, y_true: ArrayLike) -> None:
        self._resolve_region_sizes(y_true)
        for metric in self.metrics:
            assert isinstance(metric, CollectionCoverage)
            metric.update(y_pred, y_true)

    def compute(self) -> Tensor:
        """Mean absolute calibration error across region sizes."""
        errors = []
        for metric in self.metrics:
            assert isinstance(metric, CollectionCoverage)
            errors.append(torch.abs(metric.compute().mean() - self.coverage_level))
        return torch.stack(errors).mean()

    def reset(self) -> None:
        for metric in self.metrics:
            assert isinstance(metric, CollectionCoverage)
            metric.reset()

    def _observed_by_size(self) -> list[float]:
        observed = []
        for metric in self.metrics:
            assert isinstance(metric, CollectionCoverage)
            observed.append(float(metric.compute().mean().item()))
        return observed

    def compute_detailed(self) -> dict[str, float]:
        """Observed coverage per region size, keyed ``collection_coverage_<size>``."""
        return {
            f"collection_coverage_{size}": value
            for size, value in zip(
                self.region_sizes, self._observed_by_size(), strict=True
            )
        }

    def plot(
        self,
        save_path: Path | str | None = None,
        title: str = "Collection Coverage",
        save_csv: bool = True,
    ):
        """Plot observed collection coverage vs region size and write a CSV."""
        if plt.get_backend().lower() != "agg":
            plt.switch_backend("Agg")
        observed = self._observed_by_size()

        if save_path is not None and save_csv:
            self._save_csv_data(save_path, observed)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axhline(self.coverage_level, color="grey", ls=":", label="Nominal")
        ax.plot(self.region_sizes, observed, "ko-", label="Observed")
        ax.set_xlabel("region size (sites summed)")
        ax.set_ylabel(f"collection coverage @ {self.coverage_level:g}")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return fig

    def _save_csv_data(self, save_path: Path | str, observed: list[float]) -> None:
        csv_path = Path(save_path).with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"region_size": self.region_sizes, "observed_mean": observed}
        ).to_csv(csv_path, index=False)
