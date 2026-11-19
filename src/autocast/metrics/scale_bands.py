"""One-step scores by band of cycles: error of the mean, spread / error, members' power.

The by-scale part of the one-step scorecard (owner 2026-10-24; bands D286), as an
eval metric so every ensemble eval of a square, periodic 2-D field reports it. Per
band of radial shells of a forward-normalised FFT (shell ``k`` holds radii
``[k - 0.5, k + 0.5)``, so the bands' sums add up to the field's mean square):

* ``error_rms``: root of the mean's squared error in the band, per field (all bands
  squared and summed give the mean square error of the ensemble mean);
* ``spread_error``: ``sqrt((M + 1) / (M - 1) * spread power / error power)``, the pooled
  corrected spread-skill ratio restricted to the band (1 = calibrated; over every
  shell it is :class:`~autocast.metrics.ensemble.PooledSpreadSkillRatio`);
* ``members_power``: members' power over the truth's (1 = members as rough as the
  truth in the band, below 1 = smoother);
* ``error_share`` / ``truth_share``: the band's share of all error / truth power.

Powers are summed over every case, lead and channel before any division (the pooled
form). Bands are inclusive ``[first, last]`` cycles; ``last: null`` runs to the grid's
corner shell. The default is the 64x64 bands of D286 (large 1-4, medium 5-16, fine
17 and up); a study passes its own through ``eval.metric_kwargs.scale_bands.bands``
(e.g. a term's range). Fields that are not square 2-D are skipped with a warning
(the eval then logs the metric as having no samples).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torchmetrics import Metric

from autocast.scripts.eval.swe_crps_spectra import compute_swe_power_spectra

log = logging.getLogger(__name__)

DEFAULT_BANDS: list[list[int | None]] = [[1, 4], [5, 16], [17, None]]
POWERS = ("error_power", "anomaly_power", "member_power", "truth_power")
#: (B, T, X, Y, C, M): the only layout the radial spectra take.
ENSEMBLE_2D_DIMS = 6


class ScaleBandScores(Metric):
    """Pooled one-step error, spread / error and members' power per band of cycles."""

    name: str = "scale_bands"
    full_state_update = False
    # States registered by ``add_state`` (one per POWERS entry, and ``fields``),
    # declared so a type checker knows them as tensors.
    error_power: Tensor
    anomaly_power: Tensor
    member_power: Tensor
    truth_power: Tensor
    fields: Tensor

    def __init__(self, bands: list[list[int | None]] | None = None) -> None:
        super().__init__()
        self.bands: list[tuple[int, int | None]] = []
        for first, last in bands if bands is not None else DEFAULT_BANDS:
            if first is None:
                msg = "a scale band needs its first cycle; only its last may be None"
                raise ValueError(msg)
            self.bands.append((int(first), None if last is None else int(last)))
        for first, last in self.bands:
            if first < 0 or (last is not None and last < first):
                msg = f"scale band [{first}, {last}] is not an ordered cycle range"
                raise ValueError(msg)
        for key in POWERS:
            self.add_state(
                key, default=torch.zeros(0, dtype=torch.float64), dist_reduce_fx="sum"
            )
        self.add_state(
            "fields",
            default=torch.tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.members = 0
        self._skipped = False

    def update(self, y_pred: Tensor, y_true: Tensor) -> None:
        if y_pred.ndim != ENSEMBLE_2D_DIMS or y_true.shape[2] != y_true.shape[3]:
            if not self._skipped:
                log.warning(
                    "scale_bands: skipped, fields are not square 2-D (%s)",
                    tuple(y_true.shape),
                )
                self._skipped = True
            return
        spectra = compute_swe_power_spectra(y_pred, y_true)
        for key in POWERS:
            summed = spectra[key].sum(dim=(0, 1, 2))  # (shells,)
            state = getattr(self, key)
            setattr(self, key, summed if state.numel() == 0 else state + summed)
        self.fields += y_true.shape[0] * y_true.shape[1] * y_true.shape[-1]
        self.members = int(y_pred.shape[-1])

    def _band_sums(self) -> dict[str, Tensor]:
        if self.truth_power.numel() == 0:
            msg = "scale_bands: no samples"
            raise RuntimeError(msg)
        top = self.truth_power.numel() - 1
        index = torch.arange(top + 1, device=self.truth_power.device)
        masks = [
            (index >= first) & (index <= (top if last is None else last))
            for first, last in self.bands
        ]
        masks.append(torch.ones_like(index, dtype=torch.bool))  # every shell
        return {
            key: torch.stack([getattr(self, key)[m].sum() for m in masks])
            for key in POWERS
        }

    def table(self) -> pd.DataFrame:
        """One row per band, then ``all`` (every shell, the field mean included)."""
        s = self._band_sums()
        m = self.members
        top = self.truth_power.numel() - 1
        tags = [f"{a}-{top if b is None else b}" for a, b in self.bands] + ["all"]
        return pd.DataFrame(
            {
                "band": tags,
                "error_rms": (s["error_power"] / self.fields).sqrt().tolist(),
                "spread_error": (
                    ((m + 1) / (m - 1)) * s["anomaly_power"] / s["error_power"]
                )
                .sqrt()
                .tolist(),
                "members_power": (s["member_power"] / s["truth_power"]).tolist(),
                "error_share": (s["error_power"] / s["error_power"][-1]).tolist(),
                "truth_share": (s["truth_power"] / s["truth_power"][-1]).tolist(),
            }
        )

    def compute(self) -> Tensor:
        """The every-shell spread / error (the pooled spread-skill ratio)."""
        return torch.tensor(float(self.table()["spread_error"].iloc[-1]))

    def plot(self, save_path: Path | str | None = None, title: str = "") -> None:
        """Write the band table as ``<save_path>.csv`` (no figure)."""
        del title
        if save_path is None:
            return
        csv_path = Path(save_path).with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.table().to_csv(csv_path, index=False)
