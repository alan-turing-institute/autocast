"""By-scale metric: the every-shell row is the pooled ratio; the bands partition."""

from __future__ import annotations

import pytest
import torch

from autocast.metrics.ensemble import PooledSpreadSkillRatio
from autocast.metrics.scale_bands import ScaleBandScores


def test_every_shell_is_pooled_ratio_bands_partition_and_1d_is_skipped():
    generator = torch.Generator().manual_seed(0)
    truth = torch.randn(3, 1, 8, 8, 1, generator=generator, dtype=torch.float64)
    pred = 0.6 * truth[..., None] + torch.randn(
        3, 1, 8, 8, 1, 5, generator=generator, dtype=torch.float64
    )
    metric = ScaleBandScores(bands=[[0, 2], [3, None]])
    for half in (slice(0, 2), slice(2, 3)):  # two updates pool like one
        metric.update(pred[half], truth[half])
    pooled = PooledSpreadSkillRatio()  # one lead: it averages per-lead ratios
    pooled.update(pred, truth)
    table = metric.table()
    assert list(table["band"]) == ["0-2", "3-6", "all"]
    assert metric.compute().item() == pytest.approx(pooled.compute().item(), rel=1e-5)
    assert table["error_share"].iloc[:2].sum() == pytest.approx(1.0)
    mse = (pred.mean(-1) - truth).square().mean().item()
    assert table["error_rms"].iloc[-1] ** 2 == pytest.approx(mse)
    # Several leads and channels: the error is per field, not per case.
    wide_truth = torch.randn(2, 2, 8, 8, 2, generator=generator, dtype=torch.float64)
    wide_pred = torch.randn(2, 2, 8, 8, 2, 4, generator=generator, dtype=torch.float64)
    wide = ScaleBandScores()
    wide.update(wide_pred, wide_truth)
    wide_mse = (wide_pred.mean(-1) - wide_truth).square().mean().item()
    assert wide.table()["error_rms"].iloc[-1] ** 2 == pytest.approx(wide_mse)

    skipped = ScaleBandScores()
    skipped.update(torch.zeros(2, 1, 8, 1, 3), torch.zeros(2, 1, 8, 1))
    with pytest.raises(RuntimeError, match="no samples"):
        skipped.compute()


@pytest.mark.parametrize("bands", [[[None, 4]], [[5, 2]], [[-1, 3]]])
def test_a_band_without_an_ordered_first_cycle_is_refused(bands):
    with pytest.raises(ValueError, match="band"):
        ScaleBandScores(bands=bands)
