"""Tests that the vectorised region tiling equals the per-region mask loop.

``CollectionCoverage.score`` used to rebuild a list of ``(d_total,)`` boolean
masks on every update, take one ``np.where`` and one host-to-device copy per
mask, then loop over regions in Python. At ``region_size=1`` on a 64x64x4 field
that is 16384 masks per update, and ~21760 loop iterations across the default
size sweep -- measured at 33.9 s per batch, a third of the whole per-timestep
metric pass.

``_region_index`` builds the same tiling arithmetically and the score gathers
every region at once. These tests pin that the replacement is exact by comparing
against ``_region_masks``, which is retained purely as the reference the loop
used to consume.
"""

import numpy as np
import pytest
import torch

from autocast.metrics.collection_coverage import (
    CollectionCoverage,
    _region_index,
    _region_masks,
)

LEVEL = 0.90


def _index_from_masks(spatial, n_channels, region_size):
    """The tiling as the old loop derived it: np.where over each boolean mask."""
    masks = _region_masks(spatial, n_channels, region_size)
    return [np.where(mask)[0] for mask in masks]


@pytest.mark.parametrize(
    ("spatial", "n_channels", "region_size"),
    [
        ((8, 8), 1, 1),
        ((8, 8), 1, 4),
        ((8, 8), 3, 4),
        ((8, 8), 4, 16),
        ((16, 8), 2, 64),
        ((12,), 1, 1),
        ((12,), 1, 4),
    ],
)
def test_region_index_matches_the_mask_loop(spatial, n_channels, region_size):
    """Same regions, same order, same sites within each region."""
    expected = _index_from_masks(spatial, n_channels, region_size)
    got = _region_index(spatial, n_channels, region_size)

    assert got.shape[0] == len(expected)
    for region, want in zip(got.tolist(), expected, strict=True):
        assert sorted(region) == sorted(want.tolist())


@pytest.mark.parametrize(
    ("spatial", "n_channels", "region_size"),
    [((8, 8), 1, 1), ((8, 8), 4, 4), ((8, 8), 2, 16), ((12,), 1, 4)],
)
def test_regions_partition_the_frame_exactly(spatial, n_channels, region_size):
    """Every site of the flattened frame belongs to exactly one region."""
    got = _region_index(spatial, n_channels, region_size).flatten().tolist()
    d_total = int(np.prod(spatial)) * n_channels
    assert sorted(got) == list(range(d_total))


def _loop_score(metric, y_pred, y_true):
    """The pre-2026-08-19 score(): a Python loop over per-region boolean masks."""
    y_pred_t, y_true_t = metric._check_input(y_pred, y_true)
    b, t = y_true_t.shape[0], y_true_t.shape[1]
    n_channels = y_true_t.shape[-1]
    spatial = tuple(y_true_t.shape[2:-1])
    m = y_pred_t.shape[-1]

    masks = _region_masks(spatial, n_channels, metric.region_size)
    mask_idx = [
        torch.as_tensor(np.where(mask)[0], device=y_pred_t.device, dtype=torch.long)
        for mask in masks
    ]
    d_total = int(np.prod(spatial)) * n_channels
    pred_flat = y_pred_t.reshape(b, t, d_total, m)
    true_flat = y_true_t.reshape(b, t, d_total)

    a = (1.0 - metric.coverage_level) / 2.0
    q = torch.tensor([a, 1.0 - a], device=y_pred_t.device, dtype=y_pred_t.dtype)

    per_region = []
    for idx in mask_idx:
        ens_sum = pred_flat[:, :, idx, :].sum(dim=2)
        truth_sum = true_flat[:, :, idx].sum(dim=2)
        quantiles = torch.quantile(ens_sum, q, dim=-1)
        covered = (truth_sum >= quantiles[0]) & (truth_sum <= quantiles[1])
        per_region.append(covered.float())
    coverage_bt = torch.stack(per_region, dim=0).mean(dim=0)
    return coverage_bt.unsqueeze(-1).expand(b, t, n_channels)


@pytest.mark.parametrize(
    ("spatial", "n_channels", "region_size"),
    [((8, 8), 1, 1), ((8, 8), 4, 4), ((8, 8), 4, 16), ((16, 8), 2, 64)],
)
def test_score_matches_the_loop_implementation(spatial, n_channels, region_size):
    """The vectorised score is numerically identical to the loop it replaced."""
    torch.manual_seed(0)
    b, t, members = 3, 2, 8
    y_pred = torch.randn(b, t, *spatial, n_channels, members, dtype=torch.float64)
    y_true = torch.randn(b, t, *spatial, n_channels, dtype=torch.float64)

    metric = CollectionCoverage(
        coverage_level=LEVEL, region_size=region_size, reduce_all=False
    )
    torch.testing.assert_close(
        metric.score(y_pred, y_true),
        _loop_score(metric, y_pred, y_true),
        rtol=0.0,
        atol=0.0,
    )


def test_tiling_is_built_once_and_reused():
    """The tiling is cached: repeated updates must not rebuild it.

    Rebuilding was the cost, so a cache that silently misses would restore it
    without changing any number. Pinned by identity, not by timing.
    """
    torch.manual_seed(0)
    y_pred = torch.randn(2, 1, 8, 8, 2, 8)
    y_true = torch.randn(2, 1, 8, 8, 2)

    metric = CollectionCoverage(coverage_level=LEVEL, region_size=4, reduce_all=False)
    metric.score(y_pred, y_true)
    first = metric._index
    metric.score(y_pred, y_true)

    assert first is not None
    assert metric._index is first


def test_bad_tiling_still_raises():
    """The arithmetic path keeps the guards the mask path had."""
    torch.manual_seed(0)
    y_pred = torch.randn(1, 1, 8, 8, 1, 4)
    y_true = torch.randn(1, 1, 8, 8, 1)
    metric = CollectionCoverage(region_size=9, reduce_all=False)
    with pytest.raises(ValueError, match="does not tile"):
        metric.score(y_pred, y_true)


def test_multichannel_1d_still_rejected():
    """The 1-D interleaved-layout guard survives the rewrite."""
    with pytest.raises(ValueError, match="single channel only"):
        _region_index((12,), 2, 4)
