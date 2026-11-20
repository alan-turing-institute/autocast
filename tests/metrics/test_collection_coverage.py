"""Tests for the collection (regional-sum) coverage metric.

The headline diagnostic: a spatially-white ensemble with the right per-site
marginal still under-covers regional sums (the correlation-structure miss), while
a spatially-coherent ensemble of the same marginal does not. The metric must also
agree numerically with the harness ``patch_coverage`` kernel it ports.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from autocast.metrics.collection_coverage import (
    CollectionCoverage,
    MultiCollectionCoverage,
    default_region_sizes,
)
from autocast.metrics.coverage import Coverage

# Import the harness reference kernel (git-ignored cal_study tree) directly by
# path so the production metric is gated against the exact semantics it ports.
_DIFF_SHARED = (
    pathlib.Path(__file__).resolve().parents[2]
    / "local_hydra"
    / "local_experiment"
    / "cal_study"
    / "_diff_shared.py"
)


def _load_diff_shared():
    if not _DIFF_SHARED.exists():
        return None
    spec = importlib.util.spec_from_file_location("_diff_shared_ref", _DIFF_SHARED)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collection_coverage_1d_rejects_multichannel():
    """1-D region masks support one channel only; multichannel must fail loudly.

    The interleaved flat layout would otherwise leave channels >= 1 uncovered, so
    the metric raises rather than silently mis-measuring (guard added in review).
    """
    metric = CollectionCoverage(region_size=1)
    y_pred = torch.randn(1, 1, 4, 2, 3)  # (B, T, S=4, C=2, M=3)
    y_true = torch.randn(1, 1, 4, 2)
    with pytest.raises(ValueError, match="single channel"):
        metric.update(y_pred, y_true)


def test_white_ensemble_undercovers_regional_sum():
    """A spatially-white ensemble with the right per-site marginal under-covers the
    regional sum, while a spatially-coherent ensemble of the same marginal does not
    -- the correlation-structure collapse in miniature (research piece sec. 6.6).

    The truth is drawn from a shared-mode + per-site-noise law (positive spatial
    correlation), so its whole-field sum is far more dispersed than the field
    size alone implies. The coherent ensemble shares that law (covers the sum);
    the white ensemble matches the per-site marginal but is spatially independent,
    so its summed members under-disperse and the truth lands in the tails.
    """
    torch.manual_seed(0)
    n, s, c, m = 256, 40, 1, 200
    sigma_g, sigma_e = 1.0, 0.5  # shared-mode and per-site noise std

    def _draw(num: int) -> torch.Tensor:
        g = sigma_g * torch.randn(num, 1)  # one shared mode per sample
        e = sigma_e * torch.randn(num, s)  # per-site noise
        return (g + e).view(num, 1, s, c)

    truth = _draw(n)[:, 0]  # (N, S, C) at T=1 below
    truth = truth.unsqueeze(1)  # (N, T=1, S, C)
    marginal_std = (sigma_g**2 + sigma_e**2) ** 0.5

    # coherent: members share the truth's shared-mode + per-site law.
    g_mem = sigma_g * torch.randn(n, 1, 1, c, m)
    e_mem = sigma_e * torch.randn(n, 1, s, c, m)
    coherent = g_mem + e_mem  # (N, 1, S, C, M), correct joint law, mean 0
    # white: iid per-site with the matched per-site marginal (no spatial coherence).
    white = marginal_std * torch.randn(n, 1, s, c, m)

    cc = CollectionCoverage(coverage_level=0.90, region_size=s)  # whole-field sum
    cov_coherent = float(cc(coherent, truth).item())
    cov_white = float(cc(white, truth).item())
    assert cov_coherent > 0.80, cov_coherent
    assert cov_white < 0.60, cov_white  # the regional-sum coverage collapse


def test_white_and_coherent_share_per_site_coverage():
    """Control: at region_size=1 the white and coherent ensembles both cover the
    per-site marginal -- the collapse is purely a joint (regional-sum) effect."""
    torch.manual_seed(5)
    n, s, c, m = 256, 40, 1, 200
    sigma_g, sigma_e = 1.0, 0.5
    g = sigma_g * torch.randn(n, 1)
    truth = (g + sigma_e * torch.randn(n, s)).view(n, 1, s, c)
    marginal_std = (sigma_g**2 + sigma_e**2) ** 0.5
    white = marginal_std * torch.randn(n, 1, s, c, m)
    g_mem = sigma_g * torch.randn(n, 1, 1, c, m)
    coherent = g_mem + sigma_e * torch.randn(n, 1, s, c, m)

    per_site = Coverage(coverage_level=0.90)
    assert abs(float(per_site(white, truth).item()) - 0.90) < 0.05
    assert abs(float(per_site(coherent, truth).item()) - 0.90) < 0.05


def test_per_site_region_matches_plain_coverage():
    """At region_size=1 the metric reduces to ordinary symmetric-quantile coverage."""
    torch.manual_seed(1)
    y_pred = torch.randn(4, 2, 12, 1, 32)
    y_true = torch.randn(4, 2, 12, 1)
    cc = CollectionCoverage(coverage_level=0.90, region_size=1)
    plain = Coverage(coverage_level=0.90)
    assert torch.allclose(cc(y_pred, y_true), plain(y_pred, y_true), atol=1e-6)


def test_matches_harness_patch_coverage():
    """The metric agrees with the harness ``patch_coverage`` on a fixed ensemble."""
    ref = _load_diff_shared()
    if ref is None:
        pytest.skip("harness _diff_shared not present")

    torch.manual_seed(2)
    n, s, c, m = 6, 40, 1, 48
    # harness layout: ens_l (N, *S, C, M), truth_l (N, *S, C)
    ens_l = torch.randn(n, s, c, m)
    truth_l = torch.randn(n, s, c)

    masks = ref.build_masks((s,), c)
    idx_by_size = {
        sz: [torch.tensor(np.where(mask)[0]) for mask in ms] for sz, ms in masks.items()
    }
    harness_cov = ref.patch_coverage(ens_l, truth_l, idx_by_size, lv=0.9)

    # production layout adds a singleton time axis: (B, T=1, S, C[, M])
    y_pred = ens_l.unsqueeze(1)
    y_true = truth_l.unsqueeze(1)
    for sz, expected in harness_cov.items():
        cc = CollectionCoverage(coverage_level=0.90, region_size=int(sz))
        got = float(cc(y_pred, y_true).item())
        assert abs(got - float(expected)) < 1e-5, f"size {sz}: {got} vs {expected}"


def test_2d_square_tiles_match_harness():
    """2-D square-tile regions match the harness patch coverage on a torus field."""
    ref = _load_diff_shared()
    if ref is None:
        pytest.skip("harness _diff_shared not present")

    torch.manual_seed(3)
    n, ns, c, m = 5, 8, 1, 40
    ens_l = torch.randn(n, ns, ns, c, m)  # (N, H, W, C, M)
    truth_l = torch.randn(n, ns, ns, c)

    masks = ref.build_masks((ns, ns), c)
    idx_by_size = {
        sz: [torch.tensor(np.where(mask)[0]) for mask in ms] for sz, ms in masks.items()
    }
    harness_cov = ref.patch_coverage(ens_l, truth_l, idx_by_size, lv=0.9)

    y_pred = ens_l.unsqueeze(1)  # (B, 1, H, W, C, M)
    y_true = truth_l.unsqueeze(1)
    for sz, expected in harness_cov.items():
        cc = CollectionCoverage(coverage_level=0.90, region_size=int(sz))
        got = float(cc(y_pred, y_true).item())
        assert abs(got - float(expected)) < 1e-5, f"size {sz}: {got} vs {expected}"


def test_rejects_indivisible_region_size():
    """A region size that does not tile the field is rejected at score time."""
    cc = CollectionCoverage(coverage_level=0.90, region_size=7)
    y_pred = torch.randn(2, 1, 40, 1, 16)
    y_true = torch.randn(2, 1, 40, 1)
    with pytest.raises(ValueError, match="tile"):
        cc(y_pred, y_true)


def test_multi_collection_coverage_per_size_csv(tmp_path):
    """MultiCollectionCoverage records one observed coverage per region size and
    writes a CSV with a per-region-size column."""
    torch.manual_seed(4)
    y_pred = torch.randn(6, 1, 40, 1, 32)
    y_true = torch.randn(6, 1, 40, 1)
    mcc = MultiCollectionCoverage(coverage_level=0.90, region_sizes=[1, 5, 40])
    mcc.update(y_pred, y_true)
    detailed = mcc.compute_detailed()
    assert set(detailed) == {
        "collection_coverage_1",
        "collection_coverage_5",
        "collection_coverage_40",
    }
    csv_path = tmp_path / "collection.csv"
    mcc.plot(save_path=str(csv_path.with_suffix(".png")))
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert list(df["region_size"]) == [1, 5, 40]
    assert "observed_mean" in df.columns


def test_default_region_sizes_1d_includes_unit_and_whole_field():
    """1-D default sweep is the field-length divisors, always with the per-site size
    (1) and the whole field (the length itself)."""
    sizes = default_region_sizes((40,))
    assert sizes == [1, 2, 4, 5, 8, 10, 20, 40]
    assert sizes[0] == 1
    assert sizes[-1] == 40


def test_default_region_sizes_2d_square_tiles():
    """2-D default sweep is square tiles whose side divides the grid (1, 4, 16, 64),
    capped so a 16x16 grid stops at 64-site tiles (no whole 256-site field)."""
    assert default_region_sizes((8, 8)) == [1, 4, 16, 64]
    assert default_region_sizes((16, 16)) == [1, 4, 16, 64]


def test_multi_collection_resolves_region_sizes_from_first_batch_1d():
    """region_sizes=None defers the sweep to the first batch's field shape (1-D)."""
    mcc = MultiCollectionCoverage(coverage_level=0.90, region_sizes=None)
    assert mcc.region_sizes == []  # unresolved until data arrives
    torch.manual_seed(0)
    y_pred = torch.randn(3, 1, 40, 1, 16)
    y_true = torch.randn(3, 1, 40, 1)
    mcc.update(y_pred, y_true)
    assert mcc.region_sizes == [1, 2, 4, 5, 8, 10, 20, 40]
    assert set(mcc.compute_detailed()) == {
        f"collection_coverage_{s}" for s in [1, 2, 4, 5, 8, 10, 20, 40]
    }


def test_multi_collection_resolves_region_sizes_from_first_batch_2d():
    """The lazy sweep also fires for a 2-D field (square tiles)."""
    mcc = MultiCollectionCoverage(coverage_level=0.90, region_sizes=None)
    torch.manual_seed(1)
    y_pred = torch.randn(2, 1, 8, 8, 1, 12)
    y_true = torch.randn(2, 1, 8, 8, 1)
    mcc.update(y_pred, y_true)
    assert mcc.region_sizes == [1, 4, 16, 64]


def test_multi_collection_explicit_region_sizes_not_overridden():
    """An explicit sweep is kept verbatim; the lazy default never fires."""
    mcc = MultiCollectionCoverage(coverage_level=0.90, region_sizes=[1, 40])
    torch.manual_seed(2)
    y_pred = torch.randn(3, 1, 40, 1, 16)
    y_true = torch.randn(3, 1, 40, 1)
    mcc.update(y_pred, y_true)
    assert mcc.region_sizes == [1, 40]
