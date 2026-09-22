import numpy as np
import torch

from autocast.metrics.coverage import Coverage
from autocast.metrics.ensemble import WinklerScore
from autocast.scripts.conformal import scoring
from autocast.scripts.conformal.calibrate import fit_conformal, fit_raw

from .conftest import make_synthetic_dump


def test_band_coverage_matches_repo_coverage_class_on_raw_interval():
    """band_coverage on the raw ensemble's own quantile interval == Coverage."""
    torch.manual_seed(0)
    pred = torch.randn(6, 4, 3, 3, 2, 10)
    true = torch.randn(6, 4, 3, 3, 2)
    alpha = 0.1
    level = round(1 - alpha, 2)

    lower, upper = scoring.raw_interval_multi(pred, [level])
    band_value = float(scoring.band_coverage(true, lower[..., 0], upper[..., 0]).mean())

    reference = Coverage(coverage_level=level)
    reference.update(pred, true)
    reference_value = float(reference.compute())

    assert abs(band_value - reference_value) < 1e-6


def test_band_winkler_matches_repo_winkler_class_on_raw_interval():
    """band_winkler on the raw ensemble's own quantile interval == WinklerScore."""
    torch.manual_seed(1)
    pred = torch.randn(6, 4, 3, 3, 2, 10)
    true = torch.randn(6, 4, 3, 3, 2)
    alpha = 0.1
    level = round(1 - alpha, 2)

    lower, upper = scoring.raw_interval_multi(pred, [level])
    band_value = float(
        scoring.band_winkler(true, lower[..., 0], upper[..., 0], alpha).mean()
    )

    reference = WinklerScore(alpha=alpha)
    reference.update(pred, true)
    reference_value = float(reference.compute())

    assert abs(band_value - reference_value) < 1e-6


def test_window_row_reconstructable_from_per_frame_ingredients():
    """Any window rebuilt from per_frame_ingredients.csv == the direct value."""
    dump = make_synthetic_dump(b_total=30, n_frames=10, seed=2)
    fitted = fit_raw(dump["preds"])
    true = dump["trues"]

    ingredients = scoring.per_frame_ingredients(fitted, true)

    for window in ((0, 1), (0, 4), (2, 7), (0, 10)):
        direct = scoring.window_row(fitted, true, window)
        for metric in ("nrmse", "vrmse", "crps", "ssr", "spread", "skill", "winkler"):
            rebuilt = scoring.reconstruct_window_metric(ingredients, window, metric)
            assert abs(rebuilt - direct[metric]) < 1e-4, metric
        rebuilt_coverage = scoring.reconstruct_window_coverage(ingredients, window)
        assert abs(rebuilt_coverage - direct["coverage"]) < 1e-6


def test_conformal_coverage_close_to_nominal_on_exchangeable_data():
    """Exchangeable, well-calibrated synthetic data -> ~90% conformal coverage."""
    calibration = make_synthetic_dump(b_total=200, n_frames=4, seed=3)
    test = make_synthetic_dump(b_total=100, n_frames=4, seed=4)

    fitted, _ = fit_conformal(calibration["trues"], calibration["preds"], test["preds"])
    level_index = scoring.LEVELS.index(scoring.NOMINAL_LEVEL)
    lower = fitted.lower[..., level_index]
    upper = fitted.upper[..., level_index]

    observed = float(scoring.band_coverage(test["trues"], lower, upper).mean())
    assert abs(observed - scoring.NOMINAL_LEVEL) < 0.05


def test_rank_histogram_shape():
    dump = make_synthetic_dump(b_total=10, n_frames=1, n_members=6, seed=5)
    histogram = scoring.rank_histogram(dump["trues"][:, 0], dump["preds"][:, 0])
    assert histogram.shape == (7,)
    assert histogram.sum() == dump["trues"][:, 0].numel()


def test_excess_kurtosis_zero_for_gaussian():
    generator = np.random.default_rng(0)
    z = generator.standard_normal(200_000)
    assert abs(scoring.excess_kurtosis(z)) < 0.05


def test_ecc_verdict_clean_and_degenerate():
    assert scoring.ecc_verdict(raw=1.0, indep=0.5, ecc=1.0) == "CLEAN ECC win"
    assert scoring.ecc_verdict(raw=1.0, indep=0.9, ecc=1.0) == "degenerate"
    assert scoring.ecc_verdict(raw=1.0, indep=0.5, ecc=2.0) == "degenerate"


def test_spatial_mean_spread_skill_shape():
    dump = make_synthetic_dump(b_total=12, n_frames=4, n_channels=3, seed=6)
    result = scoring.spatial_mean_spread_skill(dump["preds"], dump["trues"])
    assert result.shape == (3,)
    assert torch.all(result > 0)


def test_bootstrap_summary_has_boot_std_keys():
    dump = make_synthetic_dump(b_total=20, n_frames=4, seed=7)
    fitted = fit_raw(dump["preds"])
    summary = scoring.bootstrap_summary(fitted, dump["trues"], n_boot=10, seed=0)
    for key in ("coverage_90", "winkler_90", "nrmse", "vrmse", "crps", "ssr"):
        assert key in summary
        assert f"{key}_boot_std" in summary
        assert summary[f"{key}_boot_std"] >= 0
