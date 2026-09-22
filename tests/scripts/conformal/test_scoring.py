import numpy as np
import torch

from autocast.metrics.coverage import Coverage, MultiCoverage
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


def test_coverage_calibration_error_matches_repo_multicoverage():
    """Per-frame, per-channel absolute errors, as MultiCoverage -- nothing pooled."""
    torch.manual_seed(2)
    pred = torch.randn(6, 4, 3, 3, 3, 10)
    true = torch.randn(6, 4, 3, 3, 3)
    true[..., 0] *= 3.0  # channel 0 under-covered ...
    true[..., 1] *= 0.2  # ... channel 1 over-covered: the two orders now differ
    # channel 2 goes from over- to under-covered across the frames, so pooling
    # the frames before the absolute error would understate it
    true[..., 2] *= torch.linspace(0.2, 3.0, 4).view(1, 4, 1, 1)
    levels = [0.5, 0.8, 0.9]

    lower, upper = scoring.raw_interval_multi(pred, levels)
    value = scoring.coverage_calibration_error(true, lower, upper, levels)

    reference = MultiCoverage(coverage_levels=levels)
    reference.update(pred, true)
    assert abs(value - float(reference.compute())) < 1e-6


def test_window_row_reconstructable_from_per_frame_ingredients():
    """Any window rebuilt from per_frame_ingredients.csv == the direct value."""
    dump = make_synthetic_dump(b_total=30, n_frames=10, seed=2)
    fitted = fit_raw(dump["preds"])
    # over-covered early, under-covered late: a window's coverage error then
    # depends on taking the absolute error frame by frame
    true = dump["trues"] * torch.linspace(0.3, 2.5, 10).view(1, 10, 1, 1, 1)

    ingredients = scoring.per_frame_ingredients(fitted, true)

    for window in ((0, 1), (0, 4), (2, 7), (0, 10)):
        direct = scoring.window_row(fitted, true, window)
        for metric in ("nrmse", "vrmse", "crps", "ssr", "spread", "skill", "winkler"):
            rebuilt = scoring.reconstruct_window_metric(ingredients, window, metric)
            assert abs(rebuilt - direct[metric]) < 1e-4, metric
        rebuilt_coverage = scoring.reconstruct_window_coverage(ingredients, window)
        assert abs(rebuilt_coverage - direct["coverage"]) < 1e-6
        reference = MultiCoverage(coverage_levels=list(scoring.LEVELS))
        start, end = window
        reference.update(dump["preds"][:, start:end], true[:, start:end])
        assert abs(direct["coverage"] - float(reference.compute())) < 1e-6


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


def _reference_rank_histogram(true, ensemble):
    """July's ``field_autouq.py::rank_hist`` at one frame, pooled over all axes."""
    ranks = (ensemble < true.unsqueeze(-1)).sum(dim=-1).flatten()
    return torch.bincount(ranks, minlength=ensemble.shape[-1] + 1).numpy()


def _reference_excess_kurtosis(true, samples):
    """July's ``field_autouq.py`` excess kurtosis of z-scores at one frame."""
    std = samples.std(dim=-1).clamp_min(scoring._Z_STD_FLOOR)
    z = ((true - samples.mean(dim=-1)) / std).flatten().double().numpy()
    return ((z - z.mean()) ** 4).mean() / z.var() ** 2 - 3.0


def test_rank_histogram_per_frame_matches_reference():
    dump = make_synthetic_dump(b_total=10, n_frames=5, n_members=6, seed=5)
    trues, preds = dump["trues"], dump["preds"]
    histograms = scoring.rank_histogram_per_frame(trues, preds, frame_block_size=2)
    assert histograms.shape == (5, 7)
    for frame in range(5):
        np.testing.assert_array_equal(
            histograms[frame],
            _reference_rank_histogram(trues[:, frame], preds[:, frame]),
        )


def test_per_frame_excess_kurtosis_matches_reference():
    dump = make_synthetic_dump(b_total=10, n_frames=4, n_members=6, seed=6)
    trues, preds = dump["trues"], dump["preds"]
    kurtosis = scoring.per_frame_excess_kurtosis(trues, preds)
    for frame in range(4):
        reference = _reference_excess_kurtosis(trues[:, frame], preds[:, frame])
        assert abs(kurtosis[frame] - reference) < 1e-4


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


def test_spatial_mean_spread_skill_is_one_for_a_calibrated_ensemble():
    """Truth and members drawn alike -> ratio ~1 (the old formula gave ~1.25)."""
    generator = torch.Generator().manual_seed(7)
    field = torch.randn(400, 5, 1, 1, 2, 11, generator=generator)
    field = field.expand(-1, -1, 4, 4, -1, -1)  # spatially constant fields
    true, members = field[..., 0], field[..., 1:]
    ratio = scoring.spatial_mean_spread_skill(members, true)
    assert ratio.shape == (2,)
    assert torch.allclose(ratio, torch.ones(2), atol=0.05)
