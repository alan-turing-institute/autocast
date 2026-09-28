import json

import numpy as np
import pandas as pd
import torch

from autocast.metrics.coverage import MultiCoverage
from autocast.scripts.conformal import scoring, writers
from autocast.scripts.conformal.calibrate import fit_conformal, fit_emos, fit_raw
from autocast.scripts.conformal.scoring import WINDOWS
from autocast.utils.plots import compute_metrics_per_timestep_from_dataloader

from .conftest import make_synthetic_dump

#: Headers of the eval's own ``rollout_metrics.csv`` and
#: ``rollout_coverage_window_<w>.csv``, embedded so the layout tests do not
#: depend on files outside this repo.
PAPER_ROLLOUT_METRICS_HEADER = ["window", "batch_idx"]
PAPER_COVERAGE_WINDOW_HEADER = ["coverage_level", "observed_mean", "channel_0"]


def _fit_all(calibration, test):
    raw = fit_raw(test["preds"])
    emos, _ = fit_emos(calibration["trues"], calibration["preds"], test["preds"])
    conformal, _ = fit_conformal(
        calibration["trues"], calibration["preds"], test["preds"]
    )
    return {"raw": raw, "EMOS": emos, "conformal": conformal}


def test_write_method_outputs_layout_and_headers(tmp_path):
    # n_frames=100 matches the real T=100 rollout so every configured window
    # (up to 65-99) survives unclipped -- this is the layout/header
    # test, so it deliberately exercises the full WINDOWS list rather than
    # the smaller T used for speed elsewhere.
    n_frames = 100
    calibration = make_synthetic_dump(b_total=40, n_frames=n_frames, seed=10)
    test = make_synthetic_dump(b_total=20, n_frames=n_frames, seed=11)
    fitted_by_method = _fit_all(calibration, test)

    for name, fitted in fitted_by_method.items():
        method_dir = tmp_path / name
        writers.write_method_outputs(method_dir, fitted, test["trues"])

        rollout_metrics = pd.read_csv(method_dir / "rollout_metrics.csv")
        assert list(rollout_metrics.columns[:2]) == PAPER_ROLLOUT_METRICS_HEADER
        assert set(rollout_metrics["batch_idx"]) == {"all"}
        assert set(rollout_metrics["window"]) == {
            f"{start}-{end}" for start, end in WINDOWS
        }

        for start, end in WINDOWS:
            path = method_dir / f"rollout_coverage_window_{start}-{end}.csv"
            assert path.exists()
            coverage_window = pd.read_csv(path)
            assert list(coverage_window.columns[:3]) == PAPER_COVERAGE_WINDOW_HEADER

        per_timestep = pd.read_csv(
            method_dir / "rollout_metrics_per_timestep_channel_all.csv", index_col=0
        )
        assert per_timestep.shape[1] == n_frames
        if fitted.samples is not None:
            assert "exkurt" in per_timestep.index

        ingredients = pd.read_csv(method_dir / "per_frame_ingredients.csv")
        assert len(ingredients) == n_frames

        rank_histogram_path = method_dir / "rank_histogram.csv"
        if fitted.samples is not None:
            assert rank_histogram_path.exists()
            rank_df = pd.read_csv(rank_histogram_path, index_col=0)
            assert rank_df.shape[1] == fitted.samples.shape[-1] + 1
        else:
            assert not rank_histogram_path.exists()


def test_per_timestep_coverage_rows_match_eval_and_ingredients(tmp_path):
    """``coverage_<level>`` rows: the eval's names and values, rebuildable exactly."""
    n_frames = 6
    dump = make_synthetic_dump(b_total=20, n_frames=n_frames, seed=14)
    # under-covered early, over-covered late, so each frame's value differs
    true = dump["trues"] * torch.linspace(2.0, 0.3, n_frames).view(1, -1, 1, 1, 1)
    fitted = fit_raw(dump["preds"])
    writers.write_method_outputs(tmp_path, fitted, true)

    per_timestep = pd.read_csv(
        tmp_path / "rollout_metrics_per_timestep_channel_all.csv", index_col=0
    )
    reference = compute_metrics_per_timestep_from_dataloader(
        dataloader=[None],
        metric_fns={"coverage": MultiCoverage},
        predict_fn=lambda _: (dump["preds"], true),
    )
    assert [row for row in per_timestep.index if row.startswith("coverage_")] == list(
        reference
    )
    ingredients = pd.read_csv(tmp_path / "per_frame_ingredients.csv")
    rebuilt = writers.per_level_coverage_rows(
        scoring.observed_coverage_from_ingredients(ingredients)
    )
    for name, values in reference.items():
        written = per_timestep.loc[name].to_numpy()
        np.testing.assert_allclose(written, values.mean(axis=1), atol=1e-6)
        np.testing.assert_allclose(rebuilt[name], written, rtol=0, atol=1e-12)


def test_write_coverage_map_and_bands(tmp_path):
    calibration = make_synthetic_dump(b_total=40, n_frames=6, seed=12)
    test = make_synthetic_dump(b_total=20, n_frames=6, seed=13)
    fitted_by_method = _fit_all(calibration, test)
    level_index = scoring.LEVELS.index(scoring.NOMINAL_LEVEL)

    coverage_by_method = {
        name: scoring.coverage_map_slice(
            test["trues"],
            fitted.lower[..., level_index],
            fitted.upper[..., level_index],
        )
        for name, fitted in fitted_by_method.items()
    }
    coverage_by_method_window = {
        name: scoring.coverage_map_by_window(
            test["trues"],
            fitted.lower[..., level_index],
            fitted.upper[..., level_index],
        )
        for name, fitted in fitted_by_method.items()
    }
    writers.write_coverage_map(tmp_path, coverage_by_method, coverage_by_method_window)
    coverage_map = torch.load(tmp_path / "coverage_map.pt", weights_only=False)
    assert set(coverage_map) == {"raw", "EMOS", "conformal"}
    for value in coverage_map.values():
        assert value.shape == test["trues"].shape[2:]
    windows_map = torch.load(tmp_path / "coverage_map_windows.pt", weights_only=False)
    assert set(windows_map) == {"raw", "EMOS", "conformal"}
    for name, by_label in windows_map.items():
        # T=6: (0,1), (0,4) and (6,12)->(6,6) empty, so only two windows.
        assert list(by_label) == ["0-1", "0-4"]
        torch.testing.assert_close(
            by_label["0-4"], coverage_by_method_window[name][(0, 4)]
        )

    writers.write_bands(tmp_path, fitted_by_method["conformal"])
    bands = torch.load(tmp_path / "bands.pt", weights_only=False)
    assert bands["lower"].shape == test["trues"].shape
    assert bands["upper"].shape == test["trues"].shape
    assert torch.all(bands["upper"] >= bands["lower"])


def test_write_calibrator_moves_emos_tensors_to_cpu(tmp_path):
    """`calibrator.pt` must load on any machine, regardless of the fit device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conformal_multiplier = torch.ones(4, 3, 3, 2, device=device)
    emos_state = {
        "per": ("time",),
        "beta0": torch.zeros(4, device=device),
        "beta1": torch.ones(4, device=device),
    }

    writers.write_calibrator(tmp_path, conformal_multiplier, emos_state)

    calibrator = torch.load(tmp_path / "calibrator.pt", weights_only=False)
    assert calibrator["conformal_multiplier"].device.type == "cpu"
    assert calibrator["emos"]["beta0"].device.type == "cpu"
    assert calibrator["emos"]["beta1"].device.type == "cpu"
    assert calibrator["emos"]["per"] == ("time",)


def test_write_dependence_csv(tmp_path):
    raw_ssr = torch.tensor([1.0, 2.0])
    indep_ssr = torch.tensor([0.5, 1.9])
    ecc_ssr = torch.tensor([1.1, 3.0])

    writers.write_dependence_csv(tmp_path, ["smoke", "u"], raw_ssr, indep_ssr, ecc_ssr)
    rows = pd.read_csv(tmp_path / "dependence.csv")
    assert list(rows.columns) == ["channel", "raw", "indep", "ecc"]
    assert list(rows["channel"]) == ["smoke", "u"]
    np.testing.assert_allclose(rows["ecc"], [1.1, 3.0], rtol=1e-6)


def test_write_manifest(tmp_path):
    writers.write_manifest(tmp_path, {"a": 1})
    assert json.loads((tmp_path / "manifest.json").read_text()) == {"a": 1}
