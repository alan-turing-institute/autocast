import json

import pandas as pd
import torch

from autocast.scripts.conformal import calibrate
from autocast.scripts.conformal.data import CalibrationSource, TestSource

from .conftest import make_grouped_dump, make_synthetic_dump, save_dump


def test_calibrate_writes_full_layout_for_all_four_combinations(tmp_path):
    """End-to-end: all four `calib-X__test-Y/` subtrees, every fixed file."""
    new_path = tmp_path / "new.pt"
    paper_valid_path = tmp_path / "paper_valid.pt"
    paper_test_path = tmp_path / "paper_test.pt"

    # b_total=90 so the default 50-trajectory test split still leaves a
    # 40-trajectory calibration pool -- comfortably above the conformal
    # per-alpha floor for every level down to 0.05 (needs >= 19).
    save_dump(make_synthetic_dump(b_total=90, n_frames=8, seed=20), new_path)
    save_dump(make_synthetic_dump(b_total=20, n_frames=8, seed=21), paper_valid_path)
    save_dump(make_synthetic_dump(b_total=20, n_frames=8, seed=22), paper_test_path)

    out_dir = tmp_path / "eval_conformal"
    calibrate.calibrate(
        new_path=new_path,
        paper_valid_path=paper_valid_path,
        paper_test_path=paper_test_path,
        out_dir=out_dir,
        device="cpu",  # tiny synthetic data -- GPU launch overhead dominates
    )

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["split"]["balanced_by_scalars"] is False
    assert len(manifest["split"]["new_calibration_idx"]) == 40
    assert len(manifest["split"]["new_test_idx"]) == 50

    for calibration_source in CalibrationSource:
        for test_source in TestSource:
            combo_dir = (
                out_dir / f"calib-{calibration_source.value}__test-{test_source.value}"
            )
            assert combo_dir.is_dir()
            for method_name in ("raw", "EMOS", "conformal"):
                method_dir = combo_dir / method_name
                assert (method_dir / "rollout_metrics.csv").exists()
                assert (
                    method_dir / "rollout_metrics_per_timestep_channel_all.csv"
                ).exists()
                assert (method_dir / "per_frame_ingredients.csv").exists()
            assert (combo_dir / "coverage_map.pt").exists()
            assert (combo_dir / "bands.pt").exists()
            assert (combo_dir / "calibrator.pt").exists()
            assert (combo_dir / "sample_fields.pt").exists()
            assert (combo_dir / "dependence.csv").exists()
            assert (combo_dir / "summary.csv").exists()

    calibrator = torch.load(
        out_dir / "calib-new__test-new" / "calibrator.pt", weights_only=False
    )
    assert calibrator["conformal_multiplier"].shape == (8, 4, 4, 2)
    assert calibrator["emos"]["per"] == ("time",)


def test_calibrate_with_balanced_split(tmp_path):
    new_path = tmp_path / "new.pt"
    paper_valid_path = tmp_path / "paper_valid.pt"
    paper_test_path = tmp_path / "paper_test.pt"

    # Paper valid/test sets are 20 (24 for Gray-Scott) trajectories in
    # production -- comfortably above the conformal per-alpha floor for
    # every level down to 0.05 (needs >= 19); mirror that here.
    save_dump(make_grouped_dump(seed=30), new_path)
    save_dump(
        make_synthetic_dump(b_total=24, n_frames=4, height=3, width=3, n_channels=1),
        paper_valid_path,
    )
    save_dump(
        make_synthetic_dump(b_total=24, n_frames=4, height=3, width=3, n_channels=1),
        paper_test_path,
    )

    out_dir = tmp_path / "eval_conformal"
    calibrate.calibrate(
        new_path=new_path,
        paper_valid_path=paper_valid_path,
        paper_test_path=paper_test_path,
        out_dir=out_dir,
        balance_by_scalars=True,
        device="cpu",  # tiny synthetic data -- GPU launch overhead dominates
    )

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["split"]["balanced_by_scalars"] is True
    assert len(manifest["split"]["new_calibration_idx"]) == 6 * 17
    assert len(manifest["split"]["new_test_idx"]) == 6 * 8
    assert (out_dir / "calib-new__test-new" / "raw" / "rollout_metrics.csv").exists()


def test_run_combination_is_reproducible_across_runs(tmp_path):
    """Same seed, same inputs -> identical EMOS-sampled diagnostics.

    `.sample()` (EMOS's own and the EMOS+ECC composition's) draws from an
    unseeded generator by default; `run_combination`'s `sample_seed` fixes
    this. Covers `rollout_metrics.csv` (EMOS's crps/ssr, which read
    `.sample()`), `rank_histogram.csv`, and `dependence.csv` (indep/ecc SSR).
    """
    calibration = make_synthetic_dump(b_total=30, n_frames=6, seed=50)
    test = make_synthetic_dump(b_total=20, n_frames=6, seed=51)

    def run_once(out_dir):
        calibrate.run_combination(
            out_dir,
            true_cal=calibration["trues"],
            pred_cal=calibration["preds"],
            true_test=test["trues"],
            pred_test=test["preds"],
            sample_seed=123,
        )

    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    run_once(first_dir)
    run_once(second_dir)

    for relative_path in (
        "EMOS/rollout_metrics.csv",
        "EMOS/rank_histogram.csv",
        "dependence.csv",
    ):
        first = pd.read_csv(first_dir / relative_path)
        second = pd.read_csv(second_dir / relative_path)
        pd.testing.assert_frame_equal(first, second)
