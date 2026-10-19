import json

import pandas as pd
import pytest
import torch
from autouq.calibrators import EMOS, AxisRole
from autouq.calibrators.mathutils import gaussian_crps

from autocast.scripts.conformal import calibrate, writers
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
    # paths relative to the output folder, so it can move between machines
    assert manifest["inputs"]["new"]["path"] == "../new.pt"
    assert manifest["inputs"]["paper_test"]["n_trajectories"] == 20

    readme = (out_dir / "README.md").read_text()
    assert "split once (at random, seed 20260709) into 40 for calibration" in readme
    for calibration_source in CalibrationSource:
        for test_source in TestSource:
            name = f"calib-{calibration_source.value}__test-{test_source.value}"
            assert f"- `{name}/`: calibrated on" in readme
    assert "Amended on" not in readme
    manifest["amendments"] = [
        {"date": "2026-09-22", "git_commit": "abc1234", "change": "Added rows."}
    ]
    writers.write_readme(out_dir, manifest)
    assert (
        "every split.\n\nAmended on 2026-09-22 at commit `abc1234`: Added rows."
        "\n\n## Forecasts"
    ) in (out_dir / "README.md").read_text()

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
            assert (combo_dir / "coverage_map_windows.pt").exists()
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


def _heterogeneous_frames(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Frames running from badly under- to badly over-dispersed, with drifting bias."""
    generator = torch.Generator().manual_seed(seed)
    b, t, h, w, c, m = 40, 30, 6, 6, 2, 8
    trues = torch.randn(b, t, h, w, c, generator=generator)
    spread = torch.logspace(-1.5, 1.2, t).view(1, t, 1, 1, 1, 1)
    bias = torch.linspace(-1.0, 1.0, t).view(1, t, 1, 1, 1)
    noise = torch.randn(b, t, h, w, c, m, generator=generator)
    return trues, (trues + bias).unsqueeze(-1) + spread * noise


def _calibration_crps_per_frame(emos, trues, preds):
    mu, sigma = emos._calibrated_mean_std(preds.double())
    return gaussian_crps(mu, sigma, trues.double()).mean(dim=(0, 2, 3, 4))


def test_fit_emos_per_frame_reaches_each_frames_optimum():
    """The joint `EMOS.calibrate` stops early on heterogeneous frames; this does not."""
    trues, preds = _heterogeneous_frames()
    joint = EMOS(per=(AxisRole.TIME,))
    joint.calibrate(trues, preds)
    per_frame = calibrate.fit_emos_per_frame(trues, preds)

    crps_joint = _calibration_crps_per_frame(joint, trues, preds)
    crps_frame = _calibration_crps_per_frame(per_frame, trues, preds)
    assert (crps_frame <= crps_joint + 1e-6).all()
    # the joint fit is measurably short of the optimum on some frame (27-30% here)
    assert ((crps_joint - crps_frame) / crps_frame).max() > 0.1


def test_fit_emos_per_frame_combines_the_single_frame_fits():
    trues, preds = _heterogeneous_frames(seed=1)
    combined = calibrate.fit_emos_per_frame(trues, preds)
    state = calibrate.emos_state_dict(combined)
    beta0 = state["beta0"]
    assert isinstance(beta0, torch.Tensor)
    assert beta0.shape == (trues.shape[1],)

    frame = 7  # the reference is fit the way the function fits: in float64
    single = EMOS(per=(AxisRole.TIME,))
    single.calibrate(
        trues[:, frame : frame + 1].double(), preds[:, frame : frame + 1].double()
    )
    for key, value in calibrate.emos_state_dict(single).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(state[key][frame : frame + 1], value.float())
    torch.testing.assert_close(
        combined.predict(preds, alphas=0.1)[:, frame : frame + 1],
        single.predict(preds[:, frame : frame + 1], alphas=0.1).float(),
    )


def test_fit_emos_per_frame_does_not_stall_in_float32():
    """Float32 inputs reach the same per-frame optimum as float64 inputs.

    Fitting these frames in float32 left 2 of 8 frames' ``gamma0`` at its
    initial value (calibration CRPS 0.2192 vs 0.2141 at frame 1).
    """
    data = make_synthetic_dump(
        b_total=25, n_frames=8, height=16, width=16, n_channels=2, n_members=6, seed=60
    )
    trues, preds = data["trues"], data["preds"]
    from_f32 = calibrate.fit_emos_per_frame(trues, preds)
    from_f64 = calibrate.fit_emos_per_frame(trues.double(), preds.double())
    beta0 = calibrate.emos_state_dict(from_f32)["beta0"]
    assert isinstance(beta0, torch.Tensor)
    assert beta0.dtype == torch.float32
    torch.testing.assert_close(
        _calibration_crps_per_frame(from_f32, trues, preds),
        _calibration_crps_per_frame(from_f64, trues, preds),
        rtol=1e-5,
        atol=0.0,
    )


def test_emos_state_dict_refuses_an_unfitted_emos():
    with pytest.raises(RuntimeError, match="calibrate must be called"):
        calibrate.emos_state_dict(EMOS(per=(AxisRole.TIME,)))


def test_calibrate_balanced_split_needs_constant_scalars(tmp_path):
    for name in ("new", "paper_valid", "paper_test"):
        save_dump(make_synthetic_dump(b_total=30, n_frames=2), tmp_path / f"{name}.pt")
    with pytest.raises(ValueError, match="requires constant_scalars"):
        calibrate.calibrate(
            new_path=tmp_path / "new.pt",
            paper_valid_path=tmp_path / "paper_valid.pt",
            paper_test_path=tmp_path / "paper_test.pt",
            out_dir=tmp_path / "out",
            balance_by_scalars=True,
            device="cpu",
        )
