"""Regression checks for frozen-cohort RB diagnostic completion."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from autocast.metrics.ensemble import EnergyScore

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from slurm_scripts.comparison.the_well.rayleigh_benard import (  # noqa: E402
    complete_diagnostics as runner,
)
from slurm_scripts.comparison.the_well.rayleigh_benard import (  # noqa: E402
    plot_diagnostics as plots,
)
from slurm_scripts.comparison.the_well.rayleigh_benard.paired_test_windows import (  # noqa: E402
    make_selection,
)


@pytest.mark.parametrize("spatial_shape", [(64,), (8, 8)])
def test_chunked_scores_match_single_frame_reference_and_joint_window_differs(
    spatial_shape,
):
    torch.manual_seed(29)
    truth = torch.randn(1, 4, *spatial_shape, 2)
    prediction = torch.randn(1, 4, *spatial_shape, 2, 16)
    scores = runner.score_leads(prediction, truth)
    chunked = runner.score_leads(prediction, truth, chunk_size=3)
    for name in scores:
        np.testing.assert_allclose(scores[name], chunked[name], rtol=1e-6, atol=1e-7)
    expected = np.concatenate(
        [
            EnergyScore()
            .score(prediction[:, i : i + 1], truth[:, i : i + 1])[0]
            .numpy()
            for i in range(4)
        ]
    )
    np.testing.assert_allclose(scores["energy"], expected)
    joint = EnergyScore().score(prediction, truth)[0].numpy()
    assert not np.allclose(joint, scores["energy"].mean(axis=0))


def test_energy_for_collapsed_ensemble_is_euclidean_error_per_lead():
    truth = torch.randn(1, 3, 64, 2)
    mean = torch.randn_like(truth)
    prediction = mean.unsqueeze(-1).expand(*mean.shape, 16)
    scores = runner.score_leads(prediction, truth)
    expected = torch.linalg.vector_norm(mean - truth, dim=2)[0].numpy()
    np.testing.assert_allclose(scores["energy"], expected, rtol=1e-6)


@pytest.mark.parametrize("spatial_shape", [(512 * 128,), (512, 128), (128, 512)])
def test_native_resolution_channels_and_members_are_required(spatial_shape):
    truth = SimpleNamespace(shape=(1, 4, *spatial_shape, 4))
    prediction = SimpleNamespace(shape=(*truth.shape, 16))
    runner.validate_prediction(prediction, truth, frames=4)
    for wrong_shape in [(1, 4, 64 * 64, 4, 16), (1, 4, 512 * 128, 4, 8)]:
        with pytest.raises(ValueError, match="native 512x128"):
            runner.validate_prediction(
                SimpleNamespace(shape=wrong_shape), truth, frames=4
            )


def test_standard_error_uses_trajectories_not_pixels_windows_or_channels():
    rows = []
    for trajectory, value in enumerate([1.0, 3.0, 5.0]):
        for lead in [1, 2]:
            for channel in range(4):
                rows.append(
                    {
                        "trajectory_idx": trajectory,
                        "lead": lead,
                        "channel": channel,
                        **dict.fromkeys(runner.METRICS + plots.COVERAGES, value),
                    }
                )
    frame = pd.DataFrame(rows)
    overall = plots.trajectory_stats(frame, []).iloc[0]
    expected_se = np.std([1.0, 3.0, 5.0], ddof=1) / np.sqrt(3)
    assert overall.vrmse_v2_mean == 3
    assert overall.vrmse_v2_se == pytest.approx(expected_se)
    per_lead = plots.trajectory_stats(frame, ["lead"])
    np.testing.assert_allclose(per_lead.vrmse_v2_se, expected_se)
    # Duplicating all dependent rows must not shrink the uncertainty band.
    doubled = plots.trajectory_stats(pd.concat([frame, frame]), []).iloc[0]
    assert doubled.vrmse_v2_se == pytest.approx(expected_se)


def test_calibration_mae_se_retains_cross_level_covariance():
    levels = np.array(runner.LEVELS)
    rows = []
    for trajectory, shift in enumerate([-0.01, 0.01, 0.03, 0.05]):
        rows.append(
            {
                "trajectory_idx": trajectory,
                **dict(zip(plots.COVERAGES, levels - 0.1 + shift, strict=True)),
            }
        )
    mean, se = plots.calibration_mae_stats(pd.DataFrame(rows))
    assert mean == pytest.approx(0.08)
    assert se == pytest.approx(np.std([-0.01, 0.01, 0.03, 0.05], ddof=1) / 2)


def test_archived_fm_sampler_and_raw_weight_policy_are_guarded():
    cfg = OmegaConf.create(
        {
            "eval": {"mode": "encode_once", "use_ema": False},
            "datamodule": {
                "well_dataset_name": "rayleigh_benard",
                "use_normalization": False,
                "n_steps_input": 1,
                "n_steps_output": 4,
                "min_dt_stride": 1,
                "max_dt_stride": 1,
            },
            "model": {
                "encoder": {},
                "processor": {
                    "_target_": "FlowMatchingProcessor",
                    "integrator": "ab3",
                    "flow_ode_steps": 16,
                },
            },
        }
    )
    entry = {"mode": "encode_once"}
    runner.validate_config(cfg, entry)
    cfg.model.processor.integrator = "euler"
    with pytest.raises(ValueError, match="sampler"):
        runner.validate_config(cfg, entry)
    cfg.model.processor.integrator = "ab3"
    cfg.eval.use_ema = True
    with pytest.raises(ValueError, match="raw weights"):
        runner.validate_config(cfg, entry)


def test_prepare_pins_inputs_generates_eval_only_scripts_and_refuses_overwrite(
    tmp_path, monkeypatch
):
    outputs = tmp_path / "outputs"
    archive = outputs / "archived"
    archive.mkdir(parents=True)
    checkpoint = archive / "frozen.ckpt"
    checkpoint.write_bytes(b"frozen test weights")
    stats = archive / "stats.yaml"
    stats.write_text("mean: 1\nstd: 2\n")
    cohort = yaml.safe_load((runner.HERE / "test64_seed0.yaml").read_text())
    (archive / "window_selection.yaml").write_text(
        yaml.safe_dump(make_selection(cohort))
    )
    config = {
        "eval": {"mode": "ambient", "use_ema": False, "checkpoint": str(checkpoint)},
        "datamodule": {
            "well_dataset_name": "rayleigh_benard",
            "use_normalization": True,
            "normalization_path": str(stats),
            "n_steps_input": 1,
            "n_steps_output": 4,
            "min_dt_stride": 1,
            "max_dt_stride": 1,
        },
        "model": {},
        "logging": {"wandb": {"enabled": True}},
    }
    (archive / "resolved_test_config.yaml").write_text(yaml.safe_dump(config))
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "ambient_test": {
                        "label": "Test",
                        "mode": "ambient",
                        "expected_step": 262144,
                        "config": "archived/resolved_test_config.yaml",
                        "checkpoint_sha256": runner.sha256(checkpoint),
                    }
                }
            }
        )
    )
    monkeypatch.setattr(
        runner,
        "source_record",
        lambda root: {"source_commit": "abc123", "source_root": str(root)},
    )
    args = argparse.Namespace(
        recipe=recipe,
        outputs_root=outputs,
        output=tmp_path / "campaign",
        raw_data_root=None,
        autoencoder_root=None,
        workers=2,
        horizon=180,
        account="test.account",
    )
    runner.prepare(args)
    manifest = json.loads((args.output / "manifest.json").read_text())
    entry = manifest["models"]["ambient_test"]
    assert entry["dependencies"][str(checkpoint)] == runner.sha256(checkpoint)
    pinned = OmegaConf.load(entry["config"])
    assert Path(pinned.datamodule.normalization_path).read_bytes() == stats.read_bytes()
    for phase in ["pilot", "production"]:
        script = args.output / f"ambient_test-{phase}.sbatch"
        subprocess.run(["bash", "-n", str(script)], check=True)
        text = script.read_text()
        assert "#SBATCH --gpus=1" in text
        assert "--time=" in text
        assert "sbatch " not in text
        assert " train " not in text
        assert ("--pilot" in text) == (phase == "pilot")
    with pytest.raises(FileExistsError):
        runner.prepare(args)


def test_comparison_refuses_missing_production_outputs(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": {"model": {}}, "horizon": 180}))
    with pytest.raises(ValueError, match="incomplete"):
        plots.load_results(manifest)


def test_full_comparison_export_and_changed_context_rejection(tmp_path):
    """Exercise complete plotting and provenance with synthetic production rows."""
    cohort = yaml.safe_load((runner.HERE / "test64_seed0.yaml").read_text())
    selection = make_selection(cohort)
    (tmp_path / "cohort.yaml").write_text(yaml.safe_dump(cohort))
    (tmp_path / "windows.yaml").write_text(yaml.safe_dump(selection))
    state = {
        "horizon": 100,
        "source_commit": "synthetic-test-only",
        "rollout_seed": 0,
        "paired_seed": 42,
        "cohort_sha256": runner.sha256(tmp_path / "cohort.yaml"),
        "windows_sha256": runner.sha256(tmp_path / "windows.yaml"),
        "models": {
            name: {"label": name, "expected_step": 262144}
            for name in ["ambient_test", "latent_fm_test"]
        },
    }
    manifest = tmp_path / "manifest.json"
    runner.save_json(manifest, state)
    for name in state["models"]:
        folder = tmp_path / name / "production"
        folder.mkdir(parents=True)
        for task, starts, leads in [("paired", 512, 4), ("rollout", 64, 100)]:
            position, lead, channel = np.meshgrid(
                np.arange(starts), np.arange(1, leads + 1), np.arange(4), indexing="ij"
            )
            position, lead, channel = position.ravel(), lead.ravel(), channel.ravel()
            identities = (
                selection["windows"]
                if task == "paired"
                else [{"trajectory_idx": i} for i in cohort["indices"]]
            )
            frame = pd.DataFrame(
                {
                    "selection_position": position,
                    "lead": lead,
                    "channel": channel,
                    **{
                        key: [identities[i][key] for i in position]
                        for key in identities[0]
                    },
                }
            )
            shift = frame.trajectory_idx.to_numpy() / 10000
            for metric in runner.METRICS:
                frame[metric] = 1 + shift
            for level, key in zip(runner.LEVELS, plots.COVERAGES, strict=True):
                frame[key] = level - 0.02 + shift
            frame.to_csv(folder / f"{task}.csv", index=False)
        runner.save_json(
            folder / "latency.json",
            {
                "seconds_per_16_member_forecast": [0.1] * 50,
                "gpu": "synthetic-test-GPU",
                "members": 16,
                "warmup": 5,
                "repeats": 50,
            },
        )
        runner.save_json(
            folder / "provenance.json",
            {
                "pilot": False,
                "manifest_sha256": runner.sha256(manifest),
                "members": 16,
                "global_step": 262144,
                "horizon": 100,
                "rollout_trajectories": cohort["indices"],
                "cohort_sha256": state["cohort_sha256"],
                "windows_sha256": state["windows_sha256"],
                "rollout_seed": 0,
                "paired_seed": 42,
                "files": {
                    f: runner.sha256(folder / f)
                    for f in ["paired.csv", "rollout.csv", "latency.json"]
                },
            },
        )
        (folder / "COMPLETE").touch()
    plots.build(manifest, tmp_path / "figures")
    assert len(list((tmp_path / "figures").glob("*.png"))) == 3
    assert (tmp_path / "figures/overall_mean_se.csv").is_file()
    assert (tmp_path / "figures/inference_latency.csv").is_file()
    # A count-preserving context change is still an unpaired comparison.
    folder = tmp_path / "latent_fm_test/production"
    frame = pd.read_csv(folder / "paired.csv")
    frame.loc[frame.selection_position.eq(0), "context_frame"] += 1
    frame.to_csv(folder / "paired.csv", index=False)
    record = json.loads((folder / "provenance.json").read_text())
    record["files"]["paired.csv"] = runner.sha256(folder / "paired.csv")
    runner.save_json(folder / "provenance.json", record)
    with pytest.raises(ValueError, match="Unpaired paired identities"):
        plots.load_results(manifest)
