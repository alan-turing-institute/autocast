"""Check immutable campaign preparation without allocating GPUs or submitting."""

import importlib.util
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf, open_dict


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    path = (
        Path(__file__).resolve().parents[2]
        / "slurm_scripts/ablations/gray_scott_data_scaling.py"
    )
    spec = importlib.util.spec_from_file_location("gs_scaling", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_source_commit", lambda: "abc123")
    for name in (
        "AUTOCAST_GS_5X_DATA",
        "AUTOCAST_GS_REFERENCE_DATA",
        "AUTOCAST_GS_5X_CACHE",
        "AUTOCAST_GS_5X_AE_CHECKPOINT",
    ):
        monkeypatch.setenv(name, "unused")
    root = tmp_path / "campaign"
    state = module.prepare(root, tmp_path / "expanded", tmp_path / "original")
    return module, root, state


def test_preparation_keeps_reference_horizons_and_manual_dependencies(campaign):
    _, root, state = campaign
    assert {key for key, stage in state["stages"].items() if "epochs" in stage} == {
        "ae",
        "crps24",
        "crps48",
        "fm24",
    }
    assert state["stages"]["cache"]["requires"] == ["ae"]
    assert state["stages"]["fm24"]["requires"] == ["cache"]
    for name, epochs in (("ae", 512), ("crps24", 399), ("crps48", 798), ("fm24", 2830)):
        cfg = OmegaConf.load(state["stages"][name]["config"])
        assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == epochs
        assert cfg.trainer.limit_train_batches == 0.2
        assert cfg.trainer.plugins[0].auto_requeue is False
        assert cfg.resume_weights_only is cfg.reset_resume_time_budget is False
        assert cfg.trainer.callbacks[-1]._target_.endswith("AllocationDeadline")
        script = (root / f"{name}.sh").read_text()
        assert "#SBATCH --nodes=1" in script
        assert "#SBATCH --gpus=4" in script
        assert "#SBATCH --time=24:00:00" in script
        assert "exec srun uv run --frozen --no-sync" in script
        assert "scontrol" not in script
    assert (
        OmegaConf.load(state["stages"]["crps48"]["config"]).trainer.max_time
        == "02:00:00:00"
    )


def test_evaluation_uses_raw_cohorts_and_preserves_training_identity(campaign):
    _, root, state = campaign
    for parent in ("fm24", "crps24", "crps48"):
        for cohort, directory in (("original", "original"), ("expanded", "expanded")):
            stage = state["stages"][f"eval_{parent}_{cohort}"]
            cfg = OmegaConf.load(stage["config"])
            assert cfg.datamodule._target_.endswith("SpatioTemporalDataModule")
            assert cfg.datamodule.test_data_path == str(
                root.parent / directory / "test/data.pt"
            )
            assert cfg.datamodule.normalization_path == str(
                root.parent / "expanded/stats.yml"
            )
            assert cfg.datamodule.stride == 1
            assert cfg.eval.max_rollout_steps == 25
            assert cfg.eval.n_members == 10
            assert cfg.eval.devices == 1
            assert cfg.eval.trajectory_statistics.enabled
            assert stage["requires"] == [parent]
            if parent.startswith("fm"):
                assert cfg.eval.mode == "encode_once"
                assert cfg.autoencoder_checkpoint == str(root / "ae/autoencoder.ckpt")
                assert cfg.model.encoder.periodic


def test_changed_configuration_is_rejected(campaign):
    module, root, state = campaign
    path = Path(state["stages"]["ae"]["config"])
    path.write_text(path.read_text() + "\n# edited\n")
    with pytest.raises(RuntimeError, match="config changed"):
        module.load_stage(root / "campaign.json", "ae")


def test_repeated_prepare_and_unallocated_execution_are_rejected(campaign, monkeypatch):
    module, root, _ = campaign
    with pytest.raises(FileExistsError):
        module.prepare(root, root / "data", root / "reference")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="inside an allocation"):
        module.execute(root / "campaign.json", "ae", None)


def test_manual_continuation_refuses_completed_and_stale_checkpoints(campaign):
    module, _, state = campaign
    stage = state["stages"]["crps48"]
    checkpoint = Path(stage["checkpoint"])
    torch.save({"global_step": 10}, checkpoint)
    status = Path(stage["workdir"]) / "status.json"
    module._write(status, {"status": "complete", "global_step": 10})
    with pytest.raises(ValueError, match="already completed"):
        module._continuation_checkpoint(stage, checkpoint)
    identity = module._digest(checkpoint)
    module._write(
        status,
        {
            "status": "continuation_required",
            "global_step": 11,
            "checkpoint_sha256": identity,
        },
    )
    with pytest.raises(ValueError, match="latest recorded"):
        module._continuation_checkpoint(stage, checkpoint)
    module._write(
        status,
        {
            "status": "continuation_required",
            "global_step": 10,
            "checkpoint_sha256": identity,
        },
    )
    actual, step = module._continuation_checkpoint(stage, checkpoint)
    assert actual == checkpoint.resolve()
    assert step == 10


def test_continuation_rejects_another_run_at_the_same_step(campaign):
    module, _, state = campaign
    stage = state["stages"]["crps48"]
    checkpoint = Path(stage["checkpoint"])
    torch.save({"global_step": 10, "weight": torch.tensor(1.0)}, checkpoint)
    module._write(
        Path(stage["workdir"]) / "status.json",
        {
            "status": "continuation_required",
            "global_step": 10,
            "checkpoint_sha256": module._digest(checkpoint),
        },
    )
    other = Path(state["stages"]["crps24"]["checkpoint"])
    torch.save({"global_step": 10, "weight": torch.tensor(99.0)}, other)
    with pytest.raises(ValueError, match="checkpoint identity"):
        module._continuation_checkpoint(stage, other)
    other.write_bytes(checkpoint.read_bytes())
    assert module._continuation_checkpoint(stage, other) == (other.resolve(), 10)
    torch.save({"global_step": 10, "weight": torch.tensor(99.0)}, checkpoint)
    with pytest.raises(ValueError, match="checkpoint identity"):
        module._continuation_checkpoint(stage, checkpoint)


@pytest.fixture
def baseline_campaign(campaign):
    module, root, state = campaign
    inputs = root.parent / "baseline-inputs"
    inputs.mkdir()
    manifest = {}
    for name, source in (("fm", "fm24"), ("crps", "crps24"), ("ae", "cache")):
        cfg = OmegaConf.load(state["stages"][source]["config"])
        with open_dict(cfg):
            cfg.datamodule.data_path = str(root.parent / f"original-{name}")
            cfg.eval.csv_path = "/published/results.csv"
            cfg.eval.video_dir = "/published/videos"
            cfg.eval.rollout_snapshot_dir = "/published/snapshots"
            cfg.eval.trajectory_statistics.output_dir = "/published/statistics"
            cfg.datamodule.normalization_path = str(
                root.parent / f"original-{name}/stats.yml"
            )
            if name == "ae":
                cfg.model.encoder.periodic = False
        path = inputs / f"{name}.yaml"
        OmegaConf.save(cfg, path, resolve=True)
        if name != "ae":
            checkpoint = inputs / f"{name}.ckpt"
            checkpoint.touch()
            manifest[name] = {"config": str(path), "checkpoint": str(checkpoint)}
    manifest["fm"].update(
        {
            "autoencoder_config": str(inputs / "ae.yaml"),
            "autoencoder_checkpoint": str(inputs / "ae.ckpt"),
        }
    )
    manifest_path = inputs / "baselines.yaml"
    OmegaConf.save(OmegaConf.create(manifest), manifest_path)
    prepared = root.parent / "with-baselines"
    result = module.prepare(
        prepared,
        root.parent / "expanded",
        root.parent / "original",
        baseline_manifest=manifest_path,
    )
    return module, root, inputs, manifest, result


def test_baselines_preserve_fitted_models_on_both_raw_cohorts(
    baseline_campaign, monkeypatch
):
    module, root, inputs, manifest, result = baseline_campaign
    calls = []

    def capture(cfg, work_dir):
        calls.append((cfg.eval.checkpoint, work_dir))

    monkeypatch.setattr(
        "autocast.scripts.eval.encoder_processor_decoder.run_evaluation", capture
    )
    for method, normalization in (("fm", "ae"), ("crps", "crps")):
        for cohort in ("original", "expanded"):
            stage = result["stages"][f"eval_baseline_{method}_{cohort}"]
            cfg = OmegaConf.load(stage["config"])
            assert cfg.datamodule._target_.endswith("SpatioTemporalDataModule")
            assert cfg.datamodule.normalization_path == str(
                root.parent / f"original-{normalization}/stats.yml"
            )
            assert cfg.datamodule.test_data_path == str(
                root.parent / cohort / "test/data.pt"
            )
            assert cfg.eval.trajectory_statistics.enabled
            assert cfg.eval.trajectory_statistics.sampling_seed == 42
            assert cfg.eval.use_ema is False
            assert cfg.eval.csv_path is None
            assert cfg.eval.video_dir is None
            assert cfg.eval.rollout_snapshot_dir is None
            assert cfg.eval.trajectory_statistics.output_dir is None
            assert cfg.eval.n_members == 10
            assert "vrmse" in cfg.eval.metrics
            assert stage["requires"] == []
            if method == "fm":
                assert cfg.eval.mode == "encode_once"
                assert cfg.autoencoder_checkpoint == str(inputs / "ae.ckpt")
                assert cfg.model.encoder.periodic is False
            else:
                original = OmegaConf.load(inputs / "crps.yaml")
                assert cfg.model.processor == original.model.processor
            module._execute_eval(result, stage, cfg, Path(stage["workdir"]))
            assert calls[-1][0] == manifest[method]["checkpoint"]
    assert len(calls) == 4


def test_clean_segment_does_not_unlock_downstream_jobs(campaign):
    module, _, state = campaign
    ae_status = Path(state["stages"]["ae"]["workdir"]) / "status.json"
    module._write(ae_status, {"status": "continuation_required"})
    with pytest.raises(RuntimeError, match="has not completed"):
        module._require_completed(state, "ae")
    module._write(ae_status, {"status": "complete"})
    module._require_completed(state, "ae")


@pytest.mark.parametrize(
    ("epochs", "elapsed", "expected"),
    [
        (100, 80000, "continuation_required"),
        (798, 160000, "complete"),
        (700, 172801, "complete"),
    ],
)
def test_completion_requires_full_epoch_or_cumulative_time_budget(
    campaign, epochs, elapsed, expected
):
    module, _, state = campaign
    stage = state["stages"]["crps48"]
    cfg = OmegaConf.load(stage["config"])
    torch.save(
        {
            "loops": {"fit_loop": {"epoch_progress": {"total": {"completed": epochs}}}},
            "callbacks": {"Timer": {"time_elapsed": {"train": elapsed}}},
            "global_step": 500,
        },
        stage["checkpoint"],
    )
    status = module._checkpoint_status(stage, cfg)
    assert status["status"] == expected
    assert status["completed_epochs"] == epochs
    assert status["epoch_horizon_complete"] is (epochs >= 798)
    assert status["checkpoint_sha256"] == module._digest(Path(stage["checkpoint"]))
