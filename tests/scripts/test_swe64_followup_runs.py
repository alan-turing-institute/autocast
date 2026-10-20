"""Preview the two matched SWE64 follow-ups without data, W&B or Slurm."""

import runpy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from autocast.scripts.workflow.cli import build_parser

ROOT = Path(__file__).resolve().parents[2]
HELPER = runpy.run_path(str(ROOT / "scripts/swe64s/prepare_followups.py"))


@pytest.fixture
def plan(tmp_path):
    return HELPER["build_plan"](
        datasets_root=tmp_path / "unmounted data",
        output_root=tmp_path / "new outputs",
        account="example.account",
    )


def _compose(job):
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        return compose(
            config_name=job["config_name"],
            overrides=job["overrides"],
            return_hydra_config=True,
        )


def test_plan_has_no_data_or_submission_side_effects(plan, tmp_path):
    assert not (tmp_path / "new outputs").exists()
    assert not (tmp_path / "unmounted data").exists()
    assert not plan["training_executed"]
    assert not plan["jobs_submitted"]
    assert not plan["target_data_verified"]
    assert len(plan["source_commit"]) == 40
    assert plan["raw_dataset"]["splits"] == {"train": 200, "valid": 20, "test": 20}
    assert plan["latent_cache"]["autoencoder_training_trajectories"] == 1000
    assert plan["latent_cache"]["sample_shape"] == [321, 16, 16, 8]
    assert plan["latent_cache"]["autoencoder_checkpoint_sha256"] == (
        "e514ebf07d4350521da8fd25c8875eb2842eb668a8b059bb207989361935573c"
    )
    assert "final raw weights" in plan["checkpoint_selection"]


@pytest.mark.parametrize("index", [0, 1])
def test_standard_cli_and_ddp_preview(plan, index):
    job = plan["jobs"][index]
    parsed = build_parser().parse_args(job["preview_command"][5:])
    assert parsed.dry_run
    assert parsed.mode == "slurm"
    assert parsed.command == job["command"]
    assert parsed.workdir == job["output"]
    assert parsed.resume_from is None
    assert parsed.overrides == job["overrides"]
    assert "--dry-run" not in job["launch_command"]
    cfg = _compose(job)
    assert cfg.trainer.devices == cfg.hydra.launcher.gpus_per_node == 4
    assert cfg.trainer.num_nodes == cfg.hydra.launcher.nodes == 1
    assert cfg.hydra.launcher.tasks_per_node == 4
    assert cfg.hydra.launcher.cpus_per_task == 72
    assert cfg.hydra.launcher.additional_parameters.account == "example.account"
    assert cfg.hydra.launcher.timeout_min == job["allocation_minutes"]
    assert cfg.trainer.max_time == job["trainer_max_time"]
    assert cfg.logging.wandb.enabled
    assert cfg.logging.wandb.mode == "online"
    assert cfg.logging.wandb.resume is None
    assert cfg.datamodule.n_steps_input == cfg.datamodule.n_steps_output == 1
    assert cfg.datamodule.stride == 1
    assert cfg.datamodule.batch_size == job["batch_per_gpu"]
    assert cfg.model.n_members == job["training_members"]
    assert cfg.optimizer.learning_rate == job["learning_rate"]
    assert cfg.optimizer.scheduler == "cosine"
    assert cfg.optimizer.scheduler_interval == job["scheduler_interval"]
    assert cfg.optimizer.min_lr_ratio == cfg.optimizer.warmup == 0
    assert OmegaConf.select(cfg, "resume_from_checkpoint") is None
    assert cfg.output.skip_test
    assert cfg.provenance.source_commit == plan["source_commit"]


def test_global_fft_preserves_the_existing_epoch_control(plan):
    cfg = _compose(plan["jobs"][0])
    assert cfg.model.processor.noise_mode == "global"
    assert cfg.model.processor.n_noise_input_channels == 256
    assert cfg.model.processor.n_noise_channels == 256
    assert not cfg.model.processor.include_global_cond
    assert cfg.model.residual_prediction
    assert cfg.model.residual_use_delta_stats
    assert not cfg.model.train_in_latent_space
    assert cfg.model.loss_func.weights == [1.0, 0.1]
    assert cfg.trainer.max_steps == -1
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == 256
    windows = 200 * (321 - 1)
    assert windows // (32 * 4) * 256 == plan["jobs"][0]["expected_updates"] == 128000
    assert cfg.datamodule.data_path == plan["raw_dataset"]["path"]
    assert cfg.datamodule.normalization_path == cfg.datamodule.data_path + "/stats.yml"
    assert cfg.autoencoder_checkpoint is None


def test_latent_fm_uses_the_fixed_cache_and_step_budget(plan):
    cfg = _compose(plan["jobs"][1])
    assert cfg.datamodule.data_path == plan["latent_cache"]["path"]
    assert cfg.datamodule._target_.endswith("EncodedDataModule")
    assert cfg.trainer.max_steps == cfg.optimizer.cosine_steps == 128000
    assert cfg.trainer.max_epochs > 128000
    assert cfg.optimizer.cosine_epochs is None
    assert cfg.model.processor._target_.endswith("ResidualFlowMatchingProcessor")
    assert cfg.model.processor.reference._target_.endswith("LastFrameReference")
    assert cfg.model.processor.residual_normalizer.granularity == "lead_channel"
    assert not OmegaConf.select(cfg, "model.residual_prediction", default=False)
    assert cfg.model.processor.backbone.patch_size == 1
    assert not cfg.model.processor.backbone.include_global_cond
    assert cfg.model.processor.backbone.include_time_embedding
    assert cfg.model.processor.flow_ode_steps == 50
    assert cfg.model.processor.integrator == "euler"
    assert cfg.provenance.dataset_uuid == plan["latent_cache"]["uuid"]
    assert cfg.provenance.raw_dataset_uuid == plan["raw_dataset"]["uuid"]
    assert (
        cfg.provenance.autoencoder_checkpoint_sha256
        == plan["latent_cache"]["autoencoder_checkpoint_sha256"]
    )


def test_existing_output_cannot_be_reused(tmp_path):
    destination = tmp_path / "outputs/global_crps_fft/fit"
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="reuse training output"):
        HELPER["build_plan"](
            datasets_root=tmp_path / "data", output_root=tmp_path / "outputs"
        )


@pytest.mark.parametrize("mode", ["adaln", "concat"])
@pytest.mark.parametrize("loss", ["afcrps", "crps_fft"])
def test_multiscale_run_is_opt_in_and_uses_standard_preview(tmp_path, mode, loss):
    name = f"multiscale_{mode}_{loss}"
    plan = HELPER["build_plan"](
        datasets_root=tmp_path / "data",
        output_root=tmp_path / "outputs",
        run_names=[name],
    )
    assert len(plan["jobs"]) == 1
    job = plan["jobs"][0]
    assert job["name"] == name
    assert "--dry-run" in job["preview_command"]
    assert not plan["jobs_submitted"]
    assert not plan["training_executed"]
    assert not (tmp_path / "outputs").exists()
    cfg = _compose(job)
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == 256
    assert cfg.trainer.max_time == "00:07:45:00"
    assert cfg.hydra.launcher.timeout_min == 480
    assert cfg.logging.wandb.enabled
    assert cfg.logging.wandb.mode == "online"
    assert cfg.datamodule.batch_size == 32
    assert cfg.model.n_members == 8
    if mode == "adaln":
        assert cfg.model.processor.noise_source.half_power_wavenumbers == [1, 2, 4, 6]
    else:
        assert cfg.model.input_noise_injector.noise_source.half_power_wavenumbers == [
            2,
            4,
            8,
            16,
        ]
    if loss == "crps_fft":
        assert cfg.model.loss_func.weights == [1.0, 0.1]
    else:
        assert cfg.model.loss_func.alpha == 0.95


def test_selected_pair_preserves_explicit_order(tmp_path):
    names = ["multiscale_concat_afcrps", "multiscale_adaln_afcrps"]
    plan = HELPER["build_plan"](
        datasets_root=tmp_path / "data",
        output_root=tmp_path / "outputs",
        run_names=names,
    )
    assert [job["name"] for job in plan["jobs"]] == names


@pytest.mark.parametrize(
    "names", [[], ["unknown"], ["global_crps_fft", "global_crps_fft"]]
)
def test_invalid_run_selection_fails_without_outputs(tmp_path, names):
    with pytest.raises(ValueError, match=r"Unknown|duplicate"):
        HELPER["build_plan"](
            datasets_root=tmp_path / "data",
            output_root=tmp_path / "outputs",
            run_names=names,
        )
    assert not (tmp_path / "outputs").exists()
