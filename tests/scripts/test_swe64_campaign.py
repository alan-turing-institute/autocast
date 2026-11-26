"""Check that nested SWE64 previews keep the intended matched protocol."""

import runpy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from autocast.losses import (
    AlphaFairCRPSLoss,
    AlphaFairEnergyScoreLoss,
    CoefficientCRPSLoss,
    ComponentBranch,
    ComponentObjective,
    JointCoefficientEnergyScore,
    TransformedLoss,
    WeightedSumLoss,
)
from autocast.losses.mode_groups import SampledModeEnergyScore
from autocast.models.optimizer_mixin import OptimizerMixin
from autocast.scripts.workflow.cli import build_parser

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = runpy.run_path(str(ROOT / "scripts/swe64s/prepare_campaign.py"))


def _compose(command):
    parsed = build_parser().parse_args(command[5:])
    assert parsed.dry_run
    assert parsed.mode == "slurm"
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        return compose(
            config_name="encoder_processor_decoder",
            overrides=parsed.overrides,
            return_hydra_config=True,
        )


@pytest.mark.parametrize("name", CAMPAIGN["RUNS"])
def test_campaign_arms_match_protocol(tmp_path, name):
    pool_root = tmp_path / "pool views"
    run = CAMPAIGN["RUNS"][name]
    plan = CAMPAIGN["build_plan"](
        pool_root=pool_root,
        output_root=tmp_path / "outputs",
        run_names=[name],
        parent_checkpoint=tmp_path / "parent.ckpt" if run.finetune_steps else None,
    )
    (job,) = plan["jobs"]
    cfg = _compose(job["preview_command"])
    loss = instantiate(cfg.model.loss_func)
    if run.objective == "afcrps":
        assert isinstance(loss, AlphaFairCRPSLoss)
    else:
        assert isinstance(loss, WeightedSumLoss)
        assert cfg.model.loss_func.weights == [1.0, 0.1]
    assert cfg.provenance.normalization_sha256 == plan["normalization_sha256"]
    assert cfg.datamodule.data_path == str(pool_root / f"train{run.pool}")
    assert cfg.datamodule.normalization_path == str(pool_root / "train200/stats.yml")
    assert cfg.datamodule.valid_trajectories == plan["valid_trajectories"] == 20
    assert cfg.datamodule.test_trajectories == plan["test_trajectories"] == 100
    assert cfg.model.residual_prediction
    assert cfg.model.residual_use_delta_stats
    assert cfg.model.residual_delta_stats_path is None
    assert cfg.datamodule.batch_size == 32
    assert cfg.model.n_members == 8
    assert cfg.trainer.devices == cfg.hydra.launcher.gpus_per_node == 4
    assert cfg.trainer.limit_train_batches == 500
    proc = cfg.model.processor
    assert (proc.hidden_dim, proc.num_heads, proc.n_layers, proc.patch_size) == (
        224,
        8,
        12,
        4,
    )
    assert proc.modulation_mode == "adaln"
    assert not proc.include_global_cond
    assert not proc.rope
    assert proc.graph_attention_backend == "masked"
    assert proc.rpb == run.rpb
    assert proc.graph_hops == run.hops
    assert proc.noise_mode == ("global" if run.global_noise else "spatial")
    assert proc.n_noise_input_channels == (256 if run.global_noise else 1)
    _assert_training_budget(cfg, run, job["expected_updates"])


def _assert_training_budget(cfg, run, expected_updates):
    if run.finetune_steps:
        assert cfg.datamodule.n_steps_output == 2
        assert cfg.model.prediction_steps == 1
        assert cfg.model.train_rollout_steps == run.finetune_steps
        assert cfg.model.validation_rollout_steps == 2
        assert not cfg.model.detach_rollout_feedback
        assert cfg.resume_weights_only
        assert cfg.optimizer.learning_rate == 2e-5
        assert cfg.optimizer.cosine_steps == cfg.trainer.max_steps == 32000
        assert cfg.optimizer.scheduler_interval == "step"
    else:
        assert cfg.resume_from_checkpoint is None
        assert not cfg.resume_weights_only
        assert cfg.datamodule.n_steps_output == 1
        assert cfg.trainer.max_steps == -1
        assert (
            cfg.trainer.max_epochs * cfg.trainer.limit_train_batches == expected_updates
        )
        if run.fixed_rate:
            assert cfg.optimizer.scheduler is None
            assert cfg.optimizer.learning_rate == 1e-4
            OptimizerMixin._reject_ignored_scheduler_keys(dict(cfg.optimizer), None)
            assert cfg.trainer.max_epochs == 512
        else:
            assert cfg.optimizer.scheduler_interval == "epoch"
            assert cfg.optimizer.cosine_epochs == cfg.trainer.max_epochs == 256
            assert cfg.optimizer.learning_rate == 2e-4


def test_score_ablation_campaign_matches_all_eleven_cells(tmp_path):
    plan = CAMPAIGN["build_plan"](
        pool_root=tmp_path / "pool",
        output_root=tmp_path / "scores",
        score_ablations=True,
    )
    assert len(plan["jobs"]) == 11
    assert len({job["output"] for job in plan["jobs"]}) == 11
    for job in plan["jobs"]:
        backbone, primary, spectral = job["name"].split("_")
        cfg = _compose(job["preview_command"])
        loss = instantiate(cfg.model.loss_func)
        primary_cls = (
            AlphaFairEnergyScoreLoss if primary == "afes" else AlphaFairCRPSLoss
        )
        if spectral == "none":
            assert isinstance(loss, primary_cls)
        else:
            assert isinstance(loss, WeightedSumLoss)
            assert isinstance(loss.losses[0], primary_cls)
            assert loss.weights == (1.0, 1.0)
            spectral_loss = loss.losses[1]
            assert isinstance(spectral_loss, TransformedLoss)
            objective = spectral_loss.objective
            assert isinstance(objective, ComponentObjective)
            joint_cls = (
                SampledModeEnergyScore
                if backbone == "fno"
                else JointCoefficientEnergyScore
            )
            classes = {
                "blend": (CoefficientCRPSLoss, joint_cls),
                "component": (CoefficientCRPSLoss,),
                "joint": (joint_cls,),
            }[spectral]
            assert objective.weights == (0.1,) * len(classes)
            for branch, score_cls in zip(objective.branches, classes, strict=True):
                assert isinstance(branch, ComponentBranch)
                assert type(branch.score) is score_cls
                if isinstance(branch.score, SampledModeEnergyScore):
                    assert branch.score.group == "singletons"
        validation = instantiate(cfg.model.validation_loss_func)
        assert isinstance(validation, WeightedSumLoss)
        assert isinstance(validation.losses[0], AlphaFairCRPSLoss)
        assert validation.weights == (1.0, 0.1)
        assert cfg.compile_backbone == (backbone == "vit")
        assert cfg.float32_matmul_precision == "high"
        assert cfg.trainer.devices == 4
        assert cfg.trainer.limit_train_batches == 1.0
        assert cfg.trainer.max_steps == cfg.optimizer.cosine_steps == 128000
        assert cfg.optimizer.scheduler_interval == "step"
        assert cfg.trainer.callbacks[0].every_n_train_steps == 32000
        assert cfg.datamodule.batch_size == 32
        assert cfg.model.n_members == 8
        assert cfg.model.residual_prediction
        assert cfg.logging.wandb.mode == "online"
        assert job["allocation_minutes"] == (480 if backbone == "vit" else 720)


def test_score_ablation_preflight_uses_one_short_fit(tmp_path):
    (job,) = CAMPAIGN["build_plan"](
        pool_root=tmp_path / "pool",
        output_root=tmp_path / "scores",
        run_names=["fno_afes_blend"],
        preflight=True,
    )["jobs"]
    cfg = _compose(job["preview_command"])
    assert cfg.trainer.max_steps == job["expected_updates"] == 128
    assert cfg.optimizer.cosine_steps == 128000
    assert cfg.trainer.limit_val_batches == 8
    assert not cfg.logging.wandb.enabled
    assert "resume_preview_command" not in job


@pytest.mark.parametrize("name", CAMPAIGN["PREFLIGHT_RUNS"])
def test_preflight_resume_keeps_scheduler_and_data(tmp_path, name):
    (job,) = CAMPAIGN["build_plan"](
        pool_root=tmp_path / "pools",
        output_root=tmp_path / "outputs",
        run_names=[name],
        preflight=True,
    )["jobs"]
    initial = _compose(job["preview_command"])
    resumed = _compose(job["resume_preview_command"])
    assert initial.model == resumed.model
    assert initial.datamodule == resumed.datamodule
    assert initial.optimizer == resumed.optimizer
    assert initial.trainer.max_epochs == 2
    assert resumed.trainer.max_epochs == 3
    assert (
        initial.hydra.launcher.timeout_min == resumed.hydra.launcher.timeout_min == 30
    )
    assert (
        initial.trainer.limit_train_batches
        == resumed.trainer.limit_train_batches
        == 500
    )
    assert not resumed.resume_weights_only
    assert not resumed.reset_resume_time_budget
    assert resumed.resume_from_checkpoint == str(Path(job["output"]) / "preflight.ckpt")
    assert not initial.logging.wandb.enabled
    assert not resumed.logging.wandb.enabled


def test_default_batch_covers_new_pool_sizes(tmp_path):
    plan = CAMPAIGN["build_plan"](
        pool_root=tmp_path / "pools",
        output_root=tmp_path / "outputs",
    )
    assert [job["name"] for job in plan["jobs"]] == [
        "fft_1000",
        "fft_5000",
        "fixed_1000",
        "fixed_5000",
    ]
    previous_trainer = OmegaConf.load(
        ROOT / "local_hydra/trainer/swe64s_noise_ablations.yaml"
    )
    expected_callbacks = [
        callback
        for callback in previous_trainer.callbacks
        if not callback._target_.endswith("ValidationMetricPlotCallback")
    ]
    for job in plan["jobs"]:
        cfg = _compose(job["preview_command"])
        assert cfg.trainer.callbacks == expected_callbacks
        assert cfg.logging.wandb.enabled
        assert cfg.logging.wandb.mode == "online"
        assert not cfg.logging.wandb.log_model


def test_finetune_rejects_mixing_graph_and_dense_parents(tmp_path):
    with pytest.raises(ValueError, match="one architecture"):
        CAMPAIGN["build_plan"](
            pool_root=tmp_path / "pools",
            output_root=tmp_path / "outputs",
            run_names=["full_off_continue_1000", "graph2_rollout2_1000"],
            parent_checkpoint=tmp_path / "parent.ckpt",
        )
