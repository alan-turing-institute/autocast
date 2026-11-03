"""Matched vector configurations and their training-only coefficient check."""

import runpy
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from autocast.losses import AlphaFairCRPSLoss, VelocityScoreLoss

ROOT = Path(__file__).resolve().parents[2]
PRESETS = (
    "crps_fft_joint_velocity_spatial1d_256ep",
    "crps_fft_helmholtz_velocity_spatial1d_256ep",
)
MODULE = runpy.run_path(str(ROOT / "scripts/swe64s/calibrate_vector_weight.py"))


def _compose(name, overrides=()):
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        return compose(
            config_name="encoder_processor_decoder",
            overrides=[
                f"local_experiment=swe64s/{name}",
                f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
                *overrides,
            ],
            return_hydra_config=True,
        )


@pytest.mark.parametrize("name", PRESETS)
def test_vector_preserves_matched_task_model_and_schedule(name):
    cfg = _compose(name)
    base = _compose("crps_fft_spatial1d_residual_one_output_12block_256ep_8h")
    assert cfg.datamodule == base.datamodule
    assert cfg.optimizer == base.optimizer
    for part in ("processor", "encoder", "decoder", "n_members", "residual_prediction"):
        assert cfg.model[part] == base.model[part]
    assert cfg.trainer.max_epochs == 256
    assert cfg.optimizer.learning_rate == 2e-4
    assert cfg.optimizer.scheduler == "cosine"
    assert cfg.optimizer.min_lr_ratio == 0
    assert cfg.model.validation_loss_func == base.model.loss_func
    assert OmegaConf.is_missing(cfg, "vector_weight")
    assert cfg.logging.wandb.enabled
    assert cfg.logging.wandb.mode == "offline"
    assert cfg.get("resume_from_checkpoint") is None
    assert cfg.output.get("resume_from_checkpoint") is None


def test_vector_pair_differs_only_in_decomposition():
    first, second = (_compose(name, ["vector_weight=0.1"]) for name in PRESETS)
    first_model = cast(
        "dict[str, Any]", OmegaConf.to_container(first.model, resolve=True)
    )
    second_model = OmegaConf.to_container(second.model, resolve=True)
    first_model["loss_func"]["losses"][2]["mode"] = "helmholtz"
    assert first_model == second_model
    assert first.trainer == second.trainer
    assert first.seed == second.seed == 42


@pytest.mark.parametrize("name", PRESETS)
def test_stop_and_checkpoint_settings_survive_launch_composition(name):
    cfg = _compose(
        name,
        [
            "vector_weight=0.1",
            "+distributed=ddp_4gpu_slurm",
            "trainer.max_time=00:07:45:00",
        ],
    )
    assert cfg.trainer.devices == 4
    assert cfg.trainer.max_time == "00:07:45:00"
    assert cfg.hydra.launcher.timeout_min == 480
    callbacks = cfg.trainer.callbacks
    stop = next(c for c in callbacks if c._target_.endswith("SustainedValidationRise"))
    assert stop.monitor == "val_loss"
    assert (stop.relative_tolerance, stop.patience, stop.min_steps) == (0.01, 10, 10000)
    best = next(
        c
        for c in callbacks
        if c.get("monitor") == "val_loss" and c._target_.endswith("ModelCheckpoint")
    )
    assert best.save_top_k == 1
    assert best.save_last
    assert not best.save_on_train_epoch_end


@pytest.mark.parametrize("name", PRESETS)
def test_composed_vector_loss_and_base_are_finite_with_distinct_gradients(name):
    cfg = _compose(name, ["vector_weight=0.1"])
    loss = instantiate(cfg.model.loss_func)
    base = instantiate(cfg.model.validation_loss_func)
    gen = torch.Generator().manual_seed(42)
    prediction = torch.randn(2, 1, 8, 8, 3, 8, generator=gen, requires_grad=True)
    target = torch.randn(2, 1, 8, 8, 3, generator=gen)
    objective, base_score = loss(prediction, target), base(prediction, target)
    torch.testing.assert_close(
        objective, base_score + 0.1 * loss.losses[2](prediction, target)
    )
    gradient = torch.autograd.grad(objective, prediction)[0]
    assert torch.isfinite(gradient).all()
    assert objective != base_score


def test_calibration_shared_cap_and_scale_check():
    gen = torch.Generator().manual_seed(11)
    prediction = torch.randn(2, 1, 8, 8, 3, 8, generator=gen)
    target = torch.randn(2, 1, 8, 8, 3, generator=gen)
    auxiliaries = {
        mode: VelocityScoreLoss([0.1211, 0.1214], channels=[1, 2], mode=mode)
        for mode in ("joint", "helmholtz")
    }
    result = MODULE["calibrate_batches"](
        AlphaFairCRPSLoss(), auxiliaries, [(prediction, target)], gradient_fraction=0.1
    )
    assert result["weight"] > 0
    assert len(result["rows"]) == 2
    assert max(r["weighted_gradient_ratio"] for r in result["rows"]) == pytest.approx(
        0.1
    )
    assert all(r["weighted_gradient_ratio"] <= 0.1000001 for r in result["rows"])
    assert prediction.grad is None
    check = MODULE["check_velocity_statistics"]
    check(auxiliaries["joint"], torch.tensor([0.0116, 0.1211, 0.1214]))
    with pytest.raises(ValueError, match="normalization"):
        check(auxiliaries["joint"], torch.tensor([1.0, 1.0, 1.0]))


@pytest.mark.parametrize("fraction", [0, -1, float("nan"), 2])
def test_calibration_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="gradient_fraction"):
        MODULE["calibrate_batches"](
            AlphaFairCRPSLoss(), {}, [], gradient_fraction=fraction
        )
