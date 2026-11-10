"""The combined control changes noise while retaining the original V2 score."""

import runpy
from pathlib import Path

import torch
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
PRESET = "crps_fft_helmholtz_learned_spatial1d_256ep_4h"
WEIGHT = 0.2983579021838257


def config(name=PRESET, overrides=()):
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


def test_combination_preserves_v2_model_data_and_optimizer():
    combined = config()
    v2 = config(
        "crps_fft_helmholtz_velocity_spatial1d_256ep",
        [f"vector_weight={WEIGHT}"],
    )
    v2.model.processor.modulation_mode = "spectral_noise"
    expected = OmegaConf.to_container(v2.model, resolve=True)
    assert OmegaConf.to_container(combined.model, resolve=True) == expected
    assert combined.datamodule == v2.datamodule
    assert combined.optimizer == v2.optimizer
    assert combined.seed == v2.seed == 42
    assert combined.model.validation_loss_func.weights == [1.0, 0.1]
    assert combined.model.loss_func.weights == [1.0, 0.1, WEIGHT]
    assert combined.model.loss_func.losses[2].mode == "helmholtz"
    assert combined.model.processor.n_noise_input_channels == 1
    assert combined.model.processor.n_noise_channels == 256
    assert combined.model.processor.noise_radial_knots == 16
    assert combined.model.processor.noise_source is None
    assert not combined.model.processor.include_global_cond
    assert combined.model.residual_prediction
    assert combined.model.residual_use_delta_stats


def test_opt_in_submission_keeps_caps_and_base_score_stopping(tmp_path):
    helper = runpy.run_path(str(ROOT / "scripts/swe64s/prepare_followups.py"))
    plan = helper["build_plan"](
        datasets_root=tmp_path / "data",
        output_root=tmp_path / "outputs",
        run_names=["learned_spatial_helmholtz_crps_fft"],
    )
    (job,) = plan["jobs"]
    cfg = config(overrides=job["overrides"][2:])
    assert cfg.trainer.devices == cfg.hydra.launcher.gpus_per_node == 4
    assert cfg.trainer.max_time == "00:03:45:00"
    assert cfg.hydra.launcher.timeout_min == job["allocation_minutes"] == 240
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == 256
    assert cfg.optimizer.learning_rate == 2e-4
    assert cfg.optimizer.min_lr_ratio == 0
    assert cfg.logging.wandb.enabled
    assert cfg.logging.wandb.mode == "online"
    assert cfg.logging.wandb.resume is None
    assert cfg.logging.wandb.id is None
    assert cfg.get("resume_from_checkpoint") is None
    assert cfg.output.get("resume_from_checkpoint") is None
    stop = next(
        c
        for c in cfg.trainer.callbacks
        if c._target_.endswith("SustainedValidationRise")
    )
    assert (stop.monitor, stop.relative_tolerance, stop.patience, stop.min_steps) == (
        "val_loss",
        0.01,
        10,
        10000,
    )
    checkpoint = next(
        c
        for c in cfg.trainer.callbacks
        if c._target_.endswith("ModelCheckpoint") and c.get("monitor") == "val_loss"
    )
    assert checkpoint.save_last
    assert checkpoint.save_top_k == 1
    assert not checkpoint.get("save_weights_only", False)
    assert not plan["jobs_submitted"]
    assert not (tmp_path / "outputs").exists()


def test_eight_member_combination_has_finite_learned_noise_gradients():
    cfg = config()
    cfg.model.processor.hidden_dim = 16
    cfg.model.processor.n_layers = 3
    cfg.model.processor.num_heads = 2
    torch.manual_seed(42)
    processor = instantiate(
        cfg.model.processor, in_channels=3, out_channels=3, spatial_resolution=(64, 64)
    )
    state = repeat(torch.randn(3, 64, 64), "c h w -> m c h w", m=8)
    prediction = rearrange(
        processor(state, torch.randn(8, 256, 1)), "m c h w -> 1 1 h w c m"
    )
    target = torch.randn(1, 1, 64, 64, 3)
    loss = instantiate(cfg.model.loss_func)
    base = instantiate(cfg.model.validation_loss_func)
    objective = loss(prediction, target)
    base_score = base(prediction, target)
    torch.testing.assert_close(
        objective, base_score + WEIGHT * loss.losses[2](prediction, target)
    )
    assert torch.isfinite(objective)
    objective.backward()
    for parameter in processor.noise_filter.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
