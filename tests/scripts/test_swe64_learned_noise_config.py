"""Matched training budgets and differentiable learned-noise objectives."""

from pathlib import Path

import pytest
import torch
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from autocast.losses import AlphaFairCRPSLoss, WeightedSumLoss

ROOT = Path(__file__).resolve().parents[2]
BASE_SUFFIX = "spatial1d_residual_one_output_12block_256ep_8h"
SUFFIX = "spatial1d_residual_one_output_12block_256ep_4h"


def config(name, *, baseline=False):
    overrides = [
        f"local_experiment=swe64s/{name}",
        f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
    ]
    if baseline:
        overrides.append("trainer=swe64s_noise_ablations")
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        return compose(config_name="encoder_processor_decoder", overrides=overrides)


@pytest.mark.parametrize("objective", ["afcrps", "crps_fft"])
def test_training_protocol_matches_spatial_baseline(objective):
    baseline = config(f"{objective}_{BASE_SUFFIX}", baseline=True)
    learned = config(f"{objective}_learned_{SUFFIX}")
    # Only the safety cap changes; preserve optimizer horizon and callbacks.
    baseline.trainer.max_time = "00:03:45:00"
    expected = OmegaConf.to_container(baseline.model, resolve=True)
    assert isinstance(expected, dict)
    expected["processor"]["modulation_mode"] = "spectral_noise"
    assert OmegaConf.to_container(learned.model, resolve=True) == expected
    for key in ("datamodule", "trainer", "optimizer"):
        assert OmegaConf.to_container(
            learned[key], resolve=True
        ) == OmegaConf.to_container(baseline[key], resolve=True)
    assert learned.logging.wandb.enabled
    assert learned.logging.wandb.mode == "online"
    assert learned.logging.wandb.id is None
    assert learned.logging.wandb.resume is None
    assert learned.model.processor.noise_source is None
    assert learned.model.processor.n_noise_input_channels == 1
    assert learned.model.processor.n_noise_channels == 256
    assert learned.model.processor.noise_radial_knots == 16
    assert not learned.model.processor.include_global_cond
    assert learned.trainer.max_epochs == learned.optimizer.cosine_epochs == 256
    assert learned.optimizer.learning_rate == 2e-4
    assert learned.optimizer.min_lr_ratio == 0
    assert learned.model.residual_prediction
    assert learned.model.residual_use_delta_stats
    assert learned.model.n_members == 8
    assert learned.datamodule.batch_size == 32
    assert learned.trainer.max_time == "00:03:45:00"
    callbacks = learned.trainer.callbacks
    guard = [c for c in callbacks if c._target_.endswith("SustainedValidationRise")]
    assert len(guard) == 1
    assert guard[0].monitor == "val_loss"
    assert guard[0].patience == 10
    assert guard[0].relative_tolerance == 0.01
    assert guard[0].min_steps == 10000
    checkpoints = [
        c
        for c in callbacks
        if c._target_.endswith("ModelCheckpoint") and c.get("monitor") == "val_loss"
    ]
    assert len(checkpoints) == 1
    assert checkpoints[0].save_last
    assert not checkpoints[0].get("save_weights_only", False)


@pytest.mark.parametrize("objective", ["afcrps", "crps_fft"])
def test_real_eight_member_objective_updates_scale_and_relative_phase(objective):
    cfg = config(f"{objective}_learned_{SUFFIX}")
    cfg.model.processor.hidden_dim = 16
    cfg.model.processor.n_layers = 3
    cfg.model.processor.num_heads = 2
    torch.manual_seed(42)
    processor = instantiate(
        cfg.model.processor, in_channels=3, out_channels=3, spatial_resolution=(64, 64)
    )
    state = repeat(torch.randn(3, 64, 64), "c h w -> m c h w", m=8)
    predictions = rearrange(
        processor(state, torch.randn(8, 256, 1)), "m c h w -> 1 1 h w c m"
    )
    loss_func = instantiate(cfg.model.loss_func)
    if objective == "crps_fft":
        assert isinstance(loss_func, WeightedSumLoss)
        assert loss_func.weights == (1.0, 0.1)
    else:
        assert isinstance(loss_func, AlphaFairCRPSLoss)
        assert loss_func.alpha == 0.95
    loss = loss_func(predictions, torch.randn(1, 1, 64, 64, 3))
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in processor.noise_filter.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_production_capacity_and_identity_initialization():
    cfg = config(f"afcrps_learned_{SUFFIX}")
    kwargs = {"in_channels": 3, "out_channels": 3, "spatial_resolution": (64, 64)}
    torch.manual_seed(42)
    learned = instantiate(cfg.model.processor, **kwargs)
    torch.manual_seed(42)
    baseline = instantiate(config(f"afcrps_{BASE_SUFFIX}").model.processor, **kwargs)
    assert (
        sum(p.numel() for p in learned.parameters())
        - sum(p.numel() for p in baseline.parameters())
        == 556
    )
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(learned.state_dict()[name], value, rtol=0, atol=0)
    # Real width/depth, but a smaller rectangular grid for CPU verification.
    state, noise = torch.randn(1, 3, 16, 24), torch.randn(1, 24, 1)
    with torch.no_grad():
        torch.testing.assert_close(
            learned(state, noise), baseline(state, noise), rtol=2e-5, atol=2e-6
        )
