"""Check the patch FNO comparison and its stochastic processor on CPU."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch import LightningDataModule
from omegaconf import OmegaConf

from autocast.scripts.setup import setup_epd_model
from autocast.types import Batch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _compose(preset):
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO_ROOT / "src/autocast/configs")
    ):
        return compose(
            config_name="encoder_processor_decoder",
            overrides=[
                f"local_experiment=swe64s/{preset}",
                f"hydra.searchpath=[file://{REPO_ROOT / 'local_hydra'}]",
                "swe64_pool_root=/example/pool",
                "swe64_train_trajectories=1000",
            ],
        )


def test_patch_fno_matches_vit_recipe_and_parameter_budget():
    baseline = _compose("campaign_nested_crps_fft")
    candidate = _compose("campaign_nested_crps_fft_fno")
    assert OmegaConf.to_container(candidate.datamodule, resolve=True) == (
        OmegaConf.to_container(baseline.datamodule, resolve=True)
    )
    assert candidate.model.loss_func == baseline.model.loss_func
    assert candidate.model.residual_prediction == baseline.model.residual_prediction
    assert candidate.model.n_members == baseline.model.n_members == 8
    assert candidate.trainer.limit_train_batches == 1.0
    assert candidate.trainer.max_steps == candidate.optimizer.cosine_steps == 128000
    assert candidate.optimizer.learning_rate == baseline.optimizer.learning_rate == 2e-4
    assert candidate.optimizer.min_lr_ratio == 0
    assert candidate.resume_from_checkpoint is None
    assert candidate.logging.wandb.mode == "online"
    assert not any("Plot" in cb._target_ for cb in candidate.trainer.callbacks)

    def count(config):
        processor = instantiate(
            config, in_channels=3, out_channels=3, spatial_resolution=[64, 64]
        )
        return sum(p.numel() for p in processor.parameters())

    assert count(candidate.model.processor) == pytest.approx(
        count(baseline.model.processor), rel=0.01
    )


def test_patch_fno_with_spatial_noise_gradients_and_checkpoint():
    cfg = _compose("campaign_nested_crps_fft_fno")
    opts = {
        "in_channels": 3,
        "out_channels": 3,
        "spatial_resolution": (8, 12),
        "hidden_dim": 16,
        "n_layers": 2,
        "spectral_channels": 4,
    }
    processor = instantiate(cfg.model.processor, **opts)
    field = torch.randn(1, 3, 8, 12).repeat(2, 1, 1, 1)
    noise = torch.randn(2, 6, 1, requires_grad=True)
    prediction = processor(field, noise)
    assert prediction.shape == field.shape
    assert not torch.allclose(prediction[0], prediction[1])
    torch.testing.assert_close(processor(field[:1], noise[:1]), prediction[:1])
    torch.testing.assert_close(
        processor(field.permute(0, 2, 3, 1).unsqueeze(1), noise),
        prediction.permute(0, 2, 3, 1).unsqueeze(1),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed_precision = processor(field, noise)
    assert torch.isfinite(mixed_precision).all()

    prediction.square().mean().backward()
    assert noise.grad is not None
    assert torch.isfinite(noise.grad).all()
    for parameter in processor.parameters():
        assert not parameter.is_complex()
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    restored = instantiate(cfg.model.processor, **opts)
    restored.load_state_dict(processor.state_dict(), strict=True)
    torch.testing.assert_close(restored(field, noise), prediction)
    with pytest.raises(ValueError, match="Expected field grid"):
        restored(field.transpose(-1, -2), noise)


def test_patch_fno_uses_ensemble_residual_training_path():
    cfg = _compose("campaign_nested_crps_fft_fno")
    cfg.model.processor.hidden_dim = 16
    cfg.model.processor.n_layers = 1
    cfg.model.processor.spectral_channels = 4
    field = torch.randn(2, 1, 8, 12, 3)
    batch = Batch(field, torch.randn_like(field), None, None)
    stats = {
        "channel_count": 3,
        "n_steps_input": 1,
        "n_steps_output": 1,
        "n_constant_scalars": 0,
        "n_constant_field_channels": 0,
        "input_shape": tuple(field.shape),
        "output_shape": tuple(field.shape),
        "example_batch": batch,
    }
    model = setup_epd_model(cfg, stats, LightningDataModule())
    assert model.residual_prediction
    loss, prediction = model.loss(batch)
    assert torch.isfinite(loss)
    assert prediction is not None
    assert prediction.shape == (*field.shape, 8)
    loss.backward()
