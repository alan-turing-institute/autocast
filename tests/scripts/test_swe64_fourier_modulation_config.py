"""The Fourier conditioning control changes only the modulation architecture."""

from pathlib import Path

import pytest
import torch
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from autocast.losses import AlphaFairCRPSLoss, WeightedSumLoss
from autocast.nn.fourier_vit import FourierViTBlock

ROOT = Path(__file__).resolve().parents[2]
BASE = "afcrps_global256_residual_one_output_12block_256ep_8h"
FOURIER = "afcrps_fourier_modulation_global256_residual_one_output_12block_256ep_8h"
FFT = "crps_fft_fourier_modulation_global256_residual_one_output_12block_256ep_8h"
REFERENCES = {
    FOURIER: BASE,
    FFT: "crps_fft_global256_residual_one_output_12block_256ep_8h",
}


def config(name, *, guarded_reference=False):
    overrides = [
        f"local_experiment=swe64s/{name}",
        f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
    ]
    if guarded_reference:
        overrides.append("trainer=swe64s_noise_ablations")
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        return compose(
            config_name="encoder_processor_decoder",
            overrides=overrides,
        )


@pytest.mark.parametrize("name", REFERENCES)
def test_fourier_control_matches_global_training_protocol(name):
    baseline = config(REFERENCES[name], guarded_reference=True)
    spectral = config(name)
    expected = OmegaConf.to_container(baseline.model, resolve=True)
    assert isinstance(expected, dict)
    expected["processor"]["modulation_mode"] = "fourier"
    assert OmegaConf.to_container(spectral.model, resolve=True) == expected
    for key in ("datamodule", "trainer", "optimizer"):
        assert OmegaConf.to_container(
            spectral[key], resolve=True
        ) == OmegaConf.to_container(baseline[key], resolve=True)
    assert spectral.logging.wandb.enabled
    assert spectral.logging.wandb.mode == "online"
    assert spectral.logging.wandb.id is None
    assert spectral.logging.wandb.resume is None
    assert not spectral.logging.wandb.log_model
    assert spectral.model.n_members == 8
    assert spectral.model.residual_prediction
    assert spectral.model.residual_use_delta_stats
    assert spectral.trainer.max_epochs == spectral.optimizer.cosine_epochs == 256
    assert spectral.optimizer.learning_rate == 2e-4
    assert spectral.optimizer.min_lr_ratio == 0
    assert spectral.model.processor.noise_mode == "global"
    assert spectral.model.processor.n_noise_input_channels == 256
    assert not spectral.model.processor.include_global_cond
    assert spectral.model.processor.fourier_num_bands == 4
    assert spectral.trainer.max_time == "00:07:45:00"
    assert spectral.datamodule.batch_size == 32
    assert spectral.datamodule.n_steps_input == spectral.datamodule.n_steps_output == 1
    assert spectral.autoencoder_checkpoint is None
    callbacks = spectral.trainer.callbacks
    guard = [c for c in callbacks if c._target_.endswith("SustainedValidationRise")]
    assert len(guard) == 1
    assert guard[0].monitor == "val_loss"
    assert guard[0].relative_tolerance == 0.01
    assert guard[0].patience == 10
    assert guard[0].min_steps == 10000
    checkpoints = [c for c in callbacks if c._target_.endswith("ModelCheckpoint")]
    best = [c for c in checkpoints if c.get("monitor") == "val_loss"]
    assert len(best) == 1
    assert best[0].save_last
    assert best[0].every_n_epochs == 1
    assert not best[0].get("save_weights_only", False)


def test_paired_configs_differ_only_in_objective_and_name():
    field, fft = config(FOURIER), config(FFT)
    # Set the name before resolving the derived W&B job type.
    field.experiment_name = fft.experiment_name
    field.model.loss_func = fft.model.loss_func
    assert OmegaConf.to_container(field, resolve=True) == OmegaConf.to_container(
        fft, resolve=True
    )


@pytest.mark.parametrize("name", REFERENCES)
def test_eight_member_objective_reaches_fourier_parameters(name):
    cfg = config(name)
    # Preserve the 64x64 grid and 4x4 patches, with a smaller CPU test backbone.
    cfg.model.processor.hidden_dim = 16
    cfg.model.processor.n_layers = 1
    cfg.model.processor.num_heads = 2
    torch.manual_seed(42)
    processor = instantiate(
        cfg.model.processor, in_channels=3, out_channels=3, spatial_resolution=(64, 64)
    )
    state = repeat(torch.randn(3, 64, 64), "c h w -> m c h w", m=8)
    prediction = rearrange(
        processor(state, torch.randn(8, 256)),
        "m c h w -> 1 1 h w c m",
    )
    objective = instantiate(cfg.model.loss_func)
    if name == FFT:
        assert isinstance(objective, WeightedSumLoss)
        assert objective.weights == (1.0, 0.1)
    else:
        assert isinstance(objective, AlphaFairCRPSLoss)
        assert objective.alpha == 0.95
    loss = objective(prediction, torch.randn(1, 1, 64, 64, 3))
    assert torch.isfinite(loss)
    loss.backward()
    for block in processor.model.vit.blocks:
        assert isinstance(block, FourierViTBlock)
        gradient = block.fourier.projection.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_production_capacity_and_small_grid_forward():
    spectral = config(FOURIER)
    kwargs = {"in_channels": 3, "out_channels": 3, "spatial_resolution": (64, 64)}
    torch.manual_seed(42)
    baseline = instantiate(config(BASE).model.processor, **kwargs)
    torch.manual_seed(42)
    model = instantiate(spectral.model.processor, **kwargs)
    assert (
        sum(p.numel() for p in model.parameters())
        - sum(p.numel() for p in baseline.parameters())
        == 37_008
    )
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    # Keep the real width/depth while limiting the CPU smoke test's token count.
    x = torch.randn(1, 3, 16, 24)
    noise = torch.randn(1, 256)
    with torch.no_grad():
        output = model(x, noise)
        torch.testing.assert_close(output, baseline(x, noise), rtol=2e-5, atol=2e-6)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
