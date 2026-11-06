"""The Fourier conditioning control changes only the modulation architecture."""

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
BASE = "afcrps_global256_residual_one_output_12block_256ep_8h"
FOURIER = "afcrps_fourier_modulation_global256_residual_one_output_12block_256ep_8h"


def config(name):
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        return compose(
            config_name="encoder_processor_decoder",
            overrides=[
                f"local_experiment=swe64s/{name}",
                f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
            ],
        )


def test_fourier_control_matches_global_training_protocol():
    baseline, spectral = config(BASE), config(FOURIER)
    expected = OmegaConf.to_container(baseline.model, resolve=True)
    assert isinstance(expected, dict)
    expected["processor"]["modulation_mode"] = "fourier"
    assert OmegaConf.to_container(spectral.model, resolve=True) == expected
    for key in ("datamodule", "trainer", "optimizer"):
        assert OmegaConf.to_container(
            spectral[key], resolve=True
        ) == OmegaConf.to_container(baseline[key], resolve=True)
    assert spectral.logging.wandb.enabled == baseline.logging.wandb.enabled
    assert spectral.logging.wandb.mode == baseline.logging.wandb.mode
    assert spectral.model.n_members == 8
    assert spectral.model.residual_prediction
    assert spectral.model.residual_use_delta_stats
    assert spectral.trainer.max_epochs == spectral.optimizer.cosine_epochs == 256
    assert spectral.optimizer.learning_rate == 2e-4
    assert spectral.optimizer.min_lr_ratio == 0
    assert spectral.model.processor.noise_mode == "global"
    assert spectral.model.processor.n_noise_input_channels == 256
    assert not spectral.model.processor.include_global_cond


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
