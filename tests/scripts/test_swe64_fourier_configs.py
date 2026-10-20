"""Check SWE64 Fourier-loss controls through decoded residual predictions."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from einops import rearrange
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from the_well.data.normalization import ZScoreNormalization

from autocast.losses import (
    AlphaFairCRPSLoss,
    CoefficientCRPSLoss,
    SpectralSpreadErrorLoss,
    WeightedSumLoss,
)
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.nn.noise import SpatialFourierSource
from autocast.scripts.setup import _build_loss_func, setup_epd_model
from autocast.scripts.workflow.slurm import (
    _build_sbatch_command,
    _resolve_launcher_submission_context,
)
from autocast.types import Batch

ROOT = Path(__file__).resolve().parents[2]
GLOBAL = "afcrps_global256_residual_one_output_12block_256ep_8h"
SPATIAL = "afcrps_spatial1d_residual_one_output_12block_256ep_8h"
STRUCTURED = "afcrps_structured_spatial1d_residual_one_output_12block_256ep_8h"
FFT_SPATIAL = "crps_fft_spatial1d_residual_one_output_12block_256ep_8h"
FFT_PRESETS = {
    "crps_fft_global256_residual_one_output_12block_256ep_8h": GLOBAL,
    FFT_SPATIAL: SPATIAL,
    "crps_fft_structured_spatial1d_residual_one_output_12block_256ep_8h": STRUCTURED,
}
WAVEVECTORS = ((3, 0), (0, 3), (2, 2), (2, -2))


def _config(name, *, distributed=False):
    overrides = [
        f"local_experiment=swe64s/{name}",
        f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
    ]
    if distributed:
        overrides += [
            "+distributed=ddp_4gpu_slurm",
            "trainer.max_time=00:07:45:00",
            "++hydra.launcher.nodes=1",
            "++hydra.launcher.cpus_per_task=72",
            "logging.wandb.enabled=true",
            "logging.wandb.mode=online",
        ]
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        return compose(
            config_name="encoder_processor_decoder",
            overrides=overrides,
            return_hydra_config=distributed,
        )


@pytest.mark.parametrize("name", FFT_PRESETS)
@pytest.mark.parametrize("distributed", [False, True])
def test_fourier_presets_change_only_the_loss(name, distributed):
    cfg = _config(name, distributed=distributed)
    reference = _config(FFT_PRESETS[name], distributed=distributed)
    expected_model = OmegaConf.to_container(reference.model, resolve=True)
    assert isinstance(expected_model, dict)
    expected_model["loss_func"] = OmegaConf.to_container(
        cfg.model.loss_func, resolve=True
    )
    assert OmegaConf.to_container(cfg.model, resolve=True) == expected_model
    for key in ("datamodule", "trainer", "optimizer"):
        assert OmegaConf.to_container(cfg[key], resolve=True) == OmegaConf.to_container(
            reference[key], resolve=True
        )
    assert cfg.datamodule.n_steps_input == cfg.datamodule.n_steps_output == 1
    assert cfg.datamodule.stride == 1
    assert cfg.datamodule.batch_size == 32
    assert cfg.model.n_members == 8
    assert cfg.model.residual_prediction
    assert cfg.model.residual_use_delta_stats
    assert not cfg.model.train_in_latent_space
    assert not cfg.model.encoder.with_constants
    assert not cfg.model.processor.include_global_cond
    assert cfg.autoencoder_checkpoint is None
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == 256
    assert cfg.trainer.max_steps == -1
    assert cfg.trainer.max_time == "00:07:45:00"
    assert cfg.optimizer.scheduler_interval == "epoch"
    assert cfg.optimizer.learning_rate == 2e-4
    assert cfg.optimizer.min_lr_ratio == cfg.optimizer.warmup == 0
    assert set(cfg.model.loss_func) == {"_target_", "weights", "losses"}
    loss = _build_loss_func(cfg.model)
    assert isinstance(loss, WeightedSumLoss)
    assert loss.requires_ambient_predictions
    assert loss.weights == (1.0, 0.1)
    assert isinstance(loss.losses[0], AlphaFairCRPSLoss)
    assert loss.losses[0].alpha == 0.95
    assert isinstance(instantiate(cfg.model.loss_func), WeightedSumLoss)
    if distributed:
        assert cfg.trainer.devices == cfg.hydra.launcher.gpus_per_node == 4
        assert cfg.trainer.num_nodes == cfg.hydra.launcher.nodes == 1
        assert cfg.hydra.launcher.tasks_per_node == 4
        assert cfg.hydra.launcher.cpus_per_task == 72
        assert cfg.hydra.launcher.timeout_min == 480
        assert cfg.logging.wandb.enabled
        assert cfg.logging.wandb.mode == "online"
    else:
        assert not cfg.logging.wandb.enabled


def test_structured_control_changes_only_the_spatial_source():
    reference, cfg = _config(SPATIAL), _config(STRUCTURED)
    expected_model = OmegaConf.to_container(reference.model, resolve=True)
    assert isinstance(expected_model, dict)
    expected_model["processor"]["noise_source"] = OmegaConf.to_container(
        cfg.model.processor.noise_source, resolve=True
    )
    assert OmegaConf.to_container(cfg.model, resolve=True) == expected_model
    for key in ("datamodule", "trainer", "optimizer"):
        assert OmegaConf.to_container(cfg[key], resolve=True) == OmegaConf.to_container(
            reference[key], resolve=True
        )
    assert cfg.model.processor.noise_mode == "spatial"
    assert cfg.model.processor.n_noise_input_channels == 1
    assert cfg.model.processor.n_noise_channels == 256
    source = instantiate(cfg.model.processor.noise_source)
    assert isinstance(source, SpatialFourierSource)
    assert source.wavevectors == WAVEVECTORS
    assert 2 * len(source.wavevectors) * cfg.model.processor.n_noise_input_channels == 8
    assert not list(source.parameters())
    assert isinstance(_build_loss_func(cfg.model), AlphaFairCRPSLoss)


@pytest.mark.parametrize("name", [STRUCTURED, *FFT_PRESETS])
def test_production_backbones_keep_capacity_and_resolved_noise(name):
    cfg = _config(name)
    assert cfg.model.processor.hidden_dim == 224
    assert cfg.model.processor.n_layers == 12
    assert cfg.model.processor.num_heads == 8
    assert cfg.model.processor.patch_size == 4
    with torch.device("meta"):
        processor = instantiate(
            cfg.model.processor,
            in_channels=3,
            out_channels=3,
            spatial_resolution=[64, 64],
        )
    expected = 10_231_504 if cfg.model.processor.noise_mode == "global" else 10_232_016
    assert sum(p.numel() for p in processor.parameters()) == expected
    if cfg.model.processor.noise_source is not None:
        assert processor.noise_source.wavevectors == WAVEVECTORS
        assert all(2 * abs(k) < 16 for pair in WAVEVECTORS for k in pair)


@pytest.mark.parametrize("name", [STRUCTURED, *FFT_PRESETS])
def test_launch_preview_preserves_the_explicit_allocation(name, monkeypatch):
    monkeypatch.chdir(ROOT)
    launcher, module_overrides = _resolve_launcher_submission_context(
        [
            f"local_experiment=swe64s/{name}",
            "+distributed=ddp_4gpu_slurm",
            "trainer.max_time=00:07:45:00",
            "++hydra.launcher.nodes=1",
            "++hydra.launcher.cpus_per_task=72",
        ]
    )
    assert launcher["timeout_min"] == 480
    assert launcher["gpus_per_node"] == launcher["tasks_per_node"] == 4
    assert launcher["nodes"] == 1
    assert launcher["cpus_per_task"] == 72
    assert "trainer.max_time=00:07:45:00" in module_overrides
    preview = _build_sbatch_command(
        job_name=name,
        log_dir=Path("/preflight"),
        launcher_cfg=launcher,
        batch_script_path=Path("/preflight/preview.sh"),
    )
    assert "--time=08:00:00" in preview
    assert "--nodes=1" in preview
    assert "--gpus-per-node=4" in preview
    assert "--ntasks-per-node=4" in preview


@pytest.mark.parametrize("name", FFT_PRESETS)
@pytest.mark.parametrize("spectral_calibration", [False, True])
def test_fourier_score_uses_reconstructed_states_with_finite_gradients(
    name, spectral_calibration
):
    cfg = _config(name)
    # Keep real grid, patching, channel/noise layout and eight ensemble members.
    cfg.model.processor.hidden_dim = 16
    cfg.model.processor.n_layers = 1
    cfg.model.processor.num_heads = 2
    if spectral_calibration:
        cfg.model.loss_func.losses.append(
            {
                "_target_": "autocast.losses.SpectralSpreadErrorLoss",
                "edges": [0.05, 0.125, 0.25, 0.5, 0.71],
            }
        )
        cfg.model.loss_func.weights.append(0.01)
    norm = ZScoreNormalization(
        stats={
            "mean": {"h": 1.0, "u": 0.2, "v": -0.3},
            "std": {"h": 0.02, "u": 0.1, "v": 0.2},
            "mean_delta": {"h": 0.001, "u": 0.02, "v": -0.04},
            "std_delta": {"h": 0.003, "u": 0.04, "v": 0.06},
        },
        core_field_names=["h", "u", "v"],
        core_constant_field_names=[],
    )
    batch = Batch(
        input_fields=torch.randn(1, 1, 64, 64, 3),
        output_fields=torch.randn(1, 1, 64, 64, 3),
        constant_scalars=torch.tensor([[0.1]]),
        constant_fields=None,
    )
    stats = {
        "channel_count": 3,
        "n_steps_input": 1,
        "n_steps_output": 1,
        "example_batch": batch,
    }
    datamodule = SimpleNamespace(train_dataset=SimpleNamespace(norm=norm))
    model = setup_epd_model(cfg, stats, cast(Any, datamodule))
    assert isinstance(model, EncoderProcessorDecoderEnsemble)
    decoded = []
    decode = model.encoder_decoder.decoder.decode

    def capture_decode(mapped):
        values = decode(mapped)
        decoded.append(values)
        return values

    with patch.object(
        model.encoder_decoder.decoder, "decode", side_effect=capture_decode
    ):
        loss, prediction = model.loss(batch)
    assert prediction is not None
    assert prediction.shape == (1, 1, 64, 64, 3, 8)
    correction = (
        decoded[0] * torch.tensor([0.003, 0.04, 0.06])
        + torch.tensor([0.001, 0.02, -0.04])
    ) / torch.tensor([0.02, 0.1, 0.2])
    expected_state = batch.input_fields[..., None] + rearrange(
        correction, "(b m) t ... c -> b t ... c m", b=1, m=8
    )
    torch.testing.assert_close(prediction, expected_state)
    coefficient_score = CoefficientCRPSLoss(alpha=0.95)(
        torch.fft.fftn(prediction, dim=(2, 3), norm="ortho"),
        torch.fft.fftn(batch.output_fields, dim=(2, 3), norm="ortho"),
    )
    expected_loss = AlphaFairCRPSLoss(alpha=0.95)(prediction, batch.output_fields)
    expected_loss += 0.1 * coefficient_score
    if spectral_calibration:
        auxiliary = SpectralSpreadErrorLoss([0.05, 0.125, 0.25, 0.5, 0.71])
        auxiliary_loss = auxiliary(prediction, batch.output_fields)
        expected_loss += 0.01 * auxiliary_loss
        auxiliary_gradients = torch.autograd.grad(
            auxiliary_loss,
            tuple(p for p in model.processor.parameters() if p.requires_grad),
            retain_graph=True,
            allow_unused=True,
        )
        assert any(
            grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
            for grad in auxiliary_gradients
        )
    torch.testing.assert_close(loss, expected_loss)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.processor.parameters()
        if p.requires_grad
    )

    if name == FFT_SPATIAL:
        batch.output_fields = torch.randn(1, 2, 64, 64, 3)
        model.eval()
        with torch.no_grad():
            for ratio in (0.0, 1.0):
                forecast, truth = model.rollout(
                    batch,
                    stride=1,
                    max_rollout_steps=2,
                    n_members=2,
                    teacher_forcing_ratio=ratio,
                    free_running_only=ratio == 0.0,
                )
                assert forecast.shape == (1, 2, 64, 64, 3, 2)
                assert torch.isfinite(forecast).all()
                torch.testing.assert_close(
                    truth, norm.denormalize_flattened(batch.output_fields, "variable")
                )
