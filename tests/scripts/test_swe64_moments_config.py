"""Check the opt-in joint-moment control against the matched SWE64 baseline."""

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

from autocast.losses import WeightedSumLoss
from autocast.losses.moments import SpectralMomentLoss
from autocast.scripts.setup import setup_epd_model
from autocast.types import Batch

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "crps_fft_spatial1d_residual_one_output_12block_256ep_8h"
PRESET = "crps_fft_moments_spatial1d_residual_one_output_12block_256ep_8h"
TEST_WEIGHT = 0.003  # Override fixture; the selected experiment default is 5e-4.


def _config(name=PRESET, *, weight=None, distributed=False):
    overrides = [
        f"local_experiment=swe64s/{name}",
        f"hydra.searchpath=[file://{ROOT / 'local_hydra'}]",
    ]
    if weight is not None:
        overrides.append(f"model.loss_func.weights.2={weight}")
    if distributed:
        overrides.extend(
            [
                "+distributed=ddp_4gpu_slurm",
                "trainer.max_time=00:07:45:00",
                "logging.wandb.enabled=true",
                "logging.wandb.mode=offline",
            ]
        )
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        return compose(
            config_name="encoder_processor_decoder",
            overrides=overrides,
            return_hydra_config=distributed,
        )


def test_moment_preset_changes_only_loss_and_experiment_name():
    baseline = _config(BASELINE)
    cfg = _config(weight=TEST_WEIGHT)
    baseline.experiment_name = cfg.experiment_name
    expected = OmegaConf.to_container(baseline, resolve=True)
    actual = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(expected, dict)
    assert isinstance(actual, dict)
    expected["model"]["loss_func"] = actual["model"]["loss_func"]
    assert actual == expected
    assert cfg.model.loss_func.losses[:2] == baseline.model.loss_func.losses
    assert cfg.model.loss_func.weights[:2] == baseline.model.loss_func.weights
    assert cfg.model.loss_func.losses[2] == {
        "_target_": "autocast.losses.moments.SpectralMomentLoss",
        "norm": "ortho",
    }
    assert cfg.model.processor.noise_mode == "spatial"
    assert cfg.model.processor.n_noise_input_channels == 1
    assert cfg.model.processor.hidden_dim == 224
    assert cfg.model.processor.n_layers == 12
    assert cfg.model.processor.num_heads == 8
    assert cfg.model.processor.patch_size == 4
    assert cfg.model.n_members == 8
    assert cfg.datamodule.batch_size == 32
    assert cfg.datamodule.n_steps_input == cfg.datamodule.n_steps_output == 1
    assert cfg.datamodule.stride == 1
    assert cfg.trainer.max_epochs == cfg.optimizer.cosine_epochs == 256
    assert cfg.trainer.max_steps == -1
    assert cfg.optimizer.scheduler_interval == "epoch"
    assert cfg.optimizer.learning_rate == 2e-4
    assert cfg.optimizer.min_lr_ratio == cfg.optimizer.warmup == 0
    assert cfg.model.residual_prediction
    assert cfg.model.residual_use_delta_stats
    assert not cfg.model.train_in_latent_space
    assert not cfg.model.encoder.with_constants
    assert not cfg.model.processor.include_global_cond
    assert cfg.autoencoder_checkpoint is None
    assert cfg.train_eval.checkpoint is None
    assert cfg.output.checkpoint_path is None
    loss = instantiate(cfg.model.loss_func)
    assert isinstance(loss, WeightedSumLoss)
    assert isinstance(loss.losses[2], SpectralMomentLoss)
    assert loss.requires_ambient_predictions
    assert loss.weights == (1.0, 0.1, TEST_WEIGHT)


def test_opt_in_auxiliary_uses_fixed_train_only_calibrated_weight():
    cfg = _config()
    loss = instantiate(cfg.model.loss_func)
    assert loss.weights == (1.0, 0.1, 5e-4)
    assert isinstance(loss.losses[2], SpectralMomentLoss)
    baseline = instantiate(_config(BASELINE).model.loss_func)
    assert baseline.weights == (1.0, 0.1)
    assert len(baseline.losses) == 2


def test_four_gpu_configuration_preserves_matched_training_budget():
    cfg = _config(weight=TEST_WEIGHT, distributed=True)
    assert cfg.trainer.devices == cfg.hydra.launcher.gpus_per_node == 4
    assert cfg.hydra.launcher.tasks_per_node == 4
    assert cfg.hydra.launcher.timeout_min == 480
    assert cfg.trainer.max_time == "00:07:45:00"
    assert cfg.logging.wandb.enabled
    assert cfg.logging.wandb.mode == "offline"
    assert 64000 // (cfg.datamodule.batch_size * 4) * cfg.trainer.max_epochs == 128000


@pytest.mark.parametrize("weight", [None, TEST_WEIGHT])
def test_fresh_small_backbone_scores_normalized_decoded_residual_states(weight):
    torch.manual_seed(153)
    cfg = _config(weight=weight)
    cfg.model.processor.hidden_dim = 16
    cfg.model.processor.n_layers = 1
    cfg.model.processor.num_heads = 2
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
        constant_scalars=None,
        constant_fields=None,
    )
    stats = {
        "channel_count": 3,
        "n_steps_input": 1,
        "n_steps_output": 1,
        "example_batch": batch,
    }
    datamodule = cast(Any, SimpleNamespace(train_dataset=SimpleNamespace(norm=norm)))
    with patch(
        "torch.load", side_effect=AssertionError("Fresh setup must not load weights")
    ):
        model = setup_epd_model(cfg, stats, datamodule)
    decoded = []
    decode = model.encoder_decoder.decoder.decode

    def capture_decode(mapped):
        values = decode(mapped)
        decoded.append(values)
        return values

    with patch.object(
        model.encoder_decoder.decoder, "decode", side_effect=capture_decode
    ):
        value, prediction = model.loss(batch)
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
    baseline_loss = instantiate(_config(BASELINE).model.loss_func)
    auxiliary = SpectralMomentLoss()
    mean_score, power_score = auxiliary.components(prediction, batch.output_fields)
    expected_loss = baseline_loss(prediction, batch.output_fields) + (
        cfg.model.loss_func.weights[2] * (mean_score + power_score)
    )
    torch.testing.assert_close(value, expected_loss)
    assert torch.isfinite(value)
    parameters = tuple(p for p in model.processor.parameters() if p.requires_grad)
    gradients = torch.autograd.grad(
        mean_score + power_score, parameters, retain_graph=True
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
    value.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    assert not any(name.startswith("loss_func.") for name in model.state_dict())


def test_joint_moment_loss_propagates_the_decoded_state_guard(
    make_toy_batch, dummy_datamodule
):
    cfg = _config(weight=TEST_WEIGHT)
    cfg.model.processor.hidden_dim = 16
    cfg.model.processor.n_layers = 1
    cfg.model.processor.num_heads = 2
    cfg.model.train_in_latent_space = True
    cfg.model.loss_func = {
        "_target_": "autocast.losses.WeightedSumLoss",
        "weights": [1.0],
        "losses": [{"_target_": "autocast.losses.moments.SpectralMomentLoss"}],
    }
    batch = make_toy_batch(t_in=1, t_out=1, w=8, h=8, c=3)
    stats = {
        "channel_count": 3,
        "n_steps_input": 1,
        "n_steps_output": 1,
        "example_batch": batch,
    }
    with pytest.raises(ValueError, match=r"set model\.train_in_latent_space=false"):
        setup_epd_model(cfg, stats, dummy_datamodule)
