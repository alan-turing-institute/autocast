"""Compose scale scores with Fourier CRPS and train on decoded residual states."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

from autocast.losses import AlphaFairCRPSLoss, TransformedLoss, WeightedSumLoss
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.scripts.setup import setup_epd_model
from autocast.types import Batch


def _config(repo_root: Path, operator: str) -> DictConfig:
    with initialize_config_dir(
        version_base=None, config_dir=str(repo_root / "src/autocast/configs")
    ):
        cfg = compose(
            config_name="encoder_processor_decoder",
            overrides=["loss@model.loss_func=ensemble_fourier_crps"],
        )
    if operator == "gaussian":
        settings = [{"sigma": 2}, {"sigma": 1}]
        operators = [
            {
                "_target_": "autocast.losses.GaussianSmoothing",
                "boundary": "periodic",
                **setting,
            }
            for setting in settings
        ]
    else:
        operators = [
            {"_target_": "autocast.losses.FourierLowPass", "cutoff": cutoff}
            for cutoff in (0.125, 0.25)
        ]
    cfg.model.loss_func.losses.append(
        {
            "_target_": "autocast.losses.TransformedLoss",
            "requires_ambient_predictions": True,
            "decomposition": {
                "_target_": "autocast.losses.ScaleDecomposition",
                "scales": [["coarse", operators[0]], ["intermediate", operators[1]]],
            },
            "score": {"_target_": "autocast.losses.AlphaFairCRPSLoss", "alpha": 0.95},
        }
    )
    cfg.model.loss_func.weights.append(0.2)
    return cfg


def _small_model_config(repo_root: Path, batch: Batch, operator: str) -> DictConfig:
    cfg = _config(repo_root, operator)
    with open_dict(cfg.model):
        cfg.model.encoder = {
            "_target_": "autocast.encoders.permute_concat.PermuteConcat",
            "in_channels": "auto",
            "n_steps_input": "auto",
            "with_constants": False,
        }
        cfg.model.decoder = {
            "_target_": "autocast.decoders.channels_last.ChannelsLast",
            "output_channels": batch.output_fields.shape[-1],
            "time_steps": batch.output_fields.shape[1],
        }
        cfg.model.processor = {
            "_target_": "autocast.processors.fno.FNOProcessor",
            "in_channels": "auto",
            "out_channels": "auto",
            "n_modes": [2, 2],
            "hidden_channels": 8,
            "n_layers": 1,
        }
        cfg.model.n_members = 3
        cfg.model.train_in_latent_space = False
        cfg.model.freeze_encoder_decoder = False
        cfg.model.residual_prediction = True
        cfg.model.residual_use_delta_stats = False
        cfg.model.train_metrics = []
        cfg.model.val_metrics = []
        cfg.model.test_metrics = []
    return cfg


def _stats(batch: Batch) -> dict:
    return {
        "channel_count": batch.input_fields.shape[-1],
        "n_steps_input": batch.input_fields.shape[1],
        "n_steps_output": batch.output_fields.shape[1],
        "example_batch": batch,
    }


@pytest.mark.parametrize("operator", ["gaussian", "low_pass"])
def test_scale_lists_compose_with_crps_and_fourier(REPO_ROOT: Path, operator: str):
    loss = instantiate(_config(REPO_ROOT, operator).model.loss_func)
    assert isinstance(loss, WeightedSumLoss)
    assert loss.weights == (1.0, 0.1, 0.2)
    assert loss.requires_ambient_predictions
    scale = loss.losses[2]
    assert isinstance(scale, TransformedLoss)
    predictions = torch.randn(2, 3, 4, 6, 2, 3, requires_grad=True)
    targets = torch.randn(2, 3, 4, 6, 2)
    pred_components = scale.decomposition(predictions, n_spatial_dims=2)
    target_components = scale.decomposition(targets, n_spatial_dims=2)
    assert tuple(pred_components) == ("coarse", "intermediate", "residual")
    score = AlphaFairCRPSLoss(alpha=0.95)
    expected_scale = torch.stack(
        [
            score(pred_components[name], target_components[name])
            for name in pred_components
        ]
    ).mean()
    expected = (
        loss.losses[0](predictions, targets)
        + 0.1 * loss.losses[1](predictions, targets)
        + 0.2 * expected_scale
    )
    actual = loss(predictions, targets)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


@pytest.mark.parametrize("operator", ["gaussian", "low_pass"])
def test_scale_loss_uses_decoded_persistence_residual_states(
    REPO_ROOT: Path, make_toy_batch, dummy_datamodule, operator: str
):
    batch = make_toy_batch(t_in=2, t_out=3, w=4, h=6, c=2)
    cfg = _small_model_config(REPO_ROOT, batch, operator)
    model = setup_epd_model(cfg, _stats(batch), dummy_datamodule)
    assert isinstance(model, EncoderProcessorDecoderEnsemble)
    loss, predictions = model.loss(batch)
    assert predictions is not None
    assert predictions.shape == (*batch.output_fields.shape, 3)
    encoded, condition = model.encoder_decoder.encoder.encode_with_cond(batch)
    residual = model.encoder_decoder.decoder.decode(
        model.processor.map(encoded, condition)
    )
    torch.testing.assert_close(
        predictions[..., 0], residual + batch.input_fields[:, -1:]
    )
    expected = instantiate(cfg.model.loss_func)(predictions, batch.output_fields)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    gradients = [parameter.grad for parameter in model.processor.parameters()]
    assert gradients
    assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients)
    assert any(grad is not None and grad.abs().sum() > 0 for grad in gradients)


@pytest.mark.parametrize("operator", ["gaussian", "low_pass"])
def test_scale_term_alone_rejects_latent_training(
    REPO_ROOT: Path, make_toy_batch, dummy_datamodule, operator: str
):
    batch = make_toy_batch(t_in=2, t_out=3, w=4, h=6, c=2)
    cfg = _small_model_config(REPO_ROOT, batch, operator)
    # Isolate the scale term so the Fourier term cannot mask a missing guard.
    cfg.model.loss_func.losses = [cfg.model.loss_func.losses[2]]
    cfg.model.loss_func.weights = [1.0]
    cfg.model.train_in_latent_space = True
    with pytest.raises(ValueError, match=r"set model\.train_in_latent_space=false"):
        setup_epd_model(cfg, _stats(batch), dummy_datamodule)
