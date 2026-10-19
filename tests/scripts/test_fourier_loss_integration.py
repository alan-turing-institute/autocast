"""Exercise the Fourier loss through configuration and decoded model training."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

from autocast.losses import (
    AlphaFairCRPSLoss,
    CoefficientCRPSLoss,
    MCDropoutMSEL2Loss,
    WeightedSumLoss,
)
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.scripts.setup import _build_loss_func, setup_epd_model
from autocast.types import Batch


def _config(repo_root: Path, overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(
        version_base=None, config_dir=str(repo_root / "src/autocast/configs")
    ):
        return compose(
            config_name="encoder_processor_decoder", overrides=overrides or []
        )


def _small_model_config(repo_root: Path, batch: Batch) -> DictConfig:
    cfg = _config(repo_root, ["loss@model.loss_func=ensemble_fourier_crps"])
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


def test_fourier_preset_is_opt_in_and_preserves_null_loss(REPO_ROOT: Path):
    cfg = _config(REPO_ROOT)
    assert cfg.model.loss_func._target_ == "torch.nn.MSELoss"
    assert isinstance(_build_loss_func(cfg.model), torch.nn.MSELoss)

    cfg = _config(REPO_ROOT, ["model.loss_func=null"])
    assert cfg.model.processor._target_ == (
        "autocast.processors.flow_matching.FlowMatchingProcessor"
    )
    assert _build_loss_func(cfg.model) is None


@pytest.mark.parametrize(
    "overrides",
    [
        [],
        ["experiment=epd_crps_vit_large_ps4_64"],
        ["local_experiment=epd/gray_scott/crps_vit_azula_large"],
        [
            "local_experiment=ablations/mc_dropout/gray_scott/"
            "mse_vit_azula_mc_dropout_large"
        ],
    ],
)
def test_fourier_preset_overrides_experiment_loss(
    REPO_ROOT: Path, overrides: list[str]
):
    cfg = _config(REPO_ROOT, [*overrides, "loss@model.loss_func=ensemble_fourier_crps"])
    assert set(cfg.model.loss_func) == {"_target_", "weights", "losses"}
    loss = _build_loss_func(cfg.model)
    assert isinstance(loss, WeightedSumLoss)
    assert loss.requires_ambient_predictions
    assert loss.weights == (1.0, 0.1)
    # Replacement must work directly through Hydra, not just EPD setup.
    assert isinstance(instantiate(cfg.model.loss_func), WeightedSumLoss)


def test_fourier_preset_preserves_architecture_and_explicit_overrides(REPO_ROOT: Path):
    experiment = (
        "local_experiment=ablations/mc_dropout/gray_scott/"
        "mse_vit_azula_mc_dropout_large"
    )
    original = _config(REPO_ROOT, [experiment])
    cfg = _config(
        REPO_ROOT,
        [
            experiment,
            "loss@model.loss_func=ensemble_fourier_crps",
            "model.loss_func.weights=[2.0,0.25]",
            "model.loss_func.losses.1.score.alpha=1.0",
        ],
    )
    loss = instantiate(cfg.model.loss_func)
    assert loss.weights == (2.0, 0.25)
    assert cfg.model.loss_func.losses[1].score.alpha == 1.0
    assert cfg.model.encoder == original.model.encoder
    assert cfg.model.processor == original.model.processor
    assert cfg.model.decoder == original.model.decoder


def test_fourier_preset_rejects_latent_training(
    REPO_ROOT: Path, make_toy_batch, dummy_datamodule
):
    batch = make_toy_batch(t_in=2, t_out=3, w=4, h=6, c=2)
    cfg = _small_model_config(REPO_ROOT, batch)
    cfg.model.train_in_latent_space = True
    with pytest.raises(ValueError, match=r"set model\.train_in_latent_space=false"):
        setup_epd_model(cfg, _stats(batch), dummy_datamodule)


@pytest.mark.parametrize("residual_prediction", [False, True])
@pytest.mark.parametrize("radial", [False, True])
def test_fourier_preset_trains_on_decoded_ensemble_states(
    REPO_ROOT: Path,
    make_toy_batch,
    dummy_datamodule,
    residual_prediction: bool,
    radial: bool,
):
    torch.manual_seed(19)
    batch = make_toy_batch(t_in=2, t_out=3, w=4, h=6, c=2)
    cfg = _small_model_config(REPO_ROOT, batch)
    cfg.model.residual_prediction = residual_prediction
    if radial:
        # Replace the decomposition mapping, keeping the same public scorer.
        cfg.model.loss_func.losses[1].decomposition = {
            "_target_": "autocast.losses.RadialFourierDecomposition",
            "edges": [0.25, 0.75],
        }
    model = setup_epd_model(cfg, _stats(batch), dummy_datamodule)
    assert isinstance(model, EncoderProcessorDecoderEnsemble)

    loss, predictions = model.loss(batch)
    assert predictions is not None
    assert predictions.shape == (*batch.output_fields.shape, 3)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    encoded, condition = model.encoder_decoder.encoder.encode_with_cond(batch)
    decoded = model.encoder_decoder.decoder.decode(
        model.processor.map(encoded, condition)
    )
    expected_state = decoded
    if residual_prediction:
        expected_state = decoded + batch.input_fields[:, -1:]
    torch.testing.assert_close(predictions[..., 0], expected_state)

    expected_fourier = CoefficientCRPSLoss(alpha=0.95)(
        torch.fft.fftn(predictions, dim=(2, 3), norm="ortho"),
        torch.fft.fftn(batch.output_fields, dim=(2, 3), norm="ortho"),
    )
    # Two complete bands with equal component weights average to half the
    # full-spectrum coefficient score; unselected bins contribute zero.
    if radial:
        expected_fourier = expected_fourier / 2
    expected_loss = (
        AlphaFairCRPSLoss(alpha=0.95)(predictions, batch.output_fields)
        + 0.1 * expected_fourier
    )
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.processor.parameters()]
    assert gradients
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )
    assert any(
        gradient is not None and gradient.abs().sum() > 0 for gradient in gradients
    )


def test_existing_processor_bound_loss_remains_available(REPO_ROOT: Path):
    cfg = _config(
        REPO_ROOT,
        [
            "local_experiment=ablations/mc_dropout/gray_scott/"
            "mse_vit_azula_mc_dropout_large"
        ],
    )
    assert cfg.model.loss_func.l2_coefficient == 1e-5
    processor = torch.nn.Linear(2, 2)
    loss = _build_loss_func(cfg.model, processor)
    assert isinstance(loss, MCDropoutMSEL2Loss)
    assert loss.processor is processor


def test_fourier_preset_runtime_typing_in_fresh_process(REPO_ROOT: Path):
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        import torch
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate

        from autocast.losses import CoefficientCRPSLoss

        config_dir = str(Path(sys.argv[1]) / "src/autocast/configs")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="encoder_processor_decoder",
                overrides=["loss@model.loss_func=ensemble_fourier_crps"],
            )
        loss = instantiate(cfg.model.loss_func)
        prediction = torch.randn(2, 3, 4, 6, 2, 3, requires_grad=True)
        target = torch.randn(2, 3, 4, 6, 2)
        loss(prediction, target).backward()
        assert torch.isfinite(prediction.grad).all()

        for dtype in (torch.complex64, torch.complex128):
            for alpha in (0, 1):
                coefficients = torch.randn(
                    2, 3, 4, 6, 2, 3, dtype=dtype, requires_grad=True
                )
                target = torch.randn(2, 3, 4, 6, 2, dtype=dtype)
                CoefficientCRPSLoss(alpha=alpha)(coefficients, target).backward()
                assert torch.isfinite(coefficients.grad).all()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(REPO_ROOT)],
        cwd=REPO_ROOT,
        env={**os.environ, "RUNTIME_TYPECHECKING": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
