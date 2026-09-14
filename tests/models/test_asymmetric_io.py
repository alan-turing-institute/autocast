"""End-to-end tests for models whose output fields differ from their inputs."""

import pytest
import torch
from conftest import get_optimizer_config
from omegaconf import OmegaConf
from torch import nn

from autocast.data.dataset import ReactionDiffusionDataset
from autocast.decoders.channels_last import ChannelsLast
from autocast.encoders.permute_concat import PermuteConcat
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.processors.base import Processor
from autocast.scripts.setup import (
    _resolve_supports_rollout,
    setup_autoencoder_components,
)
from autocast.types import Batch, EncodedBatch, FieldSpec, IOSpec, Tensor
from autocast.types.collation import collate_batches

N_TRAJ, N_T, W, H, C = 2, 6, 8, 8, 2
T_IN, T_OUT = 2, 1
OUT_CHANNEL_IDXS = (1,)  # predict V only, from both U and V


class _Tiny(Processor[EncodedBatch]):
    """Maps the flattened input stack to the flattened output stack."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:  # noqa: ARG002
        return self.conv(x)

    def loss(self, batch: EncodedBatch) -> Tensor:
        raise NotImplementedError


@pytest.fixture
def asymmetric_dataset() -> ReactionDiffusionDataset:
    return ReactionDiffusionDataset(
        data_path=None,
        data={
            "data": torch.randn(N_TRAJ, N_T, W, H, C),
            "constant_scalars": None,
            "constant_fields": None,
        },
        n_steps_input=T_IN,
        n_steps_output=T_OUT,
        output_channel_idxs=OUT_CHANNEL_IDXS,
    )


def _build_epd() -> EncoderProcessorDecoder:
    """An EPD wired for 2 input channels in and 1 output channel out."""
    encoder = PermuteConcat(in_channels=C, n_steps_input=T_IN, with_constants=False)
    decoder = ChannelsLast(output_channels=len(OUT_CHANNEL_IDXS), time_steps=T_OUT)
    return EncoderProcessorDecoder(
        encoder_decoder=EncoderDecoder(
            encoder=encoder, decoder=decoder, loss_func=nn.MSELoss()
        ),
        processor=_Tiny(C * T_IN, len(OUT_CHANNEL_IDXS) * T_OUT),
        loss_func=nn.MSELoss(),
        optimizer_config=get_optimizer_config(),
        supports_rollout=False,
    )


# --- dataset -> model, end to end ---


def test_dataset_yields_asymmetric_batches(asymmetric_dataset):
    batch = collate_batches([asymmetric_dataset[0], asymmetric_dataset[1]])

    assert batch.input_fields.shape == (2, T_IN, W, H, C)
    assert batch.output_fields.shape == (2, T_OUT, W, H, len(OUT_CHANNEL_IDXS))


def test_epd_forward_produces_output_shaped_predictions(asymmetric_dataset):
    batch = collate_batches([asymmetric_dataset[0], asymmetric_dataset[1]])

    prediction = _build_epd()(batch)

    assert prediction.shape == batch.output_fields.shape


def test_epd_training_step_runs_on_asymmetric_batches(asymmetric_dataset):
    """The full data-space path trains when outputs differ from inputs."""
    batch = collate_batches([asymmetric_dataset[0], asymmetric_dataset[1]])

    loss = _build_epd().training_step(batch, 0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_epd_backward_reaches_processor_parameters(asymmetric_dataset):
    batch = collate_batches([asymmetric_dataset[0], asymmetric_dataset[1]])
    model = _build_epd()

    model.training_step(batch, 0).backward()

    grad = model.processor.conv.weight.grad
    assert grad is not None
    assert torch.any(grad != 0)


# --- latent-space losses are refused with a usable message ---


def test_latent_space_loss_on_asymmetric_batch_explains_itself(asymmetric_dataset):
    batch = collate_batches([asymmetric_dataset[0], asymmetric_dataset[1]])
    encoder = PermuteConcat(in_channels=C, n_steps_input=T_IN, with_constants=False)

    with pytest.raises(ValueError, match="cannot encode the target fields"):
        encoder.encode_output(
            Batch(
                input_fields=batch.input_fields,
                output_fields=batch.output_fields,
                constant_scalars=None,
                constant_fields=None,
            )
        )


# --- rollout is refused, not attempted ---


def test_rollout_is_refused_for_asymmetric_models(asymmetric_dataset):
    batch = collate_batches([asymmetric_dataset[0], asymmetric_dataset[1]])

    with pytest.raises(NotImplementedError, match="supports_rollout=False"):
        _build_epd().rollout(batch, stride=T_OUT)


# --- supports_rollout is derived from the data, not hand-set ---


def _stats(in_channels: int, out_channels: int, out_res=(W, H)) -> dict:
    return {
        "io_spec": IOSpec(
            inputs=FieldSpec(T_IN, in_channels, (W, H)),
            outputs=FieldSpec(T_OUT, out_channels, out_res),
        )
    }


def test_supports_rollout_true_for_symmetric_shapes():
    assert _resolve_supports_rollout(OmegaConf.create({}), _stats(2, 2)) is True


def test_supports_rollout_false_for_differing_channels():
    assert _resolve_supports_rollout(OmegaConf.create({}), _stats(2, 1)) is False


def test_supports_rollout_false_for_differing_resolution():
    stats = _stats(2, 2, out_res=(W * 2, H * 2))
    assert _resolve_supports_rollout(OmegaConf.create({}), stats) is False


def test_explicit_config_overrides_the_derived_value():
    cfg = OmegaConf.create({"supports_rollout": False})
    assert _resolve_supports_rollout(cfg, _stats(2, 2)) is False


def test_missing_io_spec_defaults_to_allowing_rollout():
    assert _resolve_supports_rollout(OmegaConf.create({}), {}) is True


# --- the decoder follows the OUTPUT channel count ---


def test_decoder_out_channels_follow_output_channel_count():
    """`auto` decoder channels must resolve to the output count, not the input."""
    config = OmegaConf.create(
        {
            "model": {
                "encoder": {
                    "_target_": "autocast.encoders.permute_concat.PermuteConcat",
                    "in_channels": "auto",
                    "n_steps_input": "auto",
                },
                "decoder": {
                    "_target_": "autocast.decoders.channels_last.ChannelsLast",
                    "output_channels": "auto",
                    "time_steps": T_OUT,
                },
            }
        }
    )
    stats = {
        "channel_count": C,
        "output_channel_count": len(OUT_CHANNEL_IDXS),
        "n_steps_input": T_IN,
        "n_steps_output": T_OUT,
    }

    setup_autoencoder_components(config, stats)

    assert config.model.encoder.in_channels == C
    assert config.model.decoder.output_channels == len(OUT_CHANNEL_IDXS)
