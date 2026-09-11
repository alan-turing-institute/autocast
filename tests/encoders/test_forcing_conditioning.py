"""Tests for conditioning an encoder on time-varying forcing fields."""

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from autocast.data.dataset import ReactionDiffusionDataset
from autocast.decoders.channels_last import ChannelsLast
from autocast.encoders.permute_concat import PermuteConcat
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.processors.base import Processor
from autocast.scripts.setup import setup_autoencoder_components
from autocast.types import Batch, EncodedBatch, Tensor
from autocast.types.collation import collate_batches

W, H, C, C_F = 4, 4, 2, 3
T_IN, T_OUT = 2, 1


def _batch(with_forcing: bool = True, n_forcing_steps: int = T_IN + T_OUT) -> Batch:
    return Batch(
        input_fields=torch.randn(2, T_IN, W, H, C),
        output_fields=torch.randn(2, T_OUT, W, H, C),
        constant_scalars=None,
        constant_fields=None,
        forcing_fields=(
            torch.randn(2, n_forcing_steps, W, H, C_F) if with_forcing else None
        ),
    )


class _Tiny(Processor[EncodedBatch]):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:  # noqa: ARG002
        return self.conv(x)

    def loss(self, batch: EncodedBatch) -> Tensor:
        raise NotImplementedError


# --- encoder ---


def test_forcing_is_ignored_when_not_requested():
    """Default stays off, so existing encoders are unaffected by the new field."""
    encoder = PermuteConcat(in_channels=C, n_steps_input=T_IN, with_forcing=False)

    encoded = encoder.encode(_batch())

    assert encoded.shape == (2, C * T_IN, W, H)


def test_forcing_widens_the_channel_stack():
    encoder = PermuteConcat(in_channels=C + C_F, n_steps_input=T_IN, with_forcing=True)

    encoded = encoder.encode(_batch())

    assert encoded.shape == (2, (C + C_F) * T_IN, W, H)


def test_forcing_absent_from_batch_is_tolerated():
    """`with_forcing` on a batch that carries none falls back to state only."""
    encoder = PermuteConcat(in_channels=C + C_F, n_steps_input=T_IN, with_forcing=True)

    encoded = encoder.encode(_batch(with_forcing=False))

    assert encoded.shape == (2, C * T_IN, W, H)


def test_encoded_forcing_values_are_the_input_window_steps():
    """Only the steps aligned with `input_fields` are concatenated."""
    encoder = PermuteConcat(in_channels=C + C_F, n_steps_input=T_IN, with_forcing=True)
    batch = _batch()

    encoded = encoder.encode(batch)

    # Layout is (c t) with channels outermost, so forcing channel 0 step 0
    # lands immediately after the C state channels' T_IN steps.
    assert torch.equal(encoded[:, C * T_IN], batch.forcing_fields[:, 0, ..., 0])


def test_too_short_forcing_window_is_reported():
    encoder = PermuteConcat(in_channels=C + C_F, n_steps_input=T_IN, with_forcing=True)

    with pytest.raises(ValueError, match="must span at least n_steps_input"):
        encoder.encode(_batch(n_forcing_steps=T_IN - 1))


# --- channel accounting ---


def test_setup_widens_encoder_for_forcing_channels():
    config = OmegaConf.create(
        {
            "model": {
                "encoder": {
                    "_target_": "autocast.encoders.permute_concat.PermuteConcat",
                    "in_channels": "auto",
                    "n_steps_input": "auto",
                    "with_forcing": True,
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
        "output_channel_count": C,
        "n_steps_input": T_IN,
        "n_steps_output": T_OUT,
        "n_forcing_field_channels": C_F,
    }

    setup_autoencoder_components(config, stats)

    assert config.model.encoder.in_channels == C + C_F
    # The decoder predicts state only, so forcing must not widen it.
    assert config.model.decoder.output_channels == C


# --- dataset through to a training step ---


def test_forced_model_trains_end_to_end():
    dataset = ReactionDiffusionDataset(
        data_path=None,
        data={
            "data": torch.randn(2, 6, W, H, C),
            "constant_scalars": None,
            "constant_fields": None,
            "forcing_fields": torch.randn(2, 6, W, H, C_F),
        },
        n_steps_input=T_IN,
        n_steps_output=T_OUT,
    )
    batch = collate_batches([dataset[0], dataset[1]])
    assert batch.forcing_fields is not None

    encoder = PermuteConcat(in_channels=C + C_F, n_steps_input=T_IN, with_forcing=True)
    model = EncoderProcessorDecoder(
        encoder_decoder=EncoderDecoder(
            encoder=encoder,
            decoder=ChannelsLast(output_channels=C, time_steps=T_OUT),
            loss_func=nn.MSELoss(),
        ),
        processor=_Tiny((C + C_F) * T_IN, C * T_OUT),
        loss_func=nn.MSELoss(),
        optimizer_config=OmegaConf.create({"_target_": "torch.optim.Adam", "lr": 1e-3}),
    )

    loss = model.training_step(batch, 0)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.processor.conv.weight.grad is not None


def test_forcing_survives_batch_transforms():
    """The field-driven transforms pick up `forcing_fields` with no extra code."""
    batch = _batch()

    moved = batch.to("cpu")
    repeated = batch.repeat(3)

    assert moved.forcing_fields is not None
    assert repeated.forcing_fields.shape == (6, T_IN + T_OUT, W, H, C_F)
