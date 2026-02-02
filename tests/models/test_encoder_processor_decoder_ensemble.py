import pytest
import torch
from torch import nn

from autocast.decoders.channels_last import ChannelsLast
from autocast.encoders.permute_concat import PermuteConcat
from autocast.losses.ensemble import CRPSLoss
from autocast.metrics.ensemble import _common_crps_score
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.processors.base import Processor
from autocast.types import Batch, EncodedBatch, Tensor


class SimpleLatentProcessor(Processor):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:  # noqa: ARG002
        return self.conv(x)

    def loss(self, batch: EncodedBatch) -> Tensor:
        preds = self.map(batch.encoded_inputs, batch.global_cond)
        return nn.functional.mse_loss(preds, batch.encoded_output_fields)


def test_epd_ensemble_forward_shape():
    """Test that EPD ensemble forward pass returns (B, T, ..., M)."""
    n_members = 3
    batch_size = 2
    t_steps = 2
    w, h = 8, 8
    channels = 4

    input_fields = torch.randn(batch_size, t_steps, w, h, channels)
    output_fields = torch.randn(batch_size, t_steps, w, h, channels)
    batch = Batch(input_fields, output_fields, None, None)

    encoder = PermuteConcat(
        in_channels=channels, n_steps_input=t_steps, with_constants=False
    )
    encoder_output_channels = channels * t_steps
    decoder = ChannelsLast(output_channels=channels, time_steps=t_steps)
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder)

    processor = SimpleLatentProcessor(
        in_channels=encoder_output_channels, out_channels=encoder_output_channels
    )

    model = EncoderProcessorDecoderEnsemble(
        encoder_decoder=encoder_decoder,
        processor=processor,
        n_members=n_members,
    )

    output = model(batch)

    # Expected: (B, T, W, H, C, M)
    expected_shape = (batch_size, t_steps, w, h, channels, n_members)
    assert output.shape == expected_shape


def test_epd_ensemble_loss_latent_integration():
    """Test EPD ensemble loss when training in latent space."""
    n_members = 3
    batch_size = 2
    t_steps = 2
    w, h = 8, 8
    channels = 2

    input_fields = torch.randn(batch_size, t_steps, w, h, channels)
    output_fields = torch.randn(batch_size, t_steps, w, h, channels)
    batch = Batch(input_fields, output_fields, None, None)

    encoder = PermuteConcat(
        in_channels=channels, n_steps_input=t_steps, with_constants=False
    )
    encoder_output_channels = channels * t_steps

    decoder = ChannelsLast(output_channels=channels, time_steps=t_steps)
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder)

    processor = SimpleLatentProcessor(
        in_channels=encoder_output_channels, out_channels=encoder_output_channels
    )

    model = EncoderProcessorDecoderEnsemble(
        encoder_decoder=encoder_decoder,
        processor=processor,
        n_members=n_members,
        train_in_latent_space=True,
        loss_func=CRPSLoss(),
    )

    loss, _ = model.loss(batch)

    # Calculate expected loss manually
    encoded_batch_tmp = encoder.encode_batch(batch)
    with torch.no_grad():
        single_preds = processor.map(encoded_batch_tmp.encoded_inputs, None)
        preds = single_preds.unsqueeze(-1).expand(-1, -1, -1, -1, n_members)

    encoded_targets = encoded_batch_tmp.encoded_output_fields
    expected_crps = _common_crps_score(preds, encoded_targets, adjustment_factor=1.0)
    expected_loss = expected_crps.mean()

    assert loss.item() == pytest.approx(expected_loss.item())


def test_epd_ensemble_loss_fallback():
    """Test fallback when n_members=1 or train_in_latent_space=False."""
    n_members = 1
    batch_size = 2
    t_steps = 2
    w, h = 8, 8
    channels = 2

    input_fields = torch.randn(batch_size, t_steps, w, h, channels)
    output_fields = torch.randn(batch_size, t_steps, w, h, channels)
    batch = Batch(input_fields, output_fields, None, None)

    encoder = PermuteConcat(
        in_channels=channels, n_steps_input=t_steps, with_constants=False
    )
    decoder = ChannelsLast(output_channels=channels, time_steps=t_steps)
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder)

    encoder_output_channels = channels * t_steps
    processor = SimpleLatentProcessor(
        in_channels=encoder_output_channels, out_channels=encoder_output_channels
    )

    def tolerant_loss(preds, targets):
        if preds.shape[-1] == 1 and preds.ndim == targets.ndim + 1:
            preds = preds.squeeze(-1)
        return nn.functional.mse_loss(preds, targets)

    model = EncoderProcessorDecoderEnsemble(
        encoder_decoder=encoder_decoder,
        processor=processor,
        n_members=n_members,
        loss_func=tolerant_loss,
    )

    loss, _ = model.loss(batch)
    assert loss.ndim == 0
