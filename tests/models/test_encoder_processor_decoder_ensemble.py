import pytest
import torch
from einops import repeat
from torch import nn

from autocast.decoders.channels_last import ChannelsLast
from autocast.encoders.permute_concat import PermuteConcat
from autocast.metrics.ensemble import _common_crps_score
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.processors.base import Processor
from autocast.types import Batch, EncodedBatch, Tensor


class MockProcessor(Processor):
    def __init__(self, output_shape, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = output_shape
        # Dummy parameter
        self.dummy_param = nn.Parameter(torch.tensor([0.0]))

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        b = x.shape[0]
        # output_shape is (C*T, W, H)
        return torch.zeros(b, *self.output_shape) + self.dummy_param

    def loss(self, batch: EncodedBatch) -> Tensor:
        return torch.tensor(-1.0)


def test_epd_ensemble_forward_shape():
    """Test that EPD ensemble forward pass returns (B, T, ..., M)."""
    n_members = 3
    batch_size = 2
    t_steps = 2
    w, h = 8, 8
    channels = 4

    # Batch setup
    input_fields = torch.randn(batch_size, t_steps, w, h, channels)
    output_fields = torch.randn(batch_size, t_steps, w, h, channels)
    batch = Batch(input_fields, output_fields, None, None)

    # PermuteConcat outputs (B, C*T, W, H)
    encoder = PermuteConcat(with_constants=False)
    encoder_output_channels = channels * t_steps

    decoder = ChannelsLast(output_channels=channels, time_steps=t_steps)

    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder)

    # Processor output must match decoder input expectations: (B, C*T, W, H)
    processor = MockProcessor(output_shape=(encoder_output_channels, w, h))

    model = EncoderProcessorDecoderEnsemble(
        encoder_decoder=encoder_decoder, processor=processor, n_members=n_members
    )

    output = model(batch)

    # Expected: (B, T, W, H, C, M)
    # ChannelsLast.decode returns (B, T, W, H, C)
    # Ensemble wrapper adds M at end -> (B, T, W, H, C, M)
    expected_shape = (batch_size, t_steps, w, h, channels, n_members)
    assert output.shape == expected_shape


def test_epd_ensemble_loss_latent_integration():
    """Test EPD ensemble loss when training in latent space (train_in_latent_space=True)."""
    n_members = 3
    batch_size = 2
    t_steps = 2
    w, h = 8, 8
    channels = 2

    input_fields = torch.randn(batch_size, t_steps, w, h, channels)
    output_fields = torch.randn(batch_size, t_steps, w, h, channels)
    batch = Batch(input_fields, output_fields, None, None)

    encoder = PermuteConcat(with_constants=False)
    encoder_output_channels = channels * t_steps

    decoder = ChannelsLast(output_channels=channels, time_steps=t_steps)

    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder)

    # Latent shape: (B, C*T, W, H)
    processor = MockProcessor(output_shape=(encoder_output_channels, w, h))

    def crps_loss(preds, targets):
        # preds: (B, C*T, W, H, M) - latent predictions
        # targets: (B, C*T, W, H) - encoded targets

        # Verify shapes match (except M)
        assert preds.shape[:-1] == targets.shape

        score = _common_crps_score(preds, targets, adjustment_factor=1.0)
        return score.mean()

    model = EncoderProcessorDecoderEnsemble(
        encoder_decoder=encoder_decoder,
        processor=processor,
        n_members=n_members,
        train_in_latent_space=True,
        loss_func=crps_loss,
    )

    loss, _ = model.loss(batch)

    # Calculate expected
    # Processor returns zeros + param (0.0) -> (B, C*T, W, H)
    # Ensemble expands to (B, C*T, W, H, M)
    preds = torch.zeros(batch_size, encoder_output_channels, w, h, n_members)

    # Encoded output fields (targets)
    encoded_batch_tmp = encoder.encode_batch(batch)
    encoded_targets = encoded_batch_tmp.encoded_output_fields  # (B, C*T, W, H)

    expected_crps_map = _common_crps_score(
        preds, encoded_targets, adjustment_factor=1.0
    )
    expected_loss = expected_crps_map.mean().item()

    assert loss.item() == pytest.approx(expected_loss)


def test_epd_ensemble_loss_fallback():
    """Test fallback when n_members=1 or train_in_latent_space=False."""
    # This triggers the fallback "super().loss(batch)" path.
    # But since EPDEnsemble.forward adds an extra dimension (M),
    # the loss function must handle it.
    n_members = 1
    batch_size = 2
    t_steps = 2
    w, h = 8, 8
    channels = 2

    input_fields = torch.randn(batch_size, t_steps, w, h, channels)
    output_fields = torch.randn(batch_size, t_steps, w, h, channels)
    batch = Batch(input_fields, output_fields, None, None)

    encoder = PermuteConcat(with_constants=False)
    decoder = ChannelsLast(output_channels=channels, time_steps=t_steps)
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder)

    encoder_output_channels = channels * t_steps
    processor = MockProcessor(output_shape=(encoder_output_channels, w, h))

    def tolerant_loss(preds, targets):
        # preds: (..., 1)
        # targets: (...)
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

    # Not checking value exactness, just that it runs and returns scalar
    assert loss.ndim == 0
