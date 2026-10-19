"""Check the residual contract through ambient and non-identity latent codecs."""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from einops import repeat
from lightning.pytorch.plugins.precision import Precision
from torch import nn

from autocast.callbacks.residual_statistics import ResidualStatisticsCallback
from autocast.decoders.base import Decoder
from autocast.decoders.identity import IdentityDecoder
from autocast.encoders.base import EncoderWithCond
from autocast.encoders.identity import IdentityEncoder
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.models.processor import ProcessorModel
from autocast.nn.noise.source import ZeroSource
from autocast.processors.residual_flow_matching import ResidualFlowMatchingProcessor
from autocast.processors.residual_normalization import ResidualStandardizer
from autocast.processors.residual_reference import LastFrameReference


class _NonlinearEncoder(EncoderWithCond):
    latent_channels = 1
    channel_axis = -1

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.dropout = nn.Dropout(dropout)

    def encode(self, batch):
        # Deliberately lossy: latent persistence need not be physical persistence.
        fields = self.dropout(batch.input_fields[:, :, ::2, ::2])
        return torch.asinh(self.scale * fields)


class _NonlinearDecoder(Decoder):
    latent_channels = 1

    def decode(self, z):
        return repeat(torch.sinh(z) / 2, "b t h w c -> b t (h r) (w s) c", r=2, s=2)


class _ConstantField(nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity = nn.Parameter(torch.tensor(0.5))

    def forward(self, z, t, cond, global_cond=None):  # noqa: ARG002
        return torch.ones_like(z) * self.velocity


def _model(
    coordinates,
    *,
    dropout=0.0,
    model_type: type[EncoderProcessorDecoder] = EncoderProcessorDecoder,
):
    codec = EncoderDecoder(
        encoder=IdentityEncoder(in_channels=1)
        if coordinates == "ambient"
        else _NonlinearEncoder(dropout),
        decoder=IdentityDecoder(in_channels=1)
        if coordinates == "ambient"
        else _NonlinearDecoder(),
    )
    processor = ResidualFlowMatchingProcessor(
        backbone=_ConstantField(),
        reference=LastFrameReference(n_steps_output=2),
        source=ZeroSource(),
        standardizer=ResidualStandardizer(
            n_steps_output=2, n_channels=1, mean=0.25, scale=2.0
        ),
        n_steps_output=2,
    )
    return model_type(
        encoder_decoder=codec, processor=processor, train_in_latent_space=True
    )


@pytest.mark.parametrize("coordinates", ["ambient", "latent"])
def test_targets_and_forecasts_use_the_codec_coordinates(coordinates, toy_batch):
    model = _model(coordinates)
    codec = model.encoder_decoder
    processor = cast(ResidualFlowMatchingProcessor, model.processor)
    encoded = codec.encoder.encode_batch(toy_batch)
    reference = repeat(encoded.encoded_inputs[:, -1:], "b 1 ... c -> b t ... c", t=2)

    torch.testing.assert_close(
        processor.target_residual(encoded), encoded.encoded_output_fields - reference
    )
    # Zero source and constant velocity integrate to 0.5, then unstandardize.
    reconstructed = reference + 0.25 + 2.0 * 0.5
    torch.testing.assert_close(
        processor.map(encoded.encoded_inputs, encoded.global_cond), reconstructed
    )
    torch.testing.assert_close(model(toy_batch), codec.decoder.decode(reconstructed))

    if coordinates == "latent":
        assert encoded.encoded_inputs.shape[2:4] != toy_batch.input_fields.shape[2:4]
        ambient_residual_batch = replace(
            toy_batch,
            input_fields=toy_batch.output_fields - toy_batch.input_fields[:, -1:],
        )
        assert not torch.allclose(
            processor.target_residual(encoded),
            codec.encoder.encode(ambient_residual_batch),
        )
        wrongly_added_after_decoding = codec.decoder.decode(reference) + (
            codec.decoder.decode(torch.full_like(reference, 1.25))
        )
        assert not torch.allclose(model(toy_batch), wrongly_added_after_decoding)


@pytest.mark.parametrize("coordinates", ["ambient", "latent"])
@pytest.mark.parametrize(
    "model_type", [EncoderProcessorDecoder, EncoderProcessorDecoderEnsemble]
)
def test_online_and_cached_training_use_the_same_flow_objective(
    coordinates, model_type, toy_batch
):
    model = _model(coordinates, model_type=model_type)
    encoded = model.encoder_decoder.encoder.encode_batch(toy_batch)
    cached_model = ProcessorModel(processor=model.processor)

    torch.manual_seed(17)
    online_loss, prediction = model.loss(toy_batch)
    torch.manual_seed(17)
    cached_loss = cached_model.loss(encoded)

    assert prediction is None
    torch.testing.assert_close(online_loss, cached_loss)
    online_loss.backward()
    processor = cast(ResidualFlowMatchingProcessor, model.processor)
    assert cast(_ConstantField, processor.flow_matching_model).velocity.grad is not None
    assert all(
        parameter.grad is None for parameter in model.encoder_decoder.parameters()
    )


def test_zero_latent_residual_returns_codec_reconstruction(toy_batch):
    model = _model("latent")
    processor = cast(ResidualFlowMatchingProcessor, model.processor)
    processor.standardizer = None
    with torch.no_grad():
        cast(_ConstantField, processor.flow_matching_model).velocity.zero_()
    encoded = model.encoder_decoder.encoder.encode_batch(toy_batch)
    reference = processor.reference_trajectory(
        encoded.encoded_inputs, encoded.global_cond
    )

    forecast = model(toy_batch)

    torch.testing.assert_close(
        forecast, model.encoder_decoder.decoder.decode(reference)
    )
    last_ambient_frame = repeat(
        toy_batch.input_fields[:, -1:], "b 1 ... c -> b t ... c", t=2
    )
    assert not torch.allclose(forecast, last_ambient_frame)


def test_frozen_latent_codec_stays_fixed_during_training(toy_batch):
    model = _model("latent", dropout=0.5)
    model.train()

    assert model.processor.training
    assert not model.encoder_decoder.training
    encoder = cast(_NonlinearEncoder, model.encoder_decoder.encoder)
    assert not encoder.dropout.training
    first = model.encoder_decoder.encoder.encode_batch(toy_batch)
    second = model.encoder_decoder.encoder.encode_batch(toy_batch)
    torch.testing.assert_close(first.encoded_inputs, second.encoded_inputs)
    torch.testing.assert_close(
        first.encoded_output_fields, second.encoded_output_fields
    )


@pytest.mark.parametrize("coordinates", ["ambient", "latent"])
def test_online_and_cached_statistics_match(coordinates, toy_batch):
    online_model = _model(coordinates)
    online_model.train()
    processor = cast(ResidualFlowMatchingProcessor, online_model.processor)
    processor.standardizer = ResidualStandardizer(n_steps_output=2, n_channels=1)
    encoded = online_model.encoder_decoder.encoder.encode_batch(toy_batch)
    cached_processor = cast(
        ResidualFlowMatchingProcessor, _model(coordinates).processor
    )
    cached_processor.standardizer = ResidualStandardizer(n_steps_output=2, n_channels=1)
    cached_model = ProcessorModel(processor=cached_processor)

    def broadcast(value, src):  # noqa: ARG001
        return value

    for model, batch in [(online_model, toy_batch), (cached_model, encoded)]:
        trainer = SimpleNamespace(
            is_global_zero=True,
            precision_plugin=Precision(),
            strategy=SimpleNamespace(broadcast=broadcast),
            datamodule=SimpleNamespace(train_dataloader=lambda batch=batch: [batch]),
        )
        ResidualStatisticsCallback().on_fit_start(trainer, model)  # type: ignore[arg-type]

    assert processor.standardizer.fitted
    assert cached_processor.standardizer.fitted
    torch.testing.assert_close(
        processor.standardizer.mean, cached_processor.standardizer.mean
    )
    torch.testing.assert_close(
        processor.standardizer.scale, cached_processor.standardizer.scale
    )
