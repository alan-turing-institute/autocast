from typing import Any

import pytest
import torch
from einops import repeat
from torch import nn

from autocast.decoders.identity import IdentityDecoder
from autocast.encoders.identity import IdentityEncoder
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.nn.noise.source import FlowSource, IIDGaussianSource, ZeroSource
from autocast.nn.spatiotemporal_vit import SpatiotemporalViTBackbone
from autocast.nn.vit import TemporalViTBackbone
from autocast.processors.base import Processor
from autocast.processors.residual_flow_matching import ResidualFlowMatchingProcessor
from autocast.processors.residual_normalization import ResidualNormalizer
from autocast.processors.residual_reference import (
    LastFrameReference,
    ProcessorReference,
    ReferenceTrajectory,
)
from autocast.types import EncodedBatch


class _ZeroField(nn.Module):
    def forward(self, z, t, cond, global_cond=None):  # noqa: ARG002
        return torch.zeros_like(z)


class _ScaledField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, z, t, cond, global_cond=None):  # noqa: ARG002
        return self.scale * z


class _OneField(nn.Module):
    def forward(self, z, t, cond, global_cond=None):  # noqa: ARG002
        return torch.ones_like(z)


class _OffsetReferenceProcessor(Processor[EncodedBatch]):
    def map(self, x, global_cond):  # noqa: ARG002
        return repeat(x[:, -1:] + 3.0, "b 1 y x c -> b t y x c", t=4)

    def loss(self, batch):
        raise NotImplementedError


def _batch(
    *,
    batch_size: int = 2,
    n_steps_input: int = 1,
    n_steps_output: int = 4,
    spatial_size: int = 8,
    n_channels: int = 1,
    residual_scale: float = 0.1,
) -> EncodedBatch:
    inputs = torch.randn(
        batch_size,
        n_steps_input,
        spatial_size,
        spatial_size,
        n_channels,
    )
    reference = repeat(
        inputs[:, -1:],
        "b 1 y x c -> b t y x c",
        t=n_steps_output,
    )
    targets = reference + residual_scale * torch.randn_like(reference)
    return EncodedBatch(
        encoded_inputs=inputs,
        encoded_output_fields=targets,
        global_cond=torch.randn(batch_size, 2),
        encoded_info={},
    )


def _processor(
    *,
    backbone: nn.Module | None = None,
    source: FlowSource | None = None,
    n_steps_output: int = 4,
    residual_normalizer: ResidualNormalizer | None = None,
    reference: ReferenceTrajectory | None = None,
) -> ResidualFlowMatchingProcessor:
    return ResidualFlowMatchingProcessor(
        backbone=backbone if backbone is not None else _ZeroField(),
        reference=reference
        if reference is not None
        else LastFrameReference(n_steps_output=n_steps_output),
        source=source,
        n_steps_output=n_steps_output,
        n_channels_out=1,
        residual_normalizer=residual_normalizer,
    )


@pytest.mark.parametrize("random_source", [False, True])
def test_map_adds_configured_source_to_reference(random_source):
    batch = _batch()
    processor = _processor(
        source=IIDGaussianSource() if random_source else ZeroSource()
    )
    torch.manual_seed(7)
    prediction = processor.map(batch.encoded_inputs, batch.global_cond)

    torch.manual_seed(7)
    source = torch.randn_like(batch.encoded_output_fields) if random_source else 0.0
    reference = repeat(batch.encoded_inputs[:, -1:], "b 1 ... c -> b t ... c", t=4)
    assert torch.equal(prediction, reference + source)


@pytest.mark.parametrize(("reference_offset", "velocity"), [(0.0, 1.0), (3.0, 0.0)])
def test_map_denormalizes_residual_before_adding_reference(reference_offset, velocity):
    batch = _batch()
    processor = _processor(
        backbone=_OneField() if velocity else _ZeroField(),
        source=ZeroSource(),
        reference=ProcessorReference(_OffsetReferenceProcessor())
        if reference_offset
        else None,
        residual_normalizer=ResidualNormalizer(
            n_steps_output=4, n_channels=1, mean=0.25, scale=2.0
        ),
    )
    prediction = processor.map(batch.encoded_inputs, batch.global_cond)
    reference = repeat(
        batch.encoded_inputs[:, -1:] + reference_offset, "b 1 ... c -> b t ... c", t=4
    )
    assert torch.equal(prediction, reference + (0.25 + 2.0 * velocity))


@pytest.mark.parametrize("integrator", ["euler", "heun"])
def test_explicit_source_returns_the_same_residual_forecast_as_map(integrator):
    batch = _batch()
    processor = _processor(
        backbone=_OneField(),
        source=IIDGaussianSource(),
        residual_normalizer=ResidualNormalizer(
            n_steps_output=4, n_channels=1, mean=0.25, scale=2.0
        ),
    )
    processor.integrator = integrator
    torch.manual_seed(7)
    prediction = processor.map(batch.encoded_inputs, batch.global_cond)

    torch.manual_seed(7)
    source = torch.randn_like(batch.encoded_output_fields)
    explicit = processor.map_from_source(
        batch.encoded_inputs, batch.global_cond, source_states=source
    )

    reference = processor.reference_trajectory(batch.encoded_inputs, batch.global_cond)
    expected = reference + 0.25 + 2.0 * (source + 1.0)
    torch.testing.assert_close(explicit, expected)
    torch.testing.assert_close(explicit, prediction)


def test_processor_reference_defines_training_residual():
    batch = _batch(residual_scale=0.0)
    processor = ResidualFlowMatchingProcessor(
        backbone=_ZeroField(),
        reference=ProcessorReference(_OffsetReferenceProcessor()),
        source=ZeroSource(),
        n_steps_output=4,
        n_channels_out=1,
    )

    loss = processor.loss(batch)

    assert loss.item() == pytest.approx(9.0)


def test_loss_targets_residual_instead_of_full_state():
    batch = _batch(residual_scale=0.0)
    processor = _processor(source=ZeroSource())

    loss = processor.loss(batch)

    assert loss.item() == 0.0


def test_loss_targets_standardized_residual():
    batch = _batch(residual_scale=0.0)
    batch.encoded_output_fields = (
        repeat(
            batch.encoded_inputs[:, -1:],
            "b 1 y x c -> b t y x c",
            t=4,
        )
        + 5.0
    )
    residual_normalizer = ResidualNormalizer(
        n_steps_output=4,
        n_channels=1,
        mean=1.0,
        scale=2.0,
    )
    processor = _processor(source=ZeroSource(), residual_normalizer=residual_normalizer)

    loss = processor.loss(batch)

    assert loss.item() == pytest.approx(4.0)


def test_residual_flow_loss_has_finite_backbone_gradients():
    batch = _batch()
    backbone = _ScaledField()
    processor = _processor(backbone=backbone, source=IIDGaussianSource())

    loss = processor.loss(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert backbone.scale.grad is not None
    assert torch.isfinite(backbone.scale.grad)


def test_reference_shape_is_validated():
    batch = _batch()
    processor = ResidualFlowMatchingProcessor(
        backbone=_ZeroField(),
        reference=LastFrameReference(n_steps_output=3),
        source=ZeroSource(),
        n_steps_output=4,
        n_channels_out=1,
    )

    with pytest.raises(ValueError, match="Reference trajectory must have shape"):
        processor.map(batch.encoded_inputs, batch.global_cond)


def test_residual_normalizer_dimensions_are_validated():
    residual_normalizer = ResidualNormalizer(
        n_steps_output=3,
        n_channels=1,
        mean=0.0,
        scale=1.0,
    )

    with pytest.raises(ValueError, match="must match the processor"):
        _processor(residual_normalizer=residual_normalizer)


@pytest.mark.parametrize(
    "backbone_type", [TemporalViTBackbone, SpatiotemporalViTBackbone]
)
def test_vit_residual_flow_shapes_and_gradients(backbone_type):
    batch = _batch()
    spatial_kwargs: dict[str, Any] = (
        {"spatial": 2} if backbone_type is TemporalViTBackbone else {}
    )
    backbone = backbone_type(
        in_channels=1,
        out_channels=1,
        cond_channels=1,
        n_steps_output=4,
        n_steps_input=1,
        global_cond_channels=2,
        include_global_cond=True,
        mod_features=16,
        hid_channels=32,
        hid_blocks=1,
        attention_heads=4,
        patch_size=4,
        dropout=0.0,
        **spatial_kwargs,
    )
    processor = _processor(backbone=backbone, source=IIDGaussianSource())

    loss = processor.loss(batch)
    loss.backward()
    prediction = processor.map(batch.encoded_inputs, batch.global_cond)

    assert torch.isfinite(loss)
    assert prediction.shape == batch.encoded_output_fields.shape
    assert all(
        parameter.grad is not None
        for parameter in backbone.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize(
    "model_type", [EncoderProcessorDecoder, EncoderProcessorDecoderEnsemble]
)
@pytest.mark.parametrize("train_in_latent_space", [False, True])
def test_rejects_double_residual_reconstruction(model_type, train_in_latent_space):
    with pytest.raises(ValueError, match=r"model\.residual_prediction must be False"):
        model_type(
            encoder_decoder=EncoderDecoder(
                encoder=IdentityEncoder(in_channels=1),
                decoder=IdentityDecoder(in_channels=1),
            ),
            processor=_processor(),
            train_in_latent_space=train_in_latent_space,
            residual_prediction=True,
        )
