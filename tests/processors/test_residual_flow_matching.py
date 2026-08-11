import pytest
import torch
from einops import repeat
from torch import nn

from autocast.nn.noise.source import FlowSource, IIDGaussianSource, ZeroSource
from autocast.nn.vit import TemporalViTBackbone
from autocast.processors.residual_flow_matching import ResidualFlowMatchingProcessor
from autocast.processors.residual_reference import LastFrameReference
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
) -> ResidualFlowMatchingProcessor:
    return ResidualFlowMatchingProcessor(
        backbone=backbone if backbone is not None else _ZeroField(),
        reference=LastFrameReference(n_steps_output=n_steps_output),
        source=source,
        n_steps_output=n_steps_output,
        n_channels_out=1,
    )


def test_map_returns_reference_for_zero_source_and_field():
    batch = _batch()
    processor = _processor(source=ZeroSource())

    prediction = processor.map(batch.encoded_inputs, batch.global_cond)

    expected = repeat(
        batch.encoded_inputs[:, -1:],
        "b 1 y x c -> b t y x c",
        t=4,
    )
    assert torch.equal(prediction, expected)


def test_map_adds_configured_source_to_reference():
    batch = _batch()
    processor = _processor(source=IIDGaussianSource())
    torch.manual_seed(7)

    prediction = processor.map(batch.encoded_inputs, batch.global_cond)

    torch.manual_seed(7)
    source = torch.randn_like(batch.encoded_output_fields)
    expected_reference = repeat(
        batch.encoded_inputs[:, -1:],
        "b 1 y x c -> b t y x c",
        t=4,
    )
    assert torch.equal(prediction, expected_reference + source)


def test_loss_targets_residual_instead_of_full_state():
    batch = _batch(residual_scale=0.0)
    processor = _processor(source=ZeroSource())

    loss = processor.loss(batch)

    assert loss.item() == 0.0


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


def test_temporal_vit_residual_flow_shapes_and_gradients():
    batch = _batch()
    backbone = TemporalViTBackbone(
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
        spatial=2,
        dropout=0.0,
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
