"""Tests for encoder constants concatenation feature."""

import torch
from conftest import _make_batch

from autocast.encoders.identity import IdentityEncoder


def test_identity_encoder_with_constant_fields():
    """Test that constant fields are concatenated to encoded inputs."""
    batch = _make_batch(batch_size=2, t=3, w=4, h=5, c=2, const_c=3, scalar_c=1)
    encoder = IdentityEncoder(with_constant_fields=True)

    encoded_batch = encoder.encode_batch(batch)

    # Encoded inputs should have additional channels from constant_fields
    expected_input_channels = (
        batch.input_fields.shape[-1] + batch.constant_fields.shape[-1]  # type: ignore  # noqa: PGH003
    )
    assert encoded_batch.encoded_inputs.shape[-1] == expected_input_channels
    assert encoded_batch.encoded_inputs.shape[:-1] == batch.input_fields.shape[:-1]

    # Encoded outputs should NOT have additional channels (constants only on inputs)
    assert encoded_batch.encoded_output_fields.shape == batch.output_fields.shape


def test_identity_encoder_with_constant_scalars():
    """Test that constant scalars are concatenated to encoded inputs."""
    batch = _make_batch(batch_size=2, t=3, w=4, h=5, c=2, const_c=1, scalar_c=4)
    encoder = IdentityEncoder(with_constant_scalars=True)

    encoded_batch = encoder.encode_batch(batch)

    # Encoded inputs should have additional channels from constant_scalars
    expected_input_channels = (
        batch.input_fields.shape[-1] + batch.constant_scalars.shape[-1]  # type: ignore  # noqa: PGH003
    )
    assert encoded_batch.encoded_inputs.shape[-1] == expected_input_channels
    assert encoded_batch.encoded_inputs.shape[:-1] == batch.input_fields.shape[:-1]

    # Encoded outputs should NOT have additional channels
    assert encoded_batch.encoded_output_fields.shape == batch.output_fields.shape


def test_identity_encoder_with_both_constants():
    """Test that both constant fields and scalars are concatenated."""
    batch = _make_batch(batch_size=2, t=3, w=4, h=5, c=2, const_c=3, scalar_c=4)
    encoder = IdentityEncoder(with_constant_fields=True, with_constant_scalars=True)

    encoded_batch = encoder.encode_batch(batch)

    # Encoded inputs should have additional channels from both
    expected_input_channels = (
        batch.input_fields.shape[-1]
        + batch.constant_fields.shape[-1]  # type: ignore  # noqa: PGH003
        + batch.constant_scalars.shape[-1]  # type: ignore  # noqa: PGH003
    )
    assert encoded_batch.encoded_inputs.shape[-1] == expected_input_channels
    assert encoded_batch.encoded_inputs.shape[:-1] == batch.input_fields.shape[:-1]


def test_constant_fields_values_are_correct():
    """Test that the concatenated constant field values are correct."""
    batch = _make_batch(batch_size=1, t=2, w=3, h=4, c=2, const_c=1, scalar_c=1)
    encoder = IdentityEncoder(with_constant_fields=True)

    encoded_batch = encoder.encode_batch(batch)

    # The last channel(s) should be the constant field values (expanded across time)
    # constant_fields is all ones, so the last channel should be all ones
    constant_part = encoded_batch.encoded_inputs[..., -1]
    assert torch.allclose(constant_part, torch.ones_like(constant_part))


def test_constant_scalars_values_are_correct():
    """Test that the concatenated constant scalar values are correct."""
    batch = _make_batch(batch_size=1, t=2, w=3, h=4, c=2, const_c=1, scalar_c=1)
    encoder = IdentityEncoder(with_constant_scalars=True)

    encoded_batch = encoder.encode_batch(batch)

    # The last channel(s) should be the constant scalar values (expanded to all dims)
    # constant_scalars is all 5.0, so the last channel should be all 5.0
    scalar_part = encoded_batch.encoded_inputs[..., -1]
    assert torch.allclose(scalar_part, torch.full_like(scalar_part, 5.0))


def test_encoder_without_constants():
    """Test that encoder without constants option works as before."""
    batch = _make_batch()
    encoder = IdentityEncoder()

    encoded_batch = encoder.encode_batch(batch)

    # Should be identical to input (no concatenation)
    assert torch.allclose(encoded_batch.encoded_inputs, batch.input_fields)
    assert torch.allclose(encoded_batch.encoded_output_fields, batch.output_fields)


def test_encoder_with_none_constants():
    """Test encoder with constants option but None values in batch."""
    batch_size, t, w, h, c = 2, 3, 4, 5, 2
    batch = _make_batch(batch_size=batch_size, t=t, w=w, h=h, c=c)
    # Set constants to None
    batch = batch.__class__(
        input_fields=batch.input_fields,
        output_fields=batch.output_fields,
        constant_scalars=None,
        constant_fields=None,
    )
    encoder = IdentityEncoder(with_constant_fields=True, with_constant_scalars=True)

    encoded_batch = encoder.encode_batch(batch)

    # Should be identical to input (no concatenation since values are None)
    assert encoded_batch.encoded_inputs.shape == batch.input_fields.shape
