import torch
from conftest import _make_batch

from autocast.decoders.identity import IdentityDecoder
from autocast.encoders.identity import IdentityEncoder
from autocast.types import EncodedBatch, EncodedSphericalBatch, GridType, SphericalBatch


def test_identity():
    batch = _make_batch()
    in_channels = batch.input_fields.shape[-1]
    encoder = IdentityEncoder(in_channels=in_channels)
    encoded_batch = encoder.encode_batch(batch)
    assert torch.allclose(encoded_batch.encoded_inputs, batch.input_fields)
    assert torch.allclose(encoded_batch.encoded_output_fields, batch.output_fields)


def test_identity_on_a_plain_batch_gives_a_plain_encoded_batch():
    encoded = IdentityEncoder(in_channels=2).encode_batch(_make_batch())

    assert type(encoded) is EncodedBatch


def test_identity_keeps_the_grid_of_a_spherical_batch():
    batch = SphericalBatch(
        input_fields=torch.randn(2, 1, 4, 8, 3),
        output_fields=torch.randn(2, 2, 4, 8, 3),
        constant_scalars=torch.randn(2, 2),
        constant_fields=torch.randn(2, 4, 8, 2),
        grid_type=GridType.LEGENDRE_GAUSS,
    )
    encoded = IdentityEncoder(in_channels=3).encode_batch(batch)
    decoder = IdentityDecoder(in_channels=3)

    assert type(encoded) is EncodedSphericalBatch
    assert encoded.grid_type is GridType.LEGENDRE_GAUSS
    assert encoded.global_cond is batch.constant_scalars
    assert torch.equal(decoder.decode(encoded.encoded_inputs), batch.input_fields)
    assert torch.equal(
        decoder.decode(encoded.encoded_output_fields), batch.output_fields
    )
