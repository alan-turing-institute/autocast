import torch
from conftest import _make_batch

from autocast.encoders.identity import IdentityEncoder


def test_identity():
    batch = _make_batch()
    in_channels = batch.input_fields.shape[-1]
    encoder = IdentityEncoder(in_channels=in_channels)
    encoded_batch = encoder.encode_batch(batch)
    assert torch.allclose(encoded_batch.encoded_inputs, batch.input_fields)
    assert torch.allclose(encoded_batch.encoded_output_fields, batch.output_fields)
