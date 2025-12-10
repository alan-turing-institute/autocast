import torch
from conftest import _make_batch

from auto_cast.encoders.identity import IdentityEncoder


def test_identity():
    batch = _make_batch()
    encoder = IdentityEncoder()
    encoded_batch = encoder.encode_batch(batch)
    assert torch.allclose(encoded_batch.encoded_inputs, batch.input_fields)
    assert torch.allclose(encoded_batch.encoded_output_fields, batch.output_fields)
