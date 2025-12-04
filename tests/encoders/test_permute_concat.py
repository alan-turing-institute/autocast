import torch

from auto_cast.encoders.permute_concat import PermuteConcat
from auto_cast.types import Batch


def _make_batch(
    batch_size: int = 1,
    t: int = 1,
    w: int = 2,
    h: int = 3,
    c: int = 2,
    const_c: int = 1,
    scalar_c: int = 1,
) -> Batch:
    input_fields = torch.arange(batch_size * t * w * h * c, dtype=torch.float32)
    input_fields = input_fields.view(batch_size, t, w, h, c)
    output_fields = torch.zeros(batch_size, t, w, h, c)
    constant_fields = torch.ones(batch_size, w, h, const_c)
    constant_scalars = torch.full((batch_size, scalar_c), 5.0)
    return Batch(
        input_fields=input_fields,
        output_fields=output_fields,
        constant_scalars=constant_scalars,
        constant_fields=constant_fields,
    )


def test_permute_concat_with_constants():
    encoder = PermuteConcat(with_constants=True)
    batch = _make_batch()

    encoded = encoder(batch)

    expected = batch.input_fields.permute(0, 4, 1, 2, 3)

    base_channels = batch.input_fields.shape[-1]
    assert batch.constant_fields is not None
    assert batch.constant_scalars is not None
    const_channels = batch.constant_fields.shape[-1]
    scalar_channels = batch.constant_scalars.shape[-1]

    assert encoded.shape == (
        batch.input_fields.shape[0],
        base_channels + const_channels + scalar_channels,
        batch.input_fields.shape[1],
        batch.input_fields.shape[2],
        batch.input_fields.shape[3],
    )

    assert torch.allclose(encoded[:, :base_channels, ...], expected)
    const_slice = encoded[:, base_channels : base_channels + const_channels, ...]
    assert torch.allclose(const_slice, torch.ones_like(const_slice))
    scalar_slice = encoded[:, -scalar_channels:, ...]
    assert torch.allclose(scalar_slice, torch.full_like(scalar_slice, 5.0))
