import torch
from conftest import _make_batch
from einops import rearrange

from autocast.encoders.permute_concat import PermuteConcat


def test_permute_concat_with_constants():
    encoder = PermuteConcat(with_constants=True)
    batch = _make_batch()

    encoded = encoder(batch)

    # After encoder: (B, T, W, H, C) -> (B, C, T, W, H) -> (B, C*T, W, H)
    base_channels = batch.input_fields.shape[-1]
    time_steps = batch.input_fields.shape[1]
    assert batch.constant_fields is not None
    assert batch.constant_scalars is not None
    const_channels = batch.constant_fields.shape[-1]
    scalar_channels = batch.constant_scalars.shape[-1]

    expected_channels = (
        base_channels * time_steps + const_channels + scalar_channels * time_steps
    )

    assert encoded.shape == (
        batch.input_fields.shape[0],
        expected_channels,
        batch.input_fields.shape[2],
        batch.input_fields.shape[3],
    )

    expected_base = rearrange(batch.input_fields, "b t w h c -> b (c t) w h")
    assert torch.allclose(encoded[:, : base_channels * time_steps, ...], expected_base)

    # Verify constant fields (no time dimension, just added as single channel)
    const_slice = encoded[
        :, base_channels * time_steps : base_channels * time_steps + const_channels, ...
    ]
    expected_const = rearrange(batch.constant_fields, "b w h c -> b c w h")
    assert torch.allclose(const_slice, expected_const)

    # Verify constant scalars (expanded across time and merged)
    scalar_slice = encoded[:, -scalar_channels * time_steps :, ...]
    # Each scalar value is expanded to:
    # (scalar_c, T, W, H)
    # then merged to (scalar_c*T, W, H)
    expected_scalars = batch.constant_scalars.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    expected_scalars = expected_scalars.expand(
        -1, -1, time_steps, batch.input_fields.shape[2], batch.input_fields.shape[3]
    )
    expected_scalars = rearrange(expected_scalars, "b c t w h -> b (c t) w h")
    assert torch.allclose(scalar_slice, expected_scalars)
