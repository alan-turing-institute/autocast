import itertools

import pytest
import torch

from auto_cast.nn.unet import TemporalUNetBackbone

params = list(
    itertools.product(
        [1, 4],  # n_steps_output
        [1, 4],  # n_steps_input
        [1, 2],  # n_channels_in
        [1, 4],  # n_channels_out
    )
)


@pytest.mark.parametrize(
    ("n_steps_output", "n_steps_input", "n_channels_in", "n_channels_out"), params
)
def test_unet(n_steps_output, n_steps_input, n_channels_in, n_channels_out):
    unet = TemporalUNetBackbone(
        in_channels=n_channels_out * n_steps_output,
        out_channels=n_channels_out * n_steps_output,
        cond_channels=n_channels_in * n_steps_input,
        mod_features=256,
        hid_channels=(32, 64, 128),
        hid_blocks=(2, 2, 2),
        spatial=2,
        periodic=False,
    )
    x_t = torch.randn(
        1, n_steps_output, 16, 16, n_channels_out
    )  # (B, T_out, W, H, C_out)
    cond = torch.randn(1, n_steps_input, 16, 16, n_channels_in)  # (B, T_in, W, H, C_in)
    output = unet.forward(x_t, torch.ones(x_t.shape[0]), cond)
    assert output.shape == (
        1,
        n_steps_output,
        16,
        16,
        n_channels_out,
    )  # (B, T_out, W, H, C_out)
