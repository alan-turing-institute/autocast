import itertools

import pytest
import torch

from autocast.nn.unet import TemporalUNetBackbone
from autocast.nn.vit import TemporalViTBackbone

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
        in_channels=n_channels_out,
        out_channels=n_channels_out,
        cond_channels=n_channels_in,
        n_steps_output=n_steps_output,
        n_steps_input=n_steps_input,
        include_global_cond=False,
        global_cond_channels=None,
        temporal_method="attention",
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
    output = unet.forward(x_t, torch.ones(x_t.shape[0]), cond, None)
    assert output.shape == (
        1,
        n_steps_output,
        16,
        16,
        n_channels_out,
    )  # (B, T_out, W, H, C_out)


def test_unet_include_time_embedding_false():
    n_steps_output, n_steps_input, n_channels_in, n_channels_out = 2, 2, 1, 1
    unet = TemporalUNetBackbone(
        in_channels=n_channels_out,
        out_channels=n_channels_out,
        cond_channels=n_channels_in,
        n_steps_output=n_steps_output,
        n_steps_input=n_steps_input,
        include_global_cond=False,
        global_cond_channels=None,
        temporal_method="none",
        mod_features=256,
        hid_channels=(32, 64, 128),
        hid_blocks=(2, 2, 2),
        spatial=2,
        periodic=False,
        include_time_embedding=False,
    )
    assert unet.time_embedding is None
    x_t = torch.randn(1, n_steps_output, 16, 16, n_channels_out)
    cond = torch.randn(1, n_steps_input, 16, 16, n_channels_in)
    output = unet.forward(x_t, None, cond, None)
    assert output.shape == (1, n_steps_output, 16, 16, n_channels_out)


def test_vit_include_time_embedding_false():
    n_steps_output, n_steps_input, n_channels_in, n_channels_out = 2, 2, 1, 1
    vit = TemporalViTBackbone(
        in_channels=n_channels_out,
        out_channels=n_channels_out,
        cond_channels=n_channels_in,
        n_steps_output=n_steps_output,
        n_steps_input=n_steps_input,
        include_global_cond=False,
        global_cond_channels=None,
        temporal_method="none",
        mod_features=64,
        hid_channels=64,
        hid_blocks=2,
        attention_heads=4,
        patch_size=4,
        spatial=2,
        include_time_embedding=False,
    )
    assert vit.time_embedding is None
    x_t = torch.randn(1, n_steps_output, 16, 16, n_channels_out)
    cond = torch.randn(1, n_steps_input, 16, 16, n_channels_in)
    output = vit.forward(x_t, None, cond, None)
    assert output.shape == (1, n_steps_output, 16, 16, n_channels_out)
