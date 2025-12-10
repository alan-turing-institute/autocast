from conftest import (
    assert_gradients_flow,
    assert_module_initialized,
    assert_output_valid,
)
from torch import rand

from autocast.encoders.dc import DCEncoder


def test_dcencoder_basic_2d():
    """Test DCEncoder basic functionality with 2D data."""
    encoder = DCEncoder(
        in_channels=3,
        out_channels=16,
        hid_channels=(32, 64, 128),
        spatial=2,
        hid_blocks=(2, 2, 2),
        pixel_shuffle=False,
    )

    x = rand(2, 3, 64, 64)
    z = encoder.forward(x)
    assert_output_valid(z, (2, 16, 16, 16))


def test_dcencoder_3d():
    """Test DCEncoder with 3D data."""
    encoder = DCEncoder(
        in_channels=4,
        out_channels=8,
        hid_channels=(16, 32),
        spatial=3,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    x = rand(2, 4, 32, 32, 32)
    z = encoder.forward(x)
    assert_output_valid(z, (2, 8, 16, 16, 16))


def test_dcencoder_pixel_shuffle():
    """Test DCEncoder with pixel_shuffle=True."""
    encoder = DCEncoder(
        in_channels=3,
        out_channels=16,
        hid_channels=(32, 64),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=True,
    )

    x = rand(2, 3, 64, 64)
    z = encoder.forward(x)
    assert_output_valid(z, (2, 16, 32, 32))


def test_dcencoder_with_attention():
    """Test DCEncoder with self-attention."""
    encoder = DCEncoder(
        in_channels=3,
        out_channels=16,
        hid_channels=(32, 64),
        spatial=2,
        hid_blocks=(2, 2),
        attention_heads={1: 4},  # Attention only in second level
        pixel_shuffle=False,
    )

    x = rand(2, 3, 64, 64)
    z = encoder.forward(x)
    assert_output_valid(z, (2, 16, 32, 32))


def test_dcencoder_single_depth():
    """Test DCEncoder with single depth level."""
    encoder = DCEncoder(
        in_channels=3,
        out_channels=16,
        hid_channels=(32,),
        spatial=2,
        hid_blocks=(3,),
        pixel_shuffle=False,
    )

    x = rand(2, 3, 32, 32)
    z = encoder.forward(x)
    assert_output_valid(z, (2, 16, 32, 32))


def test_dcencoder_initialization():
    """Test DCEncoder parameter initialization."""
    encoder = DCEncoder(
        in_channels=3,
        out_channels=16,
        hid_channels=(32, 64),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    assert_module_initialized(encoder, "Encoder")


def test_dcencoder_gradient_flow():
    """Test that gradients flow through DCEncoder."""
    encoder = DCEncoder(
        in_channels=3,
        out_channels=16,
        hid_channels=(32, 64),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    x = rand(2, 3, 64, 64, requires_grad=True)
    assert_gradients_flow(encoder, x, "Encoder")
