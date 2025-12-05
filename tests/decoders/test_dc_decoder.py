from conftest import (
    assert_gradients_flow,
    assert_module_initialized,
    assert_output_valid,
)
from torch import rand

from auto_cast.decoders.dc import DCDecoder


def test_dcdecoder_basic_2d():
    """Test DCDecoder basic functionality with 2D data."""
    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(128, 64, 32),
        spatial=2,
        hid_blocks=(2, 2, 2),
        pixel_shuffle=False,
    )

    z = rand(2, 16, 16, 16)
    x = decoder.forward(z)
    assert_output_valid(x, (2, 64, 64, 3))


def test_dcdecoder_3d():
    """Test DCDecoder with 3D data."""
    decoder = DCDecoder(
        in_channels=8,
        out_channels=4,
        hid_channels=(32, 16),
        spatial=3,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    z = rand(2, 8, 16, 16, 16)
    x = decoder.forward(z)
    assert_output_valid(x, (2, 32, 32, 32, 4))


def test_dcdecoder_pixel_shuffle():
    """Test DCDecoder with pixel_shuffle=True."""
    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(64, 32),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=True,
    )

    # Test forward pass
    z = rand(2, 16, 16, 16)
    x = decoder.forward(z)

    assert x.shape == (2, 32, 32, 3), f"Expected (2, 32, 32, 3), got {x.shape}"
    assert not x.isnan().any(), "Output contains NaN values"


def test_dcdecoder_with_attention():
    """Test DCDecoder with self-attention."""
    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(64, 32),
        spatial=2,
        hid_blocks=(2, 2),
        attention_heads={0: 4},  # Attention only in first level
        pixel_shuffle=False,
    )

    z = rand(2, 16, 16, 16)
    x = decoder.forward(z)
    assert_output_valid(x, (2, 32, 32, 3))


def test_dcdecoder_single_depth():
    """Test DCDecoder with single depth level."""
    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(32,),
        spatial=2,
        hid_blocks=(3,),
        pixel_shuffle=False,
    )

    z = rand(2, 16, 16, 16)
    x = decoder.forward(z)
    assert_output_valid(x, (2, 16, 16, 3))


def test_dcdecoder_initialization():
    """Test DCDecoder parameter initialization."""
    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(64, 32),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    assert_module_initialized(decoder, "Decoder")


def test_dcdecoder_gradient_flow():
    """Test that gradients flow through DCDecoder."""
    decoder = DCDecoder(
        in_channels=16,
        out_channels=3,
        hid_channels=(64, 32),
        spatial=2,
        hid_blocks=(2, 2),
        pixel_shuffle=False,
    )

    z = rand(2, 16, 16, 16, requires_grad=True)
    assert_gradients_flow(decoder, z, "Decoder")
