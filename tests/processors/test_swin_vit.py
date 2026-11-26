import pytest
import torch

from autocast.processors.swin_tv_vit import SwinTVProcessor
from autocast.processors.swin_vit import SwinViTBlock


def test_swin_processor_forward_and_map_shape():
    processor = SwinTVProcessor(
        in_channels=3,
        out_channels=5,
        spatial_resolution=(64, 64),
    )

    x = torch.randn(2, 3, 64, 64)
    y = processor.forward(x)
    y_map = processor.map(x, global_cond=None)

    assert y.shape == (2, 5, 64, 64)
    assert y_map.shape == (2, 5, 64, 64)


def test_swin_processor_noise_conditioning_changes_outputs():
    processor = SwinTVProcessor(
        in_channels=3,
        out_channels=3,
        spatial_resolution=(64, 64),
        n_noise_channels=16,
        zero_init=False,
    )

    x = torch.randn(2, 3, 64, 64)
    y_1 = processor.map(x, global_cond=None)
    y_2 = processor.map(x, global_cond=None)

    assert y_1.shape == y_2.shape == (2, 3, 64, 64)
    assert not torch.allclose(y_1, y_2)


def test_periodic_shifted_windows_mix_across_both_seams():
    def build(periodic: bool = False) -> SwinViTBlock:
        return SwinViTBlock(
            hidden_dim=4,
            num_heads=1,
            n_spatial_dims=2,
            n_noise_channels=None,
            window_size=(4, 4),
            shift=True,
            use_ada_ln=False,
            periodic=periodic,
        )

    periodic = build(periodic=True)
    bounded = build()
    with torch.no_grad():
        # Uniform attention lets a single impulse reveal its entire window.
        periodic.qkv_proj.weight.zero_()
        periodic.qkv_proj.weight[8:].copy_(torch.eye(4))
        periodic.qkv_proj.bias.zero_()
        periodic.output_proj.weight.copy_(torch.eye(4))
        periodic.output_proj.bias.zero_()
        for parameter in periodic.mlp.parameters():
            parameter.zero_()
        bounded.load_state_dict(periodic.state_dict(), strict=True)

        field = torch.zeros(1, 8, 8, 4)
        field[0, 0, 0] = torch.tensor([1.0, -1.0, 0.0, 0.0])
        wrapped = periodic(field)
        masked = bounded(field)
        expected = periodic.norm1(field[0, 0, 0]) / 16
        for row, column in ((0, 7), (7, 0), (7, 7)):
            torch.testing.assert_close(wrapped[0, row, column], expected)
            torch.testing.assert_close(masked[0, row, column], torch.zeros(4))
        torch.testing.assert_close(wrapped[0, 4, 4], torch.zeros(4))


def test_periodic_swin_rejects_windows_that_need_padding():
    block = SwinViTBlock(
        hidden_dim=4,
        num_heads=1,
        n_spatial_dims=2,
        n_noise_channels=None,
        window_size=(4, 4),
        shift=True,
        periodic=True,
    )
    with pytest.raises(ValueError, match="windows must tile"):
        block(torch.zeros(1, 7, 8, 4))
