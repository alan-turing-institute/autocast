import pytest
import torch

from autocast.processors.swin_vit import SwinViTProcessor


@pytest.mark.parametrize("spatial_shape", [(9, 13), (9, 13, 5)])
@pytest.mark.parametrize("patch_size", [1, 4, None])
def test_partial_patches_preserve_grid_and_edge_gradients(spatial_shape, patch_size):
    processor = SwinViTProcessor(
        in_channels=3,
        out_channels=2,
        spatial_resolution=spatial_shape,
        window_size=(4, 4, 4),
        hidden_dim=24,
        encoder_depths=(1, 1),
        encoder_num_heads=(3, 6),
        decoder_depths=(1, 1),
        decoder_num_heads=(6, 3),
        patch_size=patch_size,
    )
    inputs = torch.randn(1, 3, *spatial_shape, requires_grad=True)

    outputs = processor.map(inputs)

    assert outputs.shape == (1, 2, *spatial_shape)
    outputs.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    edge = (slice(None), slice(None), *(-1 for _ in spatial_shape))
    assert inputs.grad[edge].abs().sum() > 0
