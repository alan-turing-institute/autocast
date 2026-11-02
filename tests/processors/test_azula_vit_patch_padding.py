import pytest
import torch
from einops import rearrange

from autocast.processors.azula_vit import AzulaViTProcessor


def make_processor(**kwargs):
    return AzulaViTProcessor(
        **(
            {
                "in_channels": 3,
                "out_channels": 2,
                "spatial_resolution": (9, 13),
                "hidden_dim": 16,
                "num_heads": 2,
                "n_layers": 2,
                "patch_size": 4,
                "temporal_method": "none",
                "n_noise_channels": 8,
            }
            | kwargs
        )
    )


@pytest.mark.parametrize("patch_size", [4, 16, (4, 8)])
@pytest.mark.parametrize("with_time", [False, True])
def test_partial_patches_preserve_grid_and_edge_gradients(patch_size, with_time):
    processor = make_processor(
        patch_size=patch_size,
        n_steps_input=2 if with_time else 1,
        n_steps_output=3 if with_time else 1,
    )
    input_shape = (1, 2, 9, 13, 3) if with_time else (1, 3, 9, 13)
    inputs = torch.randn(input_shape, requires_grad=True)

    outputs = processor.map(inputs)

    expected_shape = (1, 3, 9, 13, 2) if with_time else (1, 2, 9, 13)
    assert outputs.shape == expected_shape
    outputs.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    edge = inputs.grad[:, :, -1, -1, :] if with_time else inputs.grad[:, :, -1, -1]
    assert edge.abs().sum() > 0


def test_divisible_grid_matches_backbone_without_padding():
    processor = make_processor()
    inputs = torch.randn(1, 3, 8, 12)
    noise = torch.randn(1, 8)
    state_keys = tuple(processor.state_dict())
    expected = processor.model(
        rearrange(inputs, "b c h w -> b 1 h w c"), noise, cond=None
    )

    outputs = processor(inputs, noise)

    torch.testing.assert_close(
        outputs, rearrange(expected, "b 1 h w c -> b c h w"), rtol=0, atol=0
    )
    assert tuple(processor.state_dict()) == state_keys
