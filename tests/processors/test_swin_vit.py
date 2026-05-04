import torch

from autocast.processors.swin_tv_vit import SwinTVProcessor


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
