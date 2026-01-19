import torch

from autocast.nn.conditional_layer_norm import ConditionalLayerNorm


def test_conditional_layer_norm():
    # TODO: consider ensemble dim test
    sample_batch = torch.randn(2, 1, 64, 64, 3)
    norm_layer = ConditionalLayerNorm(
        normalized_shape=[64, 64, 3], n_noise_channels=64 * 64
    )
    x_noise = torch.randn(2, 1, 64, 64, 1)
    output1 = norm_layer.forward(sample_batch, x_noise)
    output2 = norm_layer.forward(sample_batch, x_noise)
    # Same conditioning should give same output
    assert torch.allclose(output1, output2)

    x_noise2 = torch.randn(2, 1, 64, 64, 1)
    output3 = norm_layer.forward(sample_batch, x_noise2)
    # Different conditioning should give different output
    assert not torch.allclose(output1, output3)


def test_conditional_layer_norm_no_noise_is_same_as_layer_norm():
    # Check when no noise
    sample_batch = torch.randn(2, 1, 64, 64, 3)
    norm_layer = torch.nn.LayerNorm(normalized_shape=[64, 64, 3])
    cond_norm_layer = ConditionalLayerNorm(
        normalized_shape=[64, 64, 3], n_noise_channels=None
    )
    output1 = norm_layer(sample_batch)
    output2 = cond_norm_layer(sample_batch)
    assert torch.allclose(output1, output2)


def test_conditional_layer_norm_with_noise_channels_is_same_as_layer_norm_upon_init():
    sample_batch = torch.randn(2, 1, 64, 64, 3)
    norm_layer = torch.nn.LayerNorm(normalized_shape=[64, 64, 3])
    cond_norm_layer = ConditionalLayerNorm(
        normalized_shape=[64, 64, 3], n_noise_channels=64 * 64 * 3
    )
    # Initialize such that output of gamma is 1 and output of beta is 0 to begin with
    torch.nn.init.zeros_(cond_norm_layer.gamma.weight)  # type: ignore  # noqa: PGH003
    torch.nn.init.ones_(cond_norm_layer.gamma.bias)  # type: ignore  # noqa: PGH003
    torch.nn.init.zeros_(cond_norm_layer.beta.weight)  # type: ignore  # noqa: PGH003
    x_noise = torch.randn(2, 64 * 64 * 3)
    output1 = norm_layer(sample_batch)
    output2 = cond_norm_layer(sample_batch, x_noise)
    assert torch.allclose(output1, output2)
