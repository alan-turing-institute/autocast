import torch

from autocast.nn.conditional_layer_norm import ConditionalLayerNorm


def test_conditional_layer_norm():
    # TODO: consider ensemble dim test
    sample_batch = torch.randn(2, 1, 64, 64, 3)

    norm_layer = ConditionalLayerNorm(
        normalized_shape=[1, 2, 3], n_noise_channels=10, n_channels=3
    )

    cond = torch.randn(2, 10)
    output1 = norm_layer.forward(sample_batch, cond)
    output2 = norm_layer.forward(sample_batch, cond)
    assert torch.allclose(output1, output2)
    cond2 = torch.randn(2, 10)
    output3 = norm_layer.forward(sample_batch, cond2)
    print(sample_batch)
    assert not torch.allclose(output1, output3)
