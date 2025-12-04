import torch

from auto_cast.decoders.channels_last import ChannelsLast


def test_channels_last_reorders_dimensions():
    decoder = ChannelsLast()
    x = torch.randn(2, 3, 4, 5, 6)

    output = decoder(x)

    assert output.shape == (2, 4, 5, 6, 3)
    # Spot-check a value to ensure permutation matches expectation
    assert torch.allclose(output[0, 0, 0, 0, 0], x[0, 0, 0, 0, 0])
