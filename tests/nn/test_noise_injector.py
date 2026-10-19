"""Input-noise sources preserve layout and the clean input channels."""

import pytest
import torch

from autocast.nn.noise import FlowSource, SpatialFourierSource, TemporalOUSource
from autocast.nn.noise.noise_injector import ConcatenatedNoiseInjector


class BadSource(FlowSource):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def sample_like(self, reference):
        if self.error == "shape":
            return reference[..., :1]
        if self.error == "dtype":
            return reference.double()
        return reference.to(device="meta")


def test_default_concat_rng_and_checkpoint_are_unchanged():
    x = torch.randn(2, 3, 8, 12, 2)
    injector = ConcatenatedNoiseInjector(n_channels=4, std=0.3)
    torch.manual_seed(9)
    expected = torch.cat((x, torch.randn(2, 3, 8, 12, 4) * 0.3), dim=-1)
    torch.manual_seed(9)
    assert torch.equal(injector(x), expected)
    assert injector.state_dict() == {}
    assert injector.get_additional_channels() == 4


def test_structured_concat_preserves_input_and_gradient():
    x = torch.randn(2, 3, 8, 12, 2, dtype=torch.float64, requires_grad=True)
    clean = x.detach().clone()
    source = SpatialFourierSource(((1, 0), (0, 2)))
    injector = ConcatenatedNoiseInjector(n_channels=4, std=0.3, noise_source=source)
    torch.manual_seed(9)
    expected_noise = source.sample_like(x.new_empty(2, 3, 8, 12, 4)) * 0.3
    torch.manual_seed(9)
    output = injector(x)
    assert output.shape == (2, 3, 8, 12, 6)
    assert output.dtype == x.dtype
    assert output.device == x.device
    assert torch.equal(output[..., :2], clean)
    assert torch.equal(x.detach(), clean)
    assert torch.equal(output[..., 2:], expected_noise)
    assert not torch.equal(output[..., 2:], injector(x)[..., 2:])
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_existing_temporal_source_only_correlates_input_slots():
    injector = ConcatenatedNoiseInjector(
        n_channels=2, noise_source=TemporalOUSource(correlation_time=float("inf"))
    )
    x = torch.zeros(2, 3, 8, 12, 1)
    first, second = injector(x), injector(x)
    assert torch.equal(first[:, 0, ..., 1:], first[:, 1, ..., 1:])
    assert not torch.equal(first[..., 1:], second[..., 1:])


@pytest.mark.parametrize("error", ["shape", "dtype", "device"])
def test_source_contract_is_checked(error):
    injector = ConcatenatedNoiseInjector(n_channels=2, noise_source=BadSource(error))
    with pytest.raises(ValueError, match="Noise source"):
        injector(torch.zeros(2, 1, 8, 12, 3))


def test_disabled_concat_does_not_draw_or_validate_source():
    injector = ConcatenatedNoiseInjector(n_channels=0, noise_source=BadSource("shape"))
    x = torch.randn(2, 3, 4)
    before = torch.random.get_rng_state()
    assert injector(x) is x
    assert torch.equal(before, torch.random.get_rng_state())


def test_structured_concat_rejects_flattened_grid():
    injector = ConcatenatedNoiseInjector(noise_source=SpatialFourierSource(((1, 0),)))
    with pytest.raises(ValueError, match="reference must have shape"):
        injector(torch.empty(2, 12, 3))
