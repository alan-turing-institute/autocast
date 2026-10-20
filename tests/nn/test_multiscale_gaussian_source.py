"""Radial Gaussian envelopes preserve scale units and ensemble statistics."""

import math

import pytest
import torch

from autocast.nn.noise import MultiscaleGaussianSource
from autocast.nn.noise.spatial import multiscale_gaussian_spatial_filter

CUTOFFS = (1.0, 2.0, 4.0, 6.0)


def _transfer(shape, cutoffs, lengths=(1.0, 1.0)):
    reference = torch.empty(shape, dtype=torch.float64)
    torch.manual_seed(153)
    sampled = MultiscaleGaussianSource(cutoffs, lengths).sample_like(reference)
    torch.manual_seed(153)
    white = torch.randn_like(reference)
    spectrum = torch.fft.fft2(sampled, dim=(2, 3), norm="ortho")
    original = torch.fft.fft2(white, dim=(2, 3), norm="ortho")
    return (spectrum / original)[0, 0], sampled


@pytest.mark.parametrize(
    "cutoffs", [[], [0], [-1], [True], [math.nan], [math.inf], ["1"]]
)
def test_rejects_invalid_half_power_wavenumbers(cutoffs):
    with pytest.raises(ValueError, match="half_power_wavenumbers"):
        MultiscaleGaussianSource(cutoffs)


@pytest.mark.parametrize(
    "lengths", [[], [1], [1, 1, 1], [0, 1], [-1, 1], [True, 1], [math.inf, 1]]
)
def test_rejects_invalid_domain_lengths(lengths):
    with pytest.raises(ValueError, match="domain_lengths"):
        MultiscaleGaussianSource(CUTOFFS, lengths)


@pytest.mark.parametrize(
    "shape", [(2, 16, 16, 4), (2, 0, 16, 16, 4), (2, 1, 0, 16, 4), (0, 1, 16, 16, 4)]
)
def test_requires_nonempty_explicit_grid(shape):
    with pytest.raises(ValueError, match=r"shape|positive"):
        MultiscaleGaussianSource(CUTOFFS).sample_like(torch.empty(shape))


def test_requires_float_and_matching_channel_count():
    source = MultiscaleGaussianSource(CUTOFFS)
    with pytest.raises(TypeError, match="floating point"):
        source.sample_like(torch.zeros(1, 1, 16, 16, 4, dtype=torch.int64))
    with pytest.raises(ValueError, match="channels must match"):
        source.sample_like(torch.empty(1, 1, 16, 16, 3))


@pytest.mark.parametrize(
    ("shape", "cutoffs", "lengths"),
    [
        ((1, 1, 16, 16, 1), [8], [1, 1]),
        ((1, 1, 16, 16, 1), [9], [1, 1]),
        ((1, 1, 16, 12, 1), [6], [1, 1]),
        ((1, 1, 16, 32, 1), [4], [2, 1]),
    ],
)
def test_unresolved_scales_fail_before_sampling(shape, cutoffs, lengths):
    torch.manual_seed(22)
    rng_state = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="Nyquist"):
        MultiscaleGaussianSource(cutoffs, lengths).sample_like(torch.empty(shape))
    assert torch.equal(rng_state, torch.random.get_rng_state())


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_shape_seed_dtype_independence_and_statelessness(dtype):
    reference = torch.empty(2, 3, 16, 24, 4, dtype=dtype)
    source = MultiscaleGaussianSource(CUTOFFS)
    torch.manual_seed(42)
    first = source.sample_like(reference)
    torch.manual_seed(42)
    repeated = source.sample_like(reference)
    other = source.sample_like(reference)
    assert first.shape == reference.shape
    assert first.device == reference.device
    assert first.dtype == reference.dtype
    assert torch.isfinite(first).all()
    assert torch.equal(first, repeated)
    assert not torch.equal(first, other)
    assert not torch.equal(first[0], first[1])
    assert not torch.equal(first[:, 0], first[:, 1])
    assert not torch.equal(first[..., 0], first[..., 1])
    assert source.state_dict() == {}
    assert list(source.parameters()) == []


def test_exact_radial_power_envelopes_and_half_power_definition():
    transfer, _ = _transfer((1, 1, 16, 16, 4), CUTOFFS)
    frequency = torch.fft.fftfreq(16, d=1 / 16, dtype=torch.float64)
    radius_squared = frequency[:, None].square() + frequency[None, :].square()
    power = torch.exp(
        -math.log(2) * radius_squared[..., None] / torch.tensor(CUTOFFS).square()
    )
    expected = (power / power.mean((0, 1))).sqrt()
    torch.testing.assert_close(
        transfer.imag, torch.zeros_like(expected), atol=2e-13, rtol=0
    )
    torch.testing.assert_close(transfer.real, expected, atol=2e-13, rtol=1e-12)
    for channel, cutoff in enumerate(CUTOFFS):
        relative_power = transfer[int(cutoff), 0, channel].abs().square()
        relative_power /= transfer[0, 0, channel].abs().square()
        assert float(relative_power) == pytest.approx(0.5, abs=1e-12)
    # Equal-radius directions receive equal amplitudes; no shell rounding.
    torch.testing.assert_close(transfer[3, 4], transfer[0, 5], atol=2e-13, rtol=0)
    torch.testing.assert_close(transfer[1, 2], transfer[-1, -2], atol=2e-13, rtol=0)


@pytest.mark.parametrize("size", [16, 24, 32, 64])
def test_physical_half_power_scale_does_not_follow_resolution(size):
    transfer, _ = _transfer((1, 1, size, size, 2), (2.0, 4.0))
    for channel, cutoff in enumerate((2, 4)):
        ratio = transfer[cutoff, 0, channel].abs().square()
        ratio /= transfer[0, 0, channel].abs().square()
        assert float(ratio) == pytest.approx(0.5, abs=1e-12)


def test_rectangular_physical_grid_and_domain_unit_conversion():
    transfer, _ = _transfer((1, 1, 16, 32, 2), (1.0, 3.0), (2.0, 1.0))
    # Two height cycles / length two equals one width cycle / length one.
    torch.testing.assert_close(transfer[2, 0], transfer[0, 1], atol=1e-12, rtol=0)
    _, original = _transfer((1, 1, 16, 16, 2), (1.0, 3.0))
    _, converted = _transfer((1, 1, 16, 16, 2), (0.5, 1.5), (2.0, 2.0))
    torch.testing.assert_close(original, converted, atol=0, rtol=0)


def test_expected_unit_variance_independent_channels_and_fluctuating_draws():
    torch.manual_seed(4)
    noise = MultiscaleGaussianSource(CUTOFFS).sample_like(
        torch.empty(2048, 1, 16, 16, 4)
    )
    torch.testing.assert_close(
        noise.square().mean((0, 1, 2, 3)), torch.ones(4), atol=0.04, rtol=0
    )
    assert float(noise[..., 0].mean((1, 2, 3)).std()) > 0.2  # DC is retained.
    assert float(noise[..., 0].square().mean((1, 2, 3)).std()) > 0.2
    for channel in range(1, 4):
        assert float((noise[..., 0] * noise[..., channel]).mean().abs()) < 0.04
    # Coarse and fine channels have intentionally different spatial coherence.
    adjacent = (noise * noise.roll(1, dims=2)).mean((0, 1, 2, 3))
    assert torch.all(adjacent[:-1] > adjacent[1:])


def test_filter_is_linear_differentiable_and_finite_at_tiny_cutoff():
    field = torch.randn(2, 1, 16, 16, 2, dtype=torch.float64, requires_grad=True)
    filtered = multiscale_gaussian_spatial_filter(
        field, half_power_wavenumbers=(1, 3), domain_lengths=(1, 1)
    )
    filtered.square().mean().backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()
    limiting = MultiscaleGaussianSource([1e-100]).sample_like(
        torch.empty(2, 1, 8, 8, 1)
    )
    assert torch.isfinite(limiting).all()
    torch.testing.assert_close(limiting, limiting[:, :, :1, :1].expand_as(limiting))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
def test_cuda_sampling():
    reference = torch.empty(2, 1, 16, 16, 4, device="cuda", dtype=torch.bfloat16)
    result = MultiscaleGaussianSource(CUTOFFS).sample_like(reference)
    assert result.device == reference.device
    assert result.dtype == reference.dtype
    assert torch.isfinite(result).all()
