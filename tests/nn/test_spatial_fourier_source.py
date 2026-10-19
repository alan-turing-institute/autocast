"""Prescribed Fourier fields preserve covariance without draw normalization."""

import math

import pytest
import torch

from autocast.nn.noise import SpatialFourierSource

WAVES = ((3, 0), (0, 3), (2, 2), (2, -2))


@pytest.mark.parametrize(
    "waves",
    [
        [],
        [[0, 0]],
        [[1, 0], [-1, 0]],
        [[1, 0], [1, 0]],
        [[1.5, 0]],
        [[True, 0]],
        [[1]],
        [[1, 0, 1]],
    ],
)
def test_invalid_wavevectors(waves):
    with pytest.raises(ValueError, match="Spatial noise"):
        SpatialFourierSource(waves)


@pytest.mark.parametrize("shape", [(2, 16, 24, 4), (2, 0, 16, 24, 4)])
def test_explicit_time_and_grid_are_required(shape):
    with pytest.raises(ValueError, match="reference must have shape"):
        SpatialFourierSource(WAVES).sample_like(torch.empty(shape))


def test_floating_reference_is_required():
    with pytest.raises(TypeError, match="floating point"):
        SpatialFourierSource(WAVES).sample_like(torch.zeros(2, 1, 16, 24, 4).long())


@pytest.mark.parametrize("shape", [(2, 1, 0, 24, 4), (2, 1, 16, 24, 0)])
def test_nonempty_grid_and_channels_are_required(shape):
    with pytest.raises(ValueError, match="positive"):
        SpatialFourierSource(WAVES).sample_like(torch.empty(shape))


@pytest.mark.parametrize("waves", [[[4, 0]], [[0, -6]], [[5, 0]]])
def test_modes_must_be_strictly_below_grid_nyquist(waves):
    with pytest.raises(ValueError, match="Nyquist"):
        SpatialFourierSource(waves).sample_like(torch.empty(2, 1, 8, 12, 1))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_shape_dtype_seed_and_independent_draws(dtype):
    source = SpatialFourierSource(WAVES)
    reference = torch.empty(2, 3, 16, 24, 4, dtype=dtype)
    torch.manual_seed(42)
    first = source.sample_like(reference)
    torch.manual_seed(42)
    repeated = source.sample_like(reference)
    other = source.sample_like(reference)
    assert first.shape == reference.shape
    assert first.dtype == dtype
    assert first.device == reference.device
    assert torch.equal(first, repeated)
    assert not torch.equal(first, other)
    assert not torch.equal(first[0], first[1])
    assert not torch.equal(first[:, 0], first[:, 1])
    assert not torch.equal(first[..., 0], first[..., 1])
    assert source.state_dict() == {}


def test_exact_basis_phase_covariance_spectrum_and_unit_variance():
    reference = torch.empty(1024, 1, 16, 24, 4, dtype=torch.float64)
    torch.manual_seed(5)
    noise = SpatialFourierSource(WAVES).sample_like(reference)
    torch.testing.assert_close(
        noise.mean((2, 3)), torch.zeros_like(noise.mean((2, 3))), atol=1e-14, rtol=0
    )
    assert torch.allclose(noise.var(0), torch.ones_like(noise[0]), atol=0.16, rtol=0)
    assert abs(float(noise.square().mean()) - 1) < 0.03
    # Gaussian draw amplitudes fluctuate; per-draw normalization would fail this.
    assert noise.square().mean((1, 2, 3)).std() > 0.3
    power = (
        torch.fft.fft2(noise, dim=(2, 3), norm="forward").abs().square().mean((0, 1, 4))
    )
    allowed = torch.zeros(16, 24, dtype=torch.bool)
    for kh, kw in WAVES:
        allowed[kh % 16, kw % 24] = allowed[-kh % 16, -kw % 24] = True
    assert float(power[~allowed].sum() / power.sum()) < 1e-25
    torch.manual_seed(5)
    coefficients = torch.randn(1024, 4, 4, 2, dtype=reference.dtype)
    for h, w in [(0, 0), (1, 3), (15, 7)]:
        phase = torch.tensor(
            [2 * math.pi * (kh * h / 16 + kw * w / 24) for kh, kw in WAVES],
            dtype=reference.dtype,
        )
        expected = (
            coefficients[..., 0] * phase.cos() + coefficients[..., 1] * phase.sin()
        ).sum(-1) / math.sqrt(len(WAVES))
        torch.testing.assert_close(noise[:, 0, h, w], expected)
    expected_cov = sum(math.cos(2 * math.pi * kh / 16) for kh, _ in WAVES) / len(WAVES)
    observed_cov = (noise * noise.roll(1, dims=2)).mean()
    assert float(observed_cov) == pytest.approx(expected_cov, abs=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
def test_cuda_sampling():
    reference = torch.empty(2, 1, 16, 24, 4, device="cuda")
    sample = SpatialFourierSource(WAVES).sample_like(reference)
    assert sample.device == reference.device
    assert torch.isfinite(sample).all()
