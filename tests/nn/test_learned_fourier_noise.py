"""Exact noise endpoints, covariance and relative-phase behavior."""

import math

import pytest
import torch
from einops import rearrange, repeat

from autocast.nn.noise.learned_fourier import LearnedFourierNoise2d


@pytest.mark.parametrize("grid", [(8, 12), (9, 7), (8, 7), (1, 1)])
def test_identity_and_dc_only_endpoints(grid):
    layer = LearnedFourierNoise2d(3).double()
    noise = torch.randn(4, *grid, 2, dtype=torch.float64)
    torch.testing.assert_close(
        layer(noise),
        repeat(noise, "b h w c -> b l h w c", l=3),
        rtol=1e-13,
        atol=1e-13,
    )
    with torch.no_grad():
        layer.radial_gains.zero_()
    expected = noise.sum((1, 2)) / math.sqrt(math.prod(grid))
    torch.testing.assert_close(
        layer(noise),
        repeat(expected, "b c -> b l h w c", l=3, h=grid[0], w=grid[1]),
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize("grid", [(6, 8), (5, 7), (6, 7)])
def test_learned_transfer_is_hermitian_and_has_unit_expected_variance(grid):
    torch.manual_seed(10)
    layer = LearnedFourierNoise2d(3, radial_knots=8).double()
    assert layer.phase_coefficients is not None
    with torch.no_grad():
        layer.radial_gains.normal_()
        layer.dc_gain.normal_()
        layer.phase_coefficients.normal_()
    transfer = layer.transfer(*grid)
    torch.testing.assert_close(
        transfer.abs().square().mean((1, 2)), torch.ones(3).double()
    )
    impulse = torch.fft.ifft2(transfer, norm="ortho")
    torch.testing.assert_close(
        impulse.imag, torch.zeros_like(impulse.imag), atol=1e-13, rtol=0
    )
    # A complete input basis computes the covariance exactly, without Monte Carlo.
    height, width = grid
    basis = rearrange(
        torch.eye(height * width).double(), "b (h w) -> b h w 1", h=height
    )
    maps = rearrange(layer(basis), "b l h w 1 -> l (h w) b")
    covariance = maps @ maps.mT
    torch.testing.assert_close(
        covariance.diagonal(dim1=-2, dim2=-1), torch.ones(3, height * width).double()
    )


def test_relative_phase_changes_cross_block_covariance_but_not_marginal_power():
    layer = LearnedFourierNoise2d(2, radial_knots=5).double()
    assert layer.phase_coefficients is not None
    basis = rearrange(torch.eye(64).double(), "b (h w) -> b h w 1", h=8)
    initial = rearrange(layer(basis), "b l h w 1 -> l (h w) b")
    with torch.no_grad():
        layer.phase_coefficients[:, 0].fill_(math.atanh(0.5))
    shifted = rearrange(layer(basis), "b l h w 1 -> l (h w) b")
    torch.testing.assert_close(shifted[0], initial[0], rtol=0, atol=0)
    for matrix in shifted:
        torch.testing.assert_close(
            matrix @ matrix.mT, torch.eye(64).double(), atol=1e-13, rtol=1e-13
        )
    assert not torch.allclose(shifted[0] @ shifted[1].mT, initial[0] @ initial[1].mT)
    # The fixed first phase is a reference; only blocks after it have parameters.
    assert layer.phase_coefficients.shape == (1, 2, 5)


def test_phase_moves_an_isolated_wave_and_does_not_modify_the_input():
    layer = LearnedFourierNoise2d(2).double()
    assert layer.phase_coefficients is not None
    height, width = 15, 20
    h, w = torch.meshgrid(
        torch.arange(height).double(), torch.arange(width).double(), indexing="ij"
    )
    angle = 2 * math.pi * (2 * h / height + 3 * w / width)
    wave = rearrange(angle.cos(), "h w -> 1 h w 1")
    saved = wave.clone()
    with torch.no_grad():
        layer.phase_coefficients[:, 0].fill_(math.atanh(0.3))
        layer.phase_coefficients[:, 1].fill_(math.atanh(-0.2))
    output = layer(wave)
    shift = 2 * math.pi * (2 * 0.3 / height - 3 * 0.2 / width)
    torch.testing.assert_close(
        output[:, 1],
        rearrange((angle + shift).cos(), "h w -> 1 h w 1"),
        atol=1e-13,
        rtol=1e-13,
    )
    torch.testing.assert_close(wave, saved, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_dtype_autocast_determinism_and_gradients(dtype):
    torch.manual_seed(11)
    layer = LearnedFourierNoise2d(3, radial_knots=6).to(dtype=dtype)
    noise = torch.randn(2, 7, 10, 2, dtype=dtype, requires_grad=True)
    state = torch.random.get_rng_state().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = layer(noise)
    assert torch.equal(state, torch.random.get_rng_state())
    assert output.shape == (2, 3, 7, 10, 2)
    assert output.dtype == dtype
    (output * torch.randn_like(output)).sum().backward()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert noise.grad is not None
    assert torch.isfinite(noise.grad).all()


def test_zero_filter_stays_finite_including_backward():
    layer = LearnedFourierNoise2d(1)
    assert layer.phase_coefficients is None
    with torch.no_grad():
        layer.radial_gains.zero_()
        layer.dc_gain.zero_()
    output = layer(torch.randn(2, 5, 7, 1))
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_phase_keeps_real_checkerboard_nyquist_modes():
    layer = LearnedFourierNoise2d(3).double()
    assert layer.phase_coefficients is not None
    with torch.no_grad():
        layer.phase_coefficients.normal_(std=2)
    h, w = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    checkerboards = torch.stack(
        ((-1.0) ** h, (-1.0) ** w, (-1.0) ** (h + w)), dim=-1
    ).double()
    inputs = rearrange(checkerboards, "h w c -> 1 h w c")
    torch.testing.assert_close(
        layer(inputs),
        repeat(inputs, "b h w c -> b l h w c", l=3),
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"blocks": 0},
        {"blocks": True},
        {"blocks": 2, "radial_knots": 1},
        {"blocks": 2, "radial_knots": 2.5},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError, match="must be an integer"):
        LearnedFourierNoise2d(**kwargs)


@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 0, 4, 1)])
def test_invalid_input_shape_rejected(shape):
    with pytest.raises(ValueError, match="positive shape"):
        LearnedFourierNoise2d(2)(torch.empty(shape))
