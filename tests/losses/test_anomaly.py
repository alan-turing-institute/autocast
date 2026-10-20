"""Check online spectral spread/error moments and their gradient routing."""

import math

import pytest
import torch
from einops import rearrange

from autocast.losses.anomaly import SpectralSpreadErrorLoss
from autocast.losses.composite import WeightedSumLoss
from autocast.losses.ensemble import AlphaFairCRPSLoss


def _reference_spectra(preds, targets, edges, include_dc=True):
    """Pool explicit FFT powers independently of the implementation helpers."""
    spatial = tuple(range(2, targets.ndim - 1))
    mean = preds.mean(-1)
    anomalies = torch.fft.fftn(preds - mean.unsqueeze(-1), dim=spatial, norm="ortho")
    errors = torch.fft.fftn(targets - mean.detach(), dim=spatial, norm="ortho")
    spread = anomalies.abs().square().sum(-1) / (preds.shape[-1] - 1)
    error = errors.abs().square()
    frequencies = torch.meshgrid(
        *[
            torch.fft.fftfreq(targets.shape[dim], dtype=torch.float64)
            for dim in spatial
        ],
        indexing="ij",
    )
    radius = torch.stack(frequencies).square().sum(0).sqrt().flatten()
    masks = [radius == 0] if include_dc else []
    lower = 0
    for index, upper in enumerate(edges):
        mask = (radius > 0) & (radius >= lower)
        mask &= radius <= upper if index == len(edges) - 1 else radius < upper
        if mask.any():
            masks.append(mask)
        lower = upper
    pooled = []
    for power in (spread, error):
        flat = rearrange(power, "b t ... c -> b t (...) c")
        pooled.append(torch.stack([flat[:, :, mask].mean((0, 2)) for mask in masks], 1))
    return pooled


@pytest.mark.parametrize("spatial_shape", [(5,), (4, 6), (3, 4, 2)])
@pytest.mark.parametrize("include_dc", [False, True])
def test_power_spectra_match_reference_without_mixing_channels_or_leads(
    spatial_shape, include_dc
):
    rng = torch.Generator().manual_seed(23)
    targets = torch.randn(3, 2, *spatial_shape, 2, dtype=torch.float64, generator=rng)
    preds = torch.randn(*targets.shape, 4, dtype=torch.float64, generator=rng)
    edges = [0.15, 0.35, 1.0]
    loss = SpectralSpreadErrorLoss(edges, include_dc=include_dc, power_scale=2)

    actual = loss.power_spectra(preds, targets)
    expected = _reference_spectra(preds, targets, edges, include_dc)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference)
    spread, error = expected
    expected_loss = (((1 + 1 / 4) * spread - error) / 2).square().mean()
    torch.testing.assert_close(loss(preds, targets), expected_loss)


def test_averages_power_not_anomaly_fields_and_is_stateless():
    pattern = torch.tensor([1.0, -1.0, 1.0, -1.0])
    preds = rearrange(torch.stack([pattern, -pattern]), "m x -> 1 1 x 1 m")
    targets = torch.zeros(1, 1, 4, 1)
    loss = SpectralSpreadErrorLoss([0.3, 0.5])

    spread, error = loss.power_spectra(preds, targets)

    assert torch.count_nonzero(preds.mean(-1)) == 0
    torch.testing.assert_close(spread.flatten(), torch.tensor([0.0, 0.0, 8.0]))
    assert torch.count_nonzero(error) == 0
    assert not loss.state_dict()
    assert loss.requires_ambient_predictions


def test_pools_batch_before_squaring_but_not_channels_or_leads():
    preds = torch.tensor([-1.0, 1.0]).expand(2, 1, 4, 1, 2)
    targets = torch.zeros(2, 1, 4, 1)
    targets[1] = math.sqrt(6)
    loss = SpectralSpreadErrorLoss([0.1])

    # Corrected spread is 12 in both cases. Error powers are 0 and 24.
    torch.testing.assert_close(
        loss(preds, targets), torch.tensor(0.0), atol=1e-10, rtol=0
    )
    assert loss(preds[:1], targets[:1]) > 0
    assert loss(preds[1:], targets[1:]) > 0
    split_channels = torch.cat([preds[:1], preds[1:]], dim=-2)
    split_targets = torch.cat([targets[:1], targets[1:]], dim=-1)
    assert loss(split_channels, split_targets) > 0
    assert loss(preds.transpose(0, 1), targets.transpose(0, 1)) > 0


@pytest.mark.parametrize("members", [2, 8, 32])
def test_gaussian_moments_include_finite_ensemble_mean_error(members):
    rng = torch.Generator().manual_seed(123)
    targets = torch.randn(131072, 1, 1, 1, dtype=torch.float64, generator=rng)
    preds = torch.randn(*targets.shape, members, dtype=torch.float64, generator=rng)
    loss = SpectralSpreadErrorLoss([0.5])

    spread, error = loss.power_spectra(preds, targets)

    spread_tolerance = 4 * math.sqrt(2 / (targets.shape[0] * (members - 1)))
    error_tolerance = 4 * (1 + 1 / members) * math.sqrt(2 / targets.shape[0])
    torch.testing.assert_close(
        spread, torch.ones_like(spread), atol=spread_tolerance, rtol=0
    )
    torch.testing.assert_close(
        error, torch.full_like(error, 1 + 1 / members), atol=error_tolerance, rtol=0
    )
    corrected = ((1 + 1 / members) * spread - error).square().mean()
    uncorrected = (spread - error).square().mean()
    torch.testing.assert_close(loss(preds, targets), corrected)
    assert corrected < uncorrected


def test_observed_residual_still_contains_mean_bias():
    preds = torch.tensor([-1.0, 1.0]).expand(1, 1, 4, 1, 2)
    targets = torch.zeros(1, 1, 4, 1)
    loss = SpectralSpreadErrorLoss([0.5])
    spread, error = loss.power_spectra(preds, targets)
    shifted_spread, shifted_error = loss.power_spectra(preds + 3, targets)

    torch.testing.assert_close(shifted_spread, spread)
    assert error[0, 0, 0] == 0
    assert shifted_error[0, 0, 0] == 36


def test_stop_gradient_blocks_mean_target_but_preserves_spread_learning():
    rng = torch.Generator().manual_seed(33)
    location = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    amplitude = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    noise = torch.randn(3, 2, 4, 6, 2, 8, dtype=torch.float64, generator=rng)
    targets = torch.randn(3, 2, 4, 6, 2, dtype=torch.float64, generator=rng)
    preds = location + amplitude * noise
    preds.retain_grad()
    loss = SpectralSpreadErrorLoss([0.2, 0.4, 0.71])

    loss(preds, targets).backward()

    assert location.grad is not None
    torch.testing.assert_close(
        location.grad, torch.zeros_like(location), atol=1e-12, rtol=0
    )
    assert amplitude.grad is not None
    assert amplitude.grad.abs() > 0
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()
    torch.testing.assert_close(
        preds.grad.sum(-1), torch.zeros_like(targets), atol=1e-12, rtol=0
    )


def test_gradients_equal_explicit_frozen_target_reference():
    rng = torch.Generator().manual_seed(38)
    targets = torch.randn(2, 2, 4, 6, 2, dtype=torch.float64, generator=rng)
    preds = torch.randn(*targets.shape, 4, dtype=torch.float64, generator=rng)
    preds.requires_grad_()
    edges = [0.2, 0.4, 0.71]
    loss = SpectralSpreadErrorLoss(edges)
    spread, error = _reference_spectra(preds, targets, edges)
    expected = ((1 + 1 / 4) * spread - error).square().mean()

    reference_gradient = torch.autograd.grad(expected, preds)[0]
    actual_gradient = torch.autograd.grad(loss(preds, targets), preds)[0]

    torch.testing.assert_close(actual_gradient, reference_gradient)


def test_joint_loss_retains_primary_mean_gradient_and_adds_spread_gradient():
    rng = torch.Generator().manual_seed(41)
    location = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    amplitude = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    preds = location + amplitude * torch.randn(
        3, 1, 4, 1, 8, dtype=torch.float64, generator=rng
    )
    targets = torch.zeros(3, 1, 4, 1, dtype=torch.float64)
    base = AlphaFairCRPSLoss(alpha=0.95)
    auxiliary = SpectralSpreadErrorLoss([0.3, 0.5])
    joint = WeightedSumLoss([base, auxiliary], [1.0, 0.01])

    base_location, base_amplitude = torch.autograd.grad(
        base(preds, targets), (location, amplitude), retain_graph=True
    )
    joint_location, joint_amplitude = torch.autograd.grad(
        joint(preds, targets), (location, amplitude)
    )

    assert base_location.abs() > 0
    torch.testing.assert_close(joint_location, base_location)
    assert not torch.isclose(joint_amplitude, base_amplitude)


def test_channels_and_leads_are_not_pooled_into_radial_geometry():
    rng = torch.Generator().manual_seed(39)
    preds = torch.randn(2, 3, 4, 6, 2, 4, dtype=torch.float64, generator=rng)
    targets = torch.randn(2, 3, 4, 6, 2, dtype=torch.float64, generator=rng)
    loss = SpectralSpreadErrorLoss([0.2, 0.4, 0.71])
    reference = torch.stack(
        [
            loss(
                preds[:, lead : lead + 1, ..., channel : channel + 1, :],
                targets[:, lead : lead + 1, ..., channel : channel + 1],
            )
            for lead in range(3)
            for channel in range(2)
        ]
    ).mean()
    torch.testing.assert_close(loss(preds, targets), reference)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_precision_promotion_and_finite_gradients(dtype):
    rng = torch.Generator().manual_seed(37)
    preds = torch.randn(2, 2, 5, 7, 2, 3, generator=rng).to(dtype).requires_grad_()
    targets = torch.randn(2, 2, 5, 7, 2, generator=rng).to(dtype)
    loss = SpectralSpreadErrorLoss([0.2, 0.4, 0.71])

    value = loss(preds, targets)
    value.backward()

    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert value.dtype == expected_dtype
    assert value.device == preds.device
    assert preds.grad is not None
    assert preds.grad.dtype == dtype
    assert torch.isfinite(preds.grad).all()


def test_mixed_precision_uses_one_geometry_and_common_precision():
    preds = torch.randn(2, 1, 10, 2, 4, dtype=torch.float32, requires_grad=True)
    targets = torch.randn(2, 1, 10, 2, dtype=torch.float64)
    loss = SpectralSpreadErrorLoss([0.3, 0.5])
    torch.testing.assert_close(loss(preds, targets), loss(preds.double(), targets))


def test_band_cutoff_dc_toggle_and_empty_bands():
    rng = torch.Generator().manual_seed(17)
    preds = torch.randn(2, 1, 4, 1, 4, dtype=torch.float64, generator=rng)
    targets = torch.randn(2, 1, 4, 1, dtype=torch.float64, generator=rng)
    compact = SpectralSpreadErrorLoss([0.25, 0.5], include_dc=False)
    empty_band = SpectralSpreadErrorLoss([0.1, 0.25, 0.5], include_dc=False)

    torch.testing.assert_close(compact(preds, targets), empty_band(preds, targets))
    # A Nyquist-only wave is outside the configured cutoff.
    wave = rearrange(torch.tensor([1.0, -1.0, 1.0, -1.0]), "x -> 1 1 x 1 1")
    with_wave = preds + wave * torch.tensor([-1.0, 1.0, -1.0, 1.0])
    low = SpectralSpreadErrorLoss([0.3], include_dc=False)
    torch.testing.assert_close(low(preds, targets), low(with_wave, targets))


@pytest.mark.parametrize(
    "edges", [[], [0], [-0.1], [0.3, 0.2], [0.2, 0.2], [math.nan], [math.inf]]
)
def test_rejects_invalid_band_edges(edges):
    with pytest.raises(ValueError, match=r"(band edge|Band edges)"):
        SpectralSpreadErrorLoss(edges)


@pytest.mark.parametrize("power_scale", [0, -1, math.nan, math.inf])
def test_rejects_invalid_fixed_power_scale(power_scale):
    with pytest.raises(ValueError, match="power_scale"):
        SpectralSpreadErrorLoss([0.5], power_scale=power_scale)


@pytest.mark.parametrize("shape", [(2, 1, 4, 2), (2, 1, 4, 2, 1), (2, 1, 4, 2, 0)])
def test_rejects_missing_or_too_small_ensemble(shape):
    with pytest.raises(ValueError, match="at least two ensemble members"):
        SpectralSpreadErrorLoss([0.5])(torch.zeros(shape), torch.zeros(2, 1, 4, 2))


def test_rejects_incompatible_shapes_and_no_spatial_axes():
    loss = SpectralSpreadErrorLoss([0.5])
    with pytest.raises(ValueError, match="Predictions must match targets"):
        loss(torch.zeros(2, 1, 4, 2, 3), torch.zeros(2, 1, 5, 2))
    with pytest.raises(ValueError, match="n_spatial_dims"):
        loss(torch.zeros(2, 1, 2, 3), torch.zeros(2, 1, 2))


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_rejects_non_real_floating_fields(dtype):
    with pytest.raises(ValueError, match="real floating"):
        SpectralSpreadErrorLoss([0.5])(
            torch.zeros(2, 1, 4, 2, 3, dtype=dtype), torch.zeros(2, 1, 4, 2)
        )


def test_rejects_empty_fields_and_no_scored_modes():
    loss = SpectralSpreadErrorLoss([0.1], include_dc=False)
    with pytest.raises(ValueError, match="nonempty"):
        loss(torch.zeros(0, 1, 4, 1, 3), torch.zeros(0, 1, 4, 1))
    with pytest.raises(ValueError, match="no nonempty"):
        loss(torch.zeros(2, 1, 4, 1, 3), torch.zeros(2, 1, 4, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_keeps_loss_and_gradients_on_device():
    preds = torch.randn(2, 1, 5, 7, 2, 4, device="cuda", requires_grad=True)
    targets = torch.randn(2, 1, 5, 7, 2, device="cuda")
    loss = SpectralSpreadErrorLoss([0.2, 0.4, 0.71]).cuda()
    value = loss(preds, targets)
    value.backward()
    assert value.device.type == "cuda"
    assert preds.grad is not None
    assert preds.grad.device.type == "cuda"
    assert torch.isfinite(preds.grad).all()
