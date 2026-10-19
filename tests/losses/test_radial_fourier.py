import math
from itertools import product

import pytest
import torch
from einops import rearrange

from autocast.losses import RadialFourierDecomposition
from autocast.losses.composite import WeightedSumLoss
from autocast.losses.decomposition import (
    ComponentBranch,
    ComponentObjective,
    IdentityDecomposition,
    TransformedLoss,
)
from autocast.losses.ensemble import CoefficientCRPSLoss
from autocast.losses.spectral import FourierDecomposition
from losses.test_coefficient_crps import _reference_score
from losses.test_spectral import _explicit_dft


def _reference_masks(spatial_shape, edges):
    """Enumerate signed integer modes independently of FFT frequency helpers."""
    frequencies = [
        [(index if 2 * index < size else index - size) / size for index in range(size)]
        for size in spatial_shape
    ]
    radii = [
        math.sqrt(sum(value**2 for value in mode)) for mode in product(*frequencies)
    ]
    masks = []
    lower = 0.0
    for index, upper in enumerate(edges):
        masks.append(
            torch.tensor(
                [
                    lower <= radius
                    and (radius <= upper if index == len(edges) - 1 else radius < upper)
                    for radius in radii
                ]
            ).reshape(spatial_shape)
        )
        lower = upper
    return masks


@pytest.mark.parametrize("spatial_shape", [(5,), (6,), (3, 4), (4, 5), (2, 3, 4)])
@pytest.mark.parametrize("members", [None, 3])
@pytest.mark.parametrize("band_mean", [False, True])
def test_radial_bands_match_dft_and_preserve_field_axes(
    spatial_shape, members, band_mean
):
    generator = torch.Generator().manual_seed(21)
    shape = (2, 3, *spatial_shape, 2)
    if members is not None:
        shape = (*shape, members)
    tensor = torch.randn(shape, dtype=torch.float64, generator=generator)
    edges = [0.2, 0.4, math.sqrt(len(spatial_shape)) / 2]
    masks = _reference_masks(spatial_shape, edges)
    coefficients = _explicit_dft(tensor, range(2, 2 + len(spatial_shape)), "ortho")

    bands = RadialFourierDecomposition(edges, band_mean=band_mean)(
        tensor, n_spatial_dims=len(spatial_shape)
    )

    assert list(bands) == ["band_0", "band_1", "band_2"]
    assert sum(int(mask.sum()) for mask in masks) == math.prod(spatial_shape)
    reconstructed = torch.zeros_like(coefficients)
    for band, mask in zip(bands.values(), masks, strict=True):
        count = int(mask.sum())
        broadcast_mask = mask.reshape(
            1, 1, *spatial_shape, 1, *([1] if members else [])
        )
        scale = math.sqrt(max(1, count)) if band_mean else 1
        assert band.shape == tensor.shape
        assert band.dtype == torch.complex128
        assert band.device == tensor.device
        assert torch.count_nonzero(band.masked_select(~broadcast_mask)) == 0
        torch.testing.assert_close(
            band, coefficients * broadcast_mask / scale, rtol=1e-12, atol=1e-12
        )
        reconstructed = reconstructed + band * scale
    torch.testing.assert_close(reconstructed, coefficients, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
def test_radial_bands_respect_fft_normalization(norm):
    tensor = torch.arange(24, dtype=torch.float64).reshape(1, 1, 3, 4, 2)
    masks = _reference_masks((3, 4), [0.3, 1])
    expected = _explicit_dft(tensor, (2, 3), norm)

    bands = RadialFourierDecomposition([0.3, 1], norm=norm)(tensor, n_spatial_dims=2)

    for band, mask in zip(bands.values(), masks, strict=True):
        torch.testing.assert_close(
            band, expected * mask.reshape(1, 1, 3, 4, 1), rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize(
    ("last_edge", "last_indices"),
    [(0.375, [2, 3, 5, 6]), (0.5, [2, 3, 4, 5, 6])],
)
def test_dc_internal_edges_last_cutoff_and_nyquist(last_edge, last_indices):
    tensor = torch.zeros(1, 1, 8, 1, dtype=torch.float64)
    tensor[0, 0, 0, 0] = math.sqrt(8)

    bands = RadialFourierDecomposition([0.25, last_edge])(tensor, n_spatial_dims=1)

    first = torch.zeros(8, dtype=torch.complex128)
    first[[0, 1, 7]] = 1
    last = torch.zeros_like(first)
    last[last_indices] = 1
    torch.testing.assert_close(bands["band_0"].flatten(), first)
    torch.testing.assert_close(bands["band_1"].flatten(), last)


@pytest.mark.parametrize(("size", "edge"), [(30, 0.1), (10, 0.3)])
def test_mixed_precision_fields_use_identical_masks_at_rational_cutoffs(size, edge):
    preds = torch.zeros(1, 1, size, 1, 2, dtype=torch.float32)
    preds[:, :, 0] = 1
    targets = preds[..., 0].double()
    decomposition = RadialFourierDecomposition([edge], norm="backward")

    pred_band = decomposition(preds, n_spatial_dims=1)["band_0"]
    target_band = decomposition(targets, n_spatial_dims=1)["band_0"]

    expected = torch.zeros(size, dtype=torch.complex128)
    expected[[0, 1, 2, 3, size - 3, size - 2, size - 1]] = 1
    torch.testing.assert_close(pred_band[0, 0, :, 0, 0].cdouble(), expected)
    torch.testing.assert_close(target_band.flatten(), expected)
    loss = TransformedLoss(decomposition, CoefficientCRPSLoss(alpha=0.95))
    assert loss(preds, targets).item() == 0


@pytest.mark.parametrize(("edge", "count"), [(0.5, 11), (math.sqrt(2) / 2, 16)])
def test_radial_corner_frequencies_exceed_per_axis_nyquist(edge, count):
    tensor = torch.zeros(1, 1, 4, 4, 1, dtype=torch.float64)
    tensor[0, 0, 0, 0, 0] = 4

    band = RadialFourierDecomposition([edge])(tensor, n_spatial_dims=2)["band_0"]

    assert torch.count_nonzero(band) == count
    assert (band[0, 0, 2, 2, 0] != 0) == (count == 16)


def test_a_known_plane_wave_keeps_its_complex_phase_in_one_band():
    x = torch.arange(6, dtype=torch.float64)
    y = torch.arange(8, dtype=torch.float64)
    phase = 2 * math.pi * (x[:, None] * 2 / 6 + y[None, :] / 8) + math.pi / 3
    tensor = rearrange(phase.mul(1j).exp(), "x y -> 1 1 x y 1")

    bands = RadialFourierDecomposition([0.25, 0.4, 1])(tensor, n_spatial_dims=2)

    expected = torch.zeros_like(tensor)
    expected[0, 0, 2, 1, 0] = math.sqrt(48) * torch.tensor(
        complex(0.5, math.sqrt(3) / 2), dtype=torch.complex128
    )
    torch.testing.assert_close(bands["band_1"], expected, rtol=1e-12, atol=1e-12)
    for name in ("band_0", "band_2"):
        torch.testing.assert_close(
            bands[name], torch.zeros_like(tensor), atol=1e-12, rtol=0
        )


def test_band_mean_keeps_empty_bands_zero_and_counts_only_spatial_modes():
    tensor = torch.zeros(2, 3, 4, 2, 3, dtype=torch.float64)
    tensor[:, :, 0] = 2
    bands = RadialFourierDecomposition([0.1, 0.2, 0.5], band_mean=True)(
        tensor, n_spatial_dims=1
    )

    assert torch.count_nonzero(bands["band_1"]) == 0
    assert torch.isfinite(bands["band_1"]).all()
    torch.testing.assert_close(
        bands["band_0"][:, :, 0], torch.ones(2, 3, 2, 3, dtype=torch.complex128)
    )
    torch.testing.assert_close(
        bands["band_2"][:, :, 1:],
        torch.ones(2, 3, 3, 2, 3, dtype=torch.complex128) / math.sqrt(3),
    )


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (torch.float16, torch.complex64),
        (torch.bfloat16, torch.complex64),
        (torch.float32, torch.complex64),
        (torch.float64, torch.complex128),
        (torch.complex64, torch.complex64),
        (torch.complex128, torch.complex128),
    ],
)
def test_radial_dtype_device_and_gradients(dtype, expected_dtype):
    generator = torch.Generator().manual_seed(8)
    tensor = torch.randn(
        1, 2, 3, 4, 2, 3, dtype=dtype, generator=generator, requires_grad=True
    )
    masks = _reference_masks((3, 4), [0.3, 1])
    coefficients = _explicit_dft(tensor.detach(), (2, 3), "ortho")

    bands = RadialFourierDecomposition([0.3, 1], band_mean=True)(
        tensor, n_spatial_dims=2
    )

    for band, mask in zip(bands.values(), masks, strict=True):
        assert band.dtype == expected_dtype
        assert band.device == tensor.device
        expected = (
            coefficients * mask.reshape(1, 1, 3, 4, 1, 1) / math.sqrt(int(mask.sum()))
        )
        torch.testing.assert_close(band.cdouble(), expected, rtol=1e-5, atol=1e-6)
    torch.stack([band.abs().sum() for band in bands.values()]).sum().backward()
    assert tensor.grad is not None
    assert tensor.grad.dtype == tensor.dtype
    assert torch.isfinite(tensor.grad).all()
    assert torch.count_nonzero(tensor.grad) > 0


@pytest.mark.parametrize("reduction", ["mean", "none"])
@pytest.mark.parametrize("band_mean", [False, True])
def test_field_bands_and_full_fourier_composition_matches_independent_reference(
    reduction, band_mean
):
    generator = torch.Generator().manual_seed(10)
    preds = torch.randn(
        2, 2, 3, 4, 2, 3, dtype=torch.float64, generator=generator, requires_grad=True
    )
    targets = torch.randn(2, 2, 3, 4, 2, dtype=torch.float64, generator=generator)
    score = CoefficientCRPSLoss(alpha=0.95, reduction=reduction)
    radial = TransformedLoss(
        RadialFourierDecomposition([0.3, 1], band_mean=band_mean),
        objective=ComponentObjective(
            [ComponentBranch(score, component_weights={"band_0": 1, "band_1": 3})]
        ),
    )
    loss = WeightedSumLoss(
        [
            TransformedLoss(IdentityDecomposition(), score),
            radial,
            TransformedLoss(FourierDecomposition(), score),
        ],
        weights=[0.2, 0.3, 0.5],
    )
    pred_coefficients = _explicit_dft(preds, (2, 3), "ortho")
    target_coefficients = _explicit_dft(targets, (2, 3), "ortho")
    band_scores = []
    for mask in _reference_masks((3, 4), [0.3, 1]):
        scale = math.sqrt(int(mask.sum())) if band_mean else 1
        band_scores.append(
            _reference_score(
                pred_coefficients * mask.reshape(1, 1, 3, 4, 1, 1) / scale,
                target_coefficients * mask.reshape(1, 1, 3, 4, 1) / scale,
                0.95,
            )
        )
    expected = (
        0.2 * _reference_score(preds, targets, 0.95)
        + 0.3 * (band_scores[0] + 3 * band_scores[1]) / 4
        + 0.5 * _reference_score(pred_coefficients, target_coefficients, 0.95)
    )
    if reduction == "mean":
        expected = expected.mean()

    actual = loss(preds, targets)

    assert actual.shape == (targets.shape if reduction == "none" else ())
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual.mean().backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()
    assert torch.count_nonzero(preds.grad) > 0


def test_band_mean_does_not_equalize_coefficient_crps_over_zeroed_positions():
    preds = torch.zeros(1, 1, 8, 1, 2, dtype=torch.float64)
    preds[:, :, 0] = math.sqrt(8)
    targets = torch.zeros(1, 1, 8, 1, dtype=torch.float64)
    score = CoefficientCRPSLoss(alpha=0.95)
    values = []
    for band_mean in (False, True):
        decomposition = RadialFourierDecomposition([0.1, 0.5], band_mean=band_mean)
        pred_bands = decomposition(preds, n_spatial_dims=1)
        target_bands = decomposition(targets, n_spatial_dims=1)
        values.append(
            [score(pred_bands[name], target_bands[name]) for name in pred_bands]
        )

    torch.testing.assert_close(
        torch.stack(values[0]), torch.tensor([1 / 8, 7 / 8], dtype=torch.float64)
    )
    torch.testing.assert_close(
        torch.stack(values[1]),
        torch.tensor([1 / 8, math.sqrt(7) / 8], dtype=torch.float64),
    )
    assert values[1][1] / values[1][0] > 2


def test_full_partition_averages_coefficient_scores_over_every_band():
    generator = torch.Generator().manual_seed(20)
    preds = torch.randn(1, 2, 3, 4, 2, 3, dtype=torch.float64, generator=generator)
    targets = torch.randn(1, 2, 3, 4, 2, dtype=torch.float64, generator=generator)
    score = CoefficientCRPSLoss(alpha=0.95)
    edges = [0.1, 0.3, 1]

    radial = TransformedLoss(RadialFourierDecomposition(edges), score)(preds, targets)
    full = TransformedLoss(FourierDecomposition(), score)(preds, targets)

    torch.testing.assert_close(radial, full / len(edges), rtol=1e-12, atol=1e-12)


def test_radial_coefficient_objective_passes_double_gradcheck():
    generator = torch.Generator().manual_seed(14)
    preds = torch.randn(
        1, 1, 4, 1, 2, dtype=torch.float64, generator=generator, requires_grad=True
    )
    targets = torch.randn(
        1, 1, 4, 1, dtype=torch.float64, generator=generator, requires_grad=True
    )
    loss = TransformedLoss(
        RadialFourierDecomposition([0.25, 0.5], band_mean=True),
        CoefficientCRPSLoss(alpha=0.95),
    )
    assert torch.autograd.gradcheck(loss, (preds, targets))


def test_cpu_autocast_promotes_radial_fft_and_backpropagates():
    preds = torch.arange(48.0).reshape(1, 1, 3, 4, 2, 2).bfloat16().requires_grad_()
    targets = torch.zeros(1, 1, 3, 4, 2, dtype=torch.bfloat16)
    loss = TransformedLoss(
        RadialFourierDecomposition([0.3, 1], band_mean=True),
        CoefficientCRPSLoss(alpha=0.95),
    )
    expected = loss(preds.float(), targets.float())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = loss(preds, targets)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_autocast_keeps_radial_fft_and_gradients_on_device():
    preds = torch.randn(
        1, 2, 3, 5, 2, 3, device="cuda", dtype=torch.float16, requires_grad=True
    )
    targets = torch.randn(1, 2, 3, 5, 2, device="cuda", dtype=torch.float16)
    loss = TransformedLoss(
        RadialFourierDecomposition([0.3, 1], band_mean=True),
        CoefficientCRPSLoss(alpha=0.95),
    )
    with torch.autocast("cuda", dtype=torch.float16):
        actual = loss(preds, targets)
    assert actual.dtype == torch.float32
    assert actual.device == preds.device
    actual.backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()


@pytest.mark.parametrize(
    ("edges", "match"),
    [
        ([], "at least one band edge"),
        ([0.3, 0.1], "must increase"),
        ([0.2, 0.2], "must increase"),
        ([0.0, 0.3], "positive"),
        ([-0.1], "positive"),
        ([float("nan")], "finite"),
        ([float("inf")], "finite"),
        ([float("-inf")], "finite"),
        ([0.1, float("nan")], "finite"),
        ([0.1, float("inf")], "finite"),
    ],
)
def test_radial_fourier_rejects_invalid_edges(edges, match):
    with pytest.raises(ValueError, match=match):
        RadialFourierDecomposition(edges)


def test_radial_fourier_rejects_invalid_normalization():
    with pytest.raises(ValueError, match="Unknown Fourier normalization"):
        RadialFourierDecomposition([0.5], norm="none")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("n_spatial_dims", [0, -1, 3])
def test_radial_fourier_rejects_invalid_spatial_rank(n_spatial_dims):
    with pytest.raises(ValueError, match="Invalid n_spatial_dims"):
        RadialFourierDecomposition([0.5])(
            torch.ones(1, 1, 3, 2), n_spatial_dims=n_spatial_dims
        )
