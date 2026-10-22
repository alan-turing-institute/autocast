"""Check finite-member spectral moments against independent population scores."""

from itertools import combinations, product

import pytest
import torch
from einops import rearrange, repeat

from autocast.losses.moments import SpectralMomentLoss


def _pairwise_components(preds, targets, norm="ortho"):
    spatial = tuple(range(2, targets.ndim - 1))
    z = torch.fft.fftn(preds, dim=spatial, norm=norm)
    y = torch.fft.fftn(targets, dim=spatial, norm=norm)
    errors = z - y.unsqueeze(-1)
    power_errors = z.abs().square() - y.abs().square().unsqueeze(-1)
    pairs = list(combinations(range(preds.shape[-1]), 2))
    mean_score = torch.stack(
        [(errors[..., i] * errors[..., j].conj()).real for i, j in pairs]
    ).mean()
    power_score = torch.stack(
        [power_errors[..., i] * power_errors[..., j] for i, j in pairs]
    ).mean()
    return mean_score, power_score


@pytest.mark.parametrize("members", [2, 8])
@pytest.mark.parametrize("spatial_shape", [(5,), (3, 4), (2, 3, 4)])
@pytest.mark.parametrize("norm", ["ortho", "forward", "backward"])
def test_values_and_gradients_equal_pairwise_u_statistics(members, spatial_shape, norm):
    rng = torch.Generator().manual_seed(37)
    targets = torch.randn(2, 3, *spatial_shape, 2, dtype=torch.float64, generator=rng)
    preds = torch.randn(*targets.shape, members, dtype=torch.float64, generator=rng)
    preds.requires_grad_()
    targets.requires_grad_()
    score = SpectralMomentLoss(norm=norm)
    actual = score.components(preds, targets)
    expected = _pairwise_components(preds, targets, norm)
    for value, reference in zip(actual, expected, strict=True):
        assert value.ndim == 0
        torch.testing.assert_close(value, reference, rtol=1e-11, atol=1e-11)
    total = score(preds, targets)
    torch.testing.assert_close(total, sum(actual))
    actual_gradients = torch.autograd.grad(total, (preds, targets))
    expected_gradients = torch.autograd.grad(
        expected[0] + expected[1], (preds, targets)
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-10, atol=1e-10
        )


@pytest.mark.parametrize("members", [2, 3, 8])
@pytest.mark.parametrize("matched", [False, True])
def test_exact_finite_member_expectations_and_population_gradients(members, matched):
    # Exhaust every equally likely IID member/target combination. The
    # asymmetric spatial pattern produces both real and complex FFT modes.
    signs = torch.tensor(list(product([-1.0, 1.0], repeat=members + 1)))
    signs = signs.to(torch.float64)
    bias = torch.tensor([1.0, 0.3, -0.4], dtype=torch.float64)
    pattern = torch.tensor([0.7, -1.2, 0.2], dtype=torch.float64)
    truth_location, truth_amplitude = 0.2, 1.1
    location = torch.tensor(
        truth_location if matched else 0.6, dtype=torch.float64, requires_grad=True
    )
    amplitude = torch.tensor(
        truth_amplitude if matched else 0.7, dtype=torch.float64, requires_grad=True
    )
    preds = location * rearrange(bias, "x -> 1 1 x 1 1") + amplitude * (
        rearrange(signs[:, :-1], "b m -> b 1 1 1 m")
        * rearrange(pattern, "x -> 1 1 x 1 1")
    )
    targets = truth_location * rearrange(bias, "x -> 1 1 x 1") + truth_amplitude * (
        rearrange(signs[:, -1], "b -> b 1 1 1") * rearrange(pattern, "x -> 1 1 x 1")
    )
    z_population = torch.fft.fft(
        torch.stack(
            [
                location * bias - amplitude * pattern,
                location * bias + amplitude * pattern,
            ]
        ),
        norm="ortho",
    )
    y_population = torch.fft.fft(
        torch.stack(
            [
                truth_location * bias - truth_amplitude * pattern,
                truth_location * bias + truth_amplitude * pattern,
            ]
        ),
        norm="ortho",
    )
    mean_z, mean_y = z_population.mean(0), y_population.mean(0)
    qz, qy = z_population.abs().square(), y_population.abs().square()
    expected = (
        (
            (mean_z - mean_y).abs().square()
            + (y_population - mean_y).abs().square().mean(0)
        ).mean(),
        ((qz.mean(0) - qy.mean(0)).square() + qy.var(0, correction=0)).mean(),
    )
    actual = SpectralMomentLoss().components(preds, targets)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=1e-12, atol=1e-12)
        actual_gradients = torch.autograd.grad(
            value, (location, amplitude), retain_graph=True
        )
        expected_gradients = torch.autograd.grad(
            reference, (location, amplitude), retain_graph=True
        )
        for gradient, expected_gradient in zip(
            actual_gradients, expected_gradients, strict=True
        ):
            torch.testing.assert_close(
                gradient, expected_gradient, rtol=1e-11, atol=1e-11
            )
            if matched:
                torch.testing.assert_close(
                    gradient, torch.zeros_like(gradient), atol=1e-11, rtol=0
                )


@pytest.mark.parametrize("mode", ["dc", "diagonal_nyquist"])
def test_dc_and_diagonal_nyquist_are_scored_without_band_scaling(mode):
    field = torch.ones(4, 6, dtype=torch.float64)
    if mode == "diagonal_nyquist":
        field[::2] *= -1
        field[:, ::2] *= -1
    preds = repeat(field, "x y -> 1 1 x y 1 8")
    targets = torch.zeros(1, 1, 4, 6, 1, dtype=torch.float64)
    mean_score, power_score = SpectralMomentLoss().components(preds, targets)
    torch.testing.assert_close(mean_score, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(power_score, torch.tensor(24.0, dtype=torch.float64))


def test_scores_are_formed_before_averaging_inputs_channels_or_leads():
    # Opposite errors must not cancel in a pooled field or pooled raw power.
    targets = torch.tensor([0.0, 2.0], dtype=torch.float64)
    targets = rearrange(targets, "b -> b 1 1 1")
    preds = torch.ones(2, 1, 1, 1, 8, dtype=torch.float64)
    score = SpectralMomentLoss()
    expected = (
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(5.0, dtype=torch.float64),
    )
    for prediction, target in (
        (preds, targets),
        (preds.transpose(0, 1), targets.transpose(0, 1)),
        (preds.transpose(0, -2), targets.transpose(0, -1)),
    ):
        for value, reference in zip(
            score.components(prediction, target), expected, strict=True
        ):
            torch.testing.assert_close(value, reference)


@pytest.mark.parametrize("value", [0.0, 1.7])
def test_coincident_members_and_target_have_zero_scores_and_gradients(value):
    targets = torch.full((2, 1, 3, 2), value, dtype=torch.float64, requires_grad=True)
    preds = torch.full(
        (*targets.shape, 8), value, dtype=torch.float64, requires_grad=True
    )
    score = SpectralMomentLoss()
    for component in score.components(preds, targets):
        assert component.item() == 0
    gradients = torch.autograd.grad(score(preds, targets), (preds, targets))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize(
    ("values", "target", "expected"),
    [([-1.0, 1.0], 0.0, (-1.0, 1.0)), ([0.0, 2.0], 2**0.5, (2 - 2 * 2**0.5, -4.0))],
)
def test_negative_unbiased_estimates_are_not_clipped(values, target, expected):
    preds = rearrange(torch.tensor(values, dtype=torch.float64), "m -> 1 1 1 1 m")
    targets = torch.full((1, 1, 1, 1), target, dtype=torch.float64)
    score = SpectralMomentLoss()
    for component, reference in zip(
        score.components(preds, targets), expected, strict=True
    ):
        assert component.item() == pytest.approx(reference, abs=1e-12)
    assert score(preds, targets).item() == pytest.approx(sum(expected), abs=1e-12)


@pytest.mark.parametrize(
    ("prediction_dtype", "target_dtype", "result_dtype"),
    [
        (torch.float16, torch.float16, torch.float32),
        (torch.bfloat16, torch.bfloat16, torch.float32),
        (torch.float32, torch.float32, torch.float32),
        (torch.float64, torch.float64, torch.float64),
        (torch.float32, torch.float64, torch.float64),
        (torch.float64, torch.float32, torch.float64),
    ],
)
def test_real_dtypes_are_promoted_before_fourier_and_power_scores(
    prediction_dtype, target_dtype, result_dtype
):
    rng = torch.Generator().manual_seed(38)
    preds = (
        torch.randn(2, 1, 3, 2, 8, generator=rng).to(prediction_dtype).requires_grad_()
    )
    targets = torch.randn(2, 1, 3, 2, generator=rng).to(target_dtype)
    score = SpectralMomentLoss()
    actual = score(preds, targets)
    expected = score(preds.to(result_dtype), targets.to(result_dtype))
    assert actual.dtype == result_dtype
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex64])
@pytest.mark.parametrize("invalid_prediction", [False, True])
def test_rejects_nonreal_or_nonfloating_inputs(dtype, invalid_prediction):
    preds = torch.zeros(1, 1, 3, 1, 2)
    targets = torch.zeros(1, 1, 3, 1)
    if invalid_prediction:
        preds = preds.to(dtype)
    else:
        targets = targets.to(dtype)
    with pytest.raises(ValueError, match="real floating"):
        SpectralMomentLoss()(preds, targets)


@pytest.mark.parametrize(
    ("prediction_shape", "target_shape"),
    [
        ((1, 1, 3, 1), (1, 1, 3, 1)),
        ((1, 1, 3, 1, 1), (1, 1, 3, 1)),
        ((1, 1, 3, 1, 0), (1, 1, 3, 1)),
        ((1, 1, 4, 1, 2), (1, 1, 3, 1)),
        ((1, 1, 1, 2), (1, 1, 1)),
        ((1, 1, 0, 1, 2), (1, 1, 0, 1)),
        ((0, 1, 3, 1, 2), (0, 1, 3, 1)),
        ((1, 1, 3, 0, 2), (1, 1, 3, 0)),
    ],
)
def test_rejects_mismatched_empty_or_nonensemble_shapes(prediction_shape, target_shape):
    with pytest.raises(ValueError, match=r"(Predictions|Spectral)"):
        SpectralMomentLoss()(torch.zeros(prediction_shape), torch.zeros(target_shape))


def test_rejects_mismatched_devices_before_fft():
    with pytest.raises(ValueError, match="device"):
        SpectralMomentLoss()(
            torch.zeros(1, 1, 3, 1, 2), torch.zeros(1, 1, 3, 1, device="meta")
        )


def test_invalid_normalization_is_rejected():
    with pytest.raises(ValueError, match="normalization"):
        SpectralMomentLoss(norm="invalid")  # pyright: ignore[reportArgumentType]


def test_stateless_loss_adds_no_persistent_checkpoint_keys():
    loss = SpectralMomentLoss()
    assert loss.requires_ambient_predictions
    assert not loss.state_dict()
    assert not list(loss.parameters())
    assert not list(loss.buffers())
    loss(torch.randn(2, 1, 3, 2, 8), torch.randn(2, 1, 3, 2))
    assert not loss.state_dict()
    loss.load_state_dict({}, strict=True)
