import math

import pytest
import torch

from autocast.losses.ensemble import (
    AlphaFairCRPSLoss,
    CoefficientCRPSLoss,
    CRPSLoss,
    FairCRPSLoss,
)


def _distance(value):
    return torch.linalg.vector_norm(torch.view_as_real(value), dim=-1)


def _reference_score(preds, targets, alpha):
    """Independent double-precision distances and mixture of endpoint scores."""
    preds = preds.to(torch.complex128)
    targets = targets.to(torch.complex128)
    members = preds.shape[-1]
    truth = torch.stack(
        [_distance(preds[..., i] - targets) for i in range(members)]
    ).mean(dim=0)
    if members == 1:
        return truth
    pairs = torch.stack(
        [
            _distance(preds[..., i] - preds[..., j])
            for i in range(members)
            for j in range(i + 1, members)
        ]
    ).sum(dim=0)
    plug_in = truth - pairs / members**2
    fair = truth - pairs / (members * (members - 1))
    return (1 - alpha) * plug_in + alpha * fair


@pytest.mark.parametrize("alpha", [0, 0.25, 0.95, 1])
@pytest.mark.parametrize("members", [2, 3, 7])
@pytest.mark.parametrize("spatial_shape", [(5,), (3, 5), (2, 3, 5)])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_score_matches_independent_distances(alpha, members, spatial_shape, dtype):
    generator = torch.Generator().manual_seed(24)
    target_shape = (2, 3, *spatial_shape, 4)
    preds = torch.randn(*target_shape, members, dtype=dtype, generator=generator)
    targets = torch.randn(*target_shape, dtype=dtype, generator=generator)

    actual = CoefficientCRPSLoss(alpha, reduction="none")(preds, targets)
    expected = _reference_score(preds, targets, alpha)

    assert actual.shape == targets.shape
    assert not actual.is_complex()
    single_precision = dtype in (torch.float32, torch.complex64)
    tolerance = (1e-5, 1e-6) if single_precision else (1e-12, 1e-12)
    torch.testing.assert_close(
        actual, expected.to(actual.dtype), rtol=tolerance[0], atol=tolerance[1]
    )


@pytest.mark.parametrize("alpha", [0, 0.95, 1])
def test_hand_calculated_fairness_endpoints(alpha):
    preds = torch.tensor([-1.0, 1.0], dtype=torch.float64).reshape(1, 1, 1, 1, 2)
    targets = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
    actual = CoefficientCRPSLoss(alpha)(preds, targets)
    assert actual.item() == pytest.approx((1 - alpha) / 2, abs=1e-15)


def test_complex_distance_retains_phase_and_does_not_sum_marginal_scores():
    preds = torch.full((1, 1, 1, 1, 2), 1j, dtype=torch.complex128)
    targets = torch.ones(1, 1, 1, 1, dtype=torch.complex128)
    actual = CoefficientCRPSLoss(alpha=0.95)(preds, targets)
    assert actual.item() == pytest.approx(math.sqrt(2), abs=1e-12)


def test_channels_are_scored_independently():
    preds = torch.tensor([3 + 0j, 4j], dtype=torch.complex128).reshape(1, 1, 1, 2, 1)
    targets = torch.zeros(1, 1, 1, 2, dtype=torch.complex128)
    assert CoefficientCRPSLoss()(preds, targets).item() == pytest.approx(3.5)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_reductions_include_every_remaining_axis(reduction):
    preds = torch.arange(48.0).reshape(2, 2, 3, 2, 2).to(torch.complex128)
    targets = torch.ones(2, 2, 3, 2, dtype=torch.complex128)
    expected = _reference_score(preds, targets, 0.95)
    if reduction == "mean":
        expected = expected.mean()
    elif reduction == "sum":
        expected = expected.sum()
    torch.testing.assert_close(
        CoefficientCRPSLoss(0.95, reduction)(preds, targets),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "transform", ["translation", "rotation", "conjugation", "scale"]
)
def test_distance_invariances(transform):
    generator = torch.Generator().manual_seed(29)
    preds = torch.randn(2, 3, 4, 2, 5, dtype=torch.complex128, generator=generator)
    targets = torch.randn(2, 3, 4, 2, dtype=torch.complex128, generator=generator)
    score = CoefficientCRPSLoss(0.95, reduction="none")
    expected = score(preds, targets)
    if transform == "translation":
        shift = targets * (2 + 3j)
        preds, targets = preds + shift.unsqueeze(-1), targets + shift
    elif transform == "rotation":
        factor = complex(math.cos(0.7), math.sin(0.7))
        preds, targets = preds * factor, targets * factor
    elif transform == "conjugation":
        preds, targets = preds.conj(), targets.conj()
    else:
        preds, targets = preds * -2.5, targets * -2.5
        expected = expected * 2.5
    torch.testing.assert_close(score(preds, targets), expected, rtol=1e-12, atol=1e-12)


def test_member_permutation_preserves_values_and_permutes_gradients():
    preds = torch.tensor([1 + 1j, 1 + 1j, -2j], dtype=torch.complex128)
    preds = preds.reshape(1, 1, 1, 1, 3).requires_grad_()
    targets = torch.zeros(1, 1, 1, 1, dtype=torch.complex128)
    order = torch.tensor([2, 0, 1])
    shuffled = preds.detach()[..., order].requires_grad_()
    score = CoefficientCRPSLoss(0.95)
    original, permuted = score(preds, targets), score(shuffled, targets)
    original_grad = torch.autograd.grad(original, preds)[0]
    permuted_grad = torch.autograd.grad(permuted, shuffled)[0]
    torch.testing.assert_close(permuted, original, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        permuted_grad, original_grad[..., order], rtol=1e-9, atol=1e-10
    )


@pytest.mark.parametrize("alpha", [0, 0.95, 1])
@pytest.mark.parametrize("value", [0j, 3 + 4j])
def test_coincident_members_and_target_have_zero_value_and_gradient(alpha, value):
    preds = torch.full(
        (1, 1, 2, 1, 3), value, dtype=torch.complex128, requires_grad=True
    )
    targets = torch.full(
        (1, 1, 2, 1), value, dtype=torch.complex128, requires_grad=True
    )
    result = CoefficientCRPSLoss(alpha)(preds, targets)
    gradients = torch.autograd.grad(result, (preds, targets))
    assert result.item() == 0
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize(
    ("dtype", "separation"), [(torch.complex64, 1e-5), (torch.complex128, 1e-10)]
)
def test_small_separations_keep_unsmoothed_modulus_gradients(dtype, separation):
    preds = torch.tensor([0j, complex(separation, 0)], dtype=dtype)
    preds = preds.reshape(1, 1, 1, 1, 2).requires_grad_()
    targets = torch.ones(1, 1, 1, 1, dtype=dtype)
    gradient = torch.autograd.grad(CoefficientCRPSLoss(0.95)(preds, targets), preds)[0]
    expected = torch.tensor([-0.0125 + 0j, -0.9875 + 0j], dtype=dtype).reshape_as(preds)
    torch.testing.assert_close(gradient, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("separation", [1e-3, 1e-8, 1e-12])
def test_small_distances_are_not_hidden_by_absolute_tolerances(separation):
    preds = torch.tensor([-separation, separation], dtype=torch.complex128)
    preds = preds.reshape(1, 1, 1, 1, 2)
    targets = torch.zeros(1, 1, 1, 1, dtype=torch.complex128)
    normalized = CoefficientCRPSLoss(0.95)(preds, targets) / separation
    assert normalized.item() == pytest.approx(0.025, rel=1e-12, abs=1e-14)


def test_complex_gradcheck_away_from_coincidences():
    preds = torch.tensor([-2 + 1j, 1 + 2j, 3 - 1j], dtype=torch.complex128)
    preds = preds.reshape(1, 1, 1, 1, 3).requires_grad_()
    targets = torch.full(
        (1, 1, 1, 1), 0.2 + 0.3j, dtype=torch.complex128, requires_grad=True
    )
    assert torch.autograd.gradcheck(CoefficientCRPSLoss(0.95), (preds, targets))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_gradients_match_independent_distances(dtype):
    generator = torch.Generator().manual_seed(31)
    preds = torch.randn(
        2, 3, 4, 2, 5, dtype=dtype, generator=generator, requires_grad=True
    )
    targets = torch.randn(
        2, 3, 4, 2, dtype=dtype, generator=generator, requires_grad=True
    )
    actual = CoefficientCRPSLoss(0.95)(preds, targets)
    expected = _reference_score(preds, targets, 0.95).mean()
    actual_gradients = torch.autograd.grad(actual, (preds, targets))
    expected_gradients = torch.autograd.grad(expected, (preds, targets))
    rtol, atol = (1e-4, 1e-6) if dtype == torch.complex64 else (1e-9, 1e-10)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=rtol, atol=atol
        )


@pytest.mark.parametrize(
    ("alpha", "real_score"),
    [(0, CRPSLoss()), (0.95, AlphaFairCRPSLoss()), (1, FairCRPSLoss())],
)
@pytest.mark.parametrize("tied", [False, True])
def test_real_values_match_existing_scores_and_gradients_match_away_from_ties(
    alpha, real_score, tied
):
    members = [-2.0, -2.0, 3.0] if tied else [-2.0, 0.5, 3.0]
    preds = (
        torch.tensor(members, dtype=torch.float64)
        .reshape(1, 1, 1, 1, 3)
        .requires_grad_()
    )
    targets = torch.full((1, 1, 1, 1), 0.2, dtype=torch.float64)
    actual, expected = (
        CoefficientCRPSLoss(alpha)(preds, targets),
        real_score(preds, targets),
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    if not tied:
        torch.testing.assert_close(
            torch.autograd.grad(actual, preds)[0],
            torch.autograd.grad(expected, preds)[0],
            rtol=1e-9,
            atol=1e-10,
        )


def test_single_member_is_absolute_error_only_at_alpha_zero():
    preds = torch.full(
        (1, 1, 1, 1, 1), 3 + 4j, dtype=torch.complex128, requires_grad=True
    )
    targets = torch.zeros(1, 1, 1, 1, dtype=torch.complex128)
    result = CoefficientCRPSLoss(alpha=0)(preds, targets)
    assert result.item() == 5
    torch.testing.assert_close(
        torch.autograd.grad(result, preds)[0], torch.full_like(preds, 0.6 + 0.8j)
    )
    for alpha in (0.01, 0.95, 1):
        with pytest.raises(ValueError, match="at least 2 members"):
            CoefficientCRPSLoss(alpha)(preds, targets)


@pytest.mark.parametrize(
    "alpha", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")]
)
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="finite and in"):
        CoefficientCRPSLoss(alpha)


@pytest.mark.parametrize("loss_cls", [CRPSLoss, CoefficientCRPSLoss])
def test_invalid_reduction_is_rejected(loss_cls):
    with pytest.raises(ValueError, match="Unknown reduction"):
        loss_cls(reduction="median")


@pytest.mark.parametrize("target_shape", [(1, 1, 1, 1, 3), (1, 1, 1, 2), (1, 1, 1)])
def test_shapes_must_match_exactly_without_broadcasting(target_shape):
    with pytest.raises(ValueError, match="final ensemble axis"):
        CoefficientCRPSLoss()(torch.ones(1, 1, 1, 1, 3), torch.ones(target_shape))


def test_empty_ensemble_is_rejected():
    with pytest.raises(ValueError, match="at least one ensemble member"):
        CoefficientCRPSLoss()(torch.ones(1, 1, 1, 1, 0), torch.ones(1, 1, 1, 1))


@pytest.mark.parametrize("integer_target", [False, True])
def test_integer_inputs_are_rejected(integer_target):
    preds = torch.ones(
        1, 1, 1, 1, 2, dtype=torch.float32 if integer_target else torch.int64
    )
    targets = torch.ones(
        1, 1, 1, 1, dtype=torch.int64 if integer_target else torch.float32
    )
    with pytest.raises(TypeError, match="floating point or complex"):
        CoefficientCRPSLoss()(preds, targets)


def test_device_mismatch_is_rejected_before_computation():
    preds = torch.ones(1, 1, 1, 1, 2)
    targets = torch.ones(1, 1, 1, 1, device="meta")
    with pytest.raises(ValueError, match="same device"):
        CoefficientCRPSLoss()(preds, targets)


@pytest.mark.parametrize(
    ("preds_dtype", "targets_dtype", "output_dtype"),
    [
        (torch.float32, torch.complex128, torch.float64),
        (torch.float16, torch.float64, torch.float64),
        (torch.float16, torch.complex128, torch.float64),
        (torch.bfloat16, torch.complex64, torch.float32),
        (torch.float64, torch.float16, torch.float64),
        (torch.complex128, torch.bfloat16, torch.float64),
    ],
)
def test_native_dtype_promotion_and_real_output(
    preds_dtype, targets_dtype, output_dtype
):
    preds = torch.ones(1, 1, 1, 1, 2, dtype=preds_dtype, requires_grad=True)
    target_value = 1j if targets_dtype.is_complex else 0.25
    targets = torch.full((1, 1, 1, 1), target_value, dtype=targets_dtype)
    result = CoefficientCRPSLoss()(preds, targets)
    assert result.dtype == output_dtype
    expected = _reference_score(preds, targets, 0).mean().to(output_dtype)
    torch.testing.assert_close(result, expected)
    result.backward()
    assert preds.grad is not None
    assert preds.grad.dtype == preds.dtype
    assert torch.isfinite(preds.grad).all()


def test_half_precision_spread_matches_full_precision_without_overflow():
    preds = (1000 * torch.arange(8, dtype=torch.float16)).reshape(1, 1, 1, 1, 8)
    preds.requires_grad_()
    targets = torch.full((1, 1, 1, 1), 3500.0, requires_grad=True)

    actual = CoefficientCRPSLoss(0.95)(preds, targets)
    expected = _reference_score(preds, targets, 0.95).mean()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected.float(), rtol=1e-6, atol=0)
    actual_gradients = torch.autograd.grad(actual, (preds, targets))
    expected_gradients = torch.autograd.grad(expected, (preds, targets))
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-3, atol=1e-6
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("single_member", [False, True])
def test_low_precision_distances_and_gradients_match_full_precision(
    dtype, autocast, single_member
):
    values = [60000.0] if single_member else [-60000.0, 60000.0]
    target_value = -60000.0 if single_member else 0.0
    alpha = 0.0 if single_member else 0.95
    preds = torch.tensor(values, dtype=dtype).reshape(1, 1, 1, 1, -1)
    preds.requires_grad_()
    targets = torch.full((1, 1, 1, 1), target_value, dtype=dtype, requires_grad=True)

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = CoefficientCRPSLoss(alpha)(preds, targets)
    expected = _reference_score(preds, targets, alpha).mean()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected.float(), rtol=1e-5, atol=0)
    actual_gradients = torch.autograd.grad(actual, (preds, targets))
    expected_gradients = torch.autograd.grad(expected, (preds, targets))
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert actual_gradient.dtype == dtype
        torch.testing.assert_close(actual_gradient, expected_gradient)
