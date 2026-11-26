import os
import subprocess
import sys
import textwrap

import pytest
import torch

from autocast.losses import (
    AlphaFairCRPSLoss,
    AlphaFairEnergyScoreLoss,
    CoefficientCRPSLoss,
    CRPSLoss,
    EnsembleMAELoss,
    FairCRPSLoss,
)


def test_ensemble_mae_loss_uses_ensemble_mean():
    preds = torch.tensor(
        [
            [[[[[0.0, 2.0], [2.0, 4.0]]]]],
        ]
    )
    targets = torch.tensor(
        [
            [[[[1.0, 2.0]]]],
        ]
    )

    loss = EnsembleMAELoss(reduction="none")(preds, targets)

    expected = torch.tensor(
        [
            [[[[0.0, 1.0]]]],
        ]
    )
    assert torch.allclose(loss, expected)


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [("mean", 0.5), ("sum", 1.0)],
)
def test_ensemble_mae_loss_reductions(reduction, expected):
    preds = torch.tensor(
        [
            [[[[[0.0, 2.0], [2.0, 4.0]]]]],
        ]
    )
    targets = torch.tensor(
        [
            [[[[1.0, 2.0]]]],
        ]
    )

    loss = EnsembleMAELoss(reduction=reduction)(preds, targets)

    assert torch.isclose(loss, torch.tensor(expected))


def test_ensemble_mae_loss_rejects_targets_with_member_dim():
    preds = torch.ones((1, 1, 1, 2, 3))
    targets = torch.ones((1, 1, 1, 2, 3))

    with pytest.raises(
        ValueError, match="Targets should not have the ensemble dimension"
    ):
        EnsembleMAELoss()(preds, targets)


@pytest.mark.parametrize(
    "loss_cls",
    [CRPSLoss, FairCRPSLoss, AlphaFairCRPSLoss, EnsembleMAELoss, CoefficientCRPSLoss],
)
def test_ensemble_loss_rejects_unknown_reduction(loss_cls):
    with pytest.raises(ValueError, match="Unknown reduction"):
        loss_cls(reduction="men")


@pytest.mark.parametrize(
    "loss_cls",
    [CRPSLoss, FairCRPSLoss, AlphaFairCRPSLoss, EnsembleMAELoss, CoefficientCRPSLoss],
)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ensemble_loss_accepts_known_reductions(loss_cls, reduction):
    assert loss_cls(reduction=reduction).reduction == reduction


def test_alpha_fair_crps_integer_alpha_with_runtime_typechecking():
    script = textwrap.dedent(
        """
        import torch

        from autocast.losses import AlphaFairCRPSLoss, CRPSLoss, FairCRPSLoss

        predictions = torch.tensor(
            [[[[[0.0, 1.0, 3.0], [1.0, 2.0, 5.0]]]]], requires_grad=True
        )
        targets = torch.tensor([[[[2.0, 1.5]]]])
        for alpha, endpoint in ((0, CRPSLoss), (1, FairCRPSLoss)):
            loss = AlphaFairCRPSLoss(alpha=alpha)(predictions, targets)
            expected = AlphaFairCRPSLoss(alpha=float(alpha))(predictions, targets)
            torch.testing.assert_close(loss, expected, rtol=0, atol=0)
            torch.testing.assert_close(loss, endpoint()(predictions, targets))
            grad, = torch.autograd.grad(loss, predictions)
            expected_grad, = torch.autograd.grad(expected, predictions)
            assert torch.isfinite(grad).all()
            torch.testing.assert_close(grad, expected_grad, rtol=0, atol=0)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "RUNTIME_TYPECHECKING": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("alpha", [0, 0.95, 1])
def test_alpha_fair_energy_score_matches_explicit_joint_distances(alpha):
    generator = torch.Generator().manual_seed(19)
    targets = torch.randn(2, 1, 4, 3, 3, generator=generator, dtype=torch.float64)
    preds = torch.randn(
        *targets.shape, 8, generator=generator, dtype=torch.float64, requires_grad=True
    )
    vectors = preds.flatten(1, -2).transpose(-1, -2)
    truth = targets.flatten(1)
    distance_to_truth = (vectors - truth[:, None]).norm(dim=-1).mean(dim=-1)
    first, second = torch.triu_indices(8, 8, offset=1)
    between_members = (vectors[:, first] - vectors[:, second]).norm(dim=-1).sum(dim=-1)
    weight = (1 - alpha) / 8**2 + alpha / (8 * 7)
    expected = (distance_to_truth - weight * between_members) / truth.shape[-1] ** 0.5
    actual = AlphaFairEnergyScoreLoss(alpha=alpha, reduction="none")(preds, targets)

    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected.sum(), preds, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual.sum(), preds)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_alpha_fair_energy_score_zero_distances_in_low_precision():
    targets = torch.ones((2, 1, 3, 4, 3), dtype=torch.bfloat16)
    preds = targets[..., None].repeat(1, 1, 1, 1, 1, 8).requires_grad_()
    loss = AlphaFairEnergyScoreLoss()(preds, targets)

    torch.testing.assert_close(loss, torch.zeros((), dtype=torch.float32))
    loss.backward()
    torch.testing.assert_close(preds.grad, torch.zeros_like(preds))
