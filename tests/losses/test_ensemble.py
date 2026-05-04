import pytest
import torch

from autocast.losses import EnsembleMAELoss


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
