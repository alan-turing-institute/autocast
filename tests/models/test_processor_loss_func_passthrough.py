"""An explicit ``loss_func=None`` must survive into the model.

Regression test: ``ProcessorModel`` used to substitute ``MSELoss`` for a None,
which made ``ProcessorModelEnsemble.loss``'s ``loss_func is None`` branch
unreachable -- so a processor that owns its objective was silently trained on
mean-squared error instead of its own loss.
"""

import torch
from torch import nn

from autocast.models.processor import ProcessorModel


class _OwnLossProcessor(nn.Module):
    """Stand-in for a processor that computes its own objective."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def loss(self, batch: object) -> torch.Tensor:  # noqa: ARG002
        return self.linear.weight.sum()


def test_explicit_none_loss_func_is_not_replaced_by_mse():
    model = ProcessorModel(processor=_OwnLossProcessor(), loss_func=None)
    assert model.loss_func is None


def test_a_supplied_loss_func_is_kept():
    loss_func = nn.L1Loss()
    model = ProcessorModel(processor=_OwnLossProcessor(), loss_func=loss_func)
    assert model.loss_func is loss_func
