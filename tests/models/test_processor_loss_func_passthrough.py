"""An explicit ``loss_func=None`` must survive into the model.

Regression test: ``ProcessorModel`` used to substitute ``MSELoss`` for a None,
which made ``ProcessorModelEnsemble.loss``'s ``loss_func is None`` branch
unreachable -- so a processor that owns its objective was silently trained on
mean-squared error instead of its own loss.
"""

from typing import Any

from torch import nn

from autocast.models.processor import ProcessorModel
from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor


class OwnLossProcessor(Processor):
    """Stand-in for a processor that computes its own objective."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.linear = nn.Linear(4, 4)

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:  # noqa: ARG002
        return self.linear(x)

    def loss(self, batch: EncodedBatch) -> Tensor:
        preds = self.map(batch.encoded_inputs, batch.global_cond)
        return nn.functional.mse_loss(preds, batch.encoded_output_fields)


def test_explicit_none_loss_func_is_not_replaced_by_mse():
    model = ProcessorModel(processor=OwnLossProcessor(), loss_func=None)
    assert model.loss_func is None


def test_a_supplied_loss_func_is_kept():
    loss_func = nn.L1Loss()
    model = ProcessorModel(processor=OwnLossProcessor(), loss_func=loss_func)
    assert model.loss_func is loss_func
