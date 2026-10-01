from copy import deepcopy
from typing import cast

import pytest
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel

from autocast.losses import MCDropoutMSEL2Loss


class _ToyProcessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.norm = nn.LayerNorm(2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(inputs))


class _Parent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.processor = _make_processor()
        self.loss_func = MCDropoutMSEL2Loss(self.processor)


def _make_processor() -> _ToyProcessor:
    processor = _ToyProcessor()
    with torch.no_grad():
        processor.linear.weight.fill_(2.0)
        processor.linear.bias.fill_(100.0)
        processor.norm.weight.fill_(100.0)
        processor.norm.bias.fill_(100.0)
    return processor


def test_mc_dropout_mse_l2_has_fixed_weight_scale():
    processor = _make_processor()
    loss_func = MCDropoutMSEL2Loss(
        processor=processor,
        l2_coefficient=1e-3,
    )
    prediction = torch.zeros(3, 2)
    targets = torch.ones_like(prediction)

    loss = loss_func(prediction, targets)

    # MSE = 1. The 2x2 matrix has sum(W**2) = 16; biases and one-dimensional
    # normalization parameters are excluded.
    assert loss.item() == pytest.approx(1.0 + 16.0e-3)
    assert loss_func.l2_penalty(loss).item() == pytest.approx(16.0e-3)


def test_mc_dropout_mse_l2_backpropagates_through_processor_weights():
    processor = _make_processor()
    loss_func = MCDropoutMSEL2Loss(processor, l2_coefficient=1e-3)

    loss = loss_func(torch.zeros(2, 2), torch.ones(2, 2))
    loss.backward()

    assert processor.linear.weight.grad is not None
    assert torch.isfinite(processor.linear.weight.grad).all()
    assert torch.count_nonzero(processor.linear.weight.grad) > 0
    assert processor.linear.bias.grad is None
    assert processor.norm.weight.grad is None


def test_mc_dropout_mse_l2_uses_plain_mse_outside_training():
    processor = _make_processor()
    loss_func = MCDropoutMSEL2Loss(processor, l2_coefficient=1e-3)
    loss_func.eval()
    prediction = torch.zeros(3, 2)
    targets = torch.ones_like(prediction)

    loss = loss_func(prediction, targets)

    assert loss.item() == pytest.approx(1.0)


def test_mc_dropout_mse_l2_does_not_register_processor_twice():
    parent = _Parent()

    state_keys = list(parent.state_dict())

    assert any(key.startswith("processor.") for key in state_keys)
    assert not any(key.startswith("loss_func.processor.") for key in state_keys)
    assert list(parent.loss_func.parameters()) == []


def test_mc_dropout_mse_l2_rebinds_processor_in_deepcopy():
    parent = _Parent()

    copied = deepcopy(parent)

    assert copied.processor is not parent.processor
    assert copied.loss_func.processor is copied.processor
    assert copied.loss_func.processor is not parent.processor


def test_mc_dropout_mse_l2_binds_ema_processor_copy():
    parent = _Parent()

    averaged = AveragedModel(parent)
    averaged_parent = cast(_Parent, averaged.module)

    assert averaged_parent.loss_func.processor is averaged_parent.processor
    assert averaged_parent.loss_func.processor is not parent.processor
    assert not any(
        key.startswith("module.loss_func.processor.") for key in averaged.state_dict()
    )


def test_mc_dropout_mse_l2_ignores_frozen_matrix_weights():
    processor = nn.Linear(2, 1)
    processor.weight.requires_grad_(False)
    loss_func = MCDropoutMSEL2Loss(processor, l2_coefficient=1.0)

    prior = loss_func.l2_penalty(torch.zeros(()))

    assert prior.shape == ()
    assert prior.item() == 0.0


def test_mc_dropout_mse_l2_rejects_negative_coefficient():
    with pytest.raises(ValueError, match="l2_coefficient"):
        MCDropoutMSEL2Loss(nn.Linear(1, 1), l2_coefficient=-1e-5)
