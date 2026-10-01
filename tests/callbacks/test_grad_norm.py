from typing import Any, cast

import pytest
import torch

from autocast.callbacks.grad_norm import GradNormCallback


class _LoggedModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.logged: dict[str, torch.Tensor | float] = {}

    def log_dict(self, values: dict[str, torch.Tensor | float], **_: object) -> None:
        self.logged.update(values)


def test_grad_norm_callback_logs_pre_clip_norms_and_lr() -> None:
    module = _LoggedModule()
    module.weight.grad = torch.tensor([3.0, 4.0])
    module.bias.grad = torch.tensor([12.0])
    optimizer = torch.optim.SGD(module.parameters(), lr=0.125)

    GradNormCallback().on_before_optimizer_step(
        trainer=cast(Any, None),
        pl_module=cast(Any, module),
        optimizer=optimizer,
    )

    assert torch.as_tensor(
        module.logged["grad_2.0_norm_total"]
    ).item() == pytest.approx(13.0)
    assert torch.as_tensor(module.logged["grad_2.0_norm_max"]).item() == pytest.approx(
        12.0
    )
    assert module.logged["lr"] == pytest.approx(0.125)
    assert "grad_2.0_norm/weight" not in module.logged


def test_grad_norm_callback_noops_without_gradients() -> None:
    module = _LoggedModule()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.125)

    GradNormCallback().on_before_optimizer_step(
        trainer=cast(Any, None),
        pl_module=cast(Any, module),
        optimizer=optimizer,
    )

    assert module.logged == {}


def test_grad_norm_callback_logs_per_param_when_enabled() -> None:
    module = _LoggedModule()
    module.weight.grad = torch.tensor([3.0, 4.0])
    module.bias.grad = torch.tensor([12.0])
    optimizer = torch.optim.SGD(module.parameters(), lr=0.125)

    GradNormCallback(log_per_param=True).on_before_optimizer_step(
        trainer=cast(Any, None),
        pl_module=cast(Any, module),
        optimizer=optimizer,
    )

    assert torch.as_tensor(
        module.logged["grad_2.0_norm/weight"]
    ).item() == pytest.approx(5.0)
    assert torch.as_tensor(module.logged["grad_2.0_norm/bias"]).item() == pytest.approx(
        12.0
    )
    assert torch.as_tensor(
        module.logged["grad_2.0_norm_total"]
    ).item() == pytest.approx(13.0)
