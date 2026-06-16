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

    def log(self, name: str, value: torch.Tensor | float, **_: object) -> None:
        self.logged[name] = value


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

    assert torch.as_tensor(module.logged["grad_norm"]).item() == pytest.approx(13.0)
    assert torch.as_tensor(module.logged["grad_norm_max"]).item() == pytest.approx(12.0)
    assert module.logged["lr"] == pytest.approx(0.125)


def test_grad_norm_callback_noops_without_gradients() -> None:
    module = _LoggedModule()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.125)

    GradNormCallback().on_before_optimizer_step(
        trainer=cast(Any, None),
        pl_module=cast(Any, module),
        optimizer=optimizer,
    )

    assert module.logged == {}
