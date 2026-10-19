from types import SimpleNamespace
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
    torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)

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


@pytest.mark.parametrize("norm_type", [0.5, 1.0, 2.0, 3.5, float("inf")])
def test_grad_norm_callback_matches_concatenated_gradient(norm_type: float) -> None:
    module = _LoggedModule()
    module.weight.grad = torch.tensor([-3.0, 4.0])
    module.bias.grad = torch.tensor([12.0])
    optimizer = torch.optim.SGD(module.parameters(), lr=0.125)

    GradNormCallback(norm_type=norm_type).on_before_optimizer_step(
        trainer=cast(Any, None), pl_module=cast(Any, module), optimizer=optimizer
    )

    expected = torch.tensor([-3.0, 4.0, 12.0]).norm(norm_type)
    torch.testing.assert_close(module.logged[f"grad_{norm_type}_norm_total"], expected)


@pytest.mark.parametrize(
    ("global_step", "should_log"),
    [
        (0, False),
        (1, False),
        (2, True),
        (3, False),
        (5, True),
        (100, False),
        (101, True),
    ],
)
def test_grad_norm_callback_uses_global_step_cadence(
    global_step: int, should_log: bool
) -> None:
    module = _LoggedModule()
    module.weight.grad = torch.tensor([3.0, 4.0])
    optimizer = torch.optim.SGD(module.parameters(), lr=0.125)

    GradNormCallback(every_n_steps=3).on_before_optimizer_step(
        trainer=cast(Any, SimpleNamespace(global_step=global_step)),
        pl_module=cast(Any, module),
        optimizer=optimizer,
    )

    assert bool(module.logged) == should_log


def test_grad_norm_callback_skips_before_inspecting_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _LoggedModule()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.125)

    def fail_if_called():
        pytest.fail("Skipped diagnostics must not inspect parameter gradients.")

    monkeypatch.setattr(module, "named_parameters", fail_if_called)
    GradNormCallback(every_n_steps=64).on_before_optimizer_step(
        trainer=cast(Any, SimpleNamespace(global_step=128)),
        pl_module=cast(Any, module),
        optimizer=optimizer,
    )
    assert module.logged == {}


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_grad_norm_callback_rejects_invalid_interval(interval: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GradNormCallback(every_n_steps=interval)


@pytest.mark.parametrize("norm_type", [0.0, -1.0])
def test_grad_norm_callback_rejects_invalid_norm(norm_type: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        GradNormCallback(norm_type=norm_type)


def test_grad_norm_callback_promotes_mixed_dtypes_without_host_scalar_reads() -> None:
    module = _LoggedModule()
    module.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    module.weight.grad = torch.tensor([3.0, 4.0])
    module.bias.grad = torch.tensor([12.0], dtype=torch.float64)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.125)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        GradNormCallback().on_before_optimizer_step(
            trainer=cast(Any, None), pl_module=cast(Any, module), optimizer=optimizer
        )

    assert "aten::_local_scalar_dense" not in {
        event.key for event in prof.key_averages()
    }
    torch.testing.assert_close(
        module.logged["grad_2.0_norm_total"], torch.tensor(13.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        module.logged["grad_2.0_norm_max"], torch.tensor(12.0, dtype=torch.float64)
    )


def test_grad_norm_callback_legacy_instance_defaults_to_every_step() -> None:
    callback = GradNormCallback()
    del callback.every_n_steps
    callback.norm_type = 2
    module = _LoggedModule()
    module.weight.grad = torch.tensor([3.0, 4.0])
    optimizer = torch.optim.SGD(module.parameters(), lr=0.125)

    callback.on_before_optimizer_step(
        trainer=cast(Any, None), pl_module=cast(Any, module), optimizer=optimizer
    )

    assert "grad_2.0_norm_total" in module.logged
