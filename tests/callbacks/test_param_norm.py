"""Unit tests for :mod:`autocast.callbacks.param_norm`."""

from typing import Any

import lightning.pytorch as L
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autocast.callbacks.param_norm import ParamNormCallback, _param_norm


class _ConstantModule(nn.Module):
    """Module whose parameter values are fixed at construction time.

    Each tensor's L2 norm is therefore exactly known and the total
    ``||theta||_2`` is ``sqrt(sum_i ||theta_i||_2^2)`` — the formula
    :class:`ParamNormCallback` is supposed to compute.
    """

    def __init__(
        self,
        sizes: list[tuple[int, ...]],
        fill: float = 1.0,
    ) -> None:
        super().__init__()
        self.params = nn.ParameterList(
            [nn.Parameter(torch.full(size, fill_value=fill)) for size in sizes]
        )


def _manual_total_p_norm(module: nn.Module, p: float) -> float:
    """Compute ``||theta||_p`` over all trainable params, by hand."""
    per_tensor_powers = [
        float(param.detach().norm(p) ** p)
        for param in module.parameters()
        if param.requires_grad
    ]
    return sum(per_tensor_powers) ** (1.0 / p)


def test_constructor_rejects_zero_norm_type():
    with pytest.raises(ValueError, match="positive"):
        ParamNormCallback(norm_type=0.0)


def test_constructor_rejects_negative_norm_type():
    with pytest.raises(ValueError, match="positive"):
        ParamNormCallback(norm_type=-1.0)


def test_param_norm_utility_total_matches_manual_l2():
    """Spot-check the underlying ``_param_norm`` utility.

    Built three constant tensors of known shape and value so the L2 norm
    is fully predictable. If this drifts, the callback's logged total
    drifts with it.
    """
    module = _ConstantModule(sizes=[(4,), (2, 3), (5,)], fill=2.0)
    norms = _param_norm(module, norm_type=2.0)

    assert "param_2.0_norm_total" in norms
    expected_total = _manual_total_p_norm(module, p=2.0)
    torch.testing.assert_close(
        norms["param_2.0_norm_total"], torch.tensor(expected_total)
    )


def test_param_norm_utility_excludes_frozen_params():
    """Frozen weights (``requires_grad=False``) must not contribute.

    Weight decay only acts on params the optimizer sees; the diagnostic
    should reflect the same surface.
    """
    module = _ConstantModule(sizes=[(4,), (2, 3)], fill=2.0)
    # Freeze the second tensor.
    module.params[1].requires_grad_(False)

    norms = _param_norm(module, norm_type=2.0)
    # Only the (4,) tensor contributes — ||[2,2,2,2]||_2 = sqrt(16) = 4.
    torch.testing.assert_close(norms["param_2.0_norm_total"], torch.tensor(4.0))
    # The frozen tensor's per-param entry must not appear.
    assert not any("params.1" in k for k in norms)


def test_param_norm_utility_empty_module_returns_empty_dict():
    """A module with no trainable params should not crash or log junk."""
    module = nn.Module()
    norms = _param_norm(module, norm_type=2.0)
    assert norms == {}


@pytest.mark.parametrize("norm_type", [1.0, 2.0, 3.5])
def test_param_norm_utility_supports_general_p(norm_type):
    """General p-norm matches the manual closed-form formula."""
    module = _ConstantModule(sizes=[(2, 2), (3,)], fill=1.5)
    norms = _param_norm(module, norm_type=norm_type)
    expected = _manual_total_p_norm(module, p=norm_type)
    torch.testing.assert_close(
        norms[f"param_{norm_type}_norm_total"], torch.tensor(expected)
    )


# ---------------------------------------------------------------------------
# Lightning integration — fit a tiny model for 1 epoch and inspect what the
# callback emits via the trainer's ``logged_metrics`` dict.
# ---------------------------------------------------------------------------


class _ScalarFitModule(L.LightningModule):
    """Minimal Lightning module so we can exercise the callback via Trainer."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        x, y = batch
        pred = self.linear(x).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)


class _TinyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.randn(4), torch.tensor(0.0)


def _fit_one_epoch(callback: ParamNormCallback) -> dict[str, Any]:
    """Run one epoch of training with the given callback; return logged metrics."""
    torch.manual_seed(0)
    module = _ScalarFitModule()
    loader = DataLoader(_TinyDataset(), batch_size=2, num_workers=0)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        callbacks=[callback],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    trainer.fit(module, train_dataloaders=loader)
    return dict(trainer.logged_metrics)


def test_callback_logs_total_norm_only_by_default():
    """log_per_param=False (default) → only the aggregate key appears."""
    metrics = _fit_one_epoch(ParamNormCallback())
    assert "param_2.0_norm_total" in metrics
    per_param_keys = [k for k in metrics if k.startswith("param_2.0_norm/")]
    assert per_param_keys == []


def test_callback_logs_per_param_when_enabled():
    """log_per_param=True logs total AND per-tensor keys."""
    metrics = _fit_one_epoch(ParamNormCallback(log_per_param=True))
    assert "param_2.0_norm_total" in metrics
    per_param_keys = [k for k in metrics if k.startswith("param_2.0_norm/")]
    # _ScalarFitModule has linear.weight + linear.bias → two per-param keys.
    assert {"param_2.0_norm/linear.weight", "param_2.0_norm/linear.bias"} <= set(
        per_param_keys
    )


def test_callback_total_norm_is_finite_and_positive_after_fit():
    """End-to-end sanity check: the value Lightning sees is finite and > 0."""
    metrics = _fit_one_epoch(ParamNormCallback())
    total = float(metrics["param_2.0_norm_total"])
    assert torch.isfinite(torch.tensor(total))
    assert total > 0.0
