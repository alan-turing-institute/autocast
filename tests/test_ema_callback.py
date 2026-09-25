from unittest.mock import Mock

import lightning as L
import torch
from torch import nn

from autocast.callbacks.ema import EMA_CHECKPOINT_KEY, EMACallback


class _ModelWithNestedState(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.factorization = {"core": torch.tensor([2.0])}

    def get_extra_state(self) -> dict[str, torch.Tensor]:
        return self.factorization

    def set_extra_state(self, state: dict[str, torch.Tensor]) -> None:
        self.factorization = state


def test_ema_checkpoint_deep_copies_nested_state() -> None:
    model = _ModelWithNestedState()
    trainer = Mock(spec=L.Trainer)
    callback = EMACallback()
    callback.on_fit_start(trainer, model)
    assert callback.ema_model is not None
    state = callback.ema_model.module.state_dict()
    checkpoint = {}

    callback.on_save_checkpoint(trainer, model, checkpoint)

    saved_state = checkpoint[EMA_CHECKPOINT_KEY]
    assert torch.equal(saved_state["weight"], state["weight"])
    assert torch.equal(
        saved_state["_extra_state"]["core"], state["_extra_state"]["core"]
    )
    assert saved_state["weight"] is not state["weight"]
    assert saved_state["_extra_state"] is not state["_extra_state"]
    assert saved_state["_extra_state"]["core"] is not state["_extra_state"]["core"]
