from types import SimpleNamespace

import torch

from autocast.callbacks.ema import EMA_CHECKPOINT_KEY, EMACallback


def test_ema_checkpoint_deep_copies_nested_state() -> None:
    tensor = torch.tensor([1.0])
    nested_tensor = torch.tensor([2.0])
    state = {
        "weight": tensor,
        "factorization": {"core": nested_tensor},
    }
    callback = EMACallback()
    callback.ema_model = SimpleNamespace(
        module=SimpleNamespace(state_dict=lambda: state)
    )
    checkpoint = {}

    callback.on_save_checkpoint(None, None, checkpoint)

    saved_state = checkpoint[EMA_CHECKPOINT_KEY]
    assert torch.equal(saved_state["weight"], tensor)
    assert torch.equal(saved_state["factorization"]["core"], nested_tensor)
    assert saved_state["weight"] is not tensor
    assert saved_state["factorization"] is not state["factorization"]
    assert saved_state["factorization"]["core"] is not nested_tensor
