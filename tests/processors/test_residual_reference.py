import pytest
import torch
from einops import repeat
from torch import nn

from autocast.processors.base import Processor
from autocast.processors.residual_reference import (
    LastFrameReference,
    ProcessorReference,
)
from autocast.types import EncodedBatch, Tensor


class _RecordingProcessor(Processor[EncodedBatch]):
    def __init__(self, n_steps_output: int = 3) -> None:
        super().__init__()
        self.n_steps_output = n_steps_output
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.received_global_cond: Tensor | None = None

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        self.received_global_cond = global_cond
        return repeat(
            self.scale * x[:, -1:],
            "b 1 ... c -> b t ... c",
            t=self.n_steps_output,
        )

    def loss(self, batch: EncodedBatch) -> Tensor:
        raise NotImplementedError


def test_last_frame_reference_repeats_selected_channels():
    x = torch.arange(2 * 3 * 4 * 5 * 2).reshape(2, 3, 4, 5, 2)
    reference = LastFrameReference(n_steps_output=4, channel_indices=(1,))

    actual = reference(x)

    expected = repeat(x[:, -1:, ..., 1:2], "b 1 ... c -> b t ... c", t=4)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_steps_output": 0}, "must be positive"),
        ({"n_steps_output": 1, "channel_indices": ()}, "cannot be empty"),
    ],
)
def test_last_frame_reference_validates_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LastFrameReference(**kwargs)


def test_frozen_processor_reference_passes_conditioning_without_gradients():
    processor = _RecordingProcessor()
    reference = ProcessorReference(processor)
    reference.train()
    x = torch.randn(2, 1, 4, 4, 1, requires_grad=True)
    global_cond = torch.randn(2, 2)

    output = reference(x, global_cond)

    assert processor.received_global_cond is global_cond
    assert not processor.training
    assert not processor.scale.requires_grad
    assert not output.requires_grad


def test_trainable_processor_reference_preserves_gradients():
    processor = _RecordingProcessor()
    reference = ProcessorReference(processor, frozen=False)
    x = torch.randn(2, 1, 4, 4, 1, requires_grad=True)

    reference(x).sum().backward()

    assert processor.scale.grad is not None
    assert x.grad is not None
