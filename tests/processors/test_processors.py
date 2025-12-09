import torch
from torch import nn

from auto_cast.models.processor import ProcessorModel
from auto_cast.processors.base import Processor
from auto_cast.types import EncodedBatch, Tensor


def _toy_encoded_batch(
    batch_size: int = 2,
    t_in: int = 2,
    t_out: int | None = None,
    w: int = 4,
    h: int = 4,
    c: int = 1,
) -> EncodedBatch:
    t_out = t_out or t_in
    encoded_inputs = torch.randn(batch_size, t_in, w, h, c)
    encoded_outputs = torch.randn(batch_size, t_out, w, h, c)
    return EncodedBatch(
        encoded_inputs=encoded_inputs,
        encoded_output_fields=encoded_outputs,
        encoded_info={},
    )


class _IdentityProcessor(Processor):
    def __init__(self) -> None:
        super().__init__(
            loss_func=nn.MSELoss(),
        )

    def map(self, x: Tensor) -> Tensor:
        return x

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        return self.loss_func(output, target)


def test_processor_rollout_handles_encoded_batches():
    processor = ProcessorModel(
        processor=_IdentityProcessor(),
        stride=1,
        max_rollout_steps=2,
    )
    encoded_batch = _toy_encoded_batch(t_in=2, t_out=2)

    preds, gts = processor.rollout(encoded_batch)

    assert preds.shape[0] == 2
    assert gts is not None
    assert gts.shape[0] == 2
