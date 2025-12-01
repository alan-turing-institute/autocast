from typing import Any

import lightning as L
import torch
from torch import nn

from auto_cast.models.encoder_decoder import EncoderDecoder
from auto_cast.processors.base import Processor
from auto_cast.types import Batch, RolloutOutput, Tensor


class EncoderProcessorDecoder(L.LightningModule):
    """Encoder-Processor-Decoder Model."""

    encoder_decoder: EncoderDecoder
    processor: Processor
    teacher_forcing_ratio: float
    stride: int
    max_rollout_steps: int
    loss_func: nn.Module

    def __init__(self): ...

    def from_encoder_processor_decoder(
        self, encoder_decoder: EncoderDecoder, processor: Processor
    ) -> None:
        self.encoder_decoder = encoder_decoder
        self.processor = processor

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.encoder_decoder.decoder(
            self.processor(self.encoder_decoder.encoder(*args, **kwargs))
        )

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        output = self(batch)
        loss = self.processor.loss_func(output, batch.output_fields)
        return loss  # noqa: RET504

    def configure_optimizers(self): ...

    def rollout(self, batch: Batch) -> RolloutOutput:
        """Rollout over multiple time steps."""
        pred_outs, gt_outs = [], []
        for _ in range(0, self.max_rollout_steps, self.stride):
            x = self.encoder_decoder.encoder(batch)
            pred_outs.append(self.processor.map(x))
            # TODO: combining teacher forcing logic
            gt_outs.append(batch.output_fields)  # This assumes we have output fields
        return torch.stack(pred_outs), torch.stack(gt_outs)


# TODO: consider if separate rollout class would be better
class Rollout:
    max_rollout_steps: int
    stride: int

    def rollout(
        self,
        batch: Batch,
        model: Processor | EncoderProcessorDecoder,
    ) -> RolloutOutput:
        """Rollout over multiple time steps."""
        pred_outs, gt_outs = [], []
        for _ in range(0, self.max_rollout_steps, self.stride):
            output = model(batch)
            pred_outs.append(output)
            # TODO: logic for moving window with teacher forcing that assigns
            gt_outs.append(batch.output_fields)  # This assumes we have output fields
        return torch.stack(pred_outs), torch.stack(gt_outs)
