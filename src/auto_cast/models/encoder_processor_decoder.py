from typing import Any, Self

import lightning as L
import torch
from torch import nn

from auto_cast.models.encoder_decoder import EncoderDecoder
from auto_cast.nn.base import Module
from auto_cast.processors.base import Processor
from auto_cast.types import Batch, RolloutOutput, Tensor


class EncoderProcessorDecoder(L.LightningModule):
    """Encoder-Processor-Decoder Model."""

    encoder_decoder: EncoderDecoder
    processor: Processor | nn.Module
    teacher_forcing_ratio: float
    stride: int
    max_rollout_steps: int
    loss_func: nn.Module

    def __init__(
        self,
        learning_rate: float = 1e-3,
        stride: int = 1,
        teacher_forcing_ratio: float = 0.5,
        max_rollout_steps: int = 10,
        loss_func: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.stride = stride
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_rollout_steps = max_rollout_steps
        self.loss_func = loss_func or nn.MSELoss()
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_encoder_processor_decoder(
        cls,
        encoder_decoder: EncoderDecoder,
        processor: Processor | Module,
        **kwargs: Any,
    ) -> Self:
        instance = cls(**kwargs)
        instance.encoder_decoder = encoder_decoder
        instance.processor = processor
        for key, value in kwargs.items():
            setattr(instance, key, value)
        return instance

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.decode(self.processor(self.encode(*args, **kwargs)))

    def encode(self, x: Batch) -> Tensor:
        return self.encoder_decoder.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.encoder_decoder.decoder(x)

    def map(self, x: Batch) -> Tensor:
        return self.forward(x)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        y_pred = self.map(batch)
        y_true = batch.output_fields
        loss = self.loss_func(y_pred, y_true)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        y_pred = self(batch)
        y_true = batch.output_fields
        loss = self.loss_func(y_pred, y_true)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        y_pred = self(batch)
        y_true = batch.output_fields
        loss = self.loss_func(y_pred, y_true)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def rollout(self, batch: Batch) -> RolloutOutput:
        """Rollout over multiple time steps."""
        pred_outs, gt_outs = [], []
        for _ in range(0, self.max_rollout_steps, self.stride):
            x = self.encoder_decoder.encoder(batch)
            pred_outs.append(self.processor(x))
            # TODO: combining teacher forcing logic
            gt_outs.append(batch.output_fields)  # This assumes we have output fields
        return torch.stack(pred_outs), torch.stack(gt_outs)


# # TODO: consider if separate rollout class would be better
# class Rollout:
#     max_rollout_steps: int
#     stride: int

#     def rollout(
#         self,
#         batch: Batch,
#         model: Processor | EncoderProcessorDecoder,
#     ) -> RolloutOutput:
#         """Rollout over multiple time steps."""
#         pred_outs, gt_outs = [], []
#         for _ in range(0, self.max_rollout_steps, self.stride):
#             output = model(batch)
#             pred_outs.append(output)
#             # TODO: logic for moving window with teacher forcing that assigns
#             gt_outs.append(batch.output_fields)  # This assumes we have output fields
#         return torch.stack(pred_outs), torch.stack(gt_outs)
