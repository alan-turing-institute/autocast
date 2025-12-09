from typing import Any

import lightning as L
import torch
from torch import nn

from auto_cast.models.encoder_decoder import EncoderDecoder
from auto_cast.processors.base import Processor
from auto_cast.processors.rollout import RolloutMixin
from auto_cast.types import Batch, EncodedBatch, Tensor, TensorBMStarL, TensorBTSPlusC


class EncoderProcessorDecoder(RolloutMixin[Batch], L.LightningModule):
    """Encoder-Processor-Decoder Model."""

    encoder_decoder: EncoderDecoder
    processor: Processor
    teacher_forcing_ratio: float
    stride: int
    max_rollout_steps: int
    loss_func: nn.Module

    def __init__(
        self,
        encoder_decoder: EncoderDecoder,
        processor: Processor,
        learning_rate: float = 1e-3,
        stride: int = 1,
        teacher_forcing_ratio: float = 0.5,
        max_rollout_steps: int = 10,
        loss_func: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.encoder_decoder = encoder_decoder
        self.processor = processor
        self.learning_rate = learning_rate
        self.stride = stride
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_rollout_steps = max_rollout_steps
        self.loss_func = loss_func or nn.MSELoss()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, batch: Batch) -> TensorBTSPlusC:
        return self.decode(self.processor(self.encode(batch)))

    def encode(self, x: Batch) -> TensorBMStarL:
        return self.encoder_decoder.encoder(x)

    def decode(self, z: TensorBMStarL) -> TensorBTSPlusC:
        return self.encoder_decoder.decoder(z)

    def map(self, x: EncodedBatch) -> TensorBMStarL:
        return self.processor.map(x.encoded_inputs)

    def forward(self, batch: Batch) -> TensorBTSPlusC:
        return self.decode(self.processor(self.encode(batch)))

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        y_pred = self(batch)
        y_true = batch.output_fields
        loss = self.loss_func(y_pred, y_true)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        y_pred = self(batch)
        y_true = batch.output_fields
        loss = self.loss_func(y_pred, y_true)
        self.log(
            "val_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        y_pred = self(batch)
        y_true = batch.output_fields
        loss = self.loss_func(y_pred, y_true)
        self.log(
            "test_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _clone_batch(self, batch: Batch) -> Batch:
        return Batch(
            input_fields=batch.input_fields.clone(),
            output_fields=batch.output_fields.clone(),
            constant_scalars=(
                batch.constant_scalars.clone()
                if batch.constant_scalars is not None
                else None
            ),
            constant_fields=(
                batch.constant_fields.clone()
                if batch.constant_fields is not None
                else None
            ),
        )

    def _predict(self, batch: Batch) -> Tensor:
        return self(batch)

    def _true_slice(
        self,
        batch: Batch,
        stride: int,
    ) -> tuple[Tensor, bool]:
        if batch.output_fields.shape[1] >= stride:
            return batch.output_fields[:, :stride, ...], True
        return batch.output_fields, False

    def _advance_batch(self, batch: Batch, next_inputs: Tensor, stride: int) -> Batch:
        """Shift the input/output windows forward by `stride` using `next_inputs`."""
        next_inputs = torch.cat(
            [batch.input_fields[:, stride:, ...], next_inputs[:, :stride, ...]],
            dim=1,
        )

        next_outputs = (
            batch.output_fields[:, stride:, ...]
            if batch.output_fields.shape[1] > stride
            else batch.output_fields[:, 0:0, ...]  # Empty tensor with correct shape
        )

        return Batch(
            input_fields=next_inputs,
            output_fields=next_outputs,
            constant_scalars=batch.constant_scalars,
            constant_fields=batch.constant_fields,
        )
