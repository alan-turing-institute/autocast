from typing import Any, Self

import lightning as L
import torch
from torch import nn

from auto_cast.models.encoder_decoder import EncoderDecoder
from auto_cast.processors.base import Processor
from auto_cast.types import Batch, EncodedBatch, RolloutOutput, Tensor


class EncoderProcessorDecoder(L.LightningModule):
    """Encoder-Processor-Decoder Model."""

    encoder_decoder: EncoderDecoder
    processor: Processor
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
        processor: Processor,
        **kwargs: Any,
    ) -> Self:
        instance = cls(**kwargs)
        instance.encoder_decoder = encoder_decoder
        instance.processor = processor
        for key, value in kwargs.items():
            setattr(instance, key, value)
        return instance

    def __call__(self, batch: Batch) -> Tensor:
        return self.decode(self.processor(self.encode(batch)))

    def encode(self, x: Batch) -> Tensor:
        return self.encoder_decoder.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.encoder_decoder.decoder(x)

    def map(self, x: EncodedBatch) -> Tensor:
        return self.processor.map(x.encoded_inputs)

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

    def rollout(self, batch: Batch) -> RolloutOutput:
        """Rollout over multiple time steps with optional teacher forcing."""
        pred_outs: list[Tensor] = []
        gt_outs: list[Tensor] = []

        # Initialize the current batch for rollout
        current_batch = Batch(
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

        # Rollout loop with teacher forcing
        for _ in range(0, self.max_rollout_steps, self.stride):
            output = self(current_batch)
            pred_outs.append(output)

            if current_batch.output_fields.shape[1] >= self.stride:
                gt_slice = current_batch.output_fields[:, : self.stride, ...]
                gt_outs.append(gt_slice)
            else:
                gt_slice = current_batch.output_fields

            # Simple teacher forcing logic with Bernoulli sampling
            rand_val = torch.rand(1, device=output.device).item()
            teacher_force = (
                gt_slice.numel() > 0 and rand_val < self.teacher_forcing_ratio
            )
            feedback = gt_slice if teacher_force else output.detach()

            if feedback.shape[1] < self.stride:
                break

            current_batch = self._advance_batch(current_batch, feedback, self.stride)

        # Stack predictions and ground truths and return
        predictions = torch.stack(pred_outs)
        if gt_outs:
            return predictions, torch.stack(gt_outs)
        return predictions, None

    @staticmethod
    def _advance_batch(batch: Batch, feedback: Tensor, stride: int) -> Batch:
        """Shift the input/output windows forward by `stride` using `feedback`."""
        next_inputs = torch.cat(
            [batch.input_fields[:, stride:, ...], feedback[:, :stride, ...]],
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
