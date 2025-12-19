from abc import ABC
from collections.abc import Sequence
from typing import Any

import lightning as L
import torch
from torch import nn
from torchmetrics import Metric

from autocast.metrics.utils import MetricsMixin
from autocast.processors.base import Processor
from autocast.processors.rollout import RolloutMixin
from autocast.types import EncodedBatch, Tensor, TensorBNC


class ProcessorModel(RolloutMixin[EncodedBatch], ABC, L.LightningModule, MetricsMixin):
    """Processor Base Class."""

    processor: Processor
    learning_rate: float

    def __init__(
        self,
        processor: Processor,
        stride: int = 1,
        loss_func: nn.Module | None = None,
        learning_rate: float | None = None,
        train_metrics: Sequence[Metric] | None = [],
        val_metrics: Sequence[Metric] | None = None,
        test_metrics: Sequence[Metric] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.processor = processor  # Register nn.Module parameters
        self.stride = stride
        self.loss_func = loss_func or nn.MSELoss()
        # Use processor's learning_rate if not explicitly provided
        self.learning_rate = (
            learning_rate
            if learning_rate is not None
            else getattr(processor, "learning_rate", 1e-3)
        )
        self.train_metrics = self._build_metrics(train_metrics, "train_")
        self.val_metrics = self._build_metrics(val_metrics, "val_")
        self.test_metrics = self._build_metrics(test_metrics, "test_")
        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self, x: TensorBNC) -> TensorBNC:
        return self.processor.map(x)

    def training_step(
        self,
        batch: EncodedBatch,
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        loss = self.processor.loss(batch)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch.encoded_inputs.shape[0]
        )
        if self.train_metrics is not None:
            y_pred = self._predict(batch)
            y_true = batch.encoded_output_fields
            self._update_and_log_metrics(
                self, self.train_metrics, y_pred, y_true, batch.encoded_inputs.shape[0]
            )
        return loss

    def validation_step(
        self,
        batch: EncodedBatch,
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        loss = self.processor.loss(batch)
        self.log(
            "val_loss", loss, prog_bar=True, batch_size=batch.encoded_inputs.shape[0]
        )
        if self.val_metrics is not None:
            y_pred = self._predict(batch)
            y_true = batch.encoded_output_fields
            self._update_and_log_metrics(
                self, self.val_metrics, y_pred, y_true, batch.encoded_inputs.shape[0]
            )
        return loss

    def test_step(self, batch: EncodedBatch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss = self.processor.loss(batch)
        self.log(
            "test_loss", loss, prog_bar=True, batch_size=batch.encoded_inputs.shape[0]
        )
        if self.test_metrics is not None:
            y_pred = self._predict(batch)
            y_true = batch.encoded_output_fields
            self._update_and_log_metrics(
                self, self.test_metrics, y_pred, y_true, batch.encoded_inputs.shape[0]
            )
        return loss

    def configure_optimizers(self):
        """Configure optimizers for training.

        Returns Adam optimizer with learning_rate. Subclasses can override
        to use different optimizers or learning rate schedules.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _clone_batch(self, batch: EncodedBatch) -> EncodedBatch:
        return EncodedBatch(
            encoded_inputs=batch.encoded_inputs.clone(),
            encoded_output_fields=batch.encoded_output_fields.clone(),
            label=None,
            encoded_info={
                key: value.clone() if hasattr(value, "clone") else value
                for key, value in batch.encoded_info.items()
            },
        )

    def _predict(self, batch: EncodedBatch) -> Tensor:
        return self.processor.map(batch.encoded_inputs)

    def map(self, x: Tensor) -> Tensor:
        """Map input tensor through the processor."""
        return self.processor.map(x)

    def _true_slice(self, batch: EncodedBatch, stride: int) -> tuple[Tensor, bool]:
        if batch.encoded_output_fields.shape[1] >= stride:
            return batch.encoded_output_fields[:, :stride, ...], True
        return batch.encoded_output_fields, False

    def _advance_batch(
        self, batch: EncodedBatch, next_inputs: Tensor, stride: int
    ) -> EncodedBatch:
        # Get the original number of input time steps to maintain consistency
        n_steps_input = batch.encoded_inputs.shape[1]

        # Concatenate remaining inputs with new predictions
        remaining_inputs = batch.encoded_inputs[:, stride:, ...]
        new_predictions = next_inputs[:, :stride, ...]

        if remaining_inputs.shape[1] == 0:
            # No remaining inputs, use most recent n_steps_input from predictions
            combined = new_predictions[:, -n_steps_input:, ...]
        else:
            combined = torch.cat([remaining_inputs, new_predictions], dim=1)
            # Keep only the most recent n_steps_input time steps
            combined = combined[:, -n_steps_input:, ...]

        next_outputs = (
            batch.encoded_output_fields[:, stride:, ...]
            if batch.encoded_output_fields.shape[1] > stride
            else batch.encoded_output_fields[:, 0:0, ...]
        )
        return EncodedBatch(
            encoded_inputs=combined,
            encoded_output_fields=next_outputs,
            label=batch.label,
            encoded_info=batch.encoded_info,
        )


class DiscreteProcessor(Processor, ABC):
    """DiscreteProcessor."""


class FlowBasedGenerativeProcessor(DiscreteProcessor):
    """Flow-based generative processor."""
