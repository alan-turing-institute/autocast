from collections.abc import Sequence
from typing import Any

import lightning as L
import torch
from torch import nn
from torchmetrics import Metric, MetricCollection

from autocast.metrics.utils import MetricsMixin
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.optimizer_mixin import OptimizerMixin
from autocast.processors.base import Processor
from autocast.processors.rollout import RolloutMixin
from autocast.types import Batch, EncodedBatch, Tensor, TensorBNC, TensorBTSC


class EncoderProcessorDecoder(
    OptimizerMixin, RolloutMixin[Batch], L.LightningModule, MetricsMixin
):
    """Encoder-Processor-Decoder Model.""" ""

    encoder_decoder: EncoderDecoder
    processor: Processor
    train_metrics: MetricCollection | None
    val_metrics: MetricCollection | None
    test_metrics: MetricCollection | None

    def __init__(
        self,
        encoder_decoder: EncoderDecoder,
        processor: Processor,
        learning_rate: float = 1e-3,
        optimizer_config: dict[str, Any] | None = None,
        stride: int = 1,
        rollout_stride: int | None = None,
        teacher_forcing_ratio: float = 0.5,
        max_rollout_steps: int = 10,
        train_in_latent_space: bool = False,
        loss_func: nn.Module | None = None,
        train_metrics: Sequence[Metric] | None = [],
        val_metrics: Sequence[Metric] | None = None,
        test_metrics: Sequence[Metric] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.encoder_decoder = encoder_decoder
        self.processor = processor
        self.learning_rate = learning_rate
        self.optimizer_config = optimizer_config
        self.stride = stride
        self.rollout_stride = rollout_stride if rollout_stride is not None else stride
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_rollout_steps = max_rollout_steps
        self.train_in_latent_space = train_in_latent_space
        if self.train_in_latent_space:
            self.encoder_decoder.freeze()
        self.loss_func = loss_func

        self.train_metrics = self._build_metrics(train_metrics, "train_")
        self.val_metrics = self._build_metrics(val_metrics, "val_")
        self.test_metrics = self._build_metrics(test_metrics, "test_")

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, batch: Batch) -> TensorBTSC:
        return self.decode(self.processor(self.encode(batch)))

    def encode(self, x: Batch) -> TensorBNC:
        return self.encoder_decoder.encoder(x)

    def decode(self, z: TensorBNC) -> TensorBTSC:
        return self.encoder_decoder.decoder(z)

    def map(self, x: EncodedBatch) -> TensorBNC:
        return self.processor.map(x.encoded_inputs)

    def forward(self, batch: Batch) -> TensorBTSC:
        encoded = self.encoder_decoder.encoder.encode(batch)
        mapped = self.processor.map(encoded)
        decoded = self.encoder_decoder.decoder.decode(mapped)
        return decoded

    def loss(self, batch: Batch) -> tuple[Tensor, Tensor | None]:
        if self.train_in_latent_space:
            encoded_batch = self.encoder_decoder.encoder.encode_batch(batch)
            loss = self.processor.loss(encoded_batch)
            y_pred = None
        else:
            if self.loss_func is None:
                msg = "loss_func must be provided when training full EPD model."
                raise ValueError(msg)
            # Otherwise, train full EPD model
            y_pred = self(batch)
            y_true = batch.output_fields
            loss = self.loss_func(y_pred, y_true)
        return loss, y_pred

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss, y_pred = self.loss(batch)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        if self.train_metrics is not None:
            if y_pred is None:
                y_pred = self(batch)
            y_true = batch.output_fields
            self._update_and_log_metrics(
                self, self.train_metrics, y_pred, y_true, batch.input_fields.shape[0]
            )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss, y_pred = self.loss(batch)
        self.log(
            "val_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        if self.val_metrics is not None:
            if y_pred is None:
                y_pred = self(batch)
            y_true = batch.output_fields
            self._update_and_log_metrics(
                self, self.val_metrics, y_pred, y_true, batch.input_fields.shape[0]
            )
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss, y_pred = self.loss(batch)
        self.log(
            "test_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        if self.test_metrics is not None:
            if y_pred is None:
                y_pred = self(batch)
            y_true = batch.output_fields
            self._update_and_log_metrics(
                self, self.test_metrics, y_pred, y_true, batch.input_fields.shape[0]
            )
        return loss

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
        """Shift the input/output windows forward by `stride` using `next_inputs`.

        Note: stride parameter overrides self.stride to allow different strides
        for training vs evaluation.
        """
        # Get the original number of input time steps to maintain consistency
        n_steps_input = batch.input_fields.shape[1]

        # Concatenate remaining inputs with new predictions
        remaining_inputs = batch.input_fields[:, stride:, ...]
        new_predictions = next_inputs[:, :stride, ...]

        if remaining_inputs.shape[1] == 0:
            # No remaining inputs, use most recent n_steps_input from predictions
            combined = new_predictions[:, -n_steps_input:, ...]
        else:
            combined = torch.cat([remaining_inputs, new_predictions], dim=1)
            # Keep only the most recent n_steps_input time steps
            combined = combined[:, -n_steps_input:, ...]

        next_outputs = (
            batch.output_fields[:, stride:, ...]
            if batch.output_fields.shape[1] > stride
            else batch.output_fields[:, 0:0, ...]  # Empty tensor with correct shape
        )

        return Batch(
            input_fields=combined,
            output_fields=next_outputs,
            constant_scalars=batch.constant_scalars,
            constant_fields=batch.constant_fields,
        )


class EPDTrainProcessor(EncoderProcessorDecoder):
    """Encoder-Processor-Decoder Model training on processor."""

    train_processor: Processor

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
        super().__init__(
            encoder_decoder=encoder_decoder,
            processor=processor,
            learning_rate=learning_rate,
            stride=stride,
            teacher_forcing_ratio=teacher_forcing_ratio,
            max_rollout_steps=max_rollout_steps,
            loss_func=loss_func,
            **kwargs,
        )

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        encoded_batch = self.encoder_decoder.encoder.encode_batch(batch)
        loss = self.processor.loss(encoded_batch)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        return loss

    def validation_step(self, batch, batch_idx: int):  # noqa: ARG002
        encoded_batch = self.encoder_decoder.encoder.encode_batch(batch)
        loss = self.processor.loss(encoded_batch)
        self.log(
            "valid_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        return loss
