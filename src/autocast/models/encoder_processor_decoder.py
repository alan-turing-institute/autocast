from typing import Any

import lightning as L
import torch
from torch import nn
from torchmetrics import Metric, MetricCollection

from autocast.metrics import ALL_METRICS
from autocast.models.denorm_mixin import DenormMixin
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.processors.base import Processor
from autocast.processors.rollout import RolloutMixin
from autocast.types import Batch, EncodedBatch, Tensor, TensorBNC, TensorBTSC


class EncoderProcessorDecoder(RolloutMixin[Batch], L.LightningModule, DenormMixin):
    """Encoder-Processor-Decoder Model."""

    encoder_decoder: EncoderDecoder
    processor: Processor
    teacher_forcing_ratio: float
    stride: int
    max_rollout_steps: int
    val_metrics: MetricCollection | None
    test_metrics: MetricCollection | None

    def __init__(
        self,
        encoder_decoder: EncoderDecoder,
        processor: Processor,
        learning_rate: float = 1e-3,
        stride: int = 1,
        teacher_forcing_ratio: float = 0.5,
        max_rollout_steps: int = 10,
        train_processor_only: bool = False,
        loss_func: nn.Module | None = None,
        val_metrics: list[Metric] | None = None,
        test_metrics: list[Metric] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.encoder_decoder = encoder_decoder
        self.processor = processor
        self.learning_rate = learning_rate
        self.stride = stride
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_rollout_steps = max_rollout_steps
        self.train_processor_only = train_processor_only
        if self.train_processor_only:
            self.encoder_decoder.freeze()
        self.loss_func = loss_func

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
        return decoded  # noqa: RET504

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        if self.train_processor_only:
            with torch.no_grad():
                encoded_batch = self.encoder_decoder.encoder.encode_batch(batch)
            loss = self.processor.loss(encoded_batch)
            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                batch_size=batch.input_fields.shape[0],
            )
            return loss
        if self.loss_func is None:
            msg = "loss_func must be provided when training full EPD model."
            raise ValueError(msg)

        # Otherwise, train full EPD model
        y_pred = self(batch)
        y_true = batch.output_fields
        loss = self.loss_func(y_pred, y_true)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        if self.train_processor_only:
            with torch.no_grad():
                encoded_batch = self.encoder_decoder.encoder.encode_batch(batch)
            loss = self.processor.loss(encoded_batch)
            self.log(
                "val_loss",
                loss,
                prog_bar=True,
                batch_size=batch.input_fields.shape[0],
            )
            return loss

        if self.loss_func is None:
            msg = "loss_func must be provided validating full EPD model."
            raise ValueError(msg)
        y_pred = self(batch)
        y_true = batch.output_fields
        loss = self.loss_func(y_pred, y_true)
        self.log(
            "val_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        if self.val_metrics is not None:
            self.val_metrics.update(y_pred, y_true)
            self.log_dict(
                self.val_metrics,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch.input_fields.shape[0],
            )
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        y_pred = self(batch)
        y_true = batch.output_fields
        if self.train_processor_only:
            with torch.no_grad():
                encoded_batch = self.encoder_decoder.encoder.encode_batch(batch)
            loss = self.processor.loss(encoded_batch)
            self.log(
                "test_loss",
                loss,
                prog_bar=True,
                batch_size=batch.input_fields.shape[0],
            )
            return loss
        if self.loss_func is None:
            msg = "loss_func must be provided testing full EPD model."
            raise ValueError(msg)
        loss = self.loss_func(y_pred, y_true)
        self.log(
            "test_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        if self.test_metrics is not None:
            self.test_metrics.update(y_pred, y_true)
            self.log_dict(
                self.test_metrics,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch.input_fields.shape[0],
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

    @staticmethod
    def _build_metrics(
        metrics: list[Metric] | None,
        prefix: str,
    ) -> MetricCollection | None:
        # If no metrics provided, default to a single MSE
        metrics_list = [] if metrics is None else metrics

        metric_dict: dict[str, Metric | MetricCollection] = {}

        for metric in metrics_list:
            if not isinstance(metric, ALL_METRICS):
                allowed = ", ".join(cls.__name__ for cls in ALL_METRICS)
                raise ValueError(
                    f"Invalid metric '{metric}'. Allowed metrics: {allowed}"
                )

            # Determine metric name
            name = getattr(metric, "name", None)
            if not isinstance(name, str):
                name = metric.__class__.__name__.lower()

            if name in metric_dict:
                raise ValueError(f"Duplicate metric name '{name}'")

            metric_dict[name] = metric

        if not metric_dict:
            return None

        return MetricCollection(metric_dict).clone(prefix=prefix)


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
        # TODO: ensure no grads propagate through encoder_decoder
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
