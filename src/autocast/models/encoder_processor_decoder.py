from collections.abc import Sequence
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import OmegaConf
from torch import nn
from torchmetrics import Metric, MetricCollection

from autocast.metrics.utils import MetricsMixin
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.processors.base import Processor
from autocast.processors.rollout import RolloutMixin
from autocast.types import Batch, EncodedBatch, Tensor, TensorBNC, TensorBTSC


class EncoderProcessorDecoder(RolloutMixin[Batch], L.LightningModule, MetricsMixin):
    """Encoder-Processor-Decoder Model."""

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
        train_processor_only: bool = False,
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
        self.train_processor_only = train_processor_only
        if self.train_processor_only:
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
        return decoded  # noqa: RET504

    def loss(self, batch: Batch) -> tuple[Tensor, Tensor | None]:
        if self.train_processor_only:
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

    def _create_optimizer(self, cfg: dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        optimizer_name = str(cfg.get("optimizer", "adam")).lower()
        lr = cfg.get("learning_rate", self.learning_rate)
        weight_decay = cfg.get("weight_decay", 0.0)

        if optimizer_name == "adamw":
            betas = cfg.get("betas", [0.9, 0.999])
            return torch.optim.AdamW(
                self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
            )
        if optimizer_name == "adam":
            betas = cfg.get("betas", [0.9, 0.999])
            return torch.optim.Adam(
                self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
            )
        if optimizer_name == "sgd":
            momentum = cfg.get("momentum", 0.9)
            return torch.optim.SGD(
                self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        msg = f"Unsupported optimizer: {optimizer_name}"
        raise ValueError(msg)

    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer, cfg: dict[str, Any]
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler from config."""
        scheduler_name = str(cfg.get("scheduler", "")).lower()

        if scheduler_name == "cosine":
            max_epochs = 1
            if (
                getattr(self, "trainer", None) is not None
                and self.trainer.max_epochs is not None
            ):
                max_epochs = int(self.trainer.max_epochs)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=0
            )
        if scheduler_name == "step":
            step_size = cfg.get("step_size", 30)
            gamma = cfg.get("gamma", 0.1)
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        if scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=10
            )
        msg = f"Unsupported scheduler: {scheduler_name}"
        raise ValueError(msg)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizers for training."""
        # Backwards compatibility: if no optimizer_config, use simple Adam
        if self.optimizer_config is None:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Accept both plain dict and Hydra DictConfig
        cfg_any: Any = self.optimizer_config
        if not isinstance(cfg_any, dict):
            cfg_any = OmegaConf.to_container(cfg_any, resolve=True)
        if not isinstance(cfg_any, dict):
            msg = (
                "optimizer_config must be a mapping (dict-like). "
                f"Got: {type(cfg_any).__name__}"
            )
            raise TypeError(msg)
        cfg = cfg_any

        optimizer = self._create_optimizer(cfg)
        scheduler_name = cfg.get("scheduler", None)

        # Return optimizer only if no scheduler
        if scheduler_name is None:
            return optimizer

        scheduler = self._create_scheduler(optimizer, cfg)

        # ReduceLROnPlateau needs special handling
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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
