import warnings
from collections.abc import Sequence
from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig
from the_well.data.normalization import ZScoreNormalization
from torch import nn
from torchmetrics import Metric, MetricCollection

from autocast.metrics.utils import MetricsMixin
from autocast.models.denorm_mixin import DenormMixin
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.optimizer_mixin import OptimizerMixin
from autocast.nn.noise.noise_injector import NoiseInjector
from autocast.processors.base import Processor
from autocast.processors.rollout import RolloutMixin
from autocast.types import Batch, EncodedBatch, Tensor, TensorBTSC
from autocast.types.types import TensorBTSCM


class EncoderProcessorDecoder(
    DenormMixin, OptimizerMixin, RolloutMixin[Batch], L.LightningModule, MetricsMixin
):
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
        optimizer_config: DictConfig | dict[str, Any] | None = None,
        stride: int = 1,
        rollout_stride: int | None = None,
        teacher_forcing_ratio: float = 0.5,
        max_rollout_steps: int = 10,
        train_in_latent_space: bool | None = None,
        freeze_encoder_decoder: bool = False,
        loss_func: nn.Module | None = None,
        ambient_loss_weight: float | None = None,
        latent_loss_weight: float | None = None,
        train_metrics: Sequence[Metric] | None = [],
        val_metrics: Sequence[Metric] | None = None,
        test_metrics: Sequence[Metric] | None = None,
        input_noise_injector: NoiseInjector | None = None,
        latent_noise_injector: NoiseInjector | None = None,
        norm: ZScoreNormalization | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.encoder_decoder = encoder_decoder
        self.processor = processor
        self.optimizer_config = optimizer_config
        self.stride = stride
        self.rollout_stride = rollout_stride if rollout_stride is not None else stride
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_rollout_steps = max_rollout_steps
        self.freeze_encoder_decoder = freeze_encoder_decoder
        self.input_noise_injector = input_noise_injector
        self.latent_noise_injector = latent_noise_injector
        self.norm = norm

        # Resolve loss weights from train_in_latent_space or explicit weights.
        self.ambient_loss_weight, self.latent_loss_weight = self._resolve_loss_weights(
            train_in_latent_space, ambient_loss_weight, latent_loss_weight
        )

        if self._pure_latent or self.freeze_encoder_decoder:
            self.encoder_decoder.freeze()
        self.loss_func = loss_func

        self.train_metrics = self._build_metrics(train_metrics, "train_")
        self.val_metrics = self._build_metrics(val_metrics, "val_")
        self.test_metrics = self._build_metrics(test_metrics, "test_")

        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def _resolve_loss_weights(
        train_in_latent_space: bool | None,
        ambient_loss_weight: float | None,
        latent_loss_weight: float | None,
    ) -> tuple[float, float]:
        """Resolve loss weights from legacy and new-style arguments.

        Legacy ``train_in_latent_space`` is mapped to the weight pair and a
        deprecation warning is emitted when the old flag is used.
        """
        explicit_weights = (
            ambient_loss_weight is not None or latent_loss_weight is not None
        )

        if train_in_latent_space is not None and explicit_weights:
            msg = (
                "Cannot specify both 'train_in_latent_space' and explicit loss weights "
                "('ambient_loss_weight' / 'latent_loss_weight'). Use the weight "
                "parameters instead."
            )
            raise ValueError(msg)

        if train_in_latent_space is not None:
            warnings.warn(
                "'train_in_latent_space' is deprecated. Use 'ambient_loss_weight' and "
                "'latent_loss_weight' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if train_in_latent_space:
                return 0.0, 1.0
            return 1.0, 0.0

        return (
            ambient_loss_weight if ambient_loss_weight is not None else 1.0,
            latent_loss_weight if latent_loss_weight is not None else 0.0,
        )

    @property
    def _pure_latent(self) -> bool:
        """True when training exclusively in latent space (encoder/decoder frozen)."""
        return self.latent_loss_weight > 0 and self.ambient_loss_weight == 0

    @property
    def train_in_latent_space(self) -> bool:
        """Backward-compatible property: True when only latent loss is active."""
        return self._pure_latent

    def _apply_input_noise(self, batch: Batch) -> Batch:
        """Apply input noise if self.input_noise_injector is set."""
        if self.input_noise_injector is not None:
            noisy_input = self.input_noise_injector(batch.input_fields)
            batch = Batch(
                input_fields=noisy_input,
                output_fields=batch.output_fields,
                constant_scalars=batch.constant_scalars,
                constant_fields=batch.constant_fields,
            )
        return batch

    def _apply_latent_noise(self, encoded_batch: EncodedBatch) -> EncodedBatch:
        """Apply noise in latent space after encoding."""
        if self.latent_noise_injector is not None:
            noisy_encoded = self.latent_noise_injector(encoded_batch.encoded_inputs)
            encoded_batch = EncodedBatch(
                encoded_inputs=noisy_encoded,
                encoded_output_fields=encoded_batch.encoded_output_fields,
                global_cond=encoded_batch.global_cond,
                encoded_info=encoded_batch.encoded_info,
            )
        return encoded_batch

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # PyTorch can serialise _metadata as a real key in the state_dict rather
        # than as an OrderedDict private attribute.  Strict load_state_dict then
        # rejects it as an unexpected key.  Remove it before Lightning loads.
        state_dict = checkpoint.get("state_dict", checkpoint)
        if isinstance(state_dict, dict):
            state_dict.pop("_metadata", None)

    def forward(self, batch: Batch) -> TensorBTSC | TensorBTSCM:
        batch = self._apply_input_noise(batch)
        encoded, global_cond = self.encoder_decoder.encoder.encode_with_cond(batch)
        if self.latent_noise_injector is not None:
            encoded = self.latent_noise_injector(encoded)
        mapped = self.processor.map(encoded, global_cond)
        decoded = self.encoder_decoder.decoder.decode(mapped)
        return decoded

    def _latent_loss(self, batch: Batch) -> Tensor:
        """Compute loss in latent (encoded) space via the processor."""
        batch = self._apply_input_noise(batch)
        encoded_batch = self.encoder_decoder.encoder.encode_batch(batch)
        encoded_batch = self._apply_latent_noise(encoded_batch)
        return self.processor.loss(encoded_batch)

    def _ambient_loss(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """Compute loss in ambient (decoded) space."""
        if self.loss_func is None:
            msg = "loss_func must be provided when ambient_loss_weight > 0."
            raise ValueError(msg)
        y_pred = self(batch)
        y_true = batch.output_fields
        return self.loss_func(y_pred, y_true), y_pred

    def loss(self, batch: Batch) -> tuple[Tensor, Tensor | None]:
        total_loss = torch.tensor(0.0, device=batch.input_fields.device)
        y_pred = None

        if self.ambient_loss_weight > 0:
            ambient_loss, y_pred = self._ambient_loss(batch)
            total_loss = total_loss + self.ambient_loss_weight * ambient_loss

        if self.latent_loss_weight > 0:
            latent_loss = self._latent_loss(batch)
            total_loss = total_loss + self.latent_loss_weight * latent_loss

        return total_loss, y_pred

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss, y_pred = self.loss(batch)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.input_fields.shape[0],
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
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.input_fields.shape[0],
        )
        if self.val_metrics is not None:
            if y_pred is None:
                y_pred = self(batch)
            y_true = batch.output_fields
            y_pred = self.denormalize_tensor(y_pred)
            y_true = self.denormalize_tensor(y_true)
            self._update_and_log_metrics(
                self, self.val_metrics, y_pred, y_true, batch.input_fields.shape[0]
            )
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss, y_pred = self.loss(batch)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.input_fields.shape[0],
        )
        if self.test_metrics is not None:
            if y_pred is None:
                y_pred = self(batch)
            y_true = batch.output_fields
            y_pred = self.denormalize_tensor(y_pred)
            y_true = self.denormalize_tensor(y_true)
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
            boundary_conditions=(
                batch.boundary_conditions.clone()
                if batch.boundary_conditions is not None
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
            boundary_conditions=batch.boundary_conditions,
        )
