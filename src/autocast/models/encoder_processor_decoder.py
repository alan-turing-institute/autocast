from collections.abc import Sequence
from typing import Any, Self

import lightning as L
import torch
from omegaconf import DictConfig
from the_well.data.normalization import RMSNormalization, ZScoreNormalization
from torch import nn
from torchmetrics import Metric, MetricCollection

from autocast.metrics.utils import MetricsMixin
from autocast.models.denorm_mixin import DenormMixin
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.optimizer_mixin import OptimizerMixin
from autocast.nn.noise.noise_injector import NoiseInjector
from autocast.processors.base import Processor
from autocast.processors.residual_flow_matching import ResidualFlowMatchingProcessor
from autocast.processors.rollout import RolloutMixin
from autocast.types import Batch, Tensor, TensorBTSC
from autocast.types.types import TensorBTSCM


class EncoderProcessorDecoder(
    DenormMixin, OptimizerMixin, RolloutMixin[Batch], L.LightningModule, MetricsMixin
):
    """Encoder-Processor-Decoder Model.

    Optionally decode cumulative residuals from the last clean input frame.
    ``residual_use_delta_stats`` selects dataset delta scaling; otherwise
    residuals use the input state's units. Outputs remain complete states.
    ``delta_norm`` optionally supplies increment statistics independently.

    ``prediction_steps=1`` permits a longer target window without changing the
    one-transition head. ``train_rollout_steps`` and ``validation_rollout_steps``
    count predicted-feedback calls; their losses are averaged across leads.
    Feedback retains gradients unless ``detach_rollout_feedback`` is true.
    Test batches use the validation horizon and optional validation objective.
    """

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
        train_in_latent_space: bool = False,
        freeze_encoder_decoder: bool = False,
        loss_func: nn.Module | None = None,
        train_metrics: Sequence[Metric] | None = [],
        val_metrics: Sequence[Metric] | None = None,
        test_metrics: Sequence[Metric] | None = None,
        input_noise_injector: NoiseInjector | None = None,
        norm: ZScoreNormalization | None = None,
        residual_prediction: bool = False,
        residual_use_delta_stats: bool = True,
        delta_norm: ZScoreNormalization | RMSNormalization | None = None,
        prediction_steps: int | None = None,
        train_rollout_steps: int = 1,
        validation_rollout_steps: int = 1,
        detach_rollout_feedback: bool = False,
        validation_loss_func: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if residual_prediction and isinstance(processor, ResidualFlowMatchingProcessor):
            msg = (
                "ResidualFlowMatchingProcessor already reconstructs complete states "
                "in processor coordinates; model.residual_prediction must be False."
            )
            raise ValueError(msg)
        if residual_prediction and train_in_latent_space:
            msg = (
                "Residual prediction requires train_in_latent_space=False "
                "and a loss on decoded states."
            )
            raise ValueError(msg)
        self.encoder_decoder = encoder_decoder
        self.processor = processor
        self.optimizer_config = optimizer_config
        self.stride = stride
        self.rollout_stride = rollout_stride if rollout_stride is not None else stride
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_rollout_steps = max_rollout_steps
        self.train_in_latent_space = train_in_latent_space
        self.freeze_encoder_decoder = freeze_encoder_decoder
        self.input_noise_injector = input_noise_injector
        self.norm = norm
        self.residual_prediction = residual_prediction
        self.residual_use_delta_stats = residual_use_delta_stats
        self.delta_norm = delta_norm
        for name, value in (
            ("train_rollout_steps", train_rollout_steps),
            ("validation_rollout_steps", validation_rollout_steps),
            ("prediction_steps", prediction_steps),
        ):
            if value is None and name == "prediction_steps":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                msg = f"{name} must be a positive integer."
                raise ValueError(msg)
        if max(train_rollout_steps, validation_rollout_steps) > 1 and (
            prediction_steps != 1 or train_in_latent_space
        ):
            msg = (
                "Training/validation rollouts require prediction_steps=1 "
                "and train_in_latent_space=False."
            )
            raise ValueError(msg)
        if validation_loss_func is not None and train_in_latent_space:
            msg = "validation_loss_func requires train_in_latent_space=False."
            raise ValueError(msg)
        self.prediction_steps = prediction_steps
        self.train_rollout_steps = train_rollout_steps
        self.validation_rollout_steps = validation_rollout_steps
        self.detach_rollout_feedback = detach_rollout_feedback
        self.validation_loss_func = validation_loss_func

        if self.train_in_latent_space or self.freeze_encoder_decoder:
            self.encoder_decoder.freeze()
        self.loss_func = loss_func

        self.train_metrics = self._build_metrics(train_metrics, "train_")
        self.val_metrics = self._build_metrics(val_metrics, "val_")
        self.test_metrics = self._build_metrics(test_metrics, "test_")

        for key, value in kwargs.items():
            setattr(self, key, value)

    def train(self, mode: bool = True) -> Self:
        """Keep frozen residual-flow coordinates fixed during training."""
        super().train(mode)
        if isinstance(self.processor, ResidualFlowMatchingProcessor) and (
            self.train_in_latent_space or self.freeze_encoder_decoder
        ):
            # Fitting uses an eval-mode codec. Dropout and running statistics
            # must not change those coordinates in the subsequent loss pass.
            self.encoder_decoder.eval()
        return self

    def _apply_input_noise(self, batch: Batch) -> Batch:
        """Apply input noise if self.input_noise_injector is set."""
        if self.input_noise_injector is not None:
            inputs = batch.input_fields
            if self.residual_prediction:
                inputs = inputs.clone()  # Protect rollout inputs from in-place noise.
            noisy_input = self.input_noise_injector(inputs)
            batch = Batch(
                input_fields=noisy_input,
                output_fields=batch.output_fields,
                constant_scalars=batch.constant_scalars,
                constant_fields=batch.constant_fields,
                boundary_conditions=batch.boundary_conditions,
            )
        return batch

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # PyTorch can serialise _metadata as a real key in the state_dict rather
        # than as an OrderedDict private attribute.  Strict load_state_dict then
        # rejects it as an unexpected key.  Remove it before Lightning loads.
        state_dict = checkpoint.get("state_dict", checkpoint)
        if isinstance(state_dict, dict):
            state_dict.pop("_metadata", None)

    def forward(self, batch: Batch) -> TensorBTSC | TensorBTSCM:
        latest_input = None
        if self.residual_prediction:
            # Preserve the clean baseline through the encoder/processor pipeline.
            latest_input = batch.input_fields[:, -1:, ...].clone()
        batch = self._apply_input_noise(batch)
        encoded, global_cond = self.encoder_decoder.encoder.encode_with_cond(batch)
        mapped = self.processor.map(encoded, global_cond)
        decoded = self.encoder_decoder.decoder.decode(mapped)
        if latest_input is not None:
            decoded = self._add_residual_to_latest_input(decoded, latest_input)
        return decoded

    def _add_residual_to_latest_input(
        self, residual: TensorBTSC, latest_input: TensorBTSC
    ) -> TensorBTSC:
        """Reconstruct all leads from one baseline; members are folded into batch."""
        if (
            residual.ndim != latest_input.ndim
            or residual.shape[0] != latest_input.shape[0]
            or residual.shape[2:] != latest_input.shape[2:]
            or residual.shape[1] == 0
            or latest_input.shape[1] != 1
        ):
            msg = (
                "Residual prediction requires matching batch, grid and channels "
                f"and nonempty frames: {residual.shape} vs {latest_input.shape}."
            )
            raise ValueError(msg)

        delta_norm = self.delta_norm if self.delta_norm is not None else self.norm
        if delta_norm is None or not self.residual_use_delta_stats:
            return latest_input + residual

        physical_delta = delta_norm.delta_denormalize_flattened(residual, "variable")
        if self.norm is None:
            return latest_input + physical_delta
        if isinstance(self.norm, ZScoreNormalization):
            # Avoid losing small increments when the physical state mean is large.
            state_scale = self.norm.flattened_stds["variable"].to(physical_delta.device)
            return latest_input + physical_delta / state_scale

        current_state = self.norm.denormalize_flattened(latest_input, "variable")
        return self.norm.normalize_flattened(current_state + physical_delta, "variable")

    def _denormalize_prediction(self, prediction: Tensor) -> Tensor:
        """Denormalize a full forward prediction, including any member axis."""
        return self.denormalize_tensor(prediction)

    def predict_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        return self._denormalize_prediction(self(batch))

    def _prepare_loss_rollout(self, batch: Batch) -> Batch:
        """Prepare member paths once, before any autoregressive calls."""
        return batch

    def _forward_loss_rollout(self, batch: Batch) -> Tensor:
        """Predict one transition without changing the number of paths."""
        return self(batch)

    def _restore_loss_rollout(self, prediction: Tensor, batch_size: int) -> Tensor:  # noqa: ARG002
        """Restore the loss layout after a transition (identity without members)."""
        return prediction

    def _rollout_loss(
        self, batch: Batch, steps: int, loss_func: nn.Module
    ) -> tuple[Tensor, Tensor]:
        """Score predicted-feedback transitions in normalized state coordinates.

        Unlike inference ``rollout``, this performs exactly one call per lead,
        does not denormalize feedback and retains the graph unless requested.
        """
        n_output = self.prediction_steps
        if n_output is None:
            msg = "An explicit prediction_steps is required for rollout losses."
            raise ValueError(msg)
        if batch.output_fields.shape[1] < steps * n_output:
            msg = "Target window is shorter than the requested rollout horizon."
            raise ValueError(msg)
        current = self._prepare_loss_rollout(batch)
        predictions, losses = [], []
        for lead in range(steps):
            flat_prediction = self._forward_loss_rollout(current)
            if flat_prediction.shape[1] != n_output:
                msg = "Model output frames do not match prediction_steps."
                raise ValueError(msg)
            prediction = self._restore_loss_rollout(
                flat_prediction, batch.input_fields.shape[0]
            )
            target = batch.output_fields[:, lead * n_output : (lead + 1) * n_output]
            predictions.append(prediction)
            losses.append(loss_func(prediction, target))
            if lead + 1 < steps:
                feedback = (
                    flat_prediction.detach()
                    if self.detach_rollout_feedback
                    else flat_prediction
                )
                current = self._advance_batch(current, feedback, n_output)
        return torch.stack(losses).mean(), torch.cat(predictions, dim=1)

    def loss(
        self,
        batch: Batch,
        *,
        rollout_steps: int | None = None,
        loss_func: nn.Module | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        active_loss = self.loss_func if loss_func is None else loss_func
        steps = self.train_rollout_steps if rollout_steps is None else rollout_steps
        if self.train_in_latent_space:
            batch = self._apply_input_noise(batch)
            encoded_batch = self.encoder_decoder.encoder.encode_batch(batch)
            loss = self.processor.loss(encoded_batch)
            y_pred = None
        else:
            if active_loss is None:
                msg = "loss_func must be provided when training full EPD model."
                raise ValueError(msg)
            if self.prediction_steps is not None:
                return self._rollout_loss(batch, steps, active_loss)
            # Otherwise, train full EPD model
            y_pred = self(batch)
            y_true = batch.output_fields
            loss = active_loss(y_pred, y_true)
        return loss, y_pred

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss, y_pred = self.loss(batch, rollout_steps=self.train_rollout_steps)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.input_fields.shape[0],
        )
        self._log_processor_diagnostics("train", batch.input_fields.shape[0])
        if self.validation_loss_func is not None and y_pred is not None:
            with torch.no_grad():
                base_loss = self.validation_loss_func(
                    y_pred, batch.output_fields[:, : y_pred.shape[1]]
                )
            self.log(
                "train_base_loss",
                base_loss,
                sync_dist=True,
                batch_size=batch.input_fields.shape[0],
            )
        if self.train_metrics is not None:
            if y_pred is None:
                y_pred = self(batch)
            y_true = batch.output_fields[:, : y_pred.shape[1]]
            self._update_and_log_metrics(
                self, self.train_metrics, y_pred, y_true, batch.input_fields.shape[0]
            )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss, y_pred = self.loss(
            batch,
            rollout_steps=self.validation_rollout_steps,
            loss_func=self.validation_loss_func,
        )
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.input_fields.shape[0],
        )
        self._log_processor_diagnostics("val", batch.input_fields.shape[0])
        if (
            self.validation_loss_func is not None
            and self.loss_func is not None
            and y_pred is not None
        ):
            self.log(
                "val_objective",
                self.loss_func(y_pred, batch.output_fields[:, : y_pred.shape[1]]),
                sync_dist=True,
                batch_size=batch.input_fields.shape[0],
            )
        if self.val_metrics is not None:
            if y_pred is None:
                y_pred = self(batch)
            y_true = batch.output_fields[:, : y_pred.shape[1]]
            y_pred = self._denormalize_prediction(y_pred)
            y_true = self.denormalize_tensor(y_true)
            self._update_and_log_metrics(
                self, self.val_metrics, y_pred, y_true, batch.input_fields.shape[0]
            )
        return loss

    def _log_processor_diagnostics(self, prefix: str, batch_size: int) -> None:
        """Forward processor-internal diagnostics to the logger.

        Processors may stash detached scalar tensors on
        ``self.processor._latest_diagnostics`` (e.g. ``DriftingProcessor``
        records within-cohort spread and per-scale force-RMS values to
        catch the n_+ = 1 contrastive-collapse failure mode). The
        Lightning module reads them here and writes them via
        ``log_dict`` with a ``train/`` or ``val/`` prefix. Processors
        that don't set the attribute are silently skipped.
        """
        diagnostics = getattr(self.processor, "_latest_diagnostics", None)
        if diagnostics:
            # sync_dist=False: each scalar is a per-rank batch-mean (cohort
            # spread, force-RMS) — the per-rank value is a fine estimate
            # and avoids a separate DDP all-reduce per diagnostic key.
            self.log_dict(
                {f"{prefix}/{name}": value for name, value in diagnostics.items()},
                sync_dist=False,
                batch_size=batch_size,
            )

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        # Test batches use the same predicted-feedback horizon and selection
        # objective as validation; inference rollout remains separately configured.
        loss, y_pred = self.loss(
            batch,
            rollout_steps=self.validation_rollout_steps,
            loss_func=self.validation_loss_func,
        )
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.input_fields.shape[0],
        )
        self._log_processor_diagnostics("test", batch.input_fields.shape[0])
        if self.test_metrics is not None:
            if y_pred is None:
                y_pred = self(batch)
            y_true = batch.output_fields[:, : y_pred.shape[1]]
            y_pred = self._denormalize_prediction(y_pred)
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
