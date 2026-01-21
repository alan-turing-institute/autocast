from typing import Any

import torch
from einops import rearrange

from autocast.models.processor import ProcessorModel
from autocast.types import EncodedBatch, RolloutOutput, Tensor, TensorBNC, TensorBNCM


class ProcessorModelEnsemble(ProcessorModel):
    """Processor Model Ensemble.

    A wrapper for ProcessorModel that ensures ensemble outputs by
    repeating the batch internally and unflattening the result.
    """

    def __init__(self, *args: Any, n_members: int = 10, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.n_members = n_members

    def forward(self, x: TensorBNC, global_cond: Tensor | None) -> TensorBNCM:
        """Forward pass through the ensemble model.

        Repeats the batch, processes it via the flattened pipeline, and returns BTSCM.
        """
        # Expand batch for ensemble members
        ensemble_x = torch.repeat_interleave(x, self.n_members, dim=0)
        ensemble_global_cond = (
            torch.repeat_interleave(global_cond, self.n_members, dim=0)
            if global_cond is not None
            else None
        )

        # Process with base class forward - input noise is applied inside
        # super().forward for each member independently
        out = super().forward(ensemble_x, ensemble_global_cond)

        # Rearrange concatenated batch dim to separate ensemble dim at end
        return rearrange(out, "(b m) ... -> b ... m", b=x.shape[0], m=self.n_members)

    def _predict(self, batch: EncodedBatch) -> Tensor:
        """Prediction for rollout (retains flattened batch dim)."""
        return super()._predict(batch)

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Compute ensemble-aware loss.

        We manually handle the ensemble expansion and loss computation.

        This ensures y_pred has shape [B, ..., M] (TensorBTSCM), suitable for ensemble
        metrics like CRPS.
        """
        # If no ensemble-aware loss_func provided, we can just use the base class loss
        if self.n_members <= 1 or self.loss_func is None:
            return super().loss(batch)

        B = batch.encoded_inputs.shape[0]
        ensemble_batch = batch.repeat(self.n_members)
        ensemble_batch = self._apply_input_noise(ensemble_batch)

        # Predictions (stochastic latent maps)
        preds_flat = self.processor.map(
            ensemble_batch.encoded_inputs, global_cond=ensemble_batch.global_cond
        )
        preds = rearrange(preds_flat, "(b m) ... -> b ... m", b=B, m=self.n_members)

        # Targets (encoded ground truth) - take the first from each member group
        targets = ensemble_batch.encoded_output_fields[:: self.n_members]

        # If the loss_func can handle BTSCM and BTSC (like CRPS), it works here too
        loss = self.loss_func(preds, targets)
        return loss

    def rollout(
        self,
        batch: EncodedBatch,
        stride: int,
        max_rollout_steps: int = 10,
        teacher_forcing_ratio: float = 0.0,
        free_running_only: bool = False,
        return_windows: bool = False,
        n_members: int | None = None,
    ) -> RolloutOutput:
        """Rollout as an ensemble by running independent trajectories for members."""
        b = batch.encoded_inputs.shape[0]

        # Expand batch for ensemble members.
        expanded_batch = batch.repeat(n_members) if n_members is not None else batch

        # Run rollout on the expanded batch.
        preds, trues = super().rollout(
            expanded_batch,
            stride=stride,
            max_rollout_steps=max_rollout_steps,
            teacher_forcing_ratio=teacher_forcing_ratio,
            free_running_only=free_running_only,
            return_windows=return_windows,
        )

        # Rearrange concatenated batch dim to separate ensemble dim at end
        def rearange_outputs(
            preds: Tensor, trues: Tensor | None
        ) -> tuple[Tensor, Tensor | None]:
            preds = rearrange(
                preds, "(b m) ... -> b ... m", b=b, m=n_members or self.n_members
            )
            if trues is not None:
                trues = rearrange(
                    trues, "(b m) ... -> b ... m", b=b, m=(n_members or self.n_members)
                )
                return preds, trues
            return preds, trues

        # If n_members is specified, rearrange outputs accordingly
        if n_members is not None:
            preds, trues = rearange_outputs(preds, trues)

        # Return outputs with trues if not None
        if trues is not None:
            return preds, trues

        return preds, None
