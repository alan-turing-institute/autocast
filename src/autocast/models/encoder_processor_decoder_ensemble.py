from typing import Any

from einops import rearrange

from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.types import Batch, RolloutOutput, Tensor, TensorBTSCM


class EncoderProcessorDecoderEnsemble(EncoderProcessorDecoder):
    """Encoder-Processor-Decoder Ensemble Model.

    A wrapper for EncoderProcessorDecoder that ensures ensemble outputs by
    repeating the batch internally and unflattening the result.
    """

    def __init__(self, *args: Any, n_members: int = 10, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.n_members = n_members

    def forward(self, batch: Batch) -> TensorBTSCM:
        """Forward pass through the ensemble model.

        Repeats the batch, processes it via the flattened pipeline, and returns BTSCM.
        """
        b = batch.input_fields.shape[0]

        # Expand batch for ensemble members
        ensemble_batch = batch.repeat(self.n_members)

        # Process with base class forward - input noise is applied inside
        # super().forward for each member independently
        out = super().forward(ensemble_batch)

        # Rearrange concatenated batch dim to separate ensemble dim at end
        return rearrange(out, "(b m) t ... c -> b t ... c m", b=b, m=self.n_members)

    def _predict(self, batch: Batch) -> Tensor:
        """Prediction for rollout (retains flattened batch dim)."""
        return super().forward(batch)

    def loss(self, batch: Batch) -> tuple[Tensor, Tensor | None]:
        """Compute ensemble-aware loss.

        If training in latent space (train_in_latent_space=True), we manually handle
        the ensemble expansion and loss computation.

        If training the full model (train_in_latent_space=False), we rely on
        super().loss(batch). super().loss(batch) calls self(batch), which calls
        EncoderProcessorDecoderEnsemble.forward.

        This ensures y_pred has shape [B, ..., M] (TensorBTSCM), suitable for ensemble
        metrics like CRPS.
        """
        # If not training in latent space OR if no ensemble-aware loss_func is provided,
        # we can just use the base class loss.
        if (
            not self.train_in_latent_space
            or self.n_members <= 1
            or self.loss_func is None
        ):
            return super().loss(batch)

        # TODO: consider removing this logic here as this might be better not
        # implemented with the ProcessorModel being a better place for this.
        # Latent Training with potential ensemble loss (e.g. CRPS in latent space)
        if self.n_members > 1 and self.loss_func is not None:
            B = batch.input_fields.shape[0]
            ensemble_batch = batch.repeat(self.n_members)
            ensemble_batch = self._apply_input_noise(ensemble_batch)
            encoded_batch = self.encoder_decoder.encoder.encode_batch(ensemble_batch)

            # Predictions (stochastic latent maps)
            preds_flat = self.processor.map(
                encoded_batch.encoded_inputs, global_cond=encoded_batch.global_cond
            )
            preds = rearrange(preds_flat, "(b m) ... -> b ... m", b=B, m=self.n_members)

            # Targets (encoded ground truth) - take the first from each member group
            # (assuming encoding of target is deterministic or we want a single target)
            targets = encoded_batch.encoded_output_fields[:: self.n_members]

            # If the loss_func can handle BTSCM and BTSC (like CRPS), it works here too
            # (though BTSC might be BNC in latent space, CRPS works on ndim >= 5)
            loss = self.loss_func(preds, targets)
            # Return None so metrics use decoded outputs via self(batch).
            return loss, None

        # Fallback to standard processor loss (e.g. Diffusion loss) on flattened batch
        return super().loss(batch)

    def rollout(
        self,
        batch: Batch,
        stride: int,
        max_rollout_steps: int = 10,
        teacher_forcing_ratio: float = 0.0,
        free_running_only: bool = False,
        return_windows: bool = False,
        detach: bool = True,
        n_members: int | None = None,
    ) -> RolloutOutput:
        """Rollout as an ensemble by running independent trajectories for members."""
        b = batch.input_fields.shape[0]

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
            detach=detach,
        )

        # If n_members is specified, rearrange outputs accordingly
        if n_members is not None:
            preds = rearrange(preds, "(b m) ... -> b ... m", b=b, m=n_members)
            if trues is not None:
                trues = rearrange(trues, "(b m) ... -> b ... m", b=b, m=n_members)
                trues = trues[..., 0]  # Only keep one since same across members

        # Return outputs with trues if not None
        if trues is not None:
            return preds, trues

        return preds, None
