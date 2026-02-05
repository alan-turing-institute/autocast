import logging

import lightning as L
from the_well.data.normalization import ZScoreNormalization

from autocast.types.batch import Batch
from autocast.types.types import Tensor

log = logging.getLogger(__name__)


class DenormMixin(L.LightningModule):
    """
    Mixin class to provide denormalization functionality for models.

    Based on The Well Trainer.denormalize(), see:
    https://github.com/PolymathicAI/the_well/blob/6cd3c44ef832855a5abae87d555bf0f0f52b1fa7/the_well/benchmark/trainer/training.py#L190
    """

    norm: ZScoreNormalization | None = None

    def denormalize_tensor(
        self,
        tensor: Tensor,
        delta=False,
    ) -> Tensor:
        """
        Denormalize a tensor (e.g., a prediction).

        Parameters
        ----------
        tensor : Tensor
            The normalized tensor to be denormalized.
        delta : bool, optional
            Whether to apply delta denormalization. Default is False.

        Returns
        -------
        Tensor
            The denormalized tensor.
        """
        if self.norm is None:
            return tensor

        if delta:
            denorm_tensor = self.norm.delta_denormalize_flattened(tensor, "variable")
        else:
            denorm_tensor = self.norm.denormalize_flattened(tensor, "variable")

        return denorm_tensor

    def predict_step(
        self,
        batch: Batch,
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """
        Override predict_step to include denormalization.

        Parameters
        ----------
        batch : Batch
            The input batch.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        Tensor
            The (optionally denormalized) predictions.
        """
        predictions = self(batch)

        if self.norm is None:
            return predictions

        log.info("Denormalizing predictions.")
        return self.denormalize_tensor(predictions)
