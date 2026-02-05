from collections.abc import Sequence
from typing import Any

from omegaconf import DictConfig
from the_well.data.normalization import ZScoreNormalization
from torch import nn
from torchmetrics import Metric

from autocast.decoders import Decoder
from autocast.encoders.base import EncoderWithCond
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.types import Batch, Tensor, TensorBNC, TensorBTSC


class AELoss(nn.Module):
    """Autoencoder Loss Function."""

    @staticmethod
    def get_loss(loss: str) -> nn.Module:
        if loss.lower() == "mse":
            return nn.MSELoss()
        raise ValueError(f"{loss} not currently supported.")

    def __init__(
        self,
        losses: Sequence[nn.Module] | None = None,
        weights: Sequence[float] | None = None,
    ):
        super().__init__()
        losses = losses or [nn.MSELoss()]
        self.losses = losses
        self.weights = weights or [1.0] * len(self.losses)

    def forward(self, model: EncoderDecoder, batch: Batch) -> Tensor:
        decoded, _ = model.forward_with_latent(batch)
        total_loss = decoded.new_zeros(())
        target = batch.output_fields
        for loss, weight in zip(self.losses, self.weights, strict=True):
            total_loss = total_loss + loss(decoded, target) * weight
        return total_loss


class AE(EncoderDecoder):
    """Autoencoder Model."""

    encoder: EncoderWithCond
    decoder: Decoder

    def __init__(
        self,
        encoder: EncoderWithCond,
        decoder: Decoder,
        loss_func: AELoss | None = None,
        optimizer_config: DictConfig | dict[str, Any] | None = None,
        train_metrics: Sequence[Metric] | None = [],
        val_metrics: Sequence[Metric] | None = None,
        test_metrics: Sequence[Metric] | None = None,
        normalization_type: ZScoreNormalization | None = None,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            loss_func=loss_func or AELoss(),
            optimizer_config=optimizer_config,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            normalization_type=normalization_type,
        )

    def forward(self, batch: Batch) -> TensorBNC:
        return self.forward_with_latent(batch)[0]

    def forward_with_latent(self, batch: Batch) -> tuple[TensorBTSC, TensorBNC]:
        encoded = self.encode(batch)
        decoded = self.decode(encoded)
        return decoded, encoded

    def _compute_loss(self, batch: Batch) -> Tensor:
        assert self.loss_func is not None
        return self.loss_func(self, batch)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss = self._compute_loss(batch)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        if self.train_metrics is not None:
            y_pred = self(batch)
            y_true = batch.output_fields
            self._update_and_log_metrics(
                self, self.train_metrics, y_pred, y_true, batch.input_fields.shape[0]
            )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss = self._compute_loss(batch)
        self.log(
            "val_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        if self.val_metrics is not None:
            y_pred = self(batch)
            y_true = batch.output_fields
            self._update_and_log_metrics(
                self, self.val_metrics, y_pred, y_true, batch.input_fields.shape[0]
            )
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss = self._compute_loss(batch)
        self.log(
            "test_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        if self.test_metrics is not None:
            y_pred = self(batch)
            y_true = batch.output_fields
            self._update_and_log_metrics(
                self, self.test_metrics, y_pred, y_true, batch.input_fields.shape[0]
            )
        return loss
