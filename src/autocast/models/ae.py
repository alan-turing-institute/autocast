from collections.abc import Sequence

from torch import nn

from autocast.decoders import Decoder
from autocast.encoders import Encoder
from autocast.models.denorm_mixin import DenormMixin
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


class AE(EncoderDecoder, DenormMixin):
    """Autoencoder Model."""

    encoder: Encoder
    decoder: Decoder

    def __init__(
        self, encoder: Encoder, decoder: Decoder, loss_func: AELoss | None = None
    ):
        super().__init__(
            encoder=encoder, decoder=decoder, loss_func=loss_func or AELoss()
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
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss = self._compute_loss(batch)
        self.log(
            "val_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        return self._compute_loss(batch)
