from torch import nn

from auto_cast.decoders import Decoder
from auto_cast.encoders import Encoder
from auto_cast.models.encoder_decoder import EncoderDecoder
from auto_cast.types import Batch, Tensor, TensorBTSC


class AELoss(nn.Module):
    """Autoencoder Loss Function."""

    def __init__(
        self, losses: list[nn.Module] | None = None, weights: list[float] | None = None
    ):
        super().__init__()
        losses = losses or [nn.MSELoss()]
        weights = weights or [1.0] * len(losses)
        self.losses = losses
        self.weights = weights

    def forward(self, model: EncoderDecoder, batch: Batch) -> Tensor:
        decoded, _ = model.forward_with_latent(batch)
        total_loss = decoded.new_zeros(())
        target = batch.output_fields
        for loss, weight in zip(self.losses, self.weights, strict=True):
            total_loss = total_loss + loss(decoded, target) * weight
        return total_loss


class AE(EncoderDecoder):
    """Autoencoder Model."""

    encoder: Encoder
    decoder: Decoder

    def __init__(
        self, encoder: Encoder, decoder: Decoder, loss_func: AELoss | None = None
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_func = loss_func or AELoss()

    def forward(self, batch: Batch) -> TensorBTSC:
        return self.forward_with_latent(batch)[0]

    def forward_with_latent(self, batch: Batch) -> tuple[TensorBTSC, TensorBTSC]:
        encoded = self.encode(batch)
        decoded = self.decode(encoded)
        return decoded, encoded

    def _compute_loss(self, batch: Batch) -> Tensor:
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
