import torch
from azula.nn.layers import ConvNd
from einops import rearrange
from torch import nn

from autocast.decoders import Decoder
from autocast.encoders.base import EncoderWithCond
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.types import Batch, Tensor, TensorBNC, TensorBTSC


class VAELoss(nn.Module):
    """Variational Autoencoder Loss Function."""

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, model: EncoderDecoder, batch: Batch) -> Tensor:
        """Compute VAE loss as reconstruction loss + beta * KL divergence."""
        decoded, encoded = model.forward_with_latent(batch)

        return self.beta * self.kl_divergence(encoded) + nn.functional.mse_loss(
            decoded, batch.output_fields
        )

    def kl_divergence(self, encoded: TensorBNC) -> Tensor:
        """Compute the KL divergence loss.

        Parameters
        ----------
        encoded: Tensor
            Encoded tensor containing mean and log variance.
            Shape: [B, 2*C, H, W, ...] for spatial or
            [B, 2*latent_dim] for flat.

        Returns
        -------
        Tensor
            KL divergence loss.
        """
        # Split along the appropriate dimension
        split_dim = 1 if encoded.dim() > 2 else -1
        mean, log_var = encoded.chunk(2, dim=split_dim)
        # Compute KL divergence, sum over all non-batch dimensions
        kl_div = -0.5 * torch.sum(
            1 + log_var - mean.pow(2) - log_var.exp(),
            dim=list(range(1, encoded.dim())),
        )
        return kl_div.mean()


class VAE(EncoderDecoder):
    """Variational Autoencoder Model.

    Supports both flat (B, latent_dim) and spatial (B, C, H, W, ...)
    latent representations.
    """

    encoder: EncoderWithCond
    decoder: Decoder
    fc_mean: nn.Module
    fc_log_var: nn.Module

    def __init__(
        self, encoder: EncoderWithCond, decoder: Decoder, spatial: int | None = None
    ):
        """Initialize VAE.

        Parameters
        ----------
        encoder : Encoder
            Encoder network.
        decoder : Decoder
            Decoder network.
        spatial : int | None
            Number of spatial dimensions in latent space (e.g., 2 for images).
            If None, assumes flat 1D latent representation.
        """
        super().__init__(encoder=encoder, decoder=decoder)
        self.spatial = spatial
        latent_dim = encoder.latent_dim
        if encoder.latent_dim != decoder.latent_dim:
            msg = "Encoder and Decoder latent dimensions must match for VAE."
            raise ValueError(msg)

        # For spatial latents, use 1x1 convolutions; for flat, use linear
        if spatial is not None:
            self.fc_mean = ConvNd(
                latent_dim,
                latent_dim,
                spatial=spatial,
                kernel_size=1,
            )
            self.fc_log_var = ConvNd(
                latent_dim,
                latent_dim,
                spatial=spatial,
                kernel_size=1,
            )
        else:
            self.fc_mean = nn.Linear(latent_dim, latent_dim)
            self.fc_log_var = nn.Linear(latent_dim, latent_dim)

        self.loss_func = VAELoss()

    def forward(self, batch: Batch) -> TensorBTSC:
        return self.forward_with_latent(batch)[0]

    def forward_with_latent(self, batch: Batch) -> tuple[TensorBTSC, TensorBNC]:
        encoded = self.encode(batch)
        # Split along channel dim (last dim)
        mean, log_var = encoded.chunk(2, dim=-1)
        z = self.reparametrize(mean, log_var)
        decoded = self.decode(z)
        return decoded, encoded

    def reparametrize(self, mean: TensorBNC, log_var: TensorBNC) -> TensorBNC:
        """Reparameterisation trick.

        Samples z ~ N(mean, sigma) during training, but returns the mean
        deterministically in evaluation mode. This makes `model.eval()` produce
        deterministic reconstructions while training remains stochastic.
        """
        if not self.training:
            return mean

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, batch: Batch) -> TensorBNC:
        # Shape: (B, T, spatial..., C) or (B, T, C) for flat
        h = self.encoder.encode(batch)

        # Check if latent is spatial (has spatial dims) or flat
        is_spatial = self.spatial is not None and h.dim() > 3

        # Process each timestep separately through fc layers
        outputs_mean = []
        outputs_logvar = []
        for idx in range(h.shape[1]):  # Loop over time dimension
            h_t = h[:, idx, ...]  # (B, spatial..., C) or (B, C) for flat

            if is_spatial:
                # Rearrange to (B, C, spatial...) for conv layers
                h_t = rearrange(h_t, "B ... C -> B C ...")

            mean_t = self.fc_mean(h_t)
            logvar_t = self.fc_log_var(h_t)

            if is_spatial:
                # Rearrange back to (B, spatial..., C)
                mean_t = rearrange(mean_t, "B C ... -> B ... C")
                logvar_t = rearrange(logvar_t, "B C ... -> B ... C")

            outputs_mean.append(mean_t)
            outputs_logvar.append(logvar_t)

        # Stack back into (B, T, spatial..., C) or (B, T, C)
        mean = torch.stack(outputs_mean, dim=1)
        log_var = torch.stack(outputs_logvar, dim=1)

        concat_dim = -1  # Always concatenate along channel dimension (last dim)
        return torch.cat([mean, log_var], dim=concat_dim)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        assert self.loss_func is not None
        loss = self.loss_func(self, batch)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        return loss
