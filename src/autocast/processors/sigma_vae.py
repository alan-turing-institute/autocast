from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor

_LOG_2PI = math.log(2.0 * math.pi)


def _mlp(in_features: int, out_features: int, hidden: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_features
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.SiLU()]
        prev = h
    layers.append(nn.Linear(prev, out_features))
    return nn.Sequential(*layers)


class SigmaVAEProcessor(Processor):
    """Conditional variational autoencoder with a calibrated (learned-sigma) decoder.

    Models the one-step predictive density ``p(x_{t+1} | x_t)`` as a conditional
    VAE: an encoder ``q(z | x_{t+1}, x_t)``, a standard-normal prior ``p(z)``, and
    a Gaussian decoder ``p(x_{t+1} | z, x_t) = N(mu(z, x_t), sigma^2)``. The
    decoder output standard deviation ``sigma`` is a **learned** per-channel
    parameter (the "sigma-VAE" construction of Rybkin et al., 2021): a calibrated
    decoder automatically balances the reconstruction term against the KL, so the
    ELBO weight ``beta`` stays at 1 and no per-problem tuning of the
    reconstruction/KL trade-off is needed. This is what lets one configuration
    calibrate across fields with very different noise scales.

    Everything is handled on the flattened field, so the same model applies to
    scalar, vector, 1D and 2D fields alike (an ensemble member is a full
    generative draw ``z ~ p(z)``, ``x = mu(z, x_t) + sigma * xi``, including the
    decoder noise so the predictive spread is calibrated, not just the latent
    spread).

    The output sigma is a single per-channel scalar, so a decaying learning rate
    cannot move it far within a budget: started at the wrong scale it ships at the
    wrong scale (a fixed ``init_log_sigma`` over-disperses any field whose
    one-step residual is not ``exp(init_log_sigma)``). It therefore defaults to
    ``init_log_sigma="auto"``: on the first training batch sigma is set to the
    data's per-channel one-step residual scale ``std(x_{t+1} - x_t)`` so it starts
    calibrated and only fine-tunes. This is data-driven, so it holds identically
    on toy and real data with no per-dataset constant. Pass a float to override.
    """

    _sigma_inited: Tensor  # registered buffer: has the data-driven init run yet

    def __init__(
        self,
        *,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        spatial_shape: Sequence[int] = (1, 1),
        global_cond_features: int = 0,
        latent_dim: int = 32,
        hidden_features: Sequence[int] = (128, 128),
        beta: float = 1.0,
        init_log_sigma: float | str = "auto",
        min_log_sigma: float = -7.0,
        max_log_sigma: float = 5.0,
    ) -> None:
        super().__init__()
        self.n_steps_input = int(n_steps_input)
        self.n_steps_output = int(n_steps_output)
        self.n_channels_in = int(n_channels_in)
        self.n_channels_out = int(n_channels_out)
        self.spatial_shape = tuple(int(s) for s in spatial_shape)
        self.global_cond_features = int(global_cond_features)
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)

        spatial = int(math.prod(self.spatial_shape)) if self.spatial_shape else 1
        self.spatial = spatial
        self.features = self.n_steps_output * spatial * self.n_channels_out
        self.context = (
            self.n_steps_input * spatial * self.n_channels_in
            + self.global_cond_features
        )

        hidden = tuple(int(h) for h in hidden_features)
        self.encoder = _mlp(self.features + self.context, 2 * self.latent_dim, hidden)
        self.decoder = _mlp(self.latent_dim + self.context, self.features, hidden)
        # Learned per-channel output log-sigma (the calibration knob). Tiled over
        # time/space at use; channel is the fastest-varying flattened axis.
        # "auto" defers the value to the first training batch (data-driven init,
        # see _maybe_init_sigma); a float is taken as-is and marked initialised.
        self._auto_sigma = isinstance(init_log_sigma, str) and init_log_sigma == "auto"
        init_value = 0.0 if self._auto_sigma else float(init_log_sigma)
        self.log_sigma = nn.Parameter(torch.full((self.n_channels_out,), init_value))
        # Buffer (not a plain attr) so the "already initialised" state survives
        # checkpointing -- scoring loads a trained sigma and must never re-init.
        self.register_buffer(
            "_sigma_inited", torch.tensor(not self._auto_sigma), persistent=True
        )

    def _context(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        ctx = x.reshape(x.shape[0], -1)
        if global_cond is not None and self.global_cond_features > 0:
            ctx = torch.cat(
                [ctx, global_cond.reshape(global_cond.shape[0], -1)], dim=-1
            )
        if ctx.shape[1] != self.context:
            msg = (
                "SigmaVAEProcessor conditioning has "
                f"{ctx.shape[1]} features per sample but expected {self.context} = "
                "n_steps_input * prod(spatial_shape) * n_channels_in + "
                "global_cond_features. Check the processor dims against the data."
            )
            raise ValueError(msg)
        return ctx

    def _sigma_flat(self) -> Tensor:
        """Per-feature sigma, (features,), from the per-channel learned log-sigma."""
        log_sigma = self.log_sigma.clamp(self.min_log_sigma, self.max_log_sigma)
        reps = self.n_steps_output * self.spatial
        return log_sigma.repeat(reps).exp()

    def _reshape_out(self, flat: Tensor) -> Tensor:
        return flat.reshape(
            flat.shape[0], self.n_steps_output, *self.spatial_shape, self.n_channels_out
        )

    def forward(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        return self.map(x, global_cond)

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Draw one generative member: z ~ p(z), x = mu(z, x_t) + sigma * xi."""
        ctx = self._context(x, global_cond)
        z = torch.randn(
            ctx.shape[0], self.latent_dim, device=ctx.device, dtype=ctx.dtype
        )
        mu = self.decoder(torch.cat([z, ctx], dim=-1))
        sample = mu + self._sigma_flat() * torch.randn_like(mu)
        return self._reshape_out(sample)

    @torch.no_grad()
    def _maybe_init_sigma(self, batch: EncodedBatch) -> None:
        """Set the output sigma to the data's one-step residual scale (once).

        Runs on the first *training* batch only. It is skipped outside training so
        it never fires on Lightning's validation sanity-check pass -- that runs
        under ``inference_mode`` (the in-place init would raise) and a validation
        batch is the wrong source for the calibration anyway.

        The per-channel target is ``std(x_{t+1} - x_t)`` -- the conditional spread
        the decoder must match -- falling back to ``std(x_{t+1})`` when input/output
        channels differ so the subtraction is undefined. Clamped to the model's
        log-sigma range. Under DDP each rank initialises from its own first batch,
        so the value is broadcast from rank 0 to keep the parameter identical
        across ranks (DDP synchronises gradients, not this in-place init).
        """
        if not self.training or bool(self._sigma_inited):
            return
        y = batch.encoded_output_fields
        x = batch.encoded_inputs
        same = x.shape[-1] == y.shape[-1]
        resid = (y - x) if same else y
        std = resid.reshape(-1, self.n_channels_out).std(dim=0).clamp_min(1e-6)
        self.log_sigma.copy_(std.log().clamp(self.min_log_sigma, self.max_log_sigma))
        self._sigma_inited.fill_(True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(self.log_sigma.data, src=0)

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Negative ELBO with a calibrated (learned-sigma) Gaussian decoder."""
        ctx = self._context(batch.encoded_inputs, batch.global_cond)
        y = batch.encoded_output_fields.reshape(
            batch.encoded_output_fields.shape[0], -1
        )
        if y.shape[1] != self.features:
            msg = (
                "SigmaVAEProcessor output field has "
                f"{y.shape[1]} features per sample but expected {self.features} = "
                "n_steps_output * prod(spatial_shape) * n_channels_out. Check the "
                "processor dims against the data."
            )
            raise ValueError(msg)
        self._maybe_init_sigma(batch)

        mu_z, logvar_z = self.encoder(torch.cat([y, ctx], dim=-1)).chunk(2, dim=-1)
        std_z = torch.exp(0.5 * logvar_z)
        z = mu_z + std_z * torch.randn_like(std_z)
        mu = self.decoder(torch.cat([z, ctx], dim=-1))

        sigma = self._sigma_flat()
        log_sigma = sigma.log()
        recon = 0.5 * (((y - mu) / sigma) ** 2 + 2.0 * log_sigma + _LOG_2PI).sum(dim=-1)
        kl = 0.5 * (mu_z**2 + logvar_z.exp() - 1.0 - logvar_z).sum(dim=-1)
        return (recon + self.beta * kl).mean()
