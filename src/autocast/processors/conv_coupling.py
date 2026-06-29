from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor

_LOG_2PI = math.log(2.0 * math.pi)


class _ConvAffineCoupling(nn.Module):
    """One conditional affine-coupling layer with a convolutional conditioner.

    A spatial checkerboard mask splits the field into a *kept* half ``y_a`` and
    a *transformed* half ``y_b``. A small CNN reads the kept half (transformed
    positions zeroed) together with the conditioning field ``x_t`` and emits a
    per-pixel log-scale ``s`` and shift ``t`` for the transformed half. The map
    is an exact bijection with a triangular Jacobian, so the log-determinant is
    just the sum of ``s`` over the transformed positions.

    The data->latent direction (``encode``) is used for the log-likelihood; the
    latent->data direction (``decode``) is used for sampling. Both run in a
    single network pass (no autoregressive unrolling).
    """

    mask: Tensor  # checkerboard split, registered as a buffer in __init__

    def __init__(
        self,
        *,
        channels: int,
        cond_channels: int,
        spatial_shape: tuple[int, int],
        parity: int,
        hidden_channels: int,
        n_blocks: int,
        kernel_size: int,
        init_scale_bound: float,
    ) -> None:
        super().__init__()
        h, w = spatial_shape
        # Spatial checkerboard: mask True = kept (conditioning) position.
        ii = torch.arange(h).view(h, 1)
        jj = torch.arange(w).view(1, w)
        mask = ((ii + jj) % 2 == parity).to(torch.float32).view(1, 1, h, w)
        self.register_buffer("mask", mask)

        pad = kernel_size // 2
        layers: list[nn.Module] = []
        in_ch = channels + cond_channels
        for _ in range(n_blocks):
            layers += [
                nn.Conv2d(in_ch, hidden_channels, kernel_size, padding=pad),
                nn.ReLU(),
            ]
            in_ch = hidden_channels
        # Final conv emits (s, t); zero-initialised so the layer starts as the
        # identity (s=t=0) for a stable, unit-Jacobian training start.
        final = nn.Conv2d(in_ch, 2 * channels, kernel_size, padding=pad)
        nn.init.zeros_(final.weight)
        if final.bias is not None:
            nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)
        # Learnable per-channel bound on the log-scale: s = tanh(s_raw) * bound.
        # It MUST be initialised non-zero. With both the conv (s_raw=0) and the
        # bound at zero, the scale is a dead saddle -- d s / d s_raw = bound = 0
        # AND d s / d bound = tanh(s_raw) = 0 -- so neither the scale weights nor
        # the bound ever receive gradient and the flow can only shift, never
        # scale (predictive std frozen at the base std of 1). A positive init
        # keeps the identity start (s = tanh(0) * bound = 0) while making
        # d s / d s_raw = bound != 0, so the proper NLL can learn the scale.
        self.log_scale_bound = nn.Parameter(
            torch.full((1, channels, 1, 1), float(init_scale_bound))
        )

    def _params(self, kept: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        """Return (s, t) for the transformed half, masked to those positions."""
        h = self.net(torch.cat([kept, cond], dim=1))
        s_raw, t = h.chunk(2, dim=1)
        # tanh-bound the log-scale, then restrict s and t to transformed cells.
        s = torch.tanh(s_raw) * self.log_scale_bound
        free = 1.0 - self.mask
        return s * free, t * free

    def encode(self, y: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        """Map data -> latent; return (z, log|det dz/dy|) summed over features."""
        s, t = self._params(y * self.mask, cond)
        z = y * self.mask + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        ladj = (-s).flatten(1).sum(dim=1)
        return z, ladj

    def decode(self, z: Tensor, cond: Tensor) -> Tensor:
        """Map latent -> data (inverse of :meth:`encode`)."""
        s, t = self._params(z * self.mask, cond)
        return z * self.mask + (1.0 - self.mask) * (z * torch.exp(s) + t)


class ConvCouplingFlowProcessor(Processor):
    """Convolutional affine-coupling normalizing flow (one-pass exact likelihood).

    A stack of spatial checkerboard affine-coupling layers, each conditioned on
    the current state ``x_t`` through a CNN, models the one-step predictive
    density ``p(x_{t+1} | x_t)`` exactly. Unlike the autoregressive flow, both
    log-likelihood and sampling cost a single forward pass per layer (no
    sequential dimension unrolling), so scoring is fast for spatial fields.

    The field is handled channels-last as ``(B, T, *spatial, C)`` externally and
    folded to ``(B, T*C, H, W)`` for the convolutions. Targets spatial fields
    (1D as an ``[L, 1]`` grid, 2D as ``[H, W]``); scalar fields have no spatial
    structure and use a flat autoregressive flow instead.
    """

    def __init__(
        self,
        *,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        spatial_shape: Sequence[int] = (1, 1),
        global_cond_features: int = 0,
        transforms: int = 8,
        hidden_channels: int = 64,
        n_blocks: int = 2,
        kernel_size: int = 3,
        init_scale_bound: float = 2.0,
    ) -> None:
        super().__init__()
        spatial = tuple(int(s) for s in spatial_shape)
        if len(spatial) != 2:
            msg = (
                "ConvCouplingFlowProcessor expects a 2D spatial_shape (H, W); "
                f"got {spatial}. Represent 1D fields as [L, 1]."
            )
            raise ValueError(msg)
        self.n_steps_input = int(n_steps_input)
        self.n_steps_output = int(n_steps_output)
        self.n_channels_in = int(n_channels_in)
        self.n_channels_out = int(n_channels_out)
        self.spatial_shape = spatial
        self.global_cond_features = int(global_cond_features)

        # Conv channel counts: target = T_out * C_out; conditioner = T_in * C_in
        # plus any global-conditioning features broadcast as constant channels.
        self.target_channels = self.n_steps_output * self.n_channels_out
        self.cond_channels = (
            self.n_steps_input * self.n_channels_in + self.global_cond_features
        )

        self.layers = nn.ModuleList(
            _ConvAffineCoupling(
                channels=self.target_channels,
                cond_channels=self.cond_channels,
                spatial_shape=spatial,
                parity=i % 2,
                hidden_channels=int(hidden_channels),
                n_blocks=int(n_blocks),
                kernel_size=int(kernel_size),
                init_scale_bound=float(init_scale_bound),
            )
            for i in range(int(transforms))
        )

    # -- layout helpers -------------------------------------------------------
    def _to_conv(self, field: Tensor, steps: int, channels: int) -> Tensor:
        """(B, T, H, W, C) channels-last -> (B, T*C, H, W) channels-first."""
        h, w = self.spatial_shape
        expected = steps * h * w * channels
        per_sample = math.prod(field.shape[1:])
        if per_sample != expected:
            msg = (
                "ConvCouplingFlowProcessor received a field with "
                f"{per_sample} elements per sample but expected {expected} = "
                f"{steps} (n_steps) x {h} x {w} (spatial_shape) x {channels} "
                "(n_channels). Check the processor dims against the data."
            )
            raise ValueError(msg)
        x = field.reshape(field.shape[0], steps, h, w, channels)
        return x.permute(0, 1, 4, 2, 3).reshape(field.shape[0], steps * channels, h, w)

    def _cond(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Build the (B, cond_channels, H, W) conditioning stack from x_t."""
        cond = self._to_conv(x, self.n_steps_input, self.n_channels_in)
        if global_cond is not None and self.global_cond_features > 0:
            h, w = self.spatial_shape
            g = global_cond.reshape(
                global_cond.shape[0], self.global_cond_features, 1, 1
            )
            cond = torch.cat([cond, g.expand(-1, -1, h, w)], dim=1)
        return cond

    # -- flow -----------------------------------------------------------------
    def coupling_layers(self) -> list[_ConvAffineCoupling]:
        """The coupling layers in order, typed concretely.

        ``nn.ModuleList`` iteration is statically typed as plain ``nn.Module``,
        which hides ``encode``/``decode``; recover the concrete type here so the
        flow passes (and tests) read cleanly.
        """
        return [cast(_ConvAffineCoupling, layer) for layer in self.layers]

    def _log_prob(self, y_cf: Tensor, cond: Tensor) -> Tensor:
        z = y_cf
        ladj = y_cf.new_zeros(y_cf.shape[0])
        for layer in self.coupling_layers():
            z, dl = layer.encode(z, cond)
            ladj = ladj + dl
        base = (-0.5 * (z**2 + _LOG_2PI)).flatten(1).sum(dim=1)
        return base + ladj

    def _sample(self, cond: Tensor) -> Tensor:
        h, w = self.spatial_shape
        z = torch.randn(
            cond.shape[0],
            self.target_channels,
            h,
            w,
            device=cond.device,
            dtype=cond.dtype,
        )
        for layer in reversed(self.coupling_layers()):
            z = layer.decode(z, cond)
        return z  # (B, T_out*C_out, H, W)

    # -- Processor API --------------------------------------------------------
    def forward(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        return self.map(x, global_cond)

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Draw one stochastic next-state member, (B, T_out, *spatial, C_out)."""
        cond = self._cond(x, global_cond)
        y_cf = self._sample(cond)
        h, w = self.spatial_shape
        y = y_cf.reshape(x.shape[0], self.n_steps_output, self.n_channels_out, h, w)
        return y.permute(0, 1, 3, 4, 2)  # -> (B, T_out, H, W, C_out)

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Exact negative log-likelihood of the next state given the current."""
        cond = self._cond(batch.encoded_inputs, batch.global_cond)
        y_cf = self._to_conv(
            batch.encoded_output_fields, self.n_steps_output, self.n_channels_out
        )
        log_prob = self._log_prob(y_cf, cond)
        return -log_prob.mean()
