"""One-pass stochastic spacetime prediction for ensemble objectives."""

from collections.abc import Callable
from functools import partial

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from autocast.nn.spacetime_vit import SpatiotemporalViTBackbone
from autocast.processors.base import Processor
from autocast.processors.masking import masked_mean, random_context_mask
from autocast.types import EncodedBatch


class SpacetimeViTProcessor(Processor[EncodedBatch]):
    """Predict future frame slots with global noise modulation, e.g. for CRPS.

    The existing ensemble model supplies independent members and the CRPS
    loss. Future slots contain zeros; target states are never model inputs.
    """

    conditioning = "frames"

    def __init__(
        self, backbone: SpatiotemporalViTBackbone, n_noise_input_channels: int = 1024
    ) -> None:
        super().__init__()
        if (
            backbone.conditioning != self.conditioning
            or not backbone.use_precomputed_modulation
        ):
            msg = (
                f"Expected {self.conditioning} conditioning and precomputed modulation."
            )
            raise ValueError(msg)
        if n_noise_input_channels <= 0:
            msg = "n_noise_input_channels must be positive."
            raise ValueError(msg)
        self.backbone = backbone
        self.n_noise_input_channels = n_noise_input_channels
        self.modulation_proj = (
            nn.Identity()
            if n_noise_input_channels == backbone.mod_features
            else nn.Linear(n_noise_input_channels, backbone.mod_features)
        )

    def forward(
        self, x: Tensor, x_noise: Tensor, global_cond: Tensor | None = None
    ) -> Tensor:
        if x_noise.shape != (x.shape[0], self.n_noise_input_channels):
            msg = "Expected one noise vector per sample."
            raise ValueError(msg)
        future = x.new_zeros(
            x.shape[0],
            self.backbone.n_steps_output,
            *x.shape[2:4],
            self.backbone.in_channels,
        )
        return self.backbone(
            future, t=self.modulation_proj(x_noise), cond=x, global_cond=global_cond
        )

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:
        noise = torch.randn(
            x.shape[0], self.n_noise_input_channels, device=x.device, dtype=x.dtype
        )
        return self(x, noise, global_cond)

    def loss(self, batch: EncodedBatch) -> Tensor:
        return self.loss_func(
            self.map(batch.encoded_inputs, batch.global_cond),
            batch.encoded_output_fields,
        )


class SpacetimeViTMaskedWindowProcessor(SpacetimeViTProcessor):
    """One-pass masked-window ensembles with a configurable unreduced score.

    Own the training ensemble so a mask is drawn once per real example,
    before expanding members. Configure ``model.loss_func=null`` to use
    this loss through the existing ensemble model. Inference remains fixed
    context forecasting through ``map`` and the model's usual member expansion.
    """

    conditioning = "mask"

    def __init__(
        self,
        backbone: SpatiotemporalViTBackbone,
        *,
        n_steps_input: int,
        n_steps_output: int,
        n_members: int,
        loss_func: nn.Module,
        n_noise_input_channels: int = 1024,
        context_mask: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__(backbone, n_noise_input_channels)
        if min(n_steps_input, n_steps_output) <= 0 or n_members < 2:
            msg = "Expected positive context/forecast lengths and at least two members."
            raise ValueError(msg)
        full_steps = n_steps_input + n_steps_output
        if (backbone.n_steps_input, backbone.n_steps_output) != (
            full_steps,
            full_steps,
        ):
            msg = "Masked backbone must process the full context+forecast window."
            raise ValueError(msg)
        self.n_steps_input, self.n_steps_output = n_steps_input, n_steps_output
        self.n_members, self.loss_func = n_members, loss_func
        self.context_mask = (
            partial(random_context_mask, atleast=1)
            if context_mask is None
            else context_mask
        )

    def _predict_window(
        self, clean: Tensor, mask: Tensor, noise: Tensor, global_cond: Tensor | None
    ) -> Tensor:
        pred = self.backbone(
            clean,
            t=self.modulation_proj(noise),
            cond=mask.to(clean.dtype),
            global_cond=global_cond,
        )
        return torch.where(mask, clean, pred)

    def forward(
        self, x: Tensor, x_noise: Tensor, global_cond: Tensor | None = None
    ) -> Tensor:
        if x.shape[1] != self.n_steps_input:
            msg = "Input context length does not match n_steps_input."
            raise ValueError(msg)
        if x_noise.shape != (x.shape[0], self.n_noise_input_channels):
            msg = "Expected one noise vector per sample."
            raise ValueError(msg)
        future = x.new_zeros(x.shape[0], self.n_steps_output, *x.shape[2:])
        clean = torch.cat((x, future), dim=1)
        mask = torch.zeros_like(clean, dtype=torch.bool)
        mask[:, : self.n_steps_input] = True
        pred = self._predict_window(clean, mask, x_noise, global_cond)
        return pred[:, self.n_steps_input :]

    def loss(self, batch: EncodedBatch) -> Tensor:
        if (
            batch.encoded_inputs.shape[1] != self.n_steps_input
            or batch.encoded_output_fields.shape[1] != self.n_steps_output
        ):
            msg = "Batch context/forecast lengths do not match the processor."
            raise ValueError(msg)
        window = torch.cat((batch.encoded_inputs, batch.encoded_output_fields), dim=1)
        mask = self.context_mask(window).expand_as(window)
        clean = torch.where(mask, window, 0)
        members = self.n_members
        noise = torch.randn(
            window.shape[0] * members,
            self.n_noise_input_channels,
            device=window.device,
            dtype=window.dtype,
        )
        global_cond = (
            repeat(batch.global_cond, "b ... -> (b m) ...", m=members)
            if batch.global_cond is not None
            else None
        )
        pred = self._predict_window(
            repeat(clean, "b ... -> (b m) ...", m=members),
            repeat(mask, "b ... -> (b m) ...", m=members),
            noise,
            global_cond,
        )
        pred = rearrange(pred, "(b m) ... -> b ... m", m=members)
        score = self.loss_func(pred, window)
        if score.shape != window.shape:
            msg = "Masked ensemble loss requires an unreduced score per target element."
            raise ValueError(msg)
        return masked_mean(score, ~mask)
