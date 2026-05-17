from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor


class MaskedFlowMatchingProcessor(Processor):
    """Processor that wraps a flow-matching generative model."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        flow_ode_steps: int = 1,
        n_steps_output: int = 4,
        n_channels_out: int = 1,
        mask: Tensor | None = None,
    ) -> None:
        # Store core hyperparameters and optional prebuilt backbone.
        super().__init__()
        self.flow_matching_model = backbone
        self.flow_ode_steps = max(flow_ode_steps, 1)
        self.n_steps_output = n_steps_output
        self.n_channels_out = n_channels_out
        # Assume 2D mask for now. Store as a buffer so it's moved with the module.
        if mask is not None:
            if mask.ndim != 2:
                raise ValueError(f"Mask must be 2D tensor, got shape {mask.shape}")
            mask = rearrange(mask, "w h -> 1 1 w h 1")
            # register as buffer so `.to(device)` moves it
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    def flow_field(
        self, z: Tensor, t: Tensor, x: Tensor, global_cond: Tensor | None = None
    ) -> Tensor:
        """Flow matching vector field.

        The vector field over the tangent space of output states (z).
        conditioned on input states (x) at time (t).

        Args:
            z: Current output states of shape (B, T_out, *spatial, C_out).
            t: Time tensor of shape (B,).
            x: Conditioning inputs of shape (B, T_in, *spatial, C_in).
            global_cond: Optional non-spatial conditioning/modulation tensor.

        Returns
        -------
            Time derivative of output states with the same shape as `z`.
        """
        return self.flow_matching_model(z, t=t, cond=x, global_cond=global_cond)

    def forward(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Alias to map for Lightning/PyTorch compatibility."""
        return self.map(x, global_cond)

    def _apply_mask(self, z: Tensor) -> Tensor:
        """Apply mask to tensor if mask is set."""
        if self.mask is None:
            return z

        return z * self.mask

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Map inputs states (x) to output states (z) by integrating the flow ODE.

        Starting from noise, Euler-integrate the learned vector field until t=1.

        Args:
            x: Conditioning inputs of shape (B, T_in, *spatial, C_in).

        Returns
        -------
            Generated outputs of shape (B, T_out, *spatial, C_out).
        """
        batch_size = x.shape[0]
        device, dtype = x.device, x.dtype

        # Initialize noisy sample and scalar time for each batch element.
        spatial_shape = tuple(x.shape[2:-1])
        z_shape = (batch_size, self.n_steps_output, *spatial_shape, self.n_channels_out)
        z = torch.randn(z_shape, device=device, dtype=dtype)
        t = torch.zeros(batch_size, device=device, dtype=dtype)

        # Simple fixed-step Euler integration over the flow field.
        dt = torch.tensor(1.0 / self.flow_ode_steps, device=device, dtype=dtype)

        # Apply mask to inputs and noise if mask is set
        x = self._apply_mask(x)
        z = self._apply_mask(z)
        for _ in range(self.flow_ode_steps):
            z = z + dt * self.flow_field(z, t, x, global_cond)
            t = t + dt
            # Apply mask to updated state of z if mask is set
            z = self._apply_mask(z)
        return z

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Compute flow-matching loss for a batch."""
        input_states = batch.encoded_inputs
        target_states = batch.encoded_output_fields

        if (
            target_states.shape[1] != self.n_steps_output
            or target_states.shape[-1] != self.n_channels_out
        ):
            msg = (
                "Target shape does not match configured output dimensions "
                f"(expected T_out={self.n_steps_output}, C_out={self.n_channels_out}, "
                f"got T_out={target_states.shape[1]}, C_out={target_states.shape[-1]})."
            )
            raise ValueError(msg)

        batch_size = target_states.shape[0]

        z0 = torch.randn_like(target_states)

        # Apply mask to initial noise, inputs, and target states if mask is set
        z0 = self._apply_mask(z0)
        input_states = self._apply_mask(input_states)
        target_states = self._apply_mask(target_states)

        t = torch.rand(
            batch_size, device=target_states.device, dtype=target_states.dtype
        )
        t_broadcast = t.view(batch_size, *([1] * (target_states.ndim - 1)))
        zt = (1 - t_broadcast) * z0 + t_broadcast * target_states

        target_velocity = target_states - z0
        v_pred = self.flow_field(zt, t, input_states, global_cond=batch.global_cond)

        squared_diff = (v_pred - target_velocity) ** 2
        if self.mask is not None:
            # Compute mean loss over valid masked elements
            mask = self.mask.to(dtype=squared_diff.dtype)
            return (squared_diff * mask).sum() / mask.expand_as(squared_diff).sum()

        return torch.mean(squared_diff)
