from __future__ import annotations

from typing import Any

import torch
from torch import nn

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor


class FlowMatchingProcessor(Processor):
    """Processor that wraps a flow-matching generative model."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        schedule: Any | None = None,
        denoiser_type: str | None = None,
        teacher_forcing_ratio: float = 0.0,
        max_rollout_steps: int = 10,
        loss_func: nn.Module | None = None,
        learning_rate: float = 1e-4,
        flow_ode_steps: int = 1,
        n_steps_output: int = 4,
        n_channels_out: int = 1,
        **kwargs: Any,
    ) -> None:
        # Store core hyperparameters and optional prebuilt backbone.
        super().__init__(
            teacher_forcing_ratio=teacher_forcing_ratio,
            max_rollout_steps=max_rollout_steps,
            loss_func=loss_func or nn.MSELoss(),
            **kwargs,
        )
        self.flow_matching_model = backbone
        self.schedule = schedule  # accepted for API compatibility
        self.denoiser_type = denoiser_type
        self.learning_rate = learning_rate
        self.flow_ode_steps = max(flow_ode_steps, 1)
        self.n_steps_output = n_steps_output
        self.n_channels_out = n_channels_out

    def flow_field(self, z: Tensor, t: Tensor, x: Tensor) -> Tensor:
        """Flow matching vector field.

        The vector field over the tangent space of output states (z).
        conditioned on input states (x) at time (t).

        Args:
            z: Current output states of shape (B, T_out, *spatial, C_out).
            t: Time tensor of shape (B,).
            x: Conditioning inputs of shape (B, T_in, *spatial, C_in).

        Returns
        -------
            Time derivative of output states with the same shape as `z`.
        """
        return self.flow_matching_model(z, t, x)

    def forward(self, x: Tensor) -> Tensor:
        """Alias to map for Lightning/PyTorch compatibility."""
        return self.map(x)

    def map(self, x: Tensor) -> Tensor:
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
        for _ in range(self.flow_ode_steps):
            z = z + dt * self.flow_field(z, t, x)
            t = t + dt
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

        z0 = torch.randn_like(target_states, requires_grad=True)
        t = torch.rand(
            batch_size, device=target_states.device, dtype=target_states.dtype
        )
        t_broadcast = t.view(batch_size, *([1] * (target_states.ndim - 1)))
        zt = (1 - t_broadcast) * z0 + t_broadcast * target_states

        target_velocity = target_states - z0
        v_pred = self.flow_field(zt, t, input_states)
        return torch.mean((v_pred - target_velocity) ** 2)
