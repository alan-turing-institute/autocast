from __future__ import annotations

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
        flow_ode_steps: int = 1,
        n_steps_output: int = 4,
        n_channels_out: int = 1,
        time_sampling_method: str = "uniform",
    ) -> None:
        # Store core hyperparameters and optional prebuilt backbone.
        super().__init__()
        self.flow_matching_model = backbone
        self.flow_ode_steps = max(flow_ode_steps, 1)
        self.n_steps_output = n_steps_output
        self.n_channels_out = n_channels_out
        self.time_sampling_method = time_sampling_method

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

        dt = torch.tensor(1.0 / self.flow_ode_steps, device=device, dtype=dtype)
        history = []

        for _ in range(self.flow_ode_steps):
            f_i = self.flow_field(z, t, x, global_cond)
            history.append(f_i)

            if len(history) == 1:
                # Step 1: Euler
                z = z + dt * history[0]
            elif len(history) == 2:
                # Step 2: Adams-Bashforth 2
                z = z + dt / 2.0 * (3.0 * history[1] - history[0])
            else:
                # Step 3+: Adams-Bashforth 3
                z = z + dt / 12.0 * (
                    23.0 * history[2] - 16.0 * history[1] + 5.0 * history[0]
                )
                history.pop(0)

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

        # Use f32 for calculating variables and velocities to prevent bfloat16 underflow
        # near t=1
        target_states_f32 = target_states.to(dtype=torch.float32)
        z0 = torch.randn_like(target_states_f32, requires_grad=True)

        if self.time_sampling_method == "logit-normal":
            u = torch.randn(
                batch_size, device=target_states.device, dtype=torch.float32
            )
            t = torch.sigmoid(u)
        else:
            t = torch.rand(batch_size, device=target_states.device, dtype=torch.float32)

        t_broadcast = t.view(batch_size, *([1] * (target_states.ndim - 1)))
        zt = (1 - t_broadcast) * z0 + t_broadcast * target_states_f32

        target_velocity = target_states_f32 - z0

        # Keep inference flow compute in its native precision (e.g. bf16 if mixed
        # precision)
        v_pred = self.flow_field(
            zt.to(dtype=target_states.dtype),
            t.to(dtype=target_states.dtype),
            input_states,
            global_cond=batch.global_cond,
        )
        return torch.mean((v_pred.to(dtype=torch.float32) - target_velocity) ** 2)
