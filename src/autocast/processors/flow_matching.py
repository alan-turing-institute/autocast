from __future__ import annotations

from typing import Any

import torch
from torch import nn

from autocast.nn.unet import TemporalUNetBackbone
from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor


class FlowMatchingProcessor(Processor):
    """Processor that wraps a flow-matching generative model."""

    def __init__(
        self,
        *,
        flow_matching_model: nn.Module | None = None,
        backbone: nn.Module | None = None,
        schedule: Any | None = None,
        denoiser_type: str | None = None,
        stride: int = 1,
        teacher_forcing_ratio: float = 0.0,
        max_rollout_steps: int = 10,
        loss_func: nn.Module | None = None,
        learning_rate: float = 1e-4,
        flow_ode_steps: int = 1,
        n_steps_output: int = 4,
        n_channels_out: int = 1,
        backbone_kwargs: dict[str, Any] | None = None,
        residual: bool = False,
        **kwargs: Any,
    ) -> None:
        # Store core hyperparameters and optional prebuilt backbone.
        super().__init__(
            stride=stride,
            teacher_forcing_ratio=teacher_forcing_ratio,
            max_rollout_steps=max_rollout_steps,
            loss_func=loss_func or nn.MSELoss(),
            residual=residual,
            **kwargs,
        )
        self.flow_matching_model = flow_matching_model or backbone
        self.schedule = schedule  # accepted for API compatibility
        self.denoiser_type = denoiser_type
        self.learning_rate = learning_rate
        self.flow_ode_steps = max(flow_ode_steps, 1)
        self.n_steps_output = n_steps_output
        self.n_channels_out = n_channels_out
        self.backbone_kwargs = backbone_kwargs or {}

    def _maybe_build_backbone(self, x: Tensor) -> None:
        """Lazily build TemporalUNetBackbone when no model is provided."""
        if self.flow_matching_model is not None:
            return

        # Infer in/out channels from configured temporal/channel counts.
        t_in = x.shape[1]
        c_in = x.shape[-1]
        t_out = self.n_steps_output
        c_out = self.n_channels_out

        self.flow_matching_model = TemporalUNetBackbone(
            in_channels=t_out * c_out,
            out_channels=t_out * c_out,
            cond_channels=t_in * c_in,
            **self.backbone_kwargs,
        )

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
        self._maybe_build_backbone(x)
        assert self.flow_matching_model is not None  # for type checkers
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
            z_abs = z
            if self.residual:
                base = x[:, -1:, ...]
                if base.shape[1] != z.shape[1]:
                    base = base.expand(
                        batch_size, z.shape[1], *spatial_shape, self.n_channels_out
                    )
                z_abs = z + base
            z = z + dt * self.flow_field(z_abs, t, x)
            t = t + dt
        if self.residual:
            base = x[:, -1:, ...]
            if base.shape[1] != z.shape[1]:
                base = base.expand(
                    batch_size, z.shape[1], *spatial_shape, self.n_channels_out
                )
            return base + z
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

        self._maybe_build_backbone(input_states)

        batch_size = target_states.shape[0]

        if self.residual:
            base = input_states[:, -1:, ...]
            if base.shape[1] != target_states.shape[1]:
                base = base.expand(batch_size, target_states.shape[1], *base.shape[2:])
            target_states = target_states - base
        z0 = torch.randn_like(target_states, requires_grad=True)
        t = torch.rand(
            batch_size, device=target_states.device, dtype=target_states.dtype
        )
        t_broadcast = t.view(batch_size, *([1] * (target_states.ndim - 1)))
        zt = (1 - t_broadcast) * z0 + t_broadcast * target_states

        target_velocity = target_states - z0
        if self.residual:
            base = input_states[:, -1:, ...]
            if base.shape[1] != zt.shape[1]:
                base = base.expand(batch_size, zt.shape[1], *base.shape[2:])
            z_abs = zt + base
        else:
            z_abs = zt
        v_pred = self.flow_field(z_abs, t, input_states)
        return torch.mean((v_pred - target_velocity) ** 2)
