from __future__ import annotations

from typing import Any

import torch
from torch import nn

from auto_cast.nn.unet import TemporalUNetBackbone
from auto_cast.processors.base import Processor
from auto_cast.types import EncodedBatch, Tensor


class FlowMatchingProcessor(Processor):
    """Processor that wraps a flow-matching generative model."""

    def __init__(
        self,
        *,
        flow_matching_model: nn.Module | None = None,
        stride: int = 1,
        teacher_forcing_ratio: float = 0.0,
        max_rollout_steps: int = 1,
        loss_func: nn.Module | None = None,
        learning_rate: float = 1e-3,
        flow_ode_steps: int = 1,
        n_steps_output: int = 1,
        n_channels_out: int = 1,
        backbone_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Store core hyperparameters and optional prebuilt backbone.
        super().__init__(
            stride=stride,
            teacher_forcing_ratio=teacher_forcing_ratio,
            max_rollout_steps=max_rollout_steps,
            loss_func=loss_func or nn.MSELoss(),
            **kwargs,
        )
        self.flow_matching_model = flow_matching_model
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

    def forward(self, z: Tensor, t: Tensor, x: Tensor) -> Tensor:
        """
        The vector field over the tangent space of output states (z)
        conditioned on input states (x) at time (t).

        Args:
            z: Current output states of shape (B, T_out, *spatial, C_out).
            t: Time tensor of shape (B,).
            x: Conditioning inputs of shape (B, T_in, *spatial, C_in).
        Returns:
            Time derivative of output states with the same shape as `z`.
        """
        self._maybe_build_backbone(x)
        assert self.flow_matching_model is not None  # for type checkers
        return self.flow_matching_model(z, t, x)

    def map(self, x: Tensor) -> Tensor:
        """
        Maps inputs states (x) to output states (z) by integrating the flow ODE.
        Starting from noise, Euler-integrate the learned vector field until t=1.

        Args:
            x: Conditioning inputs of shape (B, T_in, *spatial, C_in).
        Returns:
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
            z = z + dt * self.forward(z, t, x)
            t = t + dt
        return z

    def training_step(self, batch: EncodedBatch, batch_idx: int) -> Tensor:  # noqa: ARG002
        """
        Perform a single flow-matching training step on encoded inputs/targets.

        Args:
            batch: EncodedBatch containing inputs and target outputs.
            batch_idx: Index of the batch (unused).
        Returns:
            Scalar training loss tensor.
        """
        input_states = batch.encoded_inputs
        target_states = batch.encoded_output_fields

        # Validate target shape against configured output dimensions.
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

        # Sample initial noise and interpolation time, then compute target velocity.
        z0 = torch.randn_like(target_states, requires_grad=True)
        t = torch.rand(batch_size, device=target_states.device, dtype=target_states.dtype)
        t_broadcast = t.view(batch_size, *([1] * (target_states.ndim - 1)))
        zt = (1 - t_broadcast) * z0 + t_broadcast * target_states

        target_velocity = target_states - z0
        v_pred = self.forward(zt, t, input_states)
        flow_loss = torch.mean((v_pred - target_velocity) ** 2)

        batch_size = batch.encoded_inputs.shape[0]
        self.log("train_loss", flow_loss, prog_bar=True, batch_size=batch_size)
        self.log(
            "train_flow_matching_loss",
            flow_loss,
            prog_bar=False,
            batch_size=batch_size,
        )
        return flow_loss

    def validation_step(self, batch: EncodedBatch, batch_idx: int) -> Tensor:
        """
        Compute validation loss directly on the map output (no rollout).

        Args:
            batch: EncodedBatch containing inputs and target outputs.
            batch_idx: Index of the batch (unused).
        Returns:
            Scalar validation loss tensor.
        """
        target_states = batch.encoded_output_fields
        if (
            target_states.shape[1] != self.n_steps_output
            or target_states.shape[-1] != self.n_channels_out
        ):
            msg = (
                "Validation target shape does not match configured output dimensions "
                f"(expected T_out={self.n_steps_output}, C_out={self.n_channels_out}, "
                f"got T_out={target_states.shape[1]}, C_out={target_states.shape[-1]})."
            )
            raise ValueError(msg)

        preds = self.map(batch.encoded_inputs)
        loss = self.loss_func(preds, target_states)

        batch_size = batch.encoded_inputs.shape[0]
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("val_rollout_loss", loss, prog_bar=False, batch_size=batch_size)
        return loss
