from __future__ import annotations

import torch
from torch import nn

from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.types import EncodedBatch, Tensor


class FlowMatchingMaskedWindowProcessor(FlowMatchingProcessor):
    """Flow matching over a masked input+output window.

    Training concatenates the input and output windows into one trajectory,
    samples a random temporal context mask, clamps observed context states
    along the flow path, and learns to denoise the full trajectory window.
    At inference time, the input window is clamped as the observed context
    and the generated output slice is returned.

    The context-mask sampler mirrors ``random_context_mask`` in
    https://github.com/francois-rozet/lola/blob/main/lola/emulation.py.
    """

    def __init__(
        self,
        *,
        backbone: nn.Module,
        flow_ode_steps: int = 1,
        n_steps_input: int | None = None,
        n_steps_output: int = 4,
        n_channels_out: int = 1,
        lmbda: float = 1.0,
        rho: float = 1.0,
        atleast: int = 0,
    ) -> None:
        super().__init__(
            backbone=backbone,
            flow_ode_steps=flow_ode_steps,
            n_steps_output=n_steps_output,
            n_channels_out=n_channels_out,
        )
        self.n_steps_input = n_steps_input
        self.lmbda = lmbda
        self.rho = rho
        self.atleast = atleast

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Generate the output window while clamping the observed input window."""
        input_steps = self._resolve_input_steps(x)
        full_steps = input_steps + self.n_steps_output
        self._validate_full_window_backbone(full_steps)
        self._validate_input_channels(x)

        batch_size = x.shape[0]
        device, dtype = x.device, x.dtype
        spatial_shape = tuple(x.shape[2:-1])
        full_shape = (
            batch_size,
            full_steps,
            *spatial_shape,
            self.n_channels_out,
        )

        clean_context = torch.zeros(full_shape, device=device, dtype=dtype)
        clean_context[:, :input_steps, ...] = x
        context_mask = self._forecast_context_mask(x, full_steps)

        z = torch.randn(full_shape, device=device, dtype=dtype)
        z = torch.where(context_mask, clean_context, z)
        t = torch.zeros(batch_size, device=device, dtype=dtype)
        dt = torch.tensor(1.0 / self.flow_ode_steps, device=device, dtype=dtype)

        for _ in range(self.flow_ode_steps):
            z = torch.where(context_mask, clean_context, z)
            v = self.flow_field(
                z,
                t,
                context_mask.to(dtype=dtype),
                global_cond=global_cond,
            )
            v = torch.where(context_mask, torch.zeros_like(v), v)
            z = z + dt * v
            t = t + dt

        z = torch.where(context_mask, clean_context, z)
        return z[:, input_steps:, ...]

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Compute masked full-window flow-matching loss for a batch."""
        input_states = batch.encoded_inputs
        target_states = batch.encoded_output_fields
        self._validate_output_shape(target_states)
        self._validate_input_steps(input_states)
        self._validate_input_channels(input_states)

        full_window = self._build_full_window(input_states, target_states)
        full_steps = full_window.shape[1]
        self._validate_full_window_backbone(full_steps)

        context_mask = self._random_context_mask(full_window)
        z0 = torch.randn_like(full_window)
        z0 = torch.where(context_mask, full_window, z0)

        batch_size = full_window.shape[0]
        t = torch.rand(batch_size, device=full_window.device, dtype=full_window.dtype)
        t_broadcast = t.view(batch_size, *([1] * (full_window.ndim - 1)))
        zt = (1 - t_broadcast) * z0 + t_broadcast * full_window

        target_velocity = full_window - z0
        v_pred = self.flow_field(
            zt,
            t,
            context_mask.to(dtype=full_window.dtype),
            global_cond=batch.global_cond,
        )
        return torch.mean((v_pred - target_velocity) ** 2)

    def _build_full_window(self, input_states: Tensor, target_states: Tensor) -> Tensor:
        if input_states.shape[0] != target_states.shape[0]:
            msg = (
                "Input and target batch sizes must match "
                f"(got {input_states.shape[0]} and {target_states.shape[0]})."
            )
            raise ValueError(msg)
        if input_states.shape[2:-1] != target_states.shape[2:-1]:
            msg = (
                "Input and target spatial shapes must match "
                f"(got {input_states.shape[2:-1]} and {target_states.shape[2:-1]})."
            )
            raise ValueError(msg)
        if input_states.shape[-1] != target_states.shape[-1]:
            msg = (
                "Masked-window flow matching requires input and output channel "
                f"counts to match (got {input_states.shape[-1]} and "
                f"{target_states.shape[-1]})."
            )
            raise ValueError(msg)
        return torch.cat([input_states, target_states], dim=1)

    def _resolve_input_steps(self, input_states: Tensor) -> int:
        input_steps = input_states.shape[1]
        if self.n_steps_input is not None and input_steps != self.n_steps_input:
            msg = (
                "Input shape does not match configured input steps "
                f"(expected T_in={self.n_steps_input}, got T_in={input_steps})."
            )
            raise ValueError(msg)
        return input_steps

    def _validate_input_steps(self, input_states: Tensor) -> None:
        self._resolve_input_steps(input_states)

    def _validate_input_channels(self, input_states: Tensor) -> None:
        if input_states.shape[-1] != self.n_channels_out:
            msg = (
                "Masked-window flow matching requires input channels to match "
                f"`n_channels_out`={self.n_channels_out}, got "
                f"{input_states.shape[-1]}."
            )
            raise ValueError(msg)

    def _validate_full_window_backbone(self, full_steps: int) -> None:
        model_steps = getattr(self.flow_matching_model, "n_steps_output", None)
        if model_steps is not None and model_steps != full_steps:
            msg = (
                "Masked-window flow matching requires the backbone "
                f"`n_steps_output` to be the full input+output window "
                f"({full_steps}), got {model_steps}."
            )
            raise ValueError(msg)

        cond_steps = getattr(self.flow_matching_model, "n_steps_input", None)
        if cond_steps is not None and cond_steps != full_steps:
            msg = (
                "Masked-window flow matching conditions on a full-window mask, "
                f"so backbone `n_steps_input` must be {full_steps}, got "
                f"{cond_steps}."
            )
            raise ValueError(msg)

        cond_channels = getattr(self.flow_matching_model, "cond_channels", None)
        if cond_channels is not None and cond_channels != self.n_channels_out:
            msg = (
                "Masked-window flow matching passes the mask as conditioning, "
                f"so backbone `cond_channels` must be {self.n_channels_out}, "
                f"got {cond_channels}."
            )
            raise ValueError(msg)

    def _mask_atleast_for(self, total_steps: int) -> int:
        atleast = self.atleast
        if atleast < 0:
            msg = f"`atleast` must be non-negative, got {atleast}."
            raise ValueError(msg)
        return min(atleast, total_steps - 1)

    def _random_context_mask(self, x: Tensor) -> Tensor:
        """Sample a random prefix/suffix temporal context mask.

        The number of observed context states is drawn from a Poisson rate
        and wrapped into ``[atleast, total_steps)``. With probability ``rho``,
        the context is a prefix; otherwise it is a suffix.

        Mirrors ``random_context_mask`` in
        https://github.com/francois-rozet/lola/blob/main/lola/emulation.py.
        """
        batch_size, total_steps = x.shape[:2]
        atleast = self._mask_atleast_for(total_steps)

        rate = torch.full(
            (batch_size, 1),
            fill_value=self.lmbda,
            device=x.device,
        )
        context = torch.poisson(rate).long()
        context = context % (total_steps - atleast) + atleast

        index = torch.arange(total_steps, device=x.device)
        if self.rho <= 0.0:
            mask = index >= total_steps - context
        elif self.rho >= 1.0:
            mask = index < context
        else:
            prefix = index < context
            suffix = index >= total_steps - context
            choose_prefix = torch.rand((batch_size, 1), device=x.device) < self.rho
            mask = torch.where(choose_prefix, prefix, suffix)

        mask = mask.reshape(batch_size, total_steps, *([1] * (x.ndim - 2)))
        return mask.expand_as(x)

    def _forecast_context_mask(self, x: Tensor, full_steps: int) -> Tensor:
        mask = torch.zeros(
            x.shape[0],
            full_steps,
            *x.shape[2:-1],
            self.n_channels_out,
            device=x.device,
            dtype=torch.bool,
        )
        mask[:, : x.shape[1], ...] = True
        return mask
