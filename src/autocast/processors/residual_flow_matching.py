"""Flow matching for residual forecast trajectories."""

from __future__ import annotations

from torch import nn

from autocast.nn.noise.source import FlowSource
from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.processors.residual_normalization import ResidualStandardizer
from autocast.processors.residual_reference import ReferenceTrajectory
from autocast.types import EncodedBatch, Tensor


class ResidualFlowMatchingProcessor(FlowMatchingProcessor):
    """Transport noise to a standardized processor-coordinate residual."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        reference: ReferenceTrajectory,
        flow_ode_steps: int = 1,
        n_steps_output: int = 4,
        n_channels_out: int = 1,
        integrator: str = "euler",
        source: FlowSource | None = None,
        standardizer: ResidualStandardizer | None = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            flow_ode_steps=flow_ode_steps,
            n_steps_output=n_steps_output,
            n_channels_out=n_channels_out,
            integrator=integrator,
            source=source,
        )
        self.reference = reference
        if standardizer is not None and (
            standardizer.n_steps_output != n_steps_output
            or standardizer.n_channels != n_channels_out
        ):
            msg = (
                "Residual standardizer dimensions must match the processor "
                f"(expected T={n_steps_output}, C={n_channels_out}; received "
                f"T={standardizer.n_steps_output}, C={standardizer.n_channels})."
            )
            raise ValueError(msg)
        self.standardizer = standardizer

    def reference_trajectory(
        self,
        x: Tensor,
        global_cond: Tensor | None,
    ) -> Tensor:
        """Generate and validate the configured reference trajectory."""
        reference = self.reference(x, global_cond)
        expected_shape = self._output_shape(x)
        if reference.shape != expected_shape:
            msg = (
                f"Reference trajectory must have shape {expected_shape}; "
                f"received {tuple(reference.shape)}."
            )
            raise ValueError(msg)
        return reference

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Generate a residual trajectory and add it to the reference."""
        reference = self.reference_trajectory(x, global_cond)
        generated_residual = super().map(x, global_cond)
        if self.standardizer is not None:
            generated_residual = self.standardizer.unstandardize(generated_residual)
        return reference + generated_residual

    def target_residual(self, batch: EncodedBatch) -> Tensor:
        """Return and validate target-minus-reference residuals for a batch."""
        target = batch.encoded_output_fields
        self._validate_output_shape(target)
        reference = self.reference_trajectory(
            batch.encoded_inputs,
            batch.global_cond,
        )
        if reference.shape != target.shape:
            msg = (
                "Reference and target trajectories must have identical shapes; "
                f"received {tuple(reference.shape)} and {tuple(target.shape)}."
            )
            raise ValueError(msg)
        return target - reference

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Match the source distribution to target-minus-reference residuals."""
        target_residual = self.target_residual(batch)
        if self.standardizer is not None:
            target_residual = self.standardizer.standardize(target_residual)
        return self._flow_matching_loss(
            target_states=target_residual,
            conditioning=batch.encoded_inputs,
            global_cond=batch.global_cond,
        )
