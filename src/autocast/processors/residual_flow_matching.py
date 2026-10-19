"""Flow matching for residual forecast trajectories."""

from __future__ import annotations

from typing import Any

from torch import nn

from autocast.nn.noise.source import SourceDistribution
from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.processors.residual_normalization import ResidualNormalizer
from autocast.processors.residual_reference import ReferenceTrajectory
from autocast.types import EncodedBatch, Tensor


class ResidualFlowMatchingProcessor(FlowMatchingProcessor):
    """Transport noise to a standardized processor-coordinate residual.

    Targets are full output states minus a deterministic reference in the same
    coordinates. With an encoder E, this is E(target) - reference(E(input)),
    rather than E(target - ambient_reference). Sampling reconstructs complete
    processor states before any decoder is applied.
    """

    standardizer: ResidualNormalizer | None  # Legacy attribute alias.

    def __init__(
        self,
        *,
        backbone: nn.Module,
        reference: ReferenceTrajectory,
        flow_ode_steps: int = 1,
        n_steps_output: int = 4,
        n_channels_out: int = 1,
        integrator: str = "euler",
        source: SourceDistribution | None = None,
        residual_normalizer: ResidualNormalizer | None = None,
        standardizer: ResidualNormalizer | None = None,
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
        if standardizer is not None:
            if residual_normalizer is not None:
                msg = "Cannot provide both standardizer and residual_normalizer."
                raise ValueError(msg)
            residual_normalizer = standardizer
        if residual_normalizer is not None and not isinstance(
            residual_normalizer, ResidualNormalizer
        ):
            msg = "residual_normalizer must be a ResidualNormalizer."
            raise TypeError(msg)
        if residual_normalizer is not None and (
            residual_normalizer.n_steps_output != n_steps_output
            or residual_normalizer.n_channels != n_channels_out
        ):
            msg = (
                "Residual normalizer dimensions must match the processor "
                f"(expected T={n_steps_output}, C={n_channels_out}; received "
                f"T={residual_normalizer.n_steps_output}, "
                f"C={residual_normalizer.n_channels})."
            )
            raise ValueError(msg)
        self.residual_normalizer = residual_normalizer
        self.register_load_state_dict_pre_hook(self._load_legacy_statistics)

    def __getattr__(self, name: str) -> Any:
        """Resolve the legacy name whether normalization is configured or omitted."""
        if name == "standardizer":
            return self.residual_normalizer
        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        """Map legacy writes to the single registered residual normalizer."""
        super().__setattr__(
            "residual_normalizer" if name == "standardizer" else name, value
        )

    @staticmethod
    def _load_legacy_statistics(_module, state_dict, prefix, *args) -> None:
        # Prefix-aware loading also covers nested models and EMA callback state.
        for key in list(state_dict):
            if key.startswith(prefix + "standardizer."):
                canonical = (
                    prefix
                    + "residual_normalizer."
                    + key.removeprefix(prefix + "standardizer.")
                )
                if canonical in state_dict:
                    msg = (
                        "Checkpoint contains both standardizer and residual_normalizer."
                    )
                    raise RuntimeError(msg)
                state_dict[canonical] = state_dict.pop(key)

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

    def map_from_source(
        self,
        x: Tensor,
        global_cond: Tensor | None,
        *,
        source_states: Tensor,
    ) -> Tensor:
        """Generate a residual from supplied noise and add it to the reference.

        The inherited ``map`` samples noise and dispatches here, so sampled and
        explicit-source forecasts use the same output coordinates.
        """
        reference = self.reference_trajectory(x, global_cond)
        generated_residual = super().map_from_source(
            x, global_cond, source_states=source_states
        )
        if self.residual_normalizer is not None:
            generated_residual = self.residual_normalizer.denormalize(
                generated_residual
            )
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
        if self.residual_normalizer is not None:
            target_residual = self.residual_normalizer.normalize(target_residual)
        return self._flow_matching_loss(
            target_states=target_residual,
            conditioning=batch.encoded_inputs,
            global_cond=batch.global_cond,
        )
