from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    """Shape contract for one side (input or output) of a model.

    Describes a channels-last field stack `(T, *spatial, C)` without its batch
    dimension. Encoder, processor and decoder construction all need the same
    three numbers, so they are carried together rather than threaded around as
    loose `n_channels` / `n_steps` / `spatial_resolution` arguments.

    Args:
        n_steps: Number of time steps.
        n_channels: Number of channels per time step.
        spatial_resolution: Size of each spatial dimension.
    """

    n_steps: int
    n_channels: int
    spatial_resolution: tuple[int, ...]

    @property
    def n_flat_channels(self) -> int:
        """Channel count once the time axis is folded into channels."""
        return self.n_steps * self.n_channels

    @classmethod
    def from_batch_shape(cls, shape: Sequence[int]) -> "FieldSpec":
        """Build a spec from a channels-last batch shape `(B, T, *spatial, C)`.

        Args:
            shape: Shape of a channels-last field tensor, batch dimension first.

        Returns:
            The corresponding `FieldSpec`.

        Raises:
            ValueError: If `shape` has fewer than three dimensions, and so
                cannot carry a batch, time and channel axis.
        """
        if len(shape) < 3:
            msg = (
                "Expected a channels-last batch shape (B, T, *spatial, C) with "
                f"at least 3 dimensions, got {tuple(shape)}"
            )
            raise ValueError(msg)
        return cls(
            n_steps=int(shape[1]),
            n_channels=int(shape[-1]),
            spatial_resolution=tuple(int(size) for size in shape[2:-1]),
        )


@dataclass(frozen=True)
class IOSpec:
    """Input and output shape contracts for a model.

    Keeping both sides explicit is what allows a model to predict fields that
    differ from the ones it consumes — a different channel count, a different
    number of steps, or a different spatial resolution. Code that needs the
    two sides to agree should ask via `is_autoregressive` rather than assuming
    it.

    An `IOSpec` describes whichever space its owner works in: data space for a
    dataset or encoder/decoder pair, latent space for a processor.

    Args:
        inputs: Shape contract for the fields the model consumes.
        outputs: Shape contract for the fields the model predicts.
    """

    inputs: FieldSpec
    outputs: FieldSpec

    @property
    def is_autoregressive(self) -> bool:
        """Whether predictions can be fed back in as inputs.

        True when outputs occupy the same channel and spatial layout as
        inputs, which is what autoregressive rollout requires. The step counts
        may differ, since rollout advances a sliding window.
        """
        return (
            self.inputs.n_channels == self.outputs.n_channels
            and self.inputs.spatial_resolution == self.outputs.spatial_resolution
        )

    @classmethod
    def from_batch_shapes(
        cls, input_shape: Sequence[int], output_shape: Sequence[int]
    ) -> "IOSpec":
        """Build a spec from a batch's input and output field shapes.

        Args:
            input_shape: Shape of the input fields, `(B, T_in, *spatial, C_in)`.
            output_shape: Shape of the output fields, `(B, T_out, *spatial, C_out)`.

        Returns:
            The corresponding `IOSpec`.
        """
        return cls(
            inputs=FieldSpec.from_batch_shape(input_shape),
            outputs=FieldSpec.from_batch_shape(output_shape),
        )
