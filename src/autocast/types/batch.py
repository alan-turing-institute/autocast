from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import torch

from autocast.types.types import (
    Tensor,
    TensorBC,
    TensorBNC,
    TensorBSC,
    TensorBTSC,
    TensorC,
    TensorNC,
    TensorS,
    TensorSC,
    TensorTSC,
)

# Generic batch type variable
BatchT = TypeVar("BatchT")


@dataclass
class Sample:
    """A batch in input data space."""

    input_fields: TensorTSC
    output_fields: TensorTSC
    constant_scalars: TensorC | None
    constant_fields: TensorSC | None
    boundary_conditions: TensorS | None


@dataclass
class EncodedSample:
    """A batch after being processed by an Encoder."""

    encoded_inputs: TensorBNC
    encoded_output_fields: TensorBNC
    global_cond: TensorNC | None
    encoded_info: dict[str, Tensor]


@dataclass
class Batch:
    """A batch in input data space."""

    input_fields: TensorBTSC
    output_fields: TensorBTSC
    constant_scalars: TensorBC | None
    constant_fields: TensorBSC | None
    boundary_conditions: TensorS | None = None

    def repeat(self, m: int) -> "Batch":
        """Repeat batch members.

        This interleaves the batch dimension by repeating each sample m times.

        For example, for m=3, a batch with samples
        0, 1, 2, ...
        becomes
        0, 0, 0, 1, 1, 1, 2, 2, 2, ...
        """
        return Batch(
            input_fields=self.input_fields.repeat_interleave(m, dim=0),
            output_fields=self.output_fields.repeat_interleave(m, dim=0),
            constant_scalars=(
                self.constant_scalars.repeat_interleave(m, dim=0)
                if self.constant_scalars is not None
                else None
            ),
            constant_fields=(
                self.constant_fields.repeat_interleave(m, dim=0)
                if self.constant_fields is not None
                else None
            ),
            boundary_conditions=(
                self.boundary_conditions.repeat_interleave(m, dim=0)
                if self.boundary_conditions is not None
                else None
            ),
        )

    def to(self, device: torch.device | str) -> "Batch":
        """Move batch to device."""
        return Batch(
            input_fields=self.input_fields.to(device),
            output_fields=self.output_fields.to(device),
            constant_scalars=(
                self.constant_scalars.to(device)
                if self.constant_scalars is not None
                else None
            ),
            constant_fields=(
                self.constant_fields.to(device)
                if self.constant_fields is not None
                else None
            ),
            boundary_conditions=(
                self.boundary_conditions.to(device)
                if self.boundary_conditions is not None
                else None
            ),
        )


@dataclass
class EncodedBatch:
    """A batch after being processed by an Encoder."""

    encoded_inputs: TensorBNC
    encoded_output_fields: TensorBNC
    global_cond: TensorBNC | None
    encoded_info: dict[str, Tensor]

    def repeat(self, m: int) -> "EncodedBatch":
        """Repeat batch members.

        This interleaves the batch dimension by repeating each sample m times.

        For example, for m=3, a batch with samples
        0, 1, 2, ...
        becomes
        0, 0, 0, 1, 1, 1, 2, 2, 2, ...

        """
        return EncodedBatch(
            encoded_inputs=self.encoded_inputs.repeat_interleave(m, dim=0),
            encoded_output_fields=self.encoded_output_fields.repeat_interleave(
                m, dim=0
            ),
            global_cond=(
                self.global_cond.repeat_interleave(m, dim=0)
                if self.global_cond is not None
                else None
            ),
            encoded_info={
                k: v.repeat_interleave(m, dim=0) for k, v in self.encoded_info.items()
            },
        )

    def to(self, device: torch.device | str) -> "EncodedBatch":
        """Move batch to device."""
        return EncodedBatch(
            encoded_inputs=self.encoded_inputs.to(device),
            encoded_output_fields=self.encoded_output_fields.to(device),
            global_cond=(
                self.global_cond.to(device) if self.global_cond is not None else None
            ),
            encoded_info={k: v.to(device) for k, v in self.encoded_info.items()},
        )


def collate_batches(samples: Sequence[Sample]) -> Batch:
    """Stack a sequence of `Batch` instances along the batch dimension."""
    if len(samples) == 0:
        msg = "collate_batches expects at least one sample"
        raise ValueError(msg)

    def _stack_optional(getter: str) -> Tensor | None:
        values = [getattr(sample, getter) for sample in samples]
        if all(v is None for v in values):
            return None
        if any(v is None for v in values):
            msg = f"Field '{getter}' is inconsistently None across samples"
            raise ValueError(msg)
        return torch.stack(values, dim=0)  # type: ignore[arg-type]

    input_fields = torch.stack([sample.input_fields for sample in samples], dim=0)
    output_fields = torch.stack([sample.output_fields for sample in samples], dim=0)
    constant_scalars = _stack_optional("constant_scalars")
    constant_fields = _stack_optional("constant_fields")
    boundary_conditions = _stack_optional("boundary_conditions")

    return Batch(
        input_fields=input_fields,
        output_fields=output_fields,
        constant_scalars=constant_scalars,
        constant_fields=constant_fields,
        boundary_conditions=boundary_conditions,
    )


def collate_encoded_samples(samples: Sequence[EncodedSample]) -> EncodedBatch:
    """Stack a sequence of `EncodedSample` instances along the batch dimension."""
    if len(samples) == 0:
        msg = "collate_encoded_samples expects at least one sample"
        raise ValueError(msg)

    def _stack_optional(getter: str) -> Tensor | None:
        values = [getattr(sample, getter) for sample in samples]
        if all(v is None for v in values):
            return None
        if any(v is None for v in values):
            msg = f"Field '{getter}' is inconsistently None across samples"
            raise ValueError(msg)
        return torch.stack(values, dim=0)  # type: ignore[arg-type]

    encoded_inputs = torch.stack([sample.encoded_inputs for sample in samples], dim=0)
    encoded_output_fields = torch.stack(
        [sample.encoded_output_fields for sample in samples], dim=0
    )
    global_cond = _stack_optional("global_cond")

    # Merge encoded_info dicts
    encoded_info: dict[str, Tensor] = {}
    first_info = samples[0].encoded_info
    for key in first_info:
        values = [sample.encoded_info.get(key) for sample in samples]
        if all(v is not None for v in values):
            encoded_info[key] = torch.stack(values, dim=0)  # type: ignore[arg-type]

    return EncodedBatch(
        encoded_inputs=encoded_inputs,
        encoded_output_fields=encoded_output_fields,
        global_cond=global_cond,
        encoded_info=encoded_info,
    )
