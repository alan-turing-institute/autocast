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
    TensorDBM,
    TensorDM,
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
            output_fields=(
                self.output_fields.repeat_interleave(m, dim=0)
                if self.output_fields is not None
                else self.output_fields
            ),  # type: ignore[arg-type]
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
            output_fields=(
                self.output_fields.to(device)
                if self.output_fields is not None
                else self.output_fields
            ),  # type: ignore[arg-type]
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


@dataclass
class ListSample:
    """A sample containing a list of Samples (one per dataset) and a mask."""

    inner: list[Sample]
    # Dataset-by-ensemble mask for missing data combinations across datasets.
    mask: TensorDM | None


@dataclass
class ListBatch:
    """A batch containing a list of Batches (one per dataset) and a mask."""

    inner: list[Batch]
    mask: TensorDBM | None
    output_fields: TensorBTSC | None = None

    @property
    def input_fields(self) -> TensorBTSC:
        return self.inner[0].input_fields

    def repeat(self, m: int) -> "ListBatch":
        return ListBatch(
            inner=[b.repeat(m) for b in self.inner],
            mask=self.mask.repeat_interleave(m, dim=0)
            if self.mask is not None
            else None,
            output_fields=self.output_fields.repeat_interleave(m, dim=0)
            if self.output_fields is not None
            else None,
        )

    def to(self, device: torch.device | str) -> "ListBatch":
        return ListBatch(
            inner=[b.to(device) for b in self.inner],
            mask=self.mask.to(device) if self.mask is not None else None,
            output_fields=self.output_fields.to(device)
            if self.output_fields is not None
            else None,
        )
