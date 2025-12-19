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


@dataclass
class EncodedSample:
    """A batch after being processed by an Encoder."""

    encoded_inputs: TensorBNC
    encoded_output_fields: TensorBNC
    label: TensorNC | None
    encoded_info: dict[str, Tensor]


@dataclass
class Batch:
    """A batch in input data space."""

    input_fields: TensorBTSC
    output_fields: TensorBTSC
    constant_scalars: TensorBC | None
    constant_fields: TensorBSC | None


@dataclass
class EncodedBatch:
    """A batch after being processed by an Encoder."""

    encoded_inputs: TensorBTSC
    encoded_output_fields: TensorBTSC
    label: TensorBNC | None
    encoded_info: dict[str, Tensor]


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

    return Batch(
        input_fields=input_fields,
        output_fields=output_fields,
        constant_scalars=constant_scalars,
        constant_fields=constant_fields,
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
    label = _stack_optional("label")

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
        label=label,
        encoded_info=encoded_info,
    )
