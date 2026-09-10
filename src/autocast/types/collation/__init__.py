from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any, TypeVar

import torch

from autocast.types.batch import Batch, EncodedBatch, EncodedSample, Sample
from autocast.types.types import Tensor

BatchClsT = TypeVar("BatchClsT")


def _stack_optional(name: str, values: Sequence[Any]) -> Tensor | None:
    """Stack a tensor field, tolerating a field that is absent from every sample."""
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        msg = f"Field '{name}' is inconsistently None across samples"
        raise ValueError(msg)
    return torch.stack(list(values), dim=0)


def _stack_mappings(values: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]:
    """Stack each entry of a mapping field, keeping only keys present in all samples."""
    stacked: dict[str, Tensor] = {}
    for key in values[0]:
        entries = [value.get(key) for value in values]
        if all(entry is not None for entry in entries):
            stacked[key] = torch.stack(entries, dim=0)  # type: ignore[arg-type]
    return stacked


def _collate(
    samples: Sequence[Any], batch_cls: type[BatchClsT], what: str
) -> BatchClsT:
    """Stack every field of `samples` along a new leading batch dimension.

    Fields are read from `dataclasses.fields()` of the first sample and passed
    to `batch_cls` by name, so a field added to both the sample and batch
    dataclasses is collated without changing this function.

    Args:
        samples: Samples to stack. Must be non-empty and share a field layout.
        batch_cls: Batch dataclass to construct; its field names must match
            those of the samples.
        what: Name of the calling collate function, used in error messages.

    Returns:
        An instance of `batch_cls` holding the stacked fields.
    """
    if len(samples) == 0:
        msg = f"{what} expects at least one sample"
        raise ValueError(msg)

    collated: dict[str, Any] = {}
    for field in fields(samples[0]):
        values = [getattr(sample, field.name) for sample in samples]
        if isinstance(values[0], Mapping):
            collated[field.name] = _stack_mappings(values)
        else:
            collated[field.name] = _stack_optional(field.name, values)
    return batch_cls(**collated)


def collate_batches(samples: Sequence[Sample]) -> Batch:
    """Stack a sequence of `Sample` instances along the batch dimension."""
    return _collate(samples, Batch, "collate_batches")


def collate_encoded_samples(samples: Sequence[EncodedSample]) -> EncodedBatch:
    """Stack a sequence of `EncodedSample` instances along the batch dimension."""
    return _collate(samples, EncodedBatch, "collate_encoded_samples")
