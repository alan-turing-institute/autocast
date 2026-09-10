from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, TypeVar, cast

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

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

# Generic batch type variable
BatchT = TypeVar("BatchT")

# Self type for the field-transform mixin, so transforms return the concrete
# dataclass they were called on rather than the mixin.
TensorFieldsT = TypeVar("TensorFieldsT", bound="TensorFieldsMixin")


def _pin_memory_if_available(tensor: Tensor) -> Tensor:
    if not torch.cuda.is_available():
        return tensor
    return tensor.pin_memory()


class TensorFieldsMixin:
    """Batch-lifecycle transforms for dataclasses of batched tensor fields.

    Each transform walks `dataclasses.fields()` rather than naming its fields,
    so adding a field to a batch dataclass extends `to`, `pin_memory` and
    `repeat` without editing them.

    A field is transformed if it holds a `Tensor`, or a mapping whose values
    are tensors (e.g. `EncodedBatch.encoded_info`). Anything else is passed
    through unchanged, which is how absent optional fields (`None`) survive.

    Only the batch types mix this in: every transform here assumes a leading
    batch dimension, and the pre-collation `Sample` types are never pinned,
    moved or repeated individually.
    """

    def map_tensors(
        self: TensorFieldsT, fn: Callable[[Tensor], Tensor]
    ) -> TensorFieldsT:
        """Return a copy of this instance with `fn` applied to every tensor field.

        Args:
            fn: Transform applied to each tensor field, and to each tensor
                value of a mapping field.

        Returns:
            A new instance of the same type. Non-tensor fields are shared with
            the original rather than copied.
        """
        instance = cast("DataclassInstance", self)
        updates: dict[str, Any] = {}
        for field in fields(instance):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                updates[field.name] = fn(value)
            elif isinstance(value, Mapping):
                updates[field.name] = {
                    key: fn(item) if isinstance(item, Tensor) else item
                    for key, item in value.items()
                }
        return cast("TensorFieldsT", replace(instance, **updates))

    def to(self: TensorFieldsT, device: torch.device | str) -> TensorFieldsT:
        """Move every tensor field to `device`."""
        return self.map_tensors(lambda tensor: tensor.to(device))

    def pin_memory(self: TensorFieldsT) -> TensorFieldsT:
        """Pin every CPU tensor field for faster host-to-device transfer.

        This is the hook `DataLoader(pin_memory=True)` looks for on a collated
        batch, rather than something callers invoke directly.
        """
        return self.map_tensors(_pin_memory_if_available)

    def repeat(self: TensorFieldsT, m: int) -> TensorFieldsT:
        """Repeat batch members.

        This interleaves the batch dimension by repeating each sample m times.

        For example, for m=3, a batch with samples
        0, 1, 2, ...
        becomes
        0, 0, 0, 1, 1, 1, 2, 2, 2, ...
        """
        return self.map_tensors(lambda tensor: tensor.repeat_interleave(m, dim=0))


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
class Batch(TensorFieldsMixin):
    """A batch in input data space."""

    input_fields: TensorBTSC
    output_fields: TensorBTSC
    constant_scalars: TensorBC | None
    constant_fields: TensorBSC | None
    boundary_conditions: TensorS | None = None


@dataclass
class EncodedBatch(TensorFieldsMixin):
    """A batch after being processed by an Encoder."""

    encoded_inputs: TensorBNC
    encoded_output_fields: TensorBNC
    global_cond: TensorBNC | None
    encoded_info: dict[str, Tensor]
