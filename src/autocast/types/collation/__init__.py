from collections.abc import Sequence

import torch

from autocast.types.batch import (
    Batch,
    EncodedBatch,
    EncodedSample,
    Sample,
    SphericalBatch,
)
from autocast.types.types import Tensor


def collate_batches(samples: Sequence[Sample]) -> Batch:
    """Stack a sequence of `Batch` instances along the batch dimension.

    Samples on a grid on the sphere give a `SphericalBatch` on that grid. Samples on
    different grids, or spherical samples mixed with plain ones, raise `ValueError`.
    """
    if len(samples) == 0:
        msg = "collate_batches expects at least one sample"
        raise ValueError(msg)

    grid_types = {getattr(sample, "grid_type", None) for sample in samples}
    if len(grid_types) > 1:
        names = sorted(str(grid_type) for grid_type in grid_types)
        msg = f"collate_batches got samples on different grids: {names}"
        raise ValueError(msg)
    (grid_type,) = grid_types

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

    if grid_type is not None:
        return SphericalBatch(
            input_fields=input_fields,
            output_fields=output_fields,
            constant_scalars=constant_scalars,
            constant_fields=constant_fields,
            boundary_conditions=boundary_conditions,
            grid_type=grid_type,
        )
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
