from collections.abc import Sequence

import torch

from autocast.data.multidataset import ListBatch, ListSample
from autocast.types.batch import Batch, EncodedBatch, EncodedSample, Sample
from autocast.types.types import Tensor


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


def collate_list_batches(samples: Sequence[ListSample]) -> ListBatch:
    """Stack a sequence of `ListSample` instances along the batch dimension.

    Each ListSample contains a list of Samples (one per dataset) and a mask.
    This function collates across the batch dimension, producing a ListBatch
    containing a list of Batches (one per dataset) and a stacked mask.
    """
    if len(samples) == 0:
        msg = "collate_list_batches expects at least one sample"
        raise ValueError(msg)

    # Collate each dataset batch separately
    num_datasets = len(samples[0].inner)
    inner_batches: list[Batch] = []
    for dataset_idx in range(num_datasets):
        # Get all samples for this dataset across the batch
        dataset_samples = [sample.inner[dataset_idx] for sample in samples]

        # Use the existing collate_batches function
        batch = collate_batches(dataset_samples)
        inner_batches.append(batch)

    # Stack masks along the batch dimension (TensorDM -> TensorDBM)
    masks = torch.stack([sample.mask for sample in samples], dim=0)

    return ListBatch(inner=inner_batches, mask=masks)
