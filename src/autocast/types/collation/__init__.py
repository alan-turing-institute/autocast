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

    # class ListSample:
    #     inner: list[Sample]
    #     mask: TensorDM  # Dataset by ensemble mask (e.g. for different combinations of
    #         missing data across datasets)

    # TODO: For each batch of ListSamples (where a ListSample contains a Sample per dataset)
    # we'd like to return a ListBatch containing a Batch per dataset, where each Batch
    # is constructed by stacking the samples across the outer sequence that's passed
    # to collate_list_batches.
    # The masks per sample should be collated across the batch dimension as well,
    # resulting in a TensorDBM mask in the ListBatch.

    for sample in samples:
        if not isinstance(sample, ListSample):
            msg = f"Expected ListSample, got {type(sample)}"
            raise TypeError(msg)

        inner_batches: list[Batch] = []
        for inner_sample in sample.inner:
            if not isinstance(inner_sample, Sample):
                msg = f"Expected inner samples to be of type Sample, got {type(inner_sample)}"
                raise TypeError(msg)
            input_fields = torch.stack(
                [inner_sample.input_fields for inner_sample in sample.inner], dim=0
            )
            output_fields = torch.stack(
                [inner_sample.output_fields for inner_sample in sample.inner], dim=0
            )
            constant_scalars = _stack_optional("constant_scalars")
            constant_fields = _stack_optional("constant_fields")
            boundary_conditions = _stack_optional("boundary_conditions")

            inner_batch = Batch(
                input_fields=input_fields,
                output_fields=output_fields,
                constant_scalars=constant_scalars,
                constant_fields=constant_fields,
                boundary_conditions=boundary_conditions,
            )
            # Inner batch per dataset
            inner_batches.append(inner_batch)
