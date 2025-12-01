from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

Tensor = torch.Tensor
Input = Tensor | DataLoader
RolloutOutput = tuple[Tensor, None] | tuple[Tensor, Tensor]

# Batch = dict[str, Tensor]
# EncodedBatch = dict[str, Tensor]


# TODO: Could be a dataclass if we want more structure
@dataclass
class Batch:  # noqa: D101
    input_fields: Tensor  # (B, T, W, H, C)
    output_fields: Tensor  # (B, T, W, H, C)
    constant_scalars: Tensor | None  # (B, C)
    constant_fields: Tensor | None  # (B, W, H, C)


@dataclass
class EncodedBatch:  # noqa: D101
    encoded_inputs: Tensor
    encoded_output_fields: Tensor
    encoded_info: dict[str, Tensor]


class EncoderForBatch:
    """EncoderForBatch."""

    def __call__(self, batch: Batch) -> EncodedBatch:
        return EncodedBatch(
            encoded_inputs=batch.input_fields,
            encoded_output_fields=batch.output_fields,
            encoded_info={},
        )


def collate_batches(samples: Sequence[Batch]) -> Batch:
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
