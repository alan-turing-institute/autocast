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
    constant_scalars: Tensor  # (B, C)
    constant_fields: Tensor  # (B, W, H, C)


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
