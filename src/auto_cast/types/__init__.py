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
class Batch:
    input_fields: Tensor
    output_fields: Tensor
    constant_scalars: Tensor
    constant_fields: Tensor


@dataclass
class EncodedBatch:
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
