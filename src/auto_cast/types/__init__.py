from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch.utils.data import DataLoader

Tensor = torch.Tensor

TensorBC = Float[Tensor, "batch channel"]
TensorBTWHC = Float[Tensor, "batch time width height channel"]
TensorBTWHLC = Float[Tensor, "batch time width height length channel"]
TensorBTSC = Float[Tensor, "batch time *spatial channel"]
TensorBCTWH = Float[Tensor, "batch channel time width height"]
TensorBCTWHL = Float[Tensor, "batch channel time width height length"]
TensorBCTHW = Float[Tensor, "batch channel time height width"]
TensorBCTS = Float[Tensor, "batch channel time *spatial"]
TensorBWHC = Float[Tensor, "batch width height channel"]
TensorBWHLC = Float[Tensor, "batch width height length channel"]
TensorBSC = Float[Tensor, "batch *spatial channel"]
TensorBTCHW = Float[Tensor, "batch time channel height width"]
TensorBTC = Float[Tensor, "batch time channel"]

Input = Tensor | DataLoader
RolloutOutput = tuple[Tensor, None] | tuple[Tensor, Tensor]

# Batch = dict[str, Tensor]
# EncodedBatch = dict[str, Tensor]


# TODO: Could be a dataclass if we want more structure
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
    encoded_info: dict[str, Tensor]


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
