from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import torch
from jaxtyping import Float
from torch.utils.data import DataLoader

# Alias for torch.Tensor
Tensor = torch.Tensor

# Type hints for various tensor shapes:
# - B: batch dimension
# - T: time dimension
# - S: spatial dimension (could be multiple spatial dims)
# - C: channel dimension
# - L: latent dimension
# - M: indicates multiple combined dimensions (e.g. spatiotemporal)
# - Star (*): indicates zero or more dimensions
# - Plus (+): indicates one or more dimensions

# Vector of scalars
TensorC = Float[Tensor, "channel"]

# Only batch and channel
TensorBC = Float[Tensor, "batch channel"]

# Combined optional spatiotemporal
TensorBMStarC = Float[Tensor, "batch *spatiotemporal channel"]
TensorBMStarL = Float[Tensor, "batch *spatiotemporal latent"]

# Separated spatiotemporal
TensorBTSStarC = Float[Tensor, "batch time *spatial channel"]
TensorBTSPlusC = Float[Tensor, "batch time spatial *spatial channel"]
TensorBTWHC = Float[Tensor, "batch time width height channel"]
TensorBTWHDC = Float[Tensor, "batch time width height depth channel"]
TensorBTCHW = Float[Tensor, "batch time channel height width"]
TensorBCTWH = Float[Tensor, "batch channel time width height"]
TensorBCTWHD = Float[Tensor, "batch channel time width height depth"]
TensorBCTHW = Float[Tensor, "batch channel time height width"]
TensorBCTSPlus = Float[Tensor, "batch channel time spatial *spatial"]

# Spatial only (no time dimension)
TensorBCSPlus = Float[Tensor, "batch channel spatial *spatial"]
TensorBWHC = Float[Tensor, "batch width height channel"]
TensorBCWH = Float[Tensor, "batch channel width height"]
TensorBWHDC = Float[Tensor, "batch width height depth channel"]
TensorBSPlusC = Float[Tensor, "batch spatial *spatial channel"]

# Items without batch dimension
TensorTSPlusC = Float[Tensor, "time spatial *spatial channel"]
TensorSPlusC = Float[Tensor, "spatial *spatial channel"]

# Generic input type
Input = Tensor | DataLoader

# Rollout output type
RolloutOutput = tuple[Tensor, None] | tuple[Tensor, Tensor]

# Generic batch type variable
BatchT = TypeVar("BatchT")


@dataclass
class Sample:
    """A batch in input data space."""

    input_fields: TensorTSPlusC
    output_fields: TensorTSPlusC
    constant_scalars: TensorC | None
    constant_fields: TensorSPlusC | None


@dataclass
class Batch:
    """A batch in input data space."""

    input_fields: TensorBTSPlusC
    output_fields: TensorBTSPlusC
    constant_scalars: TensorBC | None
    constant_fields: TensorBSPlusC | None


@dataclass
class EncodedBatch:
    """A batch after being processed by an Encoder."""

    encoded_inputs: TensorBTSPlusC
    encoded_output_fields: TensorBTSPlusC
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
