"""Field ordering and reversible preprocessing for LoLA codecs."""

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor

from autocast.decoders.identity import IdentityDecoder
from autocast.encoders.identity import IdentityEncoder
from autocast.types.batch import Batch


def named_field_indices(
    well_field_names: Iterable[Sequence[str]], codec_fields: Sequence[str]
) -> list[int]:
    """Map the codec's named channels to The Well's flattened field order."""
    fields = [field for group in well_field_names for field in group]
    if len(fields) != len(codec_fields) or set(fields) != set(codec_fields):
        raise ValueError(
            f"Well fields {fields} do not match codec fields {codec_fields}"
        )
    return [fields.index(field) for field in codec_fields]


def standardize_fields(
    x: Tensor, mean: Tensor, std: Tensor, log1p_indices: Sequence[int] = ()
) -> Tensor:
    """Apply LoLA's field transforms before checkpoint training statistics."""
    if log1p_indices:
        indices = list(log1p_indices)
        x = x.clone()
        x[..., indices] = torch.log1p(x[..., indices])
    return (x - mean) / std


def restore_fields(
    x: Tensor, mean: Tensor, std: Tensor, log1p_indices: Sequence[int] = ()
) -> Tensor:
    """Invert LoLA's normalization and return fields in physical units."""
    x = x * std + mean
    if log1p_indices:
        indices = list(log1p_indices)
        x[..., indices] = torch.expm1(x[..., indices])
    return x


class StandardizedFieldEncoder(IdentityEncoder):
    """Apply the codec's field transforms while retaining the native grid."""

    mean: Tensor
    std: Tensor

    def __init__(
        self,
        in_channels: int,
        mean: Sequence[float],
        std: Sequence[float],
        log1p_indices: Sequence[int],
        field_indices: Sequence[int] = (),
    ) -> None:
        super().__init__(in_channels)
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.log1p_indices = list(log1p_indices)
        self.field_indices = list(field_indices)

    def encode(self, batch: Batch) -> Tensor:
        fields = batch.input_fields
        if self.field_indices:
            fields = fields[..., self.field_indices]
        return standardize_fields(fields, self.mean, self.std, self.log1p_indices)


class StandardizedFieldDecoder(IdentityDecoder):
    """Restore physical fields on the native grid."""

    mean: Tensor
    std: Tensor

    def __init__(
        self,
        in_channels: int,
        mean: Sequence[float],
        std: Sequence[float],
        log1p_indices: Sequence[int],
    ) -> None:
        super().__init__(in_channels)
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.log1p_indices = list(log1p_indices)

    def decode(self, z: Tensor) -> Tensor:
        return restore_fields(z, self.mean, self.std, self.log1p_indices)
