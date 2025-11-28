from __future__ import annotations

from collections.abc import Sequence

import torch

from auto_cast.preprocessor.base import Preprocessor
from auto_cast.types import Batch, Tensor


class FNOInputPreprocessor(Preprocessor):
    """Prepare FNO inputs by selecting channels and appending constants."""

    def __init__(
        self,
        channels: Sequence[int] = (0,),
        with_constants: bool = False,
    ) -> None:
        self.channels = tuple(channels)
        self.with_constants = with_constants

    def __call__(self, batch: Batch) -> Tensor:
        if "input_fields" not in batch:
            msg = "Batch is missing 'input_fields' required for FNO input preparation"
            raise KeyError(msg)

        x = batch["input_fields"][:, list(self.channels), ...]

        if self.with_constants and "constant_fields" in batch:
            constants = batch["constant_fields"]
            while constants.dim() < x.dim():
                constants = constants.unsqueeze(-1)
            constants = constants.expand(x.shape[0], -1, *x.shape[2:])
            x = torch.cat([x, constants], dim=1)

        return x
