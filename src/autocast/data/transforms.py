# ruff: noqa: ARG002
"""Deterministic input transforms for The Well datasets.

These are *transforms* (deterministic, applied to every sample of every split),
not stochastic data augmentations. They are injected through The Well's
``WellDataset(transform=...)`` hook -- whose interface happens to be the
``Augmentation`` base class -- and run on the name->tensor ``constant_scalars``
dict before The Well flattens it into the conditioning channel dim.

Because they are deterministic, they must be applied identically to *all*
dataloaders (train/val/test/rollout) so the model sees the same conditioning at
train and inference time -- unlike a stochastic augmentation, which would be
train-only.
"""

from collections.abc import Sequence

from the_well.data import Augmentation
from the_well.data.datasets import TrajectoryData, TrajectoryMetadata


class LogScalars(Augmentation):
    """Natural-log transform of named constant scalars.

    Applies ``value.log()`` to each constant scalar whose name (case-insensitive)
    is in ``scalar_names``. This is a deterministic value transform, not a
    stochastic augmentation: it is meant to be applied to every split. Boundary
    conditions and non-matching scalars are left untouched.
    """

    def __init__(self, scalar_names: Sequence[str]) -> None:
        self.scalar_names = {name.lower() for name in scalar_names}

    def __call__(
        self, data: TrajectoryData, metadata: TrajectoryMetadata
    ) -> TrajectoryData:
        scalars = data.get("constant_scalars")
        if scalars:
            for key, value in scalars.items():
                if key.lower() in self.scalar_names:
                    scalars[key] = value.log()
            data["constant_scalars"] = scalars
        return data
