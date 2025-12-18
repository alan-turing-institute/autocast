from collections.abc import Iterable

import h5py
import torch
from einops import repeat
from torch.utils.data import ConcatDataset, Dataset

from autocast.types.batch import EncodedSample
from autocast.types.types import Tensor, TensorNC


class EncodedBatchMixin:
    """A mixin class to provide EncodedBatch conversion functionality."""

    @staticmethod
    def to_sample(data: dict) -> EncodedSample:
        """Convert a dictionary of tensors to a Sample object."""
        return EncodedSample(
            encoded_inputs=data["input_fields"],
            encoded_output_fields=data["output_fields"],
            label=data.get("label"),
            encoded_info=data.get("encoded_info", {}),
        )


class EncodedDataset(Dataset, EncodedBatchMixin):
    """A base class for encoded datasets."""

    def __init__(self):
        super().__init__()


class MiniWellDataset(Dataset):
    r"""Creates a mini-Well dataset.

    From LOLA:
        https://github.com/PolymathicAI/lola/blob/bd4bdf2a9fc024e6b2aa95eb4e24a800fec98dae/lola/data.py#L399
    """

    def __init__(
        self,
        file: str,
        steps: int = 1,
        stride: int = 1,
    ):
        self.file = h5py.File(file, mode="r")

        self.trajectories = self.file["state"].shape[0]  # type: ignore  # noqa: PGH003
        self.steps_per_trajectory = self.file["state"].shape[1]  # type: ignore  # noqa: PGH003

        self.steps = steps
        self.stride = stride

    def __len__(self) -> int:  # noqa: D105
        return self.trajectories * (
            self.steps_per_trajectory - (self.steps - 1) * self.stride
        )

    def __getitem__(self, i: int) -> dict[str, Tensor]:  # noqa: D105
        crops_per_trajectory = (
            self.steps_per_trajectory - (self.steps - 1) * self.stride
        )

        i, j = i // crops_per_trajectory, i % crops_per_trajectory

        state = self.file["state"][  # type: ignore  # noqa: PGH003
            i, slice(j, j + (self.steps - 1) * self.stride + 1, self.stride)
        ]
        label = self.file["label"][i]  # type: ignore  # noqa: PGH003

        return {
            "state": torch.as_tensor(state),
            "label": torch.as_tensor(label),
        }

    @staticmethod
    def from_files(files: Iterable[str], **kwargs) -> Dataset:
        return ConcatDataset([MiniWellDataset(file, **kwargs) for file in files])


class MiniWellInputOutput(EncodedDataset, EncodedBatchMixin):
    """A wrapper around The Well's MiniwellDataset to provide Batch objects."""

    miniwell_dataset: MiniWellDataset

    def __init__(
        self,
        file_name: str,
        n_steps_input: int,
        n_steps_output: int,
        steps: int = 1,
        stride: int = 1,
        concat_inputs_and_label: bool = True,
    ):
        Dataset.__init__(self)
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.concat_inputs_and_label = concat_inputs_and_label
        self.miniwell_dataset = MiniWellDataset(
            file=file_name, steps=steps, stride=stride
        )

    @staticmethod
    def from_files(files: Iterable[str], **kwargs) -> Dataset:
        return ConcatDataset([MiniWellDataset(file, **kwargs) for file in files])

    def __len__(self) -> int:  # noqa: D105
        return len(self.miniwell_dataset)

    def __getitem__(self, index) -> EncodedSample:  # noqa: D105
        data = self.miniwell_dataset.__getitem__(index)

        input_fields = data["state"][: self.n_steps_input]
        output_fields = data["state"][
            self.n_steps_input : self.n_steps_input + self.n_steps_output
        ]
        label: TensorNC = data.get("label")  # type: ignore  # noqa: PGH003
        if self.concat_inputs_and_label:
            # Broadcast label across spatial dims to match input_fields shape
            # input_fields: (T, C, *spatial), label: (*) -> (1, numel, *spatial)
            spatial_dims = input_fields.shape[2:]  # (H, W, ...)
            label_flat = label.flatten()  # Flatten any shape to 1D
            label_expanded = repeat(
                label_flat,
                "c -> 1 c " + " ".join(f"d{i}" for i in range(len(spatial_dims))),
                **{f"d{i}": s for i, s in enumerate(spatial_dims)},
            )
            input_fields = torch.cat([input_fields, label_expanded], dim=0)

        return self.to_sample(
            {
                "input_fields": input_fields,
                "output_fields": output_fields,
                "label": label,
                "encoded_info": data.get("encoded_info", {}),
            }
        )
