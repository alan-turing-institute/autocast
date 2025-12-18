from collections.abc import Iterable
from pathlib import Path

import h5py
import torch
from einops import rearrange, repeat
from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from autocast.types.batch import EncodedSample, collate_encoded_samples
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
            t = input_fields.shape[0]
            spatial_dims = input_fields.shape[2:]  # (H, W, ...)
            label_flat = label.flatten()  # Flatten any shape to 1D
            label_expanded = repeat(
                label_flat,
                "c -> t c " + " ".join(f"d{i}" for i in range(len(spatial_dims))),
                t=t,
                **{f"d{i}": s for i, s in enumerate(spatial_dims)},
            )
            input_fields = torch.cat([input_fields, label_expanded], dim=1)

        return self.to_sample(
            {
                "input_fields": rearrange(input_fields, "t c ... -> t ... c"),
                "output_fields": rearrange(output_fields, "t c ... -> t ... c"),
                "label": label,
                "encoded_info": data.get("encoded_info", {}),
            }
        )


class EncodedDataModule(LightningDataModule):
    """DataModule for encoded datasets that produce EncodedBatch objects.

    This datamodule wraps datasets that produce EncodedSample objects (like
    MiniWellInputOutput) and provides train/val/test dataloaders that collate
    samples into EncodedBatch objects.
    """

    def __init__(
        self,
        data_path: str | None = None,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        stride: int = 1,
        batch_size: int = 16,
        num_workers: int = 0,
        concat_inputs_and_label: bool = True,
        dataset_cls: type[EncodedDataset] | None = None,
        **dataset_kwargs,
    ):
        """Initialize the EncodedDataModule.

        Args:
            data_path: Base path to the dataset files. Should contain
                train/, valid/, test/ subdirectories with data.h5 files.
            n_steps_input: Number of input time steps.
            n_steps_output: Number of output time steps to predict.
            stride: Stride for stepping through trajectories.
            batch_size: Batch size for dataloaders.
            num_workers: Number of workers for dataloaders. Default 0 for
                h5py compatibility.
            concat_inputs_and_label: Whether to concatenate labels with inputs.
            dataset_cls: Dataset class to use. Defaults to MiniWellInputOutput.
            **dataset_kwargs: Additional kwargs passed to dataset constructor.
        """
        super().__init__()
        self.data_path = Path(data_path) if data_path is not None else None
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.concat_inputs_and_label = concat_inputs_and_label
        self.dataset_cls = dataset_cls or MiniWellInputOutput
        self.dataset_kwargs = dataset_kwargs

        self.train_dataset: EncodedDataset | None = None
        self.val_dataset: EncodedDataset | None = None
        self.test_dataset: EncodedDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for the given stage."""
        if self.data_path is None:
            msg = "data_path must be provided"
            raise ValueError(msg)

        # Compute total steps needed for the MiniWell dataset
        total_steps = self.n_steps_input + self.n_steps_output

        common_kwargs = {
            "n_steps_input": self.n_steps_input,
            "n_steps_output": self.n_steps_output,
            "steps": total_steps,
            "stride": self.stride,
            "concat_inputs_and_label": self.concat_inputs_and_label,
            **self.dataset_kwargs,
        }

        if stage == "fit" or stage is None:
            train_file = self.data_path / "train" / "data.h5"
            if train_file.exists():
                self.train_dataset = self.dataset_cls(
                    file_name=str(train_file),  # type: ignore TODO: update with protocol to support different classes
                    **common_kwargs,
                )

            valid_file = self.data_path / "valid" / "data.h5"
            if valid_file.exists():
                self.val_dataset = self.dataset_cls(
                    file_name=str(valid_file),  # type: ignore TODO: update with protocol to support different classes
                    **common_kwargs,
                )

        if stage == "test" or stage is None:
            test_file = self.data_path / "test" / "data.h5"
            if test_file.exists():
                self.test_dataset = self.dataset_cls(
                    file_name=str(test_file),  # type: ignore TODO: update with protocol to support different classes
                    **common_kwargs,
                )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_dataset is None:
            self.setup(stage="fit")
        return DataLoader(
            self.train_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_encoded_samples,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.val_dataset is None:
            self.setup(stage="fit")
        return DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_encoded_samples,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        if self.test_dataset is None:
            self.setup(stage="test")
        return DataLoader(
            self.test_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_encoded_samples,
        )


class MiniWellDataModule(LightningDataModule):
    """DataModule for MiniWell datasets.

    This datamodule wraps MiniWellInputOutput datasets and provides
    train/val/test dataloaders that collate samples into EncodedBatch objects.
    Accepts a data_path with train/valid/test subdirectories containing data.h5.
    """

    def __init__(
        self,
        data_path: str | None = None,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        stride: int = 1,
        batch_size: int = 16,
        num_workers: int = 0,
        concat_inputs_and_label: bool = True,
    ):
        """Initialize the MiniWellDataModule.

        Args:
            data_path: Base path to the dataset files. Should contain
                train/, valid/, test/ subdirectories with data.h5 files.
            n_steps_input: Number of input time steps.
            n_steps_output: Number of output time steps to predict.
            stride: Stride for stepping through trajectories.
            batch_size: Batch size for dataloaders.
            num_workers: Number of workers for dataloaders. Default 0 for
                h5py compatibility.
            concat_inputs_and_label: Whether to concatenate labels with inputs.
        """
        super().__init__()
        self.data_path = Path(data_path) if data_path is not None else None
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.concat_inputs_and_label = concat_inputs_and_label

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def _create_dataset(self, dir_path: Path) -> Dataset | None:
        """Create a dataset from all h5 files in a directory."""
        if not dir_path.exists():
            return None

        # Find all h5 files in the directory
        files = sorted(dir_path.glob("*.h5")) + sorted(dir_path.glob("*.hdf5"))
        if not files:
            return None

        # Compute total steps needed for the MiniWell dataset
        total_steps = self.n_steps_input + self.n_steps_output

        common_kwargs = {
            "n_steps_input": self.n_steps_input,
            "n_steps_output": self.n_steps_output,
            "steps": total_steps,
            "stride": self.stride,
            "concat_inputs_and_label": self.concat_inputs_and_label,
        }

        if len(files) == 1:
            return MiniWellInputOutput(file_name=str(files[0]), **common_kwargs)

        # Multiple files - create ConcatDataset
        datasets = [
            MiniWellInputOutput(file_name=str(f), **common_kwargs) for f in files
        ]
        return ConcatDataset(datasets)

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for the given stage."""
        if self.data_path is None:
            msg = "data_path must be provided"
            raise ValueError(msg)

        if stage == "fit" or stage is None:
            if self.train_dataset is None:
                train_dir = self.data_path / "train"
                self.train_dataset = self._create_dataset(train_dir)
            if self.val_dataset is None:
                valid_dir = self.data_path / "valid"
                self.val_dataset = self._create_dataset(valid_dir)

        if (stage == "test" or stage is None) and self.test_dataset is None:
            test_dir = self.data_path / "test"
            self.test_dataset = self._create_dataset(test_dir)

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_dataset is None:
            self.setup(stage="fit")
        if self.train_dataset is None:
            msg = "No training files provided"
            raise ValueError(msg)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_encoded_samples,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.val_dataset is None:
            self.setup(stage="fit")
        if self.val_dataset is None:
            msg = "No validation files provided"
            raise ValueError(msg)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_encoded_samples,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        if self.test_dataset is None:
            self.setup(stage="test")
        if self.test_dataset is None:
            msg = "No test files provided"
            raise ValueError(msg)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_encoded_samples,
        )
