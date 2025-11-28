from pathlib import Path

import torch
from the_well.data.datamodule import WellDataModule
from the_well.data.normalization import ZScoreNormalization
from torch.utils.data import DataLoader

from auto_cast.data.dataset import SpatioTemporalDataset


class SpatioTemporalDataModule(WellDataModule):
    """A class for spatio-temporal data modules."""

    def __init__(
        self,
        data_path: str | None,
        data: dict[str, dict] | None = None,
        dataset_cls: type[SpatioTemporalDataset] = SpatioTemporalDataset,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        stride: int = 1,
        # TODO: support for passing data from dict
        input_channel_idxs: tuple[int, ...] | None = None,
        output_channel_idxs: tuple[int, ...] | None = None,
        batch_size: int = 4,
        dtype: torch.dtype = torch.float32,
        ftype: str = "torch",
        verbose: bool = False,
        use_normalization: bool = False,
    ):
        self.verbose = verbose
        self.use_normalization = use_normalization

        base_path = Path(data_path) if data_path is not None else None
        suffix = ".pt" if ftype == "torch" else ".h5"
        fname = f"data{suffix}"
        train_path = base_path / "train" / fname if base_path is not None else None
        valid_path = base_path / "valid" / fname if base_path is not None else None
        test_path = base_path / "test" / fname if base_path is not None else None

        # Create training dataset first (without normalization)
        self.train_dataset = dataset_cls(
            data_path=str(train_path) if train_path is not None else None,
            data=data["train"] if data is not None else None,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            stride=stride,
            input_channel_idxs=input_channel_idxs,
            output_channel_idxs=output_channel_idxs,
            dtype=dtype,
            verbose=self.verbose,
            use_normalization=False,  # Temporarily disable to compute stats
            norm=None,
        )

        # Compute normalization from training data if requested
        norm = None
        if self.use_normalization:
            if self.verbose:
                print("Computing normalization statistics from training data...")
            norm = ZScoreNormalization
            # if self.verbose:
            #     print(f"  Mean (per channel): {norm.mean}")
            #     print(f"  Std (per channel): {norm.std}")

            # Now enable normalization for training dataset
            self.train_dataset.use_normalization = True
            self.train_dataset.norm = norm

        self.val_dataset = dataset_cls(
            data_path=str(valid_path) if valid_path is not None else None,
            data=data["valid"] if data is not None else None,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            stride=stride,
            input_channel_idxs=input_channel_idxs,
            output_channel_idxs=output_channel_idxs,
            dtype=dtype,
            verbose=self.verbose,
            use_normalization=self.use_normalization,
            norm=norm,
        )
        self.test_dataset = dataset_cls(
            data_path=str(test_path) if test_path is not None else None,
            data=data["test"] if data is not None else None,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            stride=stride,
            input_channel_idxs=input_channel_idxs,
            output_channel_idxs=output_channel_idxs,
            dtype=dtype,
            verbose=self.verbose,
            use_normalization=self.use_normalization,
            norm=norm,
        )
        self.rollout_val_dataset = dataset_cls(
            data_path=str(train_path) if train_path is not None else None,
            data=data["train"] if data is not None else None,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            stride=stride,
            input_channel_idxs=input_channel_idxs,
            output_channel_idxs=output_channel_idxs,
            full_trajectory_mode=True,
            dtype=dtype,
            verbose=self.verbose,
            use_normalization=self.use_normalization,
            norm=norm,
        )
        self.rollout_test_dataset = dataset_cls(
            data_path=str(test_path) if test_path is not None else None,
            data=data["test"] if data is not None else None,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            stride=stride,
            input_channel_idxs=input_channel_idxs,
            output_channel_idxs=output_channel_idxs,
            full_trajectory_mode=True,
            dtype=dtype,
            verbose=self.verbose,
            use_normalization=self.use_normalization,
            norm=norm,
        )
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        """DataLoader for training."""
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1
        )

    def val_dataloader(self) -> DataLoader:
        """DataLoader for standard validation (not full trajectory rollouts)."""
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1
        )

    def rollout_val_dataloader(self) -> DataLoader:
        """DataLoader for full trajectory rollouts on validation data."""
        return DataLoader(
            self.rollout_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )

    def test_dataloader(self) -> DataLoader:
        """DataLoader for testing."""
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1
        )

    def rollout_test_dataloader(self) -> DataLoader:
        """DataLoader for full trajectory rollouts on test data."""
        return DataLoader(
            self.rollout_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )
