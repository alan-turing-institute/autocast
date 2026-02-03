import os
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from the_well.data.normalization import ZScoreNormalization
from torch.utils.data import DataLoader

from autocast.data.dataset import SpatioTemporalDataset, TheWell
from autocast.types import collate_batches


class TheWellDataModule(LightningDataModule):
    """DataModule for TheWell datasets."""

    def __init__(
        self,
        well_dataset_name: str,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        batch_size: int = 4,
        use_normalization: bool = False,
        # TODO: consider how to pass normalization stats/path
        normalization_type: type[ZScoreNormalization] | None = ZScoreNormalization,
        autoencoder_mode: bool = False,
        num_workers: int | None = None,  # Auto-detect if None
        normalization_path: str = "../stats.yaml",
        **well_kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.autoencoder_mode = autoencoder_mode
        # Auto-detect num_workers based on available CPUs, capped at 8
        self.num_workers = (
            num_workers if num_workers is not None else min(os.cpu_count() or 1, 8)
        )

        self.train_dataset = TheWell(
            well_dataset_name=well_dataset_name,
            well_split_name="train",
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=use_normalization,
            normalization_type=normalization_type,
            normalization_path=normalization_path,
            autoencoder_mode=autoencoder_mode,
            **well_kwargs,
        )
        self.val_dataset = TheWell(
            well_dataset_name=well_dataset_name,
            well_split_name="valid",
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=use_normalization,
            normalization_type=normalization_type,
            normalization_path=normalization_path,
            autoencoder_mode=autoencoder_mode,
            **well_kwargs,
        )
        self.test_dataset = TheWell(
            well_dataset_name=well_dataset_name,
            well_split_name="test",
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=use_normalization,
            normalization_type=normalization_type,
            normalization_path=normalization_path,
            autoencoder_mode=autoencoder_mode,
            **well_kwargs,
        )

        if not autoencoder_mode:
            self.rollout_val_dataset = TheWell(
                well_dataset_name=well_dataset_name,
                well_split_name="train",
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                normalization_path=normalization_path,
                full_trajectory_mode=True,
                **well_kwargs,
            )
            self.rollout_test_dataset = TheWell(
                well_dataset_name=well_dataset_name,
                well_split_name="test",
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                normalization_path=normalization_path,
                full_trajectory_mode=True,
                **well_kwargs,
            )

    def train_dataloader(self) -> DataLoader:
        """DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_batches,
        )

    def val_dataloader(self) -> DataLoader:
        """DataLoader for validation."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batches,
        )

    def test_dataloader(self) -> DataLoader:
        """DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batches,
        )

    def rollout_val_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """DataLoader for full trajectory rollouts on validation data."""
        if self.autoencoder_mode:
            msg = (
                "Rollout dataloaders not available when autoencoder_mode="
                f"{self.autoencoder_mode}"
            )
            raise RuntimeError(msg)
        return DataLoader(
            self.rollout_val_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batches,
        )

    def rollout_test_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """DataLoader for full trajectory rollouts on test data."""
        if self.autoencoder_mode:
            msg = (
                "Rollout dataloaders not available when autoencoder_mode="
                f"{self.autoencoder_mode}"
            )
            raise RuntimeError(msg)
        return DataLoader(
            self.rollout_test_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batches,
        )


class SpatioTemporalDataModule(LightningDataModule):
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
        autoencoder_mode: bool = False,
        use_normalization: bool = False,
        normalization_type: type[ZScoreNormalization] | None = None,
        normalization_path: None | str = None,
        normalization_stats: dict | None = None,
        num_workers: int | None = None,
    ):
        super().__init__()
        self.verbose = verbose
        self.use_normalization = use_normalization
        self.autoencoder_mode = autoencoder_mode
        # Auto-detect num_workers based on available CPUs, capped at 8
        self.num_workers = (
            num_workers if num_workers is not None else min(os.cpu_count() or 1, 8)
        )
        base_path = Path(data_path) if data_path is not None else None
        suffix = ".pt" if ftype == "torch" else ".h5"
        fname = f"data{suffix}"
        train_path = base_path / "train" / fname if base_path else None
        valid_path = base_path / "valid" / fname if base_path else None
        test_path = base_path / "test" / fname if base_path else None

        # Create training dataset first (without normalization)
        self.train_dataset = dataset_cls(
            data_path=str(train_path) if train_path is not None else None,
            data=data["train"] if data is not None else None,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            stride=stride,
            input_channel_idxs=input_channel_idxs,
            output_channel_idxs=output_channel_idxs,
            autoencoder_mode=self.autoencoder_mode,
            dtype=dtype,
            verbose=self.verbose,
            use_normalization=use_normalization,
            normalization_type=normalization_type,
            normalization_path=normalization_path,
            normalization_stats=normalization_stats,
        )

        # # Compute normalization from training data if requested
        # norm = None
        # if self.use_normalization:
        #     if self.verbose:
        #         print("Computing normalization statistics from training data...")
        #     norm = ZScoreNormalization
        #     # if self.verbose:
        #     #     print(f"  Mean (per channel): {norm.mean}")
        #     #     print(f"  Std (per channel): {norm.std}")

        #     # Now enable normalization for training dataset
        #     self.train_dataset.use_normalization = True
        #     self.train_dataset.norm = norm

        self.val_dataset = dataset_cls(
            data_path=str(valid_path) if valid_path is not None else None,
            data=data["valid"] if data is not None else None,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            stride=stride,
            input_channel_idxs=input_channel_idxs,
            output_channel_idxs=output_channel_idxs,
            autoencoder_mode=self.autoencoder_mode,
            dtype=dtype,
            verbose=self.verbose,
            use_normalization=use_normalization,
            normalization_type=normalization_type,
            normalization_path=normalization_path,
            normalization_stats=normalization_stats,
        )
        self.test_dataset = dataset_cls(
            data_path=str(test_path) if test_path is not None else None,
            data=data["test"] if data is not None else None,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            stride=stride,
            input_channel_idxs=input_channel_idxs,
            output_channel_idxs=output_channel_idxs,
            autoencoder_mode=self.autoencoder_mode,
            dtype=dtype,
            verbose=self.verbose,
            use_normalization=use_normalization,
            normalization_type=normalization_type,
            normalization_path=normalization_path,
            normalization_stats=normalization_stats,
        )

        self.batch_size = batch_size

        if not self.autoencoder_mode:
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
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                normalization_path=normalization_path,
                normalization_stats=normalization_stats,
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
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                normalization_path=normalization_path,
                normalization_stats=normalization_stats,
            )

    def train_dataloader(self) -> DataLoader:
        """DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_batches,
        )

    def val_dataloader(self) -> DataLoader:
        """DataLoader for standard validation (not full trajectory rollouts)."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batches,
        )

    def rollout_val_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """DataLoader for full trajectory rollouts on validation data."""
        if self.autoencoder_mode:
            msg = (
                "Rollout dataloaders not available when autoencoder_mode="
                f"{self.autoencoder_mode}"
            )
            raise RuntimeError(msg)
        return DataLoader(
            self.rollout_val_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batches,
        )

    def test_dataloader(self) -> DataLoader:
        """DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batches,
        )

    def rollout_test_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """DataLoader for full trajectory rollouts on test data."""
        if self.autoencoder_mode:
            msg = (
                "Rollout dataloaders not available when autoencoder_mode="
                f"{self.autoencoder_mode}"
            )
            raise RuntimeError(msg)
        return DataLoader(
            self.rollout_test_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batches,
        )
