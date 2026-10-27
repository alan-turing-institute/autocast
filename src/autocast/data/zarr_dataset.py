"""Lazy windows over AutoSim Zarr arrays, independent of their chunk layout."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import yaml
from lightning.pytorch import LightningDataModule
from the_well.data.normalization import ZScoreNormalization
from torch.utils.data import DataLoader, Dataset

from autocast.data.metadata import Metadata
from autocast.types import Sample
from autocast.types.collation import collate_batches


def _open_group(path: str) -> Any:
    try:
        import zarr  # noqa: PLC0415 -- keep the optional dependency lazy
    except ImportError as error:
        msg = "Zarr datasets require the optional dependency: uv sync --extra zarr."
        raise ImportError(msg) from error
    return zarr.open_group(path, mode="r")


def _positive_integer(name: str, value: int, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        msg = f"{name} must be an integer greater than or equal to {minimum}."
        raise ValueError(msg)


class ZarrDataset(Dataset):
    """Read bounded raw-field windows from a completed AutoSim Zarr store.

    Arrays retain ``[trajectory, time, height, width, channel]`` order. ``stride``
    advances the window start; frames inside a window remain consecutive.
    Storage shards never become trajectory boundaries. Initialization reads
    metadata only, and each loader process opens its own array on first access.
    """

    def __init__(  # noqa: PLR0915
        self,
        data_path: str | Path,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        stride: int = 1,
        start_frame: int = 0,
        channel_idxs: tuple[int, ...] | list[int] | None = None,
        autoencoder_mode: bool = False,
        full_trajectory_mode: bool = False,
        dtype: torch.dtype = torch.float32,
        use_normalization: bool = False,
        normalization_path: str | Path | None = None,
        normalization_stats: dict | None = None,
    ) -> None:
        """Configure windowing and optional training-derived normalization."""
        if full_trajectory_mode:
            msg = (
                "Full-trajectory mode is unsupported for lazy Zarr datasets. "
                "Use bounded rollout_steps on ZarrDataModule; latent caching "
                "requires a streaming encoder."
            )
            raise ValueError(msg)
        _positive_integer("n_steps_input", n_steps_input)
        _positive_integer("n_steps_output", n_steps_output, allow_zero=autoencoder_mode)
        _positive_integer("stride", stride)
        _positive_integer("start_frame", start_frame, allow_zero=True)
        self.data_path = str(data_path)
        group = _open_group(self.data_path)
        attrs = dict(group.attrs)
        if (
            attrs.get("schema") != "autosim.spatiotemporal"
            or attrs.get("schema_version") != 1
        ):
            msg = "Expected an autosim.spatiotemporal Zarr store with schema_version=1."
            raise ValueError(msg)
        if attrs.get("complete") is not True:
            msg = "The Zarr store is incomplete; wait for generation to finish."
            raise ValueError(msg)
        array = group["data"]
        if (
            not hasattr(array, "shape")
            or len(array.shape) != 5
            or any(size < 1 for size in array.shape)
        ):
            msg = (
                "Expected a non-empty [trajectory, time, height, width, channel] array"
            )
            raise ValueError(msg)
        self.n_trajectories, total_steps, self.height, self.width, n_channels = (
            array.shape
        )
        field_names = attrs.get("field_names", [])
        if (
            not isinstance(field_names, list)
            or len(field_names) != n_channels
            or any(not isinstance(name, str) or not name for name in field_names)
            or len(set(field_names)) != n_channels
        ):
            msg = "field_names must contain one unique name per stored channel."
            raise ValueError(msg)
        self.channel_idxs = tuple(channel_idxs) if channel_idxs is not None else None
        if self.channel_idxs is not None:
            if not self.channel_idxs or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < n_channels
                for index in self.channel_idxs
            ):
                msg = "channel_idxs must contain valid, non-negative channel indices."
                raise ValueError(msg)
            field_names = [field_names[index] for index in self.channel_idxs]
        self.field_names = list(field_names)
        self.n_channels = len(self.field_names)
        self.n_timesteps = total_steps - start_frame
        self.n_steps_input = 1 if autoencoder_mode else n_steps_input
        self.n_steps_output = 0 if autoencoder_mode else n_steps_output
        self.stride = stride
        self.start_frame = start_frame
        self.autoencoder_mode = autoencoder_mode
        self.full_trajectory_mode = False
        self.dtype = dtype
        self.use_normalization = use_normalization
        self.provenance = attrs.get("provenance", {})
        self._window_size = self.n_steps_input + self.n_steps_output
        if self.n_timesteps < self._window_size:
            msg = "start_frame leaves too few frames for the requested window."
            raise ValueError(msg)
        self._windows_per_trajectory = (
            self.n_timesteps - self._window_size
        ) // stride + 1
        self._array: Any = None
        self._pid: int | None = None
        self.norm: ZScoreNormalization | None = None
        if use_normalization:
            if normalization_stats is None:
                stats_path = (
                    Path(normalization_path)
                    if normalization_path is not None
                    else Path(self.data_path).parent.parent / "stats.yml"
                )
                with stats_path.open() as stream:
                    normalization_stats = yaml.safe_load(stream)
            if normalization_stats is None or "stats" not in normalization_stats:
                msg = "Normalization metadata must contain a stats mapping."
                raise ValueError(msg)
            self.norm = ZScoreNormalization(
                normalization_stats["stats"], self.field_names, []
            )
        self.metadata = Metadata(
            dataset_name=Path(self.data_path).parent.parent.name,
            n_spatial_dims=2,
            spatial_resolution=(self.height, self.width),
            scalar_names=[],
            constant_scalar_names=[],
            constant_field_names={},
            boundary_condition_types=[],
            field_names={0: self.field_names},
            n_steps_per_trajectory=[self.n_timesteps] * self.n_trajectories,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Exclude process-local array handles from worker serialization."""
        state = self.__dict__.copy()
        state["_array"] = None
        state["_pid"] = None
        return state

    def _data_array(self):
        if self._array is None or self._pid != os.getpid():
            self._array = _open_group(self.data_path)["data"]
            self._pid = os.getpid()
        return self._array

    def __len__(self) -> int:
        """Return the number of windows, including those crossing storage shards."""
        return self.n_trajectories * self._windows_per_trajectory

    @property
    def sample_trajectory_indices(self) -> list[int]:
        """Construct trajectory identifiers only when evaluation requests them."""
        return [index // self._windows_per_trajectory for index in range(len(self))]

    @property
    def sample_window_indices(self) -> list[int]:
        """Construct per-trajectory window ordinals for evaluation."""
        return [index % self._windows_per_trajectory for index in range(len(self))]

    def __getitem__(self, index: int) -> Sample:
        """Read and normalize one contiguous window without loading a trajectory."""
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        trajectory, window = divmod(index, self._windows_per_trajectory)
        start = self.start_frame + window * self.stride
        fields = torch.from_numpy(
            self._data_array()[trajectory, start : start + self._window_size]
        )
        if self.channel_idxs is not None:
            fields = fields[..., list(self.channel_idxs)]
        if self.norm is not None:
            fields = self.norm.normalize_flattened(fields, "variable")
        fields = fields.to(self.dtype)
        inputs = fields[: self.n_steps_input]
        outputs = inputs if self.autoencoder_mode else fields[self.n_steps_input :]
        return Sample(inputs, outputs, None, None, None)


class ZarrDataModule(LightningDataModule):
    """Create lazy training and bounded-rollout loaders for raw Zarr fields."""

    def __init__(
        self,
        data_path: str | Path,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        stride: int = 1,
        batch_size: int = 16,
        num_workers: int = 0,
        start_frame: int = 0,
        channel_idxs: tuple[int, ...] | list[int] | None = None,
        dtype: torch.dtype = torch.float32,
        autoencoder_mode: bool = False,
        full_trajectory_mode: bool = False,
        use_normalization: bool = True,
        normalization_path: str | Path | None = None,
        normalization_stats: dict | None = None,
        rollout_steps: int = 12,
        rollout_window_stride: int | None = None,
        pin_memory: bool = True,
        persistent_workers: bool | None = None,
        prefetch_factor: int | None = 2,
    ) -> None:
        """Configure loaders; ``rollout_steps`` caps each evaluation target window."""
        super().__init__()
        if full_trajectory_mode:
            msg = (
                "ZarrDataModule does not support full_trajectory_mode. "
                "Use bounded rollout_steps; latent caching needs a streaming encoder."
            )
            raise ValueError(msg)
        _positive_integer("batch_size", batch_size)
        _positive_integer("num_workers", num_workers, allow_zero=True)
        _positive_integer("rollout_steps", rollout_steps)
        _positive_integer("n_steps_output", n_steps_output, allow_zero=autoencoder_mode)
        if not autoencoder_mode and rollout_steps < n_steps_output:
            msg = "rollout_steps must be at least n_steps_output."
            raise ValueError(msg)
        if rollout_window_stride is not None:
            _positive_integer("rollout_window_stride", rollout_window_stride)
        if prefetch_factor is not None:
            _positive_integer("prefetch_factor", prefetch_factor)
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.autoencoder_mode = autoencoder_mode
        self.rollout_steps = rollout_steps
        self.rollout_window_stride = (
            stride if rollout_window_stride is None else rollout_window_stride
        )
        self.pin_memory = pin_memory
        self.persistent_workers = (
            num_workers > 0 if persistent_workers is None else persistent_workers
        )
        self.prefetch_factor = prefetch_factor
        self._dataset_options: dict[str, Any] = {
            "n_steps_input": n_steps_input,
            "n_steps_output": n_steps_output,
            "stride": stride,
            "start_frame": start_frame,
            "channel_idxs": channel_idxs,
            "dtype": dtype,
            "autoencoder_mode": autoencoder_mode,
            "use_normalization": use_normalization,
            "normalization_path": normalization_path or self.data_path / "stats.yml",
            "normalization_stats": normalization_stats,
        }
        self.train_dataset: ZarrDataset | None = None
        self.val_dataset: ZarrDataset | None = None
        self.test_dataset: ZarrDataset | None = None
        self._rollout_datasets: dict[str, ZarrDataset] = {}

    def _dataset(self, split: str, *, rollout: bool = False) -> ZarrDataset:
        options = dict(self._dataset_options)
        path = self.data_path / split / "data.zarr"
        dataset = ZarrDataset(path, **options)
        if rollout:
            # The ordinary window validates that at least one prediction block
            # fits; shorter trajectories can still use the available horizon.
            options.update(
                n_steps_output=min(
                    self.rollout_steps, dataset.n_timesteps - dataset.n_steps_input
                ),
                stride=self.rollout_window_stride,
            )
            return ZarrDataset(path, **options)
        return dataset

    def setup(self, stage: str | None = None) -> None:
        """Read split metadata once; no data arrays are materialized here."""
        if stage in (None, "fit") and self.train_dataset is None:
            self.train_dataset = self._dataset("train")
        if stage in (None, "fit", "validate") and self.val_dataset is None:
            self.val_dataset = self._dataset("valid")
        if stage in (None, "test", "predict") and self.test_dataset is None:
            self.test_dataset = self._dataset("test")

    def _loader(self, dataset: ZarrDataset, *, shuffle: bool, batch_size=None):
        options: dict[str, Any] = {
            "batch_size": self.batch_size if batch_size is None else batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "collate_fn": collate_batches,
        }
        if self.num_workers > 0:
            # Zarr metadata access can start asynchronous I/O threads in the
            # parent; spawning avoids inheriting their runtime through fork.
            options["multiprocessing_context"] = "spawn"
            options["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                options["prefetch_factor"] = self.prefetch_factor
        return DataLoader(dataset, **options)

    def train_dataloader(self) -> DataLoader:
        """Return shuffled training windows for the standard distributed sampler."""
        self.setup("fit")
        assert self.train_dataset is not None
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return validation windows using training-derived statistics."""
        self.setup("validate")
        assert self.val_dataset is not None
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return held-out test windows."""
        self.setup("test")
        assert self.test_dataset is not None
        return self._loader(self.test_dataset, shuffle=False)

    def _rollout_loader(self, split: str, batch_size: int | None) -> DataLoader:
        if self.autoencoder_mode:
            msg = "Rollout loaders are unavailable in autoencoder mode."
            raise ValueError(msg)
        if split not in self._rollout_datasets:
            self._rollout_datasets[split] = self._dataset(split, rollout=True)
        return self._loader(
            self._rollout_datasets[split], shuffle=False, batch_size=batch_size
        )

    def rollout_val_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Return bounded training-split rollouts, matching the existing API."""
        return self._rollout_loader("train", batch_size)

    def rollout_valid_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Return bounded held-out validation rollouts."""
        return self._rollout_loader("valid", batch_size)

    def rollout_test_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Return bounded test rollouts without reading an entire trajectory."""
        return self._rollout_loader("test", batch_size)
