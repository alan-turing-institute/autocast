import itertools
from dataclasses import dataclass
from typing import Literal

import torch
from omegaconf import DictConfig
from the_well.data.normalization import ZScoreNormalization
from torch.utils.data import Dataset

from autocast.data.dataset import BatchMixin, SpatioTemporalDataset
from autocast.types.batch import Batch, Sample
from autocast.types.types import TensorDBM, TensorDM


@dataclass
class ListSample:  # noqa: D101
    inner: list[Sample]
    mask: (
        TensorDM | None
    )  # Dataset by ensemble mask (e.g. for different combinations of missing data across datasets)

    # def __getitem__(self, idx) -> Sample:
    #     return self.inner[idx]


@dataclass
class ListBatch:  # noqa: D101
    inner: list[Batch]
    mask: TensorDBM | None


class MultiSpatioTemporalDataset(Dataset, BatchMixin):  # noqa: D101
    datasets: list[SpatioTemporalDataset]  # data: list[(N, T, S, C)]
    # masks: list[Tensor] # shape list[(N, ..., num_masks (treat like ensemble))] # num
    # masks: list[TensorBM]  # shape list[(N, M (num_masks, treat like ensemble))] # num
    # masks: TensorDBM
    masks: TensorDM | None
    # Example for reaction diffusion:
    # - Deterministic loop over all combinations of missing/not missing
    #   (apart from all missing)
    # - Could also have a hierarchy of data availability (e.g. 100, 110, 111)
    #   (if a fidelity level is available, all coarser/lower levels are available too)

    @staticmethod
    def create_mask(
        n_levels: int,
        mode: Literal["sequential", "combinatorial"] = "sequential",
    ) -> TensorDM:
        """Create a dataset-by-mask boolean tensor of shape (D, M).

        Convention: True means masked/unavailable, False means available.

                Modes:
                - sequential: if level l is available, then all lower levels are available.
                    (hierarchical fidelity availability)
                - combinatorial: all masking combinations except the all-masked case.
        """
        if n_levels <= 0:
            msg = "n_levels must be > 0."
            raise ValueError(msg)

        if mode == "sequential":
            # Same convention as attention masks: True entries are masked.
            # Example D=3 -> [[0, 1, 1],
            #                 [0, 0, 1],
            #                 [0, 0, 0]]
            return torch.triu(torch.ones((n_levels, n_levels), dtype=torch.bool), 1)

        if mode == "combinatorial":
            # Keep every masking pattern except "all masked".
            masks = list(itertools.product([False, True], repeat=n_levels))
            all_masked = tuple(True for _ in range(n_levels))
            masks.remove(all_masked)
            return torch.tensor(masks, dtype=torch.bool).T

        raise ValueError(f"Unknown mask mode: {mode}")

    @staticmethod
    def _infer_n_levels(
        data_paths: str | list[str] | tuple[str, ...] | None,
        data: list[dict] | None,
    ) -> int:
        if data is not None:
            return len(data)
        if data_paths is None:
            msg = "Cannot infer number of datasets for mask creation."
            raise ValueError(msg)
        if isinstance(data_paths, str):
            return len(data_paths.split(","))
        return len(data_paths)

    def __init__(
        self,
        data_paths: str | list[str] | tuple[str, ...] | None = None,
        data: list[dict] | None = None,
        masks: (
            TensorDM | Literal["sequential"] | Literal["combinatorial"] | None
        ) = "sequential",
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        stride: int = 1,
        input_channel_idxs: tuple[int, ...] | None = None,
        output_channel_idxs: tuple[int, ...] | None = None,
        full_trajectory_mode: bool = False,
        autoencoder_mode: bool = False,
        dtype: torch.dtype = torch.float32,
        verbose: bool = False,
        use_normalization: bool = False,
        normalization_type: type[ZScoreNormalization] | None = ZScoreNormalization,
        normalization_path: str | list[str] | None = None,
        normalization_stats: dict | DictConfig | list[dict | DictConfig] | None = None,
    ):
        """Initialize MultiSpatioTemporalDataset with multiple datasets.

        Parameters
        ----------
        data_paths: str | list[str] | tuple[str, ...] | None
            Path(s) to HDF5 files. Can be a comma-separated string or list/tuple.
        data: list[dict] | None
            List of preloaded data dictionaries. Defaults to None.
        masks: TensorDM | Literal["sequential"] | Literal["combinatorial"] | None
            Masking strategy or custom mask tensor. Defaults to "sequential".
        n_steps_input: int
            Number of input time steps.
        n_steps_output: int
            Number of output time steps.
        stride: int
            Stride for sampling the data.
        input_channel_idxs: tuple[int, ...] | None
            Indices of input channels to use. Defaults to None.
        output_channel_idxs: tuple[int, ...] | None
            Indices of output channels to use. Defaults to None.
        full_trajectory_mode: bool
            If True, use full trajectories without creating subtrajectories.
        autoencoder_mode: bool
            If True, return (input, input) pairs for autoencoder training.
        dtype: torch.dtype
            Data type for tensors. Defaults to torch.float32.
        verbose: bool
            If True, print dataset information.
        use_normalization: bool
            Whether to apply Z-score normalization. Defaults to False.
        normalization_type: type[ZScoreNormalization] | None
            Normalization class to use. Defaults to ZScoreNormalization.
        normalization_path: str | list[str] | None
            Path(s) to normalization statistics file(s) (yaml).
            Can be a single path (applied to all datasets) or a list
            of paths (one per dataset). Defaults to None.
        normalization_stats: dict | DictConfig | list[dict | DictConfig] | None
            Preloaded normalization statistics. Can be a single dict
            (applied to all datasets) or a list of dicts (one per dataset).
            Defaults to None.
        """
        # Validate inputs
        if data_paths is not None and data is not None:
            msg = "Cannot provide both data_paths and data."
            raise ValueError(msg)

        if data_paths is None and data is None:
            msg = "Must provide either data_paths or data."
            raise ValueError(msg)

        # Convert data_paths to list if string
        if isinstance(data_paths, str):
            data_paths = [p.strip() for p in data_paths.split(",")]

        # Infer number of datasets
        n_datasets = self._infer_n_levels(data_paths=data_paths, data=data)

        # Handle normalization parameters - can be single values or lists
        # Convert to lists if single values provided
        if normalization_path is not None:
            if isinstance(normalization_path, str):
                # Single path - apply to all datasets
                normalization_paths = [normalization_path] * n_datasets
            else:
                # List of paths
                normalization_paths = list(normalization_path)
                if len(normalization_paths) != n_datasets:
                    msg = (
                        f"Length of normalization_path list "
                        f"({len(normalization_paths)}) must match number "
                        f"of datasets ({n_datasets})"
                    )
                    raise ValueError(msg)
        else:
            normalization_paths = [None] * n_datasets

        if normalization_stats is not None:
            if isinstance(normalization_stats, (dict, DictConfig)):
                # Single stats dict - apply to all datasets
                normalization_stats_list = [normalization_stats] * n_datasets
            else:
                # List of stats dicts
                normalization_stats_list = list(normalization_stats)
                if len(normalization_stats_list) != n_datasets:
                    msg = (
                        f"Length of normalization_stats list "
                        f"({len(normalization_stats_list)}) must match "
                        f"number of datasets ({n_datasets})"
                    )
                    raise ValueError(msg)
        else:
            normalization_stats_list = [None] * n_datasets

        # Create datasets
        self.datasets = []

        if data_paths is not None:
            for idx, path in enumerate(data_paths):
                dataset = SpatioTemporalDataset(
                    data_path=path,
                    data=None,
                    n_steps_input=n_steps_input,
                    n_steps_output=n_steps_output,
                    stride=stride,
                    input_channel_idxs=input_channel_idxs,
                    output_channel_idxs=output_channel_idxs,
                    full_trajectory_mode=full_trajectory_mode,
                    autoencoder_mode=autoencoder_mode,
                    dtype=dtype,
                    verbose=verbose,
                    use_normalization=use_normalization,
                    normalization_type=normalization_type,
                    normalization_path=normalization_paths[idx],
                    normalization_stats=normalization_stats_list[idx],
                )
                self.datasets.append(dataset)

        elif data is not None:
            for idx, data_dict in enumerate(data):
                dataset = SpatioTemporalDataset(
                    data_path=None,
                    data=data_dict,
                    n_steps_input=n_steps_input,
                    n_steps_output=n_steps_output,
                    stride=stride,
                    input_channel_idxs=input_channel_idxs,
                    output_channel_idxs=output_channel_idxs,
                    full_trajectory_mode=full_trajectory_mode,
                    autoencoder_mode=autoencoder_mode,
                    dtype=dtype,
                    verbose=verbose,
                    use_normalization=use_normalization,
                    normalization_type=normalization_type,
                    normalization_path=normalization_paths[idx],
                    normalization_stats=normalization_stats_list[idx],
                )
                self.datasets.append(dataset)

        # Validate all datasets have the same length
        if len(self.datasets) > 1:
            lengths = [len(ds) for ds in self.datasets]
            if len(set(lengths)) > 1:
                msg = f"All datasets must have the same length. Got lengths: {lengths}"
                raise ValueError(msg)

        # Handle masks
        if isinstance(masks, str):
            n_levels = len(self.datasets)
            self.masks = self.create_mask(n_levels=n_levels, mode=masks)
        else:
            self.masks = masks

        # Validate mask dimensions if provided
        if self.masks is not None:
            if self.masks.shape[0] != len(self.datasets):
                msg = (
                    f"Mask first dimension ({self.masks.shape[0]}) must match "
                    f"number of datasets ({len(self.datasets)})"
                )
                raise ValueError(msg)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        if len(self.datasets) == 0:
            return 0
        return len(self.datasets[0])

    def __getitem__(self, idx) -> ListSample:  # noqa: D105
        outs = []
        for dataset in self.datasets:
            out: Sample = dataset.__getitem__(idx)
            outs.append(out)
        # [
        #   [mask 0 for dataset 0 in window 0, mask 0 for dataset 1 in window 0],
        #   [mask 1 for dataset 0 in window 0, mask 1 for dataset 1 in window 0]
        # ]
        # in case of 2 datasets, masks will be
        # [[0, 1],
        # [1, 0],
        # [1, 1]]
        # mask: TensorDM = self.all_masks[idx]
        return ListSample(inner=outs, mask=self.masks)
