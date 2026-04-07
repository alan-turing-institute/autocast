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
        data_paths: str | None,
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
        normalization_path: str | None = None,
        normalization_stats: dict | DictConfig | None = None,
    ):
        # TODO: handle either mutiple data_paths or multiple data dicts (e.g. with a
        # list of paths or a list of dicts with paths and other info like normalization
        # stats)

        if isinstance(masks, str):
            n_levels = self._infer_n_levels(data_paths=data_paths, data=data)
            self.masks = self.create_mask(n_levels=n_levels, mode=masks)
        else:
            self.masks = masks

        # TODO: create windowed masks like in SpatioTemporalDataset that perfectly
        # match the windowed data samples
        # E.g.
        # Window 1 (4 in, 2 out): []
        # self.all_masks = [
        #   [mask for dataset 0 in window 0, mask for dataset 1 in window 0],
        #   [mask for dataset 0 in window 1, mask for dataset 1 in window 1]]
        #   ...
        # ]
        # NOTE: each mask for a window is an ensemble Tensor (shape M)
        # TODO: these needs to be constructed such that the windowing removes the
        # "batch"/"trajetory" dim as is done in SpatioTemporalDataset
        # This leaves all_masks as a list over trajectories with each element of the
        # list being a list over the datasets with a TensorM for each dataset to capture
        # ensemble masks for that dataset in that window (e.g. different combinations of
        # missing data).
        # TODO: co-pilot to apply from SpatioTemporalDataset to create windowed masks
        # TODO: ALTERNATIVE is to just load temporal masks (user-provided)
        # self.all_masks: list[TensorDM] = []

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
