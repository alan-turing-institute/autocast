from dataclasses import dataclass

from torch.utils.data import Dataset

from autocast.data.dataset import BatchMixin, SpatioTemporalDataset
from autocast.types.batch import Sample
from autocast.types.types import TensorBM, TensorM


@dataclass
class ListSample:  # noqa: D101
    inner: list[Sample]
    mask: list[TensorM]  # shape list[(N, M (num_masks, treat like ensemble))] # num

    # def __getitem__(self, idx) -> Sample:
    #     return self.inner[idx]


class MultiSpatioTemporalDataset(Dataset, BatchMixin):  # noqa: D101
    datasets: list[SpatioTemporalDataset]  # data: list[(N, T, S, C)]
    # masks: list[Tensor] # shape list[(N, ..., num_masks (treat like ensemble))] # num
    masks: list[TensorBM]  # shape list[(N, M (num_masks, treat like ensemble))] # num
    # Example for reaction diffusion:
    # - Deterministic loop over all combinations of missing/not missing
    #   (apart from all missing)
    # - Could also have a hierarchy of data availability (e.g. 100, 110, 111)
    #   (if a fidelity level is available, all coarser/lower levels are available too)

    def __init__(self) -> None:
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
        self.all_masks: list[list[TensorM]] = [[]]

    def __getitem__(self, idx) -> ListSample:  # noqa: D105
        outs = []
        for dataset in self.datasets:
            out: Sample = dataset.__getitem__(idx)
            outs.append(out)
        # [mask for dataset 0 in window 0, mask for dataset 1 in window 0]
        mask: list[TensorM] = self.all_masks[idx]
        return ListSample(inner=outs, mask=mask)
