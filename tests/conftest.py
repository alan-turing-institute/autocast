from collections.abc import Callable

import pytest
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from auto_cast.types import Batch, EncodedBatch


def _single_item_collate(items):
    return items[0]


@pytest.fixture
def make_toy_batch() -> Callable[..., Batch]:
    """Factory that builds lightweight `Batch` instances for tests."""

    def _factory(
        batch_size: int = 2,
        t_in: int = 2,
        t_out: int | None = None,
        w: int = 4,
        h: int = 4,
        c: int = 1,
    ) -> Batch:
        t_out = t_out or t_in
        input_fields = torch.randn(batch_size, t_in, w, h, c)
        output_fields = torch.randn(batch_size, t_out, w, h, c)
        return Batch(
            input_fields=input_fields,
            output_fields=output_fields,
            constant_scalars=None,
            constant_fields=None,
        )

    return _factory


@pytest.fixture
def toy_batch(make_toy_batch: Callable[..., Batch]) -> Batch:
    """Concrete batch instance for tests that don't need custom sizes."""

    return make_toy_batch()


class _BatchDataset(Dataset):
    def __init__(self, make_batch: Callable[..., Batch]) -> None:
        super().__init__()
        self._make_batch = make_batch

    def __len__(self) -> int:  # pragma: no cover - deterministic size
        return 2

    def __getitem__(self, idx: int) -> Batch:  # pragma: no cover - simple access
        return self._make_batch(batch_size=1)


@pytest.fixture
def batch_dataset(make_toy_batch: Callable[..., Batch]) -> Dataset:
    return _BatchDataset(make_toy_batch)


@pytest.fixture
def dummy_loader(batch_dataset: Dataset) -> DataLoader:
    """Dataloader that yields toy `Batch` samples."""

    return DataLoader(
        batch_dataset,
        batch_size=1,
        collate_fn=_single_item_collate,
        num_workers=0,
    )


@pytest.fixture
def encoded_batch(make_toy_batch: Callable[..., Batch]) -> EncodedBatch:
    """Create an `EncodedBatch` by flattening time into channels."""

    batch = make_toy_batch()
    encoded_inputs = rearrange(batch.input_fields, "b t w h c -> b (t c) w h")
    encoded_outputs = rearrange(batch.output_fields, "b t w h c -> b (t c) w h")
    return EncodedBatch(
        encoded_inputs=encoded_inputs,
        encoded_output_fields=encoded_outputs,
        encoded_info={},
    )


class _EncodedBatchDataset(Dataset):
    def __init__(self, make_batch: Callable[..., Batch]) -> None:
        super().__init__()
        self._make_batch = make_batch

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> EncodedBatch:
        batch = self._make_batch(batch_size=1)
        encoded_inputs = rearrange(batch.input_fields, "b t w h c -> b (t c) w h")
        encoded_outputs = rearrange(batch.output_fields, "b t w h c -> b (t c) w h")
        return EncodedBatch(
            encoded_inputs=encoded_inputs,
            encoded_output_fields=encoded_outputs,
            encoded_info={},
        )


@pytest.fixture
def encoded_dummy_loader(make_toy_batch: Callable[..., Batch]) -> DataLoader:
    dataset = _EncodedBatchDataset(make_toy_batch)
    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=_single_item_collate,
        num_workers=0,
    )
