from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from einops import rearrange
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from autocast.types import Batch, EncodedBatch


@pytest.fixture
def REPO_ROOT() -> Path:
    return Path(__file__).parent.parent


def _make_batch(
    batch_size: int = 1,
    t: int = 1,
    w: int = 2,
    h: int = 3,
    c: int = 2,
    const_c: int = 1,
    scalar_c: int = 1,
) -> Batch:
    input_fields = torch.arange(batch_size * t * w * h * c, dtype=torch.float32)
    input_fields = input_fields.view(batch_size, t, w, h, c)
    output_fields = torch.zeros(batch_size, t, w, h, c)
    constant_fields = torch.ones(batch_size, w, h, const_c)
    constant_scalars = torch.full((batch_size, scalar_c), 5.0)
    return Batch(
        input_fields=input_fields,
        output_fields=output_fields,
        constant_scalars=constant_scalars,
        constant_fields=constant_fields,
    )


def assert_output_valid(output: Tensor, expected_shape: tuple, name: str = "Output"):
    """Assert output has expected shape and contains no NaN values."""
    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.shape}"
    )
    assert not output.isnan().any(), f"{name} contains NaN values"


def assert_module_initialized(module: nn.Module, module_name: str):
    """Assert module is properly initialized with valid parameters."""
    assert isinstance(module, nn.Module), f"{module_name} is not a Module"
    params = list(module.parameters())
    assert len(params) > 0, f"{module_name} has no parameters"
    for param in params:
        assert not param.isnan().any(), "Parameters contain NaN values"


def assert_gradients_flow(module: nn.Module, input_tensor: Tensor, module_name: str):
    """Assert gradients flow through module to input and all parameters."""
    output = module.forward(input_tensor)
    loss = output.sum()
    loss.backward()

    # Check input gradients
    assert input_tensor.grad is not None, f"{module_name}: No gradients on input"
    assert not input_tensor.grad.isnan().any(), (
        f"{module_name}: Input gradients contain NaN"
    )

    # Check parameter gradients
    for name, param in module.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, (
                f"{module_name}: No gradients on parameter {name}"
            )
            assert not param.grad.isnan().any(), (
                f"{module_name}: Parameter {name} gradients contain NaN"
            )


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
        label=None,
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
            label=None,
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


if __name__ == "__main__":
    print(Path(__file__).parent.parent)
    print("banana")
