"""Unit tests for autocast.scripts.data module."""

import torch

from autocast.scripts.data import batch_to_device
from autocast.types import Batch


def test_batch_to_device_moves_required_fields():
    batch = Batch(
        input_fields=torch.randn(2, 2, 8, 8, 3),
        output_fields=torch.randn(2, 2, 8, 8, 3),
        constant_scalars=None,
        constant_fields=None,
    )
    result = batch_to_device(batch, torch.device("cpu"))
    assert result.input_fields.device.type == "cpu"
    assert result.output_fields.device.type == "cpu"


def test_batch_to_device_moves_optional_fields():
    batch = Batch(
        input_fields=torch.randn(2, 2, 8, 8, 3),
        output_fields=torch.randn(2, 2, 8, 8, 3),
        constant_scalars=torch.randn(2, 4),
        constant_fields=torch.randn(2, 8, 8, 2),
    )
    result = batch_to_device(batch, torch.device("cpu"))
    assert result.constant_scalars is not None
    assert result.constant_scalars.device.type == "cpu"
    assert result.constant_fields is not None
    assert result.constant_fields.device.type == "cpu"


def test_batch_to_device_preserves_none_fields():
    batch = Batch(
        input_fields=torch.randn(2, 2, 8, 8, 3),
        output_fields=torch.randn(2, 2, 8, 8, 3),
        constant_scalars=None,
        constant_fields=None,
    )
    result = batch_to_device(batch, torch.device("cpu"))
    assert result.constant_scalars is None
    assert result.constant_fields is None
