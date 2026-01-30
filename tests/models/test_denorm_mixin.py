"""Tests for DenormMixin functionality."""

import contextlib

import lightning as L
import pytest
import torch
from the_well.data.normalization import ZScoreNormalization

from autocast.models.denorm_mixin import DenormMixin
from autocast.types.batch import Batch


class SimpleDenormModel(DenormMixin, L.LightningModule):
    """Simple model for testing DenormMixin."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, batch: Batch) -> torch.Tensor:
        # Simple forward pass for testing
        return batch.input_fields


@pytest.fixture
def mock_normalizer():
    """Create a mock ZScoreNormalization object."""
    # Create simple normalization stats following the format used in the_well
    # The stats dict should have mean, std, mean_delta, std_delta at the top level
    stats = {
        "mean": {"U": 2.0, "V": 4.0},
        "std": {"U": 1.0, "V": 2.0},
        "mean_delta": {"U": 0.0, "V": 0.0},
        "std_delta": {"U": 0.1, "V": 0.2},
    }
    norm = ZScoreNormalization(
        stats=stats,
        core_field_names=["U", "V"],
        core_constant_field_names=[],
    )
    return norm


@pytest.fixture
def normalized_batch():
    """Create a batch with normalized data."""
    # Create data that's been normalized
    # Original: Channel 0 = [2, 3, 4], Channel 1 = [4, 6, 8]
    # Normalized: Channel 0 = [0, 1, 2], Channel 1 = [0, 1, 2]
    input_fields = torch.tensor(
        [
            [  # Batch 1
                [  # Timestep 1
                    [[0.0, 0.0], [1.0, 1.0]],  # Spatial grid
                    [[2.0, 2.0], [0.0, 0.0]],
                ]
            ]
        ]
    )
    output_fields = input_fields.clone()
    return Batch(
        input_fields=input_fields,
        output_fields=output_fields,
        constant_scalars=None,
        constant_fields=None,
    )


def test_denorm_mixin_no_normalizer():
    """Test that DenormMixin returns unchanged data when no normalizer is set."""
    model = SimpleDenormModel()
    tensor = torch.randn(2, 4, 8, 8, 3)

    # Without normalizer, should return same tensor
    result = model.denormalize_tensor(tensor)
    assert torch.allclose(result, tensor)


def test_denorm_mixin_with_normalizer(mock_normalizer):
    """Test that DenormMixin correctly denormalizes when normalizer is set."""
    model = SimpleDenormModel()
    model.norm = mock_normalizer

    # Create normalized tensor (z-score normalized)
    # Original values: [2.0, 4.0] -> normalized: [0.0, 0.0]
    normalized = torch.zeros(1, 1, 2, 2, 2)

    result = model.denormalize_tensor(normalized)

    # Should denormalize back to original: x = z * std + mean
    # Channel 0: 0 * 1.0 + 2.0 = 2.0
    # Channel 1: 0 * 2.0 + 4.0 = 4.0
    expected = torch.zeros_like(normalized)
    expected[..., 0] = 2.0
    expected[..., 1] = 4.0

    assert torch.allclose(result, expected)


def test_predict_step_without_normalizer():
    """Test predict_step returns raw predictions when no normalizer is set."""
    model = SimpleDenormModel()
    batch = Batch(
        input_fields=torch.randn(2, 4, 8, 8, 3),
        output_fields=torch.randn(2, 4, 8, 8, 3),
        constant_scalars=None,
        constant_fields=None,
    )

    predictions = model.predict_step(batch, batch_idx=0)
    # Should just be forward pass output
    assert torch.allclose(predictions, batch.input_fields)


def test_predict_step_with_normalizer(mock_normalizer):
    """Test predict_step denormalizes predictions when normalizer is set."""
    model = SimpleDenormModel()
    model.norm = mock_normalizer

    # Create batch with normalized data (zero mean)
    batch = Batch(
        input_fields=torch.zeros(1, 1, 2, 2, 2),
        output_fields=torch.zeros(1, 1, 2, 2, 2),
        constant_scalars=None,
        constant_fields=None,
    )

    predictions = model.predict_step(batch, batch_idx=0)

    # Predictions should be denormalized back to mean values
    # Channel 0: 0 * 1.0 + 2.0 = 2.0
    # Channel 1: 0 * 2.0 + 4.0 = 4.0
    assert torch.allclose(predictions[..., 0], torch.full((1, 1, 2, 2), 2.0))
    assert torch.allclose(predictions[..., 1], torch.full((1, 1, 2, 2), 4.0))


def test_denormalize_tensor_shapes():
    """Test denormalize_tensor handles different tensor shapes correctly."""
    model = SimpleDenormModel()

    # Test various shapes
    shapes = [
        (1, 1, 8, 8, 2),  # 2D spatial
        (2, 4, 16, 16, 3),  # Different batch/time
        (1, 1, 8, 8, 8, 4),  # 3D spatial
    ]

    for shape in shapes:
        tensor = torch.randn(*shape)
        result = model.denormalize_tensor(tensor)
        assert result.shape == tensor.shape, f"Shape mismatch for {shape}"


def test_connect_normalizer_from_datamodule(mock_normalizer):
    """Test that _connect_normalizer pulls normalizer from datamodule."""
    model = SimpleDenormModel()

    # Create mock datamodule with normalizer
    class MockDataModule:
        class MockDataset:
            def __init__(self):
                self.norm = mock_normalizer

        def __init__(self):
            self.train_dataset = self.MockDataset()

    class MockTrainer:
        def __init__(self):
            self.datamodule = MockDataModule()

    # Directly set _trainer to avoid property check
    model._trainer = MockTrainer()  # type: ignore[assignment]
    model._connect_normalizer()

    assert model.norm is mock_normalizer


def test_connect_normalizer_no_trainer():
    """Test that _connect_normalizer handles missing trainer gracefully."""
    model = SimpleDenormModel()
    # Lightning raises error when accessing trainer property if not attached
    # _connect_normalizer should handle this gracefully with hasattr check
    with contextlib.suppress(RuntimeError):
        model._connect_normalizer()
    # Should still have no normalizer set
    assert model.norm is None


def test_connect_normalizer_no_datamodule():
    """Test that _connect_normalizer handles missing datamodule gracefully."""
    model = SimpleDenormModel()

    class MockTrainer:
        pass

    # Directly set _trainer to avoid property check
    model._trainer = MockTrainer()  # type: ignore[assignment]
    model._connect_normalizer()
    assert model.norm is None


def test_connect_normalizer_no_norm_attribute():
    """Test that _connect_normalizer handles missing norm attribute gracefully."""
    model = SimpleDenormModel()

    class MockDataModule:
        class MockDataset:
            pass

        def __init__(self):
            self.train_dataset = self.MockDataset()

    class MockTrainer:
        def __init__(self):
            self.datamodule = MockDataModule()

    # Directly set _trainer to avoid property check
    model._trainer = MockTrainer()  # type: ignore[assignment]
    model._connect_normalizer()
    assert model.norm is None


def test_on_fit_start_connects_normalizer(mock_normalizer):
    """Test that on_fit_start automatically connects normalizer."""
    model = SimpleDenormModel()

    class MockDataModule:
        class MockDataset:
            def __init__(self):
                self.norm = mock_normalizer

        def __init__(self):
            self.train_dataset = self.MockDataset()

    class MockTrainer:
        def __init__(self):
            self.datamodule = MockDataModule()

    # Directly set _trainer to avoid property check
    model._trainer = MockTrainer()  # type: ignore[assignment]
    model.on_fit_start()

    assert model.norm is mock_normalizer


def test_on_predict_start_connects_normalizer(mock_normalizer):
    """Test that on_predict_start automatically connects normalizer."""
    model = SimpleDenormModel()

    class MockDataModule:
        class MockDataset:
            def __init__(self):
                self.norm = mock_normalizer

        def __init__(self):
            self.train_dataset = self.MockDataset()

    class MockTrainer:
        def __init__(self):
            self.datamodule = MockDataModule()

    # Directly set _trainer to avoid property check
    model._trainer = MockTrainer()  # type: ignore[assignment]
    model.on_predict_start()

    assert model.norm is mock_normalizer
