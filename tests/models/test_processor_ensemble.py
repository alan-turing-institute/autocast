import pytest
import torch
from torch import nn

from autocast.metrics.ensemble import _common_crps_score
from autocast.models.processor_ensemble import ProcessorModelEnsemble
from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor


class MockProcessor(Processor):
    def __init__(self, output_shape, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = output_shape
        # Dummy parameter to make it a valid module with state
        self.dummy_param = nn.Parameter(torch.tensor([0.0]))

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        # Return deterministic output based on input shape for shape verification
        # But for loss calculation, we might want something we can predict.
        # Here we just return random matching the batch size of x.
        b = x.shape[0]
        # output_shape is (T, H, W, C)
        return torch.zeros(b, *self.output_shape) + self.dummy_param

    def loss(self, batch: EncodedBatch) -> Tensor:
        # Base processor loss shouldn't be called if ensemble logic works in ProcessorModelEnsemble.loss
        return torch.tensor(-1.0)


def test_processor_ensemble_forward_shape():
    """Test that the ensemble forward pass returns the correct shape (B, ..., M)."""
    n_members = 3
    batch_size = 2
    # Input: (B, T, H, W, C) = (2, 1, 8, 8, 4)
    input_shape = (batch_size, 1, 8, 8, 4)
    # Output: (T, H, W, C) = (2, 8, 8, 4)
    output_field_shape = (2, 8, 8, 4)

    x = torch.randn(*input_shape)

    processor = MockProcessor(output_shape=output_field_shape)
    ensemble = ProcessorModelEnsemble(processor=processor, n_members=n_members)

    output = ensemble(x, global_cond=None)

    # Expected: (B, T, H, W, C, M)
    expected_shape = (batch_size, *output_field_shape, n_members)
    assert output.shape == expected_shape


def test_processor_ensemble_loss_integration():
    """Test the custom loss logic in ProcessorModelEnsemble using CRPS."""
    n_members = 3
    batch_size = 2

    # Shapes
    # Input: (B, T, H, W, C)
    input_shape = (batch_size, 1, 8, 8, 4)
    # Output field: (T, H, W, C)
    output_field_shape = (2, 8, 8, 4)
    # Full Output Batch: (B, T, H, W, C)
    output_batch_shape = (batch_size, *output_field_shape)

    inputs = torch.randn(*input_shape)
    targets = torch.randn(*output_batch_shape)

    batch = EncodedBatch(
        encoded_inputs=inputs,
        encoded_output_fields=targets,
        global_cond=None,
        encoded_info={},
    )

    def crps_loss(preds, targets):
        # preds: (B, ..., M)
        # targets: (B, ...)
        # _common_crps_score returns spatial map of CRPS
        score = _common_crps_score(preds, targets, adjustment_factor=1.0)
        return score.mean()

    processor = MockProcessor(output_shape=output_field_shape)
    # Ensure preds (which are 0.0 from MockProcessor) are predictable
    # MockProcessor.map returns zeros + dummy_param (0.0).

    ensemble = ProcessorModelEnsemble(
        processor=processor, n_members=n_members, loss_func=crps_loss
    )

    loss = ensemble.loss(batch)

    # Calculate expected loss manually
    # Preds are all 0.0 (from MockProcessor)
    preds = torch.zeros(batch_size, *output_field_shape, n_members)
    expected_crps_map = _common_crps_score(preds, targets, adjustment_factor=1.0)
    expected_loss = expected_crps_map.mean().item()

    assert loss.item() == pytest.approx(expected_loss)


def test_processor_ensemble_loss_fallback():
    """Test that it falls back to processor.loss if n_members=1."""
    n_members = 1
    batch_size = 2
    input_shape = (batch_size, 1, 8, 8, 4)
    output_field_shape = (2, 8, 8, 4)
    output_batch_shape = (batch_size, *output_field_shape)

    inputs = torch.randn(*input_shape)
    targets = torch.randn(*output_batch_shape)

    batch = EncodedBatch(
        encoded_inputs=inputs,
        encoded_output_fields=targets,
        global_cond=None,
        encoded_info={},
    )

    processor = MockProcessor(output_shape=output_field_shape)
    # MockProcessor.loss returns -1.0

    ensemble = ProcessorModelEnsemble(
        processor=processor,
        n_members=n_members,
        loss_func=nn.MSELoss(),  # Even with loss func, n_members=1 should fallback
    )

    loss = ensemble.loss(batch)
    assert loss.item() == -1.0
