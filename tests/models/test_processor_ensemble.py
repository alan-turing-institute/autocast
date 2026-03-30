import pytest
import torch
from torch import nn

from autocast.losses.ensemble import CRPSLoss
from autocast.models.processor_ensemble import ProcessorModelEnsemble
from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor


class SimpleLinearProcessor(Processor):
    def __init__(self, input_dim, output_dim, output_time_steps, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(input_dim, output_dim)
        self.output_time_steps = output_time_steps

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:  # noqa: ARG002
        y = self.linear(x)
        # Simple time expansion for testing T_in=1 -> T_out=N
        if x.shape[1] == 1 and self.output_time_steps > 1:
            y = y.expand(-1, self.output_time_steps, -1, -1, -1)

        return y

    def loss(self, batch: EncodedBatch) -> Tensor:
        preds = self.map(batch.encoded_inputs, batch.global_cond)
        return nn.functional.mse_loss(preds, batch.encoded_output_fields)


def test_processor_ensemble_forward_shape():
    """Test that the ensemble forward pass returns the correct shape (B, ..., M)."""
    n_members = 3
    batch_size = 2
    input_shape = (batch_size, 1, 8, 8, 4)
    output_field_shape = (2, 8, 8, 4)

    x = torch.randn(*input_shape)

    processor = SimpleLinearProcessor(input_dim=4, output_dim=4, output_time_steps=2)
    ensemble = ProcessorModelEnsemble(processor=processor, n_members=n_members)

    output = ensemble(x, global_cond=None)

    # Expected: (B, T, H, W, C, M)
    expected_shape = (batch_size, *output_field_shape, n_members)
    assert output.shape == expected_shape


def test_processor_ensemble_loss_integration():
    """Test the custom loss logic in ProcessorModelEnsemble using CRPS."""
    n_members = 3
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

    processor = SimpleLinearProcessor(input_dim=4, output_dim=4, output_time_steps=2)
    ensemble = ProcessorModelEnsemble(
        processor=processor, n_members=n_members, loss_func=CRPSLoss()
    )

    loss = ensemble.loss(batch)

    # Calculate expected loss manually - identical predictions across ensemble members
    with torch.no_grad():
        single_preds = processor.map(inputs, None)  # (B, T, H, W, C)
        # CRPS with M replicated identical predictions reduces to MAE
        expected_loss = torch.mean(torch.abs(single_preds - targets))

    assert loss.item() == pytest.approx(expected_loss.item())


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

    processor = SimpleLinearProcessor(input_dim=4, output_dim=4, output_time_steps=2)

    ensemble = ProcessorModelEnsemble(
        processor=processor,
        n_members=n_members,
        loss_func=nn.MSELoss(),  # Even with loss func, n_members=1 should fallback
    )

    loss = ensemble.loss(batch)

    # Fallback calls super().loss() which calls processor.loss()
    expected_loss = processor.loss(batch)

    assert loss.item() == pytest.approx(expected_loss.item())


def test_processor_ensemble_predict_returns_flat_batch():
    """Test that _predict returns no ensemble dim for rollout compatibility."""
    n_members = 3
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

    processor = SimpleLinearProcessor(input_dim=4, output_dim=4, output_time_steps=2)
    ensemble = ProcessorModelEnsemble(processor=processor, n_members=n_members)

    preds = ensemble._predict(batch)

    # _predict returns flat (B, T, H, W, C) — ensemble expansion is handled by callers
    expected_shape = (batch_size, *output_field_shape)
    assert preds.shape == expected_shape
