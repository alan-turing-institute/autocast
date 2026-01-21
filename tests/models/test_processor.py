"""Tests for ProcessorModel with EncodedBatch training."""

import lightning as L
import pytest
import torch
from azula.noise import VPSchedule
from torch.utils.data import DataLoader, Dataset

from autocast.models.processor import ProcessorModel
from autocast.nn.unet import TemporalUNetBackbone
from autocast.processors.diffusion import DiffusionProcessor
from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.types import EncodedBatch


def _single_item_collate(items):
    """Collate function that returns the single item (already batched)."""
    return items[0]


class MockEncodedDataset(Dataset):
    """Mock dataset that generates EncodedBatch samples for testing."""

    def __init__(
        self,
        *,
        n_steps_input: int = 1,
        n_steps_output: int = 4,
        n_channels_in: int = 8,
        n_channels_out: int = 8,
        spatial_size: int = 8,
        batch_size: int = 2,
        length: int = 4,
    ) -> None:
        super().__init__()
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.spatial_size = spatial_size
        self.batch_size = batch_size
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _: int) -> EncodedBatch:
        encoded_inputs = torch.randn(
            self.batch_size,
            self.n_steps_input,
            self.spatial_size,
            self.spatial_size,
            self.n_channels_in,
        )
        encoded_outputs = torch.randn(
            self.batch_size,
            self.n_steps_output,
            self.spatial_size,
            self.spatial_size,
            self.n_channels_out,
        )
        return EncodedBatch(
            encoded_inputs=encoded_inputs,
            encoded_output_fields=encoded_outputs,
            global_cond=None,
            encoded_info={},
        )


def _build_dataloader(
    *,
    n_steps_input: int = 1,
    n_steps_output: int = 4,
    n_channels_in: int = 8,
    n_channels_out: int = 8,
    spatial_size: int = 8,
    batch_size: int = 2,
) -> DataLoader:
    """Build a DataLoader with mock EncodedBatch data."""
    dataset = MockEncodedDataset(
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        spatial_size=spatial_size,
        batch_size=batch_size,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=_single_item_collate,
        num_workers=0,
    )


def _build_diffusion_processor(
    *,
    n_steps_input: int = 1,
    n_steps_output: int = 4,
    n_channels_in: int = 8,
    n_channels_out: int = 8,
) -> DiffusionProcessor:
    """Build a DiffusionProcessor with a TemporalUNetBackbone."""
    backbone = TemporalUNetBackbone(
        in_channels=n_channels_out,
        out_channels=n_channels_out,
        cond_channels=n_channels_in,
        n_steps_output=n_steps_output,
        n_steps_input=n_steps_input,
        mod_features=64,
        hid_channels=(16, 32),
        hid_blocks=(1, 1),
        spatial=2,
        periodic=False,
    )
    return DiffusionProcessor(
        backbone=backbone,
        schedule=VPSchedule(),
        n_steps_output=n_steps_output,
        n_channels_out=n_channels_out,
    )


def _build_flow_matching_processor(
    *,
    n_steps_input: int = 1,
    n_steps_output: int = 4,
    n_channels_in: int = 8,
    n_channels_out: int = 8,
) -> FlowMatchingProcessor:
    """Build a FlowMatchingProcessor with a TemporalUNetBackbone."""
    backbone = TemporalUNetBackbone(
        in_channels=n_channels_out,
        out_channels=n_channels_out,
        cond_channels=n_channels_in,
        n_steps_output=n_steps_output,
        n_steps_input=n_steps_input,
        mod_features=64,
        hid_channels=(16, 32),
        hid_blocks=(1, 1),
        spatial=2,
        periodic=False,
    )
    return FlowMatchingProcessor(
        backbone=backbone,
        n_steps_output=n_steps_output,
        n_channels_out=n_channels_out,
        flow_ode_steps=2,
    )


def _make_encoded_batch(
    *,
    n_steps_input: int = 1,
    n_steps_output: int = 4,
    n_channels_in: int = 8,
    n_channels_out: int = 8,
    batch_size: int = 2,
    spatial_size: int = 8,
    global_cond: torch.Tensor | None = None,
    encoded_info: dict | None = None,
) -> EncodedBatch:
    """Create an EncodedBatch for testing."""
    return EncodedBatch(
        encoded_inputs=torch.randn(
            batch_size, n_steps_input, spatial_size, spatial_size, n_channels_in
        ),
        encoded_output_fields=torch.randn(
            batch_size, n_steps_output, spatial_size, spatial_size, n_channels_out
        ),
        global_cond=global_cond,
        encoded_info=encoded_info or {},
    )


# --- Basic ProcessorModel Tests ---


def test_processor_model_forward():
    """Test that ProcessorModel forward pass works with encoded inputs."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(processor=processor, learning_rate=1e-4)

    batch = _make_encoded_batch()
    output = model(batch.encoded_inputs)

    assert output.shape == (2, 4, 8, 8, 8), f"Unexpected output shape: {output.shape}"


def test_processor_model_training_step():
    """Test that training_step computes loss correctly with EncodedBatch."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(processor=processor, learning_rate=1e-4)

    batch = _make_encoded_batch()
    loss = model.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be scalar"
    assert not loss.isnan(), "Loss should not be NaN"
    assert loss.item() > 0, "Loss should be positive"


def test_processor_model_validation_step():
    """Test that validation_step computes loss correctly with EncodedBatch."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(processor=processor, learning_rate=1e-4)

    batch = _make_encoded_batch()
    loss = model.validation_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be scalar"
    assert not loss.isnan(), "Loss should not be NaN"


def test_processor_model_test_step():
    """Test that test_step computes loss correctly with EncodedBatch."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(processor=processor, learning_rate=1e-4)

    batch = _make_encoded_batch()
    loss = model.test_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be scalar"
    assert not loss.isnan(), "Loss should not be NaN"


def test_processor_model_diffusion_trainer_fit():
    """Test full training loop with Lightning Trainer and DiffusionProcessor."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(processor=processor, learning_rate=1e-3)

    train_loader = _build_dataloader()
    val_loader = _build_dataloader()

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, train_loader, val_loader)


def test_processor_model_flow_matching_trainer_fit():
    """Test full training loop with Lightning Trainer and FlowMatchingProcessor."""
    processor = _build_flow_matching_processor()
    model = ProcessorModel(processor=processor, learning_rate=1e-3)

    train_loader = _build_dataloader()
    val_loader = _build_dataloader()

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, train_loader, val_loader)


def test_processor_model_clone_batch():
    """Test that _clone_batch creates independent copy of EncodedBatch."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(processor=processor, learning_rate=1e-4)

    original = _make_encoded_batch(encoded_info={"key": torch.tensor([1.0, 2.0])})
    cloned = model._clone_batch(original)

    # Modify original
    original.encoded_inputs.fill_(0)
    original.encoded_output_fields.fill_(0)

    # Cloned should be unchanged
    assert cloned.encoded_inputs.abs().sum() > 0, "Clone should be independent"
    assert cloned.encoded_output_fields.abs().sum() > 0, "Clone should be independent"


def test_processor_model_advance_batch():
    """Test that _advance_batch correctly advances the batch for rollout."""
    processor = _build_diffusion_processor(n_steps_input=2, n_steps_output=4)
    model = ProcessorModel(processor=processor, learning_rate=1e-4, stride=2)

    batch = EncodedBatch(
        encoded_inputs=torch.randn(2, 2, 8, 8, 8),  # n_steps_input=2
        encoded_output_fields=torch.randn(2, 8, 8, 8, 8),  # 8 output steps
        global_cond=None,
        encoded_info={},
    )

    next_inputs = torch.randn(2, 4, 8, 8, 8)  # Predictions
    advanced = model._advance_batch(batch, next_inputs, stride=2)

    assert advanced.encoded_inputs.shape[1] == 2, "Should maintain n_steps_input"
    assert advanced.encoded_output_fields.shape[1] == 6, (
        "Should have 8-2=6 outputs left"
    )


def test_processor_model_configure_optimizers():
    """Test that configure_optimizers returns Adam optimizer."""
    processor = _build_diffusion_processor()
    lr = 5e-4
    model = ProcessorModel(processor=processor, learning_rate=lr)

    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == lr


def test_processor_model_map():
    """Test that map method works correctly."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(processor=processor, learning_rate=1e-4)

    x = torch.randn(2, 1, 8, 8, 8)
    output = model.map(x, None)

    assert output.shape == (2, 4, 8, 8, 8), f"Unexpected shape: {output.shape}"
    assert not output.isnan().any(), "Output should not contain NaN"


# --- Parametrized Tests for Different Configurations ---


@pytest.mark.parametrize("n_steps_input", [1, 2, 4])
@pytest.mark.parametrize("n_steps_output", [1, 2, 4])
def test_varying_temporal_steps(n_steps_input: int, n_steps_output: int):
    """Test ProcessorModel with different input/output temporal configurations."""
    n_channels = 8
    processor = _build_diffusion_processor(
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        n_channels_in=n_channels,
        n_channels_out=n_channels,
    )
    model = ProcessorModel(processor=processor, learning_rate=1e-4)

    batch = _make_encoded_batch(
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        n_channels_in=n_channels,
        n_channels_out=n_channels,
    )

    loss = model.training_step(batch, batch_idx=0)
    assert not loss.isnan(), (
        f"Loss NaN for steps_in={n_steps_input}, steps_out={n_steps_output}"
    )


@pytest.mark.parametrize(("n_channels_in", "n_channels_out"), [(4, 4), (8, 8), (4, 8)])
def test_varying_channels(n_channels_in: int, n_channels_out: int):
    """Test ProcessorModel with different channel configurations."""
    processor = _build_diffusion_processor(
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
    )
    model = ProcessorModel(processor=processor, learning_rate=1e-4)

    batch = _make_encoded_batch(
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
    )

    loss = model.training_step(batch, batch_idx=0)
    assert not loss.isnan(), (
        f"Loss NaN for channels_in={n_channels_in}, channels_out={n_channels_out}"
    )


# --- Tests with Labels ---


def test_encoded_batch_with_global_conds():
    """Test that ProcessorModel works when global_conds are present."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(processor=processor, learning_rate=1e-4)

    batch = _make_encoded_batch(global_cond=torch.randn(2, 1, 3))
    loss = model.training_step(batch, batch_idx=0)

    assert not loss.isnan(), "Loss should not be NaN with global_conds"


def test_clone_batch_preserves_global_conds():
    """Test that _clone_batch preserves global_cond correctly."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(processor=processor, learning_rate=1e-4)

    batch = _make_encoded_batch(global_cond=torch.randn(2, 1, 3))
    cloned = model._clone_batch(batch)

    assert cloned.global_cond is not None, "Clone should preserve global_cond"
    assert torch.allclose(batch.global_cond, cloned.global_cond), (  # type: ignore  # noqa: PGH003
        "Clone should have same values"
    )
    assert batch.global_cond is not cloned.global_cond, "Clone should be a new tensor"
