"""Tests for ProcessorModel with EncodedBatch training."""

from types import SimpleNamespace
from unittest.mock import Mock

import lightning as L
import pytest
import torch
from azula.noise import VPSchedule
from conftest import CondCaptureProcessor, get_optimizer_config
from torch.utils.data import DataLoader, Dataset

from autocast.metrics import MSE
from autocast.models.processor import ProcessorModel
from autocast.nn.unet import TemporalUNetBackbone
from autocast.nn.vit import TemporalViTBackbone
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
    include_global_cond: bool = False,
    global_cond_channels: int | None = None,
) -> DiffusionProcessor:
    """Build a DiffusionProcessor with a TemporalUNetBackbone."""
    backbone = TemporalUNetBackbone(
        in_channels=n_channels_out,
        out_channels=n_channels_out,
        cond_channels=n_channels_in,
        n_steps_output=n_steps_output,
        n_steps_input=n_steps_input,
        include_global_cond=include_global_cond,
        global_cond_channels=global_cond_channels,
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
        include_global_cond=False,
        global_cond_channels=None,
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
    model = ProcessorModel(processor=processor, optimizer_config=get_optimizer_config())

    batch = _make_encoded_batch()
    output = model(batch.encoded_inputs, batch.global_cond)

    assert output.shape == (2, 4, 8, 8, 8), f"Unexpected output shape: {output.shape}"


def test_processor_model_training_step():
    """Test that training_step computes loss correctly with EncodedBatch."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(processor=processor, optimizer_config=get_optimizer_config())

    batch = _make_encoded_batch()
    loss = model.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be scalar"
    assert not loss.isnan(), "Loss should not be NaN"
    assert loss.item() > 0, "Loss should be positive"


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_train_loss_log_interval_requires_positive_integer(interval):
    with pytest.raises(ValueError, match="must be a positive integer"):
        ProcessorModel(
            processor=CondCaptureProcessor(), train_loss_log_every_n_steps=interval
        )


@pytest.mark.parametrize("interval", [None, 1, 3, "legacy"])
def test_train_loss_logging_preserves_loss_metrics_and_diagnostics(
    monkeypatch, interval
):
    kwargs = (
        {}
        if interval in (None, "legacy")
        else {"train_loss_log_every_n_steps": interval}
    )
    model = ProcessorModel(
        processor=CondCaptureProcessor(), train_metrics=[MSE()], **kwargs
    )
    if interval == "legacy":
        # Full-model pickles created before this option have no instance field.
        del model.train_loss_log_every_n_steps
    batch = _make_encoded_batch()
    loss = torch.tensor(2.5, requires_grad=True)
    compute_loss = Mock(return_value=loss)
    log = Mock()
    diagnostics = Mock()
    metrics = Mock()
    monkeypatch.setattr(model.processor, "loss", compute_loss)
    monkeypatch.setattr(model, "log", log)
    monkeypatch.setattr(model, "_log_processor_diagnostics", diagnostics)
    monkeypatch.setattr(model, "_update_and_log_metrics", metrics)
    monkeypatch.setattr(
        model, "_predict_for_metrics", Mock(return_value=batch.encoded_output_fields)
    )
    trainer = SimpleNamespace(global_step=0)
    monkeypatch.setattr(model, "_trainer", trainer)

    # Repeated global steps model accumulation; batch_idx resets model epochs.
    steps = [0, 0, 1, 2, 2, 3, 5]
    logged_steps = []
    for index, step in enumerate(steps):
        trainer.global_step = step
        log.reset_mock()
        assert model.training_step(batch, batch_idx=index % 4) is loss
        if log.called:
            logged_steps.append(step)
            log.assert_called_once_with(
                "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=2
            )
    assert logged_steps == ([2, 2, 5] if interval == 3 else steps)
    assert (
        compute_loss.call_count
        == diagnostics.call_count
        == metrics.call_count
        == len(steps)
    )
    loss.backward()
    assert loss.grad == 1


@pytest.mark.parametrize("phase", ["validation", "test"])
def test_train_loss_interval_does_not_change_evaluation_logging(monkeypatch, phase):
    model = ProcessorModel(
        processor=CondCaptureProcessor(),
        train_loss_log_every_n_steps=64,
        val_metrics=[],
        test_metrics=[],
    )
    log = Mock()
    monkeypatch.setattr(model, "log", log)
    loss = getattr(model, f"{phase}_step")(_make_encoded_batch(), batch_idx=0)
    name = "val_loss" if phase == "validation" else "test_loss"
    log.assert_called_once_with(name, loss, prog_bar=True, sync_dist=True, batch_size=2)


def test_train_loss_log_cadence_survives_accumulation_and_resume(monkeypatch, tmp_path):
    model = ProcessorModel(
        processor=_build_diffusion_processor(),
        optimizer_config=get_optimizer_config(),
        train_loss_log_every_n_steps=3,
    )
    logged_steps = []
    original_log = model.log

    def record_log(name, *args, **kwargs):
        if name == "train_loss":
            logged_steps.append((model.current_epoch, model.global_step))
        return original_log(name, *args, **kwargs)

    monkeypatch.setattr(model, "log", record_log)
    trainer_kwargs = {
        "accelerator": "cpu",
        "accumulate_grad_batches": 2,
        "log_every_n_steps": 3,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    trainer = L.Trainer(max_epochs=2, **trainer_kwargs)
    loader = _build_dataloader()
    trainer.fit(model, loader)
    assert trainer.global_step == 4
    assert logged_steps == [(1, 2), (1, 2)]
    checkpoint = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(checkpoint)

    resumed = L.Trainer(max_epochs=3, **trainer_kwargs)
    resumed.fit(model, loader, ckpt_path=checkpoint)
    assert resumed.global_step == 6
    assert logged_steps == [(1, 2), (1, 2), (2, 5), (2, 5)]


def test_processor_model_validation_step():
    """Test that validation_step computes loss correctly with EncodedBatch."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(processor=processor, optimizer_config=get_optimizer_config())

    batch = _make_encoded_batch()
    loss = model.validation_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be scalar"
    assert not loss.isnan(), "Loss should not be NaN"


def test_processor_model_test_step():
    """Test that test_step computes loss correctly with EncodedBatch."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(processor=processor, optimizer_config=get_optimizer_config())

    batch = _make_encoded_batch()
    loss = model.test_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be scalar"
    assert not loss.isnan(), "Loss should not be NaN"


def test_processor_model_diffusion_trainer_fit():
    """Test full training loop with Lightning Trainer and DiffusionProcessor."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(
        processor=processor,
        optimizer_config=get_optimizer_config(learning_rate=1e-3),
    )

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
    model = ProcessorModel(
        processor=processor,
        optimizer_config=get_optimizer_config(learning_rate=1e-3),
    )

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
    model = ProcessorModel(processor=processor, optimizer_config=get_optimizer_config())

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
    model = ProcessorModel(
        processor=processor,
        optimizer_config=get_optimizer_config(),
        stride=2,
    )

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
    model = ProcessorModel(
        processor=processor,
        optimizer_config=get_optimizer_config(learning_rate=lr),
    )

    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == lr


def test_processor_model_map():
    """Test that map method works correctly."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(processor=processor, optimizer_config=get_optimizer_config())

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
    model = ProcessorModel(processor=processor, optimizer_config=get_optimizer_config())

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
    model = ProcessorModel(processor=processor, optimizer_config=get_optimizer_config())

    batch = _make_encoded_batch(
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
    )

    loss = model.training_step(batch, batch_idx=0)
    assert not loss.isnan(), (
        f"Loss NaN for channels_in={n_channels_in}, channels_out={n_channels_out}"
    )


# --- Tests with global_cond ---


def test_encoded_batch_with_global_conds():
    """Test that ProcessorModel works when global_conds are present."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(processor=processor, optimizer_config=get_optimizer_config())

    batch = _make_encoded_batch(global_cond=torch.randn(2, 1, 3))
    loss = model.training_step(batch, batch_idx=0)

    assert not loss.isnan(), "Loss should not be NaN with global_conds"


def test_global_cond_passed_to_processor_predict():
    processor = CondCaptureProcessor()
    model = ProcessorModel(processor=processor, optimizer_config=get_optimizer_config())
    global_cond = torch.randn(2, 1, 3)
    batch = _make_encoded_batch(global_cond=global_cond)

    _ = model._predict(batch)

    assert processor.last_global_cond is not None
    assert torch.allclose(processor.last_global_cond, global_cond)


def test_clone_batch_preserves_global_conds():
    """Test that _clone_batch preserves global_cond correctly."""
    processor = _build_diffusion_processor()
    model = ProcessorModel(processor=processor, optimizer_config=get_optimizer_config())

    batch = _make_encoded_batch(global_cond=torch.randn(2, 1, 3))
    cloned = model._clone_batch(batch)

    assert cloned.global_cond is not None, "Clone should preserve global_cond"
    assert torch.allclose(batch.global_cond, cloned.global_cond), (  # type: ignore  # noqa: PGH003
        "Clone should have same values"
    )
    assert batch.global_cond is not cloned.global_cond, "Clone should be a new tensor"


def test_processor_model_enforces_global_cond_usage():
    """Test that backbone actually uses global_cond when configured to do so."""
    # Configure processor to require global_cond
    global_cond_dim = 5
    processor = _build_diffusion_processor(
        include_global_cond=True,
        global_cond_channels=global_cond_dim,
    )
    model = ProcessorModel(processor=processor, optimizer_config=get_optimizer_config())

    # Test success with valid global_cond
    batch_with_cond = _make_encoded_batch(global_cond=torch.randn(2, global_cond_dim))
    loss = model.training_step(batch_with_cond, batch_idx=0)
    assert not loss.isnan()

    # Test failure when global_cond is missing (None)
    batch_without_cond = _make_encoded_batch(global_cond=None)

    # Expect ValueError from TemporalBackboneBase:
    # "Model init with global_cond_channels but no global_cond provided"
    with pytest.raises(ValueError, match="no global_cond provided"):
        model.training_step(batch_without_cond, batch_idx=0)


def test_processor_ignores_global_cond_when_disabled():
    """Test that global_cond is ignored when include_global_cond is False."""
    # Configure processor to not use global_cond even when present
    processor = _build_diffusion_processor(
        include_global_cond=False,
        global_cond_channels=None,
    )
    model = ProcessorModel(processor=processor, optimizer_config=get_optimizer_config())

    # Create dummy input
    batch_size = 2
    # Use simple shapes matching default build params
    x = torch.randn(batch_size, 1, 8, 8, 8)
    global_cond = torch.randn(batch_size, 5)

    # Run with global_cond = None
    # Fix seed to ensure same noise generation in sampler
    torch.manual_seed(42)
    output_no_cond = model(x, global_cond=None)

    # Run with global_cond provided (reset seed to get exactly same noise)
    torch.manual_seed(42)
    output_with_cond = model(x, global_cond=global_cond)

    # Outputs should be identical since global_cond is meant to be ignored
    assert torch.allclose(output_no_cond, output_with_cond), (
        "Output changed despite global_cond being disabled"
    )


def test_temporal_vit_uses_precomputed_modulation_without_embedding_params():
    backbone = TemporalViTBackbone(
        in_channels=4,
        out_channels=4,
        cond_channels=0,
        n_steps_output=1,
        n_steps_input=1,
        global_cond_channels=None,
        include_global_cond=False,
        mod_features=32,
        hid_channels=32,
        hid_blocks=1,
        attention_heads=4,
        patch_size=4,
        spatial=2,
        temporal_method="none",
        use_precomputed_modulation=True,
    )

    # In precomputed modulation mode, no trainable time embedding parameters
    # should be registered, which avoids strict-DDP unused-parameter failures.
    assert all(
        not name.startswith("time_embedding") for name, _ in backbone.named_parameters()
    )

    x = torch.randn(2, 1, 8, 8, 4)
    t = torch.randn(2, 32)
    y = backbone(x_t=x, t=t, cond=None, global_cond=None)

    assert y.shape == (2, 1, 8, 8, 4)
