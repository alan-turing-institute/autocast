import lightning as L
import pytest
import torch
from conftest import get_optimizer_config
from torch import nn

from autocast.models.processor import ProcessorModel
from autocast.processors.azula_vit import (
    AzulaViTProcessor,
    MCDropoutAzulaViTProcessor,
)
from autocast.processors.vit import AViTProcessor
from autocast.types import EncodedBatch


def test_vit_processor(encoded_batch, encoded_dummy_loader):
    input_channels = encoded_batch.encoded_inputs.shape[1]
    output_channels = encoded_batch.encoded_output_fields.shape[1]
    assert (
        encoded_batch.encoded_inputs.shape[2:]
        == encoded_batch.encoded_output_fields.shape[2:]
    )
    spatial_resolution = tuple(encoded_batch.encoded_output_fields.shape[2:])

    processor = AViTProcessor(
        in_channels=input_channels,
        out_channels=output_channels,
        spatial_resolution=spatial_resolution,
    )
    model = ProcessorModel(
        processor=processor,
        optimizer_config=get_optimizer_config(),
    )

    output = model.map(encoded_batch.encoded_inputs, encoded_batch.global_cond)
    assert output.shape == encoded_batch.encoded_output_fields.shape

    train_loss = model.training_step(encoded_batch, 0)
    assert train_loss.shape == ()
    train_loss.backward()

    # Run a full training loop
    L.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
        enable_model_summary=False,
        accelerator="cpu",
    ).fit(
        model,
        train_dataloaders=encoded_dummy_loader,
        val_dataloaders=encoded_dummy_loader,
    )


def test_azula_vit_processor_5d_multistep():
    """Cached-latent path: T_in=1, T_out=4. Processor folds T into C internally.

    Regression test for CRPS-in-latent: AzulaViTProcessor must produce a 5D
    output with shape (B, n_steps_output, H, W, C) when given a 5D input,
    mirroring what ``PermuteConcat + ChannelsLast`` achieves in ambient mode.
    """
    b, t_in, t_out, h, w, c = 2, 1, 4, 8, 8, 4
    processor = AzulaViTProcessor(
        in_channels=c,
        out_channels=c,
        spatial_resolution=(h, w),
        hidden_dim=64,
        num_heads=4,
        n_layers=2,
        patch_size=1,
        temporal_method="none",
        n_noise_channels=32,
        n_steps_input=t_in,
        n_steps_output=t_out,
    )
    x = torch.randn(b, t_in, h, w, c)
    targets = torch.randn(b, t_out, h, w, c)
    batch = EncodedBatch(
        encoded_inputs=x,
        encoded_output_fields=targets,
        global_cond=None,
        encoded_info={},
    )

    pred = processor.map(x, global_cond=None)
    assert pred.shape == targets.shape

    model = ProcessorModel(processor=processor, optimizer_config=get_optimizer_config())
    train_loss = model.training_step(batch, 0)
    assert train_loss.shape == ()
    train_loss.backward()


def test_azula_vit_processor_checkpointing(encoded_batch):
    input_channels = encoded_batch.encoded_inputs.shape[1]
    output_channels = encoded_batch.encoded_output_fields.shape[1]
    spatial_resolution = tuple(encoded_batch.encoded_output_fields.shape[2:])

    processor = AzulaViTProcessor(
        in_channels=input_channels,
        out_channels=output_channels,
        spatial_resolution=spatial_resolution,
        hidden_dim=64,
        num_heads=4,
        n_layers=2,
        patch_size=4,
        n_noise_channels=32,
        checkpointing=True,
    )
    model = ProcessorModel(
        processor=processor,
        optimizer_config=get_optimizer_config(),
    )

    train_loss = model.training_step(encoded_batch, 0)
    assert train_loss.shape == ()
    train_loss.backward()


def _make_mc_dropout_processor(*, dropout: float = 0.5) -> MCDropoutAzulaViTProcessor:
    return MCDropoutAzulaViTProcessor(
        in_channels=4,
        out_channels=4,
        spatial_resolution=(8, 8),
        hidden_dim=64,
        num_heads=4,
        n_layers=2,
        patch_size=1,
        temporal_method="none",
        dropout=dropout,
    )


def test_mc_dropout_azula_vit_uses_dropout_without_noise_modulation():
    processor = _make_mc_dropout_processor(dropout=0.2)

    assert processor.n_noise_channels is None
    assert processor.n_noise_input_channels is None
    assert processor.modulation_proj is None

    dropout_modules = [
        module for module in processor.modules() if isinstance(module, nn.Dropout)
    ]

    assert len(dropout_modules) == 2
    assert all(module.p == 0.2 for module in dropout_modules)


def test_mc_dropout_azula_vit_remains_stochastic_during_inference():
    processor = _make_mc_dropout_processor()
    processor.eval()
    x = torch.randn(2, 1, 8, 8, 4)

    first = processor.map(x)
    second = processor.map(x)

    assert not torch.equal(first, second)
    assert processor.training is False
    assert all(
        not module.training
        for module in processor.modules()
        if isinstance(module, nn.Dropout)
    )


def test_standard_azula_vit_disables_dropout_during_inference():
    processor = AzulaViTProcessor(
        in_channels=4,
        out_channels=4,
        spatial_resolution=(8, 8),
        hidden_dim=64,
        num_heads=4,
        n_layers=2,
        patch_size=1,
        temporal_method="none",
        n_noise_channels=None,
        dropout=0.5,
    )
    processor.eval()
    x = torch.randn(2, 1, 8, 8, 4)

    first = processor.map(x)
    second = processor.map(x)

    assert torch.equal(first, second)


def test_mc_dropout_azula_vit_rejects_noise_modulation():
    with pytest.raises(ValueError, match="dropout instead of stochastic modulation"):
        MCDropoutAzulaViTProcessor(
            in_channels=4,
            out_channels=4,
            spatial_resolution=(8, 8),
            n_noise_channels=32,
        )
