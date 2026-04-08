import typing

import lightning as L
import pytest
import torch
from conftest import CondCaptureProcessor, get_optimizer_config
from torch import nn

from autocast.decoders.channels_last import ChannelsLast
from autocast.encoders.permute_concat import PermuteConcat
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.nn.noise.noise_injector import (
    AdditiveNoiseInjector,
    ConcatenatedNoiseInjector,
)
from autocast.processors.base import Processor
from autocast.types import Batch, EncodedBatch, Tensor


class TinyProcessor(Processor[EncodedBatch]):
    def __init__(self, in_channels: int = 1, out_channels: int | None = None) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.loss_func = nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:  # noqa: ARG002
        return self(x)

    def loss(self, batch: EncodedBatch) -> Tensor:
        outputs = self(batch.encoded_inputs)
        return self.loss_func(outputs, batch.encoded_output_fields)


def test_on_load_checkpoint_removes_metadata_from_state_dict():
    checkpoint = {
        "state_dict": {"_metadata": {"x": 1}, "layer.weight": torch.tensor(1)}
    }
    EncoderProcessorDecoder.on_load_checkpoint(
        typing.cast(EncoderProcessorDecoder, object()), checkpoint
    )
    assert "_metadata" not in checkpoint["state_dict"]


def test_on_load_checkpoint_handles_missing_state_dict_key():
    checkpoint = {"_metadata": {"x": 1}, "layer.weight": torch.tensor(1)}
    EncoderProcessorDecoder.on_load_checkpoint(
        typing.cast(EncoderProcessorDecoder, object()), checkpoint
    )
    assert "_metadata" not in checkpoint


def test_encoder_processor_decoder_training_step_runs(make_toy_batch, dummy_loader):
    batch = make_toy_batch()
    output_channels = batch.output_fields.shape[-1]
    time_steps = batch.output_fields.shape[1]
    # Encoder merges C*T into single dimension
    merged_channels = output_channels * time_steps

    encoder = PermuteConcat(
        in_channels=output_channels, n_steps_input=time_steps, with_constants=False
    )
    decoder = ChannelsLast(output_channels=output_channels, time_steps=time_steps)
    loss = nn.MSELoss()
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder, loss_func=loss)

    processor = TinyProcessor(in_channels=merged_channels)
    model = EncoderProcessorDecoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        loss_func=loss,
        optimizer_config=get_optimizer_config(),
    )

    train_loss = model.training_step(batch, 0)

    assert train_loss.shape == ()
    train_loss.backward()

    L.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
        accelerator="cpu",
    ).fit(model, train_dataloaders=dummy_loader, val_dataloaders=dummy_loader)


def test_global_cond_passes_from_encoder_to_processor():
    batch_size = 2
    t_steps = 3
    w = 8
    h = 8
    channels = 2
    cond_dim = 4

    input_fields = torch.randn(batch_size, t_steps, w, h, channels)
    output_fields = torch.randn(batch_size, t_steps, w, h, channels)
    constant_scalars = torch.randn(batch_size, cond_dim)

    batch = Batch(
        input_fields=input_fields,
        output_fields=output_fields,
        constant_scalars=constant_scalars,
        constant_fields=None,
    )

    encoder = PermuteConcat(
        in_channels=channels, n_steps_input=t_steps, with_constants=False
    )
    decoder = ChannelsLast(output_channels=channels, time_steps=t_steps)
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder)

    processor = CondCaptureProcessor()
    model = EncoderProcessorDecoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        optimizer_config=get_optimizer_config(),
    )

    _ = model(batch)

    assert processor.last_global_cond is not None
    assert torch.allclose(processor.last_global_cond, constant_scalars)


@pytest.mark.parametrize(
    ("n_steps_input", "n_steps_output", "stride"),
    [
        (1, 4, 4),  # More output than input (n_steps_output >= n_steps_input)
        (2, 2, 2),  # Equal input/output, stride matches
        (4, 4, 4),  # Larger equal input/output
        (3, 3, 3),  # Odd number of steps
        (1, 1, 1),  # Single step
        (4, 2, 2),  # More input than output (n_steps_output <= n_steps_input)
        (6, 3, 3),  # Double input (n_steps_output <= n_steps_input)
    ],
)
def test_encoder_processor_decoder_rollout_handles_batches(
    make_toy_batch, n_steps_input, n_steps_output, stride
):
    """Test EncoderProcessorDecoder rollout with sufficient trajectory data."""
    batch_size = 10
    max_rollout_steps = 4
    trajectory_length = 100

    batch = make_toy_batch(
        batch_size=batch_size,
        t_in=n_steps_input,
        t_out=trajectory_length - n_steps_input,
    )

    output_channels = batch.output_fields.shape[-1]
    merged_input_channels = output_channels * n_steps_input
    merged_output_channels = output_channels * n_steps_output

    encoder = PermuteConcat(
        in_channels=output_channels, n_steps_input=n_steps_input, with_constants=False
    )
    decoder = ChannelsLast(output_channels=output_channels, time_steps=n_steps_output)
    loss = nn.MSELoss()
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder, loss_func=loss)
    processor = TinyProcessor(
        in_channels=merged_input_channels, out_channels=merged_output_channels
    )
    model = EncoderProcessorDecoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        loss_func=loss,
        optimizer_config=get_optimizer_config(),
        stride=stride,
        max_rollout_steps=max_rollout_steps,
    )

    preds, gts = model.rollout(
        batch, stride=stride, max_rollout_steps=max_rollout_steps, return_windows=True
    )

    assert preds.shape == (batch_size, max_rollout_steps, n_steps_output, 32, 32, 1)
    assert gts is not None
    assert gts.shape == preds.shape

    preds, gts = model.rollout(
        batch,
        stride=n_steps_output,
        max_rollout_steps=max_rollout_steps,
        return_windows=False,
    )

    assert preds.shape == (batch_size, max_rollout_steps * n_steps_output, 32, 32, 1)
    assert gts is not None
    assert gts.shape == preds.shape


@pytest.mark.parametrize(
    ("n_steps_input", "n_steps_output", "stride", "expected_gt_windows"),
    [
        (2, 2, 2, 3),  # 6 output steps / 2 stride = 3 windows
        (4, 4, 4, 1),  # 6 output steps / 4 stride = 1 full window (+ partial)
        (3, 3, 3, 2),  # 6 output steps / 3 stride = 2 windows
        (1, 1, 1, 6),  # 6 output steps / 1 stride = 6 windows
        (4, 2, 2, 3),  # Different input/output: 6 / 2 stride = 3 windows
        (6, 3, 3, 2),  # Double input: 6 / 3 stride = 2 windows
    ],
)
def test_encoder_processor_decoder_rollout_handles_short_trajectory(
    make_toy_batch, n_steps_input, n_steps_output, stride, expected_gt_windows
):
    """Test EncoderProcessorDecoder rollout when trajectory is shorter than needed.

    In free-running mode the model continues predicting using its own outputs even after
    ground truth data runs out.
    """
    batch_size = 10
    max_rollout_steps = 10
    # Short trajectory: only 6 time steps available for output
    trajectory_length = n_steps_input + 6

    batch = make_toy_batch(
        batch_size=batch_size,
        t_in=n_steps_input,
        t_out=trajectory_length - n_steps_input,
    )

    output_channels = batch.output_fields.shape[-1]
    merged_input_channels = output_channels * n_steps_input
    merged_output_channels = output_channels * n_steps_output

    encoder = PermuteConcat(
        in_channels=output_channels, n_steps_input=n_steps_input, with_constants=False
    )
    decoder = ChannelsLast(output_channels=output_channels, time_steps=n_steps_output)
    loss = nn.MSELoss()
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder, loss_func=loss)
    processor = TinyProcessor(
        in_channels=merged_input_channels, out_channels=merged_output_channels
    )
    model = EncoderProcessorDecoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        loss_func=loss,
        optimizer_config=get_optimizer_config(),
        stride=stride,
        max_rollout_steps=max_rollout_steps,
        teacher_forcing_ratio=0.0,
    )

    preds, gts = model.rollout(batch, stride=stride, return_windows=True)

    # In free-running mode, predictions continue for all max_rollout_steps
    assert preds.shape == (batch_size, max_rollout_steps, n_steps_output, 32, 32, 1)

    # Ground truth only available for windows where data exists
    assert gts is not None
    assert gts.shape == (batch_size, expected_gt_windows, n_steps_output, 32, 32, 1)

    preds, gts = model.rollout(batch, stride=n_steps_output, return_windows=False)

    # Predictions for all rollout windows concatenated
    assert preds.shape == (batch_size, max_rollout_steps * n_steps_output, 32, 32, 1)
    # Ground truth only for windows where data was available
    assert gts is not None
    assert gts.shape == (batch_size, expected_gt_windows * n_steps_output, 32, 32, 1)


def _make_epd(channels=2, t_steps=2, latent_noise_injector=None, extra_in=0):
    """Helper to build a minimal EPD with optional latent noise."""
    encoder = PermuteConcat(
        in_channels=channels, n_steps_input=t_steps, with_constants=False
    )
    merged = channels * t_steps
    decoder = ChannelsLast(output_channels=channels, time_steps=t_steps)
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder)
    processor = TinyProcessor(in_channels=merged + extra_in, out_channels=merged)
    return EncoderProcessorDecoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        loss_func=nn.MSELoss(),
        latent_noise_injector=latent_noise_injector,
        latent_loss_weight=1.0,
    )


def test_latent_noise_injector_additive_forward_and_loss():
    """Additive latent noise runs through both forward and _latent_loss."""
    model = _make_epd(latent_noise_injector=AdditiveNoiseInjector(std=0.1))
    batch = Batch(
        input_fields=torch.randn(2, 2, 8, 8, 2),
        output_fields=torch.randn(2, 2, 8, 8, 2),
        constant_scalars=None,
        constant_fields=None,
    )
    out = model(batch)
    assert out.shape == batch.output_fields.shape

    loss, _ = model.loss(batch)
    assert loss.ndim == 0
    loss.backward()


def test_latent_noise_injector_concat_apply_latent_noise():
    """ConcatenatedNoiseInjector adds channels to encoded_inputs in EncodedBatch.

    Note: ConcatenatedNoiseInjector appends on the last dimension (channels-last).
    This test verifies the _apply_latent_noise method directly.
    """
    n_extra = 3
    injector = ConcatenatedNoiseInjector(n_channels=n_extra, std=0.5)
    model = _make_epd(latent_noise_injector=injector)

    encoded_batch = EncodedBatch(
        encoded_inputs=torch.randn(2, 8, 8, 4),
        encoded_output_fields=torch.randn(2, 8, 8, 4),
        global_cond=None,
        encoded_info={},
    )
    noisy = model._apply_latent_noise(encoded_batch)
    assert noisy.encoded_inputs.shape[-1] == 4 + n_extra
    assert (
        noisy.encoded_output_fields.shape == encoded_batch.encoded_output_fields.shape
    )


def test_latent_noise_injector_none_preserves_behavior():
    """No latent noise injector gives identical results to baseline."""
    torch.manual_seed(42)
    batch = Batch(
        input_fields=torch.randn(2, 2, 8, 8, 2),
        output_fields=torch.randn(2, 2, 8, 8, 2),
        constant_scalars=None,
        constant_fields=None,
    )
    model = _make_epd(latent_noise_injector=None)
    model.eval()

    with torch.no_grad():
        out1 = model(batch)
        out2 = model(batch)

    assert torch.allclose(out1, out2)
