import math
from unittest.mock import MagicMock

import pytest
from torch import nn

from autocast.callbacks.loss_weight_schedule import (
    ConstantSchedule,
    CosineSchedule,
    LinearRampSchedule,
    LossWeightScheduleCallback,
)
from autocast.decoders.channels_last import ChannelsLast
from autocast.encoders.permute_concat import PermuteConcat
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor

# -- Schedule unit tests ----------------------------------------------------


class TestLinearRampSchedule:
    def test_before_start(self):
        s = LinearRampSchedule(0.0, 1.0, 0.2, 0.8)
        assert s(0.0) == 0.0
        assert s(0.1) == 0.0

    def test_after_end(self):
        s = LinearRampSchedule(0.0, 1.0, 0.2, 0.8)
        assert s(0.8) == 1.0
        assert s(1.0) == 1.0

    def test_midpoint(self):
        s = LinearRampSchedule(0.0, 1.0, 0.0, 1.0)
        assert s(0.5) == pytest.approx(0.5)

    def test_custom_range(self):
        s = LinearRampSchedule(2.0, 4.0, 0.25, 0.75)
        assert s(0.5) == pytest.approx(3.0)


class TestCosineSchedule:
    def test_endpoints(self):
        s = CosineSchedule(0.0, 1.0)
        assert s(0.0) == pytest.approx(0.0)
        assert s(1.0) == pytest.approx(1.0)

    def test_midpoint(self):
        s = CosineSchedule(0.0, 1.0)
        assert s(0.5) == pytest.approx(0.5)

    def test_quarter(self):
        s = CosineSchedule(0.0, 1.0)
        expected = 0.5 * (1.0 - math.cos(math.pi * 0.25))
        assert s(0.25) == pytest.approx(expected)


class TestConstantSchedule:
    def test_constant(self):
        s = ConstantSchedule(0.42)
        assert s(0.0) == 0.42
        assert s(0.5) == 0.42
        assert s(1.0) == 0.42


# -- Callback tests ---------------------------------------------------------


class SimpleProcessor(Processor):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 1)

    def map(
        self,
        x: Tensor,
        global_cond: Tensor | None = None,  # noqa: ARG002
    ) -> Tensor:
        return self.conv(x)

    def loss(self, batch: EncodedBatch) -> Tensor:
        preds = self.conv(batch.encoded_inputs)
        return nn.functional.mse_loss(preds, batch.encoded_output_fields)


def _make_model(
    ambient_loss_weight: float = 1.0,
    latent_loss_weight: float = 0.0,
    freeze_encoder_decoder: bool = False,
) -> EncoderProcessorDecoder:
    channels, t_steps = 2, 2
    encoder = PermuteConcat(
        in_channels=channels, n_steps_input=t_steps, with_constants=False
    )
    merged = channels * t_steps
    decoder = ChannelsLast(output_channels=channels, time_steps=t_steps)
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder)
    processor = SimpleProcessor(in_channels=merged)
    return EncoderProcessorDecoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        loss_func=nn.MSELoss(),
        ambient_loss_weight=ambient_loss_weight,
        latent_loss_weight=latent_loss_weight,
        freeze_encoder_decoder=freeze_encoder_decoder,
    )


def _make_trainer_mock(
    global_step: int = 50,
    estimated_stepping_batches: int = 100,
    max_epochs: int = 10,
    current_epoch: int = 5,
) -> MagicMock:
    trainer = MagicMock(
        spec=[
            "global_step",
            "estimated_stepping_batches",
            "max_epochs",
            "current_epoch",
        ]
    )
    trainer.global_step = global_step
    trainer.estimated_stepping_batches = estimated_stepping_batches
    trainer.max_epochs = max_epochs
    trainer.current_epoch = current_epoch
    return trainer


def test_callback_mutates_ambient_weight():
    model = _make_model(ambient_loss_weight=0.0, latent_loss_weight=1.0)
    callback = LossWeightScheduleCallback(
        ambient_schedule=LinearRampSchedule(0.0, 1.0),
        schedule_by="step",
    )
    trainer = _make_trainer_mock(global_step=50, estimated_stepping_batches=100)

    callback.on_train_batch_start(trainer, model, batch=None, batch_idx=0)

    assert model.ambient_loss_weight == pytest.approx(0.5)


def test_callback_mutates_latent_weight():
    model = _make_model(ambient_loss_weight=1.0, latent_loss_weight=0.0)
    callback = LossWeightScheduleCallback(
        latent_schedule=LinearRampSchedule(0.0, 0.5),
        schedule_by="step",
    )
    trainer = _make_trainer_mock(global_step=100, estimated_stepping_batches=100)

    callback.on_train_batch_start(trainer, model, batch=None, batch_idx=0)

    assert model.latent_loss_weight == pytest.approx(0.5)


def test_callback_epoch_mode():
    model = _make_model(ambient_loss_weight=0.0, latent_loss_weight=1.0)
    callback = LossWeightScheduleCallback(
        ambient_schedule=ConstantSchedule(0.3),
        schedule_by="epoch",
    )
    trainer = _make_trainer_mock(current_epoch=5, max_epochs=10)

    callback.on_train_batch_start(trainer, model, batch=None, batch_idx=0)

    assert model.ambient_loss_weight == pytest.approx(0.3)


def test_callback_unfreezes_encoder_decoder():
    model = _make_model(
        ambient_loss_weight=0.0,
        latent_loss_weight=1.0,
    )
    # Manually freeze encoder_decoder to simulate pure-latent init
    for p in model.encoder_decoder.parameters():
        p.requires_grad = False

    callback = LossWeightScheduleCallback(
        ambient_schedule=LinearRampSchedule(0.0, 1.0),
        schedule_by="step",
    )
    trainer = _make_trainer_mock(global_step=50, estimated_stepping_batches=100)

    callback.on_train_batch_start(trainer, model, batch=None, batch_idx=0)

    # Encoder/decoder should be unfrozen now
    assert all(p.requires_grad for p in model.encoder_decoder.parameters())
    assert callback._encoder_decoder_unfrozen is True


def test_callback_does_not_unfreeze_when_explicitly_frozen():
    model = _make_model(
        ambient_loss_weight=0.0,
        latent_loss_weight=1.0,
        freeze_encoder_decoder=True,
    )

    callback = LossWeightScheduleCallback(
        ambient_schedule=LinearRampSchedule(0.0, 1.0),
        schedule_by="step",
    )
    trainer = _make_trainer_mock(global_step=50, estimated_stepping_batches=100)

    callback.on_train_batch_start(trainer, model, batch=None, batch_idx=0)

    # Should remain frozen because freeze_encoder_decoder=True
    assert all(not p.requires_grad for p in model.encoder_decoder.parameters())
    assert callback._encoder_decoder_unfrozen is False


def test_callback_no_schedule_leaves_weights_unchanged():
    model = _make_model(ambient_loss_weight=0.7, latent_loss_weight=0.3)
    callback = LossWeightScheduleCallback()
    trainer = _make_trainer_mock()

    callback.on_train_batch_start(trainer, model, batch=None, batch_idx=0)

    assert model.ambient_loss_weight == pytest.approx(0.7)
    assert model.latent_loss_weight == pytest.approx(0.3)


def test_callback_state_dict_roundtrip():
    callback = LossWeightScheduleCallback()
    callback._encoder_decoder_unfrozen = True

    state = callback.state_dict()
    new_callback = LossWeightScheduleCallback()
    new_callback.load_state_dict(state)

    assert new_callback._encoder_decoder_unfrozen is True
