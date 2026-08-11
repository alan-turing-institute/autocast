from __future__ import annotations

import pytest
import torch
from torch import nn

from autocast.callbacks.residual_statistics import ResidualStatisticsCallback
from autocast.decoders.identity import IdentityDecoder
from autocast.encoders.identity import IdentityEncoder
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.models.processor import ProcessorModel
from autocast.nn.noise.noise_injector import AdditiveNoiseInjector
from autocast.nn.noise.source import ZeroSource
from autocast.processors.residual_flow_matching import ResidualFlowMatchingProcessor
from autocast.processors.residual_normalization import (
    ResidualGranularity,
    ResidualStandardizer,
)
from autocast.processors.residual_reference import (
    LastFrameReference,
    ReferenceTrajectory,
)
from autocast.types import Batch, EncodedBatch


class _ZeroField(nn.Module):
    def forward(self, z, t, cond, global_cond=None):  # noqa: ARG002
        return torch.zeros_like(z)


class _SingleProcessStrategy:
    def __init__(self) -> None:
        self.broadcasts: list[object] = []

    def broadcast(self, value, src=0):  # noqa: ARG002
        self.broadcasts.append(value)
        return value


class _DataModule:
    def __init__(self, train_batches: list[Batch] | list[EncodedBatch]) -> None:
        self.train_batches = train_batches
        self.train_calls = 0
        self.val_calls = 0

    def train_dataloader(self):
        self.train_calls += 1
        return self.train_batches

    def val_dataloader(self):
        self.val_calls += 1
        return []


class _Trainer:
    def __init__(self, datamodule: _DataModule) -> None:
        self.datamodule = datamodule
        self.strategy = _SingleProcessStrategy()
        self.is_global_zero = True


class _TrainableReference(ReferenceTrajectory):
    def __init__(self, n_steps_output: int) -> None:
        super().__init__()
        self.n_steps_output = n_steps_output
        self.offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, global_cond=None):  # noqa: ARG002
        return x[:, -1:].expand(-1, self.n_steps_output, *([-1] * (x.ndim - 2)))


def _processor(
    *,
    granularity: ResidualGranularity = "lead_channel",
    reference: ReferenceTrajectory | None = None,
    with_standardizer: bool = True,
) -> ResidualFlowMatchingProcessor:
    n_steps_output = 3
    standardizer = (
        ResidualStandardizer(
            n_steps_output=n_steps_output,
            n_channels=2,
            granularity=granularity,
        )
        if with_standardizer
        else None
    )
    return ResidualFlowMatchingProcessor(
        backbone=_ZeroField(),
        reference=reference
        if reference is not None
        else LastFrameReference(n_steps_output=n_steps_output),
        source=ZeroSource(),
        standardizer=standardizer,
        n_steps_output=n_steps_output,
        n_channels_out=2,
    )


def _encoded_batches() -> list[EncodedBatch]:
    residuals = torch.tensor(
        [
            [
                [[[1.0, 2.0], [3.0, 4.0]]],
                [[[2.0, 5.0], [4.0, 7.0]]],
                [[[6.0, 9.0], [8.0, 11.0]]],
            ],
            [
                [[[5.0, 6.0], [7.0, 8.0]]],
                [[[6.0, 9.0], [8.0, 11.0]]],
                [[[10.0, 13.0], [12.0, 15.0]]],
            ],
        ]
    )
    inputs = torch.zeros(2, 1, 1, 2, 2)
    batches = []
    for index in range(2):
        batches.append(
            EncodedBatch(
                encoded_inputs=inputs[index : index + 1],
                encoded_output_fields=residuals[index : index + 1],
                global_cond=None,
                encoded_info={},
            )
        )
    return batches


def _fit(
    callback: ResidualStatisticsCallback,
    trainer: _Trainer,
    model: nn.Module,
) -> None:
    callback.on_fit_start(trainer, model)  # type: ignore[arg-type]


@pytest.mark.parametrize("granularity", ["channel", "lead_channel"])
def test_fits_streaming_training_statistics(granularity: ResidualGranularity):
    batches = _encoded_batches()
    datamodule = _DataModule(batches)
    trainer = _Trainer(datamodule)
    processor = _processor(granularity=granularity)
    model = ProcessorModel(processor=processor)
    callback = ResidualStatisticsCallback()
    rng_state = torch.random.get_rng_state()

    _fit(callback, trainer, model)

    residual = torch.cat(
        [batch.encoded_output_fields for batch in batches],
        dim=0,
    )
    if granularity == "channel":
        values = residual.reshape(-1, residual.shape[-1])
    else:
        values = residual.movedim(1, 0).reshape(
            residual.shape[1],
            -1,
            residual.shape[-1],
        )
    expected_scale, expected_mean = torch.std_mean(values, dim=-2, correction=0)

    assert processor.standardizer is not None
    assert processor.standardizer.fitted
    assert torch.allclose(processor.standardizer.mean, expected_mean)
    assert torch.allclose(processor.standardizer.scale, expected_scale)
    assert datamodule.train_calls == 1
    assert datamodule.val_calls == 0
    assert len(trainer.strategy.broadcasts) == 3
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_skips_training_loader_when_statistics_are_restored():
    datamodule = _DataModule(_encoded_batches())
    trainer = _Trainer(datamodule)
    processor = _processor()
    assert processor.standardizer is not None
    processor.standardizer.set_statistics(mean=2.0, scale=3.0)

    _fit(ResidualStatisticsCallback(), trainer, ProcessorModel(processor=processor))

    assert datamodule.train_calls == 0
    assert torch.equal(processor.standardizer.mean, torch.full((3, 2), 2.0))
    assert torch.equal(processor.standardizer.scale, torch.full((3, 2), 3.0))
    assert len(trainer.strategy.broadcasts) == 1


def test_fits_after_frozen_encoder_for_raw_batches():
    encoded_batches = _encoded_batches()
    raw_batches = [
        Batch(
            input_fields=batch.encoded_inputs,
            output_fields=batch.encoded_output_fields,
            constant_scalars=None,
            constant_fields=None,
        )
        for batch in encoded_batches
    ]
    datamodule = _DataModule(raw_batches)
    trainer = _Trainer(datamodule)
    processor = _processor()
    encoder_decoder = EncoderDecoder(
        encoder=IdentityEncoder(in_channels=2),
        decoder=IdentityDecoder(in_channels=2),
    )
    model = EncoderProcessorDecoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        train_in_latent_space=True,
    )

    _fit(ResidualStatisticsCallback(), trainer, model)

    residual = torch.cat([batch.output_fields for batch in raw_batches], dim=0)
    values = residual.movedim(1, 0).reshape(3, -1, 2)
    expected_scale, expected_mean = torch.std_mean(values, dim=1, correction=0)
    assert processor.standardizer is not None
    assert torch.allclose(processor.standardizer.mean, expected_mean)
    assert torch.allclose(processor.standardizer.scale, expected_scale)


def test_rejects_trainable_reference():
    processor = _processor(reference=_TrainableReference(n_steps_output=3))
    trainer = _Trainer(_DataModule(_encoded_batches()))

    with pytest.raises(ValueError, match="fixed, non-trainable reference"):
        _fit(ResidualStatisticsCallback(), trainer, ProcessorModel(processor=processor))


def test_rejects_input_noise_until_its_statistics_are_defined():
    processor = _processor()
    model = ProcessorModel(
        processor=processor,
        noise_injector=AdditiveNoiseInjector(std=0.1),
    )
    trainer = _Trainer(_DataModule(_encoded_batches()))

    with pytest.raises(ValueError, match="do not yet support model input noise"):
        _fit(ResidualStatisticsCallback(), trainer, model)


def test_requires_configured_standardizer():
    processor = _processor(with_standardizer=False)
    trainer = _Trainer(_DataModule(_encoded_batches()))

    with pytest.raises(TypeError, match="processor.standardizer"):
        _fit(ResidualStatisticsCallback(), trainer, ProcessorModel(processor=processor))
