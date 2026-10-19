from __future__ import annotations

import lightning as L
import pytest
import torch
from einops import rearrange, repeat
from lightning.pytorch.plugins.precision import Precision
from torch import nn
from torch.utils.data import DataLoader

from autocast.callbacks.residual_statistics import ResidualStatisticsCallback
from autocast.decoders.identity import IdentityDecoder
from autocast.encoders.base import EncoderWithCond
from autocast.encoders.identity import IdentityEncoder
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.models.processor import ProcessorModel
from autocast.nn.noise.noise_injector import AdditiveNoiseInjector
from autocast.nn.noise.source import ZeroSource
from autocast.processors.base import Processor
from autocast.processors.residual_flow_matching import ResidualFlowMatchingProcessor
from autocast.processors.residual_normalization import (
    ResidualGranularity,
    ResidualStandardizer,
)
from autocast.processors.residual_reference import (
    LastFrameReference,
    ProcessorReference,
    ReferenceTrajectory,
)
from autocast.types import Batch, EncodedBatch
from autocast.utils import get_optimizer_config


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
        self.precision_plugin = Precision()


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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available.",
)
def test_fits_streaming_training_statistics_on_mps():
    batches = _encoded_batches()
    processor = _processor().to("mps")
    model = ProcessorModel(processor=processor)

    _fit(
        ResidualStatisticsCallback(),
        _Trainer(_DataModule(batches)),
        model,
    )

    residual = torch.cat(
        [batch.encoded_output_fields for batch in batches],
        dim=0,
    )
    values = residual.movedim(1, 0).reshape(3, -1, 2)
    expected_scale, expected_mean = torch.std_mean(values, dim=1, correction=0)
    assert processor.standardizer is not None
    assert processor.standardizer.mean.device.type == "mps"
    assert torch.allclose(processor.standardizer.mean.cpu(), expected_mean)
    assert torch.allclose(processor.standardizer.scale.cpu(), expected_scale)


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

    with pytest.raises(TypeError, match=r"processor\.standardizer"):
        _fit(ResidualStatisticsCallback(), trainer, ProcessorModel(processor=processor))


class _PrecisionEncoder(EncoderWithCond):
    latent_channels = 2
    channel_axis = -1

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)
        self.output_dtypes = []
        with torch.no_grad():
            self.conv.weight.copy_(torch.eye(2)[:, :, None, None])
            assert self.conv.bias is not None
            self.conv.bias.fill_(0.25)

    def encode(self, batch):
        fields = batch.input_fields
        encoded = self.conv(rearrange(fields, "b t h w c -> (b t) c h w"))
        self.output_dtypes.append(encoded.dtype)
        return rearrange(encoded, "(b t) c h w -> b t h w c", b=fields.shape[0])


class _PrecisionReference(Processor[EncodedBatch]):
    def __init__(self):
        super().__init__()
        self.encoder = _PrecisionEncoder()

    def map(self, x, global_cond):  # noqa: ARG002
        fields = self.encoder.encode(Batch(x, x, None, None))
        return repeat(fields[:, -1:], "b 1 ... c -> b t ... c", t=3)

    def loss(self, batch):
        raise NotImplementedError


class _TrainableField(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.125))

    def forward(self, z, t, cond, global_cond=None):  # noqa: ARG002
        return torch.zeros_like(z) + self.offset


class _PrecisionDataModule(L.LightningDataModule):
    def __init__(self, batches):
        super().__init__()
        self.batches = batches

    def train_dataloader(self):
        return DataLoader(self.batches, batch_size=None, num_workers=0)


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("32-true", torch.float32),
        ("16-true", torch.float16),
        ("bf16-true", torch.bfloat16),
        ("bf16-mixed", torch.bfloat16),
    ],
)
@pytest.mark.parametrize("processor_reference", [False, True])
def test_fitting_uses_training_precision(
    precision, expected_dtype, processor_reference
):
    batches = [
        Batch(batch.encoded_inputs, batch.encoded_output_fields, None, None)
        for batch in _encoded_batches()
    ]
    encoder = _PrecisionEncoder()
    reference_processor = _PrecisionReference() if processor_reference else None
    processor = _processor(
        reference=ProcessorReference(reference_processor)
        if reference_processor is not None
        else None
    )
    processor.flow_matching_model = _TrainableField()
    model = EncoderProcessorDecoder(
        encoder_decoder=EncoderDecoder(encoder, IdentityDecoder(in_channels=2)),
        processor=processor,
        train_in_latent_space=True,
        optimizer_config=get_optimizer_config(optimizer="sgd"),
        train_metrics=None,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        precision=precision,
        max_steps=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=[ResidualStatisticsCallback()],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=_PrecisionDataModule(batches))

    residual = torch.cat([batch.output_fields for batch in batches])
    if processor_reference:
        residual = residual - 0.25
    expected_scale, expected_mean = torch.std_mean(
        rearrange(residual, "b t h w c -> t (b h w) c"), dim=1, correction=0
    )
    standardizer = processor.standardizer
    assert standardizer is not None
    assert standardizer.fitted
    assert set(encoder.output_dtypes) == {expected_dtype}
    if reference_processor is not None:
        assert set(reference_processor.encoder.output_dtypes) == {expected_dtype}
    torch.testing.assert_close(standardizer.mean, expected_mean.to(standardizer.mean))
    torch.testing.assert_close(
        standardizer.scale, expected_scale.to(standardizer.scale)
    )
    assert all(batch.input_fields.dtype == torch.float32 for batch in batches)
    assert trainer.global_step == 1
