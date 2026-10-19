from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import cast

import lightning as L
import pytest
import torch
from einops import rearrange, repeat
from lightning.pytorch.plugins.precision import Precision
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler

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
    ResidualNormalizer,
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

    def all_gather(self, value) -> torch.Tensor:  # noqa: ARG002
        pytest.fail("Invalid fitting configuration reached a collective")


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
    def __init__(self, datamodule: _DataModule | L.LightningDataModule) -> None:
        self.datamodule = datamodule
        self.strategy = _SingleProcessStrategy()
        self.is_global_zero = True
        self.precision_plugin = Precision()
        self.world_size = 1
        self.global_rank = 0


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
    with_residual_normalizer: bool = True,
) -> ResidualFlowMatchingProcessor:
    n_steps_output = 3
    residual_normalizer = (
        ResidualNormalizer(
            n_steps_output=n_steps_output,
            n_channels=2,
            granularity=granularity,
        )
        if with_residual_normalizer
        else None
    )
    return ResidualFlowMatchingProcessor(
        backbone=_ZeroField(),
        reference=reference
        if reference is not None
        else LastFrameReference(n_steps_output=n_steps_output),
        source=ZeroSource(),
        residual_normalizer=residual_normalizer,
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


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("granularity", ["channel", "lead_channel"])
def test_fits_streaming_training_statistics(
    granularity: ResidualGranularity, raw, device
):
    batches = _encoded_batches()
    datamodule = _DataModule(
        [
            Batch(batch.encoded_inputs, batch.encoded_output_fields, None, None)
            for batch in batches
        ]
        if raw
        else batches
    )
    trainer = _Trainer(datamodule)
    processor = _processor(granularity=granularity).to(device)
    model = (
        EncoderProcessorDecoder(
            encoder_decoder=EncoderDecoder(
                IdentityEncoder(in_channels=2), IdentityDecoder(in_channels=2)
            ),
            processor=processor,
            train_in_latent_space=True,
        )
        if raw
        else ProcessorModel(processor=processor)
    )
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

    assert processor.residual_normalizer is not None
    assert processor.residual_normalizer.fitted
    assert torch.allclose(processor.residual_normalizer.mean.cpu(), expected_mean)
    assert torch.allclose(processor.residual_normalizer.scale.cpu(), expected_scale)
    assert datamodule.train_calls == 1
    assert datamodule.val_calls == 0
    assert trainer.strategy.broadcasts == [True]
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_skips_training_loader_when_statistics_are_restored():
    datamodule = _DataModule(_encoded_batches())
    trainer = _Trainer(datamodule)
    processor = _processor()
    assert processor.residual_normalizer is not None
    processor.residual_normalizer.set_statistics(mean=2.0, scale=3.0)

    _fit(ResidualStatisticsCallback(), trainer, ProcessorModel(processor=processor))

    assert datamodule.train_calls == 0
    assert torch.equal(processor.residual_normalizer.mean, torch.full((3, 2), 2.0))
    assert torch.equal(processor.residual_normalizer.scale, torch.full((3, 2), 3.0))
    assert len(trainer.strategy.broadcasts) == 1


def test_rejects_trainable_reference():
    processor = _processor(reference=_TrainableReference(n_steps_output=3))
    trainer = _Trainer(_DataModule(_encoded_batches()))

    with pytest.raises(ValueError, match="fixed, non-trainable reference"):
        _fit(ResidualStatisticsCallback(), trainer, ProcessorModel(processor=processor))


@pytest.mark.parametrize("rank", [0, 1])
def test_rejects_input_noise_until_its_statistics_are_defined(rank):
    processor = _processor()
    model = ProcessorModel(
        processor=processor,
        noise_injector=AdditiveNoiseInjector(std=0.1),
    )
    trainer = _Trainer(_DataModule(_encoded_batches()))

    trainer.global_rank = rank
    trainer.is_global_zero = rank == 0
    trainer.world_size = 2

    with pytest.raises(ValueError, match="do not yet support model input noise"):
        _fit(ResidualStatisticsCallback(), trainer, model)

    assert trainer.strategy.broadcasts == []


def test_requires_configured_residual_normalizer():
    processor = _processor(with_residual_normalizer=False)
    trainer = _Trainer(_DataModule(_encoded_batches()))

    with pytest.raises(TypeError, match=r"processor\.residual_normalizer"):
        _fit(ResidualStatisticsCallback(), trainer, ProcessorModel(processor=processor))


@pytest.mark.parametrize("n_samples", [1, 5])
def test_distributed_fitting_combines_disjoint_shards_including_empty_rank(n_samples):
    batches = [
        EncodedBatch(
            torch.zeros(1, 1, 1, 2, 2),
            torch.full((1, 3, 1, 2, 2), float(index)),
            None,
            {},
        )
        for index in range(n_samples)
    ]
    barrier = Barrier(2)
    summaries: list[torch.Tensor | None] = [None, None]
    visited = [[], []]

    class Dataset:
        def __init__(self, rank):
            self.rank = rank

        def __len__(self):
            return len(batches)

        def __getitem__(self, index):
            visited[self.rank].append(index)
            return batches[index]

    class Strategy(_SingleProcessStrategy):
        def __init__(self, rank):
            super().__init__()
            self.rank = rank

        def broadcast(self, value, src=0):  # noqa: ARG002
            return True if isinstance(value, bool) else value

        def all_gather(self, value):
            summaries[self.rank] = value
            barrier.wait(timeout=10)
            return torch.stack(cast(list[torch.Tensor], summaries))

    def fit_rank(rank):
        processor = _processor()
        model = ProcessorModel(processor=processor)
        trainer = _Trainer(_PrecisionDataModule(Dataset(rank)))
        trainer.world_size = 2
        trainer.global_rank = rank
        trainer.is_global_zero = rank == 0
        trainer.strategy = Strategy(rank)
        _fit(ResidualStatisticsCallback(), trainer, model)
        return processor.residual_normalizer

    with torch.random.fork_rng(devices=[]), ThreadPoolExecutor(max_workers=2) as pool:
        normalizers = list(pool.map(fit_rank, range(2)))
    expected_scale, expected_mean = torch.std_mean(
        torch.cat([batch.encoded_output_fields for batch in batches]),
        dim=(0, 2, 3),
        correction=0,
    )
    for normalizer in normalizers:
        assert normalizer is not None
        torch.testing.assert_close(normalizer.mean, expected_mean)
        torch.testing.assert_close(normalizer.scale, expected_scale.clamp_min(1e-6))
    assert all(summary is not None for summary in summaries)
    assert visited == [list(range(0, n_samples, 2)), list(range(1, n_samples, 2))]


@pytest.mark.parametrize("subset", [False, True])
def test_distributed_fitting_rejects_nonstandard_sampling(subset, monkeypatch):
    batches = _encoded_batches()
    sampler = (
        SubsetRandomSampler([0]) if subset else RandomSampler(batches, replacement=True)
    )
    datamodule = _PrecisionDataModule(batches)
    monkeypatch.setattr(
        datamodule,
        "train_dataloader",
        lambda: DataLoader(cast(Dataset, batches), batch_size=None, sampler=sampler),
    )
    trainer = _Trainer(datamodule)
    trainer.world_size = 2
    with pytest.raises(ValueError, match="standard samplers"):
        _fit(
            ResidualStatisticsCallback(),
            trainer,
            ProcessorModel(processor=_processor()),
        )


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
    residual_normalizer = processor.residual_normalizer
    assert residual_normalizer is not None
    assert residual_normalizer.fitted
    assert set(encoder.output_dtypes) == {expected_dtype}
    if reference_processor is not None:
        assert set(reference_processor.encoder.output_dtypes) == {expected_dtype}
    torch.testing.assert_close(
        residual_normalizer.mean, expected_mean.to(residual_normalizer.mean)
    )
    torch.testing.assert_close(
        residual_normalizer.scale, expected_scale.to(residual_normalizer.scale)
    )
    assert all(batch.input_fields.dtype == torch.float32 for batch in batches)
    assert trainer.global_step == 1
