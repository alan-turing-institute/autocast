import itertools

import lightning as L
import pytest
import torch
from azula.noise import VPSchedule
from conftest import get_optimizer_config
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autocast.models.processor import ProcessorModel
from autocast.nn.unet import TemporalUNetBackbone
from autocast.processors.diffusion import DiffusionProcessor
from autocast.processors.noise import LogLogitSchedule
from autocast.types import EncodedBatch


def _single_item_collate(items):
    return items[0]


class _DiffusionEncodedDataset(Dataset):
    """Minimal dataset that generates diffusion-friendly `EncodedBatch` samples."""

    def __init__(
        self,
        *,
        n_steps_input: int,
        n_steps_output: int,
        n_channels_in: int,
        n_channels_out: int,
        n_steps: int | None = None,
        spatial_size: int = 8,
    ) -> None:
        super().__init__()
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.n_steps = n_steps
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.spatial_size = spatial_size

    def __len__(self) -> int:
        return 2

    def __getitem__(self, _: int) -> EncodedBatch:
        encoded_inputs = torch.randn(
            1,
            self.n_steps_input,
            self.spatial_size,
            self.spatial_size,
            self.n_channels_in,
        )
        encoded_outputs = torch.randn(
            1,
            self.n_steps_output
            if self.n_steps is None
            else (self.n_steps - self.n_steps_input),
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


class _CaptureBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_t: torch.Tensor | None = None

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,  # noqa: ARG002
        global_cond: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        self.last_t = t.detach()
        return torch.zeros_like(x_t)


def _build_encoded_loader(
    *,
    n_steps_input: int,
    n_steps_output: int,
    n_channels_in: int,
    n_channels_out: int,
    n_steps: int | None = None,
) -> DataLoader[EncodedBatch]:
    dataset = _DiffusionEncodedDataset(
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        n_steps=n_steps,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=_single_item_collate,
        num_workers=0,
    )


def test_log_logit_schedule_matches_lola_formula():
    schedule = LogLogitSchedule(
        sigma_min=1e-3,
        sigma_max=1e3,
        scale=1.0,
        shift=0.0,
    )
    t = torch.tensor([0.0, 0.25, 0.75, 1.0])

    alpha, sigma = schedule(t)

    t_min = 1e-3 / (1 + 1e-3)
    t_max = 1e3 / (1 + 1e3)
    expected_sigma = torch.exp(torch.logit(t * (t_max - t_min) + t_min))

    assert torch.allclose(alpha, torch.ones_like(t))
    assert torch.allclose(sigma, expected_sigma)


def test_lola_denoiser_uses_scaled_log_sigma_modulation():
    backbone = _CaptureBackbone()
    schedule = LogLogitSchedule()
    processor = DiffusionProcessor(
        backbone=backbone,
        schedule=schedule,
        denoiser_type="lola",
        n_steps_output=4,
        n_channels_out=2,
    )
    x = torch.randn(2, 4, 3, 3, 2)
    t = torch.tensor([0.2, 0.6])

    processor.denoiser(
        x,
        t,
        cond=torch.randn(2, 1, 3, 3, 2),
        global_cond=None,
    )

    assert backbone.last_t is not None
    alpha, sigma = schedule(t)
    expected_t = 10.0 * torch.log(sigma / alpha)
    assert torch.allclose(backbone.last_t, expected_t)


def test_lola_diffusion_processor_loss_is_finite():
    processor = DiffusionProcessor(
        backbone=_CaptureBackbone(),
        schedule=LogLogitSchedule(),
        denoiser_type="lola",
        n_steps_output=2,
        n_channels_out=1,
    )
    batch = EncodedBatch(
        encoded_inputs=torch.randn(2, 1, 3, 3, 1),
        encoded_output_fields=torch.randn(2, 2, 3, 3, 1),
        global_cond=None,
        encoded_info={},
    )

    loss = processor.loss(batch)

    assert loss.shape == ()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("sampler", ["ab", "vab"])
def test_diffusion_ab_sampler_runs_and_is_deterministic(sampler: str):
    """LoLA's Adams-Bashforth samplers integrate through `map` deterministically.

    Guards the wiring for matching LoLA's inference recipe (algorithm="ab",
    order=2). Being a deterministic ODE solver, the same init noise (seed) must
    yield an identical sample.
    """
    processor = DiffusionProcessor(
        backbone=_CaptureBackbone(),
        schedule=LogLogitSchedule(),
        denoiser_type="lola",
        n_steps_output=4,
        n_channels_out=2,
        sampler=sampler,
        sampler_steps=8,
        sampler_order=2,
    )
    x = torch.randn(2, 1, 8, 8, 2)

    torch.manual_seed(0)
    out_a = processor.map(x, None)
    torch.manual_seed(0)
    out_b = processor.map(x, None)

    assert out_a.shape == (2, 4, 8, 8, 2)
    assert torch.isfinite(out_a).all()
    assert torch.allclose(out_a, out_b)


params = list(
    itertools.product(
        [1, 4],  # n_steps_output
        [1, 4],  # n_steps_input
        [1, 2],  # n_channels_in
        [1, 4],  # n_channels_out
        ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"],  # accelerator
    )
)


@pytest.mark.parametrize(
    (
        "n_steps_output",
        "n_steps_input",
        "n_channels_in",
        "n_channels_out",
        "accelerator",
    ),
    params,
)
@pytest.mark.parametrize(
    "temporal_method",
    ["none", "attention", "tcn"],
)
def test_diffusion_processor(
    n_steps_output: int,
    n_steps_input: int,
    n_channels_in: int,
    n_channels_out: int,
    accelerator: str,
    temporal_method: str,
):
    encoded_loader = _build_encoded_loader(
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
    )
    encoded_batch: EncodedBatch = next(iter(encoded_loader))

    processor = DiffusionProcessor(
        backbone=TemporalUNetBackbone(
            in_channels=n_channels_out,
            out_channels=n_channels_out,
            cond_channels=n_channels_in,
            n_steps_output=n_steps_output,
            n_steps_input=n_steps_input,
            include_global_cond=False,
            global_cond_channels=None,
            temporal_method=temporal_method,
            mod_features=256,
            hid_channels=(32, 64, 128),
            hid_blocks=(2, 2, 2),
            spatial=2,
            periodic=False,
        ),
        schedule=VPSchedule(),
        n_steps_output=n_steps_output,
        n_channels_out=n_channels_out,
        sampler_steps=5,
    )
    model = ProcessorModel(
        processor=processor,
        optimizer_config=get_optimizer_config(),
        sampler_steps=5,
        stride=n_steps_output,
    )
    output = model.map(encoded_batch.encoded_inputs, encoded_batch.global_cond)
    assert output.shape == encoded_batch.encoded_output_fields.shape

    train_loss = model.training_step(encoded_batch, 0)
    assert train_loss.shape == ()
    train_loss.backward()

    L.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
        enable_model_summary=False,
        accelerator=accelerator,
    ).fit(
        model,
        train_dataloaders=encoded_loader,
        val_dataloaders=encoded_loader,
    )

    # Testing map
    with torch.no_grad():
        model.eval()
        output = model.map(encoded_batch.encoded_inputs, encoded_batch.global_cond)
        assert output.shape == encoded_batch.encoded_output_fields.shape

    # Testing rollout (only when input and output channels match)
    if n_channels_in == n_channels_out:
        encoded_rollout_loader = _build_encoded_loader(
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
        )
        batch = next(iter(encoded_rollout_loader))
        model.rollout(batch, stride=n_steps_output)
