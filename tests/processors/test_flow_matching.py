import itertools
import math

import lightning as L
import pytest
import torch
from conftest import get_optimizer_config
from torch.utils.data import DataLoader, Dataset

from autocast.models.processor import ProcessorModel
from autocast.nn.unet import TemporalUNetBackbone
from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.processors.flow_matching_masked_window import (
    FlowMatchingMaskedWindowProcessor,
)
from autocast.types import EncodedBatch


def _single_item_collate(items):
    return items[0]


class _FlowMatchingEncodedDataset(Dataset):
    """Minimal dataset that generates flow-matching-friendly `EncodedBatch` samples."""

    def __init__(
        self,
        *,
        n_steps_input: int,
        n_steps_output: int,
        n_channels_in: int,
        n_channels_out: int,
        spatial_size: int = 8,
    ) -> None:
        super().__init__()
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
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


def _build_encoded_loader(
    *,
    n_steps_input: int,
    n_steps_output: int,
    n_channels_in: int,
    n_channels_out: int,
) -> DataLoader:
    dataset = _FlowMatchingEncodedDataset(
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=_single_item_collate,
        num_workers=0,
    )


params = list(
    itertools.product(
        [1, 4],  # n_steps_output
        [1, 4],  # n_steps_input
        [1, 2],  # n_channels_in
        [1, 4],  # n_channels_out
    )
)


@pytest.mark.parametrize(
    "temporal_method",
    ["none", "attention", "tcn"],
)
@pytest.mark.parametrize(
    ("n_steps_output", "n_steps_input", "n_channels_in", "n_channels_out"),
    params,
)
def test_flow_matching_processor(
    n_steps_output: int,
    n_steps_input: int,
    n_channels_in: int,
    n_channels_out: int,
    temporal_method: str,
):
    encoded_loader = _build_encoded_loader(
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
    )
    encoded_batch = next(iter(encoded_loader))

    processor = FlowMatchingProcessor(
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
        n_steps_output=n_steps_output,
        n_channels_out=n_channels_out,
        flow_ode_steps=1,
    )
    model = ProcessorModel(processor, optimizer_config=get_optimizer_config())
    output = model.map(encoded_batch.encoded_inputs, None)
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
        accelerator="cpu",
    ).fit(
        model,
        train_dataloaders=encoded_loader,
        val_dataloaders=encoded_loader,
    )


def test_masked_window_flow_matching_processor():
    n_steps_input = 1
    n_steps_output = 4
    n_channels = 2
    full_steps = n_steps_input + n_steps_output
    encoded_loader = _build_encoded_loader(
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        n_channels_in=n_channels,
        n_channels_out=n_channels,
    )
    encoded_batch = next(iter(encoded_loader))

    processor = FlowMatchingMaskedWindowProcessor(
        backbone=TemporalUNetBackbone(
            in_channels=n_channels,
            out_channels=n_channels,
            cond_channels=n_channels,
            n_steps_output=full_steps,
            n_steps_input=full_steps,
            include_global_cond=False,
            global_cond_channels=None,
            temporal_method="none",
            mod_features=256,
            hid_channels=(32, 64, 128),
            hid_blocks=(2, 2, 2),
            spatial=2,
            periodic=False,
        ),
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        n_channels_out=n_channels,
        flow_ode_steps=1,
    )
    model = ProcessorModel(processor, optimizer_config=get_optimizer_config())
    output = model.map(encoded_batch.encoded_inputs, None)
    assert output.shape == encoded_batch.encoded_output_fields.shape

    train_loss = model.training_step(encoded_batch, 0)
    assert train_loss.shape == ()
    train_loss.backward()


def _lola_random_context_mask(
    x: torch.Tensor,
    *,
    lmbda: float = 1.0,
    rho: float = 1.0,
    atleast: int = 0,
) -> torch.Tensor:
    batch_size, _, total_steps, *shape = x.shape

    rate = torch.full((batch_size, 1), fill_value=lmbda, device=x.device)
    context = torch.poisson(rate).long()
    context = context % (total_steps - atleast) + atleast

    index = torch.arange(total_steps, device=x.device)
    if rho <= 0.0:
        mask = index >= total_steps - context
    elif rho >= 1.0:
        mask = index < context
    else:
        prefix = index < context
        suffix = index >= total_steps - context
        choose_prefix = torch.rand((batch_size, 1), device=x.device) < rho
        mask = torch.where(choose_prefix, prefix, suffix)

    mask = mask.reshape(batch_size, 1, total_steps, *([1] * len(shape)))
    return mask.expand_as(x)


@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
def test_masked_window_context_mask_matches_lola(rho: float):
    processor = FlowMatchingMaskedWindowProcessor(
        backbone=torch.nn.Identity(),
        n_steps_output=4,
        n_channels_out=3,
        lmbda=2.0,
        rho=rho,
        atleast=1,
    )
    x = torch.empty(5, 6, 4, 4, 3)

    torch.manual_seed(12)
    actual = processor._random_context_mask(x)

    torch.manual_seed(12)
    lola_x = torch.empty(5, 3, 6, 4, 4)
    expected = _lola_random_context_mask(
        lola_x,
        lmbda=2.0,
        rho=rho,
        atleast=1,
    )
    expected = expected.permute(0, 2, 3, 4, 1)

    assert torch.equal(actual, expected)


# ---------------------------------------------------------------------------
# Integrator option (euler default / heun)
# ---------------------------------------------------------------------------


class _DecayField(torch.nn.Module):
    """Stub backbone implementing the analytic field f(z, t) = -z.

    The flow ODE dz/dt = -z has the exact solution z(1) = z(0) * exp(-1),
    giving a closed-form target to test integrator accuracy against.
    """

    include_time_embedding = True

    def forward(self, z, t=None, cond=None, global_cond=None):  # noqa: ARG002
        return -z


def _decay_processor(integrator: str, steps: int) -> FlowMatchingProcessor:
    return FlowMatchingProcessor(
        backbone=_DecayField(),
        n_steps_output=1,
        n_channels_out=1,
        flow_ode_steps=steps,
        integrator=integrator,
    )


def test_flow_matching_integrator_validation():
    with pytest.raises(ValueError, match="integrator must be one of"):
        _decay_processor("rk99", 4)


def test_flow_matching_euler_default_unchanged():
    """integrator='euler' (the default) reproduces the original sampling path
    bit-for-bit: same RNG consumption, same update rule."""
    x = torch.randn(3, 1, 2, 2, 1)
    steps = 4
    torch.manual_seed(123)
    actual = FlowMatchingProcessor(
        backbone=_DecayField(),
        n_steps_output=1,
        n_channels_out=1,
        flow_ode_steps=steps,
    ).map(x, None)

    # Reconstruct the pre-source-API implementation directly: draw with
    # torch.randn, then apply the original Euler update z <- z - dt * z.
    torch.manual_seed(123)
    expected = torch.randn(3, 1, 2, 2, 1)
    dt = 1.0 / steps
    for _ in range(steps):
        expected = expected - dt * expected

    assert torch.equal(actual, expected)


def test_flow_matching_default_loss_matches_original_formulation():
    """The default source preserves the original loss and RNG sequence."""
    inputs = torch.randn(3, 1, 2, 2, 1)
    targets = torch.randn(3, 2, 2, 2, 1)
    batch = EncodedBatch(
        encoded_inputs=inputs,
        encoded_output_fields=targets,
        global_cond=None,
        encoded_info={},
    )
    processor = FlowMatchingProcessor(
        backbone=_DecayField(),
        n_steps_output=2,
        n_channels_out=1,
    )

    torch.manual_seed(123)
    actual = processor.loss(batch)

    # Reconstruct the implementation before sources became configurable.
    torch.manual_seed(123)
    z0 = torch.randn_like(targets)
    t = torch.rand(targets.shape[0], device=targets.device, dtype=targets.dtype)
    t_broadcast = t.view(targets.shape[0], *([1] * (targets.ndim - 1)))
    zt = (1 - t_broadcast) * z0 + t_broadcast * targets
    target_velocity = targets - z0
    expected = torch.mean((-zt - target_velocity) ** 2)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("integrator", ["euler", "heun"])
def test_flow_matching_integrator_shapes(integrator):
    x = torch.randn(2, 1, 4, 4, 2)
    proc = FlowMatchingProcessor(
        backbone=_DecayField(),
        n_steps_output=1,
        n_channels_out=2,
        flow_ode_steps=3,
        integrator=integrator,
    )
    out = proc.map(x, None)
    assert out.shape == (2, 1, 4, 4, 2)
    assert torch.isfinite(out).all()


def test_flow_matching_heun_more_accurate_than_euler_on_decay_field():
    """On dz/dt = -z (exact factor e^-1), Heun's per-step factor
    (1 - dt + dt^2/2) is second-order accurate vs Euler's (1 - dt):
    at 8 steps the Heun endpoint must be strictly closer to the truth."""
    steps = 8
    x = torch.randn(4, 1, 2, 2, 1)

    torch.manual_seed(7)
    z_euler = _decay_processor("euler", steps).map(x, None)
    torch.manual_seed(7)
    z_heun = _decay_processor("heun", steps).map(x, None)

    # Recover the shared initial noise draw to compute the exact endpoint.
    torch.manual_seed(7)
    z0 = torch.randn(4, 1, 2, 2, 1)
    exact = z0 * math.exp(-1.0)

    err_euler = (z_euler - exact).abs().max().item()
    err_heun = (z_heun - exact).abs().max().item()
    assert err_heun < err_euler
    # Analytic per-step factors at dt = 1/8.
    dt = 1.0 / steps
    assert torch.allclose(z_euler, z0 * (1.0 - dt) ** steps, atol=1e-6)
    assert torch.allclose(z_heun, z0 * (1.0 - dt + 0.5 * dt * dt) ** steps, atol=1e-6)
