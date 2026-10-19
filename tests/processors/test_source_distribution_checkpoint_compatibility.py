"""Source naming must not change checkpoint keys or seeded model behaviour."""

from collections import OrderedDict
from typing import TypeVar, cast

import pytest
import torch
from einops import rearrange
from hydra.utils import instantiate
from torch import nn
from torch.optim.swa_utils import AveragedModel

from autocast.nn.noise.noise_injector import ConcatenatedNoiseInjector
from autocast.nn.noise.source import (
    FlowSource,
    SeparableGaussianSource,
    SourceDistribution,
)
from autocast.processors.azula_vit import AzulaViTProcessor
from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.processors.residual_flow_matching import ResidualFlowMatchingProcessor
from autocast.processors.residual_normalization import ResidualNormalizer
from autocast.processors.residual_reference import LastFrameReference
from autocast.types import EncodedBatch

_Module = TypeVar("_Module", bound=nn.Module)


class _Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))

    def forward(self, z, t, cond, global_cond=None):  # noqa: ARG002
        time = rearrange(t, "b -> b 1 1 1 1")
        return z * self.weight + cond[:, :1] + time


def _historical_source() -> SeparableGaussianSource:
    # Concrete Hydra targets already saved in run/checkpoint configurations must
    # keep resolving, even though the abstract interface has a new name.
    return instantiate(
        {
            "_target_": "autocast.nn.noise.source.SeparableGaussianSource",
            "temporal_correlation_time": 1.5,
            "spatial_length_scale": 0.8,
            "channel_correlation": [[1.0, 0.2], [0.2, 1.0]],
        }
    )


def _wrap(processor: _Module, prefix: str) -> tuple[nn.Module, _Module]:
    if not prefix:
        return processor, processor
    model = nn.Module()
    model.processor = processor
    if prefix.startswith("module."):
        model = AveragedModel(model)
        processor = cast(_Module, model.module.processor)
    return model, processor


@pytest.mark.parametrize("prefix", ["", "processor.", "module.processor."])
@pytest.mark.parametrize("residual", [False, True])
def test_historical_flow_checkpoint_preserves_source_and_seeded_results(
    prefix, residual
):
    options = {
        "backbone": _Field(),
        "source": _historical_source(),
        "n_steps_output": 2,
        "n_channels_out": 2,
        "flow_ode_steps": 1,
    }
    if residual:
        processor = ResidualFlowMatchingProcessor(
            **options,
            reference=LastFrameReference(n_steps_output=2),
            residual_normalizer=ResidualNormalizer(n_steps_output=2, n_channels=2),
        )
    else:
        processor = FlowMatchingProcessor(**options)
    model, processor = _wrap(processor, prefix)

    # These are the pre-rename registered names, not keys derived from the new
    # model. A nontrivial buffer makes a lost/renamed source visible under strict
    # loading; stateless IID-only checks cannot detect that regression.
    state = OrderedDict(
        {
            prefix + "flow_matching_model.weight": torch.tensor([0.2, -0.1]),
            prefix + "source.channel_factor": torch.tensor([[1.0, 0.0], [0.6, 0.8]]),
        }
    )
    if residual:
        state.update(
            {
                prefix + "residual_normalizer.mean": torch.full((2, 2), 0.3),
                prefix + "residual_normalizer.scale": torch.full((2, 2), 1.5),
                prefix + "residual_normalizer._fitted": torch.tensor(True),
            }
        )
    if prefix.startswith("module."):
        state["n_averaged"] = torch.tensor(5)
    keys = tuple(state)
    loaded = model.load_state_dict(state, strict=True)
    assert not loaded.missing_keys
    assert not loaded.unexpected_keys
    assert tuple(state) == keys
    assert set(model.state_dict()) == set(state)
    assert isinstance(processor.source, SourceDistribution)
    assert isinstance(processor.source, FlowSource)
    assert "source" in processor._modules
    assert "noise_source" not in processor._modules

    source = _historical_source()
    source.load_state_dict({"channel_factor": state[prefix + "source.channel_factor"]})
    x = torch.full((2, 1, 4, 4, 2), 0.7)
    y = torch.full((2, 2, 4, 4, 2), 1.2)
    torch.manual_seed(23)
    sample = source.sample_like(y)
    expected = sample + sample * torch.tensor([0.2, -0.1]) + x
    if residual:
        expected = x + expected * 1.5 + 0.3
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(23)
    torch.testing.assert_close(processor.map(x, None), expected)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)

    target = (y - x - 0.3) / 1.5 if residual else y
    torch.manual_seed(24)
    sample = source.sample_like(target)
    time = rearrange(torch.rand(2), "b -> b 1 1 1 1")
    interpolant = (1 - time) * sample + time * target
    velocity = interpolant * torch.tensor([0.2, -0.1]) + x + time
    expected_loss = (velocity - (target - sample)).square().mean()
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(24)
    batch = EncodedBatch(
        encoded_inputs=x,
        encoded_output_fields=y,
        global_cond=None,
        encoded_info={},
    )
    torch.testing.assert_close(processor.loss(batch), expected_loss, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)


@pytest.mark.parametrize("prefix", ["", "processor.", "module.processor."])
@pytest.mark.parametrize("kind", ["concat", "adaln"])
def test_conditioning_checkpoint_keeps_noise_source_buffer_and_rng(prefix, kind):
    source = _historical_source()
    if kind == "concat":
        processor = ConcatenatedNoiseInjector(n_channels=2, noise_source=source)
    else:
        processor = AzulaViTProcessor(
            in_channels=2,
            out_channels=2,
            spatial_resolution=(8, 8),
            hidden_dim=8,
            num_heads=2,
            n_layers=1,
            patch_size=2,
            n_noise_input_channels=2,
            n_noise_channels=4,
            noise_mode="spatial",
            noise_source=source,
            temporal_method="none",
        )
    model, processor = _wrap(processor, prefix)
    state = model.state_dict()
    key = prefix + "noise_source.channel_factor"
    assert key in state
    factor = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    state[key] = factor
    loaded = model.load_state_dict(state, strict=True)
    assert not loaded.missing_keys
    assert not loaded.unexpected_keys
    assert "noise_source" in processor._modules
    assert "source" not in processor._modules
    assert isinstance(processor.noise_source, SeparableGaussianSource)
    assert processor.noise_source.channel_factor is not None
    assert torch.equal(processor.noise_source.channel_factor, factor)

    source = _historical_source()
    source.load_state_dict({"channel_factor": factor})
    x = torch.ones(2, 1, 8, 8, 2)
    torch.manual_seed(25)
    if kind == "concat":
        expected = torch.cat((x, source.sample_like(x)), dim=-1)
    else:
        noise = source.sample_like(torch.empty(2, 1, 4, 4, 2))
        expected = processor(x, rearrange(noise, "b 1 h w c -> b (h w) c"))
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(25)
    actual = (
        processor(x)
        if isinstance(processor, ConcatenatedNoiseInjector)
        else processor.map(x, None)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
