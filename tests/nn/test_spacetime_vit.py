import json
from pathlib import Path

import pytest
import torch
from azula.nn.attention import MultiheadSelfAttention
from hydra import compose, initialize_config_dir
from torch import nn

from autocast.losses.ensemble import AlphaFairCRPSLoss
from autocast.models.processor_ensemble import ProcessorModelEnsemble
from autocast.nn.spacetime_vit import SpatiotemporalViTBackbone
from autocast.nn.vit import SpatialViTBackbone, TemporalViTBackbone
from autocast.processors.diffusion import DiffusionProcessor
from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.processors.flow_matching_masked_window import (
    FlowMatchingMaskedWindowProcessor,
)
from autocast.processors.spacetime_vit import (
    SpacetimeViTMaskedWindowProcessor,
    SpacetimeViTProcessor,
)
from autocast.scripts.setup import _build_processor
from autocast.types import EncodedBatch


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _backbone(**overrides):
    kwargs = {
        "in_channels": 2,
        "out_channels": 2,
        "cond_channels": 2,
        "n_steps_input": 1,
        "n_steps_output": 4,
        "mod_features": 4,
        "hid_channels": 8,
        "hid_blocks": 2,
        "attention_heads": 2,
        "dropout": 0.0,
    }
    return SpatiotemporalViTBackbone(**(kwargs | overrides))


def test_matches_pinned_lola_forward_and_gradient():
    # Generated independently using the pinned upstream ViT/attention source,
    # not this implementation. Fixed nonzero gates exercise the input skip.
    reference = json.loads(
        (Path(__file__).parent / "fixtures/lola_spacetime.json").read_text()
    )
    model = _backbone(n_steps_input=3, n_steps_output=3, conditioning="mask").vit
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(
                0.1 * torch.sin(torch.arange(param.numel()).reshape(param.shape) + 1)
            )
        for block in model.blocks:
            # Upstream stores RoPE as (axes, features); Azula uses Linear's
            # transposed storage convention for the same coordinate mapping.
            theta = 0.1 * torch.sin(torch.arange(12).reshape(3, 4) + 1)
            projection = block.get_submodule("msa.theta_proj")
            assert isinstance(projection, nn.Linear)
            projection.weight.copy_(theta.T)
    x = torch.linspace(-1, 1, 24).reshape(1, 2, 3, 2, 2).requires_grad_()
    mask = torch.zeros_like(x)
    mask[:, :, 0] = 1
    mod = torch.linspace(-0.5, 0.5, 4).reshape(1, 4)
    y = model(x, mod, mask)
    y.square().sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(
        y.flatten(), torch.tensor(reference["output"]), atol=1e-7, rtol=1e-5
    )
    torch.testing.assert_close(
        x.grad.flatten(), torch.tensor(reference["input_grad"]), atol=1e-7, rtol=1e-5
    )
    assert sum(p.numel() for p in model.parameters()) == reference["parameters"]


def test_rb_architecture_and_spacetime_token_count():
    reference = json.loads(
        (Path(__file__).parent / "fixtures/lola_spacetime.json").read_text()
    )
    with torch.device("meta"):
        rb = SpatiotemporalViTBackbone(64, 64, 64, 4, 1)
    assert (
        sum(p.numel() for p in rb.vit.parameters()) == reference["rb_trunk_parameters"]
    )
    assert len(rb.vit.blocks) == 16
    attention = rb.vit.blocks[0].get_submodule("msa")
    gates = rb.vit.blocks[0].get_submodule("ada_zero.2")
    assert isinstance(attention, MultiheadSelfAttention)
    assert isinstance(gates, nn.Linear)
    assert attention.heads == 8
    assert rb.vit.in_proj.out_features == 1024
    assert rb.vit.in_proj.in_features == 128  # Per-frame fields + mask, not 5*C.
    assert gates.out_features == 4 * 1024
    assert attention.bias_proj is None
    assert attention.qkv_proj.bias is None
    assert attention.y_proj.bias is not None
    assert isinstance(rb.time_embedding, nn.Sequential)
    assert rb.time_embedding[0].omega == 1e3

    model = _backbone()
    captured = []
    hook = model.vit.blocks[0].register_forward_pre_hook(
        lambda _, args: captured.append((args[0].shape, args[2].clone()))
    )
    y = model(torch.randn(1, 4, 16, 4, 2), torch.ones(1), torch.randn(1, 1, 16, 4, 2))
    hook.remove()
    assert y.shape == (1, 4, 16, 4, 2)
    shape, pos = captured[0]
    assert shape == (1, 320, 8)
    assert pos.shape == (320, 3)
    assert torch.equal(pos[:, 0].unique(), torch.arange(5, dtype=pos.dtype))


def test_checkpointing_preserves_dropout_outputs_and_gradients():
    plain = _backbone(patch_size=(1, 2, 2), dropout=0.05)
    checked = _backbone(patch_size=(1, 2, 2), dropout=0.05, checkpointing=True)
    checked.load_state_dict(plain.state_dict())
    x = torch.randn(1, 4, 4, 2, 2)
    context = torch.randn(1, 1, 4, 2, 2)
    outputs = []
    for model in (plain, checked):
        torch.manual_seed(17)
        y = model(x, torch.ones(1), context)
        assert y.shape == x.shape
        y.square().mean().backward()
        outputs.append(y)
    torch.testing.assert_close(*outputs)
    for a, b in zip(plain.parameters(), checked.parameters(), strict=True):
        assert a.grad is not None
        assert b.grad is not None
        torch.testing.assert_close(a.grad, b.grad)
    checked.eval()
    with torch.no_grad():
        assert torch.isfinite(checked(x, torch.ones(1), context)).all()


@pytest.mark.parametrize(
    "preset",
    [
        "flow_matching_spacetime_vit",
        "flow_matching_masked_window_spacetime_vit",
        "diffusion_spacetime_vit",
        "spacetime_vit",
        "spacetime_vit_masked_window",
    ],
)
def test_processor_config_and_objective(preset, REPO_ROOT):
    with initialize_config_dir(
        str(REPO_ROOT / "src/autocast/configs"), version_base=None
    ):
        cfg = compose(
            config_name="processor", overrides=[f"processor@model.processor={preset}"]
        )
    cfg.model.processor.backbone.hid_channels = 8
    cfg.model.processor.backbone.hid_blocks = 2
    cfg.model.processor.backbone.mod_features = 4
    cfg.model.processor.backbone.attention_heads = 2
    cfg.model.processor.backbone.dropout = 0.0
    processor = _build_processor(
        cfg.model,
        {
            "in_channels": 2,
            "out_channels": 2,
            "n_channels_out": 2,
            "n_steps_input": 1,
            "n_steps_output": 4,
            "spatial_resolution": (4, 2),
        },
        global_cond_channels=2,
    )
    batch = EncodedBatch(
        encoded_inputs=torch.randn(2, 1, 4, 2, 2),
        encoded_output_fields=torch.randn(2, 4, 4, 2, 2),
        global_cond=torch.randn(2, 2),
        encoded_info={},
    )
    if isinstance(processor, SpacetimeViTProcessor):
        model = ProcessorModelEnsemble(
            processor,
            n_members=8,
            loss_func=(
                None
                if isinstance(processor, SpacetimeViTMaskedWindowProcessor)
                else AlphaFairCRPSLoss()
            ),
        )
        loss = model.loss(batch)
        pred = model(batch.encoded_inputs, batch.global_cond)
        assert pred.shape == (*batch.encoded_output_fields.shape, 8)
        assert not torch.equal(pred[..., 0], pred[..., 1])
        assert processor.backbone.time_embedding is None
    else:
        assert isinstance(processor, (FlowMatchingProcessor, DiffusionProcessor))
        if isinstance(processor, FlowMatchingProcessor):
            processor.flow_ode_steps = 2
        else:
            processor.sampler_steps = 2
        loss = processor.loss(batch)
        pred = processor.map(batch.encoded_inputs, batch.global_cond)
        assert pred.shape == batch.encoded_output_fields.shape
    assert torch.isfinite(pred).all()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in processor.parameters()
    )


def test_crps_future_slots_and_noise_are_independent_of_targets():
    processor = SpacetimeViTProcessor(_backbone(use_precomputed_modulation=True), 6)
    x, noise = torch.randn(2, 1, 4, 2, 2), torch.randn(2, 6)
    seen = []
    hook = processor.backbone.register_forward_pre_hook(
        lambda _, args, kwargs: seen.append((args[0].clone(), kwargs["cond"].clone())),
        with_kwargs=True,
    )
    a = processor(x, noise)
    b = processor(x, noise)
    hook.remove()
    assert torch.equal(a, b)
    assert torch.count_nonzero(seen[0][0]) == 0
    assert torch.equal(seen[0][1], x)


def test_masked_fm_clamps_context_throughout_sampling():
    backbone = _backbone(n_steps_input=5, n_steps_output=5, conditioning="mask")
    processor = FlowMatchingMaskedWindowProcessor(
        backbone=backbone,
        n_steps_input=1,
        n_steps_output=4,
        n_channels_out=2,
        flow_ode_steps=3,
    )
    windows = []
    hook = backbone.register_forward_pre_hook(
        lambda _, args: windows.append(args[0].detach().clone()),
    )
    context = torch.randn(1, 1, 4, 2, 2)
    pred = processor.map(context, None)
    hook.remove()
    assert pred.shape == (1, 4, 4, 2, 2)
    assert len(windows) == 3
    assert all(torch.equal(window[:, :1], context) for window in windows)


def test_existing_temporal_vit_keeps_channel_packing():
    kwargs = {
        "in_channels": 2,
        "out_channels": 2,
        "cond_channels": 2,
        "n_steps_input": 1,
        "n_steps_output": 4,
        "global_cond_channels": None,
        "include_global_cond": False,
        "hid_channels": 8,
        "hid_blocks": 1,
        "attention_heads": 2,
        "mod_features": 4,
        "patch_size": 1,
    }
    model = TemporalViTBackbone(**kwargs)
    seen = []
    blocks = model.vit.get_submodule("blocks")
    assert isinstance(blocks, nn.ModuleList)
    hook = blocks[0].register_forward_pre_hook(
        lambda _, args: seen.append(args[0].shape)
    )
    x = torch.randn(1, 4, 16, 4, 2)
    context = torch.randn(1, 1, 16, 4, 2)
    output = model(x, torch.ones(1), context)
    hook.remove()
    assert output.shape == x.shape
    assert seen == [(1, 64, 8)]
    projection = model.vit.get_submodule("in_proj")
    assert isinstance(projection, nn.Linear)
    assert projection.in_features == 10
    restored = SpatialViTBackbone(**kwargs)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x, torch.ones(1), context), output)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"conditioning": "other"}, "conditioning"),
        ({"patch_size": (2, 1, 1)}, "patch_size"),
        ({"out_channels": 3}, "channel counts"),
        ({"conditioning": "mask"}, "equal full-window"),
        ({"rope": False}, "LOLA attention"),
        ({"rpb": True}, "LOLA attention"),
        ({"window_size": 3}, "LOLA attention"),
    ],
)
def test_invalid_configuration(overrides, match):
    with pytest.raises(ValueError, match=match):
        _backbone(**overrides)
