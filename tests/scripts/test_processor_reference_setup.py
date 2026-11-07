"""Configured references use the public setup paths and local synthetic checkpoints."""

from copy import deepcopy
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from autocast.decoders.identity import IdentityDecoder
from autocast.encoders.identity import IdentityEncoder
from autocast.processors.base import Processor
from autocast.scripts.config import save_resolved_config
from autocast.scripts.setup import (
    setup_datamodule,
    setup_epd_model,
    setup_processor_model,
)
from autocast.types import Batch, EncodedBatch, Tensor


class CoordinateEncoder(IdentityEncoder):
    def __init__(self, in_channels):
        super().__init__(in_channels)
        self.latent_channels = 2 * in_channels
        self.scale = nn.Parameter(torch.tensor(2.0))

    def encode(self, batch: Batch) -> Tensor:
        x = batch.input_fields[:, :, ::2] * self.scale
        return torch.cat((x, -x), dim=-1)


class CoordinateDecoder(IdentityDecoder):
    def decode(self, z: Tensor) -> Tensor:
        return z[..., : self.latent_channels // 2].repeat_interleave(2, dim=2) / 2


class ZeroField(nn.Module):
    def forward(self, z, **kwargs):
        return torch.zeros_like(z)


class TinyBackbone(nn.Module):
    def __init__(self, **dimensions):
        super().__init__()
        self.dimensions = dimensions
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout(x) * self.scale


class TinyProcessor(Processor[EncodedBatch]):
    def __init__(self, backbone, n_steps_output, **dimensions):
        super().__init__(**dimensions)
        self.backbone = backbone
        self.n_steps_output = n_steps_output
        self.received_cond = None

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:
        self.received_cond = global_cond
        return self.backbone(x[:, -1:]).expand(-1, self.n_steps_output, *x.shape[2:])

    def loss(self, batch: EncodedBatch) -> Tensor:
        return (
            (self.map(batch.encoded_inputs) - batch.encoded_output_fields)
            .square()
            .mean()
        )


def _processor_config():
    return {
        "_target_": f"{__name__}.TinyProcessor",
        "in_channels": "auto",
        "out_channels": "auto",
        "n_steps_input": "auto",
        "n_steps_output": "auto",
        "n_channels_out": "auto",
        "spatial_resolution": "auto",
        "global_cond_channels": "auto",
        "backbone": {
            "_target_": f"{__name__}.TinyBackbone",
            "in_channels": "auto",
            "out_channels": "auto",
            "cond_channels": "auto",
            "n_steps_input": "auto",
            "n_steps_output": "auto",
            "global_cond_channels": "auto",
        },
    }


@pytest.fixture
def reference_run(tmp_path, epd_config, request):
    for split in ("train", "valid", "test"):
        folder = tmp_path / split
        folder.mkdir()
        torch.save({"data": torch.ones(2, 5, 2, 3, 1)}, folder / "data.pt")
    epd_config.datamodule = {
        "_target_": "autocast.data.datamodule.SpatioTemporalDataModule",
        "data_path": str(tmp_path),
        "use_normalization": False,
        "n_steps_input": 2,
        "n_steps_output": 3,
        "stride": 1,
        "batch_size": 1,
        "num_workers": 0,
    }
    mode = getattr(request, "param", "ambient")
    if "normalized" in mode:
        epd_config.datamodule["use_normalization"] = True
        epd_config.datamodule["normalization_stats"] = {
            "core_field_names": ["u"],
            "constant_field_names": [],
            "stats": {
                "mean": {"u": 2.0},
                "std": {"u": 3.0},
                "mean_delta": {"u": 0.0},
                "std_delta": {"u": 1.0},
            },
        }
    if "latent" in mode:
        epd_config.model.encoder._target_ = f"{__name__}.CoordinateEncoder"
        epd_config.model.decoder._target_ = f"{__name__}.CoordinateDecoder"
    epd_config.model.processor = _processor_config()
    epd_config.model.freeze_encoder_decoder = True
    dm, config, stats = setup_datamodule(epd_config)
    source = cast(Any, setup_epd_model(config, stats, dm))
    with torch.no_grad():
        source.processor.backbone.scale.fill_(7)
    checkpoint = tmp_path / "reference.ckpt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    saved_config = save_resolved_config(config, tmp_path)
    parent = deepcopy(config)
    parent.model.processor = {
        "_target_": (
            "autocast.processors.residual_flow_matching.ResidualFlowMatchingProcessor"
        ),
        "backbone": {"_target_": f"{__name__}.ZeroField"},
        "source": {"_target_": "autocast.nn.noise.source.ZeroSource"},
        "n_steps_output": "auto",
        "n_channels_out": "auto",
        "reference": {
            "_target_": "autocast.processors.residual_reference.ProcessorReference",
            "processor": _processor_config(),
            "checkpoint_path": str(checkpoint),
            "config_path": str(saved_config),
        },
    }
    parent.model.loss_func = None
    parent.model.train_in_latent_space = True
    return parent, stats, dm, checkpoint, saved_config


def test_setup_resolves_nested_processor_and_backbone(reference_run):
    config, stats, dm, *_ = reference_run
    del config.model.processor.reference.checkpoint_path
    del config.model.processor.reference.config_path
    model = cast(Any, setup_epd_model(config, stats, dm))
    reference = model.processor.reference
    assert reference.processor.n_steps_output == 3
    assert reference.processor.in_channels == 1
    assert reference.processor.n_steps_input == 2
    assert list(reference.processor.spatial_resolution) == [2, 3]
    assert reference.processor.backbone.dimensions == {
        "in_channels": 1,
        "out_channels": 1,
        "cond_channels": 1,
        "n_steps_input": 2,
        "n_steps_output": 3,
        "global_cond_channels": 0,
    }


def test_setup_loads_reference_weights_and_preserves_frozen_eval(reference_run):
    config, stats, dm, *_ = reference_run
    model = cast(Any, setup_epd_model(config, stats, dm))
    model.train()
    reference = model.processor.reference
    x = torch.ones(1, 2, 2, 3, 1, requires_grad=True)
    cond = torch.ones(1, 2)
    output = reference(x, cond)
    torch.testing.assert_close(output, torch.full((1, 3, 2, 3, 1), 7.0))
    assert reference.processor.received_cond is cond
    assert not output.requires_grad
    assert not reference.processor.backbone.dropout.training
    assert not reference.processor.backbone.scale.requires_grad
    assert model.norm is dm.train_dataset.norm


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("in_channels", 2),
        ("out_channels", 2),
        ("n_steps_input", 4),
        ("n_steps_output", 4),
        ("spatial_resolution", [3, 2]),
        ("backbone.cond_channels", 2),
        ("backbone.n_steps_output", 4),
    ],
)
def test_setup_rejects_incompatible_explicit_dimensions(reference_run, field, value):
    config, stats, dm, *_ = reference_run
    OmegaConf.update(config.model.processor.reference.processor, field, value)
    with pytest.raises(ValueError, match="dimension"):
        setup_epd_model(config, stats, dm)


def test_setup_preserves_compatible_explicit_dimensions(reference_run):
    config, stats, dm, *_ = reference_run
    nested = config.model.processor.reference.processor
    nested.n_steps_input = 2
    nested.n_steps_output = 3
    nested.spatial_resolution = [2, 3]
    nested.backbone.cond_channels = 1
    setup_epd_model(config, stats, dm)
    assert nested.n_steps_input == 2
    assert nested.n_steps_output == 3
    assert nested.spatial_resolution == [2, 3]
    assert nested.backbone.cond_channels == 1


@pytest.mark.parametrize("frozen", [True, False])
def test_reference_is_independent_and_resume_state_wins(reference_run, frozen):
    config, stats, dm, checkpoint, _ = reference_run
    config.model.processor.reference.frozen = frozen
    first = cast(Any, setup_epd_model(deepcopy(config), stats, dm))
    second = cast(Any, setup_epd_model(deepcopy(config), stats, dm))
    parameter = first.processor.reference.processor.backbone.scale
    with torch.no_grad():
        parameter.fill_(11)
    assert second.processor.reference.processor.backbone.scale.item() == 7
    # Simulate restoration of a parent model checkpoint after setup. Nothing in
    # train()/forward()/state loading may overwrite it from the external source.
    second.load_state_dict(first.state_dict(), strict=True)
    checkpoint.unlink()
    second.train()
    ref = second.processor.reference
    x = torch.ones(1, 2, 2, 3, 1, requires_grad=True)
    if not frozen:
        assert ref.processor.backbone.dropout.training
        ref.eval()  # Deterministic value check; gradients must remain enabled.
    result = ref(x)
    torch.testing.assert_close(result, torch.full_like(result, 11))
    assert result.requires_grad is (not frozen)
    if not frozen:
        result.sum().backward()
        assert ref.processor.backbone.scale.grad is not None
        assert x.grad is not None


@pytest.mark.parametrize(
    "damage", ["prefix", "missing", "extra", "shape", "payload", "corrupt"]
)
def test_checkpoint_failures_are_not_silently_ignored(reference_run, damage):
    config, stats, dm, checkpoint, _ = reference_run
    payload = torch.load(checkpoint, weights_only=True)
    key = "processor.backbone.scale"
    if damage == "prefix":
        payload["state_dict"] = {
            "wrong." + k: v for k, v in payload["state_dict"].items()
        }
    elif damage == "missing":
        del payload["state_dict"][key]
    elif damage == "extra":
        payload["state_dict"]["processor.unexpected"] = torch.ones(1)
    elif damage == "shape":
        payload["state_dict"][key] = torch.ones(2)
    elif damage == "payload":
        payload = {"state_dict": 42}
    if damage == "corrupt":
        checkpoint.write_bytes(b"not a checkpoint")
    else:
        torch.save(payload, checkpoint)
    with pytest.raises(
        (ValueError, TypeError, RuntimeError), match=r"checkpoint|state_dict"
    ):
        setup_epd_model(config, stats, dm)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model.encoder._target_", "autocast.encoders.permute_concat.PermuteConcat"),
        ("model.decoder.in_channels", 2),
        ("datamodule.channel_idxs", [0]),
        ("datamodule.use_normalization", True),
        ("model.residual_prediction", True),
        ("model.input_noise_injector", {"_target_": "torch.nn.Identity"}),
        ("model.processor.backbone.dropout", 0.9),
    ],
)
def test_saved_coordinate_settings_must_match(reference_run, field, value):
    config, stats, dm, _, saved_config = reference_run
    source = OmegaConf.load(saved_config)
    OmegaConf.update(source, field, value)
    OmegaConf.save(source, saved_config)
    with pytest.raises(ValueError, match=r"reference|Reference"):
        setup_epd_model(config, stats, dm)


def test_metadata_free_checkpoint_is_not_proven_compatible(reference_run):
    config, stats, dm, _, saved_config = reference_run
    OmegaConf.save(OmegaConf.create({}), saved_config)
    with pytest.raises(ValueError, match=r"config|metadata"):
        setup_epd_model(config, stats, dm)


@pytest.mark.parametrize(
    "reference_run", ["normalized", "latent", "latent_normalized"], indirect=True
)
def test_matching_coordinate_provenance_loads(reference_run):
    config, stats, dm, *_ = reference_run
    model = cast(Any, setup_epd_model(config, stats, dm))
    reference = model.processor.reference
    encoded, cond = model.encoder_decoder.encoder.encode_with_cond(
        stats["example_batch"]
    )
    output = reference(encoded, cond)
    torch.testing.assert_close(output, (7 * encoded[:, -1:]).expand_as(output))
    nested = config.model.processor.reference.processor
    assert nested.in_channels == encoded.shape[-1]
    assert nested.spatial_resolution == list(encoded.shape[2:-1])
    assert model.encoder_decoder.encoder is not reference.processor
    assert model.norm is dm.train_dataset.norm


@pytest.mark.parametrize("reference_run", ["normalized"], indirect=True)
@pytest.mark.parametrize(
    "damage", ["missing_stats", "stats", "field_order", "live_stats"]
)
def test_normalization_provenance_is_required(reference_run, damage):
    config, stats, dm, _, saved_config = reference_run
    source = OmegaConf.load(saved_config)
    if damage == "missing_stats":
        del source.datamodule.normalization_stats
        del config.datamodule.normalization_stats
    elif damage == "stats":
        source.datamodule.normalization_stats.stats.mean.u = 10
    elif damage == "field_order":
        source.datamodule.normalization_stats.core_field_names = ["v"]
    else:
        dm.train_dataset.norm.flattened_stds["variable"].fill_(10)
    OmegaConf.save(source, saved_config)
    with pytest.raises(ValueError, match=r"[Rr]eference"):
        setup_epd_model(config, stats, dm)


@pytest.mark.parametrize("reference_run", ["latent"], indirect=True)
@pytest.mark.parametrize("damage", ["encoder", "trainable_source", "trainable_parent"])
def test_latent_coordinates_must_match_and_remain_fixed(reference_run, damage):
    config, stats, dm, checkpoint, saved_config = reference_run
    source = OmegaConf.load(saved_config)
    if damage == "encoder":
        payload = torch.load(checkpoint, weights_only=True)
        payload["state_dict"]["encoder_decoder.encoder.scale"].fill_(3)
        torch.save(payload, checkpoint)
    elif damage == "trainable_source":
        source.model.freeze_encoder_decoder = False
        OmegaConf.save(source, saved_config)
    else:
        config.model.freeze_encoder_decoder = False
        config.model.train_in_latent_space = False
    with pytest.raises(
        ValueError, match=r"[Rr]eference.*co(codec|ordinates)|frozen codecs"
    ):
        setup_epd_model(config, stats, dm)


def test_checkpoint_load_is_safe_cpu_only(reference_run, monkeypatch):
    config, stats, dm, checkpoint, _ = reference_run
    original = torch.load
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    setup_epd_model(deepcopy(config), stats, dm)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    # Tensor-only loading rejects Python module objects without an unsafe retry.
    torch.save({"state_dict": {}, "object": nn.Linear(1, 1)}, checkpoint)
    with pytest.raises(ValueError, match="safely"):
        setup_epd_model(config, stats, dm)
    assert len(calls) == 2
    assert all(call["weights_only"] is True for call in calls)


@pytest.mark.parametrize("field", ["checkpoint_path", "config_path"])
def test_remote_reference_paths_are_rejected(reference_run, field):
    config, stats, dm, *_ = reference_run
    config.model.processor.reference[field] = "https://example.invalid/reference"
    with pytest.raises(ValueError, match=r"local files|safely"):
        setup_epd_model(config, stats, dm)


def test_configured_reference_cannot_bypass_setup(reference_run):
    config, *_ = reference_run
    ref = instantiate(config.model.processor.reference)
    with pytest.raises(RuntimeError, match="require model setup"):
        ref(torch.ones(1, 2, 2, 3, 1))


def test_processor_only_setup_rejects_unproven_latent_provenance(reference_run):
    config, stats, dm, *_ = reference_run
    stats["input_shape"] = stats["example_batch"].input_fields.shape
    with pytest.raises(ValueError, match="cached latent metadata"):
        setup_processor_model(config, stats, dm)


@pytest.mark.parametrize(
    "reference_run", ["ambient", "latent_normalized"], indirect=True
)
def test_parent_forward_uses_reference_without_targets(reference_run):
    config, stats, dm, *_ = reference_run
    model = cast(Any, setup_epd_model(config, stats, dm))
    batch = stats["example_batch"]
    expected = (7 * batch.input_fields[:, -1:]).expand_as(batch.output_fields)
    torch.testing.assert_close(model(batch), expected)
    batch.output_fields.fill_(12345)
    torch.testing.assert_close(model(batch), expected)


def test_minimal_hydra_example_composes_and_loads(reference_run, REPO_ROOT):
    config, _, _, checkpoint, saved_config = reference_run
    # AViTLatentProcessor expects a conditioning tensor; use real local scalars.
    for split in ("train", "valid", "test"):
        file = saved_config.parent / split / "data.pt"
        payload = torch.load(file, weights_only=True)
        payload["constant_scalars"] = torch.ones(2, 2)
        torch.save(payload, file)
    dm, config, stats = setup_datamodule(config)
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "src/autocast/configs"), version_base=None
    ):
        example = compose(config_name="processor/residual_flow_matching_pretrained_vit")
    config.model.processor = example.processor
    reference = config.model.processor.reference
    reference.checkpoint_path = str(checkpoint)
    reference.config_path = str(saved_config)
    # Use small real CPU ViTs on both sides, retaining the example's architecture.
    reference.processor.hidden_dim = 8
    reference.processor.num_heads = 2
    reference.processor.n_layers = 1
    config.model.processor.backbone.hid_channels = 8
    config.model.processor.backbone.hid_blocks = 1
    source_config = deepcopy(config)
    source_config.model.processor = deepcopy(reference.processor)
    source = setup_epd_model(source_config, stats, dm)
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    save_resolved_config(source_config, saved_config.parent)
    model = cast(Any, setup_epd_model(config, stats, dm))
    model.train()
    assert not model.processor.reference.processor.training
    assert model.processor.reference.checkpoint_loaded
    batch = stats["example_batch"]
    result = model.processor.reference(batch.input_fields, batch.constant_scalars)
    assert result.shape == stats["example_batch"].output_fields.shape


def test_processor_only_setup_resolves_nested_autos_without_checkpoint(reference_run):
    config, stats, dm, *_ = reference_run
    del config.model.processor.reference.checkpoint_path
    del config.model.processor.reference.config_path
    stats["input_shape"] = stats["example_batch"].input_fields.shape
    model = cast(Any, setup_processor_model(config, stats, dm))
    nested = config.model.processor.reference.processor
    assert nested.n_steps_input == 2
    assert nested.n_steps_output == 3
    assert nested.backbone.cond_channels == 1
    result = model.processor.reference(stats["example_batch"].input_fields)
    torch.testing.assert_close(result, torch.full((1, 3, 2, 3, 1), 2.0))


def test_nested_reference_uses_explicit_parent_horizon(reference_run):
    config, stats, dm, *_ = reference_run
    del config.model.processor.reference.checkpoint_path
    del config.model.processor.reference.config_path
    config.model.processor.n_steps_output = 4
    model = cast(Any, setup_epd_model(config, stats, dm))
    assert config.model.processor.reference.processor.n_steps_output == 4
    assert config.model.processor.reference.processor.backbone.n_steps_output == 4
    assert model.processor.reference(stats["example_batch"].input_fields).shape[1] == 4


@pytest.mark.parametrize("damage", ["dtype", "tensor_type", "encoder_prefix"])
def test_checkpoint_tensor_metadata_is_checked(reference_run, damage):
    config, stats, dm, checkpoint, _ = reference_run
    payload = torch.load(checkpoint, weights_only=True)
    key = "processor.backbone.scale"
    if damage == "dtype":
        payload["state_dict"][key] = payload["state_dict"][key].double()
    elif damage == "tensor_type":
        payload["state_dict"][key] = 7.0
    else:
        payload["state_dict"]["encoder_decoder.encoder.unexpected"] = torch.ones(1)
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="Reference checkpoint"):
        setup_epd_model(config, stats, dm)


class ChainedReferenceProcessor(Processor[EncodedBatch]):
    def __init__(self, reference):
        super().__init__()
        self.reference = reference

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:
        return self.reference(x, global_cond)

    def loss(self, batch: EncodedBatch) -> Tensor:
        return (
            (self.map(batch.encoded_inputs) - batch.encoded_output_fields)
            .square()
            .mean()
        )


def test_outer_checkpoint_preserves_trained_nested_reference(reference_run, tmp_path):
    config, stats, dm, *_ = reference_run
    chain_config = deepcopy(config)
    chain_config.model.processor = {
        "_target_": f"{__name__}.ChainedReferenceProcessor",
        "reference": deepcopy(config.model.processor.reference),
    }
    chain_config.model.processor["reference"]["frozen"] = False
    chain = cast(Any, setup_epd_model(chain_config, stats, dm))
    with torch.no_grad():
        chain.processor.reference.processor.backbone.scale.fill_(11)
    chain_path = tmp_path / "chain"
    chain_path.mkdir()
    checkpoint = chain_path / "model.ckpt"
    torch.save({"state_dict": chain.state_dict()}, checkpoint)
    saved_config = save_resolved_config(chain_config, chain_path)
    config.model.processor.reference.processor = deepcopy(chain_config.model.processor)
    config.model.processor.reference.checkpoint_path = str(checkpoint)
    config.model.processor.reference.config_path = str(saved_config)

    model = cast(Any, setup_epd_model(config, stats, dm))

    nested = model.processor.reference.processor.reference
    assert nested.checkpoint_loaded
    assert nested.processor.backbone.scale.item() == 11
    result = model.processor.reference(stats["example_batch"].input_fields)
    torch.testing.assert_close(result, torch.full_like(result, 11))
