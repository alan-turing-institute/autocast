"""Unit tests for autocast.scripts.setup module."""

import pytest
import torch
from autosim.cli import compute_normalization_stats, save_normalization_stats
from omegaconf import OmegaConf
from the_well.data.normalization import ZScoreNormalization

from autocast.encoders.base import Encoder
from autocast.losses import MCDropoutMSEL2Loss
from autocast.scripts.setup import (
    _apply_processor_channel_defaults,
    _build_loss_func,
    _build_processor,
    _filter_kwargs_for_target,
    _get_latent_channels,
    _get_module_device,
    _resolve_module_device,
    _resolve_processor_temporal_steps,
    _set_if_auto,
    resolve_auto_params,
    setup_autoencoder_components,
    setup_processor_model,
)
from autocast.types import Batch

# --- _set_if_auto ---


def test_set_if_auto_sets_when_none():
    cfg = OmegaConf.create({"key": None})
    _set_if_auto(cfg, "key", 42)
    assert cfg["key"] == 42


def test_set_if_auto_sets_when_auto():
    cfg = OmegaConf.create({"key": "auto"})
    _set_if_auto(cfg, "key", 42)
    assert cfg["key"] == 42


def test_set_if_auto_preserves_explicit_value():
    cfg = OmegaConf.create({"key": 10})
    _set_if_auto(cfg, "key", 42)
    assert cfg["key"] == 10


def test_set_if_auto_ignores_missing_key():
    cfg = OmegaConf.create({"other": "auto"})
    _set_if_auto(cfg, "key", 42)
    assert "key" not in cfg


def test_get_module_device_prefers_parameter_device():
    module = torch.nn.Linear(4, 2)
    device = _get_module_device(module)
    assert device == next(module.parameters()).device


def test_get_module_device_uses_buffer_when_no_parameters():
    class BufferOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.ones(1))

    module = BufferOnly()
    device = _get_module_device(module)
    assert device == module.state.device


def test_get_module_device_returns_none_for_parameterless_bufferless_module():
    module = torch.nn.Identity()
    assert _get_module_device(module) is None


def test_resolve_module_device_defaults_to_cpu_for_parameterless_modules():
    module_a = torch.nn.Identity()
    module_b = torch.nn.Identity()
    assert _resolve_module_device(module_a, module_b) == torch.device("cpu")


# --- _filter_kwargs_for_target ---


def test_filter_kwargs_returns_all_when_target_none():
    kwargs = {"a": 1, "b": 2}
    result = _filter_kwargs_for_target(None, kwargs)
    assert result == kwargs


def test_filter_kwargs_filters_to_allowed_params():
    kwargs = {"in_features": 10, "out_features": 5, "extra": "ignored"}
    result = _filter_kwargs_for_target("torch.nn.Linear", kwargs)
    assert "extra" not in result
    assert result["in_features"] == 10


def test_filter_kwargs_handles_invalid_target():
    kwargs = {"a": 1}
    result = _filter_kwargs_for_target("nonexistent.Class", kwargs)
    assert result == kwargs


# --- _apply_processor_channel_defaults ---


def test_apply_processor_defaults_to_auto_values():
    cfg = OmegaConf.create(
        {"in_channels": "auto", "out_channels": None, "n_steps_input": 4}
    )
    _apply_processor_channel_defaults(
        cfg,
        in_channels=8,
        out_channels=16,
        n_steps_input=2,
        n_steps_output=2,
        n_channels_out=16,
    )
    assert cfg["in_channels"] == 8
    assert cfg["out_channels"] == 16
    assert cfg["n_steps_input"] == 4  # Was explicit, not changed


def test_apply_processor_defaults_to_backbone():
    cfg = OmegaConf.create({"backbone": {"in_channels": "auto", "cond_channels": None}})
    _apply_processor_channel_defaults(
        cfg,
        in_channels=8,
        out_channels=16,
        n_steps_input=2,
        n_steps_output=2,
        n_channels_out=16,
    )
    assert cfg["backbone"]["in_channels"] == 16  # out_channels for backbone
    assert cfg["backbone"]["cond_channels"] == 8  # in_channels for backbone


def test_apply_processor_handles_none_config():
    # Should not raise
    _apply_processor_channel_defaults(
        None,
        in_channels=8,
        out_channels=16,
        n_steps_input=2,
        n_steps_output=2,
        n_channels_out=16,
    )


def test_build_processor_prefers_explicit_processor_step_counts():
    model_cfg = OmegaConf.create(
        {
            "processor": {
                "_target_": "autocast.processors.vit_latent.AViTLatentProcessor",
                "global_cond_channels": 0,
                "include_global_cond": False,
                "n_steps_input": 7,
                "n_steps_output": 9,
                "hidden_dim": 8,
                "num_heads": 2,
                "n_layers": 1,
            }
        }
    )
    proc_kwargs = {
        "in_channels": 3,
        "out_channels": 3,
        "n_steps_input": 2,
        "n_steps_output": 4,
        "n_channels_out": 3,
        "spatial_resolution": (8, 8),
    }

    processor = _build_processor(model_cfg, proc_kwargs)

    assert processor.n_steps_input == 7
    assert processor.n_steps_output == 9


def test_build_processor_uses_inferred_steps_when_processor_steps_not_explicit():
    model_cfg = OmegaConf.create(
        {
            "processor": {
                "_target_": "autocast.processors.vit_latent.AViTLatentProcessor",
                "global_cond_channels": 0,
                "include_global_cond": False,
                "n_steps_input": "auto",
                "n_steps_output": None,
                "hidden_dim": 8,
                "num_heads": 2,
                "n_layers": 1,
            }
        }
    )
    proc_kwargs = {
        "in_channels": 3,
        "out_channels": 3,
        "n_steps_input": 2,
        "n_steps_output": 4,
        "n_channels_out": 3,
        "spatial_resolution": (8, 8),
    }

    processor = _build_processor(model_cfg, proc_kwargs)

    assert processor.n_steps_input == 2
    assert processor.n_steps_output == 4


def test_resolve_processor_temporal_steps_for_time_concatenating_encoder():
    class TimeConcatEncoder:
        outputs_time_channel_concat = True

    steps_in, steps_out = _resolve_processor_temporal_steps(
        TimeConcatEncoder(),  # type: ignore - just testing resolution logic, not actual encoder behavior
        n_steps_input=4,
        n_steps_output=2,
    )

    assert steps_in == 1
    assert steps_out == 1


def test_resolve_processor_temporal_steps_for_regular_encoder():
    class RegularEncoder:
        outputs_time_channel_concat = False

    steps_in, steps_out = _resolve_processor_temporal_steps(
        RegularEncoder(),  # type: ignore - just testing resolution logic, not actual encoder behavior
        n_steps_input=4,
        n_steps_output=2,
    )

    assert steps_in == 4
    assert steps_out == 2


def test_resolve_processor_temporal_steps_defaults_to_non_concat_when_missing_attr():
    class EncoderWithoutFlag:
        pass

    steps_in, steps_out = _resolve_processor_temporal_steps(
        EncoderWithoutFlag(),  # type: ignore - just testing resolution logic, not actual encoder behavior
        n_steps_input=4,
        n_steps_output=2,
    )

    assert steps_in == 4
    assert steps_out == 2


# --- resolve_auto_params ---


def test_resolve_auto_params_resolves_steps():
    cfg = OmegaConf.create(
        {"datamodule": {"n_steps_input": "auto", "n_steps_output": "auto"}}
    )
    input_shape = (2, 4, 8, 8, 3)  # B, T, W, H, C
    output_shape = (2, 2, 8, 8, 3)
    result = resolve_auto_params(cfg, input_shape, output_shape)
    assert result.datamodule.n_steps_input == 4
    assert result.datamodule.n_steps_output == 2


def test_resolve_auto_params_resolves_stride():
    cfg = OmegaConf.create({"datamodule": {"n_steps_output": 2, "stride": "auto"}})
    result = resolve_auto_params(cfg, (2, 4, 8, 8, 3), (2, 2, 8, 8, 3))
    assert result.datamodule.stride == 2


def test_resolve_auto_params_unchanged_if_no_datamodule():
    cfg = OmegaConf.create({"other": "value"})
    result = resolve_auto_params(cfg, (2, 4, 8, 8, 3), (2, 2, 8, 8, 3))
    assert result == cfg


# --- _build_loss_func ---


def test_build_loss_func_returns_none_when_unspecified():
    """No loss in the config means no cross-cutting loss, not a silent MSE.

    Processors that own their objective in ``processor.loss()`` are configured
    with ``loss_func: null``. ``ProcessorModelEnsemble.loss`` gates on
    ``loss_func is None`` to reach that path, so a default here would send them
    through the cross-cutting loss instead.
    """
    assert _build_loss_func(OmegaConf.create({})) is None
    assert _build_loss_func(OmegaConf.create({"loss_func": None})) is None


def test_build_loss_func_instantiates_from_config():
    cfg = OmegaConf.create({"loss_func": {"_target_": "torch.nn.L1Loss"}})
    loss = _build_loss_func(cfg)
    assert isinstance(loss, torch.nn.L1Loss)


def test_build_loss_func_injects_processor_when_requested():
    processor = torch.nn.Linear(2, 2)
    cfg = OmegaConf.create(
        {
            "loss_func": {
                "_target_": "autocast.losses.MCDropoutMSEL2Loss",
                "l2_coefficient": 1e-5,
            }
        }
    )

    loss = _build_loss_func(cfg, processor)

    assert isinstance(loss, MCDropoutMSEL2Loss)
    assert loss.processor is processor
    assert loss.l2_coefficient == pytest.approx(1e-5)


def test_build_loss_func_rejects_missing_required_processor():
    cfg = OmegaConf.create(
        {
            "loss_func": {
                "_target_": "autocast.losses.MCDropoutMSEL2Loss",
                "l2_coefficient": 1e-5,
            }
        }
    )

    with pytest.raises(ValueError, match="requires a processor-bound EPD setup"):
        _build_loss_func(cfg)


# --- _infer_latent_channels ---


def _make_mock_encoder(output_shape: tuple[int, ...]) -> Encoder:
    """Create a mock encoder returning a specific output shape."""

    class MockEncoder(Encoder):
        latent_channels = 8
        encoder_model = torch.nn.Identity()

        def encode(self, batch: Batch) -> torch.Tensor:  # noqa: ARG002
            return torch.randn(*output_shape)

    return MockEncoder()


def _make_batch(shape: tuple[int, ...] = (2, 2, 16, 16, 4)) -> Batch:
    return Batch(
        input_fields=torch.randn(*shape),
        output_fields=torch.randn(*shape),
        constant_scalars=None,
        constant_fields=None,
    )


def test_get_latent_channels_from_encoder():
    encoder = _make_mock_encoder((2, 2, 4, 4, 8))
    channels = _get_latent_channels(encoder)
    assert channels == 8


def test_get_latent_channels_requires_attribute():
    class BrokenEncoder(Encoder):
        encoder_model = torch.nn.Identity()

        def encode(self, batch: Batch) -> torch.Tensor:  # noqa: ARG002
            return torch.randn(2, 2, 4, 4, 8)

    encoder = BrokenEncoder()
    with pytest.raises(ValueError, match="must set latent_channels"):
        _get_latent_channels(encoder)


def test_setup_autoencoder_components_resolves_decoder_in_channels_auto():
    cfg = OmegaConf.create(
        {
            "model": {
                "encoder": {
                    "_target_": "autocast.encoders.identity.IdentityEncoder",
                    "in_channels": "auto",
                },
                "decoder": {
                    "_target_": "autocast.decoders.identity.IdentityDecoder",
                    "in_channels": "auto",
                },
            },
            "autoencoder_checkpoint": None,
        }
    )
    stats = {"channel_count": 3, "n_steps_input": 2, "n_steps_output": 2}

    encoder, decoder = setup_autoencoder_components(cfg, stats)

    assert encoder.latent_channels == 3
    assert decoder.latent_channels == 3


def _state_normalizer(fields=("u", "v")):
    stats = {
        name: dict.fromkeys(fields, value)
        for name, value in {
            "mean": 10.0,
            "std": 2.0,
            "mean_delta": 2.0,
            "std_delta": 4.0,
        }.items()
    }
    return ZScoreNormalization(stats, list(fields), [])


@pytest.mark.parametrize("n_members", [1, 3])
@pytest.mark.parametrize("use_normalization", [False, True])
@pytest.mark.parametrize(
    ("options", "expected_values"),
    [
        ({}, [3.0, 3.0]),
        ({"residual_prediction": True}, [6.0, 10.0]),
        ({"residual_prediction": True, "residual_use_delta_stats": False}, [6.0, 6.0]),
        ({"residual_prediction": False, "residual_use_delta_stats": False}, [3.0, 3.0]),
    ],
)
def test_setup_epd_residual_defaults_and_options(
    tmp_path,
    epd_config,
    setup_epd,
    dummy_datamodule,
    make_toy_batch,
    n_members,
    use_normalization,
    options,
    expected_values,
):
    epd_config.model.update(n_members=n_members, **options)
    if (
        not options.get("residual_prediction")
        or options.get("residual_use_delta_stats") is False
    ):
        epd_config.model.residual_delta_stats_path = str(tmp_path / "missing.yml")
    norm = _state_normalizer() if use_normalization else None
    dummy_datamodule.train_dataset.norm = norm
    batch = make_toy_batch(t_in=2, t_out=2, w=2, h=2, c=2)
    batch.input_fields.fill_(3.0)
    model = setup_epd(dummy_datamodule, batch)
    shape = batch.output_fields.shape + ((n_members,) if n_members > 1 else ())
    torch.testing.assert_close(
        model(batch), torch.full(shape, expected_values[use_normalization])
    )
    assert model.norm is norm


@pytest.mark.parametrize("use_normalization", [False, True])
@pytest.mark.parametrize("channels", [None, (0,), (1,), (1, 0)])
def test_setup_epd_autosim_delta_file_and_channel_selection(
    tmp_path,
    epd_config,
    setup_epd,
    dummy_datamodule,
    make_toy_batch,
    use_normalization,
    channels,
):
    # AutoSim's adjacent-frame deltas are [-2, 6] and [1, 5]; mean + std = [6, 5].
    payload = {
        "data": torch.tensor([[0.0, 0.0], [-2.0, 1.0], [4.0, 6.0]]).reshape(
            1, 3, 1, 1, 2
        )
    }
    path = tmp_path / "stats.yml"
    save_normalization_stats(
        compute_normalization_stats(payload, core_field_names=["u", "v"]), path
    )
    selected = channels or (0, 1)
    fields = [["u", "v"][index] for index in selected]
    dummy_datamodule.train_dataset.channel_idxs = channels
    dummy_datamodule.train_dataset.norm = (
        _state_normalizer(fields) if use_normalization else None
    )
    epd_config.model.update(
        residual_prediction=True, residual_delta_stats_path=str(path)
    )
    batch = make_toy_batch(t_in=2, t_out=2, w=2, h=2, c=len(selected))
    batch.input_fields.fill_(1.0)
    model = setup_epd(dummy_datamodule, batch)
    increment = torch.tensor([6.0, 5.0])[list(selected)]
    expected = 1 + increment / (2 if use_normalization else 1)
    torch.testing.assert_close(model(batch), expected.expand_as(batch.output_fields))


def test_setup_processor_rejects_residual_prediction_before_building_latents(
    dummy_datamodule,
):
    cfg = OmegaConf.create({"model": {"residual_prediction": True}})

    with pytest.raises(ValueError, match="requires a full EncoderProcessorDecoder"):
        setup_processor_model(cfg, {}, dummy_datamodule)
