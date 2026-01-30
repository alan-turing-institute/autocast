"""Unit tests for autocast.scripts.setup module."""

import torch
from omegaconf import OmegaConf

from autocast.encoders.base import Encoder
from autocast.scripts.setup import (
    _apply_processor_channel_defaults,
    _build_loss_func,
    _extract_config_dict,
    _filter_kwargs_for_target,
    _infer_latent_channels,
    _set_if_auto,
    resolve_auto_params,
)
from autocast.types import Batch

# --- _set_if_auto ---


def test_set_if_auto_sets_when_none():
    cfg = {"key": None}
    _set_if_auto(cfg, "key", 42)
    assert cfg["key"] == 42


def test_set_if_auto_sets_when_auto():
    cfg = {"key": "auto"}
    _set_if_auto(cfg, "key", 42)
    assert cfg["key"] == 42


def test_set_if_auto_preserves_explicit_value():
    cfg = {"key": 10}
    _set_if_auto(cfg, "key", 42)
    assert cfg["key"] == 10


def test_set_if_auto_ignores_missing_key():
    cfg = {"other": "auto"}
    _set_if_auto(cfg, "key", 42)
    assert "key" not in cfg


# --- _extract_config_dict ---


def test_extract_config_dict_extracts_subconfig():
    cfg = OmegaConf.create({"sub": {"a": 1, "b": 2}})
    result = _extract_config_dict(cfg, "sub")
    assert result == {"a": 1, "b": 2}


def test_extract_config_dict_returns_default_when_missing():
    cfg = OmegaConf.create({})
    result = _extract_config_dict(cfg, "missing", {"default": True})
    assert result == {"default": True}


def test_extract_config_dict_returns_empty_dict_when_no_default():
    cfg = OmegaConf.create({})
    result = _extract_config_dict(cfg, "missing")
    assert result == {}


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
    cfg = {"in_channels": "auto", "out_channels": None, "n_steps_input": 4}
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
    cfg = {"backbone": {"in_channels": "auto", "cond_channels": None}}
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


def test_build_loss_func_defaults_to_mse():
    loss = _build_loss_func({})
    assert isinstance(loss, torch.nn.MSELoss)


def test_build_loss_func_instantiates_from_config():
    cfg = {"loss_func": {"_target_": "torch.nn.L1Loss"}}
    loss = _build_loss_func(cfg)
    assert isinstance(loss, torch.nn.L1Loss)


# --- _infer_latent_channels ---


def _make_mock_encoder(output_shape: tuple[int, ...]) -> Encoder:
    """Create a mock encoder returning a specific output shape."""

    class MockEncoder(Encoder):
        latent_dim = 8
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


def test_infer_latent_channels_from_encoder():
    encoder = _make_mock_encoder((2, 2, 4, 4, 8))
    channels, concat = _infer_latent_channels(encoder, _make_batch())
    assert channels == 8
    assert concat is False  # Same ndim


def test_infer_latent_channels_detects_time_concat():
    encoder = _make_mock_encoder((2, 8, 4, 4))  # 4D (time bundled)
    channels, concat = _infer_latent_channels(encoder, _make_batch())
    assert channels == 4  # Last dim
    assert concat is True  # Lower ndim


def test_infer_latent_channels_falls_back_on_error():
    class BrokenEncoder(Encoder):
        latent_dim = 8
        encoder_model = torch.nn.Identity()

        def encode(self, batch: Batch) -> torch.Tensor:  # noqa: ARG002
            msg = "Broken"
            raise RuntimeError(msg)

    encoder = BrokenEncoder()
    channels, concat = _infer_latent_channels(encoder, _make_batch())
    assert channels == 4  # Falls back to input channels
    assert concat is False
