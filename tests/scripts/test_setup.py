"""Unit tests for autocast.scripts.setup module."""

import pytest
import torch
from omegaconf import OmegaConf

from autocast.encoders.base import Encoder
from autocast.scripts.setup import (
    _apply_processor_channel_defaults,
    _build_loss_func,
    _filter_kwargs_for_target,
    _get_latent_channels,
    _set_if_auto,
    resolve_auto_params,
    setup_autoencoder_components,
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
    loss = _build_loss_func(OmegaConf.create({}))
    assert isinstance(loss, torch.nn.MSELoss)


def test_build_loss_func_instantiates_from_config():
    cfg = OmegaConf.create({"loss_func": {"_target_": "torch.nn.L1Loss"}})
    loss = _build_loss_func(cfg)
    assert isinstance(loss, torch.nn.L1Loss)


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
