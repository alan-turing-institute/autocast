"""Integration tests for scripts using actual configs from src/autocast/configs/."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from autocast.scripts.setup import (
    _apply_processor_channel_defaults,
    _build_loss_func,
    _get_normalized_processor_config,
    resolve_auto_params,
)

# --- Fixtures ---


@pytest.fixture
def config_dir(REPO_ROOT: Path) -> str:
    """Path to the configs directory."""
    return str(REPO_ROOT / "src" / "autocast" / "configs")


def _load_config(
    config_dir: str, config_name: str, overrides: list[str] | None = None
) -> DictConfig:
    """Load a config by name."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name, overrides=overrides or [])


# --- Parametrized tests over top-level configs ---

TOP_LEVEL_CONFIGS = ["autoencoder", "encoder_processor_decoder", "processor"]


@pytest.mark.parametrize("config_name", TOP_LEVEL_CONFIGS)
def test_top_level_config_loads(config_dir: str, config_name: str):
    """Verify top-level configs load without errors."""
    cfg = _load_config(config_dir, config_name)
    assert cfg is not None
    assert "datamodule" in cfg
    assert "model" in cfg


@pytest.mark.parametrize("config_name", TOP_LEVEL_CONFIGS)
def test_top_level_config_has_trainer(config_dir: str, config_name: str):
    """Verify top-level configs have trainer section."""
    cfg = _load_config(config_dir, config_name)
    assert "trainer" in cfg


def test_multinode_distributed_config_sets_trainer_nodes(config_dir: str):
    cfg = _load_config(
        config_dir,
        "encoder_processor_decoder",
        overrides=["+distributed=ddp_4gpu_2node_slurm"],
    )

    assert cfg.trainer.devices == 4
    assert cfg.trainer.num_nodes == 2
    assert cfg.trainer.strategy == "ddp"


# --- Parametrized tests over component configs ---


def _glob_config_names(config_dir: str, subdir: str) -> list[str]:
    """Get list of config names from a subdirectory."""
    path = Path(config_dir) / subdir
    if not path.exists():
        return []
    return [f.stem for f in path.glob("*.yaml")]


@pytest.fixture
def encoder_configs(config_dir: str) -> list[str]:
    return _glob_config_names(config_dir, "encoder")


@pytest.fixture
def decoder_configs(config_dir: str) -> list[str]:
    return _glob_config_names(config_dir, "decoder")


@pytest.fixture
def processor_configs(config_dir: str) -> list[str]:
    return _glob_config_names(config_dir, "processor")


def test_encoder_configs_exist(encoder_configs: list[str]):
    """Verify encoder configs are found."""
    assert len(encoder_configs) > 0
    assert "dc" in encoder_configs or "identity" in encoder_configs


def test_decoder_configs_exist(decoder_configs: list[str]):
    """Verify decoder configs are found."""
    assert len(decoder_configs) > 0
    assert "dc" in decoder_configs or "identity" in decoder_configs


def test_processor_configs_exist(processor_configs: list[str]):
    """Verify processor configs are found."""
    assert len(processor_configs) > 0
    assert "flow_matching" in processor_configs or "fno" in processor_configs


# --- Tests using real configs ---


def test_extract_optimizer_returns_empty_when_missing(config_dir: str):
    cfg = _load_config(config_dir, "autoencoder")
    # Check that optimizer config exists
    assert "optimizer" in cfg
    assert cfg.optimizer is not None


def test_get_normalized_processor_config_from_epd(config_dir: str):
    cfg = _load_config(config_dir, "encoder_processor_decoder")
    proc_config = _get_normalized_processor_config(cfg.model)
    assert proc_config is not None
    assert "_target_" in proc_config


def test_apply_defaults_preserves_explicit_values(config_dir: str):
    cfg = _load_config(config_dir, "encoder_processor_decoder")
    proc_config = _get_normalized_processor_config(cfg.model)
    assert proc_config is not None

    # Store original explicit values
    original_values = {k: v for k, v in proc_config.items() if v not in (None, "auto")}

    _apply_processor_channel_defaults(
        proc_config,
        in_channels=999,
        out_channels=999,
        n_steps_input=999,
        n_steps_output=999,
        n_channels_out=999,
    )

    # Explicit values should be preserved
    for key, original in original_values.items():
        if key in proc_config:
            assert proc_config[key] == original


def test_resolve_auto_params_with_real_config(config_dir: str):
    cfg = _load_config(config_dir, "encoder_processor_decoder")
    input_shape = (2, 4, 32, 32, 2)
    output_shape = (2, 4, 32, 32, 2)

    # Make a mutable copy and set auto values
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    cfg_copy = OmegaConf.create(container)
    assert isinstance(cfg_copy, DictConfig)
    if "datamodule" in cfg_copy:
        cfg_copy.datamodule.n_steps_input = "auto"  # type: ignore[assignment]
        cfg_copy.datamodule.n_steps_output = "auto"  # type: ignore[assignment]

    resolved = resolve_auto_params(cfg_copy, input_shape, output_shape)

    if "datamodule" in resolved:
        assert resolved.datamodule.n_steps_input == 4
        assert resolved.datamodule.n_steps_output == 4


def test_build_loss_func_with_real_config(config_dir: str):
    cfg = _load_config(config_dir, "encoder_processor_decoder")
    loss = _build_loss_func(cfg.model)
    # Should return a valid nn.Module
    assert hasattr(loss, "forward")


def test_swe_crps_vit_concat_experiment_config(config_dir: str):
    cfg = _load_config(
        config_dir,
        "encoder_processor_decoder",
        overrides=["experiment=epd_crps_vit_concat_32"],
    )

    assert cfg.datamodule.n_steps_input == 1
    assert cfg.datamodule.n_steps_output == 1
    assert cfg.datamodule.stride == 1
    assert cfg.datamodule.use_normalization is True
    assert cfg.model.n_members == 4
    assert cfg.model.processor.spatial_resolution == [32, 32]
    assert cfg.model.processor.patch_size == 1
    assert cfg.model.processor.hidden_dim == 64
    assert cfg.model.processor.n_layers == 4
    assert cfg.model.processor.n_noise_channels is None
    assert cfg.model.input_noise_injector.n_channels == 1
    assert cfg.trainer.max_epochs == 50
    assert cfg.trainer.max_steps == 1024
    assert cfg.trainer.limit_val_batches == 4
    assert cfg.trainer.num_sanity_val_steps == 0
    assert all(
        callback._target_ != "autocast.callbacks.ema.EMACallback"
        for callback in cfg.trainer.callbacks
    )


def test_swe_crps_vit_adaln_experiment_matches_concat(config_dir: str):
    concat_cfg = _load_config(
        config_dir,
        "encoder_processor_decoder",
        overrides=["experiment=epd_crps_vit_concat_32"],
    )
    adaln_cfg = _load_config(
        config_dir,
        "encoder_processor_decoder",
        overrides=["experiment=epd_crps_vit_adaln_32"],
    )

    backbone_keys = [
        "spatial_resolution",
        "patch_size",
        "hidden_dim",
        "num_heads",
        "n_layers",
    ]
    for key in backbone_keys:
        assert concat_cfg.model.processor[key] == adaln_cfg.model.processor[key]
    assert concat_cfg.model.n_members == adaln_cfg.model.n_members == 4
    assert concat_cfg.model.input_noise_injector.n_channels == 1
    assert concat_cfg.model.processor.n_noise_channels is None
    assert adaln_cfg.model.get("input_noise_injector") is None
    assert adaln_cfg.model.processor.n_noise_channels == 16
    assert adaln_cfg.trainer.max_steps == concat_cfg.trainer.max_steps == 1024
    assert adaln_cfg.trainer.limit_val_batches == 4
    assert all(
        callback._target_ != "autocast.callbacks.ema.EMACallback"
        for callback in adaln_cfg.trainer.callbacks
    )


def test_swe_crps_vit_residual_experiments_match(config_dir: str):
    configs = [
        _load_config(
            config_dir,
            "encoder_processor_decoder",
            overrides=[f"experiment={experiment}"],
        )
        for experiment in (
            "epd_crps_vit_concat_residual_32",
            "epd_crps_vit_adaln_residual_32",
        )
    ]

    for cfg in configs:
        assert cfg.model.residual_prediction is True
        assert cfg.model.residual_use_delta_stats is True
        assert cfg.model.processor.zero_init_output is True
        assert cfg.model.n_members == 4
        assert cfg.trainer.max_steps == 1024
    assert configs[0].model.input_noise_injector.n_channels == 1
    assert configs[0].model.processor.n_noise_channels is None
    assert configs[1].model.get("input_noise_injector") is None
    assert configs[1].model.processor.n_noise_channels == 16


def test_swe_crps_fno_concat_experiment_config(config_dir: str):
    cfg = _load_config(
        config_dir,
        "encoder_processor_decoder",
        overrides=["experiment=epd_crps_fno_concat_32"],
    )

    assert cfg.datamodule.n_steps_input == 1
    assert cfg.datamodule.n_steps_output == 1
    assert cfg.datamodule.stride == 1
    assert cfg.datamodule.use_normalization is True
    assert cfg.model.n_members == 8
    assert cfg.model.processor.n_modes == [12, 12]
    assert cfg.model.processor.hidden_channels == 32
    assert cfg.model.processor.n_layers == 3
    assert cfg.model.input_noise_injector.n_channels == 1
    assert cfg.trainer.max_epochs == 50
    assert all(
        callback._target_ != "autocast.callbacks.ema.EMACallback"
        for callback in cfg.trainer.callbacks
    )
