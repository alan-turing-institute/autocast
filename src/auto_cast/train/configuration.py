"""Shared utilities for Hydra-based training/evaluation scripts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, open_dict

from auto_cast.train.autoencoder import build_datamodule


@dataclass
class TrainingParams:
    """Resolved training knobs that may come from config or CLI overrides."""

    n_steps_input: int
    n_steps_output: int
    autoencoder_checkpoint: Path | None
    freeze_autoencoder: bool


def compose_training_config(args) -> DictConfig:
    """Compose the Hydra config prior to applying CLI-specific overrides."""
    config_dir = args.config_dir.resolve()
    overrides: Sequence[str] = args.overrides or []
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=args.config_name, overrides=list(overrides))


def _get_field(batch, primary: str, fallback: str):
    return (
        getattr(batch, primary) if hasattr(batch, primary) else getattr(batch, fallback)
    )


def prepare_datamodule(cfg: DictConfig):
    """Instantiate the datamodule and inspect the first batch for sizing."""
    datamodule = build_datamodule(cfg.data)
    batch = next(iter(datamodule.train_dataloader()))
    train_inputs = _get_field(batch, "inputs", "input_fields")
    train_outputs = _get_field(batch, "outputs", "output_fields")
    channel_count = train_inputs.shape[-1]
    inferred_n_steps_input = train_inputs.shape[1]
    inferred_n_steps_output = train_outputs.shape[1]
    return (
        datamodule,
        channel_count,
        inferred_n_steps_input,
        inferred_n_steps_output,
        train_inputs.shape,
        train_outputs.shape,
    )


def _maybe_set(cfg_node: DictConfig | None, key: str, value: int) -> None:
    if cfg_node is None or key not in cfg_node:
        return
    current = cfg_node.get(key)
    if current not in (None, "auto"):
        return
    with open_dict(cfg_node):
        cfg_node[key] = value


def configure_module_dimensions(
    cfg: DictConfig,
    channel_count: int,
    n_steps_input: int,
    n_steps_output: int,
) -> None:
    """Populate missing dimension hints for encoder/decoder/processor modules."""
    _maybe_set(cfg.decoder, "output_channels", channel_count)
    _maybe_set(cfg.decoder, "time_steps", n_steps_output)
    _maybe_set(cfg.processor, "in_channels", channel_count * n_steps_input)
    _maybe_set(cfg.processor, "out_channels", channel_count * n_steps_output)


def normalize_processor_cfg(cfg: DictConfig) -> None:
    """Force config values into the shapes expected by processor classes."""
    processor_cfg = cfg.get("processor")
    if processor_cfg is None:
        return
    tuple_fields = ("n_modes",)
    for field in tuple_fields:
        value = processor_cfg.get(field)
        if isinstance(value, ListConfig):
            with open_dict(processor_cfg):
                processor_cfg[field] = tuple(value)


def update_data_cfg(cfg: DictConfig, n_steps_input: int, n_steps_output: int) -> None:
    """Update datamodule configuration to match resolved training step counts."""
    data_cfg = cfg.data
    with open_dict(data_cfg.datamodule):
        data_cfg.datamodule.n_steps_input = n_steps_input
        data_cfg.datamodule.n_steps_output = n_steps_output
        data_cfg.datamodule.autoencoder_mode = False


def resolve_training_params(cfg: DictConfig, args) -> TrainingParams:
    """Resolve training hyperparameters using the config plus CLI overrides."""
    training_cfg = cfg.get("training")
    n_steps_input_cfg = (
        training_cfg.get("n_steps_input", 1) if training_cfg is not None else 1
    )
    n_steps_output_cfg = (
        training_cfg.get("n_steps_output", 1) if training_cfg is not None else 1
    )
    ckpt_cfg = (
        training_cfg.get("autoencoder_checkpoint") if training_cfg is not None else None
    )
    freeze_cfg = (
        training_cfg.get("freeze_autoencoder", False)
        if training_cfg is not None
        else False
    )

    n_steps_input = args.n_steps_input or n_steps_input_cfg
    n_steps_output = args.n_steps_output or n_steps_output_cfg

    checkpoint = args.autoencoder_checkpoint
    if checkpoint is None and ckpt_cfg is not None:
        checkpoint = Path(ckpt_cfg)

    freeze_autoencoder = (
        args.freeze_autoencoder if args.freeze_autoencoder is not None else freeze_cfg
    )

    if n_steps_output < 1:
        msg = "n_steps_output must be >= 1 for processor training."
        raise ValueError(msg)

    return TrainingParams(
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        autoencoder_checkpoint=checkpoint,
        freeze_autoencoder=freeze_autoencoder,
    )


__all__ = [
    "TrainingParams",
    "compose_training_config",
    "configure_module_dimensions",
    "normalize_processor_cfg",
    "prepare_datamodule",
    "resolve_training_params",
    "update_data_cfg",
]
