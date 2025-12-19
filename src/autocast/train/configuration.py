"""Shared utilities for Hydra-based training/evaluation scripts."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, open_dict

from autocast.train.autoencoder import build_datamodule
from autocast.types import Batch

log = logging.getLogger(__name__)


@dataclass
class TrainingParams:
    """Resolved training knobs that may come from config or CLI overrides."""

    n_steps_input: int
    n_steps_output: int
    stride: int
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
    constant_scalars = getattr(batch, "constant_scalars", None)
    constant_fields = getattr(batch, "constant_fields", None)
    sample_batch = Batch(
        input_fields=train_inputs,
        output_fields=train_outputs,
        constant_scalars=constant_scalars,
        constant_fields=constant_fields,
    )
    return (
        datamodule,
        channel_count,
        inferred_n_steps_input,
        inferred_n_steps_output,
        train_inputs.shape,
        train_outputs.shape,
        sample_batch,
    )


def _infer_encoder_latent_channels(
    encoder, fallback_channels: int, example_batch: Batch | None = None
) -> int:
    latent_dim = getattr(encoder, "latent_dim", None)
    if isinstance(latent_dim, int) and latent_dim > 0:
        return latent_dim
    if example_batch is None or not hasattr(encoder, "encode"):
        return fallback_channels

    prev_training = getattr(encoder, "training", None)
    if prev_training is not None:
        encoder.eval()
    try:
        with torch.no_grad():
            encoded = encoder.encode(example_batch)  # type: ignore[attr-defined]
        inferred = int(encoded.shape[-1])
        log.debug("Inferred latent channel count=%s from sample batch", inferred)
        return inferred
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("Failed to infer latent channels via encode(): %s", exc)
        return fallback_channels
    finally:
        if prev_training is not None:
            encoder.train(prev_training)


def _override_dimension(
    cfg_node: DictConfig | None,
    key: str,
    value: int,
    fallback_values: tuple[int | None | str, ...] = (),
) -> None:
    if cfg_node is None or key not in cfg_node:
        return
    current = cfg_node.get(key)
    allowed = set(fallback_values)
    allowed.update({None, "auto"})
    if current in allowed:
        cfg_node[key] = value


def align_processor_channels_with_encoder(
    cfg: DictConfig,
    *,
    encoder,
    channel_count: int,
    n_steps_input: int,
    n_steps_output: int,
    example_batch: Batch | None = None,
) -> int:
    """Align processor/backbone channels with the encoder latent dimensionality."""
    latent_channels = _infer_encoder_latent_channels(
        encoder,
        fallback_channels=channel_count,
        example_batch=example_batch,
    )

    processor_cfg = _model_cfg(cfg).get("processor")
    if processor_cfg is None:
        return latent_channels

    raw_in = channel_count * n_steps_input
    raw_out = channel_count * n_steps_output
    latent_in = latent_channels * n_steps_input
    latent_out = latent_channels * n_steps_output

    with open_dict(processor_cfg):
        _override_dimension(
            processor_cfg,
            "n_channels_out",
            latent_channels,
            (channel_count,),
        )
        _override_dimension(
            processor_cfg,
            "out_channels",
            latent_out,
            (raw_out,),
        )
        _override_dimension(
            processor_cfg,
            "in_channels",
            latent_in,
            (raw_in,),
        )

    backbone_cfg = processor_cfg.get("backbone")
    if backbone_cfg is not None:
        with open_dict(backbone_cfg):
            _override_dimension(
                backbone_cfg,
                "in_channels",
                latent_channels,
                (channel_count,),
            )
            _override_dimension(
                backbone_cfg,
                "out_channels",
                latent_channels,
                (channel_count,),
            )
            _override_dimension(
                backbone_cfg,
                "cond_channels",
                latent_channels,
                (channel_count,),
            )
            _override_dimension(
                backbone_cfg,
                "n_steps_input",
                n_steps_input,
                (),
            )
            _override_dimension(
                backbone_cfg,
                "n_steps_output",
                n_steps_output,
                (),
            )

    return latent_channels


def _maybe_set(cfg_node: DictConfig | None, key: str, value: int) -> None:
    if cfg_node is None or key not in cfg_node:
        return
    current = cfg_node.get(key)
    if current not in (None, "auto"):
        return
    with open_dict(cfg_node):
        cfg_node[key] = value


def _model_cfg(cfg: DictConfig) -> DictConfig:
    """Return the nested model config when present, else the root config."""
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, DictConfig):
        return model_cfg
    return cfg


def configure_module_dimensions(
    cfg: DictConfig,
    channel_count: int,
    n_steps_input: int,
    n_steps_output: int,
) -> None:
    """Populate missing dimension hints for encoder/decoder/processor modules.

    Note: Backbone channel dimensions are handled separately by
    align_processor_channels_with_encoder, which uses the encoder's latent
    dimension rather than raw channel counts.
    """
    model_cfg = _model_cfg(cfg)
    encoder_cfg = model_cfg.get("encoder")
    _maybe_set(encoder_cfg, "in_channels", channel_count)
    _maybe_set(encoder_cfg, "time_steps", n_steps_input)
    decoder_cfg = model_cfg.get("decoder")
    _maybe_set(decoder_cfg, "out_channels", channel_count)
    _maybe_set(decoder_cfg, "output_channels", channel_count)  # alias
    _maybe_set(decoder_cfg, "time_steps", n_steps_output)
    processor_cfg = model_cfg.get("processor")
    _maybe_set(processor_cfg, "in_channels", channel_count * n_steps_input)
    _maybe_set(processor_cfg, "out_channels", channel_count * n_steps_output)
    _maybe_set(processor_cfg, "n_steps_output", n_steps_output)
    _maybe_set(processor_cfg, "n_channels_out", channel_count)


def normalize_processor_cfg(cfg: DictConfig) -> None:
    """Force config values into the shapes expected by processor classes."""
    processor_cfg = _model_cfg(cfg).get("processor")
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
    # Handle both nested datamodule structure and flat structure (e.g., the_well.yaml)
    if "datamodule" in data_cfg:
        with open_dict(data_cfg.datamodule):
            data_cfg.datamodule.n_steps_input = n_steps_input
            data_cfg.datamodule.n_steps_output = n_steps_output
            data_cfg.datamodule.autoencoder_mode = False
    else:
        with open_dict(data_cfg):
            data_cfg.n_steps_input = n_steps_input
            data_cfg.n_steps_output = n_steps_output
            data_cfg.autoencoder_mode = False


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
    stride_cfg = training_cfg.get("stride") if training_cfg is not None else None

    n_steps_input = args.n_steps_input or n_steps_input_cfg
    n_steps_output = args.n_steps_output or n_steps_output_cfg

    checkpoint = args.autoencoder_checkpoint
    if checkpoint is None and ckpt_cfg is not None:
        checkpoint = Path(ckpt_cfg)

    freeze_autoencoder = (
        args.freeze_autoencoder if args.freeze_autoencoder is not None else freeze_cfg
    )

    if stride_cfg in (None, "auto"):
        stride_cfg = n_steps_output
    stride_override = getattr(args, "stride", None)
    stride = stride_override or stride_cfg or n_steps_output
    if stride < 1:
        msg = "stride must be >= 1."
        raise ValueError(msg)

    if training_cfg is not None:
        with open_dict(training_cfg):
            training_cfg["stride"] = stride

    if n_steps_output < 1:
        msg = "n_steps_output must be >= 1 for processor training."
        raise ValueError(msg)

    return TrainingParams(
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        stride=stride,
        autoencoder_checkpoint=checkpoint,
        freeze_autoencoder=freeze_autoencoder,
    )


__all__ = [
    "TrainingParams",
    "align_processor_channels_with_encoder",
    "compose_training_config",
    "configure_module_dimensions",
    "normalize_processor_cfg",
    "prepare_datamodule",
    "resolve_training_params",
    "update_data_cfg",
]
