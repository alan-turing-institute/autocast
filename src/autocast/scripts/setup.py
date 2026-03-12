"""Setup utilities for AutoCast experiments."""

import inspect
import logging
from pathlib import Path
from typing import Any

import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig
from torch import nn

from autocast.data.datamodule import SpatioTemporalDataModule, TheWellDataModule
from autocast.decoders.base import Decoder
from autocast.encoders.base import Encoder, EncoderWithCond
from autocast.models.autoencoder import AE, AELoss
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.models.processor import ProcessorModel
from autocast.models.processor_ensemble import ProcessorModelEnsemble
from autocast.scripts.data import build_datamodule
from autocast.types.batch import Batch, EncodedBatch

log = logging.getLogger(__name__)


def resolve_auto_params(
    config: DictConfig, input_shape: tuple, output_shape: tuple
) -> DictConfig:
    """Resolve 'auto' values in the configuration using inferred data shapes."""
    data_config = config.get("datamodule")
    if data_config is None:
        return config
    if data_config.get("n_steps_input") == "auto":
        data_config["n_steps_input"] = input_shape[1]
    if data_config.get("n_steps_output") == "auto":
        data_config["n_steps_output"] = output_shape[1]
    if data_config.get("stride") == "auto":
        data_config["stride"] = data_config.get("n_steps_output", output_shape[1])
    if data_config.get("rollout_stride") == "auto":
        data_config["rollout_stride"] = data_config.get("stride")
    return config


def _get_optimizer_config(config: DictConfig) -> DictConfig:
    """Return optimizer config."""
    optimizer_config = config.get("optimizer")
    if optimizer_config is None:
        msg = "Optimizer config is required for training."
        raise ValueError(msg)
    return optimizer_config


def _filter_kwargs_for_target(
    target: str | None, kwargs: dict[str, Any]
) -> dict[str, Any]:
    if target is None:
        return kwargs
    try:
        cls = get_class(target)
    except Exception:
        return kwargs
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return kwargs
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    ):
        return kwargs
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


def _set_if_auto(cfg: DictConfig, key: str, value: Any) -> None:
    """Set config key to value if current value is None or 'auto'."""
    if key in cfg and cfg.get(key) in (None, "auto"):
        cfg[key] = value


def _apply_processor_channel_defaults(
    processor_config: DictConfig | None,
    *,
    in_channels: int,
    out_channels: int,
    n_steps_input: int,
    n_steps_output: int,
    n_channels_out: int,
    global_cond_channels: int | None = None,
    spatial_resolution: tuple[int, ...] | None = None,
) -> None:
    """Apply inferred channel/step defaults to processor and backbone configs."""
    if processor_config is None:
        return

    _set_if_auto(processor_config, "in_channels", in_channels)
    _set_if_auto(processor_config, "out_channels", out_channels)
    _set_if_auto(processor_config, "n_steps_input", n_steps_input)
    _set_if_auto(processor_config, "n_steps_output", n_steps_output)
    _set_if_auto(processor_config, "n_channels_out", n_channels_out)
    _set_if_auto(processor_config, "global_cond_channels", global_cond_channels)
    _set_if_auto(
        processor_config,
        "spatial_resolution",
        list(spatial_resolution) if spatial_resolution is not None else None,
    )

    backbone_config = processor_config.get("backbone")
    if backbone_config is None:
        return

    # Backbone applies steps multiplier internally, so we pass per-step channels
    _set_if_auto(backbone_config, "in_channels", n_channels_out)  # z has n_channels_out
    _set_if_auto(backbone_config, "out_channels", n_channels_out)
    _set_if_auto(backbone_config, "cond_channels", in_channels)  # x has in_channels
    _set_if_auto(backbone_config, "n_steps_input", n_steps_input)
    _set_if_auto(backbone_config, "n_steps_output", n_steps_output)
    if global_cond_channels is not None:
        _set_if_auto(backbone_config, "global_cond_channels", global_cond_channels)


def setup_datamodule(
    config: DictConfig,
) -> tuple[SpatioTemporalDataModule | TheWellDataModule, DictConfig, dict]:
    """Create the datamodule and infer data shapes."""
    datamodule = build_datamodule(config)

    datamodule.setup(stage="fit")
    batch = next(iter(datamodule.train_dataloader()))

    if isinstance(batch, Batch):
        train_inputs = batch.input_fields
        train_outputs = batch.output_fields
        n_constant_scalars = (
            batch.constant_scalars.shape[-1]
            if batch.constant_scalars is not None
            else 0
        )
        n_constant_field_channels = (
            batch.constant_fields.shape[-1] if batch.constant_fields is not None else 0
        )
    elif isinstance(batch, EncodedBatch):
        train_inputs = batch.encoded_inputs
        train_outputs = batch.encoded_output_fields
        n_constant_scalars = None
        n_constant_field_channels = None
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    input_shape = train_inputs.shape
    output_shape = train_outputs.shape

    config = resolve_auto_params(config, input_shape, output_shape)
    logic_stats = {
        "channel_count": input_shape[-1],
        "n_steps_input": input_shape[1],
        "n_steps_output": output_shape[1],
        "n_constant_scalars": n_constant_scalars,
        "n_constant_field_channels": n_constant_field_channels,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "example_batch": batch,
    }

    return datamodule, config, logic_stats


def setup_autoencoder_components(
    config: DictConfig, stats: dict, extra_input_channels: int = 0
) -> tuple[EncoderWithCond, Decoder]:
    """Build or load the autoencoder (Encoder and Decoder)."""
    model_config = config.get("model", {})
    encoder_config = model_config.get("encoder")
    decoder_config = model_config.get("decoder")

    log.info(
        "Model config before resolving auto channels:\nEncoder: %s\nDecoder: %s",
        encoder_config,
        decoder_config,
    )
    base_channels = stats.get("channel_count")
    input_channels = (
        (base_channels + extra_input_channels)
        if isinstance(base_channels, int) and extra_input_channels
        else base_channels
    )

    # Update auto channel values directly in DictConfig
    if (
        encoder_config
        and isinstance(input_channels, int)
        and encoder_config.get("in_channels") in (None, "auto")
    ):
        # TODO: add more robust approach to inlcuding extra constant channels
        # handling here is specifically for the case when the encoder_config
        # includes `with_constants` - this is currently `PermuteConcat`
        if encoder_config.get("with_constants") and input_channels is not None:
            input_channels += stats.get("n_constant_scalars", 0)
            input_channels += stats.get("n_constant_field_channels", 0)
        encoder_config["in_channels"] = input_channels

    # Update n_steps_input for encoders that need it (e.g., PermuteConcat)
    if (
        encoder_config
        and "n_steps_input" in encoder_config
        and encoder_config.get("n_steps_input") in (None, "auto")
    ):
        encoder_config["n_steps_input"] = stats.get("n_steps_input")

    if decoder_config:
        if (
            "out_channels" in decoder_config
            and isinstance(base_channels, int)
            and decoder_config.get("out_channels") in (None, "auto")
        ):
            decoder_config["out_channels"] = base_channels
        if (
            "output_channels" in decoder_config
            and isinstance(base_channels, int)
            and decoder_config.get("output_channels") in (None, "auto")
        ):
            decoder_config["output_channels"] = base_channels

    log.info(
        "Model config after resolving auto channels:\nEncoder: %s\nDecoder: %s",
        encoder_config,
        decoder_config,
    )
    encoder = instantiate(encoder_config)
    decoder = instantiate(decoder_config)
    checkpoint = config.get("autoencoder_checkpoint")

    if checkpoint is None:
        log.info("No autoencoder checkpoint supplied.")
        return encoder, decoder

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log.info("Loading autoencoder weights from %s", checkpoint_path)
    AE.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        encoder=encoder,
        decoder=decoder,
        loss_func=AELoss(),
        strict=False,
    )
    return encoder, decoder


def setup_autoencoder_model(
    config: DictConfig,
    stats: dict,
    datamodule: SpatioTemporalDataModule | TheWellDataModule,
) -> AE:
    """Build the full autoencoder model (encoder, decoder, loss)."""
    encoder, decoder = setup_autoencoder_components(config, stats)
    model_config = config.get("model", {})
    loss_config = model_config.get("loss")
    loss = instantiate(loss_config) if loss_config is not None else None
    optimizer_config = _get_optimizer_config(config)
    norm = getattr(datamodule.train_dataset, "norm", None)
    model = AE(
        encoder=encoder,
        decoder=decoder,
        loss_func=loss,
        optimizer_config=optimizer_config,
        norm=norm,
    )
    return model


def _get_latent_channels(encoder: Encoder) -> int:
    """Get latent channel count from encoder.

    All encoders must set latent_channels in their __init__.
    """
    if not hasattr(encoder, "latent_channels") or not isinstance(
        encoder.latent_channels, int
    ):
        raise ValueError(
            f"Encoder {type(encoder).__name__} must set latent_channels as an integer "
            "in its __init__ method."
        )
    return encoder.latent_channels


def _get_latent_channels_out(decoder: Decoder) -> int:
    """Get latent channel count from decoder.

    All decoders must set latent_channels in their __init__.
    """
    if not hasattr(decoder, "latent_channels") or not isinstance(
        decoder.latent_channels, int
    ):
        raise ValueError(
            f"Decoder {type(decoder).__name__} must set latent_channels as an integer "
            "in its __init__ method."
        )
    return decoder.latent_channels


def _get_normalized_processor_config(model_config: DictConfig) -> DictConfig | None:
    """Return processor config or None."""
    return model_config.get("processor")


def _build_processor(
    model_config: DictConfig,
    proc_kwargs: dict[str, Any],
    global_cond_channels: int | None = None,
) -> nn.Module:
    """Build processor from config with channel defaults applied."""
    processor_config = _get_normalized_processor_config(model_config)
    _apply_processor_channel_defaults(
        processor_config,
        in_channels=proc_kwargs["in_channels"],
        out_channels=proc_kwargs["out_channels"],
        n_steps_input=proc_kwargs["n_steps_input"],
        n_steps_output=proc_kwargs["n_steps_output"],
        n_channels_out=proc_kwargs["n_channels_out"],
        global_cond_channels=global_cond_channels,
        spatial_resolution=proc_kwargs.get("spatial_resolution"),
    )
    target = processor_config.get("_target_") if processor_config else None
    filtered_kwargs = _filter_kwargs_for_target(target, proc_kwargs)
    return instantiate(processor_config, **filtered_kwargs)


def _build_loss_func(model_config: DictConfig) -> nn.Module:
    """Build loss function from config, defaulting to MSELoss."""
    loss_func_config = model_config.get("loss_func")
    if loss_func_config is None:
        return nn.MSELoss()
    return instantiate(loss_func_config)


def setup_processor_model(
    config: DictConfig,
    stats: dict,
    datamodule: SpatioTemporalDataModule | TheWellDataModule,
) -> ProcessorModel:
    """Set up just the processor model for training on latents."""
    model_config = config.get("model", {})
    noise_injector, extra_input_channels = _resolve_input_noise_injector(model_config)

    proc_kwargs = {
        "in_channels": stats["channel_count"] + extra_input_channels,
        "out_channels": stats["channel_count"],
        "n_steps_input": stats["n_steps_input"],
        "n_steps_output": stats["n_steps_output"],
        "n_channels_out": stats["channel_count"],
        "spatial_resolution": tuple(stats["input_shape"][2:-1]),
    }
    processor = _build_processor(model_config, proc_kwargs)
    loss_func = _build_loss_func(model_config)

    is_ensemble = model_config.get("n_members", 1) > 1
    cls = ProcessorModelEnsemble if is_ensemble else ProcessorModel

    data_config = config.get("datamodule", {})
    optimizer_config = _get_optimizer_config(config)
    norm = getattr(datamodule.train_dataset, "norm", None)
    kwargs = {
        "processor": processor,
        "stride": data_config.get("stride", stats["n_steps_output"]),
        "loss_func": loss_func,
        "optimizer_config": optimizer_config,
        "noise_injector": noise_injector,
        "norm": norm,
    }
    if is_ensemble:
        kwargs["n_members"] = model_config.get("n_members")

    return cls(**kwargs)


def setup_epd_model(
    config: DictConfig,
    stats: dict,
    datamodule: SpatioTemporalDataModule | TheWellDataModule,
) -> EncoderProcessorDecoder | EncoderProcessorDecoderEnsemble:
    """Orchestrate the creation of the full Encoder-Processor-Decoder model."""
    model_config = config.get("model", {})
    noise_injector, extra_input_channels = _resolve_input_noise_injector(model_config)

    encoder, decoder = setup_autoencoder_components(
        config, stats, extra_input_channels=extra_input_channels
    )

    data_config = config.get("datamodule", {})
    freeze_encoder_decoder = bool(
        model_config.get("freeze_encoder_decoder", False)
        or data_config.get("freeze_autoencoder", False)
    )
    if freeze_encoder_decoder:
        for p in encoder.parameters():
            p.requires_grad = False
        for p in decoder.parameters():
            p.requires_grad = False

    latent_channels = _get_latent_channels(encoder)
    latent_channels_out = _get_latent_channels_out(decoder)
    stats["latent_channels"] = latent_channels
    stats["latent_channels_out"] = latent_channels_out
    log.info("Latent channel in count: %s", latent_channels)
    log.info("Latent channel out count: %s", latent_channels_out)

    global_cond_channels = None
    if hasattr(encoder, "encode_cond"):
        cond = encoder.encode_cond(stats["example_batch"])
        if cond is not None:
            global_cond_channels = cond.shape[-1]
    log.info(
        (
            "Global cond inference: constant_scalars=%s, "
            "boundary_conditions=%s, "
            "inferred_global_cond_channels=%s",
        ),
        (
            None
            if stats["example_batch"].constant_scalars is None
            else tuple(stats["example_batch"].constant_scalars.shape)
        ),
        (
            None
            if stats["example_batch"].boundary_conditions is None
            else tuple(stats["example_batch"].boundary_conditions.shape)
        ),
        global_cond_channels,
    )

    steps_in = stats["n_steps_input"]
    steps_out = stats["n_steps_output"]
    with torch.no_grad():
        encoded_example, _ = encoder.encode_with_cond(stats["example_batch"])
    latent_spatial_resolution = tuple(encoded_example.shape[2:-1])

    # TODO: currently "out_channels" and "in_channels" are only used in the config for
    # ViT and FNO, while "n_channels_out" is used in flow_matching and diffusions
    proc_kwargs = {
        "in_channels": latent_channels,
        "out_channels": latent_channels_out,
        "n_channels_out": latent_channels_out,
        "n_steps_input": steps_in,
        "n_steps_output": steps_out,
        "spatial_resolution": latent_spatial_resolution,
    }
    processor = _build_processor(model_config, proc_kwargs, global_cond_channels)
    loss_func = _build_loss_func(model_config)

    is_ensemble = model_config.get("n_members", 1) > 1
    cls = EncoderProcessorDecoderEnsemble if is_ensemble else EncoderProcessorDecoder

    optimizer_config = _get_optimizer_config(config)
    norm = getattr(datamodule.train_dataset, "norm", None)
    kwargs = {
        "encoder_decoder": EncoderDecoder(
            encoder,
            decoder,
            optimizer_config=optimizer_config,
        ),
        "processor": processor,
        "train_in_latent_space": model_config.get("train_in_latent_space", False),
        "freeze_encoder_decoder": freeze_encoder_decoder,
        "stride": data_config.get("stride", stats["n_steps_output"]),
        "optimizer_config": optimizer_config,
        "loss_func": loss_func,
        "input_noise_injector": noise_injector,
        "norm": norm,
    }
    if is_ensemble:
        kwargs["n_members"] = model_config.get("n_members")

    return cls(**kwargs)


def _resolve_input_noise_injector(model_config: DictConfig) -> tuple[Any | None, int]:
    noise_config = model_config.get("input_noise_injector", None)
    if not noise_config or "_target_" not in noise_config:
        return None, 0

    extra_channels = 0
    if "ConcatenatedNoiseInjector" in str(noise_config.get("_target_")):
        n_channels = noise_config.get("n_channels")
        if n_channels in (None, "auto"):
            msg = (
                "ConcatenatedNoiseInjector requires explicit n_channels in config. "
                "Set input_noise_injector.n_channels to an integer value."
            )
            raise ValueError(msg)
        extra_channels = int(n_channels)

    return instantiate(noise_config), extra_channels
