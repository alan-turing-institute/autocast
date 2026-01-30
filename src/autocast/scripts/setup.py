"""Setup utilities for AutoCast experiments."""

import inspect
import logging
from pathlib import Path
from typing import Any

import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

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


def _get_optimizer_config(config: DictConfig) -> dict[str, Any] | None:
    optimizer_config = config.get("optimizer")
    if optimizer_config is None:
        return None
    config_dict = OmegaConf.to_container(optimizer_config, resolve=True)
    return config_dict if isinstance(config_dict, dict) else None  # type: ignore since checked if dict


def _get_data_config(config: DictConfig) -> dict[str, Any]:
    data_config = config.get("datamodule")
    if data_config is None:
        return {}
    config_dict = OmegaConf.to_container(data_config, resolve=True)
    return config_dict if isinstance(config_dict, dict) else {}  # type: ignore since checked if dict


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


def setup_datamodule(config: DictConfig):
    """Create the datamodule and infer data shapes."""
    datamodule = build_datamodule(config)

    datamodule.setup(stage="fit")
    batch = next(iter(datamodule.train_dataloader()))

    if isinstance(batch, Batch):
        train_inputs = batch.input_fields
        train_outputs = batch.output_fields
    elif isinstance(batch, EncodedBatch):
        train_inputs = batch.encoded_inputs
        train_outputs = batch.encoded_output_fields
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    input_shape = train_inputs.shape
    output_shape = train_outputs.shape

    config = resolve_auto_params(config, input_shape, output_shape)
    data_config = _get_data_config(config)
    logic_stats = {
        "channel_count": input_shape[-1],
        "n_steps_input": data_config.get("n_steps_input", input_shape[1]),
        "n_steps_output": data_config.get("n_steps_output", output_shape[1]),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "example_batch": batch,
    }

    return datamodule, config, logic_stats


def setup_encoded_datamodule(config: DictConfig):
    """Alias for setup_datamodule, generic enough to handle both."""
    return setup_datamodule(config)


def setup_autoencoder_components(
    config: DictConfig, stats: dict, extra_input_channels: int = 0
) -> tuple[EncoderWithCond, Decoder]:
    """Build or load the autoencoder (Encoder and Decoder)."""
    model_config = config.get("model", {})
    encoder_config = model_config.get("encoder")
    decoder_config = model_config.get("decoder")

    base_channels = stats.get("channel_count")
    input_channels = (
        (base_channels + extra_input_channels)
        if isinstance(base_channels, int) and extra_input_channels
        else base_channels
    )

    if isinstance(encoder_config, DictConfig):
        encoder_config = OmegaConf.to_container(encoder_config, resolve=True)
    if isinstance(encoder_config, dict) and (
        "in_channels" in encoder_config
        and isinstance(input_channels, int)
        and encoder_config.get("in_channels") in (None, "auto")
    ):
        encoder_config["in_channels"] = input_channels

    if isinstance(decoder_config, DictConfig):
        decoder_config = OmegaConf.to_container(decoder_config, resolve=True)
    if isinstance(decoder_config, dict):
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

    encoder = instantiate(encoder_config)
    decoder = instantiate(decoder_config)

    checkpoint = config.get("training", {}).get("autoencoder_checkpoint")

    if checkpoint is None:
        log.info("No autoencoder checkpoint supplied; training from scratch.")
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


def setup_autoencoder_model(config: DictConfig, stats: dict) -> AE:
    """Build the full autoencoder model (encoder, decoder, loss)."""
    encoder, decoder = setup_autoencoder_components(config, stats)
    model_config = config.get("model", {})
    loss_config = model_config.get("loss")
    loss = instantiate(loss_config) if loss_config is not None else None
    model = AE(encoder=encoder, decoder=decoder, loss_func=loss)
    lr = model_config.get("learning_rate")
    if lr is not None:
        model.learning_rate = lr
    return model


def _infer_latent_channels(encoder: Encoder, batch: Any) -> int:
    """Run a forward pass to determine latent channel count."""
    prev_training = encoder.training
    encoder.eval()
    try:
        with torch.no_grad():
            if hasattr(encoder, "encode"):
                encoded = encoder.encode(batch)
            else:
                encoded = encoder(batch.input_fields)

            channel_dim = getattr(encoder, "channel_dim", -1)
            return encoded.shape[channel_dim]
    except Exception as e:
        msg = f"Could not infer latent channels: {e}. Defaulting to input channels."
        log.warning(msg)
        return batch.input_fields.shape[-1]
    finally:
        encoder.train(prev_training)


def setup_processor_model(config: DictConfig, stats: dict) -> ProcessorModel:
    """Set up just the processor model for training on latents."""
    model_config = config.get("model", {})
    noise_injector, extra_input_channels = _resolve_input_noise_injector(model_config)

    processor_config = model_config.get("processor")
    if isinstance(processor_config, DictConfig):
        processor_config = OmegaConf.to_container(processor_config, resolve=True)

    proc_kwargs = {
        "in_channels": (stats["channel_count"] + extra_input_channels)
        * stats["n_steps_input"],
        "out_channels": stats["channel_count"] * stats["n_steps_output"],
        "n_steps_output": stats["n_steps_output"],
        "n_channels_out": stats["channel_count"],
    }
    target = (
        processor_config.get("_target_") if isinstance(processor_config, dict) else None
    )
    proc_kwargs = _filter_kwargs_for_target(target, proc_kwargs)
    processor = instantiate(processor_config, **proc_kwargs)

    loss_func_config = model_config.get("loss_func")
    loss_func = (
        instantiate(loss_func_config) if loss_func_config is not None else nn.MSELoss()
    )

    is_ensemble = model_config.get("n_members", 1) > 1
    cls = ProcessorModelEnsemble if is_ensemble else ProcessorModel

    data_config = _get_data_config(config)
    kwargs = {
        "processor": processor,
        "stride": data_config.get("stride", stats["n_steps_output"]),
        "loss_func": loss_func,
        "learning_rate": model_config.get("learning_rate", 1e-3),
        "optimizer_config": _get_optimizer_config(config),
        "noise_injector": noise_injector,
    }
    if is_ensemble:
        kwargs["n_members"] = model_config.get("n_members")

    return cls(**kwargs)


def setup_epd_model(config: DictConfig, stats: dict) -> EncoderProcessorDecoder:
    """Orchestrate the creation of the full Encoder-Processor-Decoder model."""
    model_config = config.get("model", {})
    noise_injector, extra_input_channels = _resolve_input_noise_injector(model_config)

    encoder, decoder = setup_autoencoder_components(
        config, stats, extra_input_channels=extra_input_channels
    )

    data_config = _get_data_config(config)
    if data_config.get("freeze_autoencoder"):
        for p in encoder.parameters():
            p.requires_grad = False
        for p in decoder.parameters():
            p.requires_grad = False

    latent_channels = _infer_latent_channels(encoder, stats["example_batch"])  # type: ignore TODO
    stats["latent_channels"] = latent_channels
    log.info("Inferred latent channel count: %s", latent_channels)

    proc_config = model_config.get("processor")
    if isinstance(proc_config, DictConfig):
        proc_config = OmegaConf.to_container(proc_config, resolve=True)

    proc_kwargs = {
        "in_channels": (latent_channels + extra_input_channels)
        * stats["n_steps_input"],
        "out_channels": latent_channels * stats["n_steps_output"],
        "n_channels_out": latent_channels,
        "n_steps_output": stats["n_steps_output"],
    }
    target = proc_config.get("_target_") if isinstance(proc_config, dict) else None
    proc_kwargs = _filter_kwargs_for_target(target, proc_kwargs)
    processor = instantiate(proc_config, **proc_kwargs)

    loss_func_config = model_config.get("loss_func")
    loss_func = (
        instantiate(loss_func_config) if loss_func_config is not None else nn.MSELoss()
    )

    is_ensemble = model_config.get("n_members", 1) > 1
    cls = EncoderProcessorDecoderEnsemble if is_ensemble else EncoderProcessorDecoder

    kwargs = {
        "encoder_decoder": EncoderDecoder(
            encoder,
            decoder,
            optimizer_config=_get_optimizer_config(config),
        ),
        "processor": processor,
        "learning_rate": model_config.get("learning_rate", 1e-3),
        "train_in_latent_space": model_config.get("train_in_latent_space", False),
        "stride": data_config.get("stride", stats["n_steps_output"]),
        "optimizer_config": _get_optimizer_config(config),
        "loss_func": loss_func,
        "input_noise_injector": noise_injector,
    }
    if is_ensemble:
        kwargs["n_members"] = model_config.get("n_members")

    return cls(**kwargs)


def _resolve_input_noise_injector(
    model_config: dict | DictConfig | None,
) -> tuple[Any | None, int]:
    noise_config = model_config.get("input_noise_injector") if model_config else None
    if not noise_config or "_target_" not in noise_config:
        return None, 0

    extra_channels = 0
    if "ConcatenatedNoiseInjector" in str(noise_config.get("_target_")):
        n_channels = noise_config.get("n_channels")
        if n_channels in (None, "auto"):
            proc_config = model_config.get("processor") or {}  # type: ignore is not None
            n_channels = proc_config.get("n_noise_channels")
        extra_channels = int(n_channels) if n_channels else 0

    return instantiate(noise_config), extra_channels
