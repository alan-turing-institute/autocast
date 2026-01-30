"""High-level setup utilities for Autocast experiments."""

import inspect
import logging
from pathlib import Path
from typing import Any

import lightning as L
import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from autocast.decoders.base import Decoder
from autocast.encoders.base import Encoder, EncoderWithCond
from autocast.logging import create_wandb_logger
from autocast.models.autoencoder import AE, AELoss
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.models.processor import ProcessorModel
from autocast.models.processor_ensemble import ProcessorModelEnsemble
from autocast.scripts.config import save_resolved_config
from autocast.scripts.data import build_datamodule
from autocast.types.batch import Batch, EncodedBatch

log = logging.getLogger(__name__)


def resolve_auto_params(
    config: DictConfig, input_shape: tuple, output_shape: tuple
) -> DictConfig:
    """Resolve 'auto' values in the configuration using inferred data shapes."""
    training_cfg = config.get("training")
    if training_cfg is None:
        training_cfg = config.get("datamodule")
    if training_cfg is None:
        return config

    if training_cfg.get("n_steps_input") == "auto":
        training_cfg["n_steps_input"] = input_shape[1]
    if training_cfg.get("n_steps_output") == "auto":
        training_cfg["n_steps_output"] = output_shape[1]

    if training_cfg.get("stride") == "auto":
        training_cfg["stride"] = training_cfg.get("n_steps_output", output_shape[1])

    if training_cfg.get("rollout_stride") == "auto":
        training_cfg["rollout_stride"] = training_cfg.get("stride")

    return config


def _get_optimizer_config(config: DictConfig) -> dict[str, Any] | None:
    opt_cfg = config.get("optimizer")
    if opt_cfg is None:
        return None
    cfg = OmegaConf.to_container(opt_cfg, resolve=True)
    return cfg if isinstance(cfg, dict) else None  # type: ignore  # noqa: PGH003


def _get_training_cfg(config: DictConfig) -> dict[str, Any]:
    training_cfg = config.get("training")
    if training_cfg is None:
        training_cfg = config.get("datamodule")
    if training_cfg is None:
        return {}
    cfg = OmegaConf.to_container(training_cfg, resolve=True)
    return cfg if isinstance(cfg, dict) else {}  # type: ignore  # noqa: PGH003


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
    training_cfg = _get_training_cfg(config)

    logic_stats = {
        "channel_count": input_shape[-1],
        "n_steps_input": training_cfg.get("n_steps_input", input_shape[1]),
        "n_steps_output": training_cfg.get("n_steps_output", output_shape[1]),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "example_batch": batch,
    }

    return datamodule, config, logic_stats


def setup_encoded_datamodule(config: DictConfig):
    """Alias for setup_datamodule, generic enough to handle both."""
    return setup_datamodule(config)


def setup_autoencoder_components(
    config: DictConfig, stats: dict
) -> tuple[EncoderWithCond, Decoder]:
    """Build or load the autoencoder (Encoder and Decoder)."""
    model_cfg = config.get("model", {})
    encoder_cfg = model_cfg.get("encoder")
    decoder_cfg = model_cfg.get("decoder")

    n_channels = stats.get("channel_count")

    if isinstance(encoder_cfg, DictConfig):
        encoder_cfg = OmegaConf.to_container(encoder_cfg, resolve=True)
    if isinstance(encoder_cfg, dict) and (
        "in_channels" in encoder_cfg
        and isinstance(n_channels, int)
        and encoder_cfg.get("in_channels") in (None, "auto")
    ):
        encoder_cfg["in_channels"] = n_channels

    if isinstance(decoder_cfg, DictConfig):
        decoder_cfg = OmegaConf.to_container(decoder_cfg, resolve=True)
    if isinstance(decoder_cfg, dict):
        if (
            "out_channels" in decoder_cfg
            and isinstance(n_channels, int)
            and decoder_cfg.get("out_channels") in (None, "auto")
        ):
            decoder_cfg["out_channels"] = n_channels
        if (
            "output_channels" in decoder_cfg
            and isinstance(n_channels, int)
            and decoder_cfg.get("output_channels") in (None, "auto")
        ):
            decoder_cfg["output_channels"] = n_channels

    encoder = instantiate(encoder_cfg)
    decoder = instantiate(decoder_cfg)

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
    model_cfg = config.get("model", {})
    loss_cfg = model_cfg.get("loss")
    loss = instantiate(loss_cfg) if loss_cfg is not None else None
    model = AE(encoder=encoder, decoder=decoder, loss_func=loss)
    lr = model_cfg.get("learning_rate")
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
    model_cfg = config.get("model", {})

    proc_cfg = model_cfg.get("processor")
    if isinstance(proc_cfg, DictConfig):
        proc_cfg = OmegaConf.to_container(proc_cfg, resolve=True)

    proc_kwargs = {
        "in_channels": stats["channel_count"] * stats["n_steps_input"],
        "out_channels": stats["channel_count"] * stats["n_steps_output"],
        "n_steps_output": stats["n_steps_output"],
        "n_channels_out": stats["channel_count"],
    }
    target = proc_cfg.get("_target_") if isinstance(proc_cfg, dict) else None
    proc_kwargs = _filter_kwargs_for_target(target, proc_kwargs)
    processor = instantiate(proc_cfg, **proc_kwargs)

    loss_func_cfg = model_cfg.get("loss_func")
    loss_func = (
        instantiate(loss_func_cfg) if loss_func_cfg is not None else nn.MSELoss()
    )

    is_ensemble = model_cfg.get("n_members", 1) > 1
    cls = ProcessorModelEnsemble if is_ensemble else ProcessorModel

    training_cfg = _get_training_cfg(config)
    kwargs = {
        "processor": processor,
        "stride": training_cfg.get("stride", stats["n_steps_output"]),
        "loss_func": loss_func,
        "learning_rate": model_cfg.get("learning_rate", 1e-3),
        "optimizer_config": _get_optimizer_config(config),
    }
    if is_ensemble:
        kwargs["n_members"] = model_cfg.get("n_members")

    return cls(**kwargs)


def setup_epd_model(config: DictConfig, stats: dict) -> EncoderProcessorDecoder:
    """Orchestrate the creation of the full Encoder-Processor-Decoder model."""
    encoder, decoder = setup_autoencoder_components(config, stats)

    training_cfg = _get_training_cfg(config)
    if training_cfg.get("freeze_autoencoder"):
        for p in encoder.parameters():
            p.requires_grad = False
        for p in decoder.parameters():
            p.requires_grad = False

    latent_channels = _infer_latent_channels(encoder, stats["example_batch"])  # type: ignore TODO
    stats["latent_channels"] = latent_channels
    log.info("Inferred latent channel count: %s", latent_channels)

    model_cfg = config.get("model", {})
    proc_cfg = model_cfg.get("processor")
    if isinstance(proc_cfg, DictConfig):
        proc_cfg = OmegaConf.to_container(proc_cfg, resolve=True)

    proc_kwargs = {
        "in_channels": latent_channels * stats["n_steps_input"],
        "out_channels": latent_channels * stats["n_steps_output"],
        "n_channels_out": latent_channels,
        "n_steps_output": stats["n_steps_output"],
    }
    target = proc_cfg.get("_target_") if isinstance(proc_cfg, dict) else None
    proc_kwargs = _filter_kwargs_for_target(target, proc_kwargs)
    processor = instantiate(proc_cfg, **proc_kwargs)

    loss_func_cfg = model_cfg.get("loss_func")
    loss_func = (
        instantiate(loss_func_cfg) if loss_func_cfg is not None else nn.MSELoss()
    )

    is_ensemble = model_cfg.get("n_members", 1) > 1
    cls = EncoderProcessorDecoderEnsemble if is_ensemble else EncoderProcessorDecoder

    kwargs = {
        "encoder_decoder": EncoderDecoder(
            encoder,
            decoder,
            optimizer_config=_get_optimizer_config(config),
        ),
        "processor": processor,
        "learning_rate": model_cfg.get("learning_rate", 1e-3),
        "train_in_latent_space": model_cfg.get("train_in_latent_space", False),
        "stride": training_cfg.get("stride", stats["n_steps_output"]),
        "optimizer_config": _get_optimizer_config(config),
        "loss_func": loss_func,
    }
    if is_ensemble:
        kwargs["n_members"] = model_cfg.get("n_members")

    return cls(**kwargs)


def run_training(
    config: DictConfig,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    work_dir: Path,
    skip_test: bool = False,
    output_checkpoint_path: Path | None = None,
    job_type: str = "train",
):
    """Standardized training loop."""
    # Ensure work_dir is a Path
    work_dir = Path(work_dir)

    # Setup logger
    logging_cfg = config.get("logging")
    logging_cfg = (
        OmegaConf.to_container(logging_cfg, resolve=True)
        if logging_cfg is not None
        else {}
    )
    wandb_logger, _watch_cfg = create_wandb_logger(
        logging_cfg,  # type: ignore TODO: fix
        experiment_name=config.get("experiment_name"),
        job_type=job_type,
        work_dir=work_dir,
        config={"hydra": OmegaConf.to_container(config, resolve=True)},
    )

    # Get trainer
    trainer_cfg = config.get("trainer")
    trainer_cfg = OmegaConf.to_container(trainer_cfg, resolve=True)
    trainer = instantiate(
        trainer_cfg,
        default_root_dir=str(work_dir),
        logger=wandb_logger,
    )

    # Get output config and save resolved config if requested
    output_cfg = config.get("output", {})
    if output_cfg.get("save_config"):
        save_resolved_config(config, work_dir)

    log.info("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)

    # Run testing if not skipped
    if not skip_test:
        trainer.test(model=model, dataloaders=datamodule.test_dataloader())

    # Save final checkpoint
    ckpt_name = output_checkpoint_path or output_cfg.get(
        "checkpoint_name", "model.ckpt"
    )
    ckpt_path = work_dir / ckpt_name
    trainer.save_checkpoint(ckpt_path)
    log.info("Saved checkpoint to %s", ckpt_path)
