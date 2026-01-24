"""High-level setup utilities for Autocast experiments."""

import logging
from pathlib import Path

import lightning as L
import torch
import yaml
from hydra.utils import instantiate
from torch import nn

from autocast.config.base import Config
from autocast.logging import create_wandb_logger
from autocast.models.autoencoder import AE, AELoss
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.models.processor import ProcessorModel
from autocast.models.processor_ensemble import ProcessorModelEnsemble
from autocast.scripts.train.configuration import (
    build_datamodule,
    resolve_auto_params,
)
from autocast.types.batch import Batch, EncodedBatch

log = logging.getLogger(__name__)


def setup_datamodule(config: Config):
    """Create the datamodule and infer data shapes."""
    # Build DataModule
    data_dict = config.data.model_dump()
    datamodule = build_datamodule(data_dict)

    # Infer Shapes from first batch
    datamodule.setup(stage="fit")
    batch = next(iter(datamodule.train_dataloader()))

    # Abstract field access
    if isinstance(batch, Batch):
        train_inputs = batch.input_fields
        train_outputs = batch.output_fields
    elif isinstance(batch, EncodedBatch):
        train_inputs = batch.encoded_inputs
        train_outputs = batch.encoded_output_fields
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # Get shapes
    input_shape = train_inputs.shape
    output_shape = train_outputs.shape

    # Resolve 'auto' params in config
    config = resolve_auto_params(config, input_shape, output_shape)

    logic_stats = {
        "channel_count": input_shape[-1],
        "n_steps_input": input_shape[1],
        "n_steps_output": output_shape[1],
        "input_shape": input_shape,
        "output_shape": output_shape,
        "example_batch": batch,
    }

    return datamodule, config, logic_stats


def setup_encoded_datamodule(config: Config):
    """Alias for setup_datamodule, generic enough to handle both."""
    return setup_datamodule(config)


def setup_autoencoder_components(config: Config) -> tuple[nn.Module, nn.Module]:
    """Build or load the autoencoder (Encoder and Decoder)."""
    model_cfg = config.model
    # Ensure config dicts have correct dimensions if needed
    # (Pydantic models handled by config.model_dump later or direct access)

    encoder = instantiate(model_cfg.encoder)
    decoder = instantiate(model_cfg.decoder)

    checkpoint = config.training.autoencoder_checkpoint

    if checkpoint is None:
        log.info("No autoencoder checkpoint supplied; training from scratch.")
        return encoder, decoder

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log.info("Loading autoencoder weights from %s", checkpoint_path)
    # Load into AE wrapper to handle loading logic correctly
    AE.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        encoder=encoder,
        decoder=decoder,
        loss_func=AELoss(),  # Dummy loss for loading
        strict=False,
    )
    return encoder, decoder


def _infer_latent_channels(encoder: nn.Module, batch: Any) -> int:
    """Run a forward pass to determine latent channel count."""
    prev_training = encoder.training
    encoder.eval()
    try:
        with torch.no_grad():
            if hasattr(encoder, "encode"):
                encoded = encoder.encode(batch)
            else:
                encoded = encoder(batch.input_fields)

            # Assuming channel dim is last or typical
            channel_dim = getattr(encoder, "channel_dim", -1)
            return encoded.shape[channel_dim]
    except Exception as e:
        log.warning(
            f"Could not infer latent channels: {e}. Defaulting to input channels."
        )
        return batch.input_fields.shape[-1]
    finally:
        encoder.train(prev_training)


def setup_processor_model(config: Config, stats: dict) -> ProcessorModel:
    """Setup just the processor model for training on latents."""
    model_cfg = config.model

    # Update processor config with inferred dimensions
    # We edit the Pydantic object directly
    proc_cfg = model_cfg.processor

    # If using 'auto' or not set, we might need manual overrides.
    # Ideally, we pass these as kwargs or instantiate does it?
    # Hydra instantiate uses the dict.

    # We can use 'instantiate' with extra kwargs to override config
    processor = instantiate(
        proc_cfg,
        in_channels=stats["channel_count"] * stats["n_steps_input"],
        out_channels=stats["channel_count"] * stats["n_steps_output"],
        n_steps_output=stats["n_steps_output"],
        n_channels_out=stats["channel_count"],
    )

    loss_func = instantiate(getattr(model_cfg, "loss_func", None)) or nn.MSELoss()

    is_ensemble = getattr(model_cfg, "n_members", 1) > 1
    cls = ProcessorModelEnsemble if is_ensemble else ProcessorModel

    kwargs = {
        "processor": processor,
        "stride": config.training.stride,
        "loss_func": loss_func,
        "learning_rate": config.model.learning_rate,
    }
    if is_ensemble:
        kwargs["n_members"] = model_cfg.n_members

    return cls(**kwargs)


def setup_epd_model(config: Config, stats: dict) -> EncoderProcessorDecoder:
    """Orchestrate the creation of the full Encoder-Processor-Decoder model."""
    encoder, decoder = setup_autoencoder_components(config)

    if config.training.freeze_autoencoder:
        for p in encoder.parameters():
            p.requires_grad = False
        for p in decoder.parameters():
            p.requires_grad = False

    # Infer Latent Dimensions to configure Processor
    latent_channels = _infer_latent_channels(encoder, stats["example_batch"])
    log.info(f"Inferred latent channel count: {latent_channels}")

    # Build Processor
    # Pass inferred dimensions as kwargs to instantiate to ensure correctness
    # regardless of config defaults
    proc_cfg = config.model.processor

    # Note: Logic for 'n_channels_out' vs 'out_channels' depends on the specific processor (UNet vs ViT)
    # Generic approach:
    processor = instantiate(
        proc_cfg,
        in_channels=latent_channels * stats["n_steps_input"],
        out_channels=latent_channels * stats["n_steps_output"],
        n_channels_out=latent_channels,
        n_steps_output=stats["n_steps_output"],
    )

    loss_func = instantiate(getattr(config.model, "loss_func", None)) or nn.MSELoss()

    is_ensemble = getattr(config.model, "n_members", 1) > 1
    cls = EncoderProcessorDecoderEnsemble if is_ensemble else EncoderProcessorDecoder

    kwargs = {
        "encoder_decoder": EncoderDecoder(encoder, decoder),
        "processor": processor,
        "learning_rate": config.model.learning_rate,
        "train_in_latent_space": config.model.train_in_latent_space,
        "stride": config.training.stride,
        "loss_func": loss_func,
        # "input_noise_injector": ... (omitted for brevity, can add back if needed)
    }
    if is_ensemble:
        kwargs["n_members"] = config.model.n_members

    return cls(**kwargs)


def run_training(
    cfg: Any,  # Unused, compat
    pydantic_config: Config,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    work_dir: Path,
    skip_test: bool = False,
    output_checkpoint_path: Optional[Path] = None,
    job_type: str = "train",
):
    """Standardized training loop."""
    work_dir = Path(work_dir)

    wandb_logger, watch_cfg = create_wandb_logger(
        pydantic_config.logging.model_dump(),
        experiment_name=pydantic_config.experiment_name,
        job_type=job_type,
        work_dir=work_dir,
        config={"hydra": pydantic_config.model_dump()},
    )

    trainer = instantiate(
        pydantic_config.trainer.model_dump(),
        default_root_dir=str(work_dir),
        logger=wandb_logger,
    )

    log.info("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)

    if not skip_test:
        trainer.test(model=model, dataloaders=datamodule.test_dataloader())

    # Checkpointing
    ckpt_name = output_checkpoint_path or pydantic_config.output.get(
        "checkpoint_name", "model.ckpt"
    )
    ckpt_path = work_dir / ckpt_name
    trainer.save_checkpoint(ckpt_path)
    msg = f"Saved checkpoint to {ckpt_path}"
    log.info(msg)

    if pydantic_config.output.get("save_config"):
        with open(work_dir / "resolved_config.yaml", "w") as f:
            yaml.dump(pydantic_config.model_dump(), f)
