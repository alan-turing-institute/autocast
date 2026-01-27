"""Train an encoder-processor-decoder with optional autoencoder warm-start."""

from __future__ import annotations

import argparse
import logging
from argparse import BooleanOptionalAction
from pathlib import Path

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from autocast.logging import create_wandb_logger, maybe_watch_model
from autocast.models.autoencoder import AE, AELoss
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.scripts.train.configuration import (
    align_processor_channels_with_encoder,
    compose_training_config,
    configure_module_dimensions,
    normalize_processor_cfg,
    prepare_datamodule,
    resolve_training_params,
    update_data_cfg,
)

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the processor training utility."""
    parser = argparse.ArgumentParser(
        description=(
            "Train an encoder-processor-decoder model with Hydra-configured "
            "encoder, decoder, and processor components."
        )
    )
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--config-dir",
        "--config-path",
        dest="config_dir",
        type=Path,
        default=repo_root / "configs",
        help=(
            "Path to the Hydra config directory (accepts --config-path as an "
            "alias; defaults to <repo>/configs)."
        ),
    )
    parser.add_argument(
        "--config-name",
        default="encoder_processor_decoder",
        help="Hydra config name to compose (defaults to 'encoder_processor_decoder').",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Hydra config overrides (e.g. trainer.max_epochs=5 "
            "logging.wandb.enabled=true)"
        ),
    )
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=None,
        help="Overrides training.autoencoder_checkpoint from the config when set.",
    )
    parser.add_argument(
        "--freeze-autoencoder",
        action=BooleanOptionalAction,
        default=None,
        help="Toggle freezing of encoder/decoder parameters when loading a checkpoint.",
    )
    parser.add_argument(
        "--n-steps-input",
        type=int,
        default=None,
        help="Override training.n_steps_input (number of input time steps).",
    )
    parser.add_argument(
        "--n-steps-output",
        type=int,
        default=None,
        help="Override training.n_steps_output (number of target time steps).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override training stride (rollout interval between predictions).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.cwd(),
        help=(
            "Directory Lightning should treat as the default_root_dir "
            "(defaults to CWD)."
        ),
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional explicit checkpoint filename; falls back to "
            "output.checkpoint_name from the config when omitted."
        ),
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip running trainer.test() after training completes.",
    )
    return parser.parse_args()


def _freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _ensure_output_path(path: Path, work_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (work_dir / path).resolve()


def build_autoencoder_modules(
    encoder_cfg: DictConfig,
    decoder_cfg: DictConfig,
    checkpoint: Path | None,
):
    """Instantiate encoder/decoder modules and optionally load AE weights."""
    encoder = instantiate(encoder_cfg)
    decoder = instantiate(decoder_cfg)
    if checkpoint is None:
        log.info(
            "No autoencoder checkpoint supplied; training encoder/decoder jointly."
        )
        return encoder, decoder

    checkpoint_path = checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        msg = f"Checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)
    log.info("Loading autoencoder weights from %s", checkpoint_path)
    ae_loss = AELoss()
    autoencoder = AE.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        encoder=encoder,
        decoder=decoder,
        loss_func=ae_loss,
    )
    return autoencoder.encoder, autoencoder.decoder


def instantiate_trainer(
    cfg: DictConfig,
    work_dir: Path,
    *,
    logger=None,
):
    """Instantiate the Lightning trainer with a concrete root directory."""
    return instantiate(
        cfg.trainer,
        default_root_dir=str(work_dir),
        logger=logger,
    )


def main() -> None:  # noqa: PLR0915
    """CLI entrypoint for training the processor."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = compose_training_config(args)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    model_cfg = cfg.get("model") or cfg
    wandb_logger, watch_cfg = create_wandb_logger(
        cfg.get("logging"),
        experiment_name=cfg.get("experiment_name", "encoder_processor_decoder"),
        job_type="train-encoder-processor-decoder",
        work_dir=work_dir,
        config={"hydra": resolved_cfg} if resolved_cfg is not None else None,
    )
    training_params = resolve_training_params(cfg, args)
    update_data_cfg(
        cfg,
        training_params.n_steps_input,
        training_params.n_steps_output,
    )

    L.seed_everything(cfg.get("seed", 42), workers=True)

    (
        datamodule,
        channel_count,
        inferred_n_steps_input,
        inferred_n_steps_output,
        input_shape,
        output_shape,
        example_batch,
    ) = prepare_datamodule(cfg)

    log.info("Detected input shape %s and output shape %s", input_shape, output_shape)
    if inferred_n_steps_input != training_params.n_steps_input:
        log.warning(
            "Datamodule produced %s input steps but training calls for %s; "
            "proceeding with inferred value.",
            inferred_n_steps_input,
            training_params.n_steps_input,
        )
    if inferred_n_steps_output != training_params.n_steps_output:
        log.warning(
            "Datamodule produced %s output steps but training calls for %s; "
            "proceeding with inferred value.",
            inferred_n_steps_output,
            training_params.n_steps_output,
        )

    epd_cfg = model_cfg
    input_noise_cfg = (
        epd_cfg.get("input_noise_injector")
        or cfg.get("input_noise_injector")
        or cfg.get("nn", {}).get("noise", {}).get("input_noise_injector")
    )
    input_noise_injector = (
        instantiate(input_noise_cfg) if input_noise_cfg is not None else None
    )
    input_channel_count = channel_count
    if input_noise_injector is not None:
        log.info(
            "Found input noise injector %s; adjusting input channel count.",
            input_noise_injector.__class__.__name__,
        )
        input_channel_count += input_noise_injector.get_additional_channels()

    configure_module_dimensions(
        cfg,
        channel_count=channel_count,
        n_steps_input=inferred_n_steps_input,
        n_steps_output=inferred_n_steps_output,
        input_channel_count=input_channel_count,
        output_channel_count=channel_count,
    )
    normalize_processor_cfg(cfg)

    encoder, decoder = build_autoencoder_modules(
        model_cfg.encoder,
        model_cfg.decoder,
        training_params.autoencoder_checkpoint,
    )
    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder)
    align_processor_channels_with_encoder(
        cfg,
        encoder=encoder,
        channel_count=channel_count,
        n_steps_input=inferred_n_steps_input,
        n_steps_output=inferred_n_steps_output,
        example_batch=example_batch,
        input_noise_injector=input_noise_injector,
    )

    if training_params.freeze_autoencoder and training_params.autoencoder_checkpoint:
        log.info("Freezing encoder and decoder parameters.")
        _freeze_module(encoder_decoder.encoder)
        _freeze_module(encoder_decoder.decoder)

    processor = instantiate(model_cfg.processor)

    epd_cfg = model_cfg
    learning_rate = epd_cfg.get("learning_rate", 1e-3)
    optimizer_config = cfg.get("optimizer", None)
    train_in_latent_space = epd_cfg.get("train_in_latent_space", False)
    teacher_forcing_ratio = epd_cfg.get("teacher_forcing_ratio", 0.5)
    max_rollout_steps = epd_cfg.get("max_rollout_steps", 10)
    loss_cfg = epd_cfg.get("loss_func")
    loss_func = instantiate(loss_cfg) if loss_cfg is not None else nn.MSELoss()
    stride = training_params.stride

    # Instantiate metrics if present in the config
    metrics_kwargs = {}
    for stage in ["train", "val", "test"]:
        metric_key = f"{stage}_metrics"
        if metric_key in epd_cfg:
            # We assume the config is compatible with hydra.utils.instantiate
            # (e.g. a list of metrics or a MetricCollection config)
            metrics_val = instantiate(epd_cfg[metric_key])
            if isinstance(metrics_val, (DictConfig, dict)):
                metrics_val = [
                    instantiate(v) if isinstance(v, (DictConfig, dict)) else v
                    for v in metrics_val.values()
                ]
            metrics_kwargs[metric_key] = metrics_val

    # Check for ensemble configuration
    n_members = epd_cfg.get("n_members", 1)
    is_ensemble = int(n_members) > 1

    model_class = (
        EncoderProcessorDecoderEnsemble if is_ensemble else EncoderProcessorDecoder
    )

    model_kwargs = {
        "encoder_decoder": encoder_decoder,
        "processor": processor,
        "learning_rate": learning_rate,
        "optimizer_config": optimizer_config,
        "train_in_latent_space": train_in_latent_space,
        "stride": stride,
        "teacher_forcing_ratio": teacher_forcing_ratio,
        "max_rollout_steps": max_rollout_steps,
        "loss_func": loss_func,
        "input_noise_injector": input_noise_injector,
        **metrics_kwargs,
    }

    if is_ensemble:
        model_kwargs["n_members"] = int(n_members)

    model = model_class(**model_kwargs)

    maybe_watch_model(wandb_logger, model, watch_cfg)
    trainer = instantiate_trainer(cfg, work_dir, logger=wandb_logger)

    log.info("Starting training.")
    trainer.fit(model=model, datamodule=datamodule)

    if not args.skip_test:
        log.info("Running evaluation on the test split.")
        trainer.test(model=model, dataloaders=datamodule.test_dataloader())

    output_cfg = cfg.get("output") or {}
    checkpoint_name = output_cfg.get(
        "checkpoint_name",
        "encoder_processor_decoder.ckpt",
    )
    checkpoint_target = args.output_checkpoint or Path(checkpoint_name)
    checkpoint_path = _ensure_output_path(checkpoint_target, work_dir)
    trainer.save_checkpoint(checkpoint_path)
    log.info(
        "Saved encoder-processor-decoder checkpoint to %s",
        checkpoint_path,
    )

    if output_cfg.get("save_config", False):
        resolved_cfg_path = work_dir / "resolved_processor_config.yaml"
        OmegaConf.save(cfg, resolved_cfg_path)
        log.info("Wrote resolved config to %s", resolved_cfg_path)


if __name__ == "__main__":
    main()
