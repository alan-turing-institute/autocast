"""Train an encoder-processor-decoder with optional autoencoder warm-start."""

from __future__ import annotations

import argparse
import logging
from argparse import BooleanOptionalAction
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn

from auto_cast.models.ae import AE, AELoss
from auto_cast.models.encoder_decoder import EncoderDecoder
from auto_cast.models.encoder_processor_decoder import EncoderProcessorDecoder
from auto_cast.train.autoencoder import build_datamodule

log = logging.getLogger(__name__)


@dataclass
class TrainingParams:
    """Resolved training knobs that may come from config or CLI overrides."""

    n_steps_input: int
    n_steps_output: int
    autoencoder_checkpoint: Path | None
    freeze_autoencoder: bool


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
        default="processor",
        help="Hydra config name to compose (defaults to 'processor').",
    )
    parser.add_argument(
        "--override",
        dest="overrides",
        action="append",
        default=[],
        help="Optional Hydra override, e.g. --override trainer.max_epochs=5",
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


def _get_field(batch, primary: str, fallback: str):
    return (
        getattr(batch, primary) if hasattr(batch, primary) else getattr(batch, fallback)
    )


def _freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _ensure_output_path(path: Path, work_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (work_dir / path).resolve()


def _update_data_cfg(cfg: DictConfig, n_steps_input: int, n_steps_output: int) -> None:
    data_cfg = cfg.data
    with open_dict(data_cfg.datamodule):
        data_cfg.datamodule.n_steps_input = n_steps_input
        data_cfg.datamodule.n_steps_output = n_steps_output
        data_cfg.datamodule.autoencoder_mode = False


def compose_training_config(args: argparse.Namespace) -> DictConfig:
    """Compose the Hydra config prior to applying CLI-specific overrides."""
    config_dir = args.config_dir.resolve()
    overrides: Sequence[str] = args.overrides or []
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=args.config_name, overrides=list(overrides))


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


def _configure_module_dimensions(
    cfg: DictConfig,
    channel_count: int,
    n_steps_input: int,
    n_steps_output: int,
) -> None:
    _maybe_set(cfg.decoder, "output_channels", channel_count)
    _maybe_set(cfg.decoder, "time_steps", n_steps_output)
    _maybe_set(cfg.processor, "in_channels", channel_count * n_steps_input)
    _maybe_set(cfg.processor, "out_channels", channel_count * n_steps_output)


def resolve_training_params(
    cfg: DictConfig, args: argparse.Namespace
) -> TrainingParams:
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


def instantiate_trainer(cfg: DictConfig, work_dir: Path):
    """Instantiate the Lightning trainer with a concrete root directory."""
    return instantiate(
        cfg.trainer,
        default_root_dir=str(work_dir),
    )


def main() -> None:
    """CLI entrypoint for training the processor."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = compose_training_config(args)
    training_params = resolve_training_params(cfg, args)
    _update_data_cfg(
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

    _configure_module_dimensions(
        cfg,
        channel_count=channel_count,
        n_steps_input=inferred_n_steps_input,
        n_steps_output=inferred_n_steps_output,
    )

    encoder, decoder = build_autoencoder_modules(
        cfg.encoder,
        cfg.decoder,
        training_params.autoencoder_checkpoint,
    )
    encoder_decoder = EncoderDecoder.from_encoder_decoder(
        encoder=encoder,
        decoder=decoder,
    )

    if training_params.freeze_autoencoder and training_params.autoencoder_checkpoint:
        log.info("Freezing encoder and decoder parameters.")
        _freeze_module(encoder_decoder.encoder)
        _freeze_module(encoder_decoder.decoder)

    processor = instantiate(cfg.processor)

    epd_cfg = cfg.get("encoder_processor_decoder")
    learning_rate = epd_cfg.get("learning_rate", 1e-3) if epd_cfg is not None else 1e-3
    loss_cfg = epd_cfg.get("loss_func") if epd_cfg is not None else None
    loss_func = instantiate(loss_cfg) if loss_cfg is not None else nn.MSELoss()

    model = EncoderProcessorDecoder.from_encoder_processor_decoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        learning_rate=learning_rate,
        loss_func=loss_func,
    )

    trainer = instantiate_trainer(cfg, work_dir)

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
