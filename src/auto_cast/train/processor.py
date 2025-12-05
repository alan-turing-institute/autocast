"""Train an encoder-processor-decoder stack with optional autoencoder warm-start."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import lightning as L
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

from auto_cast.decoders import Decoder
from auto_cast.decoders.channels_last import ChannelsLast
from auto_cast.encoders import Encoder
from auto_cast.encoders.permute_concat import PermuteConcat
from auto_cast.models.ae import AE, AELoss
from auto_cast.models.encoder_decoder import EncoderDecoder
from auto_cast.models.encoder_processor_decoder import EncoderProcessorDecoder
from auto_cast.nn.fno import FNOProcessor
from auto_cast.train.autoencoder import build_datamodule

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the processor training utility."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the FNO-based processor either from scratch or by reusing a "
            "pretrained autoencoder (encoder+decoder) checkpoint."
        )
    )
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=repo_root / "configs",
        help="Path to the Hydra config directory (defaults to <repo>/configs).",
    )
    parser.add_argument(
        "--config-name",
        default="config",
        help="Hydra config name to compose (defaults to 'config').",
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
        help=(
            "Path to a Lightning checkpoint containing the pretrained autoencoder. "
            "If omitted, the encoder and decoder are trained jointly with the "
            "processor."
        ),
    )
    parser.add_argument(
        "--freeze-autoencoder",
        action="store_true",
        help="Freeze encoder/decoder weights after loading a checkpoint.",
    )
    parser.add_argument(
        "--n-steps-input",
        type=int,
        default=4,
        help="Number of input time steps for the datamodule (default: 4).",
    )
    parser.add_argument(
        "--n-steps-output",
        type=int,
        default=1,
        help="Number of output time steps for training targets (default: 1).",
    )
    parser.add_argument(
        "--with-constants",
        action="store_true",
        help="Include constant fields/scalars when encoding (PermuteConcat).",
    )
    parser.add_argument(
        "--processor-n-modes",
        type=int,
        nargs=2,
        default=(16, 16),
        metavar=("NX", "NY"),
        help="Number of Fourier modes per spatial dimension for the FNO processor.",
    )
    parser.add_argument(
        "--processor-hidden-channels",
        type=int,
        default=64,
        help="Hidden channel width for the FNO processor (default: 64).",
    )
    parser.add_argument(
        "--processor-n-layers",
        type=int,
        default=4,
        help="Number of layers in the FNO processor (default: 4).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help=(
            "Learning rate for the encoder-processor-decoder optimizer (default: 1e-3)."
        ),
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
        default=Path("encoder_processor_decoder.ckpt"),
        help="Filename to store the trained checkpoint (relative to --work-dir).",
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
    data_cfg.datamodule.n_steps_input = n_steps_input
    data_cfg.datamodule.n_steps_output = n_steps_output
    data_cfg.datamodule.autoencoder_mode = False


def compose_training_config(args: argparse.Namespace) -> DictConfig:
    """Compose the Hydra config and force datamodule settings from CLI flags."""
    config_dir = args.config_dir.resolve()
    overrides: Sequence[str] = args.overrides or []
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name, overrides=list(overrides))
    _update_data_cfg(cfg, args.n_steps_input, args.n_steps_output)
    return cfg


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


def build_autoencoder_modules(
    channel_count: int,
    time_steps: int,
    with_constants: bool,
    checkpoint: Path | None,
) -> tuple[Encoder, Decoder]:
    """Create encoder/decoder modules and optionally load checkpoint weights."""
    encoder = PermuteConcat(with_constants=with_constants)
    decoder = ChannelsLast(output_channels=channel_count, time_steps=time_steps)
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


def build_processor(
    channel_count: int,
    n_steps_input: int,
    n_steps_output: int,
    args: argparse.Namespace,
) -> FNOProcessor:
    """Instantiate the FNO processor with CLI-controlled hyperparameters."""
    return FNOProcessor(
        in_channels=channel_count * n_steps_input,
        out_channels=channel_count * n_steps_output,
        n_modes=tuple(args.processor_n_modes),
        hidden_channels=args.processor_hidden_channels,
        n_layers=args.processor_n_layers,
    )


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

    if args.n_steps_output < 1:
        msg = "n_steps_output must be >= 1 for processor training."
        raise ValueError(msg)

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = compose_training_config(args)
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
    if inferred_n_steps_input != args.n_steps_input:
        log.warning(
            "Datamodule produced %s input steps but --n-steps-input=%s; "
            "proceeding with inferred value.",
            inferred_n_steps_input,
            args.n_steps_input,
        )
    if inferred_n_steps_output != args.n_steps_output:
        log.warning(
            "Datamodule produced %s output steps but --n-steps-output=%s; "
            "proceeding with inferred value.",
            inferred_n_steps_output,
            args.n_steps_output,
        )

    encoder, decoder = build_autoencoder_modules(
        channel_count=channel_count,
        time_steps=inferred_n_steps_output,
        with_constants=args.with_constants,
        checkpoint=args.autoencoder_checkpoint,
    )
    encoder_decoder = EncoderDecoder.from_encoder_decoder(
        encoder=encoder,
        decoder=decoder,
    )

    if args.freeze_autoencoder and args.autoencoder_checkpoint is not None:
        log.info("Freezing encoder and decoder parameters.")
        _freeze_module(encoder_decoder.encoder)
        _freeze_module(encoder_decoder.decoder)

    processor = build_processor(
        channel_count=channel_count,
        n_steps_input=inferred_n_steps_input,
        n_steps_output=inferred_n_steps_output,
        args=args,
    )

    model = EncoderProcessorDecoder.from_encoder_processor_decoder(
        encoder_decoder=encoder_decoder,
        processor=processor,
        learning_rate=args.learning_rate,
        loss_func=nn.MSELoss(),
    )

    trainer = instantiate_trainer(cfg, work_dir)

    log.info("Starting training.")
    trainer.fit(model=model, datamodule=datamodule)

    if not args.skip_test:
        log.info("Running evaluation on the test split.")
        trainer.test(model=model, dataloaders=datamodule.test_dataloader())

    checkpoint_path = _ensure_output_path(args.output_checkpoint, work_dir)
    trainer.save_checkpoint(checkpoint_path)
    log.info("Saved encoder-processor-decoder checkpoint to %s", checkpoint_path)


if __name__ == "__main__":
    main()
