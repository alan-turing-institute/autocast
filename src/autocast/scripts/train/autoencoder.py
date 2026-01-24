"""Train an autoencoder stack defined by the Hydra config."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig

from autocast.config.base import Config
from autocast.data.datamodule import SpatioTemporalDataModule
from autocast.logging import create_wandb_logger, maybe_watch_model
from autocast.models.autoencoder import AE
from autocast.scripts.train.configuration import (
    build_datamodule,
    load_config,
)
from autocast.types import Batch

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the autoencoder training utility."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the autoencoder stack defined by the Hydra config under configs/."
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
        default="autoencoder",
        help="Hydra config name to compose (defaults to 'autoencoder').",
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
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Directory Lightning should use as default_root_dir and where artifacts "
            "are written (defaults to outputs/<experiment>/<timestamp>)."
        ),
    )
    return parser.parse_args()


def _resolve_work_dir(args: argparse.Namespace, cfg: DictConfig) -> Path:
    if args.work_dir is not None:
        return args.work_dir.expanduser().resolve()
    experiment = cfg.get("experiment_name", "autoencoder")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (Path.cwd() / "outputs" / str(experiment) / timestamp).resolve()


def build_model(cfg: Config) -> AE:
    """Create an autoencoder model (encoder, decoder, loss) from config."""
    model_cfg = cfg.model
    # Handle Pydantic extra fields or DictConfig
    encoder_cfg = getattr(model_cfg, "encoder", None) or model_cfg.get("encoder")
    decoder_cfg = getattr(model_cfg, "decoder", None) or model_cfg.get("decoder")
    loss_cfg = getattr(model_cfg, "loss", None) or model_cfg.get("loss")

    encoder = instantiate(encoder_cfg)
    decoder = instantiate(decoder_cfg)
    loss = instantiate(loss_cfg) if loss_cfg is not None else None

    model = AE(encoder=encoder, decoder=decoder, loss_func=loss)

    lr = getattr(model_cfg, "learning_rate", None)
    if lr is not None:
        model.learning_rate = lr
    return model


def _batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        input_fields=batch.input_fields.to(device),
        output_fields=batch.output_fields.to(device),
        constant_scalars=(
            batch.constant_scalars.to(device)
            if batch.constant_scalars is not None
            else None
        ),
        constant_fields=(
            batch.constant_fields.to(device)
            if batch.constant_fields is not None
            else None
        ),
    )


def _heatmap_slice(tensor: torch.Tensor) -> torch.Tensor:
    data = tensor.detach().cpu()
    while data.ndim > 2:
        data = data[0]
    if data.ndim == 1:
        data = data.unsqueeze(0)
    return data


def _save_reconstructions(
    model: AE,
    datamodule: SpatioTemporalDataModule,
    work_dir: Path,
    max_batches: int = 4,
    cmap: str = "viridis",
) -> None:
    output_dir = work_dir / "reconstructions"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    loader = datamodule.test_dataloader()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch_on_device = _batch_to_device(batch, device)
            outputs, latents = model.forward_with_latent(batch_on_device)
            inputs = batch_on_device.input_fields  # B, T, W, H, C

            x = inputs[0, 0, ..., 0].clone().cpu()
            y = outputs[0, 0, ..., 0].clone().cpu()
            z = latents[0, 0, ..., 0].clone().cpu()
            fig, axs = plt.subplots(1, 4, figsize=(12, 4))
            for ax in axs:
                ax.axis("off")

            axs[0].imshow(_heatmap_slice(x), cmap=cmap)
            axs[0].set_title("Input")
            axs[1].imshow(_heatmap_slice(y), cmap=cmap)
            axs[1].set_title("Reconstruction")
            difference = y - x
            axs[2].imshow(_heatmap_slice(difference), cmap=cmap)
            axs[2].set_title("Difference")
            axs[3].imshow(_heatmap_slice(z), cmap=cmap)
            axs[3].set_title("Latent")

            fig_path = output_dir / f"batch_{idx:02d}.png"
            fig.tight_layout()
            fig.savefig(fig_path)
            plt.close(fig)
            log.info("Saved reconstruction preview to %s", fig_path)

            if idx + 1 >= max_batches:
                break


def train_autoencoder(config: Config, work_dir: Path) -> Path:
    """Train the autoencoder defined in `cfg` and return the checkpoint path."""
    log.info("Starting autoencoder experiment: %s", config.experiment_name)
    L.seed_everything(config.seed, workers=True)

    # We use model_dump for instantiation which supports resolving references
    resolved_cfg = config.model_dump()

    wandb_logger, watch_cfg = create_wandb_logger(
        config.logging.model_dump(),
        experiment_name=config.experiment_name,
        job_type="train-autoencoder",
        work_dir=work_dir,
        config={"hydra": resolved_cfg},
    )

    # Data params can be in config.data (Pydantic)
    datamodule = build_datamodule(config.data.model_dump())

    model = build_model(config)
    maybe_watch_model(wandb_logger, model, watch_cfg)

    trainer = instantiate(
        config.trainer.model_dump(), logger=wandb_logger, default_root_dir=str(work_dir)
    )
    trainer.fit(model=model, datamodule=datamodule)

    checkpoint_name = config.output.get("checkpoint_name", "autoencoder.ckpt")
    checkpoint_target = Path(checkpoint_name)
    checkpoint_path = (
        checkpoint_target
        if checkpoint_target.is_absolute()
        else (work_dir / checkpoint_target)
    )
    trainer.save_checkpoint(checkpoint_path)
    log.info("Saved checkpoint to %s", checkpoint_path.resolve())

    _save_reconstructions(model, datamodule, work_dir)

    if config.output.get("save_config", False):
        resolved_cfg_path = work_dir / "resolved_autoencoder_config.yaml"
        with open(resolved_cfg_path, "w") as f:
            yaml.dump(resolved_cfg, f)
        log.info("Wrote resolved config to %s", resolved_cfg_path.resolve())

    return checkpoint_path


def main() -> None:
    """CLI entrypoint for autoencoder training."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = compose_training_config(args)
    work_dir = _resolve_work_dir(args, cfg)
    work_dir.mkdir(parents=True, exist_ok=True)
    log.inload_config(args)
    # work_dir logic might need update if it depends on DictConfig
    # But let's check _resolve_work_dir
    if args.work_dir is not None:
        work_dir = args.work_dir.expanduser().resolve()
    else:
        experiment = cfg.experiment_name  # Access as attr
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        work_dir = (Path.cwd() / "outputs" / str(experiment) / timestamp).resolve()


if __name__ == "__main__":
    main()
