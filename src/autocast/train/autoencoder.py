from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.base import SCMode

from autocast.data.datamodule import SpatioTemporalDataModule
from autocast.data.dataset import SpatioTemporalDataset
from autocast.logging import create_wandb_logger, maybe_watch_model
from autocast.models.ae import AE
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


def compose_training_config(args: argparse.Namespace) -> DictConfig:
    """Compose Hydra config using the provided CLI arguments."""
    config_dir = args.config_dir.resolve()
    overrides: Sequence[str] = args.overrides or []
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=args.config_name, overrides=list(overrides))


def _resolve_work_dir(args: argparse.Namespace, cfg: DictConfig) -> Path:
    if args.work_dir is not None:
        return args.work_dir.expanduser().resolve()
    experiment = cfg.get("experiment_name", "autoencoder")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (Path.cwd() / "outputs" / str(experiment) / timestamp).resolve()


def _configure_trainer_root(cfg: DictConfig, work_dir: Path) -> None:
    trainer_cfg = cfg.get("trainer")
    if trainer_cfg is None:
        return
    with open_dict(trainer_cfg):
        trainer_cfg["default_root_dir"] = str(work_dir)


def _as_dtype(name: str | None) -> torch.dtype:
    if name is None:
        return torch.float32
    if not hasattr(torch, name):
        msg = f"Unknown torch dtype '{name}'"
        raise ValueError(msg)
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        msg = f"Attribute '{name}' is not a torch.dtype"
        raise ValueError(msg)
    return dtype


def _generate_split(simulator: Any, split_cfg: DictConfig) -> dict[str, Any]:
    n_train = split_cfg.get("n_train", 0)
    n_valid = split_cfg.get("n_valid", 0)
    n_test = split_cfg.get("n_test", 0)
    log.info(
        "Generating synthetic dataset (train=%s, valid=%s, test=%s)",
        n_train,
        n_valid,
        n_test,
    )
    return {
        "train": simulator.forward_samples_spatiotemporal(n_train),
        "valid": simulator.forward_samples_spatiotemporal(n_valid),
        "test": simulator.forward_samples_spatiotemporal(n_test),
    }


def build_datamodule(cfg: DictConfig) -> SpatioTemporalDataModule:
    """Instantiate the `SpatioTemporalDataModule` described by `cfg`."""
    # If the config explicitly names a datamodule class via Hydra `_target_`,
    # instantiate and return directly
    if cfg.get("_target_") is not None:
        log.info("Instantiating datamodule from target: %s", cfg.get("_target_"))
        return instantiate(cfg)
    data_path = cfg.get("data_path")
    data = None
    if cfg.get("use_simulator"):
        simulator = instantiate(cfg.simulator)
        data = _generate_split(simulator, cfg.split)
    if data_path is None and data is None:
        msg = "Either 'data_path' or 'use_simulator' must be provided."
        raise ValueError(msg)

    dm_cfg_raw = OmegaConf.to_container(
        cfg.datamodule, resolve=True, structured_config_mode=SCMode.DICT
    )
    if dm_cfg_raw is None:
        dm_cfg_raw = {}
    if not isinstance(dm_cfg_raw, dict):  # pragma: no cover - defensive
        msg = "datamodule configuration must be a mapping"
        raise TypeError(msg)

    # Extract known parameters with proper types
    batch_size: int = dm_cfg_raw.pop("batch_size", 4)  # type: ignore[assignment]
    dtype_name = dm_cfg_raw.pop("dtype", "float32")
    dtype = _as_dtype(dtype_name) if isinstance(dtype_name, str) else dtype_name
    ftype: str = dm_cfg_raw.pop("ftype", "torch")  # type: ignore[assignment]
    dataset_cls = dm_cfg_raw.pop("dataset_cls", SpatioTemporalDataset)

    log.info("Instantiating SpatioTemporalDataModule")
    return SpatioTemporalDataModule(
        data_path=data_path,
        data=data,
        dataset_cls=dataset_cls,
        batch_size=batch_size,
        dtype=dtype,
        ftype=ftype,
        **dm_cfg_raw,  # type: ignore[arg-type]
    )


def build_model(cfg: DictConfig) -> AE:
    """Create an autoencoder model (encoder, decoder, loss) from config."""
    encoder = instantiate(cfg.encoder)
    decoder = instantiate(cfg.decoder)
    loss_cfg = cfg.get("loss")
    loss = instantiate(loss_cfg) if loss_cfg is not None else None
    model = AE(encoder=encoder, decoder=decoder, loss_func=loss)
    if cfg.get("learning_rate") is not None:
        model.learning_rate = cfg.learning_rate
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
            inputs = batch_on_device.input_fields

            fig, axs = plt.subplots(1, 4, figsize=(12, 4))
            for ax in axs:
                ax.axis("off")

            axs[0].imshow(_heatmap_slice(inputs[0]), cmap=cmap)
            axs[0].set_title("Input")
            axs[1].imshow(_heatmap_slice(outputs[0]), cmap=cmap)
            axs[1].set_title("Reconstruction")
            difference = outputs[0].detach() - inputs[0]
            axs[2].imshow(_heatmap_slice(difference), cmap=cmap)
            axs[2].set_title("Difference")
            axs[3].imshow(_heatmap_slice(latents[0]), cmap=cmap)
            axs[3].set_title("Latent")

            fig_path = output_dir / f"batch_{idx:02d}.png"
            fig.tight_layout()
            fig.savefig(fig_path)
            plt.close(fig)
            log.info("Saved reconstruction preview to %s", fig_path)

            if idx + 1 >= max_batches:
                break


def train_autoencoder(cfg: DictConfig, work_dir: Path) -> Path:
    """Train the autoencoder defined in `cfg` and return the checkpoint path."""
    log.info("Starting autoencoder experiment: %s", cfg.experiment_name)
    L.seed_everything(cfg.seed, workers=True)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger, watch_cfg = create_wandb_logger(
        cfg.get("logging"),
        experiment_name=cfg.get("experiment_name", "autoencoder"),
        job_type="train-autoencoder",
        work_dir=work_dir,
        config={"hydra": resolved_cfg} if resolved_cfg is not None else None,
    )
    datamodule = build_datamodule(cfg.data)
    model = build_model(cfg.model)
    maybe_watch_model(wandb_logger, model, watch_cfg)
    trainer = instantiate(cfg.trainer, logger=wandb_logger)
    trainer.fit(model=model, datamodule=datamodule)

    checkpoint_name = cfg.output.get("checkpoint_name", "autoencoder.ckpt")
    checkpoint_target = Path(checkpoint_name)
    checkpoint_path = (
        checkpoint_target
        if checkpoint_target.is_absolute()
        else (work_dir / checkpoint_target)
    )
    trainer.save_checkpoint(checkpoint_path)
    log.info("Saved checkpoint to %s", checkpoint_path.resolve())

    _save_reconstructions(model, datamodule, work_dir)

    if cfg.output.get("save_config", False):
        resolved_cfg_path = work_dir / "resolved_autoencoder_config.yaml"
        OmegaConf.save(cfg, resolved_cfg_path)
        log.info("Wrote resolved config to %s", resolved_cfg_path.resolve())

    return checkpoint_path


def main() -> None:
    """CLI entrypoint for autoencoder training."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = compose_training_config(args)
    work_dir = _resolve_work_dir(args, cfg)
    work_dir.mkdir(parents=True, exist_ok=True)
    _configure_trainer_root(cfg, work_dir)
    log.info("Using work directory %s", work_dir)
    train_autoencoder(cfg, work_dir)


if __name__ == "__main__":
    main()
