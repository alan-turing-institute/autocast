"""Training functions and utils for AutoCast experiments."""

import logging
import os
import shutil
from pathlib import Path

import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from autocast.data.datamodule import SpatioTemporalDataModule, TheWellDataModule
from autocast.logging import create_wandb_logger
from autocast.logging.wandb import maybe_watch_model
from autocast.models.autoencoder import AE
from autocast.scripts.config import save_resolved_config
from autocast.scripts.data import batch_to_device
from autocast.scripts.setup import setup_autoencoder_model, setup_datamodule

log = logging.getLogger(__name__)


def _resolve_checkpoint_path(
    work_dir: Path,
    output_cfg: DictConfig | dict,
    output_checkpoint_path: Path | str | None,
    default_name: str,
) -> Path:
    checkpoint_name = output_checkpoint_path or output_cfg.get(
        "checkpoint_name", default_name
    )
    checkpoint_target = Path(checkpoint_name)
    return (
        checkpoint_target
        if checkpoint_target.is_absolute()
        else (work_dir / checkpoint_target)
    )


def _latest_callback_checkpoint_path(trainer: L.Trainer) -> Path | None:
    candidates: list[Path] = []
    trainer_callbacks = getattr(trainer, "callbacks", [])
    for callback in trainer_callbacks:
        if not isinstance(callback, ModelCheckpoint):
            continue

        for candidate in (callback.last_model_path, callback.best_model_path):
            if not candidate:
                continue
            checkpoint = Path(candidate)
            if checkpoint.exists():
                candidates.append(checkpoint)

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _link_checkpoint_target_to_latest(trainer: L.Trainer, target_path: Path) -> bool:
    source_path = _latest_callback_checkpoint_path(trainer)
    if source_path is None:
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()

    try:
        source_resolved = source_path.resolve()
        target_parent = target_path.parent.resolve()
        relative_source = Path(os.path.relpath(source_resolved, start=target_parent))
        target_path.symlink_to(relative_source)
        log.info("Linked checkpoint %s -> %s", target_path, relative_source)
    except OSError:
        shutil.copy2(source_path, target_path)
        log.info("Copied checkpoint %s -> %s", source_path, target_path)

    return True


def _save_or_link_checkpoint_target(trainer: L.Trainer, target_path: Path) -> None:
    if not _link_checkpoint_target_to_latest(trainer, target_path):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(target_path)
        log.info("Saved checkpoint to %s", target_path.resolve())


class CheckpointAliasSymlinkCallback(Callback):
    """Refreshes a stable checkpoint alias in the work directory during training."""

    def __init__(self, target_path: Path | str):
        self.target_path = Path(target_path)

    def _refresh_alias(self, trainer: L.Trainer):
        _link_checkpoint_target_to_latest(trainer, self.target_path)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        del pl_module
        self._refresh_alias(trainer)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        del pl_module
        self._refresh_alias(trainer)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        del pl_module
        self._refresh_alias(trainer)


def run_training(
    config: DictConfig,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    work_dir: Path,
    skip_test: bool = False,
    output_checkpoint_path: Path | str | None = None,
    job_type: str = "train",
    run_name: str | None = None,
):
    """Standardized training loop."""
    # Ensure work_dir is a Path
    work_dir = Path(work_dir)

    # Setup logger
    logging_cfg = config.get("logging")
    logging_cfg_resolved = (
        OmegaConf.to_container(logging_cfg, resolve=True)
        if logging_cfg is not None
        else {}
    )
    wandb_logger, _watch_cfg = create_wandb_logger(
        logging_cfg_resolved,  # type: ignore TODO: fix
        experiment_name=config.get("experiment_name"),
        run_name=run_name,
        job_type=job_type,
        work_dir=work_dir,
        config={"hydra": OmegaConf.to_container(config, resolve=True)},
    )

    # Get output config and save resolved config if requested
    output_cfg = config.get("output", {})
    checkpoint_path = _resolve_checkpoint_path(
        work_dir,
        output_cfg,
        output_checkpoint_path,
        default_name="model.ckpt",
    )

    # Get trainer
    trainer_cfg = config.get("trainer")
    trainer_cfg = OmegaConf.to_container(trainer_cfg, resolve=True)
    if not isinstance(trainer_cfg, dict):
        msg = "trainer config must resolve to a mapping"
        raise TypeError(msg)

    callbacks = trainer_cfg.get("callbacks", [])
    if not isinstance(callbacks, list):
        callbacks = []

    for callback in callbacks:
        if isinstance(callback, dict) and callback.get("_target_", "").endswith(
            "ModelCheckpoint"
        ):
            callback.setdefault("save_last", "link")

    callbacks.append(CheckpointAliasSymlinkCallback(checkpoint_path))
    trainer_cfg["callbacks"] = callbacks

    trainer = instantiate(
        trainer_cfg,
        default_root_dir=str(work_dir),
        logger=wandb_logger,
    )

    if output_cfg.get("save_config"):
        save_resolved_config(config, work_dir)

    resume_checkpoint = config.get("resume_from_checkpoint") or output_cfg.get(
        "resume_from_checkpoint"
    )

    log.info("Starting training...")
    if resume_checkpoint is not None:
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=str(Path(resume_checkpoint).expanduser().resolve()),
        )
    else:
        trainer.fit(model=model, datamodule=datamodule)

    # Run testing if not skipped
    if not skip_test:
        trainer.test(model=model, datamodule=datamodule)

    # Save stable checkpoint target (prefer callback checkpoint)
    _save_or_link_checkpoint_target(trainer, checkpoint_path)

    # If the stable target is a symlink, replace it with a final concrete checkpoint.
    if checkpoint_path.is_symlink():
        checkpoint_path.unlink()
        trainer.save_checkpoint(checkpoint_path)
        log.info(
            "Overwrote checkpoint symlink with final checkpoint at %s",
            checkpoint_path.resolve(),
        )


@torch.no_grad()
def _save_reconstructions(
    model: AE,
    datamodule: SpatioTemporalDataModule | TheWellDataModule,
    work_dir: Path,
    max_batches: int = 4,
    cmap: str = "viridis",
) -> None:
    output_dir = work_dir / "reconstructions"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    loader = datamodule.test_dataloader()

    def _heatmap_slice(tensor: torch.Tensor) -> torch.Tensor:
        data = tensor.detach().cpu()
        while data.ndim > 2:
            data = data[0]
        if data.ndim == 1:
            data = data.unsqueeze(0)
        return data

    for idx, batch in enumerate(loader):
        batch_on_device = batch_to_device(batch, device)
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


def train_autoencoder(
    config: DictConfig,
    work_dir: Path,
    run_name: str | None = None,
) -> Path:
    """Train the autoencoder defined in `cfg` and return the checkpoint path."""
    log.info("Starting autoencoder experiment: %s", config.get("experiment_name"))
    L.seed_everything(config.get("seed", 42), workers=True)

    resolved_cfg = OmegaConf.to_container(config, resolve=True)

    logging_cfg = config.get("logging")
    logging_cfg_resolved = (
        OmegaConf.to_container(logging_cfg, resolve=True)
        if logging_cfg is not None
        else {}
    )
    wandb_logger, watch_cfg = create_wandb_logger(
        logging_cfg_resolved,  # type: ignore TODO: fix
        experiment_name=config.get("experiment_name"),
        run_name=run_name,
        job_type="train-autoencoder",
        work_dir=work_dir,
        config={"hydra": resolved_cfg},
    )

    datamodule, config, stats = setup_datamodule(config)

    model = setup_autoencoder_model(config, stats, datamodule=datamodule)
    maybe_watch_model(wandb_logger, model, watch_cfg)

    trainer_cfg = config.get("trainer")
    trainer_cfg = OmegaConf.to_container(trainer_cfg, resolve=True)
    trainer = instantiate(
        trainer_cfg, logger=wandb_logger, default_root_dir=str(work_dir)
    )
    output_cfg = config.get("output", {})
    if output_cfg.get("save_config", False):
        save_resolved_config(
            config, work_dir, filename="resolved_autoencoder_config.yaml"
        )

    resume_checkpoint = config.get("resume_from_checkpoint") or output_cfg.get(
        "resume_from_checkpoint"
    )
    if resume_checkpoint is not None:
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=str(Path(resume_checkpoint).expanduser().resolve()),
        )
    else:
        trainer.fit(model=model, datamodule=datamodule)

    checkpoint_name = output_cfg.get("checkpoint_name", "autoencoder.ckpt")
    checkpoint_target = Path(checkpoint_name)
    checkpoint_path = (
        checkpoint_target
        if checkpoint_target.is_absolute()
        else (work_dir / checkpoint_target)
    )
    trainer.save_checkpoint(checkpoint_path)
    log.info("Saved checkpoint to %s", checkpoint_path.resolve())

    _save_reconstructions(model, datamodule, work_dir)

    return checkpoint_path
