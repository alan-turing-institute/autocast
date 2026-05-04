"""Training functions and utils for AutoCast experiments."""

import logging
import os
import shutil
from pathlib import Path
from time import perf_counter
from typing import cast

import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, Timer
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


_VALID_MATMUL_PRECISIONS = {"highest", "high", "medium"}


def apply_float32_matmul_precision(
    config: DictConfig, default: str | None = None
) -> None:
    """Apply `torch.set_float32_matmul_precision` from config if set.

    Accepts "highest" (pure f32), "high" (TF32 matmul, f32 accumulate), or
    "medium" (TF32 matmul + accumulate). On Ampere+ GPUs, "high" gives most of
    the TF32 throughput win while keeping f32 accumulation — a much safer
    tradeoff than bf16-mixed for precision-sensitive objectives.

    If the config key is unset (or null), `default` is used. If `default` is
    also None, PyTorch's own default ("highest") is left untouched. Training
    entrypoints pass `default=None` (opt-in); the eval entrypoint passes
    `default="high"` for backward compatibility with the previous hardcoded
    behavior.
    """
    precision = config.get("float32_matmul_precision")
    if precision is None:
        precision = default
    if precision is None:
        return
    precision = str(precision)
    if precision not in _VALID_MATMUL_PRECISIONS:
        valid = sorted(_VALID_MATMUL_PRECISIONS)
        msg = f"float32_matmul_precision must be one of {valid}, got {precision!r}"
        raise ValueError(msg)
    torch.set_float32_matmul_precision(precision)
    log.info("Set torch.float32_matmul_precision to %r", precision)


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

    source_resolved = source_path.resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        try:
            # If target already points to the latest checkpoint, nothing to do.
            if target_path.resolve() == source_resolved:
                return True
        except OSError:
            # Broken symlink or transient filesystem state; replace below.
            pass
        target_path.unlink()

    try:
        target_parent = target_path.parent.resolve()
        relative_source = Path(os.path.relpath(source_resolved, start=target_parent))
        target_path.symlink_to(relative_source)
    except OSError:
        if target_path.exists() and target_path.resolve() == source_resolved:
            return True
        shutil.copy2(source_resolved, target_path)

    return True


def _save_or_link_checkpoint_target(trainer: L.Trainer, target_path: Path) -> None:
    if not _link_checkpoint_target_to_latest(trainer, target_path):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(target_path)
        log.info("Saved checkpoint to %s", target_path.resolve())


def _resume_weights_only(
    model: L.LightningModule,
    checkpoint_path: Path,
) -> None:
    """Load only model weights from a Lightning checkpoint.

    This intentionally does *not* restore trainer/optimizer/scheduler state.
    It is useful when extending runs that previously stopped due to limits like
    ``trainer.max_time``: full-state resume may immediately re-trigger the stop
    condition based on restored loop progress.
    """
    # PyTorch >=2.6 defaults torch.load(weights_only=True), which can reject
    # regular Lightning checkpoints that include non-tensor Python objects in
    # optimizer/callback state. For this trusted local resume path we need to
    # deserialize the full checkpoint payload to extract `state_dict`.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        msg = f"Checkpoint missing state_dict: {checkpoint_path}"
        raise ValueError(msg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("Weights-only resume: missing keys (%d)", len(missing))
    if unexpected:
        log.warning("Weights-only resume: unexpected keys (%d)", len(unexpected))


def _validate_resume_settings(
    *,
    resume_checkpoint: str | os.PathLike | None,
    resume_weights_only: bool,
    reset_resume_time_budget: bool = False,
) -> None:
    """Validate resume-related config combinations."""
    if resume_weights_only and resume_checkpoint is None:
        msg = (
            "resume_weights_only=true requires a resume checkpoint via "
            "`resume_from_checkpoint` or `output.resume_from_checkpoint`."
        )
        raise ValueError(msg)
    if reset_resume_time_budget and resume_checkpoint is None:
        msg = (
            "reset_resume_time_budget=true requires a resume checkpoint via "
            "`resume_from_checkpoint` or `output.resume_from_checkpoint`."
        )
        raise ValueError(msg)
    if reset_resume_time_budget and resume_weights_only:
        msg = (
            "reset_resume_time_budget=true is only meaningful for full-state "
            "resume; disable resume_weights_only."
        )
        raise ValueError(msg)


class CheckpointAliasSymlinkCallback(Callback):
    """Refreshes a stable checkpoint alias in the work directory during training."""

    def __init__(self, target_path: Path | str):
        self.target_path = Path(target_path)

    def _refresh_alias(self, trainer: L.Trainer):
        if not trainer.is_global_zero:
            return
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


class ResetResumeTimerCallback(Callback):
    """Reset Lightning Timer offset after restoring a full-state checkpoint."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._applied = False

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        if not self.enabled or self._applied:
            return

        reset_count = 0
        for callback in getattr(trainer, "callbacks", []):
            if isinstance(callback, Timer):
                callback._offset = 0
                reset_count += 1

        if reset_count:
            log.info(
                "Reset training timer budget on resume "
                "(cleared %d Timer callback offset%s).",
                reset_count,
                "" if reset_count == 1 else "s",
            )
        else:
            log.warning(
                "reset_resume_time_budget=true set, but no Timer callback found."
            )
        self._applied = True


def _attach_reset_timer_callback(
    trainer: L.Trainer, *, enabled: bool
) -> ResetResumeTimerCallback:
    """Insert timer-reset callback before Lightning's Timer callback.

    Timer enforces `max_time` in its `on_fit_start` hook. To ensure the restored
    elapsed-time offset is cleared first, this callback must run earlier.
    """
    reset_cb = ResetResumeTimerCallback(enabled=enabled)
    callbacks = cast(list[Callback], getattr(trainer, "callbacks", []))
    timer_index = next(
        (idx for idx, callback in enumerate(callbacks) if isinstance(callback, Timer)),
        None,
    )
    if timer_index is None:
        callbacks.append(reset_cb)
    else:
        callbacks.insert(timer_index, reset_cb)
    return reset_cb


class TrainingTimerCallback(Callback):
    """Measures wall-clock training time and persists it to the checkpoint.

    Records total training time and per-epoch durations.  Each epoch
    measurement spans the **full cycle** — training batches *and* the
    subsequent validation loop — so that the ``time-epochs`` command can
    accurately predict wall-clock budget consumption.

    Epoch boundaries are measured from one ``on_train_epoch_start`` to the
    next; the final epoch is closed out in ``on_train_end`` (which fires
    after the last validation loop).

    Note
    ----
    Lightning often saves checkpoints during ``on_train_epoch_end`` (e.g. when
    ``ModelCheckpoint(save_on_train_epoch_end=True)`` is configured). That is
    *before* ``on_train_end`` runs. To avoid mixing two meanings in one field:
    - ``training_runtime_total_s`` is only set once training has ended.
    - ``training_runtime_elapsed_s`` is a snapshot of wall-clock time *so far*.
    Consumers (e.g. eval scripts) can prefer ``*_total_s`` and fall back to
    ``*_elapsed_s`` if needed.
    """

    def __init__(self) -> None:
        self._train_start: float | None = None
        self._epoch_start: float | None = None
        self._epoch_times_s: list[float] = []
        self.training_runtime_total_s: float | None = None

    def _current_elapsed_runtime_s(self) -> float | None:
        """Return elapsed runtime since train start (wall-clock seconds)."""
        if self._train_start is None:
            return None
        return perf_counter() - self._train_start

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del trainer, pl_module
        self._train_start = perf_counter()
        self._epoch_times_s = []

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        del trainer, pl_module
        now = perf_counter()
        # Close out the *previous* epoch (training + validation + overhead).
        if self._epoch_start is not None:
            self._epoch_times_s.append(now - self._epoch_start)
        self._epoch_start = now

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del trainer, pl_module
        now = perf_counter()
        # Close out the final epoch (includes its validation loop).
        if self._epoch_start is not None:
            self._epoch_times_s.append(now - self._epoch_start)
        if self._train_start is not None:
            self.training_runtime_total_s = now - self._train_start

        # Emit human-readable timing info into stdout/stderr logs so timing
        # runs are inspectable without loading timing.ckpt.
        if self._epoch_times_s:
            n = len(self._epoch_times_s)
            mean_epoch_s = sum(self._epoch_times_s) / n
            min_epoch_s = min(self._epoch_times_s)
            max_epoch_s = max(self._epoch_times_s)
            log.info(
                "TrainingTimerCallback: epochs=%d mean=%.1fs min=%.1fs max=%.1fs",
                n,
                mean_epoch_s,
                min_epoch_s,
                max_epoch_s,
            )
            if n <= 12:
                log.info(
                    "TrainingTimerCallback epoch_times_s: %s",
                    ", ".join(f"{t:.1f}s" for t in self._epoch_times_s),
                )

    def state_dict(self) -> dict:  # type: ignore[override]
        runtime_elapsed_s = self._current_elapsed_runtime_s()
        d: dict = {
            "training_runtime_total_s": self.training_runtime_total_s,
            "training_runtime_elapsed_s": runtime_elapsed_s,
            "epoch_times_s": list(self._epoch_times_s),
        }
        if self._epoch_times_s:
            n = len(self._epoch_times_s)
            d["mean_epoch_s"] = sum(self._epoch_times_s) / n
            d["min_epoch_s"] = min(self._epoch_times_s)
            d["max_epoch_s"] = max(self._epoch_times_s)
        return d

    def load_state_dict(  # type: ignore[override]
        self, state_dict: dict
    ) -> None:
        self.training_runtime_total_s = state_dict.get("training_runtime_total_s")
        self._epoch_times_s = list(state_dict.get("epoch_times_s", []))


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

    apply_float32_matmul_precision(config)

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
    callbacks.append(TrainingTimerCallback())
    trainer_cfg["callbacks"] = callbacks

    trainer = instantiate(
        trainer_cfg,
        default_root_dir=str(work_dir),
        logger=wandb_logger,
    )

    if output_cfg.get("save_config") and trainer.is_global_zero:
        save_resolved_config(config, work_dir)

    resume_checkpoint = config.get("resume_from_checkpoint") or output_cfg.get(
        "resume_from_checkpoint"
    )
    resume_weights_only = bool(
        config.get("resume_weights_only")
        or output_cfg.get("resume_weights_only")
        or config.get("train_eval", {}).get("resume_weights_only")
    )
    reset_resume_time_budget = bool(
        config.get("reset_resume_time_budget")
        or output_cfg.get("reset_resume_time_budget")
        or config.get("train_eval", {}).get("reset_resume_time_budget")
    )
    _validate_resume_settings(
        resume_checkpoint=resume_checkpoint,
        resume_weights_only=resume_weights_only,
        reset_resume_time_budget=reset_resume_time_budget,
    )
    _attach_reset_timer_callback(
        trainer,
        enabled=(
            reset_resume_time_budget
            and resume_checkpoint is not None
            and not resume_weights_only
        ),
    )

    log.info("Starting training...")
    if resume_checkpoint is not None:
        resolved_resume_checkpoint = Path(resume_checkpoint).expanduser().resolve()
        if resume_weights_only:
            log.info(
                "Resuming training from checkpoint weights only: %s",
                resolved_resume_checkpoint,
            )
            _resume_weights_only(model, resolved_resume_checkpoint)
            trainer.fit(model=model, datamodule=datamodule)
        else:
            log.info(
                "Resuming training from checkpoint (full state): %s",
                resolved_resume_checkpoint,
            )
            trainer.fit(
                model=model,
                datamodule=datamodule,
                ckpt_path=str(resolved_resume_checkpoint),
            )
    else:
        log.info("Starting training from scratch (no resume checkpoint).")
        trainer.fit(model=model, datamodule=datamodule)

    # Save stable checkpoint target (prefer callback checkpoint) immediately
    # after fit so a checkpoint exists even if optional test later fails.
    if trainer.is_global_zero:
        _save_or_link_checkpoint_target(trainer, checkpoint_path)

        # If the stable target is a symlink, replace with a final concrete checkpoint.
        if checkpoint_path.is_symlink():
            checkpoint_path.unlink()
            trainer.save_checkpoint(checkpoint_path)

    # Ensure non-zero ranks observe the finalized checkpoint before test.
    trainer.strategy.barrier("checkpoint-alias-finalize")

    # Run testing if not skipped.
    if not skip_test:
        trainer.test(model=model, datamodule=datamodule)


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
    apply_float32_matmul_precision(config)
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
    trainer.callbacks.append(TrainingTimerCallback())
    output_cfg = config.get("output", {})
    if output_cfg.get("save_config", False) and trainer.is_global_zero:
        save_resolved_config(
            config, work_dir, filename="resolved_autoencoder_config.yaml"
        )

    resume_checkpoint = config.get("resume_from_checkpoint") or output_cfg.get(
        "resume_from_checkpoint"
    )
    resume_weights_only = bool(
        config.get("resume_weights_only") or output_cfg.get("resume_weights_only")
    )
    reset_resume_time_budget = bool(
        config.get("reset_resume_time_budget")
        or output_cfg.get("reset_resume_time_budget")
    )
    _validate_resume_settings(
        resume_checkpoint=resume_checkpoint,
        resume_weights_only=resume_weights_only,
        reset_resume_time_budget=reset_resume_time_budget,
    )
    _attach_reset_timer_callback(
        trainer,
        enabled=(
            reset_resume_time_budget
            and resume_checkpoint is not None
            and not resume_weights_only
        ),
    )
    if resume_checkpoint is not None:
        resolved_resume_checkpoint = Path(resume_checkpoint).expanduser().resolve()
        if resume_weights_only:
            log.info(
                "Resuming training from checkpoint weights only: %s",
                resolved_resume_checkpoint,
            )
            _resume_weights_only(model, resolved_resume_checkpoint)
            trainer.fit(model=model, datamodule=datamodule)
        else:
            log.info(
                "Resuming training from checkpoint (full state): %s",
                resolved_resume_checkpoint,
            )
            trainer.fit(
                model=model,
                datamodule=datamodule,
                ckpt_path=str(resolved_resume_checkpoint),
            )
    else:
        log.info("Starting training from scratch (no resume checkpoint).")
        trainer.fit(model=model, datamodule=datamodule)

    checkpoint_path = _resolve_checkpoint_path(
        work_dir,
        output_cfg,
        output_cfg.get("checkpoint_path"),
        default_name="autoencoder.ckpt",
    )
    if trainer.is_global_zero:
        trainer.save_checkpoint(checkpoint_path)
        log.info("Saved checkpoint to %s", checkpoint_path.resolve())

        _save_reconstructions(model, datamodule, work_dir)

    trainer.strategy.barrier("autoencoder-post-training-finalize")

    return checkpoint_path
