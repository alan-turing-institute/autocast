"""Utility helpers for configuring Weights & Biases logging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn


@dataclass(frozen=True)
class _WatchConfig:
    log: str | None
    log_freq: int


def _to_container(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _extract_wandb_cfg(
    logging_cfg: dict[str, Any] | DictConfig | None,
) -> dict[str, Any] | None:
    if logging_cfg is None:
        return None
    candidate = None
    if isinstance(logging_cfg, (dict, DictConfig)):
        candidate = logging_cfg.get("wandb") if "wandb" in logging_cfg else logging_cfg
    if candidate is None:
        return None
    candidate = _to_container(candidate)
    if candidate is None:
        return None
    if not isinstance(candidate, dict):
        return None
    return dict(candidate)


def _build_logger_kwargs(
    wandb_cfg: dict[str, Any],
    *,
    experiment_name: str,
    run_name: str | None,
    job_type: str | None,
    work_dir: Path | None,
    base_config: dict[str, Any] | None,
) -> tuple[dict[str, Any], _WatchConfig]:
    tags_raw = wandb_cfg.get("tags") or []
    tags = list(tags_raw) if isinstance(tags_raw, (list, tuple)) else [tags_raw]
    tags = [tag for tag in tags if tag]

    name = wandb_cfg.get("name") or run_name or experiment_name
    job = job_type or wandb_cfg.get("job_type")
    save_dir_value = wandb_cfg.get("save_dir") or work_dir
    save_dir = str(save_dir_value) if save_dir_value is not None else None

    watch_cfg_raw = wandb_cfg.get("watch") or {}
    watch_cfg = _WatchConfig(
        log=watch_cfg_raw.get("log"),
        log_freq=int(watch_cfg_raw.get("log_freq", 100)),
    )

    config_payload: dict[str, Any] = {}
    if base_config:
        config_payload.update(_to_container(base_config) or {})
    if wandb_cfg.get("config"):
        config_payload.update(_to_container(wandb_cfg["config"]) or {})

    settings = wandb_cfg.get("settings")
    settings_payload = _to_container(settings) if settings else None

    kwargs: dict[str, Any] = {
        "project": wandb_cfg.get("project", experiment_name),
        "entity": wandb_cfg.get("entity"),
        "name": name,
        "group": wandb_cfg.get("group"),
        "job_type": job,
        "tags": tags or None,
        "notes": wandb_cfg.get("notes"),
        "mode": wandb_cfg.get("mode"),
        "resume": wandb_cfg.get("resume"),
        "id": wandb_cfg.get("id"),
        "log_model": wandb_cfg.get("log_model", False),
        "save_dir": save_dir,
        "settings": settings_payload,
    }

    if config_payload:
        kwargs["config"] = config_payload

    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return kwargs, watch_cfg


def create_wandb_logger(
    logging_cfg: dict[str, Any] | DictConfig | None,
    *,
    experiment_name: str,
    run_name: str | None = None,
    job_type: str | None = None,
    work_dir: Path | None = None,
    config: dict[str, Any] | DictConfig | None = None,
) -> tuple[WandbLogger | None, _WatchConfig | None]:
    """Instantiate a WandbLogger when enabled via the Hydra logging config."""
    wandb_cfg = _extract_wandb_cfg(logging_cfg)
    if not wandb_cfg or not wandb_cfg.get("enabled", False):
        return None, None

    kwargs, watch_cfg = _build_logger_kwargs(
        wandb_cfg,
        experiment_name=experiment_name,
        run_name=run_name,
        job_type=job_type,
        work_dir=work_dir,
        base_config=_to_container(config) if config is not None else None,
    )

    logger = WandbLogger(**kwargs)
    return logger, watch_cfg


def maybe_watch_model(
    logger: WandbLogger | None,
    model: nn.Module | None,
    watch_cfg: _WatchConfig | None,
) -> None:
    """Attach gradient/parameter watching if requested by the config."""
    if logger is None or model is None or watch_cfg is None:
        return
    if watch_cfg.log is None:
        return
    logger.watch(model, log=watch_cfg.log, log_freq=watch_cfg.log_freq)


def log_metrics(
    logger: WandbLogger | None,
    metrics: dict[str, float],
    step: int | None = None,
) -> None:
    """Log scalar metrics when a trainer is not driving WandbLogger."""
    if logger is None or not metrics:
        return
    logger.log_metrics(metrics, step=step)


def create_notebook_logger(
    project: str = "autocast-notebooks",
    name: str | None = None,
    *,
    enabled: bool = True,
    tags: list[str] | None = None,
    watch: str | None = None,
) -> tuple[WandbLogger | None, _WatchConfig | None]:
    """Create a WandbLogger for notebook use with minimal configuration.

    Parameters
    ----------
    project : str, optional
        W&B project name (default: "autocast-notebooks")
    name : str, optional
        Run name (default: auto-generated by wandb)
    enabled : bool, optional
        Whether to enable wandb logging (default: True)
    tags : list[str], optional
        Tags to attach to the run
    watch : str, optional
        Model watching mode: "gradients", "parameters", "all", or None to disable

    Returns
    -------
    logger : WandbLogger | None
        Configured logger instance, or None if disabled
    watch_cfg : _WatchConfig | None
        Watch configuration for use with maybe_watch_model

    Examples
    --------
    >>> logger, watch_cfg = create_notebook_logger(
    ...     project="my-project",
    ...     name="experiment-1",
    ...     tags=["notebook", "exploration"],
    ... )
    >>> # Then pass to trainer: L.Trainer(..., logger=logger)
    >>> # And optionally: maybe_watch_model(logger, model, watch_cfg)
    """
    if not enabled:
        return None, None

    logging_cfg = {
        "wandb": {
            "enabled": True,
            "project": project,
            "name": name,
            "tags": tags or [],
            "watch": {
                "log": watch,
                "log_freq": 100,
            },
        }
    }

    return create_wandb_logger(
        logging_cfg,
        experiment_name=name or "notebook",
        job_type="notebook",
    )


__all__ = [
    "create_notebook_logger",
    "create_wandb_logger",
    "log_metrics",
    "maybe_watch_model",
]
