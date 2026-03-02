"""Benchmark CLI for encoder-processor-decoder checkpoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from autocast.benchmarking import benchmark_model
from autocast.scripts.config import save_resolved_config
from autocast.scripts.eval.encoder_processor_decoder import (
    _extract_state_dict,
    _load_checkpoint_payload,
)
from autocast.scripts.setup import setup_datamodule, setup_epd_model
from autocast.scripts.utils import get_default_config_path
from autocast.types.batch import Batch

log = logging.getLogger(__name__)


def _resolve_work_dir(work_dir: Path | None) -> Path:
    if work_dir is not None:
        return work_dir

    if HydraConfig.initialized():
        output_dir = HydraConfig.get().runtime.output_dir
        if output_dir:
            return Path(output_dir).resolve()

    return Path.cwd()


def _resolve_csv_path(eval_cfg: DictConfig, work_dir: Path) -> Path:
    benchmark_cfg = eval_cfg.get("benchmark", {})
    csv_path = benchmark_cfg.get("csv_path")
    if csv_path is not None:
        return Path(csv_path).expanduser().resolve()
    return (work_dir / "benchmark_metrics.csv").resolve()


def _resolve_checkpoint_path(eval_cfg: DictConfig, work_dir: Path) -> Path:
    checkpoint_path = eval_cfg.get("checkpoint")
    if checkpoint_path is None:
        msg = (
            "No checkpoint specified. Provide eval.checkpoint=/path/to/checkpoint.ckpt"
        )
        raise ValueError(msg)

    resolved_path = Path(checkpoint_path)
    if resolved_path.is_absolute():
        return resolved_path

    workdir_candidate = (work_dir / resolved_path).resolve()
    parent_candidate = (work_dir.parent / resolved_path).resolve()
    if workdir_candidate.exists():
        return workdir_candidate
    if parent_candidate.exists():
        return parent_candidate
    return workdir_candidate


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@hydra.main(
    version_base=None,
    config_path=get_default_config_path(),
    config_name="encoder_processor_decoder",
)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for EPD benchmark runs."""
    run_benchmark(cfg)


def run_benchmark(cfg: DictConfig, work_dir: Path | None = None) -> Path:
    """Run benchmark using an already-composed config and write a CSV."""
    logging.basicConfig(level=logging.INFO)

    umask_value = cfg.get("umask")
    if umask_value is not None:
        os.umask(int(str(umask_value), 8))
        log.info("Applied process umask %s", umask_value)

    work_dir = _resolve_work_dir(work_dir)
    eval_cfg = cfg.get("eval", {})
    benchmark_cfg = eval_cfg.get("benchmark", {})

    checkpoint_path = _resolve_checkpoint_path(eval_cfg, work_dir)

    if cfg.get("output", {}).get("save_config"):
        save_resolved_config(cfg, work_dir, filename="resolved_benchmark_config.yaml")

    datamodule, cfg, stats = setup_datamodule(cfg)
    model = setup_epd_model(cfg, stats, datamodule=datamodule)

    checkpoint_payload = _load_checkpoint_payload(checkpoint_path)
    state_dict = _extract_state_dict(checkpoint_payload)
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        msg = (
            "Checkpoint parameters do not match the instantiated model. "
            f"Missing keys: {load_result.missing_keys}. "
            f"Unexpected keys: {load_result.unexpected_keys}."
        )
        raise RuntimeError(msg)

    device = _resolve_device(eval_cfg.get("device", "auto"))
    model.to(device)
    model.eval()

    example_batch = stats.get("example_batch")
    if not isinstance(example_batch, Batch):
        msg = f"Expected Batch example for benchmarking, got {type(example_batch)}"
        raise TypeError(msg)

    batch_size = int(benchmark_cfg.get("batch_size", eval_cfg.get("batch_size", 1)))
    n_warmup = int(benchmark_cfg.get("n_warmup", 5))
    n_benchmark = int(benchmark_cfg.get("n_benchmark", 50))

    metrics = benchmark_model(
        model,
        example_batch,
        n_warmup=n_warmup,
        n_benchmark=n_benchmark,
        batch_size=batch_size,
    )

    row = {
        "checkpoint": str(checkpoint_path),
        "batch_size": batch_size,
        "n_warmup": n_warmup,
        "n_benchmark": n_benchmark,
        "device": str(device),
        **metrics,
    }

    csv_path = _resolve_csv_path(eval_cfg, work_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    log.info("Wrote benchmark CSV to %s", csv_path)
    return csv_path


if __name__ == "__main__":
    main()
