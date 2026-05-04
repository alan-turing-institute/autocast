"""Benchmark CLI for encoder-processor-decoder checkpoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from autocast.benchmarking import benchmark_model, benchmark_rollout
from autocast.scripts.config import save_resolved_config
from autocast.scripts.execution import (
    benchmark_metric_rows,
    extract_state_dict,
    load_checkpoint_payload,
    resolve_benchmark_csv_path,
    resolve_checkpoint_path,
    resolve_device,
    resolve_hydra_work_dir,
)
from autocast.scripts.setup import setup_datamodule, setup_epd_model
from autocast.scripts.utils import get_default_config_path
from autocast.types.batch import Batch

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=get_default_config_path(),
    config_name="encoder_processor_decoder",
)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for EPD benchmark runs."""
    run_benchmark(cfg)


def run_benchmark(cfg: DictConfig, work_dir: Path | None = None) -> Path:  # noqa: PLR0915
    """Run benchmark using an already-composed config and write a CSV."""
    logging.basicConfig(level=logging.INFO)

    umask_value = cfg.get("umask")
    if umask_value is not None:
        os.umask(int(str(umask_value), 8))
        log.info("Applied process umask %s", umask_value)

    work_dir = resolve_hydra_work_dir(work_dir)
    eval_cfg = cfg.get("eval", {})
    benchmark_cfg = eval_cfg.get("benchmark", {})

    checkpoint_path = resolve_checkpoint_path(
        eval_cfg,
        work_dir,
        missing_message=(
            "No checkpoint specified. Provide eval.checkpoint=/path/to/checkpoint.ckpt"
        ),
    )

    if cfg.get("output", {}).get("save_config"):
        save_resolved_config(cfg, work_dir, filename="resolved_benchmark_config.yaml")

    datamodule, cfg, stats = setup_datamodule(cfg)
    model = setup_epd_model(cfg, stats, datamodule=datamodule)

    checkpoint_payload = load_checkpoint_payload(checkpoint_path)
    state_dict = extract_state_dict(checkpoint_payload)
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        msg = (
            "Checkpoint parameters do not match the instantiated model. "
            f"Missing keys: {load_result.missing_keys}. "
            f"Unexpected keys: {load_result.unexpected_keys}."
        )
        raise RuntimeError(msg)

    # Prefer eval.accelerator to mirror Lightning API while keeping
    # eval.device as a backwards-compatible fallback.
    accelerator = eval_cfg.get("accelerator", None)
    if accelerator is None:
        accelerator = eval_cfg.get("device", "auto")
    elif "device" in eval_cfg:
        log.warning(
            "Both eval.accelerator and deprecated eval.device were provided; "
            "using eval.accelerator=%s.",
            accelerator,
        )

    device = resolve_device(str(accelerator))
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

    rows = benchmark_metric_rows(
        benchmark_type="model",
        checkpoint_path=checkpoint_path,
        device=str(device),
        batch_size=batch_size,
        n_warmup=n_warmup,
        n_benchmark=n_benchmark,
        metrics=metrics,
    )

    rollout_benchmark_cfg = eval_cfg.get("benchmark_rollout", {})
    if rollout_benchmark_cfg.get("enabled", False):
        rb_batch_size = int(rollout_benchmark_cfg.get("batch_size", batch_size))
        rb_n_warmup = int(rollout_benchmark_cfg.get("n_warmup", 5))
        rb_n_benchmark = int(rollout_benchmark_cfg.get("n_benchmark", 20))
        rb_max_rollout_steps = int(
            rollout_benchmark_cfg.get("max_rollout_steps")
            or eval_cfg.get("max_rollout_steps", 10)
        )
        rb_free_running_only = bool(
            rollout_benchmark_cfg.get(
                "free_running_only", eval_cfg.get("free_running_only", True)
            )
        )
        data_config = cfg.get("datamodule", {})
        rb_stride = int(
            rollout_benchmark_cfg.get("stride")
            or data_config.get("rollout_stride")
            or stats["n_steps_output"]
        )
        log.info(
            (
                "Running rollout benchmark "
                "(batch_size=%s, n_warmup=%s, n_benchmark=%s, "
                "stride=%s, max_rollout_steps=%s)"
            ),
            rb_batch_size,
            rb_n_warmup,
            rb_n_benchmark,
            rb_stride,
            rb_max_rollout_steps,
        )
        rollout_metrics = benchmark_rollout(
            model,
            example_batch,
            stride=rb_stride,
            max_rollout_steps=rb_max_rollout_steps,
            n_warmup=rb_n_warmup,
            n_benchmark=rb_n_benchmark,
            batch_size=rb_batch_size,
            free_running_only=rb_free_running_only,
        )
        rows.extend(
            benchmark_metric_rows(
                benchmark_type="rollout",
                checkpoint_path=checkpoint_path,
                device=str(device),
                batch_size=rb_batch_size,
                n_warmup=rb_n_warmup,
                n_benchmark=rb_n_benchmark,
                metrics=rollout_metrics,
                stride=rb_stride,
                max_rollout_steps=rb_max_rollout_steps,
            )
        )

    csv_path = resolve_benchmark_csv_path(eval_cfg, work_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.info("Wrote benchmark CSV to %s", csv_path)
    return csv_path


if __name__ == "__main__":
    main()
