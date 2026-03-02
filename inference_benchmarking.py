import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from autocast.benchmarking import benchmark_model
from autocast.scripts.execution import (
    extract_state_dict,
    load_checkpoint_payload,
    resolve_device,
)
from autocast.scripts.setup import setup_datamodule, setup_epd_model
from autocast.scripts.workflow.commands import infer_eval_checkpoint
from autocast.types.batch import Batch

DEFAULT_CONFIG = Path(__file__).parent / "inference_benchmarking.yaml"


def find_run_folder(stem: str, run_id: str) -> Path:
    """Find run folder by searching recursively under stem."""
    dirs = [p for p in Path(stem).rglob(run_id) if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"Run folder not found for: {run_id}")
    return dirs[0]


def _benchmark_run(
    run_id: str,
    *,
    stem: str,
    batch_size: int,
    n_warmup: int,
    n_benchmark: int,
    device: torch.device,
) -> dict[str, Any]:
    model = None
    try:
        folder = find_run_folder(stem, run_id)
        cfg = OmegaConf.load(folder / "resolved_config.yaml")
        if not isinstance(cfg, DictConfig):
            msg = f"Expected DictConfig in {folder / 'resolved_config.yaml'}"
            raise TypeError(msg)

        ckpt_path = infer_eval_checkpoint(folder)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found in {folder}")
        print(f"  checkpoint: {ckpt_path.name}")

        datamodule, cfg, stats = setup_datamodule(cfg)
        model = setup_epd_model(cfg, stats, datamodule=datamodule)

        checkpoint = load_checkpoint_payload(ckpt_path)
        state_dict = extract_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        example_batch = stats.get("example_batch")
        if not isinstance(example_batch, Batch):
            raise TypeError(f"Expected Batch example, got {type(example_batch)}")

        metrics = benchmark_model(
            model,
            example_batch,
            n_warmup=n_warmup,
            n_benchmark=n_benchmark,
            batch_size=batch_size,
        )

        print(f"  Throughput: {metrics['throughput_samples_per_sec']:.1f} samples/s")
        print(
            f"  Latency:    {metrics['latency_ms_per_batch']:.1f} ms/batch  "
            f"({metrics['latency_ms_per_sample']:.2f} ms/sample)"
        )
        if "peak_gpu_memory_mb" in metrics:
            print(f"  GPU memory: {metrics['peak_gpu_memory_mb']:.0f} MB (peak)")
        if "gflops_per_sample" in metrics:
            print(f"  FLOPs:      {metrics['gflops_per_sample']:.2f} GFLOPs/sample")

        return {
            "run_id": run_id,
            "processor": cfg.model.processor._target_.split(".")[-1],
            "hidden_dim": cfg.model.processor.get("hidden_dim", "N/A"),
            "n_members": cfg.model.get("n_members", 1),
            "n_steps_in": cfg.datamodule.n_steps_input,
            "n_steps_out": cfg.datamodule.n_steps_output,
            "batch_size": batch_size,
            "device": str(device),
            **metrics,
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        return {"run_id": run_id, "error": str(e)}
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Run benchmarking for configured run IDs and write an output CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to benchmark config YAML",
    )
    args = parser.parse_args()

    bcfg = OmegaConf.load(args.config)

    stem = bcfg.outputs_stem
    output_csv = bcfg.output_csv
    batch_size = int(bcfg.benchmark.batch_size)
    n_warmup = int(bcfg.benchmark.n_warmup)
    n_benchmark = int(bcfg.benchmark.n_benchmark)
    run_cfg = OmegaConf.to_container(bcfg.runs, resolve=True)
    if not isinstance(run_cfg, dict):
        msg = "Expected `runs` to be a mapping in benchmark config"
        raise TypeError(msg)
    run_ids = [str(value) for value in run_cfg.values()]

    results = []
    device = resolve_device()

    for run_id in run_ids:
        print(f"\n--- {run_id} ---")
        results.append(
            _benchmark_run(
                run_id,
                stem=stem,
                batch_size=batch_size,
                n_warmup=n_warmup,
                n_benchmark=n_benchmark,
                device=device,
            )
        )

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")


if __name__ == "__main__":
    main()
