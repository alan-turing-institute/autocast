import argparse
import re
import time
from pathlib import Path

import pandas as pd
import torch
from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.models.encoder_processor_decoder_ensemble import (
    EncoderProcessorDecoderEnsemble,
)
from autocast.scripts.eval.encoder_processor_decoder import (
    _extract_state_dict,
    _load_checkpoint_payload,
)
from autocast.scripts.workflow.commands import infer_eval_checkpoint
from autocast.types.batch import Batch

DEFAULT_CONFIG = Path(__file__).parent / "inference_benchmarking.yaml"


def find_run_folder(stem: str, run_id: str) -> Path:
    """Find run folder by searching recursively under stem."""
    dirs = [p for p in Path(stem).rglob(run_id) if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"Run folder not found for: {run_id}")
    return dirs[0]


def build_model_from_config(cfg) -> EncoderProcessorDecoder:
    """Instantiate model components directly from resolved config."""
    encoder = instantiate(cfg.model.encoder)
    decoder = instantiate(cfg.model.decoder)
    processor = instantiate(cfg.model.processor)
    enc_dec = EncoderDecoder(encoder=encoder, decoder=decoder)
    n_members = cfg.model.get("n_members", 1)
    if n_members > 1:
        return EncoderProcessorDecoderEnsemble(
            encoder_decoder=enc_dec, processor=processor, n_members=n_members
        )
    return EncoderProcessorDecoder(encoder_decoder=enc_dec, processor=processor)


def _get_spatial_resolution(cfg) -> list[int]:
    spatial = cfg.model.processor.get("spatial_resolution", None)
    if spatial is not None and spatial != "auto":
        return list(spatial)

    data_path = str(cfg.datamodule.get("data_path", ""))
    match = re.search(r"_(\d+)_(\d+)$", Path(data_path).name)
    if match:
        return [int(match.group(1)), int(match.group(2))]

    raise ValueError(
        f"Cannot determine spatial resolution: processor has no 'spatial_resolution' "
        f"and data_path '{data_path}' does not end with _H_W."
    )


def _get_output_channels(cfg) -> int:
    dec = cfg.model.decoder
    ch = dec.get("output_channels", None)
    if ch is not None:
        return int(ch)
    ch = dec.get("out_channels", None)
    if ch is not None:
        return int(ch)
    raise ValueError(
        "Cannot determine output channels: decoder has neither "
        "'output_channels' nor 'out_channels' in config."
    )


def _get_global_cond_channels(cfg) -> int | None:
    candidates = [cfg.model.processor]
    backbone = cfg.model.processor.get("backbone", None)
    if backbone is not None:
        candidates.append(backbone)

    for node in candidates:
        if not node.get("include_global_cond", False):
            continue
        ch = node.get("global_cond_channels", None)
        if ch is not None and ch != "auto":
            return int(ch)

    return None


def make_synthetic_batch(cfg, batch_size: int) -> Batch:
    core_ch = _get_output_channels(cfg)
    t_in = cfg.datamodule.n_steps_input
    t_out = cfg.datamodule.n_steps_output
    spatial = _get_spatial_resolution(cfg)

    with_const = getattr(cfg.model.encoder, "with_constants", False)
    enc_in_ch = getattr(cfg.model.encoder, "in_channels", None)
    n_const = enc_in_ch - core_ch * t_in if with_const and enc_in_ch is not None else 0

    global_cond_ch = _get_global_cond_channels(cfg)
    if global_cond_ch is not None:
        n_const = global_cond_ch

    return Batch(
        input_fields=torch.randn(batch_size, t_in, *spatial, core_ch),
        output_fields=torch.randn(batch_size, t_out, *spatial, core_ch),
        constant_scalars=torch.randn(batch_size, n_const) if n_const > 0 else None,
        constant_fields=None,
        boundary_conditions=None,
    )


class SyntheticDataset(Dataset):
    def __init__(self, batches: list):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


class ThroughputCallback(Callback):
    def __init__(self, n_warmup: int, batch_size: int):
        self.n_warmup = n_warmup
        self.batch_size = batch_size
        self._batch_times: list[float] = []
        self._t: float | None = None

    def on_predict_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx >= self.n_warmup:
            self._t = time.perf_counter()

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx >= self.n_warmup and self._t is not None:
            self._batch_times.append(time.perf_counter() - self._t)

    def metrics(self) -> dict:
        total = sum(self._batch_times)
        n = len(self._batch_times)
        latency_batch_ms = total / n * 1000
        return {
            "throughput_samples_per_sec": (n * self.batch_size) / total,
            "latency_ms_per_batch": latency_batch_ms,
            "latency_ms_per_sample": latency_batch_ms / self.batch_size,
        }


def benchmark_model(
    model, cfg, n_warmup: int, n_benchmark: int, batch_size: int
) -> dict:
    batches = [
        make_synthetic_batch(cfg, batch_size) for _ in range(n_warmup + n_benchmark)
    ]
    loader = DataLoader(
        SyntheticDataset(batches), batch_size=1, collate_fn=lambda x: x[0]
    )

    callback = ThroughputCallback(n_warmup=n_warmup, batch_size=batch_size)
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        callbacks=[callback],
        enable_progress_bar=False,
        logger=False,
    )
    trainer.predict(model, dataloaders=loader)
    return callback.metrics()


def main():
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
    batch_size = bcfg.benchmark.batch_size
    n_warmup = bcfg.benchmark.n_warmup
    n_benchmark = bcfg.benchmark.n_benchmark
    run_ids = list(OmegaConf.to_container(bcfg.runs).values())

    results = []

    for run_id in run_ids:
        print(f"\n--- {run_id} ---")
        model = None
        try:
            folder = find_run_folder(stem, run_id)
            cfg = OmegaConf.load(folder / "resolved_config.yaml")

            ckpt_path = infer_eval_checkpoint(folder)
            if ckpt_path is None:
                raise FileNotFoundError(f"No checkpoint found in {folder}")
            print(f"  checkpoint: {ckpt_path.name}")

            model = build_model_from_config(cfg)
            checkpoint = _load_checkpoint_payload(ckpt_path)
            state_dict = _extract_state_dict(checkpoint)
            model.load_state_dict(state_dict, strict=True)

            metrics = benchmark_model(model, cfg, n_warmup, n_benchmark, batch_size)

            results.append(
                {
                    "run_id": run_id,
                    "processor": cfg.model.processor._target_.split(".")[-1],
                    "hidden_dim": cfg.model.processor.get("hidden_dim", "N/A"),
                    "n_members": cfg.model.get("n_members", 1),
                    "spatial": str(_get_spatial_resolution(cfg)),
                    "n_steps_in": cfg.datamodule.n_steps_input,
                    "n_steps_out": cfg.datamodule.n_steps_output,
                    "batch_size": batch_size,
                    **metrics,
                }
            )
            print(
                f"  Throughput: {metrics['throughput_samples_per_sec']:.1f} samples/s"
            )
            print(
                f"  Latency:    {metrics['latency_ms_per_batch']:.1f} ms/batch  "
                f"({metrics['latency_ms_per_sample']:.2f} ms/sample)"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"run_id": run_id, "error": str(e)})
        finally:
            if model is not None:
                del model
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")


if __name__ == "__main__":
    main()
