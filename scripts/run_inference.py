"""Run inference with a trained AutoCast checkpoint and save predictions.

General-purpose: load any AutoCast checkpoint (full EPD or processor-only +
frozen autoencoder), run it over one or more dataloader splits, and save
{"true": ..., "pred": ...} tensors to disk under --out-dir -- so downstream
consumers (e.g. a conformal-prediction pipeline) never need to import any
model-loading code, only `torch.load` the output files.

Handles both checkpoint shapes `run_evaluation()` distinguishes (see
src/autocast/scripts/eval/encoder_processor_decoder.py):
  - Full EPD checkpoint (`encoder_decoder.*` keys present): rebuilt directly
    via `setup_epd_model` + a full `state_dict` load.
  - Processor-only checkpoint (trained via `autocast train processor` against
    a separately-trained, frozen autoencoder): the autoencoder is injected
    from `autoencoder_checkpoint` (read from the run's resolved config, or
    its `eval/resolved_eval_config.yaml` as a fallback), and the datamodule
    is swapped to ambient/raw-field mode if it was cached-latents, before
    only the processor's weights are loaded.

Ensemble vs. deterministic models: you do not need to tell the script which
kind of model it's loading. If the rebuilt model exposes an `n_members`
attribute (EncoderProcessorDecoderEnsemble), a single forward call already
returns all members, and --n-members (if given) overrides that count.
Otherwise the model is treated as producing one sample per call by default
(--n-members defaults to 1 in that case, so a genuinely deterministic model
costs exactly one forward pass, not N redundant identical ones) -- pass
--n-members explicitly to draw more independent samples from a model that is
stochastic but not wrapped in an Ensemble class (e.g. flow matching sampling
fresh ODE-initial noise per call). See resolve_n_members().

Running on unseen data (e.g. a freshly-generated AutoSim dataset, not the
data the checkpoint was trained on): pass --data-path pointing at the new
dataset's directory (containing train/valid/test/data.pt, e.g. produced by
scripts/generate_cns64_autosim_data.sh). This overrides *only* where raw
data is read from -- normalization stats keep coming from the checkpoint's
own training config (`normalization_path`/`normalization_stats`), which is
what you want: a model must always be fed inputs normalized the same way it
was trained, so new data should be normalized against training-set
statistics, not its own. `SpatioTemporalDataModule` already keeps these two
things (data location vs. normalization stats) fully decoupled, so no new
normalization logic is needed here -- see docs/NORMALIZATION_CONFIG.md. Only
pass --normalization-path too if you specifically want to override the
stats as well (rare).

Examples:
    # Score the checkpoint's own val/test splits (e.g. for conformal calibration).
    python scripts/run_inference.py \\
        --run-dir outputs/crps_cns64_vit_azula_large_bed4611_c99f534 \\
        --splits val,test

    python scripts/run_inference.py \\
        --run-dir outputs/diff_cns64_flow_matching_vit_09490da_636fcc3 \\
        --splits val,test --n-members 10

    # Score unseen data from a freshly-generated AutoSim dataset instead,
    # still normalized with the training-time stats.
    python scripts/run_inference.py \\
        --run-dir outputs/crps_cns64_vit_azula_large_bed4611_c99f534 \\
        --data-path /projects/.../conditioned_navier_stokes_cp_calib \\
        --out-dir /projects/.../conditioned_navier_stokes_cp_calib/predictions \\
        --splits val,test
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from autocast.scripts.data import batch_to_device
from autocast.scripts.eval.encoder_processor_decoder import (
    _extract_processor_state_dict,
    _is_processor_only_checkpoint,
    _maybe_inject_encoder_decoder_from_autoencoder_checkpoint,
    _maybe_swap_to_ambient_datamodule,
)
from autocast.scripts.execution import (
    extract_state_dict,
    load_checkpoint_payload,
    resolve_device,
)
from autocast.scripts.setup import setup_datamodule, setup_epd_model
from autocast.scripts.workflow.commands import infer_eval_checkpoint

log = logging.getLogger("run_inference")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing the resolved config + checkpoint.",
    )
    p.add_argument("--config-name", default="resolved_config.yaml")
    p.add_argument(
        "--checkpoint-name",
        default=None,
        help=(
            "Checkpoint filename inside --run-dir. Auto-detected if omitted "
            "via infer_eval_checkpoint (tries output.checkpoint_name / "
            "eval.checkpoint / encoder_processor_decoder.ckpt / "
            "processor.ckpt / model.ckpt)."
        ),
    )
    p.add_argument(
        "--autoencoder-checkpoint",
        default=None,
        help=(
            "TODO override: only needed if the run is a processor-only "
            "checkpoint and its resolvable autoencoder_checkpoint path "
            "(from the training config or eval/resolved_eval_config.yaml) "
            "is stale/unreachable on this filesystem."
        ),
    )
    p.add_argument(
        "--data-path",
        default=None,
        help=(
            "Override datamodule.data_path to run inference on different/unseen "
            "data (e.g. a freshly-generated AutoSim dataset directory containing "
            "train/valid/test/data.pt), while keeping the checkpoint's own "
            "training-time normalization stats. See module docstring."
        ),
    )
    p.add_argument(
        "--normalization-path",
        default=None,
        help=(
            "Override datamodule.normalization_path. Rarely needed -- omit this "
            "to keep normalizing with the checkpoint's own training-time stats, "
            "which is almost always what you want, even when --data-path points "
            "at unseen data."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/predictions/",
    )
    p.add_argument(
        "--splits", default="val,test", help="Comma-separated subset of {val,test}."
    )
    p.add_argument(
        "--n-members",
        type=int,
        default=None,
        help=(
            "Number of ensemble predictions to draw per input window. If "
            "omitted: for models with a native `n_members` attribute "
            "(EncoderProcessorDecoderEnsemble), uses whatever count the model "
            "was already configured with; otherwise defaults to 1 (a single, "
            "deterministic-style forward pass) -- see resolve_n_members()."
        ),
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help=(
            "First of --n-members consecutive seeds used for stochastic "
            "sampling. Only used for models with no native `n_members` "
            "attribute -- see get_ensemble_predictions."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override datamodule batch size (defaults to the training config's).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Force device, e.g. 'cuda' or 'cpu' (default: auto).",
    )
    p.add_argument(
        "--use-ema",
        action="store_true",
        help="Use EMA weights if the checkpoint has an ema_state_dict.",
    )
    return p.parse_args()


def resolve_autoencoder_checkpoint(
    cfg: DictConfig, run_dir: Path, override: str | None
) -> str | None:
    """Find the autoencoder checkpoint path for a processor-only run.

    Checks, in order: an explicit CLI override, the run's own resolved
    config, then a prior eval run's resolved_eval_config.yaml -- Mode-1 eval
    runs sometimes only record the resolved `autoencoder_checkpoint` there,
    not in the training-time resolved_config.yaml.
    """
    if override is not None:
        return override
    existing = cfg.get("autoencoder_checkpoint")
    if existing:
        return existing
    eval_cfg_path = run_dir / "eval" / "resolved_eval_config.yaml"
    if eval_cfg_path.exists():
        eval_cfg = OmegaConf.load(eval_cfg_path)
        ae_ckpt = (
            eval_cfg.get("autoencoder_checkpoint")
            if isinstance(eval_cfg, DictConfig)
            else None
        )
        if ae_ckpt:
            log.info("Picked up autoencoder_checkpoint from %s", eval_cfg_path)
            return ae_ckpt
    return None


def apply_data_overrides(
    cfg: DictConfig, data_path: str | None, normalization_path: str | None
) -> DictConfig:
    """Point the datamodule at different raw data, keeping normalization as-is by default.

    Must be applied to the *final* resolved `cfg.datamodule` -- i.e. after any
    ambient-datamodule swap (`_maybe_swap_to_ambient_datamodule` can replace
    `cfg.datamodule` wholesale with the autoencoder's own cached training
    datamodule config), not before it, or this override would be silently
    discarded for processor-only + cached-latents checkpoints.
    """
    if data_path is None and normalization_path is None:
        return cfg
    with open_dict(cfg):
        if data_path is not None:
            cfg.datamodule.data_path = data_path
        if normalization_path is not None:
            cfg.datamodule.normalization_path = normalization_path
    return cfg


def load_trained_model(
    run_dir: Path,
    config_name: str,
    checkpoint_name: str | None,
    autoencoder_checkpoint_override: str | None,
    data_path: str | None,
    normalization_path: str | None,
    device: torch.device,
    use_ema: bool,
):
    """Rebuild a trained model + its datamodule from a run directory."""
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / config_name)
    if not isinstance(cfg, DictConfig):
        msg = f"Expected DictConfig from {run_dir / config_name}, got {type(cfg).__name__}"
        raise TypeError(msg)

    if checkpoint_name is not None:
        checkpoint_path = run_dir / checkpoint_name
    else:
        checkpoint_path = infer_eval_checkpoint(run_dir)
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Could not find a checkpoint under {run_dir}. Pass --checkpoint-name "
            "explicitly (e.g. 'processor.ckpt' or 'encoder_processor_decoder.ckpt')."
        )
    log.info("Checkpoint: %s", checkpoint_path)

    payload = load_checkpoint_payload(Path(checkpoint_path))
    processor_only = _is_processor_only_checkpoint(payload)
    log.info("processor_only checkpoint: %s", processor_only)

    if processor_only:
        autoencoder_checkpoint = resolve_autoencoder_checkpoint(
            cfg, run_dir, autoencoder_checkpoint_override
        )
        if autoencoder_checkpoint is None:
            raise RuntimeError(
                "Processor-only checkpoint but no autoencoder_checkpoint could be "
                f"resolved from {config_name}, {run_dir}/eval/resolved_eval_config.yaml, "
                "or --autoencoder-checkpoint. TODO: pass "
                "--autoencoder-checkpoint /path/to/ae_....ckpt explicitly."
            )
        with open_dict(cfg):
            cfg.autoencoder_checkpoint = autoencoder_checkpoint
        cfg = _maybe_inject_encoder_decoder_from_autoencoder_checkpoint(cfg)

        # Probe first: if the training datamodule was cached-latents, swap to
        # the raw/ambient datamodule so predictions are comparable to raw
        # ground truth. No-op if the datamodule already yields raw Batch
        # objects.
        _probe_datamodule, cfg, probe_stats = setup_datamodule(cfg)
        cfg = _maybe_swap_to_ambient_datamodule(
            cfg, eval_mode="ambient", example_batch=probe_stats.get("example_batch")
        )
        # Apply data/normalization overrides only after the ambient swap above,
        # which can replace cfg.datamodule wholesale -- applying earlier would
        # be silently discarded for processor-only + cached-latents checkpoints.
        cfg = apply_data_overrides(cfg, data_path, normalization_path)
        datamodule, cfg, stats = setup_datamodule(cfg)

        model = setup_epd_model(cfg, stats, datamodule=datamodule)
        processor_state_dict = _extract_processor_state_dict(payload, use_ema=use_ema)
        load_result = model.processor.load_state_dict(processor_state_dict, strict=True)
    else:
        cfg = apply_data_overrides(cfg, data_path, normalization_path)
        datamodule, cfg, stats = setup_datamodule(cfg)
        model = setup_epd_model(cfg, stats, datamodule=datamodule)
        load_result = model.load_state_dict(
            extract_state_dict(payload, use_ema=use_ema), strict=True
        )

    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint parameters do not match the instantiated model. "
            f"Missing keys: {load_result.missing_keys}. "
            f"Unexpected keys: {load_result.unexpected_keys}."
        )

    model = model.eval().to(device)
    return model, datamodule, cfg


def resolve_n_members(model, requested_n_members: int | None) -> int:
    """Resolve how many samples to draw per input, without requiring the
    caller to know whether the model is an ensemble or deterministic.

    Ensemble-class models (those exposing `n_members`, e.g.
    EncoderProcessorDecoderEnsemble) already carry a member count from their
    training/eval config -- used as-is unless explicitly overridden, so a
    correctly-configured ensemble checkpoint just works with no flags.
    Everything else defaults to a single sample, so a genuinely deterministic
    model costs exactly one forward pass rather than `--n-members` redundant
    identical ones. A model that is stochastic without being wrapped in an
    Ensemble class (e.g. flow matching sampling fresh ODE-initial noise per
    call) will still only draw one sample unless --n-members is passed
    explicitly -- there is no fully general, static way to detect "this
    model is secretly stochastic" across arbitrary processor types; the
    reliable way is empirical (call the model twice with different seeds and
    compare outputs), which isn't worth the extra forward pass here given
    --n-members already covers this case with one explicit flag.
    """
    if requested_n_members is not None:
        return requested_n_members
    if hasattr(model, "n_members"):
        return model.n_members
    return 1


@torch.no_grad()
def get_ensemble_predictions(
    model, batch, n_members: int, seed_base: int
) -> torch.Tensor:
    """Return predictions shaped (B, T, S1, S2, C, M)."""
    if hasattr(model, "n_members"):
        original_n_members = model.n_members
        try:
            model.n_members = n_members
            return model(batch)
        finally:
            model.n_members = original_n_members

    samples = []
    for i in range(n_members):
        torch.manual_seed(seed_base + i)
        samples.append(model(batch))
    return torch.stack(samples, dim=-1)


def collect_predictions(model, dataloader, n_members: int, seed_base: int, device):
    """Pool true/pred tensors across an entire dataloader."""
    trues, preds = [], []
    for batch in dataloader:
        batch = batch_to_device(batch, device)
        pred = get_ensemble_predictions(model, batch, n_members, seed_base)
        trues.append(batch.output_fields.detach().cpu())
        preds.append(pred.detach().cpu())
    return torch.cat(trues), torch.cat(preds)


def main() -> None:
    """Run inference over the requested splits and dump prediction tensors."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "predictions")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device or "auto")
    log.info("Device: %s", device)

    if args.data_path is not None:
        log.info(
            "Running on unseen data at %s (normalization stats still come from "
            "the checkpoint's training config unless --normalization-path is "
            "also set).",
            args.data_path,
        )

    model, datamodule, cfg = load_trained_model(
        run_dir,
        args.config_name,
        args.checkpoint_name,
        args.autoencoder_checkpoint,
        args.data_path,
        args.normalization_path,
        device,
        args.use_ema,
    )
    log.info("Model class: %s", type(model).__name__)
    log.info("datamodule.data_path: %s", cfg.get("datamodule", {}).get("data_path"))
    log.info(
        "datamodule.normalization_path: %s",
        cfg.get("datamodule", {}).get("normalization_path"),
    )

    n_members = resolve_n_members(model, args.n_members)
    log.info(
        "n_members: %d (%s)",
        n_members,
        (
            "explicit --n-members"
            if args.n_members is not None
            else (
                "model's own configured n_members"
                if hasattr(model, "n_members")
                else "default of 1"
            )
        ),
    )

    if args.batch_size is not None:
        datamodule.batch_size = args.batch_size

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        if split == "val":
            dataloader = datamodule.val_dataloader()
        elif split == "test":
            dataloader = datamodule.test_dataloader()
        else:
            raise ValueError(f"Unknown split {split!r}; expected 'val' or 'test'.")

        true, pred = collect_predictions(
            model, dataloader, n_members, args.seed_base, device
        )
        out_path = out_dir / f"{split}_predictions.pt"
        torch.save(
            {
                "true": true,
                "pred": pred,
                "n_members": n_members,
                "run_dir": str(run_dir),
                "checkpoint_path": str(
                    run_dir / args.checkpoint_name
                    if args.checkpoint_name
                    else "auto-detected"
                ),
            },
            out_path,
        )
        log.info(
            "[%s] true=%s pred=%s -> %s",
            split,
            tuple(true.shape),
            tuple(pred.shape),
            out_path,
        )


if __name__ == "__main__":
    main()
