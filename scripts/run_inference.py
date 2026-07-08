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

A checkpoint with a stateless (parameter-free) encoder/decoder, e.g.
PermuteConcat/ChannelsLast, has no `encoder_decoder.*` keys even though it's
a genuine full EPD checkpoint -- this is reclassified using the model config
instead, mirroring `run_evaluation()`'s identical safety net.

Ensemble vs. deterministic models: you do not need to tell the script which
kind of model it's loading, or how many samples to draw -- this is inferred
automatically (see resolve_model_n_members()):
  1. `cfg.eval.n_members` from the run's own resolved config, if present --
     this is the "how many stochastic samples to evaluate with" convention
     the training pipeline already records for every run (via the `optional
     eval: encoder_processor_decoder` default group), including for models
     like flow matching that commonly have no `model.n_members` set at all.
     This is applied *before* the model is built, so e.g. a flow-matching
     checkpoint with `eval.n_members: 10` in its own config gets built as
     the Ensemble variant automatically.
  2. Otherwise, whatever `model.n_members` the checkpoint was trained with
     (e.g. 10 for an EncoderProcessorDecoderEnsemble), or 1 if the model has
     no `n_members` concept at all -- so a genuinely deterministic model
     costs exactly one forward pass, never N redundant identical ones.

Running on unseen data (e.g. a freshly-generated AutoSim dataset, not the
data the checkpoint was trained on): pass --data-path pointing at the new
dataset's directory (containing train/valid/test/data.pt). This overrides
*only* where raw data is read from -- normalization stats keep coming from
the checkpoint's own training config (`normalization_path`/
`normalization_stats`), which is what you want: a model must always be fed
inputs normalized the same way it was trained, so new data should be
normalized against training-set statistics, not its own.
`SpatioTemporalDataModule` already keeps these two things (data location vs.
normalization stats) fully decoupled -- see docs/NORMALIZATION_CONFIG.md.

Examples:
    # Score the checkpoint's own val/test splits (e.g. for conformal calibration).
    python scripts/run_inference.py \\
        --run-dir outputs/crps_cns64_vit_azula_large_bed4611_c99f534 \\
        --splits val,test

    # n_members is inferred here too, from this run's own eval.n_members if
    # present in its resolved config.
    python scripts/run_inference.py \\
        --run-dir outputs/diff_cns64_flow_matching_vit_09490da_636fcc3 \\
        --splits val,test

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
from autocast.types import Batch

log = logging.getLogger("run_inference")

# The two checkpoint filenames this script's model-loading logic actually
# knows how to distinguish (see _is_processor_only_checkpoint): the default
# `output.checkpoint_name` for `train_processor` and
# `train_encoder_processor_decoder` respectively (see configs/processor.yaml,
# configs/encoder_processor_decoder.yaml). Anything else (a renamed/
# overridden checkpoint, an autoencoder checkpoint, a raw Lightning
# `epoch=N-step=N.ckpt`) must be passed via --checkpoint-name.
DEFAULT_CHECKPOINT_NAMES = ("processor.ckpt", "encoder_processor_decoder.ckpt")


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
            "Checkpoint filename inside --run-dir. If omitted, looked for "
            "under its default name for each training mode this script "
            f"supports ({', '.join(DEFAULT_CHECKPOINT_NAMES)}); pass this "
            "explicitly if the checkpoint has a different name."
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
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/predictions/",
    )
    p.add_argument(
        "--splits", default="val,test", help="Comma-separated subset of {val,test}."
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help=(
            "First of N consecutive seeds used for stochastic sampling, "
            "where N is the number of samples inferred by "
            "resolve_model_n_members(). Only used for models with no native "
            "`n_members` attribute -- see get_ensemble_predictions."
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


def find_checkpoint(run_dir: Path, checkpoint_name: str | None) -> Path:
    """Locate the checkpoint file inside run_dir.

    If --checkpoint-name is given, use it as-is. Otherwise, look for exactly
    one of DEFAULT_CHECKPOINT_NAMES -- if the checkpoint has any other name
    (e.g. it was renamed, or belongs to a training mode this script doesn't
    handle), pass --checkpoint-name explicitly rather than guessing further.
    """
    if checkpoint_name is not None:
        return run_dir / checkpoint_name

    found = [name for name in DEFAULT_CHECKPOINT_NAMES if (run_dir / name).exists()]
    if len(found) == 1:
        return run_dir / found[0]

    raise FileNotFoundError(
        f"Could not find exactly one of {DEFAULT_CHECKPOINT_NAMES} under {run_dir} "
        f"(found: {found or 'none'}). Pass --checkpoint-name explicitly."
    )


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


def resolve_model_n_members(cfg: DictConfig) -> int | None:
    """Decide what to set cfg.model.n_members to before building the model.

    Uses `cfg.eval.n_members` -- the run's own recorded "how many stochastic
    samples to evaluate with" convention -- if present. This config key is
    populated by the `optional eval: encoder_processor_decoder` default
    group at *training* time already (see
    configs/encoder_processor_decoder.yaml), and `run_evaluation()` applies
    it the same way (`cfg.model.n_members = eval_cfg.n_members`). Concretely,
    this covers models like flow matching, which commonly have no
    `model.n_members` set (so build as a plain, non-ensemble class) but do
    have `eval.n_members: 10` recorded -- letting them get built as the
    Ensemble variant automatically. Returns None if `cfg.eval.n_members` is
    absent, leaving cfg.model.n_members (and therefore which model class
    gets built) untouched.
    """
    eval_cfg = cfg.get("eval")
    if eval_cfg is not None:
        configured = eval_cfg.get("n_members")
        if configured is not None:
            return configured
    return None


def apply_overrides(cfg: DictConfig, data_path: str | None) -> DictConfig:
    """Point the datamodule at different raw data and infer cfg.model.n_members.

    Normalization always keeps coming from the checkpoint's own training
    config (`cfg.datamodule.normalization_path`/`normalization_stats`,
    untouched here) -- a model must always be fed inputs normalized the same
    way it was trained, so new data should be normalized against
    training-set statistics, not its own. `SpatioTemporalDataModule` already
    keeps data location and normalization stats fully decoupled, so
    overriding just `data_path` is sufficient -- see
    docs/NORMALIZATION_CONFIG.md.

    Must be applied to the *final* resolved `cfg.datamodule` -- i.e. after any
    ambient-datamodule swap (`_maybe_swap_to_ambient_datamodule` can replace
    `cfg.datamodule` wholesale with the autoencoder's own cached training
    datamodule config), not before it, or the data_path override would be
    silently discarded for processor-only + cached-latents checkpoints.
    """
    n_members = resolve_model_n_members(cfg)
    with open_dict(cfg):
        if data_path is not None:
            cfg.datamodule.data_path = data_path
        if n_members is not None:
            cfg.model.n_members = n_members
    return cfg


def load_trained_model(
    run_dir: Path,
    config_name: str,
    checkpoint_name: str | None,
    autoencoder_checkpoint_override: str | None,
    data_path: str | None,
    device: torch.device,
    use_ema: bool,
):
    """Rebuild a trained model + its datamodule from a run directory."""
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / config_name)
    if not isinstance(cfg, DictConfig):
        msg = f"Expected DictConfig from {run_dir / config_name}, got {type(cfg).__name__}"
        raise TypeError(msg)

    checkpoint_path = find_checkpoint(run_dir, checkpoint_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Pass --checkpoint-name "
            "explicitly if it has a different name."
        )
    log.info("Checkpoint: %s", checkpoint_path)

    payload = load_checkpoint_payload(checkpoint_path)
    processor_only = _is_processor_only_checkpoint(payload)

    # Stateless encoders/decoders (e.g. PermuteConcat/ChannelsLast) contribute
    # no encoder_decoder.* params, so a full EPD checkpoint can look
    # processor-only from its state_dict keys alone. Reclassify using the
    # model config instead -- mirrors run_evaluation()'s identical safety net
    # in src/autocast/scripts/eval/encoder_processor_decoder.py.
    _probe_datamodule, cfg, probe_stats = setup_datamodule(cfg)
    example_batch = probe_stats.get("example_batch")
    if (
        processor_only
        and isinstance(example_batch, Batch)
        and not cfg.get("autoencoder_checkpoint")
        and cfg.get("model", {}).get("encoder") is not None
        and cfg.get("model", {}).get("decoder") is not None
    ):
        log.info(
            "Checkpoint has no encoder_decoder.* params, but the model config "
            "has its own encoder+decoder -- treating as a full EPD checkpoint "
            "with a stateless (parameter-free) encoder/decoder."
        )
        processor_only = False
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

        # If the training datamodule was cached-latents, swap to the
        # raw/ambient datamodule so predictions are comparable to raw ground
        # truth. No-op if the datamodule already yields raw Batch objects.
        # Must happen before apply_overrides below, which can be silently
        # discarded by this swap replacing cfg.datamodule wholesale.
        cfg = _maybe_swap_to_ambient_datamodule(
            cfg, eval_mode="ambient", example_batch=example_batch
        )

    cfg = apply_overrides(cfg, data_path)
    datamodule, cfg, stats = setup_datamodule(cfg)
    model = setup_epd_model(cfg, stats, datamodule=datamodule)

    if processor_only:
        processor_state_dict = _extract_processor_state_dict(payload, use_ema=use_ema)
        load_result = model.processor.load_state_dict(processor_state_dict, strict=True)
    else:
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
    return model, datamodule, cfg, checkpoint_path


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

    model, datamodule, cfg, checkpoint_path = load_trained_model(
        run_dir,
        args.config_name,
        args.checkpoint_name,
        args.autoencoder_checkpoint,
        args.data_path,
        device,
        args.use_ema,
    )
    datamodule_cfg = cfg.get("datamodule", {})
    log.info(
        "Model class: %s, data_path: %s, normalization_path: %s",
        type(model).__name__,
        datamodule_cfg.get("data_path"),
        datamodule_cfg.get("normalization_path"),
    )

    n_members = getattr(model, "n_members", 1)
    log.info("n_members (inferred): %d", n_members)

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
                "checkpoint_path": str(checkpoint_path),
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
