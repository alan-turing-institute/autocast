r"""Reproduce the paper's conformal-prediction results for the main-comparison models.

Three stages, run per model:

``predict``
    Run the evaluation in prediction-saving mode on three sets of trajectories -- the
    new simulated set, the paper dataset's validation split and its test split -- and
    write each to ``<run>/eval_conformal/predictions/<set>/rollout_tensors.pt``. The
    evaluation config is built from the paper's own resolved eval config for the run,
    so the checkpoint, ensemble size, rollout length and solver settings are exactly
    the paper's.
``calibrate``
    Calibrate on {new, paper-valid} and score on {new, paper} with
    ``autocast.scripts.conformal.calibrate``, writing ``<run>/eval_conformal/``.
``sufficiency``
    Sweep the number of calibration trajectories with
    ``autocast.scripts.conformal.sufficiency``, writing
    ``<run>/eval_conformal/data_sufficiency/``.

Every path is relative to ``--root``, a directory laid out like the Isambard project
area (``outputs/`` and ``datasets/``), so the same script runs on the cluster or on a
local mirror.

Examples
--------
Smoke-test one model on two trajectories, then run everything for it::

    python scripts/run_conformal_paper.py --root /projects/u6eo/autocast \\
        predict --model crps_ad64 --max-traj 2
    python scripts/run_conformal_paper.py --root /projects/u6eo/autocast \\
        all --model crps_ad64
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# Absolute path prefixes that appear in the paper's saved configs. After the config is
# rebuilt for ``--root``, none of them may remain unless ``--root`` is one of them.
ISAMBARD_PREFIXES = (
    "/lus/lfs1aip2/projects/u6eo/autocast",
    "/projects/u6eo/autocast",
    "/home/u6eo/",
)
EVAL_CONFIG_NAME = "eval_config"
PREDICTIONS_FILE = "rollout_tensors.pt"
N_MEMBERS = 10  # the paper's evaluation ensemble size


class PredictionSet(StrEnum):
    """Which trajectories a prediction run covers."""

    NEW = "new"
    PAPER_VALID = "paper_valid"
    PAPER_TEST = "paper_test"


class DumpSplit(StrEnum):
    """Dataset split the evaluation reads (``eval.dump_split``).

    Mirrors the eval script's own ``DumpSplit``: this launcher runs the eval as a
    subprocess and imports nothing from ``autocast``, so its ``--cfg job`` check
    stays fast (no torch or lightning import).
    """

    TEST = "test"
    VALID = "valid"


class Stage(StrEnum):
    """Pipeline stage selected on the command line."""

    PREDICT = "predict"
    CALIBRATE = "calibrate"
    SUFFICIENCY = "sufficiency"
    ALL = "all"


@dataclass(frozen=True)
class System:
    """A PDE system: the paper dataset and the new simulated set."""

    paper_dataset: str
    new_dataset: str
    balance_by_scalars: bool = False


@dataclass(frozen=True)
class Model:
    """A paper checkpoint and the eval run that produced its paper results."""

    system: str
    run: str
    checkpoint: str
    paper_eval_dir: str
    autoencoder_run: str | None = None


SYSTEMS = {
    "ad64": System("advection_diffusion_2ba25b9", "fresh_oos_20260713/ad64"),
    "cns64": System("conditioned_navier_stokes_2d_5e1f575", "fresh_oos_20260713/cns64"),
    "gpe64": System("gpe/laser_only_wake_5b51eac", "fresh_oos_20260713/gpe64"),
    # The Gray-Scott new set spans the paper's six patterns and is split evenly
    # across them.
    "gs64": System(
        "gray_scott_68b0669", "fresh_oos_20260922/gs64", balance_by_scalars=True
    ),
}

_CRPS_CKPT = "autocast/{wandb}/checkpoints/best-multiwinkler-from0p25-{tail}.ckpt"
MODELS = {
    "crps_ad64": Model(
        "ad64",
        "2026-04-24/crps_ad64_vit_azula_large_bed4611_da01a04",
        _CRPS_CKPT.format(wandb="h1pfa9dx", tail="0477-0.0001"),
        "eval_best_multiwinkler_from0p25",
    ),
    "crps_cns64": Model(
        "cns64",
        "2026-04-24/crps_cns64_vit_azula_large_bed4611_c99f534",
        _CRPS_CKPT.format(wandb="g3oxmc8x", tail="0314-0.0063"),
        "eval_best_multiwinkler_from0p25",
    ),
    "crps_gpe64": Model(
        "gpe64",
        "2026-04-24/crps_gpe64_vit_azula_large_bed4611_e0a6df5",
        _CRPS_CKPT.format(wandb="9nqjv37c", tail="0131-0.0040"),
        "eval_best_multiwinkler_from0p25",
    ),
    "crps_gs64": Model(
        "gs64",
        "2026-04-24/crps_gs64_vit_azula_large_bed4611_828a161",
        _CRPS_CKPT.format(wandb="7p2z13p6", tail="0382-0.0006"),
        "eval_best_multiwinkler_from0p25",
    ),
    "fm_ad64": Model(
        "ad64",
        "2026-04-20/diff_ad64_flow_matching_vit_09490da_dae1382",
        "processor.ckpt",
        "eval",
        "2026-04-17/ae_ad64_3a7999b_1a1e300",
    ),
    "fm_cns64": Model(
        "cns64",
        "2026-04-20/diff_cns64_flow_matching_vit_09490da_636fcc3",
        "processor.ckpt",
        "eval",
        "2026-04-17/ae_cns64_3a7999b_b9c29f8",
    ),
    "fm_gpe64": Model(
        "gpe64",
        "2026-04-20/diff_gpe64_flow_matching_vit_09490da_47bf39a",
        "processor.ckpt",
        "eval",
        "2026-04-17/ae_gpe64_3a7999b_31e1c9f",
    ),
    "fm_gs64": Model(
        "gs64",
        "2026-04-20/diff_gs64_flow_matching_vit_09490da_7e9e331",
        "processor.ckpt",
        "eval",
        "2026-04-17/ae_gs64_3a7999b_ed36b8e",
    ),
}


def _load_config(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        msg = f"expected a mapping at the top of {path}"
        raise TypeError(msg)
    return cfg


def _dataset_path(root: Path, model: Model, pred_set: PredictionSet) -> Path:
    system = SYSTEMS[model.system]
    name = system.new_dataset if pred_set is PredictionSet.NEW else system.paper_dataset
    return root / "datasets" / name


def _dump_split(pred_set: PredictionSet) -> DumpSplit:
    # The new set is stored as a dataset whose test split holds all trajectories.
    if pred_set is PredictionSet.PAPER_VALID:
        return DumpSplit.VALID
    return DumpSplit.TEST


def _datamodule(root: Path, model: Model, pred_set: PredictionSet) -> DictConfig:
    """Raw-field datamodule for a run, pointed at the requested trajectories.

    CRPS models already evaluate on raw fields, so their paper datamodule is reused.
    Flow-matching models were evaluated on cached latents; for them the datamodule the
    companion autoencoder was trained with is reused, switched from autoencoder
    training to forecasting, so channel selection and normalization match the latent
    space the processor was trained in.
    """
    run_dir = root / "outputs" / model.run
    if model.autoencoder_run is None:
        paper = _load_config(
            run_dir / model.paper_eval_dir / "resolved_eval_config.yaml"
        )
        node = paper.datamodule
    else:
        ae = _load_config(
            root
            / "outputs"
            / model.autoencoder_run
            / "resolved_autoencoder_config.yaml"
        )
        node = ae.datamodule
        node.autoencoder_mode = False
        node.full_trajectory_mode = False
    paper_stats = root / "datasets" / SYSTEMS[model.system].paper_dataset / "stats.yml"
    node.data_path = str(_dataset_path(root, model, pred_set))
    # Always normalize with the paper dataset's training statistics.
    node.normalization_path = str(paper_stats)
    return node


def _find_isambard_paths(cfg: DictConfig) -> list[str]:
    container = OmegaConf.to_container(cfg, resolve=False)
    found: list[str] = []

    def walk(value: object, key: str) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{key}.{k}" if key else str(k))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{key}[{i}]")
        elif isinstance(value, str) and value.startswith(ISAMBARD_PREFIXES):
            found.append(f"{key}={value}")

    walk(container, "")
    return found


def build_eval_config(
    root: Path,
    model: Model,
    pred_set: PredictionSet,
    out_dir: Path,
    max_traj: int | None,
    batch_size: int | None = None,
) -> DictConfig:
    """Build the eval config that saves ``model``'s predictions on ``pred_set``."""
    run_dir = root / "outputs" / model.run
    cfg = _load_config(run_dir / model.paper_eval_dir / "resolved_eval_config.yaml")
    cfg.datamodule = _datamodule(root, model, pred_set)
    if model.autoencoder_run is not None:
        cfg.autoencoder_checkpoint = str(
            root / "outputs" / model.autoencoder_run / "autoencoder.ckpt"
        )
    ev = cfg.eval
    ev.checkpoint = str(run_dir / model.checkpoint)
    ev.n_members = N_MEMBERS
    if batch_size is not None:
        ev.batch_size = batch_size
    ev.devices = 1
    ev.csv_path = str(out_dir / "evaluation_metrics.csv")
    ev.video_dir = str(out_dir / "videos")
    ev.rollout_snapshot_dir = str(out_dir / "videos" / "snapshots")
    # Dump mode reuses the rollout pass, which only runs with rollout metrics on.
    ev.compute_rollout_metrics = True
    ev.benchmark.enabled = False
    ev.benchmark_rollout.enabled = False
    # No per-trajectory videos or snapshot images: only the saved tensors are needed.
    ev.batch_indices = []
    ev.save_rollout_snapshots = False
    ev.dump_rollout_tensors = True
    ev.dump_rollout_path = str(out_dir / PREDICTIONS_FILE)
    ev.dump_split = str(_dump_split(pred_set))
    ev.dump_max_traj = max_traj
    cfg.logging.wandb.enabled = False

    if not str(root).startswith(ISAMBARD_PREFIXES):
        leftovers = _find_isambard_paths(cfg)
        if leftovers:
            msg = f"config still points at Isambard paths: {leftovers}"
            raise ValueError(msg)
    return cfg


def _dataset_split_files(dataset_dir: Path) -> list[Path]:
    """Every file one dataset directory's datamodule opens.

    The datamodule opens all three splits even when only one is scored (see
    ``_dump_split``), so a predict run needs every one of them on disk.
    """
    splits = [dataset_dir / split / "data.pt" for split in ("train", "valid", "test")]
    return [*splits, dataset_dir / "stats.yml"]


def preflight_predict(
    root: Path, key: str, pred_sets: list[PredictionSet]
) -> DictConfig:
    """Validate every artefact a real ``predict`` run would open, without running it.

    Used for the Isambard submit gate's compose-only check (``--cfg job``):
    ``isambard_submit.sh`` leg 1 runs the job script with ``AUTOCAST_PREFLIGHT=1``
    to prove the command composes before any GPU is allocated. The real
    ``predict()`` launches a subprocess per (model, set) that leg 1 must not
    run, so this checks the same artefacts a different way: ``build_eval_config``
    already fails loudly if the two YAML configs it loads are missing
    (``OmegaConf.load`` raises), but the checkpoint and dataset files it only
    stores as strings are not otherwise checked -- this checks those.

    Returns the last set's composed eval config. ``trainer.devices`` -- a
    training-run leftover the eval script itself never reads (it builds its
    ``lightning.Fabric`` from ``eval.devices``/``eval.accelerator`` instead,
    see ``encoder_processor_decoder.py``) -- is identical across sets for one
    model, so any one of them is representative for the gate's GPU-count check.
    """
    model = MODELS[key]
    run_dir = root / "outputs" / model.run
    missing: list[Path] = []

    if model.autoencoder_run is not None:
        ae_ckpt = root / "outputs" / model.autoencoder_run / "autoencoder.ckpt"
        if not ae_ckpt.is_file():
            missing.append(ae_ckpt)
    ckpt = run_dir / model.checkpoint
    if not ckpt.is_file():
        missing.append(ckpt)

    cfg: DictConfig | None = None
    for pred_set in pred_sets:
        dataset_dir = _dataset_path(root, model, pred_set)
        missing.extend(f for f in _dataset_split_files(dataset_dir) if not f.is_file())
        out_dir = run_dir / "eval_conformal" / "predictions" / str(pred_set)
        cfg = build_eval_config(root, model, pred_set, out_dir, max_traj=None)

    if missing:
        listed = "\n  ".join(str(path) for path in missing)
        msg = f"[{key}] --cfg job: missing artefact(s) this run would open:\n  {listed}"
        raise FileNotFoundError(msg)
    if cfg is None:
        msg = "pred_sets was empty -- nothing to compose"
        raise ValueError(msg)
    return cfg


def predict(
    root: Path,
    key: str,
    pred_set: PredictionSet,
    max_traj: int | None,
    batch_size: int | None = None,
) -> Path:
    """Save one model's predictions on one set; return the output directory."""
    model = MODELS[key]
    subdir = str(pred_set) if max_traj is None else f"{pred_set}_smoke{max_traj}"
    out_dir = root / "outputs" / model.run / "eval_conformal" / "predictions" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_eval_config(root, model, pred_set, out_dir, max_traj, batch_size)
    OmegaConf.save(cfg, out_dir / f"{EVAL_CONFIG_NAME}.yaml")
    cmd = [
        sys.executable,
        "-m",
        "autocast.scripts.eval.encoder_processor_decoder",
        "--config-path",
        str(out_dir),
        "--config-name",
        EVAL_CONFIG_NAME,
        f"hydra.run.dir={out_dir}",
    ]
    print(f"[{key}/{subdir}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return out_dir


def _predictions(root: Path, model: Model, pred_set: PredictionSet) -> Path:
    return (
        root
        / "outputs"
        / model.run
        / "eval_conformal"
        / "predictions"
        / str(pred_set)
        / PREDICTIONS_FILE
    )


def calibrate(root: Path, key: str, threads: int, device: str) -> None:
    """Calibrate and score all four calibration/test combinations for one model."""
    model = MODELS[key]
    cmd = [
        sys.executable,
        "-m",
        "autocast.scripts.conformal.calibrate",
        "--new",
        str(_predictions(root, model, PredictionSet.NEW)),
        "--paper-valid",
        str(_predictions(root, model, PredictionSet.PAPER_VALID)),
        "--paper-test",
        str(_predictions(root, model, PredictionSet.PAPER_TEST)),
        "--out",
        str(root / "outputs" / model.run / "eval_conformal"),
        "--threads",
        str(threads),
        "--device",
        device,
    ]
    if SYSTEMS[model.system].balance_by_scalars:
        cmd.append("--balance-by-scalars")
    print(f"[{key}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def sufficiency(root: Path, key: str, threads: int, device: str) -> None:
    """Run the calibration-set-size sweep for one model."""
    model = MODELS[key]
    cmd = [
        sys.executable,
        "-m",
        "autocast.scripts.conformal.sufficiency",
        "--new",
        str(_predictions(root, model, PredictionSet.NEW)),
        "--out",
        # The sweep writes its own data_sufficiency/ folder inside --out.
        str(root / "outputs" / model.run / "eval_conformal"),
        "--threads",
        str(threads),
        "--device",
        device,
    ]
    if SYSTEMS[model.system].balance_by_scalars:
        cmd.append("--balance-by-scalars")
    print(f"[{key}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    """Parse arguments and run the requested stage."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("stage", choices=[s.value for s in Stage])
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument(
        "--set",
        dest="pred_sets",
        action="append",
        choices=[s.value for s in PredictionSet],
        help="prediction set(s) for the predict stage (default: all three)",
    )
    parser.add_argument(
        "--max-traj",
        type=int,
        default=None,
        help="smoke test: save only this many trajectories, to a separate folder",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="eval batch size for the predict stage (default: the paper's)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="device for the calibrate and sufficiency stages",
    )
    parser.add_argument(
        "--cfg",
        choices=["job"],
        default=None,
        help=(
            "compose-only check, named after Hydra's own flag: for the predict "
            "stage, validate every artefact the run would open and print the "
            "composed eval config, without launching anything. Used by the "
            "Isambard submit gate's leg 1."
        ),
    )
    args = parser.parse_args()

    stage = Stage(args.stage)
    if args.max_traj is not None and stage in (Stage.CALIBRATE, Stage.SUFFICIENCY):
        msg = "--max-traj is for smoke-test predictions; calibration uses full sets"
        raise ValueError(msg)
    pred_sets = [PredictionSet(s) for s in args.pred_sets or list(PredictionSet)]
    if args.cfg == "job":
        if stage is not Stage.PREDICT:
            msg = "--cfg job is only implemented for the predict stage"
            raise ValueError(msg)
        cfg = preflight_predict(args.root, args.model, pred_sets)
        text = OmegaConf.to_yaml(cfg)
        print(text, end="")
        cfg_out = os.environ.get("AUTOCAST_PREFLIGHT_CFG_OUT")
        if cfg_out:
            Path(cfg_out).write_text(text)
        return
    if stage in (Stage.PREDICT, Stage.ALL):
        for pred_set in pred_sets:
            predict(args.root, args.model, pred_set, args.max_traj, args.batch_size)
    if args.max_traj is not None:
        return
    if stage in (Stage.CALIBRATE, Stage.ALL):
        calibrate(args.root, args.model, args.threads, args.device)
    if stage in (Stage.SUFFICIENCY, Stage.ALL):
        sufficiency(args.root, args.model, args.threads, args.device)


if __name__ == "__main__":
    main()
