"""Preview or submit the measured U-Net extensions from their committed manifest."""

from __future__ import annotations

import argparse
import subprocess
from datetime import date
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from autocast.scripts.workflow.commands import build_train_overrides
from autocast.scripts.workflow.slurm import _resolve_launcher_submission_context

ROOT = Path(__file__).resolve().parents[2]
RUN_IDS = {"unet_m8_crps_ad", "unet_m8_crps_gs", "unet_m8_crps_gpe"}


def _validate_config(run: dict) -> None:
    budget = run["epoch_budget"]
    if (
        run.get("production_budget_ready") is not True
        or type(budget) is not int
        or budget <= 0
    ):
        raise ValueError(f"{run['run_id']}: a measured production budget is required")
    checkpoint = ROOT / run["timing_checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing measured timing checkpoint: {checkpoint}")
    timed = OmegaConf.load(checkpoint.parent / ".hydra/config.yaml")
    cfg = compose(config_name="encoder_processor_decoder", overrides=run["overrides"])
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        msg = "Expected a mapping for the production config"
        raise TypeError(msg)

    # The measured model, data and callbacks stay fixed; only the production
    # horizon, time cap, logging and output settings change from the timing run.
    expected_optimizer = OmegaConf.to_container(timed.optimizer, resolve=True)
    expected_trainer = OmegaConf.to_container(timed.trainer, resolve=True)
    if not isinstance(expected_optimizer, dict) or not isinstance(
        expected_trainer, dict
    ):
        msg = "Expected optimizer and trainer mappings in the timing config"
        raise TypeError(msg)
    expected_optimizer["cosine_epochs"] = budget
    expected_trainer.update(max_epochs=budget, max_time="00:23:59:00")
    expected = {
        "model": OmegaConf.to_container(timed.model, resolve=True),
        "datamodule": OmegaConf.to_container(timed.datamodule, resolve=True),
        "optimizer": expected_optimizer,
        "trainer": expected_trainer,
        "seed": timed.seed,
        "float32_matmul_precision": timed.float32_matmul_precision,
    }
    for key, value in expected.items():
        if resolved[key] != value:
            raise ValueError(f"{run['run_id']}: {key} differs from its measured config")
    if (
        not cfg.logging.wandb.enabled
        or not cfg.output.skip_test
        or cfg.output.checkpoint_path == "timing.ckpt"
    ):
        raise ValueError(
            f"{run['run_id']}: expected production logging/output settings"
        )


def _prepare_launches(run_group: str) -> list[tuple[Path, list[str]]]:
    manifest_path = ROOT / "slurm_scripts/ablations/diffusion_unet_extensions.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    runs = [run for run in manifest["runs"] if run["kind"] == "epd"]
    if len(runs) != len(RUN_IDS) or {run["run_id"] for run in runs} != RUN_IDS:
        msg = "Expected exactly the AD, GS and GPE U-Net production runs"
        raise ValueError(msg)
    launches = []
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/autocast/configs")
    ):
        for run in runs:
            _validate_config(run)
            workdir, run_id, overrides = build_train_overrides(
                kind="epd",
                mode="slurm",
                dataset=run["dataset"],
                output_base="outputs",
                work_dir=None,
                resume_from=None,
                overrides=run["overrides"],
                run_group=run_group,
            )
            launcher, _ = _resolve_launcher_submission_context(overrides)
            if (
                launcher["gpus_per_node"] != 4
                or launcher["tasks_per_node"] != 4
                or launcher["timeout_min"] != 1439
            ):
                raise ValueError(f"{run['run_id']}: expected four GPUs and 23h59m")
            command = [
                "uv",
                "run",
                "--frozen",
                "--no-sync",
                "autocast",
                "epd",
                "--mode",
                "slurm",
                "--run-group",
                run_group,
                "--run-id",
                run_id,
                *run["overrides"],
            ]
            print(
                f"Validated {run['run_id']}: {run['epoch_budget']} epochs", flush=True
            )
            launches.append((workdir, command))
    return launches


def _check_output_group(launches: list[tuple[Path, list[str]]]) -> None:
    for workdir, _ in launches:
        # Generated names end in the source commit and UUID. Ignore those two
        # fields when detecting an earlier submission in the requested group.
        prefix = workdir.name.rsplit("_", 2)[0]
        existing = sorted(workdir.parent.glob(f"{prefix}_*"))
        if existing:
            raise FileExistsError(
                f"Production run already exists in this group: {existing[0]}. "
                "For an intentional repeat, choose a fresh --run-group."
            )


def main(argv: list[str] | None = None) -> None:
    """Preview by default; submit only with an explicit flag and fresh output group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true", help="Submit after previewing")
    parser.add_argument(
        "--run-group",
        default=f"{date.today().isoformat()}/diffusion_unet_extensions",
        help="Group beneath outputs/; use a fresh group for an intentional repeat",
    )
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        parser.error(f"Run from the checkout root: {ROOT}")
    group = Path(args.run_group)
    if group.is_absolute() or ".." in group.parts or str(group) == ".":
        parser.error("--run-group must be a nonempty relative path beneath outputs/")
    launches = _prepare_launches(args.run_group)
    if args.submit:
        _check_output_group(launches)
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            parser.error("Commit or clear checkout changes before submitting")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"Source commit: {revision}", flush=True)
    for _, command in launches:
        subprocess.run([*command, "--dry-run"], check=True)
    if not args.submit:
        print("Preview only: no jobs submitted. Use --submit to launch.", flush=True)
        return
    for workdir, command in launches:
        _check_output_group([(workdir, command)])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
