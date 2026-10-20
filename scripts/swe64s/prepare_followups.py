"""Print the matched SWE64 follow-up commands; never submit or train."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESETS = REPO_ROOT / "local_hydra/local_experiment/swe64s"


def build_plan(
    *, datasets_root: Path, output_root: Path, account: str | None = None
) -> dict[str, Any]:
    """Reuse standard CLI entry points with explicit data and allocation settings."""
    settings = json.loads((PRESETS / "followup_runs.json").read_text())
    raw = json.loads((PRESETS / "dataset.json").read_text())
    datasets_root = datasets_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    paths = {
        "raw": datasets_root / raw["directory"],
        "latent_cache": datasets_root / settings["latent_cache"]["directory"],
    }
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    clean = not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    ).strip()
    jobs = []
    for spec in settings["runs"]:
        destination = output_root / spec["name"] / "fit"
        run_name = f"swe64s-{spec['name']}-{output_root.name}"
        if destination.exists():
            msg = f"Refusing to reuse training output: {destination}"
            raise FileExistsError(msg)
        overrides = [
            f"local_experiment=swe64s/{spec['preset']}",
            f"hydra.searchpath=[file://{REPO_ROOT / 'local_hydra'}]",
            "+distributed=ddp_4gpu_slurm",
            f"trainer.max_time={spec['trainer_max_time']}",
            "++hydra.launcher.nodes=1",
            "++hydra.launcher.cpus_per_task=72",
            "++hydra.launcher.partition=workq",
            f"hydra.launcher.timeout_min={spec['allocation_minutes']}",
            f"datamodule.data_path={json.dumps(str(paths[spec['dataset']]))}",
            "logging.wandb.enabled=true",
            "logging.wandb.mode=online",
            "logging.wandb.entity=turing-core",
            "logging.wandb.project=autocast",
            f"logging.wandb.group={json.dumps('swe64s-' + output_root.name)}",
            f"logging.wandb.name={json.dumps(run_name)}",
            "logging.wandb.resume=null",
            f"++provenance.source_commit={source}",
            f"++provenance.raw_dataset_uuid={raw['uuid']}",
            "umask=0007",
        ]
        if account is not None:
            overrides.append(
                f"++hydra.launcher.additional_parameters.account={json.dumps(account)}"
            )
        if spec["dataset"] == "latent_cache":
            cache = settings["latent_cache"]
            overrides += [
                f"++provenance.dataset_uuid={cache['uuid']}",
                f"++provenance.autoencoder_checkpoint_sha256={cache['autoencoder_checkpoint_sha256']}",
                f"++provenance.normalization_sha256={cache['normalization_sha256']}",
            ]
        else:
            overrides.append(f"++provenance.dataset_uuid={raw['uuid']}")
        command = [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "autocast",
            spec["command"],
            "--mode",
            "slurm",
            "--workdir",
            str(destination),
            *overrides,
        ]
        # Preview through the same standard command, with no separate launcher.
        preview = [*command[:6], "--dry-run", *command[6:]]
        jobs.append(
            {
                **spec,
                "output": str(destination),
                "overrides": overrides,
                "launch_command": command,
                "preview_command": preview,
            }
        )
    return {
        "source_commit": source,
        "working_tree_clean": clean,
        "raw_dataset": {
            "path": str(paths["raw"]),
            "uuid": raw["uuid"],
            "splits": raw["splits"],
        },
        "latent_cache": {
            **settings["latent_cache"],
            "path": str(paths["latent_cache"]),
        },
        "checkpoint_selection": settings["checkpoint_selection"],
        "budget_note": (
            "128000 nominal updates each on four GPUs; not equal condition "
            "exposure, runtime or completed budgets if stopped early."
        ),
        "target_data_verified": False,
        "training_executed": False,
        "jobs_submitted": False,
        "jobs": jobs,
    }


def main() -> None:
    """Print a JSON plan and shell commands without creating any outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--account")
    args = parser.parse_args()
    plan = build_plan(**vars(args))
    print(json.dumps(plan, indent=2))
    for job in plan["jobs"]:
        print(f"\n# {job['name']}: preview only")
        print(shlex.join(job["preview_command"]))


if __name__ == "__main__":
    main()
