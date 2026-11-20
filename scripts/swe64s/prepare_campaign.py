"""Print nested SWE64 training and preflight commands in dry-run mode."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FIRST_RUNS = ("fft_200", "fft_1000", "fft_5000", "fixed_1000", "fixed_5000")
PREFLIGHT_RUNS = ("full_off_1000", "graph2_1000", "fixed_5000")


@dataclass(frozen=True)
class Run:
    """Scientific settings that vary across the campaign arms."""

    pool: int = 1000
    objective: str = "crps_fft"
    global_noise: bool = False
    rpb: bool = True
    hops: int | None = None
    fixed_rate: bool = False
    finetune_steps: int | None = None


RUNS = {
    "fft_200": Run(pool=200),
    "fft_1000": Run(),
    "fft_5000": Run(pool=5000),
    "fixed_1000": Run(fixed_rate=True),
    "fixed_5000": Run(pool=5000, fixed_rate=True),
    "fft_global_1000": Run(global_noise=True),
    "afcrps_spatial_1000": Run(objective="afcrps"),
    "afcrps_global_1000": Run(objective="afcrps", global_noise=True),
    "full_off_1000": Run(rpb=False),
    "graph2_1000": Run(rpb=False, hops=2),
    "graph1_1000": Run(rpb=False, hops=1),
    "full_off_continue_1000": Run(rpb=False, finetune_steps=1),
    "full_off_rollout2_1000": Run(rpb=False, finetune_steps=2),
    "graph2_continue_1000": Run(rpb=False, hops=2, finetune_steps=1),
    "graph2_rollout2_1000": Run(rpb=False, hops=2, finetune_steps=2),
}


def _job(
    name: str,
    run: Run,
    pool_root: Path,
    output_root: Path,
    source: str,
    *,
    stats_hash: str,
    account: str | None,
    preflight: bool,
    parent_checkpoint: Path | None,
) -> dict[str, Any]:
    preset = f"campaign_nested_{run.objective}"
    epochs = 512 if run.fixed_rate else 256
    minutes = 960 if run.fixed_rate else 480
    if run.fixed_rate:
        preset += "_constant"
    if run.finetune_steps is not None:
        preset += "_finetune"
        minutes = 180 if run.finetune_steps == 1 else 300
    if preflight:
        minutes = 30
        output_root /= "preflight"
    output = output_root / name / "fit"
    if output.exists():
        msg = f"Refusing to reuse output: {output}"
        raise FileExistsError(msg)
    hours, remainder = divmod(minutes - (5 if preflight else 15), 60)
    search_path = f"file://{REPO_ROOT / 'local_hydra'}"
    overrides = [
        f"local_experiment=swe64s/{preset}",
        f"hydra.searchpath=[{json.dumps(search_path)}]",
        "+distributed=ddp_4gpu_slurm",
        f"swe64_pool_root={json.dumps(str(pool_root))}",
        f"swe64_train_trajectories={run.pool}",
        f"experiment_name=swe64s_{name}",
        f"trainer.max_time=00:{hours:02d}:{remainder:02d}:00",
        f"hydra.launcher.timeout_min={minutes}",
        "++hydra.launcher.nodes=1",
        "++hydra.launcher.cpus_per_task=72",
        "++hydra.launcher.partition=workq",
        f"model.processor.rpb={str(run.rpb).lower()}",
        f"model.processor.graph_hops={json.dumps(run.hops)}",
        f"++provenance.source_commit={source}",
        f"++provenance.normalization_sha256={stats_hash}",
        f"logging.wandb.group={json.dumps(output_root.name)}",
        f"logging.wandb.name={json.dumps(f'swe64s-{name}-{output_root.name}')}",
    ]
    if account is not None:
        overrides.append(
            f"++hydra.launcher.additional_parameters.account={json.dumps(account)}"
        )
    if run.global_noise:
        overrides += [
            "model.processor.noise_mode=global",
            "model.processor.n_noise_input_channels=256",
        ]
    if run.finetune_steps is not None:
        overrides += [
            f"resume_from_checkpoint={json.dumps(str(parent_checkpoint))}",
            f"model.train_rollout_steps={run.finetune_steps}",
        ]
    if preflight:
        overrides += [
            "trainer.max_epochs=2",
            "++trainer.profiler=simple",
            "logging.wandb.enabled=false",
            "output.checkpoint_path=preflight.ckpt",
        ]
        if not run.fixed_rate:
            overrides.append(f"optimizer.cosine_epochs={epochs}")
    command = [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "autocast",
        "epd",
        "--dry-run",
        "--mode",
        "slurm",
        "--workdir",
        str(output),
        *overrides,
    ]
    job: dict[str, Any] = {
        "name": name,
        "train_trajectories": run.pool,
        "expected_updates": (
            1000 if preflight else 32000 if run.finetune_steps else epochs * 500
        ),
        "allocation_minutes": minutes,
        "train_path": str(pool_root / f"train{run.pool}"),
        "normalization_path": str(pool_root / "train200/stats.yml"),
        "output": str(output),
        "overrides": overrides,
        "preview_command": command,
    }
    if preflight:
        job["resume_preview_command"] = [
            *command[:10],
            str(output.parent / "resume"),
            *(item for item in overrides if item != "trainer.max_epochs=2"),
            "trainer.max_epochs=3",
            f"resume_from_checkpoint={json.dumps(str(output / 'preflight.ckpt'))}",
        ]
    return job


def build_plan(
    *,
    pool_root: Path,
    output_root: Path,
    run_names: list[str] | None = None,
    account: str | None = None,
    preflight: bool = False,
    parent_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Resolve command inputs without inspecting or mutating remote data."""
    names = run_names or (PREFLIGHT_RUNS if preflight else FIRST_RUNS)
    if len(names) != len(set(names)):
        msg = "Select each run only once."
        raise ValueError(msg)
    selected = [RUNS[name] for name in names]
    finetunes = [run for run in selected if run.finetune_steps is not None]
    if finetunes and (preflight or parent_checkpoint is None):
        msg = "Fine-tune previews require a parent checkpoint and no preflight flag."
        raise ValueError(msg)
    if len({run.hops for run in finetunes}) > 1:
        msg = "Select one architecture at a time with its matching parent checkpoint."
        raise ValueError(msg)
    if parent_checkpoint is not None:
        parent_checkpoint = parent_checkpoint.expanduser().resolve()
    pool_root = pool_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    clean = not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    ).strip()
    previous_manifest = (
        REPO_ROOT / "local_hydra/local_experiment/swe64s/followup_runs.json"
    )
    previous = json.loads(previous_manifest.read_text())
    stats_hash = previous["latent_cache"]["normalization_sha256"]
    return {
        "source_commit": source,
        "working_tree_clean": clean,
        "normalization_sha256": stats_hash,
        "valid_trajectories": 100,
        "test_trajectories": 100,
        "target_data_verified": False,
        "jobs_submitted": False,
        "jobs": [
            _job(
                name,
                run,
                pool_root,
                output_root,
                source,
                stats_hash=stats_hash,
                account=account,
                preflight=preflight,
                parent_checkpoint=parent_checkpoint,
            )
            for name, run in zip(names, selected, strict=True)
        ],
    }


def main() -> None:
    """Print the plan and its dry-run shell commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--account")
    parser.add_argument("--run", dest="run_names", action="append", choices=RUNS)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--parent-checkpoint", type=Path)
    args = parser.parse_args()
    plan = build_plan(**vars(args))
    print(json.dumps(plan, indent=2))
    for job in plan["jobs"]:
        print("\n" + shlex.join(job["preview_command"]))
        if "resume_preview_command" in job:
            print(shlex.join(job["resume_preview_command"]))


if __name__ == "__main__":
    main()
