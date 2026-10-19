"""Preview the extension evaluations; submit only explicitly selected ready runs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("diffusion_unet_evals.yaml")


def flatten(prefix: str, values: dict) -> list[str]:
    """Pin every historical evaluation value, including nested benchmarks."""
    result = []
    for key, value in values.items():
        name = f"{prefix}.{key}"
        if isinstance(value, dict):
            result.extend(flatten(name, value))
        else:
            result.append(f"++{name}={json.dumps(value, separators=(',', ':'))}")
    return result


def checkpoint_metadata(path: Path) -> dict:
    """Read metadata without materializing the model tensors in host memory."""
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return {key: checkpoint.get(key) for key in ("epoch", "global_step")}


def prepare(run: dict, manifest: dict) -> dict:
    """Resolve one selection while retaining final-path semantics for diffusion."""
    run_dir = (ROOT / run["run_dir"]).resolve(strict=True)
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    if config["logging"]["wandb"]["name"] != run["run_id"]:
        raise ValueError(f"Wrong training run: {run_dir}")
    if config["trainer"]["max_epochs"] != run["epochs"]:
        raise ValueError(f"Changed training horizon: {run_dir}")
    profile = manifest["profiles"][run["profile"]]
    final = run_dir / profile["final_checkpoint"]
    metadata = checkpoint_metadata(final) if final.is_file() else {}
    complete = metadata.get("global_step") == run["final_global_step"] and metadata.get(
        "epoch"
    ) in {run["epochs"] - 1, run["epochs"]}
    policy = run["checkpoint"]
    if policy == "last":
        # Do not resolve a running job's processor.ckpt symlink to last.ckpt.
        # The final export replaces that symlink when training finishes.
        checkpoint = final
    else:
        patterns = {
            "best_val": "best-val-*.ckpt",
            "best_multiwinkler": "best-multiwinkler-overall-*.ckpt",
        }
        matches = sorted(run_dir.glob(f"autocast/*/checkpoints/{patterns[policy]}"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {policy} checkpoint in {run_dir}: {matches}"
            )
        checkpoint = matches[0]
    eval_dir = run_dir / profile["output_subdir"]
    settings = {**manifest["eval"], **profile["eval"]}
    settings.update(
        csv_path=str(eval_dir / "evaluation_metrics.csv"),
        video_dir=str(eval_dir / "videos"),
        rollout_snapshot_dir=str(eval_dir / "videos/snapshots"),
    )
    overrides = [
        *flatten("eval", settings),
        f"eval.checkpoint={checkpoint}",
        "eval.devices=1",
        "seed=42",
        "float32_matmul_precision=high",
        f"hydra.job.name=eval_{run['run_id']}",
        "hydra.launcher.nodes=1",
        "hydra.launcher.gpus_per_node=1",
        "hydra.launcher.tasks_per_node=1",
        "hydra.launcher.cpus_per_task=72",
        "hydra.launcher.partition=workq",
        "hydra.launcher.additional_parameters.nodes=1",
        "hydra.launcher.additional_parameters.ntasks=1",
        "hydra.launcher.additional_parameters.mem=115000M",
        f"hydra.launcher.timeout_min={profile['timeout_min']}",
    ]
    if run.get("autoencoder_checkpoint"):
        ae = (ROOT / run["autoencoder_checkpoint"]).resolve(strict=True)
        cache = Path(config["datamodule"]["data_path"]).resolve(strict=True)
        if cache != ae.parent / "cached_latents":
            raise ValueError(f"Autoencoder/cache mismatch for {run['run_id']}")
        overrides.append(f"++autoencoder_checkpoint={ae}")
    command = [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "autocast",
        "eval",
        "--mode",
        "slurm",
        "--workdir",
        str(run_dir),
        "--output-subdir",
        profile["output_subdir"],
        *overrides,
    ]
    return {
        "run_id": run["run_id"],
        "training_job": str(run["training_job"]),
        "complete": complete,
        "final_checkpoint_metadata": metadata,
        "checkpoint": str(checkpoint),
        "output_dir": str(eval_dir),
        "command": command,
    }


def main(argv: list[str] | None = None) -> None:
    """Preview all seven by default; never submit an unfinished training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", nargs="+", help="Select manifest run IDs")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        parser.error(f"Run from {ROOT}")
    manifest = yaml.safe_load(MANIFEST.read_text())
    runs = manifest["runs"]
    known = {run["run_id"] for run in runs}
    selected = set(args.run_id) if args.run_id else known
    if not selected <= known:
        parser.error(f"Unknown run IDs: {sorted(selected - known)}")
    plans = [prepare(run, manifest) for run in runs if run["run_id"] in selected]
    if args.submit:
        active = set(
            subprocess.check_output(
                ["squeue", "--noheader", f"--user={os.environ['USER']}", "--format=%i"],
                text=True,
            ).split()
        )
        for plan in plans:
            if not plan["complete"] or plan["training_job"] in active:
                parser.error(f"Training is incomplete or active: {plan['run_id']}")
            output = Path(plan["output_dir"])
            if output.exists() and any(output.iterdir()):
                parser.error(f"Evaluation output already exists: {output}")
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            parser.error("Commit or clear checkout changes before submitting")
    for plan in plans:
        state = "ready" if plan["complete"] else "waiting for training completion"
        print(f"{plan['run_id']}: {state}", flush=True)
        print(f"  checkpoint: {plan['checkpoint']}", flush=True)
        print("  resources: 1 GPU, 1 task, 72 CPUs, 115000M, shared node", flush=True)
        print(shlex.join([*plan["command"], "--dry-run"]), flush=True)
        subprocess.run([*plan["command"], "--dry-run"], check=True)
    if args.submit:
        for plan in plans:
            subprocess.run(plan["command"], check=True)


if __name__ == "__main__":
    main()
