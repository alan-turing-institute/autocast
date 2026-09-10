"""Shared notebook plumbing; model choices and data generation stay in the pages."""

import json
import os
import shlex
import subprocess
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = next(
    path
    for path in Path(__file__).resolve().parents
    if (path / "pyproject.toml").exists()
)
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def prepare_tutorial(name: str) -> None:
    """Reuse complete example outputs, or run their source notebook once.

    Preparation uses a separate kernel and never writes notebook outputs back
    to the source. Incomplete existing runs need an explicit rerun by the reader,
    so setup cannot silently replace a checkpoint used by another experiment.
    """
    directory, required = {
        "autoencoder_and_latents": (
            "autoencoder",
            [
                "autoencoder/autoencoder.ckpt",
                "autoencoder/cached_latents/autoencoder_config.yaml",
                "autoencoder/cached_latents/metadata.json",
                *[
                    f"autoencoder/data/{split}/data.pt"
                    for split in ("train", "valid", "test")
                ],
                *[
                    f"autoencoder/cached_latents/{split}/*.pt"
                    for split in ("train", "valid", "test")
                ],
            ],
        ),
        "diffusion_and_flow_matching": (
            "generative_processors",
            ["processor_runs.json"],
        ),
        "deterministic_ensembles": (
            "deterministic_ensemble",
            [
                "deterministic_ensemble/encoder_processor_decoder.ckpt",
                "deterministic_ensemble/resolved_config.yaml",
                *[
                    f"deterministic_ensemble/data/{split}/data.pt"
                    for split in ("train", "valid", "test")
                ],
            ],
        ),
    }[name]
    manifest = OUTPUT_ROOT / "processor_runs.json"
    if name == "diffusion_and_flow_matching" and manifest.is_file():
        selected_runs = json.loads(manifest.read_text())
        required.extend(
            f"{run}/{filename}"
            for run in selected_runs.values()
            for filename in ("processor.ckpt", "resolved_config.yaml")
        )
    if all(
        any(path.is_file() for path in OUTPUT_ROOT.glob(pattern))
        for pattern in required
    ):
        print(f"Reusing outputs from {name}.ipynb")
        return
    if (OUTPUT_ROOT / directory).exists() or (
        name == "diffusion_and_flow_matching" and manifest.exists()
    ):
        message = (
            f"Incomplete outputs for {name}.ipynb; rerun that notebook explicitly. "
            "Existing files have been left unchanged."
        )
        raise RuntimeError(message)
    print(f"Preparing small CPU example with {name}.ipynb ...", flush=True)
    path = Path(__file__).resolve().parent / f"{name}.ipynb"
    NotebookClient(
        nbformat.read(path, as_version=4),
        timeout=180,
        allow_errors=False,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    print(f"Prepared outputs from {name}.ipynb", flush=True)


# Bound both metric passes, even when a reader substitutes a larger dataset.
EVALUATION_OPTIONS = [
    "eval.accelerator=cpu",
    "++eval.max_test_batches=2",
    "++eval.max_rollout_batches=2",
    "++eval.batch_size=1",
    "eval.benchmark.enabled=false",
    "eval.benchmark_rollout.enabled=false",
    "eval.save_rollout_snapshots=false",
]


def run_autocast(*arguments: str | Path) -> None:
    """Run the CLI in the already-installed environment, with two CPU threads."""
    command = ["uv", "run", "--no-sync", "autocast", *map(str, arguments)]
    print(shlex.join(command))
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
        check=True,
    )
