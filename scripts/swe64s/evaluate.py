"""Preview or run shared metric evaluations from saved SWE64 training configs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESET = REPO_ROOT / "local_hydra/local_experiment/swe64s/evaluation.yaml"


def _overrides(prefix: str, value: Any) -> list[str]:
    """Flatten only the shared eval settings into struct-safe Hydra overrides."""
    if isinstance(value, dict):
        return [
            item
            for key, nested in value.items()
            for item in _overrides(f"{prefix}.{key}", nested)
        ]
    return [f"++{prefix}={json.dumps(value)}"]


def build_commands(
    *,
    run_dir: Path,
    checkpoint: Path,
    output_root: Path,
    autoencoder_checkpoint: Path | None,
) -> list[list[str]]:
    """Reuse the standard evaluator without replacing saved model/data settings."""
    settings = OmegaConf.to_container(OmegaConf.load(PRESET).eval, resolve=True)
    if not isinstance(settings, dict):
        msg = "Expected an evaluation settings mapping"
        raise TypeError(msg)
    commands = []
    for mode, ratio in (("free_running", 0.0), ("teacher_forced", 1.0)):
        mode_settings = {
            **settings,
            "free_running_only": ratio == 0.0,
            "teacher_forcing_ratio": ratio,
        }
        command = [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "autocast",
            "eval",
            "--workdir",
            str(run_dir),
            "--output-subdir",
            str(output_root / mode),
            *_overrides("eval", mode_settings),
            f"++eval.checkpoint={json.dumps(str(checkpoint))}",
            "++seed=42",
        ]
        if autoencoder_checkpoint is not None:
            command.append(
                f"++autoencoder_checkpoint={json.dumps(str(autoencoder_checkpoint))}"
            )
        commands.append(command)
    return commands


def _validate_execution(args: argparse.Namespace) -> None:
    candidates = [
        args.run_dir / name
        for name in ("resolved_config.yaml", "resolved_autoencoder_config.yaml")
    ]
    config_path = next((path for path in candidates if path.is_file()), None)
    if config_path is None:
        msg = "A saved training config is required; generic defaults are not a baseline"
        raise FileNotFoundError(msg)
    cfg = OmegaConf.load(config_path)
    if OmegaConf.select(cfg, "model.processor._target_") is None:
        msg = "Select an afCRPS or FM training run, not an autoencoder run"
        raise ValueError(msg)
    if (cfg.datamodule.n_steps_input, cfg.datamodule.n_steps_output) != (1, 4):
        msg = "This evaluation protocol requires one input and four output frames"
        raise ValueError(msg)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    is_fm = str(cfg.model.processor._target_).endswith("FlowMatchingProcessor")
    if is_fm and (
        args.autoencoder_checkpoint is None or not args.autoencoder_checkpoint.is_file()
    ):
        msg = "Latent FM evaluation requires the matching SWE64 autoencoder checkpoint"
        raise FileNotFoundError(msg)
    for mode in ("free_running", "teacher_forced"):
        destination = args.output_root / mode
        if destination.exists():
            msg = f"Refusing to reuse evaluation destination: {destination}"
            raise FileExistsError(msg)


def main() -> None:
    """Print both commands by default; execution requires the explicit flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--autoencoder-checkpoint", type=Path)
    parser.add_argument(
        "--execute", action="store_true", help="Run both evaluations sequentially."
    )
    args = parser.parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.autoencoder_checkpoint is not None:
        args.autoencoder_checkpoint = args.autoencoder_checkpoint.expanduser().resolve()
    commands = build_commands(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
    )
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.execute:
        _validate_execution(args)
        for command in commands:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
