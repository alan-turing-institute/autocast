"""Preview or run shared metric evaluations from saved SWE64 training configs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
FORECAST_LEADS = 100


def build_commands(
    *,
    run_dir: Path,
    checkpoint: Path,
    output_root: Path,
    autoencoder_checkpoint: Path | None,
    n_steps_output: int = 4,
    test_data_root: Path | None = None,
    normalization_path: Path | None = None,
) -> list[list[str]]:
    """Reuse the standard evaluator without replacing saved model/data settings."""
    if n_steps_output not in (1, 4):
        msg = "The SWE64 protocol supports one or four output frames"
        raise ValueError(msg)
    commands = []
    for mode, ratio in (("free_running", 0.0), ("teacher_forced", 1.0)):
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
            "eval=swe64_paper",
            f"++eval.max_rollout_steps={FORECAST_LEADS // n_steps_output}",
            f"++eval.free_running_only={json.dumps(ratio == 0.0)}",
            f"++eval.teacher_forcing_ratio={ratio}",
            f"++datamodule.n_steps_output={n_steps_output}",
            f"++eval.checkpoint={json.dumps(str(checkpoint))}",
            "++seed=42",
        ]
        # The free-running pass already scores every next-window test example.
        if ratio == 1.0:
            command += [
                "++eval.compute_test_metrics=false",
                "++eval.trajectory_statistics.enabled=false",
            ]
        if test_data_root is not None:
            command += [
                f"++datamodule.data_path={json.dumps(str(test_data_root))}",
                "++datamodule.test_trajectories=100",
            ]
        if normalization_path is not None:
            command.append(
                f"++datamodule.normalization_path={json.dumps(str(normalization_path))}"
            )
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
    if cfg.datamodule.n_steps_input != 1 or _output_steps(cfg) not in (1, 4):
        msg = (
            "This evaluation protocol requires one input and one or four output frames"
        )
        raise ValueError(msg)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    codec = args.autoencoder_checkpoint or OmegaConf.select(
        cfg, "autoencoder_checkpoint"
    )
    identity_codec = all(
        str(OmegaConf.select(cfg, f"model.{part}._target_")).split(".")[-1]
        == f"Identity{part.title()}"
        for part in ("encoder", "decoder")
    )
    requires_autoencoder = codec is not None or (
        OmegaConf.select(cfg, "model.train_in_latent_space", default=False)
        and not identity_codec
    )
    if requires_autoencoder and (
        codec is None or not Path(codec).expanduser().is_file()
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
    parser.add_argument("--test-data-root", type=Path)
    parser.add_argument("--normalization-path", type=Path)
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
        n_steps_output=_saved_output_steps(args.run_dir),
        test_data_root=args.test_data_root,
        normalization_path=args.normalization_path,
    )
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.execute:
        _validate_execution(args)
        for command in commands:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


def _saved_output_steps(run_dir: Path) -> int:
    """Keep previews possible without data, but honor an available saved task."""
    for name in ("resolved_config.yaml", "resolved_autoencoder_config.yaml"):
        path = run_dir / name
        if path.is_file():
            return _output_steps(OmegaConf.load(path))
    return 4


def _output_steps(config: DictConfig | ListConfig) -> int:
    return int(
        OmegaConf.select(config, "model.prediction_steps")
        or OmegaConf.select(config, "datamodule.n_steps_output")
    )


if __name__ == "__main__":
    main()
