"""Collect a completed RB campaign and rebuild local comparisons; submit no jobs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=20",
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=none",
    "-o",
    "ServerAliveInterval=10",
    "-o",
    "ServerAliveCountMax=2",
]


def collect(host: str, remote: Path, local: Path, timeout_hours: float) -> None:
    """Wait on filesystem completion markers, download, then verify and plot."""
    local.mkdir(parents=True, exist_ok=True)
    status = local / "collection_status.json"

    def record(phase: str, **fields) -> None:
        status.write_text(
            json.dumps({"phase": phase, "remote": str(remote), **fields}, indent=2)
            + "\n"
        )
        print(phase, fields, flush=True)

    def rsync(source: str) -> None:
        subprocess.run(
            [
                "rsync",
                "-az",
                "-e",
                shlex.join(["ssh", *SSH_OPTIONS]),
                source,
                str(local) + "/",
            ],
            check=True,
        )

    record("fetching_manifest")
    try:
        rsync(f"{host}:{remote}/manifest.json")
        manifest = local / "manifest.json"
        models = json.loads(manifest.read_text())["models"]
        checks = " && ".join(
            shlex.join(["test", "-f", str(remote / name / "production/COMPLETE")])
            for name in models
        )
        deadline = time.monotonic() + timeout_hours * 3600
        while True:
            result = subprocess.run(
                ["ssh", *SSH_OPTIONS, host, checks],
                check=False,
                text=True,
                capture_output=True,
            )
            if result.returncode == 0:
                break
            if result.returncode != 1:
                raise subprocess.CalledProcessError(
                    result.returncode, result.args, result.stdout, result.stderr
                )
            if time.monotonic() >= deadline:
                msg = "Timed out waiting for the four production completion markers"
                raise TimeoutError(msg)
            record("waiting_for_production")
            # Only file markers are read; no scheduler polling or submissions.
            time.sleep(60)
        record("downloading_complete_outputs")
        rsync(f"{host}:{remote}/")
        record("verifying_and_plotting")
        subprocess.run(
            [
                "uv",
                "run",
                "--frozen",
                "--no-sync",
                "python",
                "-m",
                "slurm_scripts.comparison.the_well.rayleigh_benard.plot_diagnostics",
                "--manifest",
                str(manifest),
                "--output",
                str(local / "comparison"),
            ],
            check=True,
        )
        record("complete", comparison=str(local / "comparison"))
    except Exception as error:
        record("failed", error=repr(error))
        raise


def main() -> None:
    """Expose a portable, evaluation-only collection command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--timeout-hours", type=float, default=6)
    args = parser.parse_args()
    collect(args.host, args.remote_root, args.local_root.resolve(), args.timeout_hours)


if __name__ == "__main__":
    main()
