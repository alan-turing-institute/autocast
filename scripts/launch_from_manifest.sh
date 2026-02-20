#!/bin/bash

set -euo pipefail

# Launch many prewritten runs from a manifest file.
#
# Manifest format (one run per line):
#   <subcommand and args exactly as passed to `autocast`>
# Example:
#   epd --mode slurm --dataset reaction_diffusion --date rd --run-name 00 trainer.max_epochs=5
#
# Usage:
#   bash scripts/launch_from_manifest.sh run_manifests/example_runs.txt
#   bash scripts/launch_from_manifest.sh run_manifests/example_runs.txt --dry-run

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <manifest-file> [--dry-run]"
  exit 1
fi

MANIFEST="$1"
DRY_RUN="${2:-}"

if [ ! -f "${MANIFEST}" ]; then
  echo "Manifest file not found: ${MANIFEST}"
  exit 1
fi

while IFS= read -r line || [ -n "$line" ]; do
  # Skip comments and blank lines
  if [[ -z "${line}" ]] || [[ "${line}" =~ ^[[:space:]]*# ]]; then
    continue
  fi

  cmd="uv run autocast ${line}"
  if [ "${DRY_RUN}" = "--dry-run" ] && [[ "${line}" != *" --dry-run"* ]] && [[ "${line}" != *"--dry-run" ]]; then
    cmd="${cmd} --dry-run"
  fi

  echo "Running: ${cmd}"
  eval "${cmd}"
done < "${MANIFEST}"
