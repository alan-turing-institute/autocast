#!/bin/bash

set -euo pipefail

if (( $# != 4 )); then
    echo "Usage: $0 MANIFEST STATE DATASET STAGE" >&2
    exit 2
fi

launch_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
manifest_path="$1"
state_path="$2"
if [[ "${manifest_path}" != /* ]]; then
    manifest_path="${launch_dir}/${manifest_path}"
fi
if [[ "${state_path}" != /* ]]; then
    state_path="${launch_dir}/${state_path}"
fi
manifest_path="$(realpath "${manifest_path}")"
state_path="$(realpath "${state_path}")"
script_dir="$(dirname "${manifest_path}")"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

cd "${repo_root}"
exec uv run --project "${repo_root}" --frozen --no-sync python \
    "${script_dir}/pipeline.py" \
    --manifest "${manifest_path}" run-stage --state "${state_path}" \
    --dataset "$3" --stage "$4"
