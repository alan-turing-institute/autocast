#!/bin/bash

set -euo pipefail

if (( $# != 4 )); then
    echo "Usage: $0 MANIFEST STATE DATASET STAGE" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

cd "${repo_root}"
exec uv run --project "${repo_root}" --frozen --no-sync python \
    "${script_dir}/pipeline.py" \
    --manifest "$1" run-stage --state "$2" --dataset "$3" --stage "$4"
