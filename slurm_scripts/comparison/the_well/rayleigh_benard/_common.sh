#!/bin/bash

# Shared helpers for curated Rayleigh-Benard comparison submitters.

RB_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RB_REPO_ROOT="$(cd "${RB_SCRIPT_DIR}/../../../.." && pwd)"
RB_DATASETS_ROOT="${AUTOCAST_DATASETS:-${RB_REPO_ROOT}/datasets}"
RB_CACHE_DIR="${RB_CACHE_DIR:-${RB_DATASETS_ROOT}/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large/cache/rayleigh_benard}"

RB_EVAL_METRICS_DEFAULT="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,vmse_v2,vrmse_v2,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,spread,skill,ssr,winkler]"

rb_has_hdf5_split() {
    local split_dir="$1"

    compgen -G "${split_dir}/*.h5" > /dev/null || \
        compgen -G "${split_dir}/*.hdf5" > /dev/null
}

rb_require_cache() {
    local split

    for split in train valid test; do
        if ! rb_has_hdf5_split "${RB_CACHE_DIR}/${split}"; then
            echo "Missing ${split}/*.h5 or ${split}/*.hdf5 under ${RB_CACHE_DIR}" >&2
            exit 1
        fi
    done
}

rb_run_dry_states() {
    if [[ "${DRY_RUN_ONLY:-false}" == "true" ]]; then
        printf '%s\n' "true"
    else
        printf '%s\n' "true" "false"
    fi
}

rb_submit() {
    local command="$1"
    local run_dry="$2"
    shift 2

    local dry_run_arg=()
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
    fi

    (
        cd "${RB_REPO_ROOT}"
        uv run autocast "${command}" --mode slurm "${dry_run_arg[@]}" "$@"
    )
}

rb_print_mode() {
    local run_dry="$1"

    if [[ "${run_dry}" == "true" ]]; then
        printf '%s\n' "slurm --dry-run"
    else
        printf '%s\n' "slurm"
    fi
}
