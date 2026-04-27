#!/bin/bash

set -euo pipefail
# Eval submitter for the 2026-04-26 planned_02 ablation runs, including the
# m=4 follow-up (planned_02_m4_followup) under the same RUNS table.
#
# Mirrors submit_eval_planned_01.sh but spans GS and AD datamodules, so the
# crps_latent profile resolves the AE checkpoint via AE_CKPTS_BY_DATAMODULE.
#
#   crps_ambient   EPD CRPS run, eval.mode=ambient, best-multiwinkler ckpt
#                  resolver (symlink-first, then best-multiwinkler-from0p25-*).
#                  Output: <run>/eval_best_multiwinkler_from0p25/.
#
#   crps_latent    processor cached-latent CRPS run, eval.mode=auto -> encode_once
#                  with the matching AE checkpoint, best-multiwinkler ckpt
#                  resolver (same as crps_ambient).
#                  Output: <run>/eval_best_multiwinkler_from0p25/.
#
# The two-step multi-Winkler resolver:
#   1. <run>/autocast/<id>/checkpoints/best-multiwinkler.ckpt — a hand-picked
#      symlink for runs where best-multiwinkler-pre0p25-*.ckpt beats the
#      from-0.25 selection.
#   2. otherwise, the latest best-multiwinkler-from0p25-*.ckpt as the default.
#
# GPE (latent_crps_m8_gpe) is intentionally omitted: the m=8 run collapsed and
# will be rerun with m=16 only — eval will move to that follow-up script.

EVAL_BATCH_SIZE_CRPS=8
EVAL_N_MEMBERS=10
TIMEOUT_MIN_CRPS=240
EVAL_SUBDIR_MW="eval_best_multiwinkler_from0p25"
ROLLOUT_SNAPSHOT_TIMESTEPS="[0,4,12,30,99]"
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

declare -A AE_CKPTS_BY_DATAMODULE=(
    ["gray_scott"]="$HOME/autocast/outputs/2026-04-17/ae_gs64_3a7999b_ed36b8e/autoencoder.ckpt"
    ["advection_diffusion"]="$HOME/autocast/outputs/2026-04-17/ae_ad64_3a7999b_1a1e300/autoencoder.ckpt"
)

# run_id|profile|datamodule|run_dir
# datamodule slot is only consumed by crps_latent (AE lookup); crps_ambient
# rows still carry it for symmetry.
RUNS=(
    "latent_crps_m8_gs|crps_latent|gray_scott|outputs/2026-04-26/planned_02/crps_gs64_vit_azula_large_3b47441_1c8e446"
    "latent_crps_m8_ad|crps_latent|advection_diffusion|outputs/2026-04-26/planned_02/crps_ad64_vit_azula_large_3b47441_3ad3562"
    "vit_m4_gs|crps_ambient|gray_scott|outputs/2026-04-26/planned_02/crps_gs64_vit_azula_large_3b47441_4944cce"
    "vit_m4_ad|crps_ambient|advection_diffusion|outputs/2026-04-26/planned_02_m4_followup/crps_ad64_vit_azula_large_189c141_bbb0bc8"
    "vit_m4_gpe|crps_ambient|gpe_laser_only_wake|outputs/2026-04-26/planned_02_m4_followup/crps_gpe64_vit_azula_large_189c141_ce8db86"
)

resolve_multiwinkler_checkpoint() {
    local run_dir="$1"
    local -a ckpts=()

    mapfile -t ckpts < <(
        find "${run_dir}" -path '*/checkpoints/best-multiwinkler.ckpt' | sort
    )

    if (( ${#ckpts[@]} >= 1 )); then
        printf '%s\n' "${ckpts[$(( ${#ckpts[@]} - 1 ))]}"
        return 0
    fi

    mapfile -t ckpts < <(
        find "${run_dir}" -type f -path '*/checkpoints/best-multiwinkler-from0p25-*.ckpt' | sort
    )

    if (( ${#ckpts[@]} >= 1 )); then
        printf '%s\n' "${ckpts[$(( ${#ckpts[@]} - 1 ))]}"
        return 0
    fi

    return 1
}

submit_crps_ambient() {
    local run_id="$1" run_dir_abs="$2"
    local eval_ckpt
    if ! eval_ckpt="$(resolve_multiwinkler_checkpoint "${run_dir_abs}")"; then
        echo "Skipping ${run_id}: no best-multiwinkler.ckpt symlink and no best-multiwinkler-from0p25-*.ckpt" >&2
        return 0
    fi
    local eval_ckpt_abs eval_output_dir
    eval_ckpt_abs="$(realpath "${eval_ckpt}")"
    eval_output_dir="${run_dir_abs}/${EVAL_SUBDIR_MW}"

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        local dry_run_arg=() run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting CRPS-ambient eval (best multi-Winkler)"
        echo "  mode: ${run_label}"
        echo "  run_id: ${run_id}"
        echo "  run_dir: ${run_dir_abs}"
        echo "  eval.checkpoint: ${eval_ckpt_abs}"
        echo "  eval.mode: ambient"
        echo "  output_subdir: ${EVAL_SUBDIR_MW}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir_abs}" \
            --output-subdir "${EVAL_SUBDIR_MW}" \
            eval.checkpoint="${eval_ckpt_abs}" \
            eval.mode=ambient \
            eval.csv_path="${eval_output_dir}/evaluation_metrics.csv" \
            eval.video_dir="${eval_output_dir}/videos" \
            eval.save_rollout_snapshots=true \
            eval.rollout_snapshot_dir="${eval_output_dir}/videos/snapshots" \
            eval.rollout_snapshot_timesteps="${ROLLOUT_SNAPSHOT_TIMESTEPS}" \
            eval.rollout_snapshot_format=png \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE_CRPS}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN_CRPS}"
    done
}

submit_crps_latent() {
    local run_id="$1" run_dir_abs="$2" ae_ckpt="$3"
    if [[ ! -f "${ae_ckpt}" ]]; then
        echo "Skipping ${run_id}: AE checkpoint missing at ${ae_ckpt}" >&2
        return 0
    fi
    local ae_ckpt_abs
    ae_ckpt_abs="$(realpath "${ae_ckpt}")"

    local eval_ckpt
    if ! eval_ckpt="$(resolve_multiwinkler_checkpoint "${run_dir_abs}")"; then
        echo "Skipping ${run_id}: no best-multiwinkler.ckpt symlink and no best-multiwinkler-from0p25-*.ckpt" >&2
        return 0
    fi
    local eval_ckpt_abs eval_output_dir
    eval_ckpt_abs="$(realpath "${eval_ckpt}")"
    eval_output_dir="${run_dir_abs}/${EVAL_SUBDIR_MW}"

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        local dry_run_arg=() run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting CRPS cached-latent eval (best multi-Winkler)"
        echo "  mode: ${run_label}"
        echo "  run_id: ${run_id}"
        echo "  run_dir: ${run_dir_abs}"
        echo "  eval.checkpoint: ${eval_ckpt_abs}"
        echo "  autoencoder_checkpoint: ${ae_ckpt_abs}"
        echo "  eval.mode: auto"
        echo "  output_subdir: ${EVAL_SUBDIR_MW}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir_abs}" \
            --output-subdir "${EVAL_SUBDIR_MW}" \
            eval.checkpoint="${eval_ckpt_abs}" \
            +autoencoder_checkpoint="${ae_ckpt_abs}" \
            eval.csv_path="${eval_output_dir}/evaluation_metrics.csv" \
            eval.video_dir="${eval_output_dir}/videos" \
            eval.save_rollout_snapshots=true \
            eval.rollout_snapshot_dir="${eval_output_dir}/videos/snapshots" \
            eval.rollout_snapshot_timesteps="${ROLLOUT_SNAPSHOT_TIMESTEPS}" \
            eval.rollout_snapshot_format=png \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE_CRPS}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN_CRPS}"
    done
}

for run_spec in "${RUNS[@]}"; do
    IFS='|' read -r run_id profile datamodule run_dir <<< "${run_spec}"

    if [[ ! -d "${run_dir}" ]]; then
        echo "Skipping ${run_id}: run_dir missing at ${run_dir}" >&2
        continue
    fi
    run_dir_abs="$(realpath "${run_dir}")"
    if [[ ! -f "${run_dir_abs}/resolved_config.yaml" ]]; then
        echo "Skipping ${run_id}: resolved_config.yaml missing under ${run_dir}" >&2
        continue
    fi

    case "${profile}" in
        crps_ambient)
            submit_crps_ambient "${run_id}" "${run_dir_abs}"
            ;;
        crps_latent)
            ae_ckpt="${AE_CKPTS_BY_DATAMODULE[$datamodule]:-}"
            if [[ -z "${ae_ckpt}" ]]; then
                echo "Skipping ${run_id}: no AE checkpoint configured for datamodule ${datamodule}" >&2
                continue
            fi
            submit_crps_latent "${run_id}" "${run_dir_abs}" "${ae_ckpt}"
            ;;
        *)
            echo "FATAL: unknown profile '${profile}' for ${run_id}" >&2
            exit 1
            ;;
    esac
done
