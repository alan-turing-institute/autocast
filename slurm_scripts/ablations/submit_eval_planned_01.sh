#!/bin/bash

set -euo pipefail
# Eval submitter for the 2026-04-25 planned_01 CNS ablation runs.
#
# Mirrors the RUNS structure of submit_planned_01_large.sh. Each row is
# dispatched to one of three eval profiles:
#
#   crps_ambient      EPD CRPS run, eval.mode=ambient, best-multiwinkler ckpt
#                     resolver (symlink-first, then best-multiwinkler-from0p25-*).
#                     Output: <run>/eval_best_multiwinkler_from0p25/.
#
#   crps_latent       processor cached-latent CRPS run, eval.mode=auto -> encode_once
#                     with the matching AE checkpoint, best-multiwinkler ckpt
#                     resolver (same as crps_ambient).
#                     Output: <run>/eval_best_multiwinkler_from0p25/.
#
#   diffusion_ambient EPD diffusion (identity encoder/decoder), eval.mode=ambient,
#                     best-val ckpt resolver. Output: <run>/eval_ambient/.
#
# The multi-Winkler resolver tries, in order:
#   1. <run>/autocast/<id>/checkpoints/best-multiwinkler.ckpt — a hand-picked
#      symlink for runs where best-multiwinkler-pre0p25-*.ckpt beats the
#      from-0.25 selection.
#   2. best-multiwinkler-overall-*.ckpt — the no-window overall-best ckpt added
#      by the trainer default callbacks. Only present for runs trained after
#      that callback was added.
#   3. otherwise, the latest best-multiwinkler-from0p25-*.ckpt as the default.

EVAL_BATCH_SIZE_CRPS=8
EVAL_BATCH_SIZE_DIFF=4
EVAL_N_MEMBERS=10
TIMEOUT_MIN_CRPS=240
TIMEOUT_MIN_DIFF=360
EVAL_SUBDIR_MW="eval_best_multiwinkler_from0p25"
EVAL_SUBDIR_DIFF="eval_ambient"
ROLLOUT_SNAPSHOT_TIMESTEPS="[0,4,12,30,99]"
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

CNS_AE_CKPT="$HOME/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8/autoencoder.ckpt"

# run_id|profile|run_dir
RUNS=(
    "unet_m8_crps_cns|crps_ambient|outputs/2026-04-25/planned_cns/crps_cns64_unet_azula_large_9c98db0_65f8f71"
    "diffusion_cns|diffusion_ambient|outputs/2026-04-25/planned_cns/diff_cns64_diffusion_vit_9c98db0_e9bc460"
    "fair_crps_m8_cns|crps_ambient|outputs/2026-04-25/planned_cns/crps_cns64_vit_azula_large_9c98db0_d2a0496"
    "plain_crps_m8_cns|crps_ambient|outputs/2026-04-25/planned_cns/crps_cns64_vit_azula_large_9c98db0_6a91c49"
    "vit_noise256_m8_cns|crps_ambient|outputs/2026-04-25/planned_cns/crps_cns64_vit_azula_large_9c98db0_86e355d"
    "vit_m4_cns|crps_ambient|outputs/2026-04-25/planned_cns/crps_cns64_vit_azula_large_9c98db0_957ff88"
    "latent_crps_m8_cns|crps_latent|outputs/2026-04-25/planned_cns/crps_cns64_vit_azula_large_9c98db0_4b2a1a5"
    "vit_global_cond_m8_cns|crps_ambient|outputs/2026-04-25/planned_cns/crps_cns64_vit_azula_large_9c98db0_2fa67c5"
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
        find "${run_dir}" -type f -path '*/checkpoints/best-multiwinkler-overall-*.ckpt' | sort
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

submit_diffusion_ambient() {
    local run_id="$1" run_dir_abs="$2"
    # Match the existing FM/diffusion eval pattern (submit_eval_fm_ambient.sh):
    # use the saved final checkpoint symlink, eval.mode=ambient, batch_size=4,
    # timeout=360. The diffusion run has identity encoder/decoder, so it is
    # already ambient — no autoencoder_checkpoint.
    local eval_ckpt="${run_dir_abs}/encoder_processor_decoder.ckpt"
    if [[ ! -e "${eval_ckpt}" ]]; then
        echo "Skipping ${run_id}: encoder_processor_decoder.ckpt missing" >&2
        return 0
    fi
    local eval_ckpt_abs eval_output_dir
    eval_ckpt_abs="$(realpath "${eval_ckpt}")"
    eval_output_dir="${run_dir_abs}/${EVAL_SUBDIR_DIFF}"

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        local dry_run_arg=() run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting diffusion-ambient eval"
        echo "  mode: ${run_label}"
        echo "  run_id: ${run_id}"
        echo "  run_dir: ${run_dir_abs}"
        echo "  eval.checkpoint: ${eval_ckpt_abs}"
        echo "  eval.mode: ambient"
        echo "  output_subdir: ${EVAL_SUBDIR_DIFF}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir_abs}" \
            --output-subdir "${EVAL_SUBDIR_DIFF}" \
            eval.checkpoint="${eval_ckpt_abs}" \
            eval.mode=ambient \
            eval.csv_path="${eval_output_dir}/evaluation_metrics.csv" \
            eval.video_dir="${eval_output_dir}/videos" \
            eval.save_rollout_snapshots=true \
            eval.rollout_snapshot_dir="${eval_output_dir}/videos/snapshots" \
            eval.rollout_snapshot_timesteps="${ROLLOUT_SNAPSHOT_TIMESTEPS}" \
            eval.rollout_snapshot_format=png \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE_DIFF}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN_DIFF}"
    done
}

for run_spec in "${RUNS[@]}"; do
    IFS='|' read -r run_id profile run_dir <<< "${run_spec}"

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
            submit_crps_latent "${run_id}" "${run_dir_abs}" "${CNS_AE_CKPT}"
            ;;
        diffusion_ambient)
            submit_diffusion_ambient "${run_id}" "${run_dir_abs}"
            ;;
        *)
            echo "FATAL: unknown profile '${profile}' for ${run_id}" >&2
            exit 1
            ;;
    esac
done
