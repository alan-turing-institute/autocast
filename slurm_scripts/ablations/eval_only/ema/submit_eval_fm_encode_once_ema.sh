#!/bin/bash

set -euo pipefail

# Eval-only EMA sweep for the main 2026-04-20 FM cached-latent runs.
#
# Each run is evaluated once with the EMA shadow weights stored in
# processor.ckpt under ema_state_dict. Eval uses encode_once so metrics stay in
# raw data space after decoding while avoiding per-step decode->encode.
#
# Output goes to <run>/eval_encode_once_ema/.

EVAL_BATCH_SIZE=4
EVAL_N_MEMBERS=10
TIMEOUT_MIN=360
EVAL_SUBDIR="eval_encode_once_ema"
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

declare -A AE_CKPT=(
    ["conditioned_navier_stokes"]="$HOME/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8/autoencoder.ckpt"
    ["gray_scott"]="$HOME/autocast/outputs/2026-04-17/ae_gs64_3a7999b_ed36b8e/autoencoder.ckpt"
    ["gpe_laser_only_wake"]="$HOME/autocast/outputs/2026-04-17/ae_gpe64_3a7999b_31e1c9f/autoencoder.ckpt"
    ["advection_diffusion"]="$HOME/autocast/outputs/2026-04-17/ae_ad64_3a7999b_1a1e300/autoencoder.ckpt"
)

# run_id|datamodule|run_dir
RUNS=(
    "fm_gs|gray_scott|outputs/2026-04-20/diff_gs64_flow_matching_vit_09490da_7e9e331"
    "fm_gpe|gpe_laser_only_wake|outputs/2026-04-20/diff_gpe64_flow_matching_vit_09490da_47bf39a"
    "fm_cns|conditioned_navier_stokes|outputs/2026-04-20/diff_cns64_flow_matching_vit_09490da_636fcc3"
    "fm_ad|advection_diffusion|outputs/2026-04-20/diff_ad64_flow_matching_vit_09490da_dae1382"
)

submit_eval() {
    local run_id="$1" run_dir_abs="$2" ae_ckpt_abs="$3"
    local eval_ckpt="${run_dir_abs}/processor.ckpt"
    local eval_ckpt_abs eval_output_dir

    if [[ ! -f "${eval_ckpt}" ]]; then
        echo "Skipping ${run_id}: processor.ckpt missing" >&2
        return 0
    fi
    eval_ckpt_abs="$(realpath "${eval_ckpt}")"
    eval_output_dir="${run_dir_abs}/${EVAL_SUBDIR}"

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        local dry_run_arg=() run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting FM encode_once EMA eval"
        echo "  mode: ${run_label}"
        echo "  run_id: ${run_id}"
        echo "  run_dir: ${run_dir_abs}"
        echo "  eval.checkpoint: ${eval_ckpt_abs}"
        echo "  autoencoder_checkpoint: ${ae_ckpt_abs}"
        echo "  eval.mode: encode_once"
        echo "  eval.use_ema: true"
        echo "  output_subdir: ${EVAL_SUBDIR}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir_abs}" \
            --output-subdir "${EVAL_SUBDIR}" \
            eval.checkpoint="${eval_ckpt_abs}" \
            eval.mode=encode_once \
            +eval.use_ema=true \
            +autoencoder_checkpoint="${ae_ckpt_abs}" \
            eval.csv_path="${eval_output_dir}/evaluation_metrics.csv" \
            eval.video_dir="${eval_output_dir}/videos" \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
}

for run_spec in "${RUNS[@]}"; do
    IFS="|" read -r run_id datamodule run_dir <<< "${run_spec}"

    if [[ ! -d "${run_dir}" ]]; then
        echo "Skipping ${run_id}: run_dir missing at ${run_dir}" >&2
        continue
    fi
    run_dir_abs="$(realpath "${run_dir}")"
    if [[ ! -f "${run_dir_abs}/resolved_config.yaml" ]]; then
        echo "Skipping ${run_id}: resolved_config.yaml missing under ${run_dir}" >&2
        continue
    fi

    ae_ckpt="${AE_CKPT[$datamodule]:-}"
    if [[ -z "${ae_ckpt}" ]]; then
        echo "Skipping ${run_id}: no AE checkpoint configured for ${datamodule}" >&2
        continue
    fi
    if [[ ! -f "${ae_ckpt}" ]]; then
        echo "Skipping ${run_id}: AE checkpoint missing at ${ae_ckpt}" >&2
        continue
    fi
    ae_ckpt_abs="$(realpath "${ae_ckpt}")"

    submit_eval "${run_id}" "${run_dir_abs}" "${ae_ckpt_abs}"
done
