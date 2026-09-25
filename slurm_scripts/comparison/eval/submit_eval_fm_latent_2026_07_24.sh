#!/bin/bash

set -euo pipefail

# Submit the four 2026-07-24 cached-latent flow-matching evaluations after
# their corresponding training jobs leave the queue.
#
# eval.mode is deliberately pinned to encode_once. Do not change this to auto:
# older tags did not always infer the cached-latent evaluation route reliably.
#
# afterany is intentional. If training fails, the evaluation job will still be
# released, but evaluation will fail clearly if processor.ckpt is unavailable.
#
# Preview all four submissions without queueing anything:
#   ./slurm_scripts/comparison/eval/submit_eval_fm_latent_2026_07_24.sh
#
# Queue them:
#   SUBMIT=true ./slurm_scripts/comparison/eval/submit_eval_fm_latent_2026_07_24.sh

EVAL_BATCH_SIZE=4
EVAL_N_MEMBERS=10
# Comparable successful evaluations took 47-55 minutes. Allow more than 2x
# that runtime for filesystem variability and the additional snapshot output.
TIMEOUT_MIN=120
MEMORY="115G"
EVAL_SUBDIR="eval"
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

RUN_ROOT="${RUN_ROOT:-outputs/2026-07-24}"
SUBMIT="${SUBMIT:-false}"

case "${SUBMIT}" in
    true|false) ;;
    *)
        echo "SUBMIT must be true or false, got: ${SUBMIT}" >&2
        exit 2
        ;;
esac

# run_id|training_job_id|run_directory|autoencoder_checkpoint
RUNS=(
    "fm_ad|5773771|diff_ad64_flow_matching_vit_103985e_74d6bd6|${HOME}/autocast/outputs/2026-04-17/ae_ad64_3a7999b_1a1e300/autoencoder.ckpt"
    "fm_gs|5773772|diff_gs64_flow_matching_vit_103985e_fa8e0f5|${HOME}/autocast/outputs/2026-04-17/ae_gs64_3a7999b_ed36b8e/autoencoder.ckpt"
    "fm_gpe|5773773|diff_gpe64_flow_matching_vit_103985e_e498f5e|${HOME}/autocast/outputs/2026-04-17/ae_gpe64_3a7999b_31e1c9f/autoencoder.ckpt"
    "fm_cns|5773774|diff_cns64_flow_matching_vit_103985e_b0eb639|${HOME}/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8/autoencoder.ckpt"
)

had_error=false

for run_spec in "${RUNS[@]}"; do
    IFS='|' read -r run_id training_job_id run_name ae_ckpt <<< "${run_spec}"
    run_dir="${RUN_ROOT%/}/${run_name}"

    if [[ ! -f "${run_dir}/resolved_config.yaml" ]]; then
        echo "Missing resolved config for ${run_id}: ${run_dir}" >&2
        had_error=true
        continue
    fi
    if [[ ! -f "${ae_ckpt}" ]]; then
        echo "Missing autoencoder checkpoint for ${run_id}: ${ae_ckpt}" >&2
        had_error=true
        continue
    fi

    run_dir_abs="$(realpath "${run_dir}")"
    ae_ckpt_abs="$(realpath "${ae_ckpt}")"
    dry_run_arg=(--dry-run)
    run_label="preview"
    if [[ "${SUBMIT}" == "true" ]]; then
        dry_run_arg=()
        run_label="submit"
    fi

    echo "${run_label}: ${run_id}"
    echo "  training dependency: afterany:${training_job_id}"
    echo "  run_dir: ${run_dir_abs}"
    echo "  eval.checkpoint: processor.ckpt"
    echo "  autoencoder_checkpoint: ${ae_ckpt_abs}"
    echo "  eval.mode: encode_once"
    echo "  output_subdir: ${EVAL_SUBDIR}"
    echo "  time: ${TIMEOUT_MIN} minutes"
    echo "  memory: ${MEMORY}"

    uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
        --workdir "${run_dir_abs}" \
        --output-subdir "${EVAL_SUBDIR}" \
        eval.checkpoint=processor.ckpt \
        eval.mode=encode_once \
        +autoencoder_checkpoint="${ae_ckpt_abs}" \
        eval.metrics="${EVAL_METRICS}" \
        eval.batch_size="${EVAL_BATCH_SIZE}" \
        eval.n_members="${EVAL_N_MEMBERS}" \
        hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
        +hydra.launcher.additional_parameters.nodes=1 \
        +hydra.launcher.additional_parameters.mem="${MEMORY}" \
        +hydra.launcher.additional_parameters.dependency="afterany:${training_job_id}"
done

if [[ "${had_error}" == "true" ]]; then
    exit 1
fi
