#!/bin/bash

set -euo pipefail
# Evaluate the 2026-04-20 FM/diff cached-latent processor basis with explicit
# eval.mode=ambient across all 4 datasets. Eval reuses resolved_config.yaml so
# flow_ode_steps (=50), hid_channels, and backbone match training, and the
# autoencoder checkpoint enables ambient-space rollout through decode->encode.
#
# Batch size: diffusion rollout is ODE-integrated (flow_ode_steps=50) per
# rollout step, so ambient 64x64 × n_members=10 × 50 ODE substeps is the
# tightest of the three. 4/GPU fits; drop to 2 if OOM.
#
# We also pin eval.n_members explicitly here so the comparison scripts do not
# depend on the global eval default staying at 10.

EVAL_BATCH_SIZE=4
EVAL_N_MEMBERS=10
TIMEOUT_MIN=360
EVAL_SUBDIR="eval_ambient"
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

RUN_DIRS=(
    "outputs/2026-04-20/diff_gs64_flow_matching_vit_09490da_7e9e331"
    "outputs/2026-04-20/diff_gpe64_flow_matching_vit_09490da_47bf39a"
    "outputs/2026-04-20/diff_cns64_flow_matching_vit_09490da_636fcc3"
    "outputs/2026-04-20/diff_ad64_flow_matching_vit_09490da_dae1382"
)
declare -A AE_CKPT=(
    ["outputs/2026-04-20/diff_gs64_flow_matching_vit_09490da_7e9e331"]="$HOME/autocast/outputs/2026-04-17/ae_gs64_3a7999b_ed36b8e/autoencoder.ckpt"
    ["outputs/2026-04-20/diff_gpe64_flow_matching_vit_09490da_47bf39a"]="$HOME/autocast/outputs/2026-04-17/ae_gpe64_3a7999b_31e1c9f/autoencoder.ckpt"
    ["outputs/2026-04-20/diff_cns64_flow_matching_vit_09490da_636fcc3"]="$HOME/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8/autoencoder.ckpt"
    ["outputs/2026-04-20/diff_ad64_flow_matching_vit_09490da_dae1382"]="$HOME/autocast/outputs/2026-04-17/ae_ad64_3a7999b_1a1e300/autoencoder.ckpt"
)

for run_dir in "${RUN_DIRS[@]}"; do
    ae_ckpt="${AE_CKPT[$run_dir]:-}"
    if [[ -z "${ae_ckpt}" ]]; then
        echo "Skipping ${run_dir}: no autoencoder_checkpoint mapping" >&2
        continue
    fi

    run_dir_abs="$(realpath "${run_dir}")"
    if [[ ! -f "${run_dir_abs}/resolved_config.yaml" ]]; then
        echo "Skipping ${run_dir}: resolved_config.yaml missing" >&2
        continue
    fi
    eval_ckpt="${run_dir_abs}/processor.ckpt"
    if [[ ! -f "${eval_ckpt}" ]]; then
        echo "Skipping ${run_dir}: processor.ckpt missing" >&2
        continue
    fi
    if [[ ! -f "${ae_ckpt}" ]]; then
        echo "Skipping ${run_dir}: AE checkpoint missing at ${ae_ckpt}" >&2
        continue
    fi
    ae_ckpt_abs="$(realpath "${ae_ckpt}")"
    eval_output_dir="${run_dir_abs}/${EVAL_SUBDIR}"

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting FM-ambient eval"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir_abs}"
        echo "  eval.checkpoint: ${eval_ckpt}"
        echo "  autoencoder_checkpoint: ${ae_ckpt_abs}"
        echo "  eval.mode: ambient"
        echo "  output_subdir: ${EVAL_SUBDIR}"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir_abs}" \
            --output-subdir "${EVAL_SUBDIR}" \
            eval.checkpoint="${eval_ckpt}" \
            eval.mode=ambient \
            +autoencoder_checkpoint="${ae_ckpt_abs}" \
            eval.csv_path="${eval_output_dir}/evaluation_metrics.csv" \
            eval.video_dir="${eval_output_dir}/videos" \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
