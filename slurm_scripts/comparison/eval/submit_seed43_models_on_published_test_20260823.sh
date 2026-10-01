#!/bin/bash

set -euo pipefail

# Evaluate the eight seed-43-trained main-comparison models on the original
# published test sets. The four CRPS jobs use their seed-43 training
# normalization while reading the published raw data. The four flow-matching
# jobs use the original published-AE caches, which retain the published raw
# data and normalization in autoencoder_config.yaml.
#
# Preview and preflight all jobs without writing output directories:
#   bash slurm_scripts/comparison/eval/\
#     submit_seed43_models_on_published_test_20260823.sh
#
# Submit all eight independent one-GPU jobs after committing this script:
#   SUBMIT=true bash slurm_scripts/comparison/eval/\
#     submit_seed43_models_on_published_test_20260823.sh
#
# Eval artifacts are isolated in new hash/UUID subdirectories inside the
# corresponding seed-43 model directories. A dated campaign directory holds
# only Slurm logs. Both the submitter and worker refuse to reuse any path.

REPO_ROOT="/home/u6eo/ltcx7228.u6eo/autocast-02"
DATASETS_ROOT="/projects/u6eo/autocast/datasets"
RUN_GROUP="2026-08-23"
RUN_UUID="${RUN_UUID:-be0eaf9}"
MEMORY="115G"
EVAL_N_MEMBERS=10
VISUAL_BATCH_INDICES="[0,1,2,3,4,5,6,7]"
ROLLOUT_SNAPSHOT_TIMESTEPS="[0,4,12,30,99]"
TEST_WINDOWS="[null]"
ROLLOUT_WINDOWS="[[0,1],[0,4],[6,12],[13,30],[31,99]]"
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,spread,skill,ssr,winkler]"

# label|method|token|run directory|checkpoint|AE checkpoint or null|
# eval input|raw published test data|normalization or null|mode|batch size|time
RUNS=(
    "cns_crps|crps|cns64|${REPO_ROOT}/outputs/2026-08-01/main_comparison_cns_seed43/crps_vit_azula_large|${REPO_ROOT}/outputs/2026-08-01/main_comparison_cns_seed43/crps_vit_azula_large/autocast/spy78y66/checkpoints/best-multiwinkler-overall-0255-0.0066.ckpt|null|${DATASETS_ROOT}/conditioned_navier_stokes_2d_5e1f575|${DATASETS_ROOT}/conditioned_navier_stokes_2d_5e1f575|${DATASETS_ROOT}/conditioned_navier_stokes_2d_seed43_20260801/stats.yml|ambient|8|01:00:00"
    "cns_fm|fm|cns64|${REPO_ROOT}/outputs/2026-08-01/main_comparison_cns_seed43/fm_vit_large_published_ae|${REPO_ROOT}/outputs/2026-08-01/main_comparison_cns_seed43/fm_vit_large_published_ae/processor.ckpt|${REPO_ROOT}/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8/autoencoder.ckpt|${REPO_ROOT}/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8/cached_latents|${DATASETS_ROOT}/conditioned_navier_stokes_2d_5e1f575|null|encode_once|4|02:30:00"
    "ad_crps|crps|ad64|${REPO_ROOT}/outputs/2026-08-13/crps_ad64_vit_azula_large_90d59a9_20cc80f|${REPO_ROOT}/outputs/2026-08-13/crps_ad64_vit_azula_large_90d59a9_20cc80f/autocast/jdg0y0v6/checkpoints/best-multiwinkler-overall-0477-0.0001.ckpt|null|${DATASETS_ROOT}/advection_diffusion_2ba25b9|${DATASETS_ROOT}/advection_diffusion_2ba25b9|${DATASETS_ROOT}/advection_diffusion_seed43_20260813/stats.yml|ambient|8|01:00:00"
    "ad_fm|fm|ad64|${REPO_ROOT}/outputs/2026-08-13/diff_ad64_flow_matching_vit_90d59a9_6443f4c|${REPO_ROOT}/outputs/2026-08-13/diff_ad64_flow_matching_vit_90d59a9_6443f4c/processor.ckpt|${REPO_ROOT}/outputs/2026-04-17/ae_ad64_3a7999b_1a1e300/autoencoder.ckpt|${REPO_ROOT}/outputs/2026-04-17/ae_ad64_3a7999b_1a1e300/cached_latents|${DATASETS_ROOT}/advection_diffusion_2ba25b9|null|encode_once|4|02:30:00"
    "gpe_crps|crps|gpe64|${REPO_ROOT}/outputs/2026-08-13/crps_gpe64_vit_azula_large_90d59a9_11091e1|${REPO_ROOT}/outputs/2026-08-13/crps_gpe64_vit_azula_large_90d59a9_11091e1/autocast/bq9el053/checkpoints/best-multiwinkler-overall-0099-0.0042.ckpt|null|${DATASETS_ROOT}/gpe/laser_only_wake_5b51eac|${DATASETS_ROOT}/gpe/laser_only_wake_5b51eac|${DATASETS_ROOT}/gpe/laser_only_wake_seed43_20260813/stats.yml|ambient|8|01:00:00"
    "gpe_fm|fm|gpe64|${REPO_ROOT}/outputs/2026-08-13/diff_gpe64_flow_matching_vit_90d59a9_880dee2|${REPO_ROOT}/outputs/2026-08-13/diff_gpe64_flow_matching_vit_90d59a9_880dee2/processor.ckpt|${REPO_ROOT}/outputs/2026-04-17/ae_gpe64_3a7999b_31e1c9f/autoencoder.ckpt|${REPO_ROOT}/outputs/2026-04-17/ae_gpe64_3a7999b_31e1c9f/cached_latents|${DATASETS_ROOT}/gpe/laser_only_wake_5b51eac|null|encode_once|4|02:30:00"
    "gs_crps|crps|gs64|${REPO_ROOT}/outputs/2026-08-13/crps_gs64_vit_azula_large_90d59a9_329f457|${REPO_ROOT}/outputs/2026-08-13/crps_gs64_vit_azula_large_90d59a9_329f457/autocast/sfrvkyjp/checkpoints/best-multiwinkler-overall-0391-0.0007.ckpt|null|${DATASETS_ROOT}/gray_scott_68b0669|${DATASETS_ROOT}/gray_scott_68b0669|${DATASETS_ROOT}/gray_scott_seed43_20260813/stats.yml|ambient|8|01:00:00"
    "gs_fm|fm|gs64|${REPO_ROOT}/outputs/2026-08-13/diff_gs64_flow_matching_vit_90d59a9_fb95d30|${REPO_ROOT}/outputs/2026-08-13/diff_gs64_flow_matching_vit_90d59a9_fb95d30/processor.ckpt|${REPO_ROOT}/outputs/2026-04-17/ae_gs64_3a7999b_ed36b8e/autoencoder.ckpt|${REPO_ROOT}/outputs/2026-04-17/ae_gs64_3a7999b_ed36b8e/cached_latents|${DATASETS_ROOT}/gray_scott_68b0669|null|encode_once|4|02:30:00"
)

# Existing completed evaluations on the seed-43 test sets. These are checked
# explicitly during preflight and are never passed to an evaluation worker.
declare -A EXISTING_NEW_TEST_EVALS=(
    [cns_crps]="${REPO_ROOT}/outputs/2026-08-01/main_comparison_cns_seed43/crps_vit_azula_large/eval_best_multiwinkler_overall_trajectory_stats_20260804"
    [cns_fm]="${REPO_ROOT}/outputs/2026-08-01/main_comparison_cns_seed43/fm_vit_large_published_ae/eval_trajectory_stats_20260804"
    [ad_crps]="${REPO_ROOT}/outputs/2026-08-13/crps_ad64_vit_azula_large_90d59a9_20cc80f/eval_crps_ad64_90d59a9_4ad378c"
    [ad_fm]="${REPO_ROOT}/outputs/2026-08-13/diff_ad64_flow_matching_vit_90d59a9_6443f4c/eval_fm_ad64_90d59a9_cb993c2"
    [gpe_crps]="${REPO_ROOT}/outputs/2026-08-13/crps_gpe64_vit_azula_large_90d59a9_11091e1/eval_crps_gpe64_90d59a9_4b5baf5"
    [gpe_fm]="${REPO_ROOT}/outputs/2026-08-13/diff_gpe64_flow_matching_vit_90d59a9_880dee2/eval_fm_gpe64_90d59a9_9247009"
    [gs_crps]="${REPO_ROOT}/outputs/2026-08-13/crps_gs64_vit_azula_large_90d59a9_329f457/eval_crps_gs64_90d59a9_2ab77e6"
    [gs_fm]="${REPO_ROOT}/outputs/2026-08-13/diff_gs64_flow_matching_vit_90d59a9_fb95d30/eval_fm_gs64_90d59a9_aa62488"
)

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_metric_columns() {
    local csv_path="$1"
    local header metric

    header="$(head -n 1 "${csv_path}")"
    for metric in spread skill ssr; do
        case ",${header}," in
            *",${metric},"*) ;;
            *)
                echo "Missing ${metric} column in ${csv_path}" >&2
                return 1
                ;;
        esac
    done
}

run_eval() {
    if (( $# != 14 )); then
        echo "Usage: $0 --run-eval REPO COMMIT CAMPAIGN LABEL METHOD RUN_DIR CHECKPOINT AE_CHECKPOINT EVAL_INPUT RAW_TEST NORMALIZATION MODE BATCH_SIZE OUTPUT_DIR" >&2
        return 2
    fi

    local repo_root="$1"
    local source_commit="$2"
    local campaign_name="$3"
    local label="$4"
    local method="$5"
    local run_dir="$6"
    local checkpoint="$7"
    local autoencoder_checkpoint="$8"
    local eval_input="$9"
    local raw_test_data="${10}"
    local normalization_path="${11}"
    local eval_mode="${12}"
    local eval_batch_size="${13}"
    local output_dir="${14}"
    local trajectory_dir="${output_dir}/trajectory_statistics"
    local autoencoder_override cache_config cache_test_relative ae_run_name
    local -a normalization_override required_outputs metric_outputs
    local -a rollout_videos rollout_snapshots

    [[ "$(git -C "${repo_root}" rev-parse HEAD)" == "${source_commit}" ]] ||
        fail "Worker source checkout no longer matches ${source_commit}"
    git -C "${repo_root}" diff --quiet || fail "Worker source has tracked changes"
    git -C "${repo_root}" diff --cached --quiet ||
        fail "Worker source has staged changes"
    [[ -f "${run_dir}/resolved_config.yaml" ]] ||
        fail "Missing run config for ${label}: ${run_dir}/resolved_config.yaml"
    [[ -f "${checkpoint}" ]] ||
        fail "Missing checkpoint for ${label}: ${checkpoint}"
    [[ -d "${raw_test_data}" && -f "${raw_test_data}/stats.yml" ]] ||
        fail "Missing published raw test dataset for ${label}: ${raw_test_data}"
    [[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] ||
        fail "Refusing to reuse eval output path: ${output_dir}"

    normalization_override=()
    if [[ "${method}" == "crps" ]]; then
        [[ "${autoencoder_checkpoint}" == "null" ]] ||
            fail "CRPS eval unexpectedly received an AE checkpoint"
        [[ "${eval_input}" == "${raw_test_data}" ]] ||
            fail "CRPS eval input is not the published raw test dataset"
        [[ -f "${normalization_path}" ]] ||
            fail "Missing CRPS training normalization: ${normalization_path}"
        autoencoder_override="autoencoder_checkpoint=null"
        normalization_override=(
            "datamodule.normalization_path=${normalization_path}"
        )
    elif [[ "${method}" == "fm" ]]; then
        [[ "${normalization_path}" == "null" ]] ||
            fail "FM normalization must come from the published cache metadata"
        [[ -f "${autoencoder_checkpoint}" ]] ||
            fail "Missing published AE checkpoint: ${autoencoder_checkpoint}"
        cache_config="${eval_input}/autoencoder_config.yaml"
        [[ -d "${eval_input}" && -f "${cache_config}" ]] ||
            fail "Missing published latent cache metadata: ${eval_input}"
        cache_test_relative="${raw_test_data#${DATASETS_ROOT}/}"
        rg -Fq "${cache_test_relative}" "${cache_config}" ||
            fail "Published cache does not map to ${raw_test_data}"
        ae_run_name="$(basename "$(dirname "${autoencoder_checkpoint}")")"
        rg -Fq "${ae_run_name}/autoencoder.ckpt" "${cache_config}" ||
            fail "Published cache does not identify AE run ${ae_run_name}"
        autoencoder_override="+autoencoder_checkpoint=${autoencoder_checkpoint}"
    else
        fail "Unknown evaluation method: ${method}"
    fi

    mkdir "${output_dir}"

    echo "Starting ${label} on the published test set"
    echo "  source commit: ${source_commit}"
    echo "  model run: ${run_dir}"
    echo "  model checkpoint: ${checkpoint}"
    echo "  eval input: ${eval_input}"
    echo "  raw published test data: ${raw_test_data}"
    echo "  model normalization: ${normalization_path}"
    echo "  output: ${output_dir}"

    if ! srun --nodes=1 --ntasks=1 --gpus=1 \
        env \
            AUTOCAST_DATASETS="${DATASETS_ROOT}" \
            UV_CACHE_DIR="/tmp/autocast-cross-eval-${SLURM_JOB_ID}" \
        uv run --project "${repo_root}" --frozen --no-sync autocast eval \
            --mode local \
            --workdir "${run_dir}" \
            --output-subdir "${output_dir}" \
            eval.checkpoint="${checkpoint}" \
            "${autoencoder_override}" \
            datamodule.data_path="${eval_input}" \
            "${normalization_override[@]}" \
            eval.mode="${eval_mode}" \
            eval.accelerator=cuda \
            eval.devices=1 \
            eval.batch_size="${eval_batch_size}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            eval.max_test_batches=null \
            eval.max_rollout_batches=null \
            eval.max_rollout_steps=25 \
            eval.metric_windows="${TEST_WINDOWS}" \
            eval.metric_windows_rollout="${ROLLOUT_WINDOWS}" \
            eval.metrics="${EVAL_METRICS}" \
            eval.compute_rollout_coverage=true \
            eval.compute_rollout_metrics=true \
            eval.batch_indices="${VISUAL_BATCH_INDICES}" \
            eval.save_rollout_snapshots=true \
            eval.rollout_snapshot_timesteps="${ROLLOUT_SNAPSHOT_TIMESTEPS}" \
            eval.rollout_snapshot_format=png \
            eval.benchmark.enabled=true \
            eval.benchmark_rollout.enabled=true \
            eval.csv_path="${output_dir}/evaluation_metrics.csv" \
            eval.video_dir="${output_dir}/videos" \
            eval.trajectory_statistics.enabled=true \
            eval.trajectory_statistics.output_dir="${trajectory_dir}" \
            eval.trajectory_statistics.overwrite_existing=false \
            eval.trajectory_statistics.sampling_seed=42 \
            eval.trajectory_statistics.include_per_timestep=true \
            output.save_config=true \
            +provenance.evaluation_campaign="${campaign_name}" \
            +provenance.evaluation_data_role=published_test \
            +provenance.evaluation_data_path="${raw_test_data}" \
            +provenance.evaluation_source_commit="${source_commit}" \
            logging.wandb.enabled=false; then
        fail "Evaluation failed for ${label}; preserving ${output_dir}"
    fi

    required_outputs=(
        "${trajectory_dir}/resolved_eval_config.yaml"
        "${trajectory_dir}/evaluation_metrics.csv"
        "${trajectory_dir}/evaluation_metadata.csv"
        "${trajectory_dir}/benchmark_metrics.csv"
        "${trajectory_dir}/rollout_metrics.csv"
        "${trajectory_dir}/single_step_metrics_per_trajectory.csv"
        "${trajectory_dir}/rollout_metrics_per_trajectory.csv"
        "${trajectory_dir}/rollout_metrics_per_timestep_per_trajectory.csv"
        "${trajectory_dir}/rollout_metrics_per_timestep_channel_all.csv"
    )
    for required_output in "${required_outputs[@]}"; do
        [[ -s "${required_output}" ]] ||
            fail "Missing or empty required output for ${label}: ${required_output}"
    done

    metric_outputs=(
        "${trajectory_dir}/evaluation_metrics.csv"
        "${trajectory_dir}/rollout_metrics.csv"
        "${trajectory_dir}/single_step_metrics_per_trajectory.csv"
        "${trajectory_dir}/rollout_metrics_per_trajectory.csv"
        "${trajectory_dir}/rollout_metrics_per_timestep_per_trajectory.csv"
    )
    for metric_output in "${metric_outputs[@]}"; do
        require_metric_columns "${metric_output}" ||
            fail "Incomplete spread/skill output for ${label}"
    done

    rg -Fq "${raw_test_data}" \
        "${trajectory_dir}/single_step_metrics_per_trajectory.csv" ||
        fail "Trajectory output does not identify the published test set"
    rg -Fq "evaluation_data_role: published_test" \
        "${trajectory_dir}/resolved_eval_config.yaml" ||
        fail "Resolved eval config lacks cross-test provenance"
    rg -Fq "evaluation_source_commit: ${source_commit}" \
        "${trajectory_dir}/resolved_eval_config.yaml" ||
        fail "Resolved eval config lacks source-commit provenance"

    mapfile -t rollout_videos < <(
        find "${trajectory_dir}/videos" -maxdepth 1 -type f -name '*.mp4' -print
    )
    mapfile -t rollout_snapshots < <(
        find "${trajectory_dir}/videos/snapshots" \
            -maxdepth 1 -type f -name '*.png' -print
    )
    (( ${#rollout_videos[@]} >= 8 )) ||
        fail "Expected at least 8 rollout videos; found ${#rollout_videos[@]}"
    (( ${#rollout_snapshots[@]} >= 24 )) ||
        fail "Expected at least 24 rollout snapshots; found ${#rollout_snapshots[@]}"

    echo "Validated complete cross-test evaluation for ${label}"
}

if [[ "${1:-}" == "--run-eval" ]]; then
    shift
    run_eval "$@"
    exit
fi

SUBMIT="${SUBMIT:-false}"
case "${SUBMIT}" in
    true|false) ;;
    *) fail "SUBMIT must be true or false, got: ${SUBMIT}" ;;
esac
[[ "${RUN_UUID}" =~ ^[0-9a-f]{7}$ ]] ||
    fail "RUN_UUID must be exactly seven lowercase hexadecimal characters"

repo_root="$(git rev-parse --show-toplevel)"
[[ "${repo_root}" == "${REPO_ROOT}" ]] ||
    fail "Run from ${REPO_ROOT}; resolved ${repo_root}"
source_commit="$(git rev-parse HEAD)"
short_hash="${source_commit:0:7}"
campaign_name="campaign_seed43_models_published_test_${short_hash}_${RUN_UUID}"
campaign_root="${REPO_ROOT}/outputs/${RUN_GROUP}/${campaign_name}"
logs_dir="${campaign_root}/slurm_logs"
script_path="$(realpath "${BASH_SOURCE[0]}")"

if ! git diff --quiet || ! git diff --cached --quiet; then
    if [[ "${SUBMIT}" == "true" ]]; then
        fail "Commit tracked source changes before submission"
    fi
    echo "WARNING: preview uses current HEAD, but submission requires a clean commit."
fi
[[ ! -e "${campaign_root}" && ! -L "${campaign_root}" ]] ||
    fail "Refusing to reuse campaign output path: ${campaign_root}"

echo "Cross-test campaign: ${campaign_name}"
echo "Source commit: ${source_commit}"
echo "Submission enabled: ${SUBMIT}"
echo "Campaign logs: ${logs_dir}"
echo "Metrics: ${EVAL_METRICS}"

for run_spec in "${RUNS[@]}"; do
    IFS='|' read -r label method token run_dir checkpoint \
        autoencoder_checkpoint eval_input raw_test_data normalization_path \
        eval_mode eval_batch_size time_limit <<< "${run_spec}"
    existing_eval_dir="${EXISTING_NEW_TEST_EVALS[${label}]}"
    output_dir="${run_dir}/eval_published_test_${short_hash}_${RUN_UUID}"

    [[ -f "${run_dir}/resolved_config.yaml" ]] ||
        fail "Missing run config for ${label}: ${run_dir}/resolved_config.yaml"
    [[ -d "${existing_eval_dir}" && \
        -s "${existing_eval_dir}/trajectory_statistics/resolved_eval_config.yaml" ]] ||
        fail "Missing completed seed-43-test evaluation for ${label}: ${existing_eval_dir}"
    [[ "${output_dir}" != "${existing_eval_dir}" ]] ||
        fail "Published-test output aliases the existing seed-43-test evaluation"
    [[ -f "${checkpoint}" ]] ||
        fail "Missing checkpoint for ${label}: ${checkpoint}"
    [[ -d "${eval_input}" ]] || fail "Missing eval input for ${label}: ${eval_input}"
    [[ -d "${raw_test_data}" && -f "${raw_test_data}/stats.yml" ]] ||
        fail "Missing published raw test dataset for ${label}: ${raw_test_data}"
    if [[ "${method}" == "crps" ]]; then
        [[ -f "${normalization_path}" ]] ||
            fail "Missing CRPS training normalization: ${normalization_path}"
        shopt -s nullglob
        crps_overall_checkpoints=(
            "${run_dir}"/autocast/*/checkpoints/best-multiwinkler-overall-*.ckpt
        )
        shopt -u nullglob
        (( ${#crps_overall_checkpoints[@]} == 1 )) ||
            fail "Expected one overall-best multi-Winkler checkpoint for ${label}; found ${#crps_overall_checkpoints[@]}"
        [[ "$(realpath "${crps_overall_checkpoints[0]}")" == \
            "$(realpath "${checkpoint}")" ]] ||
            fail "Pinned checkpoint is not the unique overall-best checkpoint"
    else
        [[ -f "${autoencoder_checkpoint}" ]] ||
            fail "Missing published AE checkpoint: ${autoencoder_checkpoint}"
        [[ -f "${eval_input}/autoencoder_config.yaml" ]] ||
            fail "Missing published cache metadata: ${eval_input}"
        cache_test_relative="${raw_test_data#${DATASETS_ROOT}/}"
        rg -Fq "${cache_test_relative}" \
            "${eval_input}/autoencoder_config.yaml" ||
            fail "Published cache does not map to ${raw_test_data}"
        ae_run_name="$(basename "$(dirname "${autoencoder_checkpoint}")")"
        rg -Fq "${ae_run_name}/autoencoder.ckpt" \
            "${eval_input}/autoencoder_config.yaml" ||
            fail "Published cache does not identify AE run ${ae_run_name}"
    fi
    [[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] ||
        fail "Refusing to reuse eval output path: ${output_dir}"

    echo "plan: ${label}"
    echo "  trained model: seed 43"
    echo "  test data: published/original"
    echo "  run_dir: ${run_dir}"
    echo "  preserved seed-43-test eval: ${existing_eval_dir}"
    echo "  checkpoint: ${checkpoint}"
    echo "  eval input: ${eval_input}"
    echo "  raw test data: ${raw_test_data}"
    echo "  normalization: ${normalization_path}"
    echo "  eval.mode: ${eval_mode}"
    echo "  batch size / members: ${eval_batch_size} / ${EVAL_N_MEMBERS}"
    echo "  output: ${output_dir}"
    echo "  resources: 1 GPU, 16 CPUs, ${MEMORY}, ${time_limit}"
done

if [[ "${SUBMIT}" == "false" ]]; then
    echo "Preflight passed for all 8 jobs; no output directories were created."
    exit
fi

mkdir -p "${REPO_ROOT}/outputs/${RUN_GROUP}"
mkdir "${campaign_root}"
mkdir "${logs_dir}"

for run_spec in "${RUNS[@]}"; do
    IFS='|' read -r label method token run_dir checkpoint \
        autoencoder_checkpoint eval_input raw_test_data normalization_path \
        eval_mode eval_batch_size time_limit <<< "${run_spec}"
    output_dir="${run_dir}/eval_published_test_${short_hash}_${RUN_UUID}"

    job_id="$(
        sbatch --parsable \
            --job-name="xev43_${label}" \
            --nodes=1 \
            --ntasks-per-node=1 \
            --gpus-per-node=1 \
            --cpus-per-task=16 \
            --time="${time_limit}" \
            --mem="${MEMORY}" \
            --open-mode=append \
            --output="${logs_dir}/${label}-%j.out" \
            --error="${logs_dir}/${label}-%j.err" \
            --chdir="${REPO_ROOT}" \
            "${script_path}" --run-eval \
                "${REPO_ROOT}" \
                "${source_commit}" \
                "${campaign_name}" \
                "${label}" \
                "${method}" \
                "${run_dir}" \
                "${checkpoint}" \
                "${autoencoder_checkpoint}" \
                "${eval_input}" \
                "${raw_test_data}" \
                "${normalization_path}" \
                "${eval_mode}" \
                "${eval_batch_size}" \
                "${output_dir}"
    )"
    echo "submitted: ${label} -> ${job_id}"
done

echo "Submitted all eight cross-test evaluations."
