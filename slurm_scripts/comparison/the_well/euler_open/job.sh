#!/bin/bash -l
set -euo pipefail
umask 027

# Submit with --chdir at the prepared AutoCast checkout. Slurm may spool this
# script outside the checkout, so use the working directory to locate the code.
cd "$(git rev-parse --show-toplevel)"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export AUTOCAST_CONFIG_PATH="$PWD/src/autocast/configs"
export AUTOCAST_SOURCE_REVISION="$(git rev-parse HEAD)"
if [[ -n "$(git --no-optional-locks status --porcelain -- src/autocast local_hydra slurm_scripts/comparison/the_well)" ]]; then
    AUTOCAST_SOURCE_REVISION+="+dirty"
fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

operation=${1:?Specify cache, train, train-fm, train-ambient, eval or eval-ambient}
profile=${2:?Specify production or smoke}
shift 2
if [[ "$operation" == cache || "$operation" == train || "$operation" == train-fm ]]; then
    export EULER_CACHE_DIR=${1:?Specify the absolute cache path}
    shift
    [[ "$EULER_CACHE_DIR" == /* ]] || { echo "Use an absolute cache path" >&2; exit 2; }
fi

experiment=the_well/euler_open/crps_spatiotemporal_latent
if [[ "$operation" == train-fm ]]; then
    experiment=the_well/euler_open/fm_spatiotemporal_latent
fi
case "$profile" in
    production) ;;
    smoke) experiment+="_smoke" ;;
    *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac

workflow=(uv run --no-sync python -m slurm_scripts.comparison.the_well.euler_open.workflow --profile "$profile")
case "$operation" in
    cache)
        exec srun --kill-on-bad-exit=1 "${workflow[@]}" cache "${SLURM_ARRAY_TASK_ID:?Use a Slurm array}"
        ;;
    train|train-fm|train-ambient)
        run_dir=${1:?Specify an absolute training run directory}
        shift
        [[ "$run_dir" == /* ]] || { echo "Use an absolute training run directory" >&2; exit 2; }
        trainer=autocast.scripts.train.processor
        if [[ "$operation" == train-ambient ]]; then
            [[ "$profile" == production ]] || { echo "Use the production ambient profile" >&2; exit 2; }
            experiment=the_well/euler_open/crps_spatiotemporal_ambient_matched
            trainer=autocast.scripts.train.encoder_processor_decoder
            "${workflow[@]}" verify-ambient
        else
            "${workflow[@]}" verify
        fi
        if [[ "$profile" == production ]]; then
            module load brics/nccl/2.26.6-1
            export NCCL_NET="AWS Libfabric"
        fi
        exec srun --kill-on-bad-exit=1 uv run --no-sync python -m "$trainer" \
            "local_experiment=$experiment" "hydra.run.dir=$run_dir" "$@"
        ;;
    eval|eval-ambient)
        run_dir=${1:?Specify the completed training run directory}
        eval_dir=${2:?Specify the evaluation output directory}
        shift 2
        [[ "$run_dir" == /* && "$eval_dir" == /* ]] || { echo "Use absolute run and evaluation paths" >&2; exit 2; }
        checkpoint=processor.ckpt
        if [[ "$operation" == eval-ambient ]]; then
            checkpoint=encoder_processor_decoder.ckpt
        fi
        exec srun --kill-on-bad-exit=1 "${workflow[@]}" evaluate \
            "$run_dir/resolved_config.yaml" "$run_dir/$checkpoint" "$eval_dir" \
            "$eval_dir/paired_windows.yaml" "$@"
        ;;
    *) echo "Unknown operation: $operation" >&2; exit 2 ;;
esac
