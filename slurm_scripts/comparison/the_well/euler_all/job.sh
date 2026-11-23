#!/bin/bash -l
set -euo pipefail
umask 027

# Submit with --chdir at the reviewed checkout. Resource requests are supplied
# at submission, since training and paired evaluation use different layouts.
cd "$(git rev-parse --show-toplevel)"
operation=${1:?Specify check, train, eval or eval-rollout}
profile=${2:?Specify probe or production}
objective=${3:?Specify fm or crps}
run_dir=${4:?Provide an absolute run directory}
[[ "$run_dir" == /* ]]

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export AUTOCAST_CONFIG_PATH="$PWD/src/autocast/configs"
export AUTOCAST_SOURCE_REVISION="$(git rev-parse HEAD)"
export AUTOCAST_DATASETS=${AUTOCAST_DATASETS:-"$PWD/datasets"}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
codec=n6g8kix8_euler_all_dcae_f32c64_large
open_dataset=euler_multi_quadrants_openBC
periodic_dataset=euler_multi_quadrants_periodicBC
export EULER_OPEN_CACHE_DIR=${EULER_OPEN_CACHE_DIR:-"$AUTOCAST_DATASETS/$open_dataset/$codec/cache/$open_dataset"}
export EULER_PERIODIC_CACHE_DIR=${EULER_PERIODIC_CACHE_DIR:-"$AUTOCAST_DATASETS/$periodic_dataset/$codec/cache/$periodic_dataset"}
workflow=(uv run --offline --no-sync python -m slurm_scripts.comparison.the_well.euler_open.workflow)

# Raw frames between model frames. The processor emits four model frames per block.
data_stride=${EULER_DATA_STRIDE:-4}
case "$data_stride" in
    1|4) ;;
    *) echo "EULER_DATA_STRIDE must be 1 or 4" >&2; exit 2 ;;
esac
case "$objective" in
    fm) batch_per_rank=8 ;;
    crps) batch_per_rank=1 ;;
    *) echo "Unknown objective: $objective" >&2; exit 2 ;;
esac
eval_limit=()
case "$profile" in
    probe)
        schedule=(trainer.max_epochs=3 trainer.max_time=00:00:13:00
            trainer.check_val_every_n_epoch=3 trainer.callbacks.0.every_n_epochs=3)
        eval_limit=(eval.max_test_batches=2)
        ;;
    production)
        schedule=(trainer.max_epochs=4096 trainer.max_time=00:23:30:00
            trainer.check_val_every_n_epoch=8 trainer.callbacks.0.every_n_epochs=128)
        ;;
    *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac

trainer=(python -m autocast.scripts.train.processor
    "local_experiment=the_well/euler_all/${objective}_spatiotemporal_latent"
    "hydra.run.dir=$run_dir" "datamodule.batch_size=$batch_per_rank"
    "datamodule.stride=$data_stride"
    trainer.num_nodes=8 trainer.devices=4 trainer.precision=32-true
    trainer.callbacks.0.save_top_k=2 trainer.callbacks.0.save_last=false
    ++trainer.callbacks.0.monitor=step ++trainer.callbacks.0.mode=max
    ++trainer.callbacks.0.save_on_train_epoch_end=true
    ++trainer.callbacks.0.save_weights_only=false
    logging.wandb.mode=offline output.skip_test=true '++umask="0027"'
    "${schedule[@]}")

if [[ "$operation" == check ]]; then
    exec uv run --offline --no-sync "${trainer[@]}" --cfg job --resolve
fi
test "$AUTOCAST_SOURCE_REVISION" = "${AUTOCAST_EXPECTED_REVISION:?Set the reviewed commit SHA}"
test -z "$(git status --porcelain)"

case "$operation" in
    train)
        "${workflow[@]}" verify-joint-cache
        module load brics/nccl/2.26.6-1
        export NCCL_NET="AWS Libfabric" WANDB_DIR="$run_dir"
        exec srun --kill-on-bad-exit=1 uv run --offline --no-sync "${trainer[@]}"
        ;;
    eval|eval-rollout)
        test -s "$run_dir/processor.ckpt"
        selection_args=()
        rollout_limits=()
        if [[ "$operation" == eval-rollout ]]; then
            preset=the_well/euler_all
            selection=lola-rollout
            manifest_prefix=lola-rollout
            subset_key=rollout_subset
            output_prefix=eval-lola
            if [[ "$data_stride" == 1 ]]; then
                rollout_limits=(eval.max_rollout_steps=25 datamodule.max_rollout_steps=100)
                output_prefix=eval-stride1
            fi
            eval_limit=()
            if [[ "$profile" == probe ]]; then
                eval_limit=(eval.max_rollout_batches=2)
                output_prefix="$output_prefix-probe"
            fi
        else
            preset=the_well/euler_open
            selection=windows
            manifest_prefix=paired
            subset_key=test_window_subset
            output_prefix=eval
            selection_args=(--data-stride "$data_stride")
        fi
        for source in open periodic; do
            cache="$EULER_OPEN_CACHE_DIR"
            if [[ "$source" == periodic ]]; then cache="$EULER_PERIODIC_CACHE_DIR"; fi
            manifest="$run_dir/$manifest_prefix-$source.yaml"
            "${workflow[@]}" --source "$source" "$selection" "$manifest" "${selection_args[@]}"
            srun --kill-on-bad-exit=1 uv run --offline --no-sync autocast eval \
                --workdir "$run_dir" --output-subdir "$output_prefix-$source" \
                "eval=$preset" eval.mode=encode_once \
                "eval.checkpoint=$run_dir/processor.ckpt" \
                datamodule.data_paths=null "datamodule.data_path=$cache" \
                eval.devices=1 eval.num_nodes=1 eval.strategy=auto \
                eval.save_rollout_snapshots=false "eval.$subset_key=$manifest" \
                "${rollout_limits[@]}" "${eval_limit[@]}"
        done
        ;;
    *) echo "Unknown operation: $operation" >&2; exit 2 ;;
esac
