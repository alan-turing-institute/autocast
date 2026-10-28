#!/usr/bin/env bash
#SBATCH --job-name=training-timing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:15:00
#SBATCH --output=training-timing-%j.log
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg WANDB_MODE=disabled

export BENCHMARK_REPO="${BENCHMARK_REPO:-$SLURM_SUBMIT_DIR}"
export BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-$BENCHMARK_REPO/.venv/bin/python}"
: "${BENCHMARK_DATASET:?Set the path to a complete synthetic Zarr dataset}"
: "${BENCHMARK_OUTPUT_ROOT:?Set the directory for timing results}"
export BENCHMARK_DATASET BENCHMARK_OUTPUT_ROOT
export BENCHMARK_OUTPUT="$BENCHMARK_OUTPUT_ROOT/job-${SLURM_JOB_ID:?}"
mkdir -p "$BENCHMARK_OUTPUT_ROOT"
test -z "$(git -C "$BENCHMARK_REPO" status --porcelain)"
cd "$BENCHMARK_REPO"
if [[ -n "${BENCHMARK_NCCL_MODULE:-}" ]]; then
    module load "$BENCHMARK_NCCL_MODULE"
fi
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 30000))"
export NCCL_DEBUG=INFO

srun --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 bash -c '
    rank=${SLURM_PROCID:-0}
    nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?}" \
        --query-gpu=timestamp,uuid,utilization.gpu,power.draw,memory.used \
        --format=csv -l 1 > "$BENCHMARK_OUTPUT-telemetry-rank-$rank.csv" \
        2> "$BENCHMARK_OUTPUT-telemetry-rank-$rank.err" &
    telemetry_pid=$!
    trap '\''kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true'\'' EXIT
    uv run --no-project --python "$BENCHMARK_PYTHON" python \
        -m autocast.scripts.benchmark_training \
        "$BENCHMARK_DATASET" "$BENCHMARK_OUTPUT" "$@"
' _ "$@"
