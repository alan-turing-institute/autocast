#!/bin/bash
#SBATCH --account=brics.u6eo
#SBATCH --qos=normal
#SBATCH --job-name=vit_era5_5min
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=216GB
#SBATCH --output=/home/u6eo/qiencai.u6eo/autocast/logs/vit_era5_%j.log
#SBATCH --error=/home/u6eo/qiencai.u6eo/autocast/logs/vit_era5_%j.err

set -e
cd /home/u6eo/qiencai.u6eo
mkdir -p autocast/logs

source /home/u6eo/qiencai.u6eo/autocast/.venv/bin/activate

echo "Job started: $(date)"
echo "Running on: $(hostname)"

cd /home/u6eo/qiencai.u6eo/autocast
autocast train-eval experiment=seaice_vit_oneshot_14day_era5

echo "Job finished: $(date)"
