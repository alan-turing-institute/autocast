#!/bin/bash
#SBATCH --account=brics.u6eo
#SBATCH --qos=normal
#SBATCH --job-name=download_era5
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=216GB
#SBATCH --output=autocast/logs/download_era5_%j.log
#SBATCH --error=autocast/logs/download_era5_%j.err

set -e
cd /home/u6eo/qiencai.u6eo
mkdir -p autocast/logs

export MAMBA_ROOT_PREFIX=/home/u6eo/qiencai.u6eo/.local/share/mamba
eval "$(/home/u6eo/qiencai.u6eo/.local/bin/micromamba shell hook -s bash)"
micromamba activate /home/u6eo/qiencai.u6eo/.micromamba/envs/icenet

echo "Job started: $(date)"
echo "Running on: $(hostname)"

icenet_data_era5 -w 3 -v --vars uas,vas,tas,zg --levels ',,,500|250' north 2000-01-01 2020-12-31

echo "Job finished: $(date)"
