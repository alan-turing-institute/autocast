#!/bin/bash
#SBATCH --account=brics.u6eo
#SBATCH --qos=normal
#SBATCH --job-name=build_osisaf_era5
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --output=logs/build_osisaf_era5_%j.log
#SBATCH --error=logs/build_osisaf_era5_%j.err

set -e
module purge
source /home/u6eo/qiencai.u6eo/autocast/.venv/bin/activate
cd /home/u6eo/qiencai.u6eo/autocast
mkdir -p logs

echo "Job started: $(date)"
python Qien_Code/build_osisaf_era5_dataset.py
echo "Job finished: $(date)"
