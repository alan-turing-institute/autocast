#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos=turing
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=process_osisaf_full
#SBATCH --output=logs/process_osisaf_full_%j.out
#SBATCH --error=logs/process_osisaf_full_%j.err

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.11.3-GCCcore-12.3.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast

python scripts/process_osisaf_full.py
