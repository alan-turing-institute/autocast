#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=64G
#SBATCH --job-name=download_osisaf_opendap
#SBATCH --output=logs/download_osisaf_opendap_%j.out
#SBATCH --error=logs/download_osisaf_opendap_%j.err

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.11.3-GCCcore-12.3.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast

python download_osisaf_iceconc_only.py