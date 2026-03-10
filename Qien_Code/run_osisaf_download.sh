#!/bin/bash

#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 2:00:00           # 2 hours (adjust if needed)
#SBATCH --nodes 1
#SBATCH --gpus 0                 # No GPU needed for this
#SBATCH --tasks-per-node 8       # 8 CPUs for parallel ops
#SBATCH --job-name process_osisaf

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0


# Activate your virtual environment
source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast/Qien_Code

python get_osisaf_data.py