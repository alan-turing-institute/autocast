#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=osisaf_subset
#SBATCH --output=logs/subset_%j.out
#SBATCH --error=logs/subset_%j.err

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.11.3-GCCcore-12.3.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

python << 'EOF'
import torch
import os

data_path = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all/test/data.pt"
output_dir = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all/test"

print("Loading data...")
data_dict = torch.load(data_path)
data = data_dict['data']  # Extract tensor from dict

print(f"Full data shape: {data.shape}")
print(f"Data type: {data.dtype}")

# Create subset: first 100 samples
subset = data[:100]

subset_path = os.path.join(output_dir, "data_subset.pt")
torch.save(subset, subset_path)

print(f"Saved subset → {subset_path}")
print(f"Subset shape: {subset.shape}")
print(f"Subset size: {os.path.getsize(subset_path) / 1e6:.2f} MB")
EOF