#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 0:30:00
#SBATCH --nodes 1
#SBATCH --gpus 0
#SBATCH --tasks-per-node 1
#SBATCH --job-name restructure_osisaf

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

python << 'EOF'
import torch
from pathlib import Path

base_path = Path("/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all")

print("Loading train data...")
train_data = torch.load(base_path / "train" / "data.pt")
print(f"Train shape: {train_data.shape}")

print("Loading valid data...")
valid_data = torch.load(base_path / "valid" / "data.pt")
print(f"Valid shape: {valid_data.shape}")

print("Loading test data...")
test_data = torch.load(base_path / "test" / "data.pt")
print(f"Test shape: {test_data.shape}")

# Create the expected dict structure
data_dict = {
    "data": {
        "train": train_data,
        "valid": valid_data,
        "test": test_data,
    }
}

# Save as single file
output_path = base_path / "osisaf_nh_sic_all.pt"
print(f"\nSaving restructured data to: {output_path}")
torch.save(data_dict, output_path)
print("✓ Done!")
EOF
