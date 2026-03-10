#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=64G
#SBATCH --job-name inspect_filled
#SBATCH --output=logs/inspect_filled_%j.out
#SBATCH --error=logs/inspect_filled_%j.err

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.11.3-GCCcore-12.3.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

python << 'EOF'
import torch
import numpy as np

print("=" * 100)
print("INSPECTING FILLED MULTI-YEAR DATASET")
print("=" * 100)

base_path = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all"

for split in ["train", "valid", "test"]:
    path = f"{base_path}/{split}/data.pt"
    data_dict = torch.load(path)
    data = data_dict["data"]
    
    print(f"\n{split.upper()}:")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Min: {data.min():.6f}")
    print(f"  Max: {data.max():.6f}")
    print(f"  Mean: {data.mean():.6f}")
    print(f"  Median: {torch.median(data):.6f}")
    print(f"  Std: {data.std():.6f}")
    print(f"  NaN count: {torch.isnan(data).sum()}")
    
    # Check histogram
    non_zero = data[data != 0].numel()
    zero = (data == 0).sum().item()
    total = data.numel()
    
    print(f"  Zero values: {zero:,} ({100*zero/total:.1f}%)")
    print(f"  Non-zero values: {non_zero:,} ({100*non_zero/total:.1f}%)")
    
    if non_zero > 0:
        non_zero_data = data[data != 0]
        print(f"  Non-zero range: [{non_zero_data.min():.6f}, {non_zero_data.max():.6f}]")
        print(f"  Non-zero mean: {non_zero_data.mean():.6f}")

print("\n" + "=" * 100)
print("COMPARISON WITH 2018 DATASET")
print("=" * 100)

path_2018 = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic/train/data.pt"
data_2018 = torch.load(path_2018)["data"]

print(f"\n2018 TRAIN:")
print(f"  Shape: {data_2018.shape}")
print(f"  Min: {data_2018.min():.6f}")
print(f"  Max: {data_2018.max():.6f}")
print(f"  Mean: {data_2018.mean():.6f}")
print(f"  Non-zero: {(data_2018 != 0).sum().item():,} ({100*(data_2018 != 0).sum().item()/data_2018.numel():.1f}%)")

EOF
