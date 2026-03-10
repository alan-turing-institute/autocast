#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=32G
#SBATCH --job-name compare_processed
#SBATCH --output=logs/compare_processed_%j.out
#SBATCH --error=logs/compare_processed_%j.err

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.11.3-GCCcore-12.3.0

# Activate conda environment
source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

python << 'EOF'
import torch
import os

print("=" * 100)
print("COMPARING TWO PROCESSED DATASETS")
print("=" * 100)

# Path 1: Full dataset (multi-year, train/valid/test split)
path1_base = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all"
# Path 2: 2018 only dataset
path2_base = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic"

print("\n" + "=" * 100)
print("DATASET 1: osisaf_nh_sic_all (FULL MULTI-YEAR DATA)")
print("=" * 100)
print(f"Location: {path1_base}")
print(f"Subdirectories: {os.listdir(path1_base)}")

for split in ["train", "valid", "test"]:
    split_path = os.path.join(path1_base, split, "data.pt")
    if os.path.exists(split_path):
        data = torch.load(split_path, map_location='cpu')
        file_size_mb = os.path.getsize(split_path) / (1024**2)
        
        print(f"\n  [{split}]")
        print(f"    File size: {file_size_mb:.2f} MB")
        print(f"    Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"    Keys: {list(data.keys())}")
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    print(f"      {key}: shape {val.shape}, dtype {val.dtype}")
                    print(f"        Value range: [{val.min():.6f}, {val.max():.6f}], mean: {val.mean():.6f}")
                elif isinstance(val, list):
                    print(f"      {key}: list of {len(val)} items")
        elif isinstance(data, torch.Tensor):
            print(f"    Shape: {data.shape}, dtype {data.dtype}")
            print(f"    Value range: [{data.min():.6f}, {data.max():.6f}], mean: {data.mean():.6f}")

print("\n\n" + "=" * 100)
print("DATASET 2: osisaf_nh_sic (2018 ONLY DATA)")
print("=" * 100)
print(f"Location: {path2_base}")
print(f"Subdirectories: {os.listdir(path2_base)}")

for split in ["train", "valid", "test"]:
    split_path = os.path.join(path2_base, split, "data.pt")
    if os.path.exists(split_path):
        data = torch.load(split_path, map_location='cpu')
        file_size_mb = os.path.getsize(split_path) / (1024**2)
        
        print(f"\n  [{split}]")
        print(f"    File size: {file_size_mb:.2f} MB")
        print(f"    Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"    Keys: {list(data.keys())}")
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    print(f"      {key}: shape {val.shape}, dtype {val.dtype}")
                    print(f"        Value range: [{val.min():.6f}, {val.max():.6f}], mean: {val.mean():.6f}")
                elif isinstance(val, list):
                    print(f"      {key}: list of {len(val)} items")
        elif isinstance(data, torch.Tensor):
            print(f"    Shape: {data.shape}, dtype {data.dtype}")
            print(f"    Value range: [{data.min():.6f}, {data.max():.6f}], mean: {data.mean():.6f}")

print("\n\n" + "=" * 100)
print("KEY DIFFERENCES SUMMARY")
print("=" * 100)
print("""
1. DATA SCOPE:
   - osisaf_nh_sic_all: Multiple years of data (likely 1979-2023 or similar full record)
   - osisaf_nh_sic: Only 2018 data

2. STRUCTURE:
   - Both should follow dict with 'data' key structure
   - Main difference: time dimension size (multi-year vs single year)

3. SPLIT DISTRIBUTION:
   - osisaf_nh_sic_all: Large train/valid/test datasets (e.g., 70/15/15 split)
   - osisaf_nh_sic: Smaller 2018 datasets (365 days split)

4. USE CASE:
   - osisaf_nh_sic_all: Full training dataset for ML models
   - osisaf_nh_sic: Limited dataset, possibly for testing/debugging or year-specific analysis
""")

EOF
