#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=32G
#SBATCH --job-name inspect_datasets
#SBATCH --output=logs/inspect_datasets_%j.out
#SBATCH --error=logs/inspect_datasets_%j.err

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.11.3-GCCcore-12.3.0

# Activate conda environment
source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

python << 'EOF'
import xarray as xr
import torch
import numpy as np

print("=" * 80)
print("RAW OSI-SAF DATASET (netCDF)")
print("=" * 80)

# Load raw OSI-SAF 2018 netCDF file
raw_path = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw_osisaf/osisaf_nh_2018.nc"
raw_ds = xr.open_dataset(raw_path)

print(f"\nType: {type(raw_ds)}")
print(f"\nDimensions: {dict(raw_ds.dims)}")
print(f"\nCoordinates: {list(raw_ds.coords)}")
for coord in raw_ds.coords:
    print(f"  {coord}: {raw_ds.coords[coord].shape}")
print(f"\nData Variables: {list(raw_ds.data_vars)}")
print(f"\nShape of each variable:")
for var in raw_ds.data_vars:
    print(f"  {var}: {raw_ds[var].shape}")
print(f"\nFull xarray structure:")
print(raw_ds)

print("\n\n" + "=" * 80)
print("PROCESSED DATASET (PyTorch .pt)")
print("=" * 80)

# Load processed dataset
processed_path = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all/train/data.pt"
processed_data = torch.load(processed_path)

print(f"\nType: {type(processed_data)}")
if isinstance(processed_data, dict):
    print(f"Keys: {list(processed_data.keys())}")
    print(f"\nDetailed structure:")
    for key, val in processed_data.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}:")
            print(f"    Type: Tensor")
            print(f"    Shape: {val.shape}")
            print(f"    Dtype: {val.dtype}")
            print(f"    Min: {val.min():.4f}, Max: {val.max():.4f}, Mean: {val.mean():.4f}")
        elif isinstance(val, list):
            print(f"  {key}: List of {len(val)} items")
            if len(val) > 0:
                print(f"    First item type: {type(val[0])}")
                if isinstance(val[0], torch.Tensor):
                    print(f"    First item shape: {val[0].shape}")
        elif isinstance(val, np.ndarray):
            print(f"  {key}:")
            print(f"    Type: numpy array")
            print(f"    Shape: {val.shape}")
            print(f"    Dtype: {val.dtype}")
        else:
            print(f"  {key}: {type(val)}")
            if not callable(val):
                print(f"    Value: {val}")
elif isinstance(processed_data, torch.Tensor):
    print(f"Single Tensor:")
    print(f"  Shape: {processed_data.shape}")
    print(f"  Dtype: {processed_data.dtype}")
    print(f"  Min: {processed_data.min():.4f}, Max: {processed_data.max():.4f}")
elif isinstance(processed_data, list):
    print(f"List of {len(processed_data)} items")
    print(f"First item type: {type(processed_data[0])}")

print("\n" + "=" * 80)
print("COMPARISON & ANALYSIS")
print("=" * 80)
print("\nKey differences:")
print("1. RAW FORMAT: NetCDF (xarray.Dataset)")
print("   - Hierarchical structure with dimensions, coordinates, and variables")
print("   - Includes full metadata and attributes")
print("   - Human-readable format")
print("\n2. PROCESSED FORMAT: PyTorch (.pt)")
print("   - Dictionary structure for efficient ML training")
print("   - Optimized for GPU loading and batching")
print("   - Pre-normalized, pre-processed tensors")

EOF
