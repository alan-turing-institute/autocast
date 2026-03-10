#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=32G
#SBATCH --job-name investigate_structure
#SBATCH --output=logs/investigate_structure_%j.out
#SBATCH --error=logs/investigate_structure_%j.err

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
import sys

print("=" * 100)
print("INVESTIGATING DATASET LOADING DIFFERENCES")
print("=" * 100)

datasets = [
    ("osisaf_nh_sic_all", "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all/train/data.pt"),
    ("osisaf_nh_sic", "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic/train/data.pt"),
]

for name, path in datasets:
    print(f"\n{'=' * 100}")
    print(f"DATASET: {name}")
    print(f"{'=' * 100}")
    print(f"Path: {path}")
    
    try:
        # Load the data
        raw_data = torch.load(path, map_location='cpu')
        
        print(f"\n1. WHAT IS LOADED FROM torch.load():")
        print(f"   Type: {type(raw_data)}")
        print(f"   Type name: {type(raw_data).__name__}")
        
        if isinstance(raw_data, dict):
            print(f"   ✓ It's a Dictionary!")
            print(f"   Keys: {list(raw_data.keys())}")
            
            # Check if "data" key exists
            if "data" in raw_data:
                print(f"   ✓ 'data' key EXISTS")
                print(f"   Value type: {type(raw_data['data'])}")
                print(f"   Value shape: {raw_data['data'].shape}")
            else:
                print(f"   ✗ 'data' key MISSING!")
                print(f"   Available keys: {list(raw_data.keys())}")
                
        elif isinstance(raw_data, torch.Tensor):
            print(f"   ✗ It's a raw Tensor, NOT a dict!")
            print(f"   Shape: {raw_data.shape}")
            print(f"   Dtype: {raw_data.dtype}")
            print(f"   This will FAIL in SpatioTemporalDataset._from_f()")
            
        else:
            print(f"   ? Unexpected type: {type(raw_data)}")
            
        print(f"\n2. WHAT _from_f() WOULD DO:")
        print(f"   The code does: assert 'data' in f, 'HDF5 file must contain data dataset'")
        
        if isinstance(raw_data, dict) and "data" in raw_data:
            print(f"   ✓ Would PASS the assertion")
        else:
            print(f"   ✗ Would FAIL the assertion - KeyError or AssertionError")
            
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

print("\n\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print("""
ISSUE: One or both datasets may not match what SpatioTemporalDataset._from_f() expects.

The code expects:
  - When loading .pt file: torch.load() returns a dict with key "data"
  - dict["data"] contains the actual tensor

If one dataset is a raw tensor instead of a dict, you need to wrap it.
If one dataset has different keys, the code needs adjustment.
""")

EOF
