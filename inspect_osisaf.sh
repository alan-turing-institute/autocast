#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 0:10:00
#SBATCH --nodes 1
#SBATCH --gpus 0
#SBATCH --tasks-per-node 1
#SBATCH --job-name inspect_osisaf

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

python << 'EOF'
import torch

# Check what's in one of the data files
train_file = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all/train/data.pt"
print(f"Loading: {train_file}")
data = torch.load(train_file, map_location='cpu')

print(f"\nType of loaded data: {type(data)}")
print(f"Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
print(f"Data dtype: {data.dtype if hasattr(data, 'dtype') else 'N/A'}")

if isinstance(data, dict):
    print(f"\nIt's a dict with keys: {data.keys()}")
    for k, v in data.items():
        print(f"  {k}: {type(v)} shape={v.shape if hasattr(v, 'shape') else 'N/A'}")
else:
    print(f"\n❌ It's a raw {type(data).__name__}, not a dict!")
    print("This is why the dataset code fails.")
EOF
