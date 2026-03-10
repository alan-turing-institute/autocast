#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --job-name inspect_raw
#SBATCH --output=logs/inspect_raw_%j.out
#SBATCH --error=logs/inspect_raw_%j.err

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.11.3-GCCcore-12.3.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

python << 'EOF'
import xarray as xr

print("=" * 100)
print("INSPECTING RAW MULTI-YEAR DATASET")
print("=" * 100)

raw_file = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw/osisaf_nh_sic_reprocessed.nc"

try:
    ds = xr.open_dataset(raw_file)
    print(f"\nFile: {raw_file}")
    print(f"\nFull structure:")
    print(ds)
    
    print(f"\n\nTime range:")
    if 'time' in ds.coords:
        times = ds.coords['time'].values
        print(f"  First time: {times[0]}")
        print(f"  Last time: {times[-1]}")
        print(f"  Total timesteps: {len(times)}")
    
    print(f"\n\nVariables and their stats:")
    for var in ds.data_vars:
        data = ds[var]
        print(f"\n  {var}:")
        print(f"    Shape: {data.shape}")
        print(f"    Min: {float(data.min()):.6f}")
        print(f"    Max: {float(data.max()):.6f}")
        print(f"    Mean: {float(data.mean()):.6f}")

except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

EOF
