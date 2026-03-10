#!/bin/bash
#SBATCH --job-name=plot_mask
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos=turing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00

module purge
module load baskerville
module load bask-apps/live

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast

python << 'PYTHON_EOF'
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load mask
mask_path = '/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all/land_mask.pt'
mask = torch.load(mask_path)

print(f"Mask loaded: shape={mask.shape}, dtype={mask.dtype}")

# Create figure with colorbar
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

# Plot binary mask
im = ax.imshow(mask.numpy(), cmap='binary', origin='upper')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Mask Value (0=Land, 1=Ocean)', rotation=270, labelpad=20)

ax.set_title('Sea Ice Land Mask (NH, 432x432)', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude Index')
ax.set_ylabel('Latitude Index')

# Save figure
output_path = '/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast/land_mask_visualization.png'
plt.savefig(output_path, dpi=100, bbox_inches='tight')
print(f"\nMask visualization saved to: {output_path}")

# Print statistics
print(f"\nMask Statistics:")
print(f"  Land pixels (0): {torch.sum(mask == 0).item():,} ({100*torch.sum(mask==0).item()/mask.numel():.1f}%)")
print(f"  Ocean pixels (1): {torch.sum(mask == 1).item():,} ({100*torch.sum(mask==1).item()/mask.numel():.1f}%)")

PYTHON_EOF

