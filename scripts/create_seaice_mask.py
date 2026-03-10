"""Create a land mask from raw sea ice netCDF data.

For sea ice data, land regions should be masked out (0), and ocean/valid regions kept (1).
This script reads from the raw OSISAF netCDF file to generate a static land mask.
"""

import torch
import numpy as np
import xarray as xr
from pathlib import Path


def create_seaice_mask_from_nc(
    nc_path: str = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw_osisaf/osisaf_nh_2018.nc",
    method: str = "data_coverage",
    coverage_threshold: float = 0.5,
    output_path: str = None,
) -> np.ndarray:
    """Create a binary land mask from raw netCDF sea ice data.
    
    Args:
        nc_path: Path to raw netCDF file
        method: How to identify land regions:
            - "data_coverage": Land = regions with <coverage_threshold fraction of valid data
            - "nan_based": Land = regions where all values are NaN
            - "attribute": Use land mask variable if present in netCDF
        coverage_threshold: For data_coverage method, fraction of valid data required to be ocean
        output_path: Where to save mask. If None, saves to same dir as nc_path with name "land_mask.pt"
    
    Returns:
        mask: Binary numpy array (H, W) where 1=ocean/valid, 0=land/invalid
    """
    nc_path = Path(nc_path)
    output_path = Path(output_path) if output_path else nc_path.parent / "land_mask.pt"
    
    print(f"Loading netCDF file: {nc_path}")
    ds = xr.open_dataset(nc_path)
    print(f"Dataset variables: {list(ds.data_vars)}")
    print(f"Dataset coords: {list(ds.coords)}")
    
    # Try to find the sea ice concentration variable
    sic_var = None
    for var_name in ['sic', 'sea_ice_concentration', 'ice_conc', 'concentration']:
        if var_name in ds.data_vars:
            sic_var = var_name
            break
    
    if sic_var is None:
        # Use first data variable if no standard name found
        sic_var = list(ds.data_vars)[0]
    
    print(f"Using variable: {sic_var}")
    sic_data = ds[sic_var].values
    print(f"SIC data shape: {sic_data.shape}, dtype: {sic_data.dtype}")
    
    # Handle different data shapes
    # Common: (time, lat, lon) or (lat, lon)
    if sic_data.ndim == 3:
        print("Processing time series data (time, lat, lon)...")
        # Average or take coverage across time dimension
        if method == "data_coverage":
            # Count how many valid (non-NaN) timesteps each pixel has
            valid_per_pixel = np.sum(~np.isnan(sic_data), axis=0)
            total_timesteps = sic_data.shape[0]
            coverage = valid_per_pixel / total_timesteps
            mask = (coverage >= coverage_threshold).astype(np.float32)
            print(f"Coverage threshold: {coverage_threshold}")
            print(f"Valid timesteps per pixel: min={valid_per_pixel.min()}, max={valid_per_pixel.max()}, mean={valid_per_pixel.mean():.1f}")
        elif method == "nan_based":
            # Land = all timesteps are NaN
            mask = (~np.all(np.isnan(sic_data), axis=0)).astype(np.float32)
        else:
            raise ValueError(f"Unknown method for 3D data: {method}")
            
    elif sic_data.ndim == 2:
        print("Processing static data (lat, lon)...")
        if method == "data_coverage":
            # For static data, look for NaN as missing/land
            mask = (~np.isnan(sic_data)).astype(np.float32)
        elif method == "nan_based":
            mask = (~np.isnan(sic_data)).astype(np.float32)
        else:
            raise ValueError(f"Unknown method for 2D data: {method}")
    else:
        raise ValueError(f"Unexpected data shape: {sic_data.shape}")
    
    # Convert to torch and ensure proper format
    mask_tensor = torch.from_numpy(mask).float()
    
    print(f"\nMask shape: {mask_tensor.shape}")
    print(f"Land pixels (0): {(mask_tensor == 0).sum().item()}")
    print(f"Ocean pixels (1): {(mask_tensor == 1).sum().item()}")
    print(f"Land coverage: {(mask_tensor == 0).float().mean().item() * 100:.1f}%")
    print(f"Ocean coverage: {(mask_tensor == 1).float().mean().item() * 100:.1f}%")
    
    # Save mask
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mask_tensor, output_path)
    print(f"\nSaved mask to: {output_path}")
    
    ds.close()
    return mask_tensor.numpy()


if __name__ == "__main__":
    # Create mask from raw netCDF data
    print("=" * 70)
    print("Creating sea ice land mask from raw netCDF data...")
    print("=" * 70)
    
    nc_file = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw_osisaf/osisaf_nh_2018.nc"
    output_file = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all/land_mask.pt"
    
    print("\nMethod: Data coverage based (50% valid data threshold)")
    mask = create_seaice_mask_from_nc(
        nc_path=nc_file,
        method="data_coverage",
        coverage_threshold=0.5,
        output_path=output_file,
    )
    
    print("\n" + "=" * 70)
    print(f"Land mask created and saved to: {output_file}")
    print("=" * 70)
