"""Process full OSI-SAF dataset into train/valid/test splits.

Reads individual year netCDF files and creates processed dataset matching
the structure of the 2018-only dataset.
"""

import os
import numpy as np
import xarray as xr
import torch
from pathlib import Path


def process_year(nc_path: Path) -> np.ndarray:
    """Process one year of OSI-SAF data.
    
    Args:
        nc_path: Path to year's netCDF file
        
    Returns:
        Processed data array of shape (365, W, H, 1)
    """
    print(f"  Loading {nc_path.name}...")
    ds = xr.open_dataset(nc_path)
    
    # Get ice concentration variable
    sic = ds['ice_conc']
    
    # Drop Feb 29 to ensure 365 days
    time_index = sic['time'].to_index()
    mask = ~((time_index.month == 2) & (time_index.day == 29))
    sic = sic.isel(time=np.where(mask)[0])
    
    # Check we have 365 days
    if len(sic.time) != 365:
        print(f"    WARNING: Expected 365 days, got {len(sic.time)}")
        return None
    
    # Get values as numpy array: (time, y, x)
    data = sic.values
    
    # Convert NaN to 0
    data = np.nan_to_num(data, nan=0.0)
    
    # Add channel dimension: (time, y, x) -> (time, y, x, 1)
    data = data[:, :, :, None]
    
    print(f"    Shape: {data.shape}, dtype: {data.dtype}")
    
    ds.close()
    return data


def main():
    # Paths
    raw_dir = Path("/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw_osisaf")
    output_dir = Path("/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_osisaf_full")
    
    # Year splits
    train_years = list(range(1979, 2011))  # 1979-2010 (32 years)
    valid_years = list(range(2011, 2016))  # 2011-2015 (5 years)
    test_years = list(range(2016, 2021))   # 2016-2020 (5 years)
    
    print("=" * 80)
    print("PROCESSING FULL OSI-SAF DATASET")
    print("=" * 80)
    print(f"\nTrain years: {train_years[0]}-{train_years[-1]} ({len(train_years)} years)")
    print(f"Valid years: {valid_years[0]}-{valid_years[-1]} ({len(valid_years)} years)")
    print(f"Test years:  {test_years[0]}-{test_years[-1]} ({len(test_years)} years)")
    print(f"\nInput directory:  {raw_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directories
    for split in ['train', 'valid', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = {
        'train': train_years,
        'valid': valid_years,
        'test': test_years
    }
    
    for split_name, years in splits.items():
        print(f"\n{'=' * 80}")
        print(f"PROCESSING {split_name.upper()} SPLIT ({len(years)} years)")
        print('=' * 80)
        
        year_data_list = []
        
        for year in years:
            nc_path = raw_dir / f"osisaf_nh_{year}.nc"
            
            if not nc_path.exists():
                print(f"  WARNING: File not found: {nc_path.name}, skipping")
                continue
            
            try:
                data = process_year(nc_path)
                if data is not None:
                    year_data_list.append(data)
            except Exception as e:
                print(f"  ERROR processing {year}: {e}")
                continue
        
        if not year_data_list:
            print(f"  ERROR: No data processed for {split_name} split!")
            continue
        
        # Stack all years: (n_years, 365, W, H, 1)
        print(f"\n  Stacking {len(year_data_list)} years...")
        stacked_data = np.stack(year_data_list, axis=0)
        print(f"  Final shape: {stacked_data.shape}")
        print(f"  Data range: [{stacked_data.min():.3f}, {stacked_data.max():.3f}]")
        
        # Convert to torch tensor
        tensor_data = torch.from_numpy(stacked_data).float()
        
        # Wrap in dictionary to match 2018 dataset structure
        data_dict = {"data": tensor_data}
        
        # Save
        output_path = output_dir / split_name / "data.pt"
        print(f"  Saving to: {output_path}")
        torch.save(data_dict, output_path)
        
        file_size = output_path.stat().st_size / (1024**3)  # GB
        print(f"  ✓ Saved: {file_size:.2f} GB")
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    
    # Summary
    for split_name in ['train', 'valid', 'test']:
        data_path = output_dir / split_name / "data.pt"
        if data_path.exists():
            size = data_path.stat().st_size / (1024**3)
            data_dict = torch.load(data_path)
            shape = data_dict['data'].shape
            print(f"\n{split_name.upper():>6}: {shape} - {size:.2f} GB")


if __name__ == "__main__":
    main()
