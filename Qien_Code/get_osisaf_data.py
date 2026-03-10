import os
import numpy as np
import xarray as xr
import pandas as pd
import torch

# -----------------------
# Config
# -----------------------
OPENDAP_URL = "https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_450a1_nh_agg"
OUT_DIR = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all"  # change me

# Split rule (year-based). Adjust as you like.
# By default: last 5 years = test, previous 5 year = valid, rest = train.
N_TEST_YEARS = 5
N_VALID_YEARS = 5

# Keep tensor size manageable
DTYPE = np.float32  # you can switch to np.float16 to halve disk usage
CHUNK_DAYS = 30     # Dask chunking along time

# -----------------------
# Helpers
# -----------------------
def infer_sic_var(ds: xr.Dataset) -> str:
    """Try common variable names; fallback to first data_var."""
    candidates = ["ice_conc", "conc", "sic", "ice_concentration"]
    for v in candidates:
        if v in ds.data_vars:
            return v
    # fallback: pick first non-empty variable
    for v in ds.data_vars:
        return v
    raise ValueError("No data variables found in dataset.")

def drop_feb29(da: xr.DataArray) -> xr.DataArray:
    """Drop Feb 29 to ensure 365 days per year."""
    t = da["time"].to_index()
    mask = ~((t.month == 2) & (t.day == 29))
    return da.isel(time=np.where(mask)[0])

# def normalize_units(sic: xr.DataArray) -> xr.DataArray:
#     """Convert to [0,1] if necessary."""
#     # Heuristic: if values look like percent (0..100), convert.
#     # Use robust quantile to avoid occasional fill values.
#     q99 = float(sic.quantile(0.99, skipna=True).compute())
#     if q99 > 1.5:
#         sic = sic / 100.0
#     return sic

def normalize_units(sic):
    # Try to infer scale from attributes/units
    units = (sic.attrs.get("units", "") or "").lower()
    # Many SIC products are in %, some are fraction.
    if "%" in units or "percent" in units:
        sic = sic / 100.0

    return sic

def clean_fill(sic: xr.DataArray) -> xr.DataArray:
    """Handle common fill conventions & clip to [0,1]."""
    # Convert to float for NaN support
    sic = sic.astype("float32")

    # If dataset provides a _FillValue, xarray usually decodes it already,
    # but just in case:
    fill = sic.attrs.get("_FillValue", None)
    if fill is not None:
        sic = sic.where(sic != fill)

    # Some OSI SAF products can include impossible values; keep plausible range.
    sic = sic.where((sic >= 0.0) & (sic <= 1.0))

    return sic

def ensure_order(sic: xr.DataArray) -> xr.DataArray:
    """Ensure (time, y, x) ordering (or time, yc, xc etc.)."""
    # Try common spatial dim names
    spatial_dims = [d for d in sic.dims if d != "time"]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected 2 spatial dims besides time, got dims={sic.dims}")
    return sic.transpose("time", spatial_dims[0], spatial_dims[1])

def year_list(da: xr.DataArray):
    years = np.unique(da["time"].dt.year.values)
    years = years[~np.isnan(years)].astype(int)
    years.sort()
    return years.tolist()

# -----------------------
# Main
# -----------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(OUT_DIR, split), exist_ok=True)

    print(f"Opening OPENDAP: {OPENDAP_URL}")
    ds = xr.open_dataset(OPENDAP_URL, chunks={"time": CHUNK_DAYS})

    sic_var = infer_sic_var(ds)
    print(f"Using SIC variable: {sic_var}")
    sic = ds[sic_var]

    # Basic cleaning pipeline (same spirit as before)
    sic = ensure_order(sic)
    sic = drop_feb29(sic)
    sic = normalize_units(sic)
    sic = clean_fill(sic)

    # Confirm spatial shape
    T, H, W = sic.sizes["time"], sic.sizes[sic.dims[1]], sic.sizes[sic.dims[2]]
    print(f"Full time length after dropping Feb29: T={T}")
    print(f"Spatial: H={H}, W={W}")

    years = year_list(sic)
    print(f"Years available: {years[0]} .. {years[-1]}  (n={len(years)})")

    # Split by year (last years for valid/test by default)
    test_years = years[-N_TEST_YEARS:]
    valid_years = years[-(N_TEST_YEARS + N_VALID_YEARS):-N_TEST_YEARS] if N_VALID_YEARS > 0 else []
    train_years = [y for y in years if (y not in valid_years and y not in test_years)]

    print(f"Split years:")
    print(f"  train: {train_years[0]}..{train_years[-1]} (n={len(train_years)})")
    if valid_years:
        print(f"  valid: {valid_years} (n={len(valid_years)})")
    print(f"  test : {test_years} (n={len(test_years)})")

    def build_and_save(years_subset, split_name):
        # Build yearly trajectories: (traj, time=365, H, W, C=1)
        traj_list = []
        for y in years_subset:
            # Select this year
            one = sic.sel(time=str(y))

            # Reindex to full daily sequence (after Feb29 drop) to detect missing days
            t0 = pd.Timestamp(f"{y}-01-01")
            t1 = pd.Timestamp(f"{y}-12-31")
            full = pd.date_range(t0, t1, freq="D")
            full = full[~((full.month == 2) & (full.day == 29))]  # drop Feb29
            one = one.reindex(time=full)

            # Must be 365 days
            if one.sizes["time"] != 365:
                print(f"[WARN] Year {y}: time={one.sizes['time']} != 365, skipping")
                continue

            # Compute year into memory (365*432*432 ~ 68M floats -> ~270MB float32)
            arr = one.data
            if hasattr(arr, "compute"):
                arr = arr.compute()
            arr = np.asarray(arr, dtype=DTYPE)

            # Add channel dim
            arr = arr[..., None]  # (365, H, W, 1)

            # This is wrong I think!  : replace NaNs with 0, and keep a mask if you want later
            # For "same as before", we usually did NaN->0 and rely on implicit land mask.
            # arr = np.nan_to_num(arr, nan=0.0)

            traj_list.append(arr)
            print(f"  built year {y}: {arr.shape} {arr.dtype}")

        if not traj_list:
            raise RuntimeError(f"No usable trajectories for split={split_name}")

        data = np.stack(traj_list, axis=0)  # (traj, 365, H, W, 1)
        print(f"[{split_name}] stacked: {data.shape} dtype={data.dtype}")

        # Save as torch tensor
        tensor = torch.from_numpy(data)
        out_path = os.path.join(OUT_DIR, split_name, "data.pt")
        torch.save(tensor, out_path)
        print(f"[{split_name}] saved: {out_path}")

    build_and_save(train_years, "train")
    if valid_years:
        build_and_save(valid_years, "valid")
    build_and_save(test_years, "test")

    print("Done.")

if __name__ == "__main__":
    RAW_DIR = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw"
    RAW_FILE = os.path.join(RAW_DIR, "osisaf_nh_sic_reprocessed.nc")

    # Download if missing
    if not os.path.exists(RAW_FILE):
        os.makedirs(RAW_DIR, exist_ok=True)
        print(f"Downloading to {RAW_FILE}...")
        # Download logic here (cURL, xarray, etc.)
        ds = xr.open_dataset(OPENDAP_URL)
        ds.to_netcdf(RAW_FILE)
    else:
        print(f"Using cached: {RAW_FILE}")
        ds = xr.open_dataset(RAW_FILE, chunks={"time": CHUNK_DAYS})
    main()
