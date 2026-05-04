"""Build the combined OSISAF + ERA5 dataset for one-shot 14-day SIC forecasting.

Usage:
    python build_osisaf_era5_dataset.py [--out_dir PATH]

Output directory layout (default: /projects/u6eo/qiencai/seaice/processed_osisaf_era5_2000_2020):
    train/data.pt          (N_train, 365, H, W, 13) float32
    valid/data.pt          (N_valid, 365, H, W, 13) float32
    test/data.pt           (N_test,  365, H, W, 13) float32
    train/constant_fields.pt  (N_train, H, W, 1)  land mask
    valid/constant_fields.pt
    test/constant_fields.pt
    stats.yaml             per-variable z-score stats from train split
    cache/year_{Y}.pt      per-year intermediate cache (resumable)

Channel layout (axis -1):
    0  : SIC              (OSISAF ice_conc, [0,1])
    1  : land_mask
    2  : u10              10m u-component of wind
    3  : v10              10m v-component of wind
    4  : d2m              2m dewpoint temperature
    5  : t2m              2m temperature
    6  : msl              mean sea level pressure
    7  : mwd              mean wave direction
    8  : sst              sea surface temperature
    9  : swh              significant height of combined wind waves
    10 : sp               surface pressure
    11 : tp               total precipitation
    12 : uvb              downward UV radiation at the surface

Split rule: 2000-2016 train, 2017-2018 valid, 2019-2020 test
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cfgrib
import numpy as np
import torch
import xarray as xr
import yaml
from scipy.ndimage import map_coordinates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

OSISAF_DIR = Path("/projects/u6eo/qiencai/seaice/raw_osisaf")
ERA5_DIR   = Path("/projects/u6eo/qiencai/seaice/ERA5")
LAND_MASK_PATH = OSISAF_DIR / "land_mask.pt"

DEFAULT_OUT_DIR = Path("/projects/u6eo/qiencai/seaice/processed_osisaf_era5_2000_2020")

TRAIN_YEARS = list(range(2000, 2017))
VALID_YEARS = [2017, 2018]
TEST_YEARS  = [2019, 2020]

ERA5_VARS = [
    "u10", "v10", "d2m", "t2m", "msl",
    "mwd", "sst", "swh", "sp", "tp", "uvb",
]
CHANNEL_NAMES = ["sic", "land_mask"] + ERA5_VARS


# ---------------------------------------------------------------------------
# OSISAF helpers
# ---------------------------------------------------------------------------

def _drop_feb29(arr: np.ndarray, times) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    idx = pd.DatetimeIndex(times)
    mask = ~((idx.month == 2) & (idx.day == 29))
    return arr[mask], times[mask]


def load_osisaf_year(year: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    nc_path = OSISAF_DIR / f"osisaf_nh_{year}.nc"
    ds = xr.open_dataset(str(nc_path))

    candidates = ["ice_conc", "conc", "sic", "ice_concentration"]
    sic_var = next((v for v in candidates if v in ds.data_vars), None)
    if sic_var is None:
        sic_var = list(ds.data_vars)[0]
    log.info("  OSISAF %d: variable '%s'", year, sic_var)

    da = ds[sic_var]
    spatial_dims = [d for d in da.dims if d != "time"]
    da = da.transpose("time", spatial_dims[0], spatial_dims[1])

    units = (da.attrs.get("units", "") or "").lower()
    if "%" in units or "percent" in units:
        da = da / 100.0

    arr   = np.asarray(da.values, dtype=np.float32)
    times = np.asarray(ds["time"].values)
    arr, times = _drop_feb29(arr, times)

    if arr.shape[0] != 365:
        log.warning("  Year %d: %d steps (expected 365), skipping.", year, arr.shape[0])
        return None, None

    arr = np.where((arr >= 0.0) & (arr <= 1.0), arr, np.nan)
    ds.close()
    return arr, times  # (365, H, W)


def get_osisaf_grid(year: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Return 2-D lat/lon arrays (H, W) for the OSISAF Lambert Azimuthal grid."""
    nc_path = OSISAF_DIR / f"osisaf_nh_{year}.nc"
    ds = xr.open_dataset(str(nc_path))

    for lat_name in ["lat", "latitude"]:
        if lat_name in ds.coords:
            lat_2d = np.asarray(ds.coords[lat_name].values, dtype=np.float64)
            break
    else:
        raise ValueError("Cannot find lat coordinate in OSISAF file")

    for lon_name in ["lon", "longitude"]:
        if lon_name in ds.coords:
            lon_2d = np.asarray(ds.coords[lon_name].values, dtype=np.float64)
            break
    else:
        raise ValueError("Cannot find lon coordinate in OSISAF file")

    ds.close()
    if lat_2d.ndim != 2 or lon_2d.ndim != 2:
        raise ValueError(f"Expected 2-D lat/lon, got lat.ndim={lat_2d.ndim}")
    log.info("  OSISAF grid shape: %s, lat [%.1f, %.1f]",
             lat_2d.shape, float(lat_2d.min()), float(lat_2d.max()))
    return lat_2d, lon_2d  # (H, W) each


# ---------------------------------------------------------------------------
# ERA5 helpers
# ---------------------------------------------------------------------------

def _regrid_var(
    era5_arr: np.ndarray,    # (T, H_era5, W_era5)  already lat-ascending, lon sorted
    era5_lat: np.ndarray,    # (H_era5,) ascending
    era5_lon: np.ndarray,    # (W_era5,) ascending [0,360)
    tgt_lat_2d: np.ndarray,  # (H, W)
    tgt_lon_2d: np.ndarray,  # (H, W)
) -> np.ndarray:             # (T, H, W) float32
    """Bilinear regrid with map_coordinates — vectorised over all T*H*W at once."""
    T = era5_arr.shape[0]
    H, W = tgt_lat_2d.shape

    tgt_lon_norm = tgt_lon_2d % 360.0

    dlat = float(era5_lat[1] - era5_lat[0])
    dlon = float(era5_lon[1] - era5_lon[0])

    lat_idx = (tgt_lat_2d   - era5_lat[0]) / dlat  # (H, W)
    lon_idx = (tgt_lon_norm - era5_lon[0]) / dlon  # (H, W)

    lat_flat = lat_idx.ravel().astype(np.float32)
    lon_flat = lon_idx.ravel().astype(np.float32)

    # Build coordinate arrays of shape (3, T*H*W)
    t_coords   = np.repeat(np.arange(T, dtype=np.float32), H * W)
    lat_coords = np.tile(lat_flat, T)
    lon_coords = np.tile(lon_flat, T)

    result = map_coordinates(
        era5_arr.astype(np.float64),
        [t_coords, lat_coords, lon_coords],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return result.reshape(T, H, W).astype(np.float32)


def load_era5_year(
    year: int,
    tgt_lat_2d: np.ndarray,
    tgt_lon_2d: np.ndarray,
) -> np.ndarray | None:
    """Load and regrid one year of ERA5 to the OSISAF 2-D grid.

    Returns float32 array of shape (365, H, W, 11) or None on failure.
    """
    grib_path = ERA5_DIR / f"era5_single_levels_{year}.grib"
    if not grib_path.exists():
        log.error("ERA5 file missing: %s", grib_path)
        return None

    log.info("  ERA5 %d: %s", year, grib_path)
    ds_list = cfgrib.open_datasets(str(grib_path))

    era5_arrays: dict[str, np.ndarray] = {}
    era5_grids:  dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for ds in ds_list:
        if "latitude" in ds.coords:
            lat_c = np.asarray(ds.coords["latitude"].values, dtype=np.float64)
        elif "lat" in ds.coords:
            lat_c = np.asarray(ds.coords["lat"].values, dtype=np.float64)
        else:
            continue

        if "longitude" in ds.coords:
            lon_c = np.asarray(ds.coords["longitude"].values, dtype=np.float64)
        elif "lon" in ds.coords:
            lon_c = np.asarray(ds.coords["lon"].values, dtype=np.float64)
        else:
            continue

        flip_lat = lat_c[0] > lat_c[-1]
        if flip_lat:
            lat_c = lat_c[::-1]

        lon_norm = lon_c % 360.0
        sort_idx = np.argsort(lon_norm)
        lon_sorted = lon_norm[sort_idx]

        for var in ERA5_VARS:
            if var not in ds.data_vars or var in era5_arrays:
                continue
            arr   = np.asarray(ds[var].values, dtype=np.float32)
            times = np.asarray(ds["time"].values)
            arr, times = _drop_feb29(arr, times)
            if arr.shape[0] != 365:
                log.warning("  ERA5 var %s year %d: %d steps (expected 365)", var, year, arr.shape[0])
                continue
            if flip_lat:
                arr = arr[:, ::-1, :]
            arr = arr[:, :, sort_idx]
            era5_arrays[var] = arr
            era5_grids[var]  = (lat_c, lon_sorted)

        try:
            ds.close()
        except Exception:
            pass

    missing = [v for v in ERA5_VARS if v not in era5_arrays]
    if missing:
        log.warning("  ERA5 %d: missing %s", year, missing)
        return None

    H, W  = tgt_lat_2d.shape
    out   = np.full((365, H, W, len(ERA5_VARS)), np.nan, dtype=np.float32)

    for vi, var in enumerate(ERA5_VARS):
        era5_lat, era5_lon = era5_grids[var]
        out[:, :, :, vi] = _regrid_var(
            era5_arrays[var], era5_lat, era5_lon, tgt_lat_2d, tgt_lon_2d
        )
        log.info("    %s regridded OK", var)

    log.info("  ERA5 %d: done -> (365, %d, %d, %d)", year, H, W, len(ERA5_VARS))
    return out


# ---------------------------------------------------------------------------
# Per-year caching
# ---------------------------------------------------------------------------

def process_year(
    year: int,
    land_mask: np.ndarray,
    tgt_lat_2d: np.ndarray,
    tgt_lon_2d: np.ndarray,
    cache_dir: Path,
) -> np.ndarray | None:
    cache_path = cache_dir / f"year_{year}.pt"
    if cache_path.exists():
        log.info("  Year %d: cache hit %s", year, cache_path)
        return torch.load(str(cache_path), map_location="cpu").numpy()

    sic, _ = load_osisaf_year(year)
    if sic is None:
        return None
    era5 = load_era5_year(year, tgt_lat_2d, tgt_lon_2d)
    if era5 is None:
        return None

    H, W = sic.shape[1], sic.shape[2]
    sic_ch  = sic[..., np.newaxis]                                                   # (365,H,W,1)
    mask_ch = np.broadcast_to(land_mask[np.newaxis], (365, H, W, 1)).copy()         # (365,H,W,1)
    combined = np.concatenate([sic_ch, mask_ch, era5], axis=-1)                     # (365,H,W,13)

    torch.save(torch.from_numpy(combined), str(cache_path))
    log.info("  Year %d cached: %s", year, combined.shape)
    return combined


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------

def build_split(
    split_years: list[int],
    land_mask: np.ndarray,
    tgt_lat_2d: np.ndarray,
    tgt_lon_2d: np.ndarray,
    out_dir: Path,
    cache_dir: Path,
    split_name: str,
) -> list[np.ndarray]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    traj_list: list[np.ndarray] = []
    for year in split_years:
        log.info("== %s year %d ==", split_name, year)
        combined = process_year(year, land_mask, tgt_lat_2d, tgt_lon_2d, cache_dir)
        if combined is None:
            log.warning("Skipping year %d.", year)
            continue
        traj_list.append(combined)

    if not traj_list:
        raise RuntimeError(f"No usable trajectories for split={split_name}")

    data = np.stack(traj_list, axis=0)   # (N, 365, H, W, 13)
    log.info("[%s] shape=%s dtype=%s", split_name, data.shape, data.dtype)
    torch.save(torch.from_numpy(data), split_dir / "data.pt")
    log.info("[%s] saved data.pt", split_name)

    N, _, H, W, _ = data.shape
    cf = np.broadcast_to(land_mask[np.newaxis], (N, H, W, 1)).copy()
    torch.save(torch.from_numpy(cf.astype(np.float32)), split_dir / "constant_fields.pt")
    log.info("[%s] saved constant_fields.pt", split_name)

    return traj_list


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_and_save_stats(train_trajectories: list[np.ndarray], out_dir: Path) -> None:
    all_flat = np.concatenate(
        [t.reshape(-1, t.shape[-1]) for t in train_trajectories], axis=0
    )
    means = np.nanmean(all_flat, axis=0)
    stds  = np.nanstd(all_flat, axis=0)
    stds  = np.where(stds < 1e-8, 1.0, stds)

    yaml_data = {
        "stats": {
            "mean":       {n: float(means[i]) for i, n in enumerate(CHANNEL_NAMES)},
            "std":        {n: float(stds[i])  for i, n in enumerate(CHANNEL_NAMES)},
            "mean_delta": dict.fromkeys(CHANNEL_NAMES, 0.0),
            "std_delta":  dict.fromkeys(CHANNEL_NAMES, 1.0),
        },
        "core_field_names":     CHANNEL_NAMES,
        "constant_field_names": [],
    }

    with open(out_dir / "stats.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    log.info("Saved stats.yaml")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    log.info("Output: %s", out_dir)

    tgt_lat_2d, tgt_lon_2d = get_osisaf_grid(2000)

    mask_obj = torch.load(str(LAND_MASK_PATH), map_location="cpu")
    if isinstance(mask_obj, dict):
        mask_tensor = list(mask_obj.values())[0]
    else:
        mask_tensor = mask_obj
    land_mask = mask_tensor.numpy().astype(np.float32)
    if land_mask.ndim == 2:
        land_mask = land_mask[:, :, np.newaxis]
    elif land_mask.ndim == 3 and land_mask.shape[0] in (1, 2):
        land_mask = land_mask[0, :, :, np.newaxis]
    log.info("Land mask shape: %s", land_mask.shape)

    train_trajs = build_split(TRAIN_YEARS, land_mask, tgt_lat_2d, tgt_lon_2d, out_dir, cache_dir, "train")
    build_split(VALID_YEARS, land_mask, tgt_lat_2d, tgt_lon_2d, out_dir, cache_dir, "valid")
    build_split(TEST_YEARS,  land_mask, tgt_lat_2d, tgt_lon_2d, out_dir, cache_dir, "test")

    compute_and_save_stats(train_trajs, out_dir)
    log.info("Dataset build complete: %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    main(args.out_dir)
