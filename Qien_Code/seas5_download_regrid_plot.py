"""
SEAS5 sea-ice-cover download, regrid to OSI-SAF EASE2-250 grid, and
comparison plot against ViT + Flow Matching + Persistence.

Usage:
    python seas5_download_regrid_plot.py [--skip-download] [--skip-regrid]
"""

import argparse
import time
import warnings
from pathlib import Path

import cdsapi
import cfgrib
import numpy as np
import requests
import torch
import xarray as xr
from pyresample import geometry, kd_tree

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/lus/lfs1aip2/projects/u6eo/qiencai/seaice")
SEAS5_DIR = BASE / "seas5"
SEAS5_DIR.mkdir(parents=True, exist_ok=True)

SEAS5_GRIB = SEAS5_DIR / "seas5_sic_2019_2020.grib"
SEAS5_PT   = SEAS5_DIR / "seas5_regridded_2019_2020.pt"
OSISAF_NC  = BASE / "data/masks/north/siconca/2000/01/ice_conc_nh_ease2-250_cdr-v2p0_200001021200.nc"

_QIENCAI = Path("/lus/lfs1aip2/projects/u6eo/qiencai")
VIT_CSV = _QIENCAI / "outputs/2026-04-30/epd_default_supervised_vit_57329b6_14da3d7/eval_stride4/rollout_metrics_per_timestep_channel_0.csv"
FM_CSV  = _QIENCAI / "outputs/2026-03-15/diff_default_flow_matching_22fb5ef_1db57ee/eval/rollout_metrics_per_timestep_channel_0.csv"
OUT_DIR = _QIENCAI / "outputs/2026-04-30/epd_default_supervised_vit_57329b6_14da3d7/eval_stride4"
OUT_PLOT = OUT_DIR / "comparison_vit_fm_seas5.png"

# Ocean mask: 1 = ocean, 0 = land; 97777 ocean pixels out of 186624
_OCEAN_MASK = torch.load(
    str(BASE / "processed_osisaf_full/train/constant_fields.pt"),
    weights_only=False,
).bool().flatten()


# ---------------------------------------------------------------------------
# 1. Download  (with retry on CDS queue-full / 400 errors)
# ---------------------------------------------------------------------------
def download(max_retries: int = 10, base_wait: int = 300):
    """
    Retry loop for CDS queue rejections ("Number queued requests … limited").
    Waits base_wait seconds on first retry, doubling each time up to 1 hour.
    """
    REQUEST = {
        "originating_centre": "ecmwf",
        "system": "5",
        "variable": ["sea_ice_cover"],
        "year": ["2019", "2020"],
        "month": ["01","02","03","04","05","06","07","08","09","10","11","12"],
        "day": ["01"],
        "leadtime_hour": [
            "24","48","72","96","120","144","168",
            "192","216","240","264","288","312","336",
        ],
        "data_format": "grib",
        "area": [90, -180, 30, 180],
    }

    for attempt in range(1, max_retries + 1):
        print(f"=== Downloading SEAS5 from CDS (attempt {attempt}/{max_retries}) ===")
        try:
            client = cdsapi.Client()
            client.retrieve("seasonal-original-single-levels", REQUEST).download(
                str(SEAS5_GRIB)
            )
            print(f"Saved: {SEAS5_GRIB}")
            return
        except requests.exceptions.HTTPError as exc:
            msg = str(exc)
            if "queued" in msg.lower() or "rejected" in msg.lower() or "400" in msg:
                wait = min(base_wait * (5 ** (attempt - 1)), 3600)
                print(f"  CDS queue full (attempt {attempt}). Retrying in {wait}s …")
                time.sleep(wait)
            else:
                raise  # unexpected HTTP error — don't retry

    raise RuntimeError(f"CDS download failed after {max_retries} attempts.")


# ---------------------------------------------------------------------------
# 2. Regrid GRIB → OSI-SAF EASE2-250 432×432 grid
# ---------------------------------------------------------------------------
def regrid():
    print("=== Regridding SEAS5 to OSI-SAF grid ===")

    # --- Load OSI-SAF target grid ---
    ref = xr.open_dataset(OSISAF_NC)
    lat_tgt = ref["lat"].values.astype(np.float64)   # (432, 432)
    lon_tgt = ref["lon"].values.astype(np.float64)

    # land mask from processed_osisaf_full: land pixels are 0 in SIC
    # (we keep the same "no NaN mask" convention)
    target_def = geometry.GridDefinition(lons=lon_tgt, lats=lat_tgt)

    # --- Read GRIB ---
    print("  Reading GRIB with cfgrib...")
    ds_list = cfgrib.open_datasets(str(SEAS5_GRIB), indexpath=None)
    # find the dataset that has 'siconc' or 'ci' (SEAS5 sea-ice variable)
    ds = None
    for d in ds_list:
        for var in list(d.data_vars) + list(d.coords):
            if var.lower() in {"siconc", "ci", "sea_ice_cover", "sic"}:
                ds = d
                sic_var = var
                break
        if ds is not None:
            break
    if ds is None:
        # fallback: first variable in first dataset
        ds = ds_list[0]
        sic_var = list(ds.data_vars)[0]
    print(f"  Using variable: {sic_var}, dims: {ds[sic_var].dims}")
    print(f"  Shape: {ds[sic_var].shape}")

    # SEAS5 is on a regular lat/lon grid — use bilinear interpolation
    from scipy.interpolate import RegularGridInterpolator

    lons_src = ds["longitude"].values.astype(np.float64)  # 1D, e.g. [0,1,...,359]
    lats_src = ds["latitude"].values.astype(np.float64)   # 1D, e.g. [90,...,30]

    # Normalize source longitudes to [-180, 180] to match target
    lons_src_norm = (lons_src + 180.0) % 360.0 - 180.0

    # Target grid flattened for interpolation queries
    lon_tgt_flat = lon_tgt.ravel()   # (432*432,)
    lat_tgt_flat = lat_tgt.ravel()

    # Clamp target coords to source grid extent to avoid extrapolation NaNs
    lon_tgt_flat = np.clip(lon_tgt_flat, lons_src_norm.min(), lons_src_norm.max())
    lat_tgt_flat = np.clip(lat_tgt_flat, lats_src.min(), lats_src.max())

    # Iterate over (time, number/member) slices
    sic_data = ds[sic_var]  # dims: (number, time, step, latitude, longitude)
    dims = sic_data.dims
    print(f"  Dims: {dims}")

    # Average over ensemble members; keep time (init months) and step (lead days)
    if "number" in dims:
        sic_data = sic_data.mean(dim="number")
        dims = sic_data.dims
        print(f"  Averaged over ensemble members. New dims: {dims}")

    # Expect dims: (time, step, latitude, longitude)
    assert "time" in dims and "step" in dims, f"Unexpected dims: {dims}"

    n_time  = sic_data.sizes["time"]
    n_steps = sic_data.sizes["step"]

    # Extract init dates and step hours
    import pandas as pd
    init_times = pd.DatetimeIndex(sic_data["time"].values)
    step_hours = np.array([int(s / np.timedelta64(1, "h")) if hasattr(s, "astype")
                           else int(s) for s in sic_data["step"].values])
    print(f"  Init dates: {init_times[0].date()} … {init_times[-1].date()}  ({n_time} months)")
    print(f"  Step hours: {step_hours}")

    # Sort lats to ascending order (required by RegularGridInterpolator)
    if lats_src[0] > lats_src[-1]:
        lats_axis = lats_src[::-1]
        lat_flip = True
    else:
        lats_axis = lats_src
        lat_flip = False

    # Build interpolation query points once
    pts = np.column_stack([lat_tgt_flat, lon_tgt_flat])

    # Output: (n_time, n_steps, 432, 432)
    regridded = np.zeros((n_time, n_steps, 432, 432), dtype=np.float32)

    print(f"  Regridding {n_time} inits × {n_steps} lead steps with bilinear interpolation...")
    for i in range(n_time):
        for j in range(n_steps):
            frame = sic_data.isel(time=i, step=j).values.astype(np.float64)
            if lat_flip:
                frame = frame[::-1, :]
            interp = RegularGridInterpolator(
                (lats_axis, lons_src_norm),
                frame,
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            regridded[i, j] = interp(pts).reshape(432, 432).astype(np.float32)

    # SEAS5 SIC is in [0,1] — convert to [0,100] to match processed_osisaf_full
    regridded = (regridded * 100.0).clip(0.0, 100.0)

    out = {
        "sic":        torch.from_numpy(regridded),  # (n_time, n_steps, 432, 432) [0,100]
        "step_hours": step_hours,                    # [24, 48, ..., 336]
        "init_times": np.array(init_times),          # (n_time,) datetime64
    }
    torch.save(out, SEAS5_PT)
    print(f"  Saved regridded data: {SEAS5_PT}  shape={regridded.shape}")
    return out


# ---------------------------------------------------------------------------
# 3. Compute SEAS5 MSE/MAE/RMSE/VRMSE per lead step
# ---------------------------------------------------------------------------
def compute_seas5_metrics(sic_seas5_100, step_hours, init_times):
    """
    sic_seas5_100 : (n_time, n_steps, 432, 432) in [0, 100]
                    n_time  = 24 init months (Jan2019..Dec2020)
                    n_steps = 14 lead steps  (24h..336h)
    init_times    : array of n_time datetime64 values (1st of each month)
    step_hours    : array of n_steps ints, e.g. [24, 48, ..., 336]

    For each lead step t, computes MSE between seas5[i,t] and the GT frame
    at init_date[i] + (t+1) days, then averages over the 24 init months.
    GT shape: (2, 365, 432, 432), year0=2019, year1=2020, day-0=Jan1.
    2020 is stored as 365 days (Feb 29 skipped), so post-Feb dates are
    shifted by -1 day index.
    """
    import pandas as pd

    # Ground truth
    print("  Loading processed_osisaf_full ground truth...")
    full = torch.load(
        "/lus/lfs1aip2/projects/u6eo/qiencai/seaice/processed_osisaf_full/test/data.pt",
        map_location="cpu",
    )
    sic_gt = full["data"][..., 0].float()  # (2, 365, 432, 432) [0,100]

    def date_to_gt_idx(ts):
        """Return (year_idx, day_idx) for a pandas Timestamp into sic_gt."""
        year_idx = 0 if ts.year == 2019 else 1
        doy = ts.dayofyear - 1  # 0-based
        if ts.year == 2020 and ts.month > 2:
            doy -= 1  # Feb 29 is skipped in the 365-day GT
        return year_idx, doy

    step_hour_to_idx = {int(h): j for j, h in enumerate(step_hours)}
    T_out = len(step_hours)
    init_ts = pd.DatetimeIndex(init_times)

    mse_s, mae_s, vrmse_s = [], [], []

    for t in range(T_out):
        lead_days = step_hours[t] // 24  # e.g. 1..14
        all_mse, all_mae, all_vrmse = [], [], []

        for i, ts in enumerate(init_ts):
            target_date = ts + pd.Timedelta(days=int(lead_days))
            # Skip if target falls outside 2019-2020
            if target_date.year not in (2019, 2020):
                continue
            yr_idx, day_idx = date_to_gt_idx(target_date)
            if day_idx < 0 or day_idx >= 365:
                continue

            pred = sic_seas5_100[i, t].float().flatten()[_OCEAN_MASK]
            gt   = sic_gt[yr_idx, day_idx].flatten()[_OCEAN_MASK]
            d    = (pred - gt) / 100.0                      # [0,1] scale
            all_mse.append((d**2).mean().item())
            all_mae.append(d.abs().mean().item())
            all_vrmse.append(
                (d**2).mean().item() ** 0.5 / ((gt / 100.0).std().item() + 1e-7)
            )

        mse_s.append(np.mean(all_mse) if all_mse else np.nan)
        mae_s.append(np.mean(all_mae) if all_mae else np.nan)
        vrmse_s.append(np.mean(all_vrmse) if all_vrmse else np.nan)

    print(f"  SEAS5 MSE per lead day: {[f'{v:.4f}' for v in mse_s]}")
    return {
        "mse":   mse_s,
        "mae":   mae_s,
        "rmse":  [m**0.5 if not np.isnan(m) else np.nan for m in mse_s],
        "vrmse": vrmse_s,
    }


# ---------------------------------------------------------------------------
# 4. Compute persistence metrics (same as before)
# ---------------------------------------------------------------------------
def compute_persistence_metrics():
    T_out, n_in, STRIDE = 14, 5, 4
    print("  Loading processed_osisaf_full for persistence...")
    full = torch.load(
        "/lus/lfs1aip2/projects/u6eo/qiencai/seaice/processed_osisaf_full/test/data.pt",
        map_location="cpu",
    )
    sic = full["data"][..., 0].float() / 100.0  # [0,1]
    n_traj, n_days = sic.shape[:2]

    mse_p, mae_p, vrmse_p = [], [], []
    for t in range(T_out):
        all_mse, all_mae, all_vrmse = [], [], []
        for yr in range(n_traj):
            yr_sic = sic[yr]
            for s in range(0, n_days - (n_in + T_out) + 1, STRIDE):
                persist = yr_sic[s + n_in - 1].flatten()[_OCEAN_MASK]
                gt      = yr_sic[s + n_in + t].flatten()[_OCEAN_MASK]
                d = persist - gt
                all_mse.append((d**2).mean().item())
                all_mae.append(d.abs().mean().item())
                all_vrmse.append(
                    (d**2).mean().item() ** 0.5 / (gt.std().item() + 1e-7)
                )
        mse_p.append(np.mean(all_mse))
        mae_p.append(np.mean(all_mae))
        vrmse_p.append(np.mean(all_vrmse))

    return {
        "mse":   mse_p,
        "mae":   mae_p,
        "rmse":  [m**0.5 for m in mse_p],
        "vrmse": vrmse_p,
    }



# ---------------------------------------------------------------------------
# 5. Compute linear-trend baseline metrics
# ---------------------------------------------------------------------------
_TRAIN_YEARS = [
    1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
    2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
    2010, 2011, 2012, 2013, 2014, 2015, 2016,
]  # 25 years (year 2000 was gap-filtered out of OSI-SAF train set)


def compute_linear_trend_metrics(step_hours, init_times):
    """
    Per-pixel, per-DOY linear trend extrapolation baseline.

    For each target date D in the evaluation set (same dates as SEAS5):
      - DOY d = day-of-year of D (0-based, Feb-29-free for 2020)
      - Fit linear(year -> SIC) over the 25 training years for DOY d
      - Predict SIC at year(D) by extrapolating that line
    """
    import pandas as pd

    print("  Loading train data for linear trend fit...")
    train_dict = torch.load(
        "/lus/lfs1aip2/projects/u6eo/qiencai/seaice/processed_osisaf_full/train/data.pt",
        map_location="cpu",
        weights_only=False,
    )
    # (25, 365, 432, 432, 1) -> numpy (25, 365, 432, 432)
    train_np = train_dict["data"].numpy()[:, :, :, :, 0]  # float32
    del train_dict
    H, W = train_np.shape[2], train_np.shape[3]

    X = np.array(_TRAIN_YEARS, dtype=np.float32)   # (25,)
    X_mean = X.mean()
    X_c = X - X_mean                               # centred
    X_c_sq = float((X_c ** 2).sum())

    print("  Computing per-DOY per-pixel linear slopes over 25 training years...")
    # Y shape: (25, 365, H*W)
    Y = train_np.reshape(25, 365, H * W).astype(np.float64)
    Y_mean = Y.mean(axis=0)                        # (365, H*W)
    Y_c    = Y - Y_mean[None]                      # (25, 365, H*W)
    slope  = (X_c[:, None, None] * Y_c).sum(axis=0) / X_c_sq  # (365, H*W)
    slope  = slope.reshape(365, H, W).astype(np.float32)
    Y_mean = Y_mean.reshape(365, H, W).astype(np.float32)
    del Y, Y_c, train_np
    print("  Linear trend slopes computed.")

    # Ground truth
    full = torch.load(
        "/lus/lfs1aip2/projects/u6eo/qiencai/seaice/processed_osisaf_full/test/data.pt",
        map_location="cpu",
        weights_only=False,
    )
    sic_gt = full["data"][..., 0].float()  # (2, 365, 432, 432) [0,100]
    del full

    def date_to_gt_idx(ts):
        year_idx = 0 if ts.year == 2019 else 1
        doy = ts.dayofyear - 1
        if ts.year == 2020 and ts.month > 2:
            doy -= 1
        return year_idx, doy

    init_ts   = pd.DatetimeIndex(init_times)
    T_out     = len(step_hours)
    mse_l, mae_l, vrmse_l = [], [], []

    for t in range(T_out):
        lead_days = int(step_hours[t]) // 24
        all_mse, all_mae, all_vrmse = [], [], []

        for ts in init_ts:
            target_date = ts + pd.Timedelta(days=lead_days)
            if target_date.year not in (2019, 2020):
                continue
            yr_idx, day_idx = date_to_gt_idx(target_date)
            if day_idx < 0 or day_idx >= 365:
                continue

            # Linear-trend forecast for (year, doy)
            pred_np = (
                Y_mean[day_idx]
                + slope[day_idx] * (float(target_date.year) - float(X_mean))
            )  # (432, 432) in [0,100] scale
            pred = torch.from_numpy(np.clip(pred_np, 0.0, 100.0)).flatten()[_OCEAN_MASK]
            gt   = sic_gt[yr_idx, day_idx].flatten()[_OCEAN_MASK]
            d    = (pred - gt) / 100.0
            all_mse.append((d ** 2).mean().item())
            all_mae.append(d.abs().mean().item())
            all_vrmse.append(
                (d ** 2).mean().item() ** 0.5 / ((gt / 100.0).std().item() + 1e-7)
            )

        mse_l.append(np.mean(all_mse)   if all_mse else np.nan)
        mae_l.append(np.mean(all_mae)   if all_mae else np.nan)
        vrmse_l.append(np.mean(all_vrmse) if all_vrmse else np.nan)

    print(f"  Linear-trend MSE per lead day: {[f'{v:.4f}' for v in mse_l]}")
    return {
        "mse":   mse_l,
        "mae":   mae_l,
        "rmse":  [m ** 0.5 if not np.isnan(m) else np.nan for m in mse_l],
        "vrmse": vrmse_l,
    }


# ---------------------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------------------
def plot(seas5_metrics, persist_metrics, trend_metrics):
    import matplotlib
    import pandas as pd
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T_out = 14
    steps = list(range(T_out))

    vit = pd.read_csv(VIT_CSV, index_col=0)
    fm  = pd.read_csv(FM_CSV,  index_col=0)

    # FM rescaling: SIC was [0,100] during eval → divide to [0,1] scale
    for m in fm.index:
        if m in {"mse", "vmse", "nmse"}:            fm.loc[m] /= 10000.0
        elif m in {"mae", "rmse", "nmae", "nrmse"}: fm.loc[m] /= 100.0

    # Rescale ViT/FM from all-pixel to ocean-only convention.
    # Land pixels are exactly 0 in both pred and GT, so:
    #   MSE_ocean = MSE_all * (N_total / N_ocean)
    #   MAE_ocean = MAE_all * (N_total / N_ocean)
    #   RMSE_ocean = RMSE_all * sqrt(N_total / N_ocean)
    # VRMSE is left as-is (denominator sigma_GT differs non-trivially).
    _N_RATIO      = 186624 / 97777   # ~1.909
    _N_RATIO_SQRT = _N_RATIO ** 0.5  # ~1.382
    for df in (vit, fm):
        for m in df.index:
            if m in {"mse", "nmse"} or m in {"mae", "nmae"}:         df.loc[m] *= _N_RATIO
            elif m in {"rmse", "nrmse"}:      df.loc[m] *= _N_RATIO_SQRT

    metrics = ["mse", "mae", "rmse"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "ViT vs Flow Matching vs SEAS5 vs Persistence vs Linear Trend — SIC [0–1] scale\n"
        "stride=4, processed_osisaf_full (land=0), test years 2019–2020",
        fontsize=11,
    )

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.plot(steps, vit.loc[metric].values, marker="o", ms=4,
                label="ViT EPD (stride=4)")
        ax.plot(steps, fm.loc[metric].values,  marker="s", ms=4,
                label="Flow Matching (stride=4)")
        ax.plot(steps, seas5_metrics[metric],  marker="D", ms=4,
                label="SEAS5 (monthly init)")
        ax.plot(steps, persist_metrics[metric], marker="^", ms=4,
                ls="--", color="gray", label="Persistence (stride=4)")
        ax.plot(steps, trend_metrics[metric],   marker="x", ms=5,
                ls="-.", color="brown", label="Linear trend (per-DOY)")
        ax.set_title(metric.upper())
        ax.set_xlabel("Lead step (days)")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {OUT_PLOT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-regrid",   action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        download()
    else:
        print("Skipping download.")

    if not args.skip_regrid:
        seas5_data = regrid()
    else:
        print("Loading cached regridded data...")
        seas5_data = torch.load(SEAS5_PT, map_location="cpu", weights_only=False)

    sic_seas5  = seas5_data["sic"]        # (n_time, n_steps, 432, 432) [0,100]
    step_hours = seas5_data["step_hours"]
    init_times = seas5_data["init_times"]
    print(f"SEAS5 regridded shape: {sic_seas5.shape}, steps: {step_hours}")

    print("=== Computing SEAS5 metrics ===")
    seas5_metrics = compute_seas5_metrics(sic_seas5, step_hours, init_times)

    print("=== Computing persistence metrics ===")
    persist_metrics = compute_persistence_metrics()

    print("=== Computing linear-trend baseline metrics ===")
    trend_metrics = compute_linear_trend_metrics(step_hours, init_times)

    print("=== Plotting ===")
    plot(seas5_metrics, persist_metrics, trend_metrics)
