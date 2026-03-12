from .coverage import Coverage, MultiCoverage
from .deterministic import (
    MAE,
    MSE,
    NMAE,
    NMSE,
    NRMSE,
    RMSE,
    VMSE,
    VRMSE,
    LInfinity,
    PowerSpectrumRMSE,
    PowerSpectrumRMSEHigh,
    PowerSpectrumRMSELow,
    PowerSpectrumRMSEMid,
    PowerSpectrumRMSETail,
)
from .ensemble import CRPS, AlphaFairCRPS, FairCRPS

__all__ = [
    "CRPS",
    "MAE",
    "MSE",
    "NMAE",
    "NMSE",
    "NRMSE",
    "RMSE",
    "VMSE",
    "VRMSE",
    "AlphaFairCRPS",
    "Coverage",
    "FairCRPS",
    "LInfinity",
    "MultiCoverage",
    "PowerSpectrumRMSE",
    "PowerSpectrumRMSEHigh",
    "PowerSpectrumRMSELow",
    "PowerSpectrumRMSEMid",
    "PowerSpectrumRMSETail",
]

ALL_DETERMINISTIC_METRICS = (
    MSE,
    MAE,
    NMAE,
    NMSE,
    RMSE,
    NRMSE,
    VMSE,
    VRMSE,
    LInfinity,
    PowerSpectrumRMSE,
    PowerSpectrumRMSELow,
    PowerSpectrumRMSEMid,
    PowerSpectrumRMSEHigh,
    PowerSpectrumRMSETail,
)
ALL_ENSEMBLE_METRICS = (
    CRPS,
    AlphaFairCRPS,
    FairCRPS,
    Coverage,
    MultiCoverage,
)
