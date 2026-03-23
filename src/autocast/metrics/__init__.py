from .coverage import Coverage, MultiCoverage
from .deterministic import MAE, MSE, NMAE, NMSE, NRMSE, RMSE, VMSE, VRMSE, LInfinity
from .ensemble import CRPS, AlphaFairCRPS, EnergyScore, FairCRPS, VariogramScore

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
    "EnergyScore",
    "FairCRPS",
    "LInfinity",
    "MultiCoverage",
    "VariogramScore",
]

ALL_DETERMINISTIC_METRICS = (MSE, MAE, NMAE, NMSE, RMSE, NRMSE, VMSE, VRMSE, LInfinity)
ALL_ENSEMBLE_METRICS = (
    CRPS,
    AlphaFairCRPS,
    FairCRPS,
    EnergyScore,
    VariogramScore,
    Coverage,
    MultiCoverage,
)
