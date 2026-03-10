from .coverage import Coverage, MultiCoverage
from .deterministic import MAE, MSE, NMAE, NMSE, NRMSE, RMSE, VMSE, VRMSE, LInfinity
from .ensemble import CRPS, AlphaFairCRPS, FairCRPS, SpreadSkillRatio

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
    "SpreadSkillRatio",
]

ALL_DETERMINISTIC_METRICS = (MSE, MAE, NMAE, NMSE, RMSE, NRMSE, VMSE, VRMSE, LInfinity)
ALL_ENSEMBLE_METRICS = (
    CRPS,
    AlphaFairCRPS,
    FairCRPS,
    SpreadSkillRatio,
    Coverage,
    MultiCoverage,
)
