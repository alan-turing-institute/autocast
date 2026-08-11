"""Training callbacks used by autocast.

This package holds Lightning callbacks used across training and evaluation.
"""

from .checkpoint import ProgressModelCheckpoint
from .ema import EMACallback
from .grad_norm import GradNormCallback
from .metrics import ValidationMetricPlotCallback
from .residual_statistics import ResidualStatisticsCallback

__all__ = [
    "EMACallback",
    "GradNormCallback",
    "ProgressModelCheckpoint",
    "ResidualStatisticsCallback",
    "ValidationMetricPlotCallback",
]
