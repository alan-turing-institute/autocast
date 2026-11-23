from autocast.losses.anomaly import SpectralSpreadErrorLoss
from autocast.losses.composite import WeightedSumLoss
from autocast.losses.decomposition import (
    ComponentBranch,
    ComponentObjective,
    ComponentTransform,
    FieldComponents,
    FieldDecomposition,
    IdentityDecomposition,
    ScoringBranch,
    SequentialDecomposition,
    TransformedLoss,
)
from autocast.losses.ensemble import (
    AlphaFairCRPSLoss,
    CoefficientCRPSLoss,
    CRPSLoss,
    EnsembleMAELoss,
    FairCRPSLoss,
    JointCoefficientEnergyScore,
)
from autocast.losses.regularization import MCDropoutMSEL2Loss
from autocast.losses.scale import (
    FourierLowPass,
    GaussianSmoothing,
    ScaleDecomposition,
    SmoothingOperator,
)
from autocast.losses.spectral import (
    CoefficientLpLoss,
    DTCWTDecomposition,
    FourierDecomposition,
    MagnitudeTransform,
    PowerTransform,
    RadialFourierDecomposition,
    RMSAmplitudeTransform,
    WaveletDecomposition,
)
from autocast.losses.vector import HelmholtzDecomposition, VelocityScoreLoss

__all__ = [
    "AlphaFairCRPSLoss",
    "CRPSLoss",
    "CoefficientCRPSLoss",
    "CoefficientLpLoss",
    "ComponentBranch",
    "ComponentObjective",
    "ComponentTransform",
    "DTCWTDecomposition",
    "EnsembleMAELoss",
    "FairCRPSLoss",
    "FieldComponents",
    "FieldDecomposition",
    "FourierDecomposition",
    "FourierLowPass",
    "GaussianSmoothing",
    "HelmholtzDecomposition",
    "IdentityDecomposition",
    "JointCoefficientEnergyScore",
    "MCDropoutMSEL2Loss",
    "MagnitudeTransform",
    "PowerTransform",
    "RMSAmplitudeTransform",
    "RadialFourierDecomposition",
    "ScaleDecomposition",
    "ScoringBranch",
    "SequentialDecomposition",
    "SmoothingOperator",
    "SpectralSpreadErrorLoss",
    "TransformedLoss",
    "VelocityScoreLoss",
    "WaveletDecomposition",
    "WeightedSumLoss",
]
