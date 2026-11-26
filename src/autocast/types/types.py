import numpy as np
import torch
from jaxtyping import Float, Inexact, Real
from torch.utils.data import DataLoader

# Alias for torch.Tensor
Tensor = torch.Tensor

# Type hints for various tensor shapes:
# - B: exactly one batch dim
# - T: exactly one time dim
# - S: one or more spatial dims
# - C: exactly one channel dim
# - L: one or more latent dims
# - N: zero or more arbirtrary dims (e.g. NDArray that can be 0 or more dims)
# - W: exactly one width dim
# - H: exactly one height dim
# - D: exactly one depth dim
# - M: exactly one model dim (e.g. for ensembles)
# - G: one or more dims of a grid on the sphere (nlat nlon, or npix for HEALPix)

TensorC = Float[Tensor, "channel"]  # Vector of scalars
TensorBC = Float[Tensor, "batch channel"]  # Only batch and channel
TensorBNC = Float[Tensor, "batch *optional_dims channel"]
TensorBNCM = Float[Tensor, "batch *optional_dims channel ensemble"]  # ensemble
TensorBTNC = Float[Tensor, "batch time *optional_dims channel"]
TensorBSC = Float[Tensor, "batch spatial *spatial channel"]
TensorBLC = Float[Tensor, "batch latent *latent channel"]
TensorBCL = Float[Tensor, "batch channel latent *latent"]

TensorBTSC = Float[Tensor, "batch time spatial *spatial channel"]  # Channels last
TensorBTSCM = Float[Tensor, "batch time spatial *spatial channel ensemble"]  # ensemble
# Integer or floating values, for metrics that accept integer-valued fields.
TensorRealBTSC = Real[Tensor, "batch time spatial *spatial channel"]
TensorRealBTSCM = Real[Tensor, "batch time spatial *spatial channel ensemble"]
# Packed (mu, log_var) along the channel axis, so the final dim is 2*C. jaxtyping
# cannot express the 2*C arithmetic constraint, so the trailing dim uses a distinct
# name `double_channel` instead of `channel`.
TensorBTSC2C = Float[Tensor, "batch time spatial *spatial double_channel"]
TensorInexactBTSC = Inexact[
    Tensor, "batch time spatial *spatial channel"
]  # Real or complex
TensorInexactBTSCM = Inexact[
    Tensor, "batch time spatial *spatial channel ensemble"
]  # Real or complex ensemble
FieldTensor = TensorInexactBTSC | TensorInexactBTSCM
TensorBCTS = Float[Tensor, "batch channel time spatial *spatial"]  # Channels first
TensorBCS = Float[Tensor, "batch channel spatial *spatial"]  # No time dimension
TensorBSSC = Float[Tensor, "batch spatial *spatial channel"]  # No time dimension

TensorTSC = Float[Tensor, "time spatial *spatial channel"]  # No batch dimension
TensorNC = Float[Tensor, "*optional_dims channel"]  # No batch dimension
TensorSC = Float[Tensor, "spatial *spatial channel"]  # No batch dimension
TensorS = Float[Tensor, "spatial *spatial"]  # No batch or channel dimension

TensorBCWH = Float[Tensor, "batch channel width height"]  # Specific spatial dims
TensorBTCHW = Float[Tensor, "batch time channel height width"]  # Specific spatial dims

TensorBTC = Float[Tensor, "batch time channel"]

# Fields on a grid on the sphere, channels last; the batch types name the grid.
TensorBTGC = Float[Tensor, "batch time grid *grid channel"]
TensorBGC = Float[Tensor, "batch grid *grid channel"]  # No time dimension
TensorTGC = Float[Tensor, "time grid *grid channel"]  # No batch dimension
TensorGC = Float[Tensor, "grid *grid channel"]  # No batch or time dimension

ArrayLike = Tensor | np.ndarray


# # Currently not used, but kept for reference
# TensorBTWHC = Float[Tensor, "batch time width height channel"]
# TensorBTWHDC = Float[Tensor, "batch time width height depth channel"]
# TensorBCTWH = Float[Tensor, "batch channel time width height"]
# TensorBCTWHD = Float[Tensor, "batch channel time width height depth"]
# TensorBCTHW = Float[Tensor, "batch channel time height width"]

# # Spatial only (no time dimension)
# TensorBWHC = Float[Tensor, "batch width height channel"]
# TensorBWHDC = Float[Tensor, "batch width height depth channel"]


# Generic input type
Input = Tensor | DataLoader

# Rollout output type
RolloutOutput = tuple[Tensor, None] | tuple[Tensor, Tensor]
