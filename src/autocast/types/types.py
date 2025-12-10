import torch
from jaxtyping import Float
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

TensorC = Float[Tensor, "channel"]  # Vector of scalars
TensorBC = Float[Tensor, "batch channel"]  # Only batch and channel
TensorBNC = Float[Tensor, "batch *optional_dims channel"]
TensorBTNC = Float[Tensor, "batch time *optional_dims channel"]
TensorBSC = Float[Tensor, "batch spatial *spatial channel"]
TensorBLC = Float[Tensor, "batch latent *latent channel"]
TensorBCL = Float[Tensor, "batch channel latent *latent"]

TensorBTSC = Float[Tensor, "batch time spatial *spatial channel"]  # Channels last
TensorBCTS = Float[Tensor, "batch channel time spatial *spatial"]  # Channels first
TensorBCS = Float[Tensor, "batch channel spatial *spatial"]  # No time dimension
TensorBSSC = Float[Tensor, "batch spatial *spatial channel"]  # No time dimension

TensorTSC = Float[Tensor, "time spatial *spatial channel"]  # No batch dimension
TensorSC = Float[Tensor, "spatial *spatial channel"]  # No batch dimension

TensorBCWH = Float[Tensor, "batch channel width height"]  # Specific spatial dims
TensorBTCHW = Float[Tensor, "batch time channel height width"]  # Specific spatial dims

TensorBTC = Float[Tensor, "batch time channel"]

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
