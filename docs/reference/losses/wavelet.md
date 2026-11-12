# Wavelet losses

Wavelet decompositions plug into the same named-component interfaces as
[Fourier losses](spectral.md). Inputs are channels-last fields
`(B, T, *S, C)` with an optional final ensemble axis. Only spatial axes are
transformed; channels, times and members are kept separate.

## Decompositions and transforms

`WaveletDecomposition(wavelet="haar", levels=1, boundary="reflect")` applies
a separable decimated real wavelet transform over one to three spatial axes.
Named bases supported by `ptwt` are accepted. Detail names are
`level_<n>/<orientation>` with `L`/`H` in spatial-axis order; the finest level
is 1. The final approximation is `level_<levels>/approximation`.
`boundary="periodic"` uses circular padding, not critically sampled
periodization. Longer filters and boundary handling affect coefficient sizes;
energy preservation is not universal across bases, padding and grid shapes.

`DTCWTDecomposition(levels=1, biort="near_sym_a", qshift="qshift_a")` uses
the 2D dual-tree complex transform from `pytorch-wavelets`, with symmetric
boundaries. It accepts real 2D fields only. Each level exposes six complex
`level_<n>/direction_<k>` components in backend orientation order. The optional
low-pass component is `level_<levels>/lowpass`. Its registered filters follow
module device/dtype moves and checkpoint state dictionaries; use `.double()`
when evaluating double-precision fields.

Both backends promote half/bfloat16 inputs to float32 and disable reduced-
precision autocast for their convolutions. The casts remain differentiable.

The component transforms are:

- `MagnitudeTransform`: coefficient modulus.
- `PowerTransform`: squared modulus.
- `RMSAmplitudeTransform`: spatial RMS, retaining extent-one spatial axes.

`CoefficientLpLoss(p=2)` scores deterministic or ensemble coefficients with
`abs(prediction - target)**p`, with mean/sum/none reductions. At exact
agreement it chooses a zero gradient, including for fractional powers whose
mathematical derivative is singular at zero.

## Example

Score the finest directional DTCWT bands with deterministic coefficient
squared error:

```python
import torch

from autocast.losses import (
    CoefficientLpLoss,
    ComponentBranch,
    ComponentObjective,
    DTCWTDecomposition,
    TransformedLoss,
)

predictions = torch.randn(2, 1, 32, 32, 3)  # (B, T, H, W, C)
targets = torch.randn(2, 1, 32, 32, 3)

finest = [f"level_1/direction_{k}" for k in range(6)]
loss = TransformedLoss(
    DTCWTDecomposition(levels=3),
    objective=ComponentObjective(
        [ComponentBranch(CoefficientLpLoss(p=2), components=finest)]
    ),
)
value = loss(predictions, targets)
```

The same loss, configured for training, can be written as YAML. Saved under
`src/autocast/configs/loss/`, for example as `my_dtcwt_loss.yaml`, it is
selected with `loss@model.loss_func=my_dtcwt_loss`:

```yaml
defaults:
  - _reset_epd_loss
  - _self_

_target_: autocast.losses.TransformedLoss
requires_ambient_predictions: true
decomposition:
  _target_: autocast.losses.DTCWTDecomposition
  levels: 3
objective:
  _target_: autocast.losses.ComponentObjective
  branches:
    - _target_: autocast.losses.ComponentBranch
      score:
        _target_: autocast.losses.CoefficientLpLoss
        p: 2
      components:
        - level_1/direction_0
        - level_1/direction_1
        - level_1/direction_2
        - level_1/direction_3
        - level_1/direction_4
        - level_1/direction_5
```

`_reset_epd_loss` clears any previously configured loss before this one is
composed. `requires_ambient_predictions` does not change the loss value; it
makes setup reject latent-space training, because the decomposition needs
decoded fields.

The notebook `notebooks/09_spectral_losses.ipynb` walks through these
decompositions on a small simulated ensemble. It reconstructs each scale in
physical space, demonstrates standalone coefficient losses and visualizes the
effect of smoothing on the decompositions.
