Scale losses
============

``ScaleDecomposition`` returns named, physical-space scale bands that can each
be scored with ordinary grid-space CRPS. Unlike ``RadialFourierDecomposition``,
its components are fields, not Fourier coefficients. Both predictions and
targets undergo the same decomposition.

Inputs have channels-last layout ``(B, T, S1, ..., Sd, C)`` with an optional
final ensemble axis ``M``. Only the spatial axes are smoothed; batch, time,
channels and members remain separate. Direct operator calls take
``n_spatial_dims``; ``TransformedLoss`` infers it from the target layout.

Scale bands and boundaries
--------------------------

Supply smoothing operators from strongest to weakest. For two operators
``S1`` and ``S2``, the components are ``S1(x)``, ``S2(x) - S1(x)`` and the final
residual ``x - S2(x)``. Their sum reconstructs the input up to floating-point
roundoff. Each operator is applied to the original field, not to the preceding
smoothed result. Ordering is a user responsibility: it cannot be inferred for
arbitrary custom operators.

* ``GaussianSmoothing`` applies a separable Gaussian kernel. ``sigma`` is in
  grid cells; larger values give stronger smoothing. Its kernel radius is
  ``ceil(truncate * sigma)``, with ``truncate=4`` by default. The default
  ``boundary="reflect"`` requires every spatial axis to be longer than this
  radius. ``boundary="periodic"`` wraps the field, including when the radius
  exceeds the axis length. ``boundary="zero"`` pads with zeros, so constant
  fields are attenuated near the boundary.
* ``FourierLowPass`` keeps periodic Fourier modes whose radial frequency is
  less than or equal to ``cutoff``, then transforms back to physical space.
  Smaller cutoffs give stronger smoothing. The cutoff is in **cycles per grid
  cell**, in ``(0, 0.5]``. It is not an integer Fourier mode or a physical
  wavenumber. A cutoff of ``0.5`` is not an identity filter in multiple spatial
  dimensions: diagonal corner frequencies can exceed the per-axis Nyquist
  frequency. Half and bfloat16 fields are promoted to float32 before the FFT;
  single and double precision, including complex inputs, are otherwise
  preserved. Cutoff membership is independent of input precision.

These parameters describe grid-relative scales. Keeping them fixed while
changing resolution does not preserve physical smoothing lengths. All
operators preserve field shape and support differentiation. As with other
PyTorch modules, move the objective and its Gaussian kernel buffers to the
input device, for example with ``loss.to(device)``.

Composing objectives
--------------------

This example adds a scale-band CRPS term to full-field CRPS::

   from autocast.losses import (
       CRPSLoss,
       GaussianSmoothing,
       ScaleDecomposition,
       TransformedLoss,
       WeightedSumLoss,
   )

   scale_loss = TransformedLoss(
       ScaleDecomposition(
           [
               ("large", GaussianSmoothing(sigma=2, boundary="periodic")),
               ("medium", GaussianSmoothing(sigma=1, boundary="periodic")),
           ],
           residual_name="small",
       ),
       score=CRPSLoss(),
       component_weights={"large": 1, "medium": 1, "small": 0.5},
       requires_ambient_predictions=True,
   )
   loss = WeightedSumLoss([CRPSLoss(), scale_loss], weights=[1.0, 0.1])
   # predictions: (B, T, S1, ..., Sd, C, M)
   # targets:     (B, T, S1, ..., Sd, C)
   # objective = loss(predictions, targets)

``TransformedLoss`` takes a weighted **mean** of the component scores: here the
denominator is ``1 + 1 + 0.5``. ``WeightedSumLoss`` takes a weighted **sum** of
whole objectives, without dividing by the sum of its weights. These are
illustrative weights, not a guarantee of equal score or gradient contributions.
The input normalization is retained; physical-space fields are not necessarily
in physical SI units.

A Fourier-coefficient term can be included independently::

   from autocast.losses import CoefficientCRPSLoss, FourierDecomposition

   fourier_loss = TransformedLoss(
       FourierDecomposition(),
       score=CoefficientCRPSLoss(),
       requires_ambient_predictions=True,
   )
   loss = WeightedSumLoss(
       [CRPSLoss(), scale_loss, fourier_loss], weights=[1.0, 0.1, 0.1]
   )

For EPD, keep ``model.train_in_latent_space=false`` when using these examples.
Their ``requires_ambient_predictions=True`` flag propagates through
``WeightedSumLoss`` and makes setup reject latent-space training, which does
not supply this field layout. The flag does not itself decode or reshape a
tensor. Decoded residual prediction is supported: these losses score the
reconstructed states. Unmarked custom compositions still require the declared
channels-last field layout.

See :doc:`spectral` for the general composition interfaces and coefficient
scoring convention. Existing loss presets and defaults remain unchanged.

API
---

.. automodule:: autocast.losses.scale
   :members:
