# Composing grid-space and spectral losses

`ensemble_fourier_crps` adds coefficient-wise complex CRPS to grid-space
CRPS. It scores complex coefficients, not their power or magnitude alone, so
coefficient phase differences contribute to the loss.

## Using the preset

Select the loss in an ensemble EPD configuration. For example, to adapt the
existing ensemble ViT experiment to your data:

```bash
uv run autocast epd \
    experiment=epd_crps_vit_large_ps4_64 \
    datamodule.data_path=/path/to/data \
    loss@model.loss_func=ensemble_fourier_crps \
    model.train_in_latent_space=false \
    model.n_members=8
```

Keep the data, encoder, decoder and processor settings appropriate to your
dataset. The loss preset overrides an experiment's loss selection but does not
otherwise change its architecture or training settings. Without this selection,
existing loss defaults are unchanged.

This EPD preset clears the previous `model.loss_func` mapping before composing
its replacement, so old constructor arguments are not inherited. Keep the
`@model.loss_func` package shown above; explicit command-line overrides of the
new loss's weights and scores are applied afterwards as usual.

The preset requires decoded fields and at least two members. EPD setup rejects
it when `model.train_in_latent_space=true`; the latent training path does not
provide the required channels-last field layout. This is distinct from using a
Fourier transform on an explicitly arranged latent grid in custom code.
Decoded residual prediction (`model.residual_prediction=true`) is supported:
the loss scores reconstructed states, not raw predicted increments.

The equivalent Python objective is:

```python
from autocast.losses import (
    AlphaFairCRPSLoss,
    CoefficientCRPSLoss,
    FourierDecomposition,
    TransformedLoss,
    WeightedSumLoss,
)

loss = WeightedSumLoss(
    losses=[
        AlphaFairCRPSLoss(alpha=0.95),
        TransformedLoss(
            FourierDecomposition(representation="complex", norm="ortho"),
            score=CoefficientCRPSLoss(alpha=0.95),
            requires_ambient_predictions=True,
        ),
    ],
    weights=[1.0, 0.1],
)
# predictions: (B, T, S1, ..., Sd, C, M)
# targets:     (B, T, S1, ..., Sd, C)
# objective = loss(predictions, targets)
```

## Exact scoring convention

For one real or complex coefficient, with members $z_1,\ldots,z_M$ and target
$y$, `CoefficientCRPSLoss` computes

$$
S_\alpha = \frac{1}{M}\sum_i |z_i-y|
 - \left(\frac{1-\alpha}{M^2}+\frac{\alpha}{M(M-1)}\right)
   \sum_{i<j}|z_i-z_j|.
$$

Here $|\cdot|$ is complex modulus (Euclidean distance in the real-imaginary
plane). This is the coefficient-wise complex scoring convention used for
spectral CRPS in [BriS, Section 2.2](https://arxiv.org/html/2511.23043v2#S2.SS2).
It is not independent scoring of the real and imaginary parts. `alpha=0`
(the class default) gives the plug-in score, `alpha=1` gives the fair score,
and intermediate values interpolate. A single member is supported only at
`alpha=0`, returning absolute coefficient error. The preset uses `alpha=0.95`
for both terms, corresponding to a fairness adjustment of $0.05/M$.

`FourierDecomposition` transforms spatial axes only, using the full FFT with
orthonormal normalization in this preset, and returns a `"fourier"` component.
It leaves batch, time, channel and member axes untouched. Each frequency and
channel is scored independently before reduction. `reduction="none"` retains
the target shape; `"mean"` (default) and `"sum"` reduce all its axes. The preset
therefore computes **mean grid-space score + 0.1 x mean Fourier score**, without
dividing by 1.1. Full-FFT frequency multiplicities matter; replacing it with an
unweighted real FFT mean changes this objective.

The loss uses the units of its inputs, including any data normalization. It
does not denormalize them. “Grid-space” is not a promise of physical SI units.
Orthonormal FFT does not equalize CRPS magnitudes or gradients, and the weight
0.1 is not a guaranteed 10% gradient contribution. Independent coefficient
scores do not determine all cross-frequency dependencies or guarantee physical
balance.

The score uses direct modulus, without smoothing, and PyTorch's zero
subgradient at exact coincidence. The Fourier boundary promotes float16 and
bfloat16 fields to float32 on the same device; double precision is preserved.
Direct coefficient scoring also uses the common input dtype with at least
float32 precision (complex64 for complex inputs), before computing distances,
to avoid low-precision subtraction or pairwise-sum overflow. The score has the
corresponding real dtype; gradients retain each input's dtype.
FFT support still depends on the execution device. Pairwise scoring has
quadratic cost in ensemble size; it evaluates only unordered member pairs.

## Radial bands

`RadialFourierDecomposition` selects disjoint bands of complex Fourier
coefficients, named `band_0`, `band_1`, and so on. Each component retains the
full grid layout with zeros outside its band. It does **not** average phases
or replace coefficients with a radial power spectrum. Band membership depends
on grid shape and edges, not on prediction or target precision.

Edges are radial frequencies in **cycles per grid cell**, not integer Fourier
mode numbers or physical wavenumbers. The first band begins at zero; intervals
include their lower edge and exclude their upper edge, except that the final
upper edge is included. In two dimensions the corners reach approximately
0.7071 cycles/cell, beyond the per-axis Nyquist of 0.5. Use a final edge of at
least `sqrt(n_spatial_dims) / 2` to cover every mode. A shorter cutoff leaves
higher frequencies unscored by this loss. Holding edges fixed across different
resolutions preserves grid-relative, not physical, scales.

For example, this two-dimensional objective emphasizes the highest band:

```python
from autocast.losses import (
    AlphaFairCRPSLoss,
    CoefficientCRPSLoss,
    RadialFourierDecomposition,
    TransformedLoss,
    WeightedSumLoss,
)

radial_loss = TransformedLoss(
    RadialFourierDecomposition(edges=[0.1, 0.25, 0.71]),
    score=CoefficientCRPSLoss(alpha=0.95),
    component_weights={"band_0": 1.0, "band_1": 1.0, "band_2": 2.0},
    requires_ambient_predictions=True,
)
loss = WeightedSumLoss([AlphaFairCRPSLoss(alpha=0.95), radial_loss], [1.0, 0.1])
```

The existing `ensemble_fourier_crps` preset remains full-spectrum and unchanged.
A radial term can also be combined with a full Fourier term through
`WeightedSumLoss`. Memory grows with the number of full-sized band components.

**Reduction matters:** with its default reduction, coefficient CRPS averages
over the full grid, including zeroed positions. `TransformedLoss` then takes
a weighted mean across components.
Consequently, an equally weighted, complete partition into `J` bands with
`band_mean=False` gives the full Fourier score divided by `J`, not a new
cross-frequency constraint. Weights or cutoffs change the emphasis. Empty bands
score zero but still count in the component-weight denominator unless excluded
or given zero weight.

The optional `band_mean=True` divides each band's coefficients by the square
root of its mode count, so its summed squared magnitude becomes mean band power.
For coefficient CRPS it instead scales the band score by `1 / sqrt(count)`;
it does **not** compute mean CRPS over just the occupied modes or automatically
equalize bands. It is disabled by default.

## Composition and extension contracts

- `WeightedSumLoss` sums whole-loss outputs with non-negative weights.
- `FieldDecomposition` maps a field to named components.
  `IdentityDecomposition` returns `"field"`; `SequentialDecomposition`
  composes stages with slash-separated component names.
- `ComponentBranch` selects components, applies shared `ComponentTransform`s
  independently to prediction and target, then forms their weighted **mean**.
- `ComponentObjective` takes a weighted **sum** of `ScoringBranch` outputs.
- `TransformedLoss` combines a decomposition with either a `score=` convenience
  argument or an explicit `objective=`, not both. An explicit objective controls
  its own component aggregation.

All aggregated score outputs must have the same shape; scalar/array broadcasting
is rejected. Transforms and branches receive `n_spatial_dims` and must preserve
the declared field layout. The `requires_ambient_predictions` flag is optional
on `TransformedLoss`; `WeightedSumLoss` propagates it from any configured term,
including nested and zero-weight terms, which are still evaluated.

An unreduced Fourier score is indexed by frequency, not physical position.
Do not apply a physical-cell mask to those bins as though they were spatial
errors. The supplied EPD preset returns a scalar and avoids that ambiguity.

```{eval-rst}
.. automodule:: autocast.losses.composite
   :members:

.. automodule:: autocast.losses.decomposition
   :members:

.. automodule:: autocast.losses.spectral
   :members:
```
