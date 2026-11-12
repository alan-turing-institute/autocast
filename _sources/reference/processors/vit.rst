autocast.processors.vit
=======================================

.. automodule:: autocast.processors.vit
   :members:
   :undoc-members:
   :show-inheritance:

Azula ViT: global and spatial AdaLN noise
---------------------------------------

``AzulaViTProcessor`` supports two noise layouts, selected by ``noise_mode``:

* ``global`` (default): one Gaussian vector per example/ensemble member,
  shared across all patch tokens.
* ``spatial``: a vector per patch token, independent Gaussian by default. Each block's
  existing AdaLN weights are shared across tokens, but its scale, shift and
  gate can vary spatially. The noise draw is reused across blocks and redrawn
  on every ``map()`` call, including each autoregressive forecast step.

When one call predicts several output frames jointly, those frames share the
same draw. A fresh draw is sampled when the next forecast window is predicted.

For example, override an Azula processor with:

.. code-block:: yaml

   noise_mode: spatial
   n_noise_input_channels: 16
   n_noise_channels: 256
   patch_size: 1

At 32x32 this samples 16 independent values at each of 1024 tokens, rather
than 16 values for the whole field. The input projection maps each local
16-vector to 256 features using shared weights. With the same channel
dimensions, global and spatial modes have identical trainable parameter
counts and compatible state dictionaries, although spatial modulation costs
more computation and activating it changes the model's stochastic behaviour.

Unlike input noise concatenation, spatial AdaLN does not append input channels:
it modulates hidden features in every transformer block. A token corresponds
to one processor input cell only when ``patch_size: 1``; larger patches draw
one vector per patch, not per cell.

``map(x)`` samples noise automatically. For controlled draws, pass
``forward(x, x_noise)`` a tensor of shape ``(B, D)`` for global mode or
``(B, N, D)`` for spatial mode, where ``D = n_noise_input_channels``
(defaulting to ``n_noise_channels``), and
``N = (H / patch_height) * (W / patch_width)``. Tokens are flattened with
width varying fastest. Spatial dimensions must be divisible by the patch
dimensions. Optional global conditioning is broadcast to all tokens.

Spatial mode is opt-in and requires positive noise dimensions. Global noise
remains the default.

Structured noise sources
~~~~~~~~~~~~~~~~~~~~~~~~

The optional ``noise_source`` uses the shared ``SourceDistribution.sample_like``
interface to replace sampling without changing the modulation layers.
``SpatialFourierSource`` samples smooth periodic Gaussian fields from integer
Fourier wavevectors. For example:

.. code-block:: yaml

   noise_mode: spatial
   n_noise_input_channels: 4
   n_noise_channels: 256
   noise_source:
     _target_: autocast.nn.noise.SpatialFourierSource
     wavevectors: [[1, 0], [0, 1]]

The source receives ``(B, 1, H_tokens, W_tokens, D)`` and its sample is flattened
in width-fastest token order. Wavevectors count cycles across this token grid;
they must be nonzero, distinct up to sign and strictly below its Nyquist limit.
Every mode has independent Gaussian sine and cosine coefficients, with unit
variance at each site in expectation. Individual draws are not normalized.
There are ``2 * number_of_wavevectors * D`` random coefficients per member,
so changing the mode set also changes the latent dimension.

To compose the example using Hydra, add
``+source_distribution@model.processor.noise_source=spatial_fourier`` together with
``model.processor.noise_mode=spatial`` to an Azula processor preset.
Omitting the source retains the existing IID sampler and checkpoint layout.
Passing explicit ``x_noise`` to ``forward`` bypasses the source entirely.
This local modulation path does not change the older, globally conditioned
``ConditionalLayerNorm`` class.

The same source can supply input concatenation:

.. code-block:: yaml

   input_noise_injector:
     _target_: autocast.nn.noise.noise_injector.ConcatenatedNoiseInjector
     n_channels: 4
     std: 1.0
     noise_source:
       _target_: autocast.nn.noise.SpatialFourierSource
       wavevectors: [[1, 0], [0, 1]]

In the encoder-processor-decoder model, concatenation happens before encoding,
on the input field grid ``(B, T_input, H, W, C)``. It appends noise channels
without modifying the physical input channels or the clean persistence
baseline. This differs from AdaLN sampling on the patch-token grid, especially
with a compressing encoder or patches larger than one cell. The existing
``input_noise_injector@model.input_noise_injector=concat`` preset accepts the source override
``+source_distribution@model.input_noise_injector.noise_source=spatial_fourier``.

``SourceDistribution`` is exported from ``autocast.nn.noise`` and
``autocast.nn.noise.source``. ``FlowSource`` remains an exact alias for existing
imports and subclasses. The ``source_distribution`` config group reuses the
existing ``flow_source`` presets; old group overrides and FM defaults remain
supported.
Concrete sampler paths, constructor arguments and checkpoint keys are unchanged.
The interface requires sampling only, not density evaluation. The consumer
decides whether samples are initial flow states or conditioning noise.

Temporal semantics are explicit: ``SpatialFourierSource`` draws independent
noise for every frame and every call. Existing OU/separable sources can
correlate an explicit time axis, but Azula's conditioning source sees only
one effective time slot. With concatenation they correlate input-frame slots,
not successive forecasts. No source state is carried across rollout calls;
temporally persistent autoregressive noise requires a separate rollout API.
