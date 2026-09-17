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
* ``spatial``: independent Gaussian vectors per patch token. Each block's
  existing AdaLN weights are shared across tokens, but its scale, shift and
  gate can vary spatially. The noise draw is reused across blocks and redrawn
  on every ``map()`` call, including each autoregressive forecast step.

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
