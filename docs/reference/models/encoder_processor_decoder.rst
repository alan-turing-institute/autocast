autocast.models.encoder\_processor\_decoder
=============================================

Decoded-state residual prediction
---------------------------------

Both deterministic and ensemble EPD models can decode residuals relative to the
last clean input frame, captured before input noise. ``model.residual_prediction``
always reconstructs state fields after decoding; it does not enable residuals
inside a processor or in cached latent space. Internal encoder latents are
supported; training requires a differentiable processor mapping and a loss on
decoded states. Enable it with:

.. code-block:: yaml

   model:
     train_in_latent_space: false
     residual_prediction: true
     residual_use_delta_stats: true

Constructor defaults are ``residual_prediction=False`` and
``residual_use_delta_stats=True``; the latter only affects enabled residual
prediction with state or independent delta statistics. Direct prediction and
checkpoint tensor structure are unchanged. Save the model configuration alongside
the weights, since the residual flag changes their prediction semantics.

All supplied state and delta means/scales describe raw fields in physical units.
They come from precomputed training-statistics files or explicit configuration;
the model never fits statistics during training or inference. For z-score scaling,
normalized-state increment mean/scale are derived as :math:`\mu_\Delta/\sigma_s` and
:math:`\sigma_\Delta/\sigma_s`; do not supply normalized-space statistics separately.

With delta statistics enabled, The Well's z-score normalizer converts the decoded
residual to a physical increment. Dividing by the state scale gives the increment
to add directly to the normalized baseline, preserving small changes without a
physical-unit round trip. For normalized baseline :math:`x` and residual
:math:`r_k`, this gives:

.. math::

   \hat{y}_k = x +
   \frac{\sigma_\Delta r_k + \mu_\Delta}{\sigma_s}.

Here :math:`\sigma_s` is the state scale and :math:`\mu_\Delta,\sigma_\Delta`
are the delta mean and scale. RMS normalization uses RMS scales with no mean
offset. Zero z-score decoded output implies a physical increment
of :math:`\mu_\Delta`, which need not be zero. With
``residual_use_delta_stats: false``, addition is directly in state-normalized
units, giving physical increment :math:`\sigma_s r_k` without a mean offset.
For delta scaling without state normalization, explicitly supply the existing
AutoSim ``stats.yml`` or native The Well ``stats.yaml``:

.. code-block:: yaml

   datamodule:
     use_normalization: false
   model:
     residual_prediction: true
     train_in_latent_space: false
     residual_delta_stats_path: ${datamodule.normalization_path}

This reconstructs :math:`\hat{Y}_k = X + \sigma_\Delta r_k + \mu_\Delta` in
physical units. The file's field order follows the dataset's channel selection;
no statistics are recomputed. It overrides the delta statistics only, leaving
state normalization unchanged. Native The Well files use the dataset's field
metadata, configured normalizer (z-score or RMS), and filesystem for remote reads,
matching its existing state-normalization path. Python callers can instead pass
``delta_norm``.
The path defaults to ``null``: without it, the state normalizer supplies delta
statistics when available, and unnormalized states retain raw addition
:math:`X+r_k`. Setting ``residual_use_delta_stats: false`` disables either source.
Explicit delta files and requested The Well state normalization require readable,
valid statistics; loading failures raise an error rather than use raw fields.

During fitting, the loss compares reconstructed states with state targets.
With z-score delta scaling, the decoder learns
:math:`r_k=(Y_k-X-\mu_\Delta)/\sigma_\Delta`; no separate residual-target transform
is needed.

Delta scaling uses the configured statistics as supplied. The shipped
``advection_diffusion_multichannel_64_64`` config has placeholder delta means/scales
of 0/1 and disables normalization by default. Before using those values for
residual delta scaling, supply measured statistics or point
``model.residual_delta_stats_path`` at the generated AutoSim file.

Every output lead, including four jointly predicted frames, is a cumulative
residual from the same last input frame, not a successive increment or cumsum.
Existing one-step delta statistics apply to all leads; they are not lead-specific.
Each ensemble member uses its own residual and corresponding clean baseline.

Losses, metrics, and rollouts receive complete reconstructed states. Each rollout
window supplies a fresh baseline: teacher forcing inserts ground-truth states,
while free running inserts reconstructed predictions. Existing prediction and
metric denormalization conventions are preserved.

Decoded states must match the input's channels-last layout, batch size, spatial
grid, and fields in the same order; only frame counts may differ. Shape mismatches
raise an error, but callers must ensure field identity. Residuals require full EPD
training/evaluation; latent training and processor-only/latent evaluation paths
are rejected. Latent and flow-matching objectives are not reinterpreted.

.. automodule:: autocast.models.encoder_processor_decoder
   :members:
   :undoc-members:
   :show-inheritance:
