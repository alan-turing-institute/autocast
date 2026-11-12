autocast.models.encoder\_processor\_decoder
=============================================

Decoded-state residual prediction
---------------------------------

Residual prediction is disabled by default and adds decoded increments to the
last clean input frame. Every lead uses that same baseline, with independent
ensemble members. Enable decoded-state supervision with:

.. code-block:: yaml

   model:
     train_in_latent_space: false
     residual_prediction: true

Internal encoder latents are supported, but latent-space training, cached-latent
and processor-only evaluation are rejected. Input and output fields must have
the same spatial grid, channels and field order; only frame counts may differ.

All state and delta statistics describe physical units and come from precomputed
training files or explicit configuration; the model never fits them on the fly.
For physical baseline :math:`X`, decoded output :math:`r_k` and z-score scales:

.. math::

   \hat{Y}_k = X + \begin{cases}
     r_k & \text{raw states, unscaled increments},\\
     \sigma_s r_k & \text{normalized states, unscaled increments},\\
     \mu_\Delta + \sigma_\Delta r_k & \text{delta scaling, either state mode}.
   \end{cases}

For normalized states, the model adds :math:`(\mu_\Delta+\sigma_\Delta r_k)/\sigma_s`
directly to the normalized baseline, preserving small updates. Normalized-space
delta mean/scale are derived as :math:`\mu_\Delta/\sigma_s,\sigma_\Delta/\sigma_s`;
do not supply them separately. RMS uses its state/delta scales without a mean
offset. With z-score scaling, zero decoded output still adds :math:`\mu_\Delta`.

``residual_use_delta_stats`` defaults to ``true`` and reuses the state normalizer's
delta statistics. For delta-only scaling, set ``datamodule.use_normalization=false``
and ``model.residual_delta_stats_path=${datamodule.normalization_path}``.
AutoSim ``stats.yml`` follows selected channels; native The Well ``stats.yaml``
uses dataset metadata, its configured z-score/RMS normalizer and filesystem.
Python callers can pass ``delta_norm``. The path defaults to ``null``, preserving
raw addition without a state normalizer; ``residual_use_delta_stats=false``
disables either delta source. Requested stats loading raises on missing/invalid
files. The shipped ``advection_diffusion_multichannel_64_64`` config contains
placeholder delta statistics (0/1); supply measured statistics to use delta scaling.

Losses, metrics and rollouts consume reconstructed states, so no residual-target
transform is needed. Rollout baselines come from truth under teacher forcing and
predictions when free running. One-step delta statistics apply to every lead;
residuals are not summed. Save the model config and stats with trained weights.
Checkpoint tensor structure and the default direct forward path are unchanged.

.. automodule:: autocast.models.encoder_processor_decoder
   :members:
   :undoc-members:
   :show-inheritance:
