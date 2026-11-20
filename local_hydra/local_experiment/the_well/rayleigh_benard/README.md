# Rayleigh-Benard run configs

This folder keeps the Rayleigh-Benard experiment configs that are useful as
reviewable presets. The ambient configs share the local
`datamodule/rayleigh_benard.yaml` preset so Rayleigh-Benard scalar handling has
one source of truth.

The maintained SLURM entrypoints for the active RB comparison matrix live under
`slurm_scripts/comparison/the_well/rayleigh_benard/`. They intentionally avoid
mirroring every historical submitter from
[#369](https://github.com/alan-turing-institute/autocast/pull/369); the scripts
there are grouped by the two active comparison questions instead of by
recovered run history.

The ambient FM configs are kept as optional presets for future ablations. They
complete the ambient/latent method matrix, but are not wired into the maintained
comparison submitters because ambient-space FM is expected to be substantially
more expensive than the selected 24h comparison runs.

## Spatiotemporal ambient CRPS

Use `crps_spatiotemporal_ambient` with the `epd` entrypoint for the LOLA pixel
backbone: width 2048, 16 blocks, 16 heads and `[1,16,16]` patches, giving 1,280
tokens for five 512×128 frames. It reuses the raw RB loader's field statistics
and log-scalar conditioning, with identity encoder/decoder and masked CRPS.
`train_in_latent_space: true` selects the existing processor-owned loss API;
the identity codec keeps that loss in normalized ambient fields.

On four GPUs, batch 8/GPU and eight members give 32 real windows and 256
predictions per update. Activation checkpointing is enabled; memory/throughput
still need measuring. EPD synchronizes training loss every batch, whereas the
latent wrapper uses sparse logging; include that overhead in timing.
When changing accumulation, preserve
`batch_size × devices × accumulate_grad_batches × n_members = 256` and set
`limit_train_batches = 64 × accumulate_grad_batches` to retain 64 updates/epoch.
The reference cosine is 4096×64 updates; choose any shorter horizon after timing
and retain it across resumes. A 48-hour budget is only a possibility, not
encoded in this preset. The existing submitters are unchanged.
