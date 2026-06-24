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
