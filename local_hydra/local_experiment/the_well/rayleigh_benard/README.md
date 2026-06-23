# Rayleigh-Benard run configs

This folder keeps the Rayleigh-Benard experiment configs that are useful as
reviewable presets. The ambient configs share the local
`datamodule/rayleigh_benard.yaml` preset so Rayleigh-Benard scalar handling has
one source of truth.

The historical May comparison branch also contained a large set of SLURM
submitters for effective-batch, timing, recovery-eval, and `start16` plotting
runs. Those scripts are intentionally not mirrored here because they encode
run-specific output paths and overlapping launcher variants. Add curated
workflow entrypoints separately if they become active maintained workflows.
