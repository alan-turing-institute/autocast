# Local experiment presets (repo-level)

Use this folder for local/private Hydra experiment presets that should not be committed.

This folder now includes a curated subset of presets converted from the
`tests-scripts/latent-vit/slurm_scripts` branch.

Current presets:

- `ae_quick_local.yaml`
- `epd_crps_fno.yaml`
- `epd_crps_vit_large.yaml`
- `epd_crps_vit_latent.yaml`
- `epd_diffusion_dm_256.yaml`
- `epd_diffusion_fm_256.yaml`

Notes on compatibility mapping to this repo:

- Legacy `vit_latent` is mapped to `processor@model.processor=vit_large`.
- Legacy diffusion/flow-matching backbone variants `vit_256` / `vit_512` are mapped
  to existing `diffusion_vit` / `flow_matching_vit` plus
  `model.processor.backbone.hid_channels` overrides.

Create additional files like `local_hydra/local_experiment/my_private_run.yaml` with:

```yaml
# @package _global_
defaults:
  - _self_

experiment_name: my_private_run
trainer:
  max_epochs: 5
```

Run with:

```bash
uv run train_encoder_processor_decoder local_experiment=my_private_run
```

Because `hydra.searchpath` includes `${hydra:runtime.cwd}/local_hydra`, this repo-level folder is discoverable automatically when running from the repository root.
