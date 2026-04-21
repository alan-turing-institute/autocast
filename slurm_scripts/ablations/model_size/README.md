# Model size ablation

Use `slurm_scripts/comparison/README.md` as the main entry point for the
baseline comparison design, submission order, and the ~80M processor matrix.
This folder only records the CNS-only delta for the follow-up ~2x sweep.

**Status:** ready — timing and 24h submit scripts are in place.

## Goal

Double processor size from the ~80M ambient baselines to a clean ~2x larger
model while scaling width and depth jointly, not by width alone. The guiding
rule is:

```
params ~= depth x width^2
```

Canonical ViT/DiT families often co-scale heads with width, but for this
ablation we keep heads fixed at `8` in both models so width and depth are the
only scaling knobs.

There is no architectural restriction forcing odd block counts here. Even
depth works fine in this codebase; the current sweep uses `16` blocks/layers
for both variants.

## Final 2x variants

All runs are CNS-only for the first pass and inherit from the comparison-study
ambient baselines. The committed choice is the even-depth `16`-block pair:

| variant | baseline config | baseline shape | final shape | base params | final params | param scale |
|---|---|---|---|---:|---:|---:|
| CRPS ambient | `local_hydra/local_experiment/epd/conditioned_navier_stokes/crps_vit_azula_large.yaml` | `568 / 12 / 8` | `768 / 16 / 8` | 80,831,448 | 169,264,768 | 2.094046x |
| FM ambient | `local_hydra/local_experiment/epd/conditioned_navier_stokes/fm_vit_large.yaml` | `704 / 12 / 8` | `896 / 16 / 8` | 80,305,408 | 168,643,456 | 2.100026x |

Measured counts above come from instantiated CNS ambient backbones and track
the ~80M matrix in `slurm_scripts/comparison/README.md` closely.

## Exact scaling table

| variant | width scale | depth scale | heads scale | head_dim base -> final | depth x width^2 heuristic | measured param scale |
|---|---:|---:|---:|---|---:|---:|
| CRPS ambient | 1.352113x | 1.333333x | 1.000000x | `71 -> 96` | 2.437612x | 2.094046x |
| FM ambient | 1.272727x | 1.333333x | 1.000000x | `88 -> 112` | 2.159780x | 2.100026x |

The principle is being followed in the intended sense:

- depth and width both increase together;
- heads stay fixed, so attention geometry is not treated as a scaling knob;
- the measured parameter ratio lands very close to `2x` for both models.

The naive `depth x width^2` estimate overshoots somewhat, especially for CRPS,
because the full processors also include patch/embed/output/modulation terms
that do not scale as a pure transformer-core law. In practice, exact counts
need to be measured from instantiated models, which is what this table uses.

## Variant details

### CRPS ambient

- Baseline: `hidden_dim=568`, `n_layers=12`, `num_heads=8`,
  `n_noise_channels=1024`, `n_members=8`, `batch_size=32`.
- Final variant: `hidden_dim=768`, `n_layers=16`, `num_heads=8`,
  `n_noise_channels=1024`, `n_members=16`, `batch_size=16`.
- `head_dim` shifts from `568 / 8 = 71` to `768 / 8 = 96`.
- `batch_size` is reduced to 16 so CRPS effective per-GPU batch stays at
  `16 x 16 = 256`, matching the baseline `32 x 8 = 256`.

### FM ambient

- Baseline: `hid_channels=704`, `hid_blocks=12`, `attention_heads=8`.
- Final variant: `hid_channels=896`, `hid_blocks=16`, `attention_heads=8`.
- `head_dim` shifts from `704 / 8 = 88` to `896 / 8 = 112`.

## Near-160 alternatives

If we optimize strictly for being numerically closer to `160M` instead of
keeping depth even, the nearby `15`-block options are:

| variant | near-160 shape | params | param scale |
|---|---|---:|---:|
| CRPS ambient | `768 / 15 / 8` | 158,769,488 | 1.964204x |
| FM ambient | `896 / 15 / 8` | 158,245,840 | 1.970550x |

Those are perfectly valid, but the committed sweep prefers the cleaner shared
`12 -> 16` depth step.

## Files

| file | purpose |
|---|---|
| `local_hydra/local_experiment/ablations/model_size/conditioned_navier_stokes/crps_vit_azula_160m.yaml` | CRPS ambient ~2x preset |
| `local_hydra/local_experiment/ablations/model_size/conditioned_navier_stokes/fm_vit_160m.yaml` | FM ambient ~2x preset |
| `slurm_scripts/ablations/model_size/submit_model_size_timing.sh` | 5-epoch timing runs for both variants |
| `slurm_scripts/ablations/model_size/submit_model_size_large.sh` | 24h production runs after filling cosine epochs |

## Naming

The submit scripts intentionally follow the ensemble-size pattern: they start
from the comparison-study baseline presets and apply the model-size changes as
CLI overrides. That keeps auto-generated run dirs descriptive:

- CRPS large-run dirs resolve like `crps_cns64_vit_azula_large_768_<git>_<uuid>`.
- FM large-run dirs resolve like `diff_cns64_flow_matching_vit_896_<git>_<uuid>`.
- The ablation knob itself is surfaced in `logging.wandb.name`
  (`model_size_crps_160m` / `model_size_fm_160m`), which also feeds the SLURM
  job name, mirroring `ensemble_size`.

## Workflow

1. Run `submit_model_size_timing.sh`.
2. Extract per-variant `cosine_epochs` from each `timing.ckpt` via
   `uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24`.
3. Paste the values into `COSINE_EPOCHS_BY_VARIANT` in
   `submit_model_size_large.sh`.
4. Run `submit_model_size_large.sh` in dry-run mode first, then for real.
