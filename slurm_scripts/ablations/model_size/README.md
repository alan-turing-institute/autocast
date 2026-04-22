# Model size ablation

Use `slurm_scripts/comparison/README.md` as the main entry point for the
baseline comparison design, submission order, and the ~80M processor matrix.
This folder records the CNS-only delta for the follow-up model-size sweep
around the ~80M baselines.

**Status:** ready — timing and 24h submit scripts cover the full scan.

## Goal

Characterise how prediction quality scales with processor size on CNS by
sweeping a 3-point aspect-preserving, heads-fixed scan around the ~80M
ambient baselines (one smaller point, baseline, one larger point). The
guiding rule is:

```
params ~= depth x width^2
```

applied at each scan point, with:

- `num_heads` fixed at `8` for both models,
- even depth chosen symmetrically around the baseline depth of `12`,
- width chosen as the nearest multiple of `8` to the baseline width/depth
  aspect ratio (CRPS `568/12 ~= 47.3`, FM `704/12 ~= 58.7`).

This keeps a single consistent scaling rule across all scan points, including
the committed 2x leg, so the ablation varies only model size.

### Why heads are fixed

Canonical ViT/DiT families often co-scale heads with width, but for an
ablation specifically about model size we hold heads fixed so attention
partitioning is not a second moving piece. The baselines already run at
non-standard head dimensions (CRPS `568/8 = 71`, FM `704/8 = 88`), so
holding heads fixed preserves the baseline attention geometry rather than
introducing a second architectural change at any one point. Resulting head
dimensions across the scan (`47-96` CRPS, `59-112` FM) are within the range
used by prior transformer families (e.g. ViT-H uses `80`, DiT-XL uses `72`).

## Scan points

All runs are CNS-only for the first pass and inherit from the comparison-study
ambient baselines. Measured processor param counts come from instantiated
CNS ambient backbones.

| variant | CRPS shape | CRPS params | CRPS scale | FM shape | FM params | FM scale |
|---|---|---:|---:|---|---:|---:|
| `_0p4x` | `376 / 8 / 8` | 31.6M | 0.391x | `472 / 8 / 8` | 25.6M | 0.319x |
| baseline | `568 / 12 / 8` | 80.8M | 1.000x | `704 / 12 / 8` | 80.0M | 1.000x |
| `_2x` | `768 / 16 / 8` | 169.3M | 2.094x | `896 / 16 / 8` | 168.3M | 2.103x |

Adjacent-point ratios are `~2.6x` and `~2.1x` (CRPS) and `~3.1x` and `~2.1x`
(FM), spanning ~5x range end to end. This is wider than a pure `0.5x / 1x /
2x` scan and keeps the smaller point at more-standard transformer dimensions
(head_dim `47` for CRPS, `59` for FM) rather than pushing into shallow /
narrow territory.

## Exact scaling table

Per-variant width/depth factors vs baseline, with naive `depth x width^2`
heuristic against the measured parameter ratio:

| variant | model | width scale | depth scale | heads scale | head_dim base -> variant | depth x width^2 heuristic | measured scale |
|---|---|---:|---:|---:|---|---:|---:|
| `_0p4x` | CRPS | 0.662x | 0.667x | 1.000x | `71 -> 47` | 0.292x | 0.391x |
| `_0p4x` | FM | 0.670x | 0.667x | 1.000x | `88 -> 59` | 0.300x | 0.319x |
| `_2x`   | CRPS | 1.352x | 1.333x | 1.000x | `71 -> 96` | 2.438x | 2.094x |
| `_2x`   | FM | 1.273x | 1.333x | 1.000x | `88 -> 112` | 2.160x | 2.103x |

Aspect ratios (width/depth) stay close to baseline across all points:

| point | CRPS aspect | drift vs base 47.3 | FM aspect | drift vs base 58.7 |
|---|---:|---:|---:|---:|
| `_0p4x` | 47.0 | -0.6% | 59.0 | +0.5% |
| baseline | 47.3 | 0.0% | 58.7 | 0.0% |
| `_2x` | 48.0 | +1.5% | 56.0 | -4.6% |

The naive `depth x width^2` estimate underestimates at small sizes and
overshoots at large sizes because the full processors include patch, embed,
output, and modulation terms that do not scale as a pure transformer-core
law. Exact counts must be measured from instantiated models, which is what
the scan-points table uses.

## Variant details

### CRPS ambient

- Baseline: `hidden_dim=568`, `n_layers=12`, `num_heads=8`,
  `n_noise_channels=1024`, `n_members=8`, `batch_size=32`.
- All scan variants set `n_members=16`, `batch_size=16` to keep effective
  per-GPU minibatch at `16 x 16 = 256`, matching the baseline
  `32 x 8 = 256`. `num_heads=8` and `n_noise_channels=1024` are held fixed.
- Per-variant width/depth:
  - `_0p4x`: `hidden_dim=376`, `n_layers=8`, head_dim `47`.
  - `_2x`:   `hidden_dim=768`, `n_layers=16`, head_dim `96`.

### FM ambient

- Baseline: `hid_channels=704`, `hid_blocks=12`, `attention_heads=8`.
- Per-variant:
  - `_0p4x`: `hid_channels=472`, `hid_blocks=8`, head_dim `59`.
  - `_2x`:   `hid_channels=896`, `hid_blocks=16`, head_dim `112`.

## Files

| file | purpose |
|---|---|
| `local_hydra/local_experiment/ablations/model_size/conditioned_navier_stokes/crps_vit_azula_0p4x.yaml` | CRPS ambient ~0.39x preset |
| `local_hydra/local_experiment/ablations/model_size/conditioned_navier_stokes/crps_vit_azula_2x.yaml` | CRPS ambient ~2.09x preset |
| `local_hydra/local_experiment/ablations/model_size/conditioned_navier_stokes/fm_vit_0p4x.yaml` | FM ambient ~0.32x preset |
| `local_hydra/local_experiment/ablations/model_size/conditioned_navier_stokes/fm_vit_2x.yaml` | FM ambient ~2.10x preset |
| `slurm_scripts/ablations/model_size/submit_model_size_timing.sh` | 5-epoch timing runs for all variants |
| `slurm_scripts/ablations/model_size/submit_model_size_large.sh` | 24h production runs after filling cosine epochs |
| `slurm_scripts/ablations/model_size/eval/submit_eval_crps_ambient.sh` | preliminary ambient eval for the current CRPS `2x` run, with conservative `eval.batch_size=4` and explicit `eval.n_members=10` to match the comparison-study eval regime |
| `slurm_scripts/ablations/model_size/eval/submit_eval_fm_ambient.sh` | preliminary ambient eval for the current FM `2x` run, with conservative `eval.batch_size=2` for the larger 896/16 backbone |

## Naming

The submit scripts intentionally follow the ensemble-size pattern: they start
from the comparison-study baseline presets and apply the model-size changes as
CLI overrides. That keeps auto-generated run dirs descriptive:

- CRPS run dirs resolve like `crps_cns64_vit_azula_large_<width>_<git>_<uuid>`
  (e.g. `..._376_...`, `..._768_...`).
- FM run dirs resolve like `diff_cns64_flow_matching_vit_<width>_<git>_<uuid>`
  (e.g. `..._472_...`, `..._896_...`).
- The ablation knob is surfaced in `logging.wandb.name`, which also feeds the
  SLURM job name (mirroring `ensemble_size`):
  - `model_size_crps_0p4x`, `model_size_crps_2x`
  - `model_size_fm_0p4x`,   `model_size_fm_2x`

Labels use the `0p4x` / `2x` convention (dot-free to avoid Hydra
override parsing issues) and reflect the honest measured scale rather than
a round number like `160M` or `40M`. Actual param counts are recorded in
the scan-points table above.

## Eval

Model-size eval lives under `slurm_scripts/ablations/model_size/eval/` while
the sweep remains CNS-only and only the `2x` production runs are in hand.
The current preliminary submitters target:

- `outputs/2026-04-21/model_size/crps_cns64_vit_azula_large_768_...`
- `outputs/2026-04-21/model_size/diff_cns64_flow_matching_vit_896_...`

Add the matching `376` / `472` run dirs to those scripts once the `0p4x`
production runs exist.

## Workflow

1. Run `submit_model_size_timing.sh` to submit 5-epoch timing jobs for all
   four variants.
2. Extract per-variant `cosine_epochs` from each `timing.ckpt` via
   `uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24`.
3. Paste the values into `COSINE_EPOCHS_BY_VARIANT` in
   `submit_model_size_large.sh`, keyed by
   `conditioned_navier_stokes:<variant>`.
4. Run `submit_model_size_large.sh` in dry-run mode first, then for real.
   The script iterates `RUN_DRY_STATES=("true" "false")` so a single
   invocation emits both a dry-run and a real submission per variant.
