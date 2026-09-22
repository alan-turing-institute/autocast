# Main-checkpoint calibration comparisons

From the repository root, choose a **new** output root and generate the three
calibration/test pairings separately:

```sh
for split in calib-new__test-paper calib-paper-valid__test-paper calib-new__test-new; do
  uv run --frozen --no-sync python scripts/plot_calibration_comparisons.py \
    --split "$split" \
    --output-dir "outputs/2026-07-24_collated/2026-09-22_calibration_splits_committed/$split"
done
```

The script refuses to overwrite an existing directory. It reads saved CSVs;
it does not rerun inference or fit calibrators. Each split has three
subdirectories containing `paper_uq_reliability_by_lead_time.pdf` and its
matching PNG:

- `crps_calibration`: main CRPS checkpoint, EMOS, conformal prediction.
- `fm_calibration`: main latent FM checkpoint, EMOS, conformal prediction.
- `combined_calibration`: all six curves, with dashed CRPS and solid FM lines.

The paper currently uses the two separate `calib-new__test-paper`
comparisons. The other splits and combined versions are alternatives for
review. Earlier previews remain in `2026-09-22_calibration_plots/` and
`2026-09-22_calibration_splits_fixed_axes/`.

## Axes and appearance

The layout reuses the paper's Figure 4 renderer: coverage for windows
`[0:4)`, `[6:12)`, `[13:30)`, `[31:99)` and relative coverage error over lead
time at nominal levels 0.9, 0.5 and 0.1. Main CRPS is blue and main latent FM is
orange; EMOS is green and conformal prediction is purple.

Every right-hand panel uses the original Figure 4 limits,
`[-0.914115395769477, 0.914115395769477]`, with visible ticks at -0.5, 0 and 0.5.
These exact bounds were recovered by rendering the eight original evaluations
from `outputs/2026-05-15_collated/`; the reference PDF and input CSV hashes are
recorded in each new export. Lead-time axes also retain their original range.
Boundary triangles mark the largest excursion within each contiguous segment
outside the displayed range. Curve values are unchanged, and the provenance
records the numbers and extrema of off-scale observations for every curve.

## Data selection

The eight main run IDs are explicit in the script. `--split` selects a
subdirectory of each run's `eval_conformal/`:

| Split | Calibration trajectories | Test trajectories |
| --- | --- | --- |
| `calib-new__test-paper` | 100 new (102 for GS) | Paper test: 20 each for AD/CNS/GPE, 24 for GS |
| `calib-paper-valid__test-paper` | Paper validation: 20 each for AD/CNS/GPE, 24 for GS | Paper test: 20 each for AD/CNS/GPE, 24 for GS |
| `calib-new__test-new` | 100 new (102 for GS) | 50 disjoint new (48 for GS) |

The uncalibrated baseline is always the matching `raw` export alongside EMOS
and conformal. It uses the same forecast realisation as the calibrated
methods. For the paper test set it is **not numerically identical** to the
historical main-comparison CSVs, despite using the same named checkpoints,
ensemble size and test datasets. The script records that discrepancy without
assuming its cause. The new-test comparison uses a different test set, so no
repeated-evaluation comparison against the historical curves is reported.

Missing/invalid curves, incomplete lead-time data and disagreements between
window coverage and the corresponding frame averages cause an error before
any figures are written. `plotting_provenance.json` records run selections,
actual calibration/test counts, input checksums, baseline differences, source
hashes, the command and output checksums. `plotting_source/` preserves the
exact recipe, shared renderer and lockfile. `plotting_commit` records HEAD
only when those three source files exactly match its Git contents; otherwise
it remains null and the archived snapshots identify the executed code.
