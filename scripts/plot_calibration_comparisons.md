# Main-checkpoint calibration comparisons

From the repository root, choose a **new** output root and generate the three
calibration/test pairings separately:

```sh
for split in calib-new__test-paper calib-paper-valid__test-paper calib-new__test-new; do
  uv run --frozen --no-sync python scripts/plot_calibration_comparisons.py \
    --split "$split" \
    --output-dir "outputs/2026-07-24_collated/2026-09-23_coverage_difference/calibration_with_emos/$split"
done
```

The script refuses to overwrite an existing directory. It reads saved CSVs;
it does not rerun inference or fit calibrators. Each split has three
subdirectories containing `paper_uq_reliability_by_lead_time.pdf` and its
matching PNG:

- `crps_calibration`: main CRPS checkpoint, EMOS, conformal prediction.
- `fm_calibration`: main latent FM checkpoint, EMOS, conformal prediction.
- `combined_calibration`: all six curves, with dashed CRPS and solid FM lines.

The paper uses the combined original/CP comparison below with
`calib-new__test-paper`. The separate comparisons including EMOS and the
other splits remain available for review. Earlier previews remain in
`2026-09-22_calibration_plots/` and
`2026-09-22_calibration_splits_fixed_axes/`.

## Combined original and CP comparison

To generate just the four original/conformal prediction (CP) curves in one
figure, without EMOS:

```sh
uv run --frozen --no-sync python scripts/plot_calibration_comparisons.py \
  --cp-only --split calib-new__test-paper \
  --output-dir outputs/2026-07-24_collated/2026-09-23_coverage_difference/calibration/calib-new__test-paper
```

This writes one PDF/PNG pair in `combined_cp_calibration/`, plus the usual
provenance and source snapshots. CRPS remains dashed and FM remains solid
throughout. The original curves retain blue (CRPS) and orange (FM); CRPS + CP
uses a darker blue and FM + CP a darker orange. CP shades use the shared
renderer's existing 22% mix with black, keeping each model in one colour
family. The earlier green/purple preview and its source snapshot remain in
`2026-09-23_calibration_cp_only/`; the family-colour preview remains in
`2026-09-23_calibration_cp_family_colours/`. The original curves still use the
matching `raw` forecasts from the selected calibration export.
EMOS files are neither read nor included in this variant's input inventory.
The existing three-comparison recipe is unchanged.

## Axes and appearance

The layout reuses the paper's Figure 4 renderer: coverage for windows
`[0:4)`, `[6:12)`, `[13:30)`, `[31:99)` and signed coverage error
(observed minus nominal) over lead time at nominal levels 0.9, 0.5 and 0.1. Main CRPS is blue and main latent FM is
orange. In the comparisons including EMOS, EMOS is green and conformal
prediction is purple; the CP-only variant uses the darker model colours above.

Every right-hand panel uses the same symmetric limits as the updated Figure 4,
computed from observed-minus-nominal errors across its eight historical
baseline evaluations, with 10% padding. The nominal level is not used as a
divisor. This retains the source forecasts while expressing both figures in
the new units. The reference PDF and input CSV hashes are recorded in each
new export; the historical PDF itself uses the old relative units.
Lead-time axes retain their original range.
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
