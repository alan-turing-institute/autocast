# EMA ablation (eval-only)

Eval-only ablation: evaluate an already-trained run using the EMA
shadow weights vs the online weights, to measure the EMA contribution.

**Status:** stub — no scripts yet.

## Source runs

Any run trained with `EMACallback` (all main comparison runs under
`outputs/2026-04-18/`). Both the live weights and the EMA shadow are
saved inside the same `.ckpt` — the knob is which to load at eval.

## Knob

Whatever the eval-side flag is for EMA loading. Candidates (need to
verify against the eval script):

- `eval.use_ema=true|false`
- `model.load_ema=true|false`
- callback-state override

Check `src/autocast/callbacks/ema.py` for the state dict layout, then
grep eval code for how it's surfaced.

## Implementation sketch

Copy `slurm_scripts/comparison/eval/submit_eval_crps_ambient.sh`, set
`eval.use_ema=false` (or equivalent), and write to a
`eval_no_ema/evaluation_metrics.csv` output so the EMA-on numbers from
the main eval pass aren't overwritten.
