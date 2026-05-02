# EMA

Eval-only ablation: evaluate already-trained runs using the EMA shadow
weights stored in each checkpoint, without retraining.

## Script

- `submit_eval_fm_encode_once_ema.sh`

## Source Runs

Main 2026-04-20 FM cached-latent runs:

- `outputs/2026-04-20/diff_gs64_flow_matching_vit_09490da_7e9e331`
- `outputs/2026-04-20/diff_gpe64_flow_matching_vit_09490da_47bf39a`
- `outputs/2026-04-20/diff_cns64_flow_matching_vit_09490da_636fcc3`
- `outputs/2026-04-20/diff_ad64_flow_matching_vit_09490da_dae1382`

All four `processor.ckpt` files have an `ema_state_dict` key.

## Knob

- `+eval.use_ema=true`
- `eval.mode=encode_once`, so metrics are computed in raw data space after
  decoding while avoiding the ambient path's per-step decode/encode loop.
- Baseline online-weight comparison evals use the same checkpoints without
  `eval.use_ema`.

Outputs write to:

- `eval_encode_once_ema/`

The submitter also overrides `eval.csv_path` and `eval.video_dir` inside that
subdir so EMA evals do not clobber online-weight evals.
