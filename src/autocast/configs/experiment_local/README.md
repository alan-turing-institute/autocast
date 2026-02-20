# Local experiment presets

Use this folder for private/local Hydra experiment presets that should not be committed.

Create files like `src/autocast/configs/experiment_local/my_private_run.yaml` with:

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
uv run train_encoder_processor_decoder experiment_local=my_private_run
```
