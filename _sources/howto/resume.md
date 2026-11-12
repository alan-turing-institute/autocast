# Resuming from a checkpoint

The following extra CLI options can be passed to the `ae`, `epd`, and `train-eval` subcommands (or added to configuration files):

- Resume from a saved checkpoint.
  The default is to perform a full-state resume, which restores model + optimizer/scheduler/trainer loop state.

  ```bash
  --resume-from path/to/encoder_processor_decoder.ckpt
  ```

- To additionally reset the timer budget:

  ```bash
  --resume-from path/to/encoder_processor_decoder.ckpt ++trainer.max_time="00:04:00:00" ++train_eval.reset_resume_time_budget=true
  ```

- To restore only the model weights and generate a fresh optimizer/trainer state:
 
  ```bash
  --resume-from path/to/encoder_processor_decoder.ckpt ++trainer.max_time="00:04:00:00" ++train_eval.resume_weights_only=true
  ```

  In conjunction with `trainer.max_time`, this allows you to continue training with a fresh timer budget.
  Note that if `resume_weights_only=true` is set without a checkpoint, AutoCast raises an error.


