# ruff: noqa: ARG002
from lightning.pytorch.callbacks import Callback
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class EMACallback(Callback):
    """Exponential Moving Average (EMA) callback for PyTorch Lightning.

    Maintains an EMA of the model parameters and swaps them in during
    validation, testing, and prediction. This greatly stabilizes metrics
    for generative models (Diffusion/Flow Matching) and mitigates overfitting
    late in training.
    """

    def __init__(self, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self.ema_model = None
        self.original_state_dict = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update EMA parameters after every training batch."""
        if self.ema_model is not None:
            self.ema_model.update_parameters(pl_module)

    def _swap_model_weights(self, pl_module):
        """Swap current model weights with the EMA weights."""
        if self.ema_model is None:
            return

        # Store original training weights
        self.original_state_dict = {
            k: v.clone().detach() for k, v in pl_module.state_dict().items()
        }
        # Load EMA weights into the Lightning module
        # Note: AveragedModel wraps the original module in a 'module' attribute
        pl_module.load_state_dict(self.ema_model.module.state_dict())

    def _restore_original_weights(self, pl_module):
        """Restore original training weights."""
        if self.original_state_dict is not None:
            pl_module.load_state_dict(self.original_state_dict)
            self.original_state_dict = None

    def on_validation_start(self, trainer, pl_module):
        self._swap_model_weights(pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._restore_original_weights(pl_module)

    def on_test_start(self, trainer, pl_module):
        self._swap_model_weights(pl_module)

    def on_test_end(self, trainer, pl_module):
        self._restore_original_weights(pl_module)

    def on_predict_start(self, trainer, pl_module):
        self._swap_model_weights(pl_module)

    def on_predict_end(self, trainer, pl_module):
        self._restore_original_weights(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict) -> None:
        """Overwrite the saved state_dict with the EMA weights for clean inference.

        This ensures that when standalone evaluation scripts (like eval.py)
        extract checkpoint['state_dict'], they natively load the high-fidelity
        EMA parameters instead of the noisy optimizer training state!
        """
        if self.ema_model is not None:
            checkpoint["state_dict"] = {
                k: v.clone() for k, v in self.ema_model.module.state_dict().items()
            }

    def state_dict(self) -> dict:
        """Save the EMA model state to the Lightning checkpoint."""
        return {
            "ema_model_state": self.ema_model.state_dict() if self.ema_model else None
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the EMA model state from the Lightning checkpoint."""
        # Note: on_fit_start must be handled correctly if resuming
        self._loaded_state_dict = state_dict.get("ema_model_state")

    def on_fit_start(self, trainer, pl_module):
        """Initialize the EMA model using the starting weights."""
        if self.ema_model is None:
            avg_fn = get_ema_multi_avg_fn(self.decay)
            self.ema_model = AveragedModel(pl_module, multi_avg_fn=avg_fn)
            # If resuming from checkpoint, load the preserved EMA state
            if getattr(self, "_loaded_state_dict", None) is not None:
                self.ema_model.load_state_dict(self._loaded_state_dict)  # type: ignore[arg-type]
                self._loaded_state_dict = None
