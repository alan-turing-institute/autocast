# ruff: noqa: ARG002
from lightning.pytorch.callbacks import Callback
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

EMA_CHECKPOINT_KEY = "ema_state_dict"


class EMACallback(Callback):
    """Exponential Moving Average (EMA) callback for PyTorch Lightning.

    Silently maintains an EMA of the model parameters during training and
    writes them into the checkpoint under a dedicated key
    (``ema_state_dict``) so that ``state_dict`` still reflects the raw
    training weights. Validation/train metrics therefore stay comparable
    across runs; eval scripts opt in to the EMA weights explicitly.
    """

    def __init__(self, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self.ema_model = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update EMA parameters after every training batch."""
        if self.ema_model is not None:
            self.ema_model.update_parameters(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict) -> None:
        """Persist EMA weights alongside the raw training state_dict.

        Stored under a separate top-level key so eval scripts can opt in
        without affecting training-time checkpoints or resume behaviour.
        """
        if self.ema_model is not None:
            checkpoint[EMA_CHECKPOINT_KEY] = {
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
