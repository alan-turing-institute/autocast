import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch.optim import Optimizer


class GradNormCallback(Callback):
    """Log gradient p-norms (total, max, and the current LR) before each optimizer step.

    Reduce gradient norms on device and forward them through
    ``pl_module.log_dict``. Computation defaults to every optimizer step;
    set ``every_n_steps`` to the trainer's ``log_every_n_steps`` to avoid
    computing diagnostics between logger updates.

    Diagnostic only: Lightning calls ``on_before_optimizer_step`` after
    backward (and gradient all-reduce under DDP) but before any gradient
    clipping or the optimizer update, so the logged norms are the true
    values the optimizer is about to act on. A total norm that explodes in
    lockstep with the training loss indicates a stability problem (missing
    clipping, no warmup, or too-high LR) rather than under-capacity; the max
    per-parameter norm isolates a single layer blowing up when the total
    looks otherwise unremarkable.

    Parameters
    ----------
    norm_type
        Positive order of the p-norm, including infinity. Defaults to
        ``2.0`` (Euclidean).
    log_per_param
        If ``True``, also log a separate scalar per parameter tensor (keys
        like ``grad_2.0_norm/encoder.layer.weight``). If ``False`` (default),
        only the aggregate total/max and the current LR are logged. The
        per-parameter mode is useful when diagnosing per-block training
        pathologies but produces a wide W&B namespace.
    every_n_steps
        Compute before each multiple of this many optimizer updates.
        Uses the trainer's global step, including after resuming.
    """

    def __init__(
        self,
        norm_type: float = 2.0,
        log_per_param: bool = False,
        every_n_steps: int = 1,
    ) -> None:
        super().__init__()
        self.norm_type = float(norm_type)
        if self.norm_type <= 0:
            msg = "norm_type must be positive or infinity."
            raise ValueError(msg)
        if (
            isinstance(every_n_steps, bool)
            or not isinstance(every_n_steps, int)
            or every_n_steps <= 0
        ):
            msg = "every_n_steps must be a positive integer."
            raise ValueError(msg)
        self.log_per_param = log_per_param
        self.every_n_steps = every_n_steps

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: Optimizer,
    ) -> None:
        """Compute and log gradient norms + the current LR before the optimizer step."""
        # Older pickled callbacks did not store a computation interval.
        interval = getattr(self, "every_n_steps", 1)
        if interval > 1 and (trainer.global_step + 1) % interval:
            return
        norm_type = float(self.norm_type)
        prefix = f"grad_{norm_type}_norm"
        norms = {
            f"{prefix}/{name}": parameter.grad.detach().norm(norm_type)
            for name, parameter in pl_module.named_parameters()
            if parameter.grad is not None
        }
        if not norms:
            return
        device = next(iter(norms.values())).device
        # Tensor construction from scalars and Python max force CUDA host reads.
        values = torch.stack([norm.to(device) for norm in norms.values()])
        log_dict: dict[str, torch.Tensor | float] = (
            dict(norms) if self.log_per_param else {}
        )
        log_dict[f"{prefix}_total"] = values.norm(norm_type)
        log_dict[f"{prefix}_max"] = values.amax()
        if optimizer.param_groups:
            log_dict["lr"] = float(optimizer.param_groups[0]["lr"])
        pl_module.log_dict(log_dict, on_step=True, on_epoch=False, sync_dist=False)
