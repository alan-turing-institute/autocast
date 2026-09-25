from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import grad_norm
from torch.optim import Optimizer


class GradNormCallback(Callback):
    """Log gradient p-norms (total, max, and the current LR) before each optimizer step.

    Uses :func:`lightning.pytorch.utilities.grad_norm` to compute the
    p-norm of the gradient of every parameter that requires grad, then
    forwards the result through ``pl_module.log_dict`` so it lands in
    every configured logger (W&B, CSV, …) at the trainer's
    ``log_every_n_steps`` cadence.

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
        Order of the p-norm passed to ``grad_norm``. Defaults to ``2.0``
        (Euclidean).
    log_per_param
        If ``True``, also log a separate scalar per parameter tensor (keys
        like ``grad_2.0_norm/encoder.layer.weight``). If ``False`` (default),
        only the aggregate total/max and the current LR are logged. The
        per-parameter mode is useful when diagnosing per-block training
        pathologies but produces a wide W&B namespace.
    """

    def __init__(
        self,
        norm_type: float = 2.0,
        log_per_param: bool = False,
    ) -> None:
        super().__init__()
        self.norm_type = norm_type
        self.log_per_param = log_per_param

    def on_before_optimizer_step(
        self,
        trainer: Trainer,  # noqa: ARG002 — required by Lightning's hook signature
        pl_module: LightningModule,
        optimizer: Optimizer,
    ) -> None:
        """Compute and log gradient norms + the current LR before the optimizer step."""
        norms = grad_norm(pl_module, norm_type=self.norm_type)
        if not norms:
            return
        # endswith handles version-to-version drift in the formatted
        # norm_type segment of the key (e.g. ``2.0`` vs ``2``).
        total = {k: v for k, v in norms.items() if k.endswith("_norm_total")}
        per_param = {k: v for k, v in norms.items() if k not in total}
        log_dict = dict(per_param) if self.log_per_param else {}
        log_dict.update(total)
        if per_param:
            max_key = next(iter(total)).replace("_norm_total", "_norm_max")
            log_dict[max_key] = max(per_param.values())
        if optimizer.param_groups:
            log_dict["lr"] = float(optimizer.param_groups[0]["lr"])
        pl_module.log_dict(log_dict, on_step=True, on_epoch=False, sync_dist=False)
