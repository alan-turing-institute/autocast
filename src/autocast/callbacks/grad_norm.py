# ruff: noqa: ARG002
import torch
from lightning.pytorch.callbacks import Callback


class GradNormCallback(Callback):
    """Log the global pre-clip gradient norm (and LR) every optimizer step.

    Diagnostic only: it reads ``p.grad`` in ``on_before_optimizer_step``, which
    Lightning calls after backward (and gradient all-reduce under DDP) but
    *before* any gradient clipping or the optimizer update. The logged
    ``grad_norm`` is therefore the true norm the optimizer would act on.

    Use it to attribute ``train_loss`` spikes: a ``grad_norm`` that explodes in
    lockstep with the loss indicates a stability problem (missing clipping,
    no warmup, or too-high LR) rather than under-capacity. The callback never
    touches the gradients, so training dynamics are unchanged.
    """

    def __init__(self, norm_type: float = 2.0):
        super().__init__()
        self.norm_type = float(norm_type)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        grads = [p.grad.detach() for p in pl_module.parameters() if p.grad is not None]
        if not grads:
            return
        per_param = torch.stack([g.norm(self.norm_type) for g in grads])
        total = per_param.norm(self.norm_type)
        pl_module.log(
            "grad_norm",
            total,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
        )
        # Largest single-parameter contribution, to spot one layer blowing up.
        pl_module.log(
            "grad_norm_max",
            per_param.max(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
        )
        if optimizer.param_groups:
            pl_module.log(
                "lr",
                float(optimizer.param_groups[0]["lr"]),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=False,
            )
