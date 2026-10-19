"""Parameter-norm diagnostics callback.

Companion to :mod:`autocast.callbacks.grad_norm`. Where gradient norms
track per-step learning dynamics, parameter norms track the slower
weight-decay dynamics — the trajectory of ``||theta||_p`` over training
is the cleanest available signal for whether AdamW's ``weight_decay`` is
appropriately tuned: a stable ``||theta||_2`` indicates the decay term
is roughly balancing the gradient update; unbounded growth signals
``weight_decay`` is too low; collapse toward zero signals it is too
high.
"""

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


def _param_norm(
    module: torch.nn.Module,
    norm_type: float,
    group_separator: str = "/",
) -> dict[str, torch.Tensor]:
    """Compute each parameter's p-norm and their overall p-norm.

    Mirrors :func:`lightning.pytorch.utilities.grad_norm` but for the
    parameter *values* rather than their gradients. Only parameters with
    ``requires_grad=True`` are included (frozen weights — e.g. a frozen
    encoder in latent-space training — are excluded so the metric
    reflects the trainable surface that ``weight_decay`` acts on).

    Args:
        module: Module to inspect.
        norm_type: Order of the p-norm. Must be positive.
        group_separator: Separator between key prefix and parameter
            name; matches Lightning's ``grad_norm`` for parity.

    Returns:
    -------
        Dict mapping ``param_<norm_type>_norm<sep><param-name>`` to each
        per-tensor norm, with a final ``param_<norm_type>_norm_total``
        aggregate. Empty dict if no parameters require grad.
    """
    norms = {
        f"param_{norm_type}_norm{group_separator}{name}": p.detach().norm(norm_type)
        for name, p in module.named_parameters()
        if p.requires_grad
    }
    if norms:
        total_norm = torch.stack(list(norms.values())).norm(norm_type)
        norms[f"param_{norm_type}_norm_total"] = total_norm
    return norms


class ParamNormCallback(Callback):
    """Log parameter p-norms at the end of each training epoch.

    Fires ``on_train_epoch_end`` (not per-step) because parameter values
    evolve on a slow timescale; a per-step log would only inflate the W&B
    namespace without revealing new signal.

    Parameters
    ----------
    norm_type
        Order of the p-norm. Defaults to ``2.0`` (Euclidean). Must be
        positive.
    log_per_param
        If ``True``, log a separate scalar per parameter tensor (keys like
        ``param_2.0_norm/processor.backbone.layers.0.attn.weight``). If
        ``False`` (default), only the aggregate
        ``param_<norm_type>_norm_total`` is logged. The per-parameter mode
        is useful when diagnosing per-block weight-decay pathologies but
        produces a wide W&B namespace.
    """

    def __init__(
        self,
        norm_type: float = 2.0,
        log_per_param: bool = False,
    ) -> None:
        super().__init__()
        if norm_type <= 0:
            msg = (
                f"`norm_type` must be a positive number (got {norm_type}). "
                "Use 1.0 for L1, 2.0 for Euclidean, etc."
            )
            raise ValueError(msg)
        self.norm_type = float(norm_type)
        self.log_per_param = log_per_param

    def on_train_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002 — required by Lightning's hook signature
        pl_module: LightningModule,
    ) -> None:
        """Compute and log parameter norms at the end of each training epoch."""
        norms = _param_norm(pl_module, norm_type=self.norm_type)
        if not self.log_per_param:
            # endswith handles version-to-version drift in the formatted
            # norm_type segment of the key (e.g. ``2.0`` vs ``2``).
            norms = {k: v for k, v in norms.items() if k.endswith("_norm_total")}
        if norms:
            pl_module.log_dict(
                norms,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
