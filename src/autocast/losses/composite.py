import math
from collections.abc import Sequence

from torch import nn

from autocast.losses.decomposition import _sum_matching
from autocast.types import FieldTensor, Tensor, TensorInexactBTSC


class WeightedSumLoss(nn.Module):
    """Combine losses using fixed non-negative weights.

    All losses must agree on shape; that shape is preserved.
    """

    def __init__(
        self,
        losses: Sequence[nn.Module],
        weights: Sequence[int | float] | None = None,
    ) -> None:
        super().__init__()
        if not losses:
            msg = "WeightedSumLoss requires at least one loss."
            raise ValueError(msg)

        resolved_weights = weights if weights is not None else [1.0] * len(losses)
        if len(resolved_weights) != len(losses):
            msg = "Each loss must have exactly one weight."
            raise ValueError(msg)
        if any(not math.isfinite(weight) or weight < 0 for weight in resolved_weights):
            msg = "Loss weights must be finite and non-negative."
            raise ValueError(msg)

        self.losses = nn.ModuleList(losses)
        self.weights = tuple(float(weight) for weight in resolved_weights)

    @property
    def requires_ambient_predictions(self) -> bool:
        """Whether any configured term requires decoded EPD predictions."""
        # Zero-weight terms are included because forward still evaluates them.
        return any(
            getattr(loss, "requires_ambient_predictions", False) for loss in self.losses
        )

    def forward(self, preds: FieldTensor, targets: TensorInexactBTSC) -> Tensor:
        """Return the weighted sum of all configured losses."""
        terms: list[Tensor] = [
            weight * loss(preds, targets)
            for loss, weight in zip(self.losses, self.weights, strict=True)
        ]
        return _sum_matching(terms, "WeightedSumLoss")
