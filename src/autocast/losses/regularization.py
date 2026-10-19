from torch import nn

from autocast.types import Tensor


class MCDropoutMSEL2Loss(nn.MSELoss):
    r"""Mean squared error plus processor-local L2 for MC dropout.

    This implements a WeatherBench Probability-inspired empirical objective:

    .. math::

        \operatorname{MSE}(\hat y, y)
        + \lambda \sum_{W \in \theta_{\mathrm{processor}}} \lVert W \rVert_2^2.

    ``l2_coefficient`` is the directly configured :math:`\lambda`; it is not
    inferred from the number of trajectories, windows, or target scalars.
    Only trainable processor parameters with at least two dimensions are
    included, excluding biases and normalization vectors. L2 is applied during
    training only, so validation and test losses remain plain MSE.

    The processor is held as an unregistered reference. This prevents the same
    module from appearing twice in the parent model's module tree and
    checkpoint while retaining gradients and preserving the reference when an
    EMA model is created with ``deepcopy``.

    This loss is intended for ambient ``EncoderProcessorDecoder`` training,
    where the model loss receives decoded predictions. It is an
    approximate-Bayesian-inspired objective, not a calibrated posterior.
    """

    requires_processor = True
    requires_ambient_predictions = True

    def __init__(
        self,
        processor: nn.Module,
        l2_coefficient: float = 1e-5,
    ) -> None:
        super().__init__()
        if l2_coefficient < 0.0:
            msg = "l2_coefficient must be non-negative."
            raise ValueError(msg)

        object.__setattr__(self, "_processor", processor)
        self.l2_coefficient = float(l2_coefficient)

    @property
    def processor(self) -> nn.Module:
        """Return the bound processor."""
        return self.__dict__["_processor"]

    def l2_penalty(self, reference: Tensor) -> Tensor:
        """Return fixed-coefficient L2 over multidimensional processor weights."""
        squared_norm: Tensor | None = None
        for parameter in self.processor.parameters():
            if parameter.requires_grad and parameter.ndim >= 2:
                term = parameter.square().sum()
                squared_norm = term if squared_norm is None else squared_norm + term

        if squared_norm is None:
            return reference.new_zeros(())
        return squared_norm * self.l2_coefficient

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Return MSE plus processor L2 during training."""
        data_loss = super().forward(input, target)
        if not self.training or self.l2_coefficient == 0.0:
            return data_loss
        return data_loss + self.l2_penalty(data_loss)
