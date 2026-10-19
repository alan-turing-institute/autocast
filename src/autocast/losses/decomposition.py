import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from autocast.types import FieldTensor, Tensor, TensorInexactBTSC

FieldComponents = dict[str, Tensor]  # component name -> field or coefficients


def _spatial_dims(tensor: Tensor, n_spatial_dims: int) -> tuple[int, ...]:
    if n_spatial_dims < 1 or 2 + n_spatial_dims >= tensor.ndim:
        raise ValueError(
            f"Invalid n_spatial_dims={n_spatial_dims} for shape {tuple(tensor.shape)}."
        )
    return tuple(range(2, 2 + n_spatial_dims))


def _sum_matching(terms: Sequence[Tensor], owner: str) -> Tensor:
    """Sum terms that must all share a shape, preserving that shape."""
    shapes = {tuple(term.shape) for term in terms}
    if len(shapes) > 1:
        msg = f"{owner} requires every score to share one shape; got {sorted(shapes)}."
        raise ValueError(msg)
    return torch.stack(tuple(terms)).sum(dim=0)


def _align_targets(preds: Tensor, targets: Tensor) -> Tensor:
    if preds.shape == targets.shape:
        return targets
    if preds.ndim == targets.ndim + 1 and preds.shape[:-1] == targets.shape:
        return targets.unsqueeze(-1)
    raise ValueError(
        "Predictions must match targets, optionally with a final ensemble axis. "
        f"Got {tuple(preds.shape)} and {tuple(targets.shape)}."
    )


class FieldDecomposition(nn.Module, ABC):
    """Decompose or transform a field into named components."""

    @abstractmethod
    def forward(
        self, tensor: FieldTensor, *, n_spatial_dims: int
    ) -> FieldComponents: ...


class ComponentTransform(nn.Module, ABC):
    """Map a component to another component of the same layout.

    Applied to prediction and target alike before scoring. ``n_spatial_dims``
    is supplied so a transform can reduce over spatial axes; a transform that
    acts elementwise ignores it.

    A transform that removes spatial axes must leave one of extent one, since
    the layout carries at least one spatial axis throughout.
    """

    @abstractmethod
    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> Tensor: ...


class IdentityDecomposition(FieldDecomposition):
    """Return the input field as a single named component."""

    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> FieldComponents:
        del n_spatial_dims
        return {"field": tensor}


class SequentialDecomposition(FieldDecomposition):
    """Apply each decomposition to every component from the previous stage."""

    def __init__(self, decompositions: Sequence[FieldDecomposition]) -> None:
        super().__init__()
        if not decompositions:
            msg = "SequentialDecomposition requires at least one stage."
            raise ValueError(msg)
        self.decompositions = nn.ModuleList(decompositions)

    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> FieldComponents:
        components = {"": tensor}
        for decomposition in self.decompositions:
            next_components: FieldComponents = {}
            for parent, component in components.items():
                children = decomposition(component, n_spatial_dims=n_spatial_dims)
                if not children:
                    msg = "Decompositions must return at least one component."
                    raise ValueError(msg)
                for child, value in children.items():
                    if not child:
                        msg = "Component names must be nonempty."
                        raise ValueError(msg)
                    name = f"{parent}/{child}" if parent else child
                    if name in next_components:
                        raise ValueError(f"Duplicate component name {name!r}.")
                    next_components[name] = value
            components = next_components
        return components


class ScoringBranch(nn.Module, ABC):
    """Turn selected components into one score.

    A branch chooses which components it reads and how they reach a score.
    An objective combines branches without regard to how each works, so any
    routing that ends in a score can be combined with any other.
    """

    @abstractmethod
    def forward(
        self,
        preds: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
        *,
        n_spatial_dims: int,
    ) -> Tensor: ...


class ComponentBranch(ScoringBranch):
    """Transform and score selected component pairs.

    The score need not be scalar: unreduced component scores retain their shape.
    Every selected component must produce the same shape.
    """

    def __init__(
        self,
        score: nn.Module,
        transforms: Sequence[ComponentTransform] = (),
        components: Sequence[str] | None = None,
        component_weights: Mapping[str, int | float] | None = None,
    ) -> None:
        super().__init__()
        if components is not None and not components:
            msg = "Component selection must be nonempty."
            raise ValueError(msg)
        if components is not None and len(set(components)) != len(components):
            msg = "Component selection must not contain duplicates."
            raise ValueError(msg)

        self.score = score
        self.transforms = nn.ModuleList(transforms)
        self.components = tuple(components) if components is not None else None
        self.component_weights = {
            name: float(weight) for name, weight in (component_weights or {}).items()
        }
        if any(
            not math.isfinite(weight) or weight < 0
            for weight in self.component_weights.values()
        ):
            msg = "Component weights must be finite and non-negative."
            raise ValueError(msg)
        if self.components is not None:
            unknown = self.component_weights.keys() - set(self.components)
            if unknown:
                msg = f"Weights provided for unselected components: {sorted(unknown)}."
                raise ValueError(msg)

    def forward(
        self,
        preds: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
        *,
        n_spatial_dims: int,
    ) -> Tensor:
        if preds.keys() != targets.keys():
            msg = "Prediction and target components must match."
            raise ValueError(msg)

        names = tuple(preds) if self.components is None else self.components
        unknown = set(names) - preds.keys()
        unknown.update(self.component_weights.keys() - preds.keys())
        if unknown:
            raise ValueError(f"Unknown components: {sorted(unknown)}.")

        weights = {name: self.component_weights.get(name, 1.0) for name in names}
        total_weight = sum(weights.values())
        if total_weight == 0:
            msg = "At least one component weight must be positive."
            raise ValueError(msg)

        terms = []
        for name in names:
            if weights[name] == 0:
                continue
            pred_component, target_component = preds[name], targets[name]
            for transform in self.transforms:
                pred_component = transform(
                    pred_component, n_spatial_dims=n_spatial_dims
                )
                target_component = transform(
                    target_component, n_spatial_dims=n_spatial_dims
                )
            terms.append(weights[name] * self.score(pred_component, target_component))
        return _sum_matching(terms, "ComponentBranch") / total_weight


class ComponentObjective(nn.Module):
    """Return a weighted sum of component-scoring branches.

    All branches must agree on shape; that shape is preserved.
    """

    def __init__(
        self,
        branches: Sequence[ScoringBranch],
        weights: Sequence[int | float] | None = None,
    ) -> None:
        super().__init__()
        if not branches:
            msg = "ComponentObjective requires at least one branch."
            raise ValueError(msg)
        resolved_weights = weights if weights is not None else [1.0] * len(branches)
        if len(resolved_weights) != len(branches):
            msg = "Each branch must have exactly one weight."
            raise ValueError(msg)
        if any(not math.isfinite(weight) or weight < 0 for weight in resolved_weights):
            msg = "Branch weights must be finite and non-negative."
            raise ValueError(msg)

        self.branches = nn.ModuleList(branches)
        self.weights = tuple(float(weight) for weight in resolved_weights)

    def forward(
        self,
        preds: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
        *,
        n_spatial_dims: int,
    ) -> Tensor:
        terms = [
            weight * branch(preds, targets, n_spatial_dims=n_spatial_dims)
            for branch, weight in zip(self.branches, self.weights, strict=True)
        ]
        return _sum_matching(terms, "ComponentObjective")


class TransformedLoss(nn.Module):
    """Decompose prediction and target fields, then score their components.

    Passing ``score`` computes a weighted mean over components. An explicit
    ``objective`` controls its own scoring and aggregation instead.

    The result takes whatever shape the component objective returns, so callers
    that apply their own weighting and reduction can request an unreduced score.

    ``requires_ambient_predictions`` marks an objective that EPD training must
    evaluate on decoded fields. It does not infer or validate a tensor's layout;
    unmarked compositions still require ``(B, T, *S, C[, M])`` fields.
    """

    def __init__(
        self,
        decomposition: FieldDecomposition,
        score: nn.Module | None = None,
        component_weights: Mapping[str, int | float] | None = None,
        *,
        objective: nn.Module | None = None,
        requires_ambient_predictions: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(score, ComponentObjective) and objective is None:
            score, objective = None, score
        if (score is None) == (objective is None):
            msg = "Provide exactly one of score or objective."
            raise ValueError(msg)
        if objective is not None and component_weights is not None:
            msg = "Component weights belong to the component objective."
            raise ValueError(msg)

        self.decomposition = decomposition
        if objective is None:
            assert score is not None
            objective = ComponentObjective(
                [
                    ComponentBranch(
                        score,
                        component_weights=component_weights,
                    )
                ]
            )
        self.objective = objective
        self.requires_ambient_predictions = requires_ambient_predictions

    def forward(self, preds: FieldTensor, targets: TensorInexactBTSC) -> Tensor:
        _align_targets(preds, targets)
        n_spatial_dims = targets.ndim - 3
        pred_components = self.decomposition(preds, n_spatial_dims=n_spatial_dims)
        target_components = self.decomposition(targets, n_spatial_dims=n_spatial_dims)
        if not pred_components or pred_components.keys() != target_components.keys():
            msg = "Prediction and target components must match and be nonempty."
            raise ValueError(msg)

        # Every decomposition here preserves spatial rank, so a component
        # carries the same number of spatial axes as the field it came from.
        return self.objective(
            pred_components, target_components, n_spatial_dims=n_spatial_dims
        )
