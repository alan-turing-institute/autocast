from collections.abc import Mapping

import pytest
import torch
from torch import nn

from autocast.losses.composite import WeightedSumLoss
from autocast.losses.decomposition import (
    ComponentBranch,
    ComponentObjective,
    ComponentTransform,
    FieldDecomposition,
    IdentityDecomposition,
    ScoringBranch,
    SequentialDecomposition,
    TransformedLoss,
)
from autocast.losses.ensemble import CRPSLoss
from autocast.types import Tensor


class _TwoBandDecomposition(FieldDecomposition):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, tensor: Tensor, *, n_spatial_dims: int) -> dict[str, Tensor]:
        del n_spatial_dims
        self.calls += 1
        return {"low": tensor, "high": 2 * tensor}


class _BufferedScale(ComponentTransform):
    factor: Tensor

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.register_buffer("factor", torch.tensor(factor))
        self.spatial_ranks: list[int] = []

    def forward(self, tensor: Tensor, *, n_spatial_dims: int) -> Tensor:
        self.spatial_ranks.append(n_spatial_dims)
        return self.factor * tensor


class _SpatialMeanBranch(ScoringBranch):
    def forward(
        self,
        preds: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
        *,
        n_spatial_dims: int,
    ) -> Tensor:
        spatial = tuple(range(2, 2 + n_spatial_dims))
        return (preds["field"] - targets["field"]).abs().mean(dim=spatial)


class _RaggedDecomposition(FieldDecomposition):
    def forward(self, tensor: Tensor, *, n_spatial_dims: int) -> dict[str, Tensor]:
        del n_spatial_dims
        return {"all": tensor, "short": tensor[:, :1]}


def _shaped_fields() -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(2, 3, 8, 6, 4, 5, generator=generator),
        torch.randn(2, 3, 8, 6, 4, generator=generator),
    )


def test_identity_decomposition_preserves_direct_crps():
    preds, targets = _shaped_fields()

    direct = CRPSLoss()(preds, targets)
    transformed = TransformedLoss(IdentityDecomposition(), CRPSLoss())(preds, targets)

    assert torch.equal(transformed, direct)


def test_component_objective_reuses_one_decomposition():
    decomposition = _TwoBandDecomposition()
    first_transform, second_transform = _BufferedScale(2.0), _BufferedScale(0.5)
    objective = ComponentObjective(
        [
            ComponentBranch(
                nn.L1Loss(), transforms=[first_transform, second_transform]
            ),
            ComponentBranch(nn.MSELoss(), components=["low"]),
        ],
        weights=[0.5, 2.0],
    )
    preds = torch.ones((1, 1, 2, 1), requires_grad=True)
    targets = torch.zeros_like(preds)

    value = TransformedLoss(decomposition, objective)(preds, targets)
    value.backward()

    torch.testing.assert_close(value, torch.tensor(2.75))
    assert decomposition.calls == 2
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()
    assert first_transform.spatial_ranks == [1, 1, 1, 1]
    assert second_transform.spatial_ranks == [1, 1, 1, 1]


def test_component_weights_are_a_mean_and_objective_weights_are_a_sum():
    preds = torch.ones(1, 1, 2, 1)
    targets = torch.zeros_like(preds)
    loss = TransformedLoss(
        _TwoBandDecomposition(),
        nn.L1Loss(),
        component_weights={"low": 1, "high": 3},
    )

    torch.testing.assert_close(loss(preds, targets), torch.tensor(1.75))
    combined = WeightedSumLoss([loss, loss], weights=[1, 2])
    torch.testing.assert_close(combined(preds, targets), torch.tensor(5.25))


def test_custom_branch_reaches_a_transformed_loss_with_spatial_rank():
    preds = torch.ones(2, 3, 5, 7, 4, dtype=torch.complex64, requires_grad=True)
    targets = torch.zeros_like(preds)
    objective = ComponentObjective([_SpatialMeanBranch()], weights=[2.0])
    loss = TransformedLoss(IdentityDecomposition(), objective=objective)

    score = loss(preds, targets)
    torch.testing.assert_close(score, torch.full((2, 3, 4), 2.0))
    score.mean().backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()


def test_transforms_are_registered_and_follow_dtype_changes():
    transform = _BufferedScale(2.0)
    loss = TransformedLoss(
        IdentityDecomposition(),
        objective=ComponentObjective(
            [ComponentBranch(nn.L1Loss(), transforms=[transform])]
        ),
    )

    loss.double()

    assert transform.factor.dtype == torch.float64
    assert "objective.branches.0.transforms.0.factor" in loss.state_dict()


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_component_aggregation_preserves_score_shape(reduction):
    preds, targets = _shaped_fields()
    score = CRPSLoss(reduction=reduction)
    objective = ComponentObjective(
        [ComponentBranch(score), ComponentBranch(score)], weights=[0.25, 0.75]
    )
    loss = TransformedLoss(IdentityDecomposition(), objective=objective)

    expected = score(preds, targets)
    actual = loss(preds, targets)

    assert actual.shape == (targets.shape if reduction == "none" else ())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        WeightedSumLoss([loss, loss], weights=[0.25, 0.75])(preds, targets), expected
    )


def test_mismatched_component_shapes_raise():
    preds, targets = _shaped_fields()
    loss = TransformedLoss(_RaggedDecomposition(), CRPSLoss(reduction="none"))

    with pytest.raises(ValueError, match="share one shape"):
        loss(preds, targets)


def test_component_objective_rejects_mixed_shapes():
    preds, targets = _shaped_fields()
    objective = ComponentObjective(
        [ComponentBranch(CRPSLoss()), ComponentBranch(CRPSLoss(reduction="none"))]
    )

    with pytest.raises(ValueError, match="share one shape"):
        TransformedLoss(IdentityDecomposition(), objective=objective)(preds, targets)


def test_weighted_sum_rejects_mixed_shapes():
    preds, targets = _shaped_fields()

    with pytest.raises(ValueError, match="share one shape"):
        WeightedSumLoss([CRPSLoss(), CRPSLoss(reduction="none")])(preds, targets)


@pytest.mark.parametrize(
    ("components", "weights", "message"),
    [
        ([], None, "nonempty"),
        (["field", "field"], None, "duplicates"),
        (["field"], {"other": 1}, "unselected"),
        (None, {"field": -1}, "non-negative"),
        (None, {"field": float("nan")}, "finite"),
        (None, {"field": float("inf")}, "finite"),
        (None, {"field": -float("inf")}, "finite"),
    ],
)
def test_component_branch_validates_constructor(components, weights, message):
    with pytest.raises(ValueError, match=message):
        ComponentBranch(nn.L1Loss(), components=components, component_weights=weights)


@pytest.mark.parametrize(
    ("components", "weights", "message"),
    [
        (["missing"], None, "Unknown components"),
        (None, {"missing": 1}, "Unknown components"),
        (None, {"field": 0}, "positive"),
    ],
)
def test_component_branch_validates_available_components(components, weights, message):
    field = torch.zeros(1, 1, 2, 1)
    branch = ComponentBranch(
        nn.L1Loss(), components=components, component_weights=weights
    )

    with pytest.raises(ValueError, match=message):
        branch({"field": field}, {"field": field}, n_spatial_dims=1)


def test_component_branch_rejects_different_component_keys():
    field = torch.zeros(1, 1, 2, 1)

    with pytest.raises(ValueError, match="components must match"):
        ComponentBranch(nn.L1Loss())(
            {"first": field}, {"second": field}, n_spatial_dims=1
        )


@pytest.mark.parametrize("owner", [WeightedSumLoss, ComponentObjective])
@pytest.mark.parametrize(
    ("count", "weights", "message"),
    [
        (0, None, "at least one"),
        (1, [1, 2], "exactly one"),
        (1, [-1], "non-negative"),
        (1, [float("nan")], "finite"),
        (1, [float("inf")], "finite"),
        (1, [-float("inf")], "finite"),
    ],
)
def test_aggregation_validates_terms_and_weights(owner, count, weights, message):
    items = [ComponentBranch(nn.L1Loss()) for _ in range(count)]

    with pytest.raises(ValueError, match=message):
        owner(items, weights=weights)


def test_sequential_decomposition_namespaces_components():
    decomposition = SequentialDecomposition(
        [_TwoBandDecomposition(), IdentityDecomposition()]
    )
    tensor = torch.ones((1, 1, 2, 1))

    components = decomposition(tensor, n_spatial_dims=1)

    assert components.keys() == {"low/field", "high/field"}
    assert torch.equal(components["low/field"], tensor)
    assert torch.equal(components["high/field"], 2 * tensor)


def test_sequential_decomposition_requires_a_stage():
    with pytest.raises(ValueError, match="at least one stage"):
        SequentialDecomposition([])


def test_transformed_loss_requires_one_scoring_path():
    decomposition = IdentityDecomposition()
    score = nn.L1Loss()
    objective = ComponentObjective([ComponentBranch(score)])

    with pytest.raises(ValueError, match="exactly one"):
        TransformedLoss(decomposition)
    with pytest.raises(ValueError, match="exactly one"):
        TransformedLoss(decomposition, score, objective=objective)
    with pytest.raises(ValueError, match="belong to the component objective"):
        TransformedLoss(
            decomposition, component_weights={"field": 1}, objective=objective
        )


def test_transformed_loss_rejects_broadcasting_input_shapes():
    preds = torch.ones(2, 3, 4, 2, 5)
    targets = torch.ones(2, 1, 4, 2)

    with pytest.raises(ValueError, match="Predictions must match targets"):
        TransformedLoss(IdentityDecomposition(), CRPSLoss())(preds, targets)


def test_ambient_requirement_is_explicit_for_transformed_loss():
    unmarked = TransformedLoss(IdentityDecomposition(), nn.L1Loss())
    marked = TransformedLoss(
        IdentityDecomposition(), nn.L1Loss(), requires_ambient_predictions=True
    )

    assert unmarked.requires_ambient_predictions is False
    assert marked.requires_ambient_predictions is True
    assert (
        WeightedSumLoss([unmarked, nn.MSELoss()]).requires_ambient_predictions is False
    )
    assert WeightedSumLoss([unmarked, marked]).requires_ambient_predictions is True


def test_ambient_requirement_propagates_through_nested_and_zero_weight_terms():
    marked = TransformedLoss(
        IdentityDecomposition(), nn.L1Loss(), requires_ambient_predictions=True
    )
    inner = WeightedSumLoss([nn.L1Loss(), marked], weights=[1, 0])
    outer = WeightedSumLoss([nn.L1Loss(), inner], weights=[1, 0])

    assert inner.requires_ambient_predictions is True
    assert outer.requires_ambient_predictions is True
