from collections import OrderedDict
from typing import cast

import pytest
import torch
from hydra.utils import instantiate
from torch import nn
from torch.optim.swa_utils import AveragedModel

from autocast.processors.residual_flow_matching import ResidualFlowMatchingProcessor
from autocast.processors.residual_normalization import ResidualStandardizer
from autocast.processors.residual_reference import LastFrameReference


class _Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity = nn.Parameter(torch.tensor(0.5))

    def forward(self, z, t, cond, global_cond=None):  # noqa: ARG002
        return torch.ones_like(z) * self.velocity


def _processor(**kwargs):
    return ResidualFlowMatchingProcessor(
        backbone=_Field(),
        reference=LastFrameReference(n_steps_output=2),
        n_steps_output=2,
        **kwargs,
    )


@pytest.mark.parametrize("prefix", ["", "processor.", "module.processor."])
@pytest.mark.parametrize("legacy", [False, True])
def test_strict_checkpoint_load_preserves_statistics_and_predictions(prefix, legacy):
    processor = _processor(
        residual_normalizer=ResidualStandardizer(
            n_steps_output=2,
            n_channels=1,
        )
    )
    model: nn.Module = processor
    if prefix:
        model = nn.Module()
        model.processor = processor
    if prefix.startswith("module."):
        model = AveragedModel(model)
        processor = cast(ResidualFlowMatchingProcessor, model.module.processor)

    component = "standardizer" if legacy else "residual_normalizer"
    state = OrderedDict(
        {
            prefix + "flow_matching_model.velocity": torch.tensor(0.5),
            prefix + component + ".mean": torch.tensor([[1.0], [2.0]]),
            prefix + component + ".scale": torch.tensor([[2.0], [3.0]]),
            prefix + component + "._fitted": torch.tensor(True),
        }
    )
    if prefix.startswith("module."):
        state["n_averaged"] = torch.tensor(5)
    original_keys = tuple(state)
    result = model.load_state_dict(state, strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert tuple(state) == original_keys
    normalizer = processor.residual_normalizer
    assert normalizer is not None
    assert normalizer.fitted
    assert torch.equal(normalizer.mean, torch.tensor([[1.0], [2.0]]))
    assert torch.equal(normalizer.scale, torch.tensor([[2.0], [3.0]]))
    x = torch.full((1, 1, 3, 1), 10.0)
    actual = processor.map_from_source(x, None, source_states=torch.zeros(1, 2, 3, 1))
    expected = torch.tensor([12.0, 13.5]).view(1, 2, 1, 1).expand_as(actual)
    torch.testing.assert_close(actual, expected)
    assert not any("standardizer." in key for key in model.state_dict())


def test_legacy_hydra_target_and_constructor_still_work():
    normalizer = instantiate(
        {
            "_target_": (
                "autocast.processors.residual_normalization.ResidualStandardizer"
            ),
            "n_steps_output": 2,
            "n_channels": 1,
            "mean": 2.0,
            "scale": 3.0,
        }
    )
    processor = _processor(standardizer=normalizer)
    assert processor.residual_normalizer is normalizer
    assert processor.standardizer is normalizer
    assert list(processor._modules).count("residual_normalizer") == 1
    assert "standardizer" not in processor._modules
    processor.standardizer = None
    assert processor.residual_normalizer is None
    processor.standardizer = normalizer
    assert processor.residual_normalizer is normalizer
    assert "standardizer" not in processor._modules


def test_checkpoint_rejects_conflicting_names_and_unrelated_keys():
    processor = _processor(
        residual_normalizer=ResidualStandardizer(
            n_steps_output=2,
            n_channels=1,
            mean=0.0,
            scale=1.0,
        )
    )
    state = processor.state_dict()
    state["standardizer.mean"] = torch.ones(2, 1)
    with pytest.raises(RuntimeError, match=r"both.*standardizer.*residual_normalizer"):
        processor.load_state_dict(state, strict=True)
    state.pop("standardizer.mean")
    state["unexpected_weight"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        processor.load_state_dict(state, strict=True)


def test_rejects_ambiguous_configuration_and_dataset_normalizers():
    normalizer = ResidualStandardizer(n_steps_output=2, n_channels=1)
    with pytest.raises(ValueError, match="both"):
        _processor(residual_normalizer=normalizer, standardizer=normalizer)
    with pytest.raises(TypeError, match="ResidualNormalizer"):
        _processor(residual_normalizer=nn.Identity())
