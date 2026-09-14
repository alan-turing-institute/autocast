"""Check AutoCast's simulator configs resolve to the intended AutoSim classes."""

from pathlib import Path

import pytest
from autosim.simulations import AdvectionDiffusionMultichannel
from autosim.simulations.reaction_diffusion import ReactionDiffusion
from hydra.utils import instantiate
from omegaconf import OmegaConf


@pytest.mark.parametrize(
    ("config_name", "expected_type", "expected_channels"),
    [
        ("reaction_diffusion", ReactionDiffusion, ["u", "v"]),
        (
            "advection_diffusion_singlechannel",
            AdvectionDiffusionMultichannel,
            ["vorticity"],
        ),
        (
            "advection_diffusion_multichannel",
            AdvectionDiffusionMultichannel,
            ["vorticity", "u", "v", "streamfunction"],
        ),
    ],
)
def test_simulator_config(
    REPO_ROOT: Path, config_name, expected_type, expected_channels
):
    config = OmegaConf.load(
        REPO_ROOT / "src/autocast/configs/simulator" / f"{config_name}.yaml"
    )
    simulator = instantiate(config.simulator)

    assert isinstance(simulator, expected_type)
    assert simulator.output_names == expected_channels
    assert simulator.return_timeseries is True
