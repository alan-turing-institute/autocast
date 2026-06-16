import torch

from autocast.data.transforms import LogScalars


def test_log_scalars_logs_selected_constant_scalars() -> None:
    transform = LogScalars(scalar_names=("Rayleigh", "Prandtl", "rho0"))
    data = {
        "constant_scalars": {
            "Rayleigh": torch.tensor([1.0e6]),
            "Prandtl": torch.tensor([7.0]),
            "rho0": torch.tensor([3.0]),
            "T0": torch.tensor([10.0]),
            "unscaled": torch.tensor([5.0]),
        }
    }

    out = transform(data, metadata=None)  # type: ignore  # noqa: PGH003
    scalars = out["constant_scalars"]

    assert torch.allclose(scalars["Rayleigh"], torch.log(torch.tensor([1.0e6])))
    assert torch.allclose(scalars["Prandtl"], torch.log(torch.tensor([7.0])))
    assert torch.allclose(scalars["rho0"], torch.log(torch.tensor([3.0])))
    assert torch.equal(scalars["T0"], torch.tensor([10.0]))
    assert torch.equal(scalars["unscaled"], torch.tensor([5.0]))


def test_log_scalars_allows_custom_scalar_names() -> None:
    transform = LogScalars(scalar_names=("custom",))
    data = {"constant_scalars": {"custom": torch.tensor([4.0])}}

    out = transform(data, metadata=None)  # type: ignore  # noqa: PGH003

    assert torch.allclose(
        out["constant_scalars"]["custom"], torch.log(torch.tensor([4.0]))
    )
