"""Names of the grids on which datasets store their fields."""

from enum import StrEnum


class GridType(StrEnum):
    """Grid of the stored fields.

    ``CARTESIAN`` is the regular planar grid of the Cartesian datasets. The other
    members name the grids on the sphere with the strings of torch-harmonics
    (``torch_harmonics.grid_types()``). A payload stores the plain string under
    ``"grid_type"``, so ``torch.load(weights_only=True)`` reads it back.
    """

    CARTESIAN = "cartesian"
    EQUIANGULAR = "equiangular"
    FEJER1 = "fejer1"
    LEGENDRE_GAUSS = "legendre-gauss"
    LOBATTO = "lobatto"
    TRAPEZOIDAL = "trapezoidal"
    HEALPIX = "healpix"
