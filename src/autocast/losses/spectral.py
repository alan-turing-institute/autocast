import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Literal, cast

import torch
from einops import rearrange
from ptwt.conv_transform import wavedec
from ptwt.separable_conv_transform import fswavedec2, fswavedec3
from pytorch_wavelets import DTCWTForward
from torch import nn

from autocast.losses.decomposition import (
    ComponentTransform,
    FieldComponents,
    FieldDecomposition,
    _align_targets,
    _spatial_dims,
)
from autocast.losses.ensemble import _bounded_gradient_sqrt, _distance_power
from autocast.types import FieldTensor, Tensor, TensorInexactBTSC

FourierRepresentation = Literal["complex", "magnitude", "power"]
FourierNorm = Literal["backward", "forward", "ortho"]
WaveletBoundary = Literal["reflect", "periodic"]


def _radial_frequency(tensor: Tensor, spatial: tuple[int, ...]) -> Tensor:
    # Geometry must not depend on prediction/target precision: rounding a
    # cutoff differently would compare different modes. Construct masks
    # in CPU float64, including for devices without float64 support.
    # Transfer masks and optional counts; field FFTs stay on their device.
    radius_squared = torch.zeros((), dtype=torch.float64, device="cpu")
    for dim in spatial:
        size = tensor.shape[dim]
        indices = torch.arange(size, dtype=torch.float64, device="cpu")
        # Divide integer bins directly: fftfreq's reciprocal multiplication
        # can put nominal edges such as 3 / 10 just above the cutoff.
        frequency = torch.where(indices < (size + 1) // 2, indices, indices - size)
        frequency = frequency / size
        shape = [1] * tensor.ndim
        shape[dim] = size
        radius_squared = radius_squared + frequency.reshape(shape).square()
    return radius_squared.sqrt()


class FourierDecomposition(FieldDecomposition):
    """Apply a full ND Fourier transform over channels-last spatial axes.

    Inputs follow ``(B, T, S1, ..., Sd, C)`` with an optional final ensemble
    axis. The transform preserves that layout and transforms spatial axes
    only. It does not transform time, channels or ensemble members.

    Half and bfloat16 inputs are promoted to float32 on the same device before
    the FFT. Single and double precision are preserved, including complex
    inputs. The cast and transform remain differentiable.

    Args:
        representation: Return complex coefficients, their magnitudes or their
            squared magnitudes, under the component name ``fourier``.
        norm: FFT normalization. ``ortho`` scales coefficients by the reciprocal
            square root of the number of spatial cells; ``forward`` uses the
            reciprocal cell count and ``backward`` applies no scaling.
    """

    def __init__(
        self,
        representation: FourierRepresentation = "complex",
        norm: FourierNorm = "ortho",
    ) -> None:
        super().__init__()
        if representation not in ("complex", "magnitude", "power"):
            raise ValueError(f"Unknown Fourier representation {representation!r}.")
        if norm not in ("backward", "forward", "ortho"):
            raise ValueError(f"Unknown Fourier normalization {norm!r}.")
        self.representation = representation
        self.norm = norm

    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> FieldComponents:
        spatial_dims = _spatial_dims(tensor, n_spatial_dims)
        if tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()
        coefficients = torch.fft.fftn(tensor, dim=spatial_dims, norm=self.norm)
        if self.representation == "magnitude":
            coefficients = coefficients.abs()
        elif self.representation == "power":
            coefficients = coefficients.abs().square()
        return {"fourier": coefficients}


class RadialFourierDecomposition(FourierDecomposition):
    """Split complex Fourier coefficients into disjoint radial bands.

    Each ``band_<index>`` retains the full field layout, with coefficients
    outside that band set to zero. This selects coefficients, not a radial
    average of their power. Frequencies beyond the final edge are omitted.
    Spatial axes, FFT normalization and precision follow FourierDecomposition.

    ``band_mean=True`` divides coefficients by the square root of the number
    of modes in their band. Summing their squared magnitudes then yields mean
    band power. This does not give a within-band mean for coefficient CRPS:
    its reduction still includes zeroed positions across the full grid.
    Band weights and the scoring objective determine the final aggregation.
    Empty bands return zero tensors. Memory grows with the number of bands.

    Args:
        edges: Finite, positive, strictly increasing upper edges in cycles per
            grid cell. The first band starts at zero. Lower edges are included;
            upper edges are excluded except for the final edge, which is
            included. A final edge of at least ``sqrt(n_spatial_dims) / 2``
            covers the entire grid, including diagonal Nyquist modes. These
            are grid-relative frequencies, not physical wavenumbers.
        norm: FFT normalization, as in FourierDecomposition.
        band_mean: Divide each nonempty band's coefficients by sqrt(mode count).
    """

    def __init__(
        self,
        edges: Sequence[int | float],
        norm: FourierNorm = "ortho",
        band_mean: bool = False,
    ) -> None:
        super().__init__(norm=norm)
        if not edges:
            msg = "RadialFourierDecomposition requires at least one band edge."
            raise ValueError(msg)
        resolved = [float(edge) for edge in edges]
        if any(not math.isfinite(edge) for edge in resolved):
            msg = f"Band edges must be finite, got {resolved}."
            raise ValueError(msg)
        if resolved[0] <= 0:
            msg = f"Band edges must be positive, got {resolved}."
            raise ValueError(msg)
        if any(later <= earlier for earlier, later in pairwise(resolved)):
            msg = f"Band edges must increase, got {resolved}."
            raise ValueError(msg)

        self.edges = tuple(resolved)
        self.band_mean = bool(band_mean)

    def _radial_frequency(self, tensor: Tensor, spatial: tuple[int, ...]) -> Tensor:
        return _radial_frequency(tensor, spatial)

    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> FieldComponents:
        coefficients = super().forward(tensor, n_spatial_dims=n_spatial_dims)["fourier"]
        spatial = _spatial_dims(tensor, n_spatial_dims)
        radius = self._radial_frequency(tensor, spatial)

        components: FieldComponents = {}
        lower = 0.0
        for index, upper in enumerate(self.edges):
            within = (radius >= lower) & (
                radius <= upper if index == len(self.edges) - 1 else radius < upper
            )
            band = coefficients * within.to(device=coefficients.device)
            if self.band_mean:
                # A coarse grid can leave a band empty.
                count = within.sum().clamp_min(1)
                band = band / count.to(device=band.device, dtype=band.real.dtype).sqrt()
            components[f"band_{index}"] = band
            lower = upper
        return components


class DTCWTDecomposition(FieldDecomposition):
    """Apply a J-level 2D dual-tree complex wavelet transform.

    Each level exposes six complex directional components named
    ``level_<n>/direction_<k>`` in backend orientation order. The optional
    final low-pass component is named ``level_<levels>/lowpass``.

    The filters are registered buffers; move the module to the input device
    and use ``.double()`` for float64 fields. Half/bfloat16 fields are promoted
    to float32 and backend convolutions run outside reduced-precision autocast.
    """

    def __init__(
        self,
        levels: int = 1,
        *,
        biort: str = "near_sym_a",
        qshift: str = "qshift_a",
        include_lowpass: bool = True,
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be positive, got {levels}.")
        self.levels = levels
        self.include_lowpass = include_lowpass
        self.transform = DTCWTForward(
            J=levels,
            biort=biort,
            qshift=qshift,
            mode="symmetric",
        )

    @staticmethod
    def _to_channels_first(tensor: Tensor) -> tuple[Tensor, tuple[int, ...]]:
        if tensor.ndim == 5:
            batch, time = tensor.shape[:2]
            return (
                rearrange(tensor, "b t h w c -> (b t) c h w"),
                (batch, time),
            )
        if tensor.ndim == 6:
            batch, time, ensemble = tensor.shape[0], tensor.shape[1], tensor.shape[-1]
            return (
                rearrange(tensor, "b t h w c m -> (b t m) c h w"),
                (batch, time, ensemble),
            )
        raise ValueError(
            "DTCWT fields must have layout (B, T, H, W, C[, M]); "
            f"got {tuple(tensor.shape)}."
        )

    @staticmethod
    def _restore_layout(tensor: Tensor, layout: tuple[int, ...]) -> Tensor:
        if len(layout) == 2:
            batch, time = layout
            return rearrange(
                tensor,
                "(b t) c h w -> b t h w c",
                b=batch,
                t=time,
            )
        batch, time, ensemble = layout
        return rearrange(
            tensor,
            "(b t m) c h w -> b t h w c m",
            b=batch,
            t=time,
            m=ensemble,
        )

    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> FieldComponents:
        if n_spatial_dims != 2:
            msg = "DTCWT decomposition requires exactly two spatial axes."
            raise ValueError(msg)
        if tensor.is_complex():
            msg = "DTCWT decomposition requires a real-valued field."
            raise ValueError(msg)
        if min(tensor.shape[2:4]) < 2**self.levels:
            raise ValueError(
                f"Spatial axes must have at least {2**self.levels} cells; "
                f"got {tuple(tensor.shape[2:4])}."
            )

        if tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()
        channels_first, layout = self._to_channels_first(tensor)
        with torch.autocast(tensor.device.type, enabled=False):
            lowpass, highpasses = self.transform(channels_first)
        components: FieldComponents = {}
        for level, highpass in enumerate(highpasses, start=1):
            complex_highpass = torch.view_as_complex(highpass.contiguous())
            for direction in range(complex_highpass.shape[2]):
                components[f"level_{level}/direction_{direction}"] = (
                    self._restore_layout(complex_highpass[:, :, direction], layout)
                )
        if self.include_lowpass:
            components[f"level_{self.levels}/lowpass"] = self._restore_layout(
                lowpass, layout
            )
        return components


class WaveletDecomposition(FieldDecomposition):
    """Apply a separable, decimated wavelet transform to spatial axes.

    Detail components are named ``level_<n>/<orientation>`` and the final
    approximation ``level_<levels>/approximation``. Each orientation character
    is ``L`` or ``H`` in spatial-axis order. ``periodic`` uses circular padding,
    not a critically sampled periodization transform.

    Half/bfloat16 inputs are promoted to float32. Convolutions run outside
    reduced-precision autocast because ptwt only supports float32/float64.
    """

    boundary: WaveletBoundary

    def __init__(
        self,
        wavelet: str = "haar",
        levels: int = 1,
        boundary: WaveletBoundary = "reflect",
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be positive, got {levels}.")
        if boundary not in ("reflect", "periodic"):
            raise ValueError(f"Unknown wavelet boundary {boundary!r}.")
        self.wavelet = wavelet
        self.levels = levels
        self.boundary = boundary

    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> FieldComponents:
        spatial_dims = _spatial_dims(tensor, n_spatial_dims)
        spatial_shape = tuple(tensor.shape[dim] for dim in spatial_dims)
        if min(spatial_shape) < 2**self.levels:
            raise ValueError(
                f"Spatial axes must have at least {2**self.levels} cells; "
                f"got {spatial_shape}."
            )

        if tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()
        # A separable transform calls the backend repeatedly; autocast must
        # remain disabled across all axes, not only for the first convolution.
        with torch.autocast(tensor.device.type, enabled=False):
            coefficients: list[Tensor | dict[str, Tensor]]
            if n_spatial_dims == 1:
                coefficients = list(
                    wavedec(
                        tensor,
                        self.wavelet,
                        mode=self.boundary,
                        level=self.levels,
                        axis=spatial_dims[0],
                    )
                )
            elif n_spatial_dims == 2:
                coefficients = cast(
                    list[Tensor | dict[str, Tensor]],
                    fswavedec2(
                        tensor,
                        self.wavelet,
                        mode=self.boundary,
                        level=self.levels,
                        axes=cast(tuple[int, int], spatial_dims),
                    ),
                )
            elif n_spatial_dims == 3:
                coefficients = cast(
                    list[Tensor | dict[str, Tensor]],
                    fswavedec3(
                        tensor,
                        self.wavelet,
                        mode=self.boundary,
                        level=self.levels,
                        axes=cast(tuple[int, int, int], spatial_dims),
                    ),
                )
            else:
                msg = "Wavelet decomposition supports 1--3 spatial axes."
                raise ValueError(msg)

        approximation = cast(Tensor, coefficients[0])
        details = coefficients[1:]
        components: FieldComponents = {}
        for level, bands in enumerate(reversed(details), start=1):
            if isinstance(bands, Tensor):
                components[f"level_{level}/H"] = bands
            else:
                for orientation, band in bands.items():
                    name = orientation.replace("a", "L").replace("d", "H")
                    components[f"level_{level}/{name}"] = band

        components[f"level_{self.levels}/approximation"] = approximation
        return components


class MagnitudeTransform(ComponentTransform):
    """Return the magnitude of real or complex coefficients."""

    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> Tensor:
        del n_spatial_dims
        return tensor.abs()


class PowerTransform(ComponentTransform):
    """Return the squared magnitude of real or complex coefficients."""

    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> Tensor:
        del n_spatial_dims
        return tensor.abs().square()


class RMSAmplitudeTransform(ComponentTransform):
    """Return the root-mean-square amplitude over a component's spatial axes.

    Reduces a component to one value per batch, time and channel, describing
    how much a feature carries without regard to where it lies.

    The spatial axes are kept with extent one, so the result still carries the
    field layout and can be transformed, scored, or joined like any other
    component.
    """

    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> Tensor:
        spatial = _spatial_dims(tensor, n_spatial_dims)
        return _bounded_gradient_sqrt(
            tensor.abs().square().mean(dim=spatial, keepdim=True)
        )


class CoefficientLpLoss(nn.Module):
    """Coefficient-wise ``|prediction - target|**p`` loss.

    At exact agreement, choose a zero gradient, including for fractional
    powers whose derivative is singular there. Forward values are unchanged.
    """

    reduction: Literal["mean", "sum", "none"]

    def __init__(self, p: int | float = 2, reduction: str = "mean") -> None:
        super().__init__()
        if not math.isfinite(p) or p <= 0:
            raise ValueError(f"p must be finite and positive, got {p}.")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unknown reduction {reduction!r}.")
        self.p = float(p)
        self.reduction = cast(Literal["mean", "sum", "none"], reduction)

    def forward(self, preds: FieldTensor, targets: TensorInexactBTSC) -> Tensor:
        error = _distance_power((preds - _align_targets(preds, targets)).abs(), self.p)
        if self.reduction == "none":
            return error
        return getattr(error, self.reduction)()
