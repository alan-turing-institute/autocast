"""
Advection-diffusion simulator with multi-channel outputs.

Returns channels: [vorticity, u, v, streamfunction].

Implementation uses spectral Poisson solve (FFT) for streamfunction and central finite
differences (via numpy.roll) for spatial derivatives.

"""

from __future__ import annotations

import numpy as np
import torch
from autoemulate.core.types import NumpyLike, TensorLike
from autoemulate.simulations.base import Simulator
from scipy.integrate import solve_ivp

# Integrator settings
integrator_keywords = {"rtol": 1e-6, "atol": 1e-8, "method": "RK45"}


class AdvectionDiffusion(Simulator):
    r"""Differentiable advection-diffusion simulator exposing multi-channel outputs.

    Parameters
    ----------
    parameters_range: dict[str, tuple[float, float]], optional
        Bounds on the sampled viscosity (`nu`) and advection strength (`mu`).
    output_names: list[str], optional
        Human-readable names for the returned channels.
    return_timeseries: bool, default=False
        Whether `forward` returns the entire trajectory instead of a single snapshot.
    log_level: str, default="progress_bar"
        Logging verbosity passed to the base `Simulator`.
    n: int, default=50
        Number of spatial points per dimension.
    L: float, default=10.0
        Domain length in each spatial direction.
    T: float, default=80.0
        Total integration time.
    dt: float, default=0.25
        Temporal resolution used for the ODE solver outputs.
    integrator_kwargs: dict, optional
        Extra keyword arguments forwarded to `scipy.integrate.solve_ivp`.

    Notes
    -----
    Each grid point emits four channels `[vorticity, u, v, streamfunction]`.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        n: int = 50,
        L: float = 10.0,
        T: float = 80.0,
        dt: float = 0.25,
        integrator_kwargs: dict | None = None,
    ) -> None:
        if parameters_range is None:
            parameters_range = {"nu": (0.0001, 0.01), "mu": (0.5, 2.0)}
        if output_names is None:
            output_names = ["vorticity", "u", "v", "streamfunction"]

        super().__init__(parameters_range, output_names, log_level)

        self.return_timeseries = return_timeseries
        self.n = n
        self.L = L
        self.T = T
        self.dt = dt
        self.integrator_kwargs = {**integrator_keywords, **(integrator_kwargs or {})}

    def _forward(self, x: TensorLike) -> TensorLike:
        # Expect single input sample in batch
        assert x.shape[0] == 1, (
            f"Simulator._forward expects a single input, got {x.shape[0]}"
        )

        # x contains the physical parameters [nu, mu]
        sample = x.cpu().numpy()[0]

        sol = simulate_advection_diffusion(
            sample,
            self.return_timeseries,
            self.n,
            self.L,
            self.T,
            self.dt,
            self.integrator_kwargs,
        )

        # sol shape: (nt, n, n, channels) if return_timeseries else (n, n, channels)
        arr = np.asarray(sol, dtype=np.float32)

        # Flatten to (1, -1) for compatibility with the superclass
        return torch.from_numpy(arr.ravel()).reshape(1, -1)

    def forward_samples_spatiotemporal(
        self, n: int, random_seed: int | None = None
    ) -> dict:
        """Produce simulator rollouts along with the sampled parameters.

        Parameters
        ----------
        n: int
            Number of trajectories to sample.
        random_seed: int, optional
            Seed for reproducible parameter draws.

        Returns
        -------
        dict
            Dictionary with keys:
            ``data``
                Tensor of shape ``(batch, time, n, n, channels)`` if
                `return_timeseries` is ``True`` or ``(batch, 1, n, n, channels)``
                otherwise. Channels follow `[vorticity, u, v, streamfunction]`.
            ``constant_scalars``
                Sampled `[nu, mu]` parameters.
            ``constant_fields``
                Placeholder for future field inputs; always ``None`` here.
        """
        x = self.sample_inputs(n, random_seed)

        y, x = self.forward_batch(x)

        channels = 4
        features_per_step = self.n * self.n * channels
        if self.return_timeseries:
            total_features = y.shape[1]
            if total_features % features_per_step != 0:
                raise RuntimeError(
                    "Returned tensor does not align with n*n*channels; "
                    f"received {total_features} features, expected multiples of "
                    f"{features_per_step}."
                )
            n_time = total_features // features_per_step
            y_reshaped = y.reshape(y.shape[0], n_time, self.n, self.n, channels)
        else:
            if y.shape[1] != features_per_step:
                raise RuntimeError(
                    "Unexpected flattened size for single snapshot; "
                    f"received {y.shape[1]}, expected {features_per_step}."
                )
            y_reshaped = y.reshape(y.shape[0], 1, self.n, self.n, channels)

        return {
            "data": y_reshaped,
            "constant_scalars": x,
            "constant_fields": None,
        }


def _spectral_poisson_solver(w2d: np.ndarray, K3: np.ndarray) -> np.ndarray:
    r"""Solve ``laplacian(psi) = -omega`` in Fourier space.

    Parameters
    ----------
    w2d: np.ndarray
        Two-dimensional vorticity field with shape ``(n, n)``.
    K3: np.ndarray
        Pre-computed spectral multiplier ``1 / (k_x^2 + k_y^2)`` with the zero mode
        handled to keep the mean streamfunction at zero.

    Returns
    -------
    np.ndarray
        Real-valued streamfunction with shape ``(n, n)``.
    """
    psi_hat = np.fft.fft2(w2d) * K3
    psi = np.real(np.fft.ifft2(psi_hat))
    return psi  # noqa: RET504


def _laplacian_periodic(w2d: np.ndarray, dx: float) -> np.ndarray:
    """Apply the periodic Laplacian using second-order central differences.

    Parameters
    ----------
    w2d: np.ndarray
        Input field of shape ``(n, n)``.
    dx: float
        Grid spacing.

    Returns
    -------
    np.ndarray
        Field after applying the Laplacian, matching the input shape.
    """
    return (np.roll(w2d, -1, axis=0) - 2 * w2d + np.roll(w2d, 1, axis=0)) / dx**2 + (
        np.roll(w2d, -1, axis=1) - 2 * w2d + np.roll(w2d, 1, axis=1)
    ) / dx**2


def _gradient_periodic(f2d: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute periodic central-difference gradients.

    Parameters
    ----------
    f2d: np.ndarray
        Scalar field of shape ``(n, n)``.
    dx: float
        Grid spacing.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(df/dx, df/dy)`` each with shape ``(n, n)``.
    """
    dfdx = (np.roll(f2d, -1, axis=1) - np.roll(f2d, 1, axis=1)) / (2 * dx)
    dfdy = (np.roll(f2d, -1, axis=0) - np.roll(f2d, 1, axis=0)) / (2 * dx)
    return dfdx, dfdy


def advection_diffusion_rhs(
    _t: float,
    w_flat: np.ndarray,
    n: int,
    dx: float,
    nu: float,
    mu: float,
    K3: np.ndarray,
) -> np.ndarray:
    r"""Evaluate the vorticity time derivative.

    Parameters
    ----------
    _t: float
        Time (ignored, but required by `solve_ivp`).
    w_flat: np.ndarray
        Flattened vorticity field of length ``n * n``.
    n: int
        Grid resolution per spatial axis.
    dx: float
        Grid spacing.
    nu: float
        Diffusion coefficient.
    mu: float
        Advection strength.
    K3: np.ndarray
        Spectral multiplier used for the Poisson solve.

    Returns
    -------
    np.ndarray
        Flattened ``dw/dt`` matching the shape of ``w_flat``.

    Notes
    -----
    Implements ``dw/dt = nu * laplacian(w) - mu * (u * d/dx + v * d/dy)`` where
    ``(u, v)`` is recovered from the streamfunction via ``u = dpsi/dy`` and
    ``v = -dpsi/dx``.
    """
    w2d = w_flat.reshape(n, n)

    # Streamfunction via spectral Poisson solver
    psi = _spectral_poisson_solver(w2d, K3)

    # Velocity field from psi
    # u = dpsi/dy, v = -dpsi/dx
    dpsidx, dpsidy = _gradient_periodic(psi, dx)  # dpsi/dx, dpsi/dy
    u = dpsidy
    v = -dpsidx

    # Gradients of vorticity
    dw_dx, dw_dy = _gradient_periodic(w2d, dx)

    advec = u * dw_dx + v * dw_dy

    diff = _laplacian_periodic(w2d, dx)

    dwdt = nu * diff - mu * advec

    return dwdt.ravel()


def simulate_advection_diffusion(
    x: NumpyLike,
    return_timeseries: bool = False,
    n: int = 50,
    L: float = 10.0,
    T: float = 80.0,
    dt: float = 0.25,
    integrator_kwargs: dict | None = None,
) -> NumpyLike:
    """Integrate the advection-diffusion system and emit physical channels.

    Parameters
    ----------
    x: array-like
        Two-element vector ``[nu, mu]`` with viscosity and advection parameters.
    return_timeseries: bool, default=False
        If ``True``, return the entire trajectory; otherwise only the terminal state.
    n: int, default=50
        Grid resolution per spatial dimension.
    L: float, default=10.0
        Domain size along each axis.
    T: float, default=80.0
        End time for integration.
    dt: float, default=0.25
        Step between recorded solver outputs.
    integrator_kwargs: dict, optional
        Extra keyword arguments forwarded to `solve_ivp`.

    Returns
    -------
    np.ndarray
        If `return_timeseries` is ``True``, an array of shape ``(n_time, n, n, 4)``;
        otherwise shape ``(n, n, 4)``. Channels are ordered
        ``[vorticity, u, v, streamfunction]``.
    """
    nu, mu = float(x[0]), float(x[1])

    tspan = np.arange(0.0, T + 1e-12, dt)
    n_time = len(tspan)

    x_grid = np.linspace(-L / 2, L / 2, n, endpoint=False)
    dx = float(x_grid[1] - x_grid[0])

    # Initial condition: Gaussian vortex (centered)
    X, Y = np.meshgrid(x_grid, x_grid)
    w_initial = np.exp(-((X**2) + (Y**2) / 20.0))

    # Precompute spectral operator for Poisson solve: 1/(k^2)
    k = (2.0 * np.pi / L) * np.concatenate(
        [np.arange(0, n // 2), np.arange(-n // 2, 0)]
    )
    KX, KY = np.meshgrid(k, k)
    denom = KX**2 + KY**2
    # Avoid division by zero at zero frequency: set to a large number then zero psi mean
    denom[0, 0] = 1.0
    K3 = 1.0 / denom
    K3[0, 0] = 0.0

    w0_flat = w_initial.ravel()

    # Integrate
    ode_kwargs = {**integrator_keywords, **(integrator_kwargs or {})}

    sol = solve_ivp(
        fun=lambda t, w: advection_diffusion_rhs(t, w, n, dx, nu, mu, K3),
        t_span=(0.0, T),
        y0=w0_flat,
        t_eval=tspan,
        **ode_kwargs,
    )

    if not sol.success:
        raise RuntimeError("ODE solver failed: " + str(sol.message))

    if return_timeseries:
        # sol.y shape: (N, n_time)
        w_ts = sol.y.T.reshape(n_time, n, n).astype(np.float32, copy=False)

        channels = np.empty((n_time, n, n, 4), dtype=np.float32)
        for ti, w2d in enumerate(w_ts):
            psi = _spectral_poisson_solver(w2d, K3)
            dpsidx, dpsidy = _gradient_periodic(psi, dx)
            u = dpsidy
            v = -dpsidx

            channels[ti] = np.stack([w2d, u, v, psi], axis=-1).astype(
                np.float32, copy=False
            )

        return channels

    w_final = sol.y[:, -1].reshape(n, n)
    psi = _spectral_poisson_solver(w_final, K3)
    dpsidx, dpsidy = _gradient_periodic(psi, dx)
    u = dpsidy
    v = -dpsidx

    return np.stack([w_final, u, v, psi], axis=-1).astype(np.float32, copy=False)
