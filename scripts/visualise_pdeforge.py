#!/usr/bin/env python
"""Generate PDEForge trajectories and save autocast videos.

Creates full spatio-temporal trajectories for three 2-D PDEs
using PDEForge, converts them to autocast format, and renders
videos via ``plot_spatiotemporal_video``.

Models
------
- wave_2d
- stochastic_heat_2d
- fitzhugh_nagumo_2d

Usage
-----
    python scripts/visualise_pdeforge.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pdeforge import get_model

from autocast.utils import plot_spatiotemporal_video

# ── Configuration ──────────────────────────────────────────────

OUTPUT_DIR = Path("outputs/pdeforge_videos")
N_TRAJECTORIES = 4  # number of different ICs per model
N_TIME_STEPS = 50  # temporal resolution
RESOLUTION = {"x": 64, "y": 64}
SEED = 42
FPS = 5

MODELS = {
    "wave_2d": {
        "cls_name": "wave_2d",
        "params": {
            "resolution": RESOLUTION,
            "time_horizon": 1.0,
            "_n_time_steps": N_TIME_STEPS,
        },
        "title": "Wave",
    },
    "fitzhugh_nagumo_2d": {
        "cls_name": "fitzhugh_nagumo_2d",
        "params": {
            "resolution": RESOLUTION,
            "time_horizon": 1.0,
            "_n_time_steps": N_TIME_STEPS,
        },
        "title": "FitzHugh-Nagumo",
    },
}


# ── Utils ───────────────────────────────────────────────────────


def generate_trajectories(
    model,
    n_trajectories: int,
    *,
    seed: int = 42,
    verbose: bool = True,
) -> torch.Tensor:
    """Generate spatio-temporal trajectories.

    Wraps ``model.generate_ic`` /
    ``model.solve(ic, return_full=True)`` in a loop
    and returns a single tensor compatible with
    autocast's ``SpatioTemporalDataset``.

    ``model.solve(ic, return_full=True)`` is expected to
    return the **full** trajectory including the initial
    condition at ``t = 0`` as the first frame — no
    additional prepending is performed.

    Parameters
    ----------
    model
        An instantiated PDEForge model (e.g. from
        ``get_model("heat_2d")(...)``).  Must expose
        ``generate_ic(seed=...)`` and
        ``solve(ic, return_full=True)``.
    n_trajectories : int
        Number of PDE problems to generate, each with a
        different random initial condition.
    seed : int
        Base random seed.  Trajectory *i* uses
        ``seed + i``.
    verbose : bool
        Print progress information.

    Returns
    -------
    torch.Tensor
        Shape ``[N, T, W, H, C]``:

        - **N** — number of trajectories
        - **T** — time steps (including ``t=0``)
        - **W, H** — spatial grid
        - **C** — solution channels (1 for scalar PDEs)
    """
    trajectories: list[np.ndarray] = []

    for i in range(n_trajectories):
        ic = model.generate_ic(seed=seed + i)
        traj = model.solve(ic, return_full=True)
        # Expected shapes for 2-D models:
        #   scalar:        (n_t, nx, ny)
        #   multi-channel: (n_t, nx, ny, C)
        trajectories.append(traj)

        if verbose and (i + 1) % max(1, n_trajectories // 10) == 0:
            print(f"  Generated {i + 1}" f"/{n_trajectories}")

    # Stack along a new leading axis → (N, n_t, ...)
    stacked = np.stack(trajectories, axis=0)

    # Ensure channel dimension exists:
    # (N, T, W, H) → (N, T, W, H, 1)
    if stacked.ndim == 4:
        stacked = stacked[..., np.newaxis]

    if verbose:
        print(f"  Result: {list(stacked.shape)}" f" [N, T, W, H, C]")

    return torch.from_numpy(stacked).float()


# ── Main ───────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, cfg in MODELS.items():
        print(f"\n{'=' * 50}")
        print(f"  {cfg['title']}")
        print(f"{'=' * 50}")

        # 1. Instantiate PDEForge model
        ModelCls = get_model(cfg["cls_name"])
        model = ModelCls(**cfg["params"])

        # 2. Generate trajectories → [N, T, W, H, C]
        data = generate_trajectories(
            model,
            n_trajectories=N_TRAJECTORIES,
            seed=SEED,
        )
        print(f"  Tensor shape: {list(data.shape)}  " f"[N, T, W, H, C]")

        # 3. Render one video per trajectory
        for traj_idx in range(min(N_TRAJECTORIES, data.shape[0])):
            save_path = OUTPUT_DIR / f"{name}_traj{traj_idx}.mp4"
            print(f"  Rendering trajectory {traj_idx} " f"→ {save_path}")

            # plot_spatiotemporal_video expects (B, T, W, H, C)
            # Pass the same tensor as both true and pred
            plot_spatiotemporal_video(
                true=data,
                pred=data,
                batch_idx=traj_idx,
                fps=FPS,
                save_path=str(save_path),
                title=f"{cfg['title']}",
                colorbar_mode="column",
            )

    print(f"\nAll videos saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
