"""Device selection that has to happen before anything else touches CUDA."""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)


def pin_local_cuda_device() -> int | None:
    """Make this process's own GPU the current CUDA device, before anything uses CUDA.

    Lightning selects each DDP rank's GPU only when ``Trainer.fit`` starts. Until
    then the current device is ``cuda:0``, so any earlier CUDA work in ranks 1-3
    lands on GPU 0. That is harmless when GPUs accept several processes, but under
    ``Exclusive_Process`` compute mode (Isambard-AI since its October 2026
    maintenance) the driver refuses those ranks and every multi-GPU run dies at
    start-up with "CUDA-capable device(s) is/are busy or unavailable".

    The local rank is read from ``LOCAL_RANK`` (Lightning's spawned ranks,
    torchrun), else ``SLURM_LOCALID`` (srun tasks). Nothing is done without a
    local rank, without CUDA, or when the rank's index is not visible (one GPU per
    task: that GPU is already ``cuda:0``).

    Returns:
        The device index selected, or ``None`` when nothing was done.
    """
    raw = os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID")
    if raw is None or not torch.cuda.is_available():
        return None
    local_rank = int(raw)
    if local_rank >= torch.cuda.device_count():
        return None
    torch.cuda.set_device(local_rank)
    log.info(
        "Selected cuda:%d for local rank %d before any CUDA work",
        local_rank,
        local_rank,
    )
    return local_rank
