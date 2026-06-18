"""Per-rank GPU-utilization logging for multi-node sanity checks."""

import logging
import os
import socket
import statistics
import threading

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)


class GpuUtilizationLogCallback(Callback):
    """Sample the local rank's GPU utilization and log a per-epoch distribution.

    W&B's built-in system metrics are only collected on the global-rank-0 node,
    so a multi-node run shows utilization for just that node's GPUs. This
    callback closes that gap: every DDP rank samples ``torch.cuda.utilization``
    for *its own* device and logs one summary line **per epoch**, tagged with
    the global rank and host. Because all ranks log independently into the
    shared SLURM stdout, the combined log shows utilization for every GPU across
    every node, broken down by epoch.

    Sampling runs on a background thread at a fixed *wall-clock* cadence rather
    than on training-loop hooks. Loop hooks (e.g. ``on_train_batch_end``) fire
    immediately after a batch's compute, which biases the reading high and hides
    the idle gaps -- such as data-loading stalls -- that a smoke test wants to
    surface. Wall-clock sampling captures the true duty cycle, including those
    gaps. The summary reports percentiles and busy/idle fractions, not just a
    mean, which a few outliers could skew.

    Epoch boundaries follow ``TrainingTimerCallback``: a bucket spans one
    ``on_train_epoch_start`` to the next (so it includes that epoch's training
    *and* its validation loop), and the final epoch is closed out in
    ``on_train_end``.

    ``torch.cuda.utilization`` is an NVML counter read by device index: it
    launches no GPU kernels, needs no CUDA context in the sampling thread, and
    makes no collective calls -- so it neither perturbs the measurement nor
    risks a DDP hang.
    """

    def __init__(
        self,
        sample_interval_s: float = 0.5,
        busy_threshold: int = 80,
        idle_threshold: int = 10,
    ) -> None:
        self.sample_interval_s = max(0.05, float(sample_interval_s))
        self.busy_threshold = int(busy_threshold)
        self.idle_threshold = int(idle_threshold)
        self._enabled = False
        self._device: int | None = None
        self._epoch_index: int | None = None
        self._lock = threading.Lock()
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_loop(self) -> None:
        warned = False
        while not self._stop.is_set():
            try:
                util = torch.cuda.utilization(self._device)
            except Exception as exc:  # e.g. missing pynvml or NVML init failure.
                # Log the reason once so an empty summary is self-explaining
                # rather than a silent "no samples collected".
                if not warned:
                    log.warning(
                        "GpuUtilizationLogCallback[host=%s cuda:%s]: sampling "
                        "disabled, torch.cuda.utilization failed: %s: %s",
                        socket.gethostname(),
                        self._device,
                        type(exc).__name__,
                        exc,
                    )
                    warned = True
            else:
                with self._lock:
                    self._samples.append(int(util))
            # Returns early if stop is set, so shutdown is prompt.
            self._stop.wait(self.sample_interval_s)

    def _drain_samples(self) -> list[int]:
        """Atomically take the samples buffered so far and reset it."""
        with self._lock:
            samples = self._samples
            self._samples = []
        return samples

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del trainer, pl_module
        if not torch.cuda.is_available():
            log.info(
                "GpuUtilizationLogCallback[host=%s]: no CUDA device; not sampling",
                socket.gethostname(),
            )
            return
        self._enabled = True
        self._device = torch.cuda.current_device()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sample_loop, name="gpu-util-sampler", daemon=True
        )
        self._thread.start()

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        del pl_module
        if not self._enabled:
            return
        # Close out the previous epoch (its training + validation cycle).
        if self._epoch_index is not None:
            self._log_epoch(trainer, self._epoch_index)
        self._epoch_index = getattr(trainer, "current_epoch", 0)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        if not self._enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.sample_interval_s + 2.0)
        # Close out the final epoch.
        if self._epoch_index is not None:
            self._log_epoch(trainer, self._epoch_index)

    def _log_epoch(self, trainer: L.Trainer, epoch: int) -> None:
        rank = getattr(trainer, "global_rank", 0)
        local_rank = os.environ.get("LOCAL_RANK", "?")
        host = socket.gethostname()
        device = self._device if self._device is not None else 0
        ordered = sorted(self._drain_samples())
        if not ordered:
            log.info(
                "GpuUtilizationLogCallback[epoch=%d global_rank=%s local_rank=%s "
                "host=%s cuda:%d]: no samples collected",
                epoch,
                rank,
                local_rank,
                host,
                device,
            )
            return

        n = len(ordered)
        mean = sum(ordered) / n
        p50 = statistics.median(ordered)
        if n >= 2:
            # "inclusive" keeps results within [min, max] (no >100% overshoot).
            deciles = statistics.quantiles(ordered, n=10, method="inclusive")
            p10, p90 = deciles[0], deciles[8]
        else:
            p10 = p90 = float(ordered[0])
        busy_frac = 100.0 * sum(v >= self.busy_threshold for v in ordered) / n
        idle_frac = 100.0 * sum(v < self.idle_threshold for v in ordered) / n
        # Coarse shape: sample counts per 10% utilization bucket (0-10, .., 90-100).
        hist = [0] * 10
        for v in ordered:
            hist[min(v // 10, 9)] += 1

        log.info(
            "GpuUtilizationLogCallback[epoch=%d global_rank=%s local_rank=%s "
            "host=%s cuda:%d]: samples=%d mean=%.1f%% p10=%.0f%% p50=%.0f%% "
            "p90=%.0f%% min=%d%% max=%d%% busy(>=%d%%)=%.1f%% idle(<%d%%)=%.1f%% "
            "deciles[0-100]=%s",
            epoch,
            rank,
            local_rank,
            host,
            device,
            n,
            mean,
            p10,
            p50,
            p90,
            min(ordered),
            max(ordered),
            self.busy_threshold,
            busy_frac,
            self.idle_threshold,
            idle_frac,
            ",".join(str(c) for c in hist),
        )
