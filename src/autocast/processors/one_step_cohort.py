"""ABC for one-step generative processors trained by cohort inflation.

Inflate each input by ``K`` via ``EncodedBatch.repeat`` (repeat-interleave
layout), run one generator forward, reshape to ``(B, K, -1)``, and
delegate the loss body to the subclass.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import ClassVar

import torch
from torch import nn

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor

# A ``shape -> N(0, 1) tensor`` noise source, mirroring
# ``autocast.processors.latent_lowrank.BaseNoise``. When threaded into ``map`` /
# ``_draw_one`` it replaces the fresh ``torch.randn`` cohort seed, so a stateful
# source (the AR(1) ``RolloutNoiseProcess`` rollout calibrator) can make a
# sample-feed rollout's per-step draws temporally correlated. The seed still
# multiplies by ``seed_sigma`` afterwards, so the two compose.
BaseNoise = Callable[[tuple[int, ...]], Tensor]


class OneStepCohortProcessor(Processor[EncodedBatch]):
    """Base class for one-step generative processors with cohort training.

    Subclasses inherit the full forward path (``map`` / ``forward`` /
    ``_draw_one``) and a ``loss`` template that inflates the batch via
    ``EncodedBatch.repeat`` to cohorts of size ``self.n_samples``, runs a
    single generator forward, reshapes to ``(B, K, -1)``, and delegates
    the loss body to the abstract ``_compute_cohort_loss``. The
    ``cohort_spread`` diagnostic (std across cohort members) is recorded
    on ``self._latest_diagnostics`` automatically; subclasses may return
    additional scalar diagnostics from ``_compute_cohort_loss`` to be
    merged in.

    Subclass contract
    -----------------
    - Override ``_compute_cohort_loss`` to return
      ``(loss: Tensor, extra_diagnostics: dict[str, Tensor])``.
    - Optionally override the class attributes ``_MIN_N_SAMPLES`` and
      ``_MIN_N_SAMPLES_REASON`` to tighten the cohort-size floor with an
      explanatory message (Drifting uses ``_MIN_N_SAMPLES = 2`` since
      cohort-of-one collapses the negative-side dual softmax).

    Cohort layout
    -------------
    ``EncodedBatch.repeat`` uses repeat-interleave semantics, so the
    inflated batch is laid out as ``[t_0] * K + [t_1] * K + ...``. After
    the ``reshape(b, K, -1)``, the leading axis indexes the original
    inputs, the middle axis indexes the K cohort members, and the K
    target copies along the middle axis are identical. Subclasses that
    need the unique positive per input slice ``target_b[:, :1, :]``.
    """

    _MIN_N_SAMPLES: ClassVar[int] = 1
    _MIN_N_SAMPLES_REASON: ClassVar[str] = ""

    # Per-channel cohort-seed std, or None for the unit seed. Annotated here so
    # the register_buffer access narrows to a Tensor (not nn.Module's getattr).
    seed_sigma: Tensor | None

    def __init__(
        self,
        *,
        backbone: nn.Module,
        n_steps_output: int,
        n_channels_out: int,
        n_samples: int = 8,
        seed_sigma: Sequence[float] | float | None = None,
    ) -> None:
        """Build a OneStepCohortProcessor.

        Args:
            backbone: One-step generator backbone. MUST be constructed
                with ``include_time_embedding=False`` (one-step
                processors have no integration time ``t`` to embed).
            n_steps_output: Number of output time steps per sample.
                Required; configs pass ``auto`` and let ``setup.py``
                resolve from the datamodule's output shape.
            n_channels_out: Number of output channels per sample.
                Required; resolved via ``auto`` as for ``n_steps_output``.
            n_samples: Cohort size per input. Must be at least
                ``cls._MIN_N_SAMPLES``.
            seed_sigma: Optional per-channel (or scalar) standard deviation
                that scales the white-Gaussian cohort seed in ``_draw_one``.
                ``None`` (the default) keeps the unit-variance seed — byte-for-
                byte the original behaviour. A scalar broadcasts to every
                channel; a sequence must have length ``n_channels_out`` (the
                last seed axis). This is the prescribed-noise injector of the
                residual/two-stage Drifting (Approach #2): the seed *multiplies*
                by sigma (never divides), so ``sigma_c = 0`` is safe and yields an
                identical-member channel (no uncertainty to represent). Stored
                as a non-persistent buffer, so it follows ``.to(device)`` but is
                taken from config (a fixed hyperparameter / stats artifact),
                never read from or written to the checkpoint.
        """
        super().__init__()

        include_te = getattr(backbone, "include_time_embedding", False)
        if include_te is not False:
            msg = (
                f"{type(self).__name__} requires a backbone built with "
                "include_time_embedding=False (one-step generator; no "
                "integration time to embed). Got "
                f"include_time_embedding={include_te!r}."
            )
            raise ValueError(msg)

        if n_samples < self._MIN_N_SAMPLES:
            reason = self._MIN_N_SAMPLES_REASON
            tail = f"; {reason}." if reason else "."
            msg = (
                f"{type(self).__name__} requires n_samples >= "
                f"{self._MIN_N_SAMPLES} (got {n_samples}){tail}"
            )
            raise ValueError(msg)

        self.generator = backbone
        self.n_steps_output = n_steps_output
        self.n_channels_out = n_channels_out
        self.n_samples = n_samples
        self._register_seed_sigma(seed_sigma)
        # Stash for diagnostics; the LightningModule pulls these from
        # `self._latest_diagnostics` and forwards them through
        # `self.log_dict` after each training/validation step.
        self._latest_diagnostics: dict[str, Tensor] = {}

    def _register_seed_sigma(self, seed_sigma: Sequence[float] | float | None) -> None:
        """Validate and register the cohort-seed standard deviation.

        Registered as a NON-persistent buffer (``persistent=False``): sigma is a
        fixed hyperparameter supplied by config / a stats artifact, so it must
        follow ``.to(device)`` but stay out of the checkpoint — keeping old
        checkpoints (saved before this option existed) loadable under
        ``strict=True``. ``None`` registers a ``None`` buffer (unit seed).
        """
        if seed_sigma is None:
            self.register_buffer("seed_sigma", None, persistent=False)
            return
        sigma = torch.as_tensor(seed_sigma, dtype=torch.float32).reshape(-1)
        # NaN/inf compare False against 0, so finiteness needs its own check.
        if not bool(torch.isfinite(sigma).all()) or bool((sigma < 0.0).any()):
            msg = (
                f"{type(self).__name__} requires every seed_sigma entry finite "
                f"and >= 0 (the seed multiplies by sigma; got {sigma.tolist()})."
            )
            raise ValueError(msg)
        if sigma.numel() not in (1, self.n_channels_out):
            msg = (
                f"{type(self).__name__} requires seed_sigma to be a scalar or "
                f"length n_channels_out={self.n_channels_out} (got "
                f"{sigma.numel()} entries)."
            )
            raise ValueError(msg)
        self.register_buffer("seed_sigma", sigma, persistent=False)

    def generator_func(
        self,
        z: Tensor,
        x: Tensor,
        global_cond: Tensor | None = None,
        loss_weight: Tensor | None = None,
    ) -> Tensor:
        """Single forward pass through the one-step generator.

        Annotated as ``Tensor`` since one-step generators can be invoked
        in either ambient or latent space; the backbone is responsible
        for enforcing its own shape contract. ``loss_weight`` is the optional
        per-sample loss-mixing weight for lambda-conditioned generators;
        backbones built without ``include_loss_weight`` ignore it.
        """
        return self.generator(
            z, t=None, cond=x, global_cond=global_cond, loss_weight=loss_weight
        )

    def forward(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Alias to ``map`` for Lightning/PyTorch compatibility."""
        return self.map(x, global_cond)

    def map(
        self,
        x: Tensor,
        global_cond: Tensor | None,
        base_noise: BaseNoise | None = None,
    ) -> Tensor:
        """Map inputs to outputs via a single one-step generation.

        One-step generators run in either ambient or latent space, so the
        shape contract for ``x`` is left to the backbone. Ensembles are
        produced by ``ProcessorModelEnsemble`` at the model level, not here.
        ``base_noise`` (default ``None`` = fresh ``torch.randn``) lets a
        stateful AR(1) source drive a sample-feed rollout (the C calibrator).
        """
        return self._draw_one(x, global_cond, base_noise=base_noise)

    def _draw_one(
        self,
        x: Tensor,
        global_cond: Tensor | None,
        loss_weight: Tensor | None = None,
        base_noise: BaseNoise | None = None,
    ) -> Tensor:
        """Sample one cohort-of-one output from noise.

        ``loss_weight`` is forwarded to the generator for lambda-conditioned
        models; ``None`` (the default) leaves the one-step path unchanged.
        ``base_noise`` (default ``None`` = fresh ``torch.randn``) replaces the
        white cohort seed with a ``shape -> N(0, 1)`` source, mirroring the
        low-rank ``sample_packed`` hook; the ``seed_sigma`` scaling still applies
        afterwards, so the two compose. ``None`` is byte-identical to the
        original draw under the same generator.
        """
        z_shape = (
            x.shape[0],
            self.n_steps_output,
            *tuple(x.shape[2:-1]),
            self.n_channels_out,
        )
        if base_noise is None:
            eps = torch.randn(z_shape, device=x.device, dtype=x.dtype)
        else:
            eps = base_noise(z_shape).to(device=x.device, dtype=x.dtype)
        if self.seed_sigma is not None:
            # Per-channel (or scalar) scaling of the white seed; seed_sigma
            # broadcasts against the trailing channel axis of z_shape. The
            # prescribed-noise injector of residual Drifting (Approach #2).
            eps = eps * self.seed_sigma.to(eps.dtype)
        return self.generator_func(eps, x, global_cond, loss_weight=loss_weight)

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Compute the cohort training loss for a batch.

        Inflates the batch to cohorts of size ``self.n_samples`` per
        input, runs a single forward pass through the generator,
        reshapes to ``(B, K, -1)``, and delegates the loss body to the
        subclass's ``_compute_cohort_loss``. Records the
        ``cohort_spread`` diagnostic on ``self._latest_diagnostics``,
        merged with any extras returned by the subclass.
        """
        # Clear up front so a raise in _compute_cohort_loss can't leave
        # stale values from the previous step for a downstream consumer
        # to pick up.
        self._latest_diagnostics = {}

        batch_n = batch.repeat(self.n_samples)
        cond = batch_n.encoded_inputs
        target = batch_n.encoded_output_fields
        b = cond.shape[0] // self.n_samples

        gen = self._draw_one(cond, batch_n.global_cond)
        gen_b = gen.reshape(b, self.n_samples, -1)
        target_b = target.reshape(b, self.n_samples, -1)

        loss, extra_diagnostics = self._compute_cohort_loss(gen_b, target_b)

        # correction=0 (biased std) keeps cohort_spread finite at
        # n_samples=1 (the MSECohort floor); unbiased std would yield
        # NaN and corrupt the W&B metric series.
        diagnostics: dict[str, Tensor] = {
            "cohort_spread": gen_b.detach().std(dim=1, correction=0).mean(),
        }
        diagnostics.update(extra_diagnostics)
        self._latest_diagnostics = diagnostics

        return loss

    @abstractmethod
    def _compute_cohort_loss(
        self,
        gen_b: Tensor,
        target_b: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the loss body and any extra scalar diagnostics.

        Args:
            gen_b: Cohort-flattened generator output of shape
                ``(B, K, prod(T_out * spatial * C_out))``. Carries the
                live gradient through the backbone.
            target_b: Cohort-flattened target of shape
                ``(B, K, prod(T_out * spatial * C_out))``. With
                repeat-interleave layout the K target copies along the
                middle axis are identical; subclasses that need the
                unique positive use ``target_b[:, :1, :]``.

        Returns:
        -------
            ``(loss, extra_diagnostics)`` — a scalar loss tensor and a
            (possibly empty) dict of scalar tensors merged into
            ``self._latest_diagnostics`` alongside ``cohort_spread``.
        """
        ...
