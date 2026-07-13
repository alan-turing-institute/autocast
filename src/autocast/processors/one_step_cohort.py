"""ABC for one-step generative processors trained by cohort inflation.

Inflate each input by ``K`` via ``EncodedBatch.repeat`` (repeat-interleave
layout), run one generator forward, reshape to ``(B, K, -1)``, and delegate
the loss body to the subclass.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import torch
from einops import rearrange
from torch import nn

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor, TensorBTSC


class OneStepCohortProcessor(Processor[EncodedBatch]):
    """Base class for one-step generative processors with cohort training.

    Subclasses inherit the full forward path (``map`` / ``forward`` /
    ``_draw_one``) and a ``loss`` template that inflates the batch via
    ``EncodedBatch.repeat`` to cohorts of size ``self.n_samples``, runs a
    single generator forward, reshapes to ``(B, K, -1)``, and delegates the
    loss body to the abstract ``_compute_cohort_loss``. The ``cohort_spread``
    diagnostic (std across cohort members) is recorded on
    ``self._latest_diagnostics`` automatically; subclasses may return
    additional scalar diagnostics from ``_compute_cohort_loss`` to be merged
    in.

    Subclass contract: override ``_compute_cohort_loss`` to return
    ``(loss: Tensor, extra_diagnostics: dict[str, Tensor])``. Optionally
    override the class attributes ``_MIN_N_SAMPLES`` and
    ``_MIN_N_SAMPLES_REASON`` to tighten the cohort-size floor with an
    explanatory message (Drifting uses ``_MIN_N_SAMPLES = 2`` since a
    cohort-of-one collapses the negative-side dual softmax).

    Note:
        Cohort layout: ``EncodedBatch.repeat`` uses repeat-interleave
        semantics, so the inflated batch is laid out as
        ``[t_0] * K + [t_1] * K + ...``. After the ``reshape(b, K, -1)``, the
        leading axis indexes the original inputs, the middle axis indexes
        the K cohort members, and the K target copies along the middle axis
        are identical. Subclasses that need the unique positive per input
        slice ``target_b[:, :1, :]``.
    """

    _MIN_N_SAMPLES: ClassVar[int] = 1
    _MIN_N_SAMPLES_REASON: ClassVar[str] = ""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        n_steps_output: int,
        n_channels_out: int,
        n_samples: int = 8,
    ) -> None:
        """Build a OneStepCohortProcessor.

        Args:
            backbone: One-step generator backbone. MUST be constructed with
                ``include_time_embedding=False`` (one-step processors have
                no integration time ``t`` to embed).
            n_steps_output: Number of output time steps per sample. Required;
                configs pass ``auto`` and let ``setup.py`` resolve from the
                datamodule's output shape.
            n_channels_out: Number of output channels per sample. Required;
                resolved via ``auto`` as for ``n_steps_output``.
            n_samples: Cohort size per input. Must be at least
                ``cls._MIN_N_SAMPLES``.
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
        # Stash for diagnostics; the LightningModule pulls these from
        # `self._latest_diagnostics` and forwards them through
        # `self.log_dict` after each training/validation step.
        self._latest_diagnostics: dict[str, Tensor] = {}

    def generator_func(
        self,
        z: Tensor,
        x: Tensor,
        global_cond: Tensor | None = None,
    ) -> Tensor:
        """Single forward pass through the one-step generator.

        Annotated as ``Tensor`` since one-step generators can be invoked in
        either ambient or latent space; the backbone is responsible for
        enforcing its own shape contract.

        Args:
            z: Noise tensor to map through the generator.
            x: Conditioning inputs.
            global_cond: Optional non-spatial conditioning/modulation tensor.

        Returns:
            Generator output with the same shape as `z`.
        """
        return self.generator(z, t=None, cond=x, global_cond=global_cond)

    def forward(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Alias to ``map`` for Lightning/PyTorch compatibility."""
        return self.map(x, global_cond)

    def map(
        self,
        x: TensorBTSC,
        global_cond: Tensor | None,
        n_samples: int | None = None,
    ) -> Tensor:
        """Map inputs to outputs via one-step generation.

        Args:
            x: Conditioning inputs of shape ``(B, T_in, *spatial, C_in)``.
            global_cond: Optional non-spatial conditioning vector.
            n_samples: If set, produce ``K = n_samples`` independent draws
                per input by inflating the batch via ``repeat_interleave``.
                The output gains a trailing ensemble axis. If ``None``
                (default), one sample. The K draws are independent noise
                samples through the one-step generator; they do NOT share
                the training-time cohort structure that the subclass's loss
                imposes during training. See also
                ``ProcessorModelEnsemble.forward``, which applies the same
                repeat-interleave + rearrange pattern at the ensemble level.

        Returns:
            If ``n_samples is None``: shape ``(B, T_out, *spatial, C_out)``.
            If ``n_samples = K``: shape ``(B, T_out, *spatial, C_out, K)``.
        """
        if n_samples is None:
            return self._draw_one(x, global_cond)

        b = x.shape[0]
        x_rep = torch.repeat_interleave(x, n_samples, dim=0)
        gc_rep = (
            torch.repeat_interleave(global_cond, n_samples, dim=0)
            if global_cond is not None
            else None
        )
        out_rep = self._draw_one(x_rep, gc_rep)
        return rearrange(out_rep, "(b m) ... -> b ... m", b=b, m=n_samples)

    def _draw_one(
        self,
        x: TensorBTSC,
        global_cond: Tensor | None,
    ) -> TensorBTSC:
        """Sample one cohort-of-one output from noise."""
        spatial_shape = tuple(x.shape[2:-1])
        z_shape = (
            x.shape[0],
            self.n_steps_output,
            *spatial_shape,
            self.n_channels_out,
        )
        eps = torch.randn(z_shape, device=x.device, dtype=x.dtype)
        return self.generator_func(eps, x, global_cond)

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Compute the cohort training loss for a batch.

        Inflates the batch to cohorts of size ``self.n_samples`` per input,
        runs a single forward pass through the generator, reshapes to
        ``(B, K, -1)``, and delegates the loss body to the subclass's
        ``_compute_cohort_loss``. Records the ``cohort_spread`` diagnostic on
        ``self._latest_diagnostics``, merged with any extras returned by the
        subclass.

        Args:
            batch: Encoded batch to compute the loss for.

        Returns:
            Scalar loss tensor.
        """
        # Clear up front so a raise in _compute_cohort_loss can't leave stale
        # values from the previous step for a downstream consumer to pick up.
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
        # n_samples=1 (the MSECohort floor); unbiased std would yield NaN and
        # corrupt the W&B metric series.
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
                ``(B, K, prod(T_out * spatial * C_out))``. Carries the live
                gradient through the backbone.
            target_b: Cohort-flattened target of shape
                ``(B, K, prod(T_out * spatial * C_out))``. With
                repeat-interleave layout the K target copies along the
                middle axis are identical; subclasses that need the unique
                positive use ``target_b[:, :1, :]``.

        Returns:
            ``(loss, extra_diagnostics)`` — a scalar loss tensor and a
            (possibly empty) dict of scalar tensors merged into
            ``self._latest_diagnostics`` alongside ``cohort_spread``.
        """
        ...
