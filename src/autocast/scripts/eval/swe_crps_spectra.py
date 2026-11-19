"""Member-first radial power spectra for periodic, square fields (any channel count)."""

from __future__ import annotations

import torch

POWER_KEYS = ("truth_power", "member_power", "mean_field_power", "anomaly_power")
RATIO_TRUTH_FLOOR = 1e-10
#: ``(B, T, X, Y, C, M)``: trajectory, lead, two space axes, channel, member.
PREDICTION_DIMS = 6
#: A spread needs at least two members.
MIN_MEMBERS = 2
#: DC plus at least one non-DC shell.
MIN_SHELLS = 2


def compute_swe_power_spectra(
    prediction: torch.Tensor, truth: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return spectra with shape (trajectory, lead, channel, radial shell).

    FFTs are computed on physical fields, individually for every member, before
    averaging *powers*. ``mean_field_power`` is a separate diagnostic, never the
    ensemble forecast spectrum. Anomalies are member minus ensemble mean.
    ``error_power`` is the power of the ensemble mean minus the truth, so a band's
    spread and error can be compared the way the pooled spread-skill ratio does.

    Unit-width shells contain radii [k-0.5, k+0.5), with DC alone in shell zero.
    Shell sums (not per-mode averages) use forward-normalized FFTs, so summing
    all shells gives the spatial mean square (Parseval). Integer wavenumbers
    correspond to a square 2*pi-periodic domain. All corner modes are retained;
    shells beyond the axis Nyquist wavenumber are only partially populated.
    No Bessel correction is used: this is the empirical member distribution.
    """
    if prediction.ndim != PREDICTION_DIMS or truth.shape != prediction.shape[:-1]:
        msg = "Expected prediction BTXYCM and matching truth BTXYC"
        raise ValueError(msg)
    if truth.shape[2] != truth.shape[3]:
        msg = "Expected square fields (X == Y)"
        raise ValueError(msg)
    if prediction.shape[-1] < MIN_MEMBERS or min(truth.shape) < 1:
        msg = "Require nonempty fields and at least two members"
        raise ValueError(msg)
    if not prediction.isfinite().all() or not truth.isfinite().all():
        msg = "Spectral inputs must be finite"
        raise ValueError(msg)

    n = truth.shape[2]
    k = torch.fft.fftfreq(n, device=truth.device, dtype=torch.float64) * n
    radius = (k[:, None].square() + k[None, :].square()).sqrt()
    indices = (radius + 0.5).floor().long().flatten()
    counts = torch.bincount(indices)
    n_bins = counts.numel()

    def transform(field):
        return torch.fft.fft2(
            field.to(torch.float64).permute(0, 1, 4, 2, 3), norm="forward"
        )

    def shell_sum(power):
        flat = power.flatten(-2)
        result = flat.new_zeros((*flat.shape[:-1], n_bins))
        return result.scatter_add(-1, indices.expand_as(flat), flat)

    # Accumulation in double precision also makes opposite-member cancellation
    # and the member = mean-field + anomaly power identity numerically robust.
    mean_field = prediction.mean(-1, dtype=torch.float64)
    mean_fft = transform(mean_field)
    truth_fft = transform(truth)
    truth_power = shell_sum(truth_fft.abs().square())
    member_power = torch.zeros_like(truth_power)
    anomaly_power = torch.zeros_like(truth_power)
    for member in prediction.unbind(-1):
        member_fft = transform(member)
        member_power += shell_sum(member_fft.abs().square())
        anomaly_power += shell_sum((member_fft - mean_fft).abs().square())
    members = prediction.shape[-1]
    return {
        "wavenumber": torch.arange(n_bins, device=truth.device),
        "mode_count": counts,
        "axis_nyquist": torch.tensor(n / 2, device=truth.device),
        "truth_power": truth_power,
        "member_power": member_power / members,
        "mean_field_power": shell_sum(mean_fft.abs().square()),
        "anomaly_power": anomaly_power / members,
        "error_power": shell_sum((mean_fft - truth_fft).abs().square()),
    }


def log_power_ratio(
    prediction_power: torch.Tensor,
    truth_power: torch.Tensor,
    *,
    relative_truth_floor: float = RATIO_TRUTH_FLOOR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take log10 of an already-aggregated power ratio, with explicit masking.

    DC is excluded. Truth <= 1e-10 of the maximum non-DC truth power in each
    curve, and nonpositive prediction power, are masked rather than replaced
    with an arbitrary epsilon. Return NaNs and a validity mask for these bins.
    This function must be applied *after* averaging powers over trajectories
    and any requested time window, not averaged over individual log ratios.
    """
    if (
        prediction_power.shape != truth_power.shape
        or truth_power.shape[-1] < MIN_SHELLS
    ):
        msg = "Require matching spectra with DC and at least one non-DC bin"
        raise ValueError(msg)
    threshold = truth_power[..., 1:].amax(-1, keepdim=True) * relative_truth_floor
    valid = (
        (truth_power > threshold)
        & (prediction_power > 0)
        & truth_power.isfinite()
        & prediction_power.isfinite()
    )
    valid[..., 0] = False
    ratio = torch.full_like(truth_power, float("nan"))
    ratio[valid] = torch.log10(prediction_power[valid] / truth_power[valid])
    return ratio, valid
