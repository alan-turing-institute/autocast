"""Calendar helpers for selecting month-anchored rollout subtrajectories.

These utilities map integer time-step indices to calendar dates so that
rollout subtrajectories can be initialised on the first of each month — the
"monthly initialisation" protocol used for forecast evaluation.

The default calendar is *no-leap* (365 days/year, Feb 29 dropped). This matches
datasets stored with a ``(doy - 1) / 365`` day-of-year encoding (e.g. the
OSISAF/ERA5 sea-ice data): if the data drops Feb 29 at preprocessing, the
index-to-date mapping must drop it too, otherwise every month start after a
leap-year February would be shifted one step off the stored data. Set
``drop_leap_day=False`` for datasets stored on a standard (proleptic Gregorian)
calendar.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta


def calendar_dates(years: list[int], *, drop_leap_day: bool = True) -> list[date]:
    """Daily dates spanning ``years`` (inclusive), one per stored time step.

    Parameters
    ----------
    years:
        Calendar years to enumerate, in order (e.g. ``[2019, 2020]``).
    drop_leap_day:
        If True (default), Feb 29 is omitted so each year has 365 days,
        matching a no-leap stored calendar. If False, leap days are kept.
    """
    dates: list[date] = []
    for year in years:
        d = date(year, 1, 1)
        end = date(year, 12, 31)
        while d <= end:
            is_leap_day = d.month == 2 and d.day == 29
            if not (drop_leap_day and is_leap_day and calendar.isleap(d.year)):
                dates.append(d)
            d += timedelta(days=1)
    return dates


def month_start_indices(
    test_years: list[int],
    n_steps_input: int,
    max_rollout_steps: int,
    stride: int = 1,
    *,
    drop_leap_day: bool = True,
) -> tuple[list[int], list[date]]:
    """Input-window start indices for subtrajectories initialised on the 1st.

    Each subtrajectory has an input window of ``n_steps_input`` steps; its
    *initialisation* (the last input frame, from which the rollout predicts
    forward) is the first of a month. A subtrajectory at input-window start
    ``s`` therefore initialises on global step ``s + n_steps_input - 1`` and
    needs ``max_rollout_steps`` steps of lookahead beyond that.

    A candidate is kept only when both the input window and the full rollout
    horizon fit inside the data:

    - the input window fits because ``s = i * stride >= 0`` (so the very first
      month start, with no room for the lead-in, is naturally skipped); and
    - the horizon fits because ``s + n_steps_input + max_rollout_steps <= n``
      (so a trailing partial month is skipped).

    Pass ``max_rollout_steps`` equal to the dataset's ``n_steps_output`` so the
    returned starts line up with the subtrajectory window length
    ``n_steps_input + n_steps_output``.

    Returns
    -------
    (start_idxs, init_dates)
        ``start_idxs`` are input-window start indices (feed straight into
        ``SpatioTemporalDataset(subtrajectory_start_idxs=...)``); ``init_dates``
        are the matching first-of-month initialisation dates, handy for labels.
    """
    dates = calendar_dates(test_years, drop_leap_day=drop_leap_day)
    n = len(dates)
    start_idxs: list[int] = []
    init_dates: list[date] = []
    i = 0
    while True:
        start = i * stride
        init_day = start + n_steps_input - 1
        if init_day + max_rollout_steps >= n:
            break
        if dates[init_day].day == 1:
            start_idxs.append(start)
            init_dates.append(dates[init_day])
        i += 1
    return start_idxs, init_dates
