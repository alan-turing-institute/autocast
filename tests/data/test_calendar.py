"""Tests for month-anchored subtrajectory index selection."""

from datetime import date

from autocast.data.calendar import calendar_dates, month_start_indices

# First-of-month indices for a 365-day no-leap year (Jan 1 == index 0).
_NO_LEAP_MONTH_STARTS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def test_calendar_dates_no_leap_drops_feb_29():
    no_leap = calendar_dates([2004], drop_leap_day=True)  # 2004 is a leap year
    standard = calendar_dates([2004], drop_leap_day=False)
    assert len(no_leap) == 365
    assert len(standard) == 366
    assert date(2004, 2, 29) not in no_leap
    assert date(2004, 2, 29) in standard


def test_month_start_indices_skips_first_and_returns_input_window_starts():
    n_steps_input = 3
    max_rollout_steps = 5
    starts, init_dates = month_start_indices(
        test_years=[2001],
        n_steps_input=n_steps_input,
        max_rollout_steps=max_rollout_steps,
        stride=1,
    )

    # January is dropped: there is no room for the n_steps_input lead-in before
    # Jan 1, so the earliest init is Feb 1.
    expected_starts = [idx - (n_steps_input - 1) for idx in _NO_LEAP_MONTH_STARTS[1:]]
    assert starts == expected_starts
    assert [d.month for d in init_dates] == list(range(2, 13))
    assert all(d.day == 1 for d in init_dates)

    # Each start's init (last input frame) lands on the 1st of the month.
    for start, init_date in zip(starts, init_dates, strict=True):
        init_day = start + n_steps_input - 1
        assert calendar_dates([2001])[init_day] == init_date


def test_month_start_indices_drops_trailing_partial_month():
    # A long horizon means the final months cannot complete a rollout.
    starts_short, dates_short = month_start_indices([2001], 3, 5, stride=1)
    starts_long, dates_long = month_start_indices([2001], 3, 60, stride=1)
    assert dates_long[-1] < dates_short[-1]
    assert len(starts_long) < len(starts_short)


def test_month_start_indices_leap_handling_shifts_post_february():
    # With Feb 29 dropped, March starts one step earlier than on a standard
    # calendar — the reason the no-leap mapping must match the stored data.
    no_leap, _ = month_start_indices([2004], 1, 1, drop_leap_day=True)
    standard, _ = month_start_indices([2004], 1, 1, drop_leap_day=False)
    # March is the 3rd kept month start (Jan, Feb, Mar, ...).
    assert standard[2] == no_leap[2] + 1
