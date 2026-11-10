import numpy as np

from analysis.baseline import find_unusual_periods, fit_expected
from tests.conftest import daily_series

WEEKLY_SHAPE = [100, 95, 95, 100, 110, 130, 120]  # Mon..Sun


def make_series(weeks=8, noise=0.03, seed=0, growth_per_week=1.0):
    rng = np.random.default_rng(seed)
    values = []
    for w in range(weeks):
        for level in WEEKLY_SHAPE:
            values.append((level + growth_per_week * w) * (1 + rng.normal(0, noise)))
    return daily_series(values)


def test_fit_expected_learns_weekday_pattern_and_trend():
    series = make_series(noise=0.0, growth_per_week=2.0)
    expected = fit_expected(series)
    assert np.allclose(expected, series, atol=1e-6)


def test_excluded_days_do_not_drag_down_the_baseline():
    series = make_series(noise=0.0)
    hit = series.copy()
    hit.iloc[-6:] *= 0.5
    expected = fit_expected(hit, exclude=hit.index[-6:])
    assert np.allclose(expected.iloc[-6:], series.iloc[-6:], atol=1e-6)


def test_flags_a_planted_drop_with_exact_dates():
    series = make_series(seed=1)
    series.iloc[-6:] *= 0.75  # 25% drop over the last six days
    periods, _ = find_unusual_periods(series)
    assert len(periods) == 1
    top = periods[0]
    assert top.start == series.index[-6] and top.end == series.index[-1]
    assert top.direction == "below"
    assert -30 < top.gap_pct * 100 < -20


def test_flags_a_one_day_spike():
    series = make_series(seed=2)
    series.iloc[20] *= 2
    periods, _ = find_unusual_periods(series)
    assert periods and periods[0].start == periods[0].end == series.index[20]
    assert periods[0].direction == "above"


def test_no_false_alarm_on_clean_series():
    for seed in range(5):
        periods, _ = find_unusual_periods(make_series(seed=seed))
        assert periods == [], f"false alarm on seed {seed}: {[p.as_dict() for p in periods]}"
