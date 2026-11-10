"""The analysis tools the AI agent can call.

Each tool is a plain Python function: it takes the dataset plus a few simple
arguments (strings, lists, dicts) and returns a small dict of rounded numbers.
The AI never sees raw rows, only these outputs, and every number it reports
must come from one of them.

Metrics:
    "revenue"              sum of the main metric column (e.g. `total`)
    "orders"               number of rows
    "average_order_value"  revenue / orders

Dates are "YYYY-MM-DD" strings; periods include both the start and end day.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from analysis.baseline import fit_expected
from analysis.baseline import find_unusual_periods as _find_periods
from analysis.columns import ColumnRoles, detect_columns

METRICS = ("revenue", "orders", "average_order_value")
# Segment mix ("what share of revenue is mobile?") is measured over the weeks
# just before the period being explained, so it reflects the recent normal.
MIX_LOOKBACK_DAYS = 28
# A driver must account for at least this share of the total gap.
MIN_DRIVER_SHARE = 0.6
# Ignore segment combinations smaller than this share of normal revenue.
MIN_COMBO_SHARE = 0.02


# --------------------------------------------------------------------------
# Dataset: the uploaded table, prepared once
# --------------------------------------------------------------------------


@dataclass
class Dataset:
    """An uploaded table plus its detected column roles.

    Personal columns are dropped when the dataset is created, so no tool can
    ever read them.
    """

    df: pd.DataFrame
    roles: ColumnRoles

    @classmethod
    def from_frame(cls, df: pd.DataFrame, roles: ColumnRoles | None = None) -> "Dataset":
        roles = roles or detect_columns(df)
        if roles.date is None or roles.metric is None:
            raise ValueError("Need a date column and a numeric metric column.")
        clean = df.drop(columns=roles.personal).copy()
        clean["_timestamp"] = pd.to_datetime(clean[roles.date], format="mixed")
        clean["_day"] = clean["_timestamp"].dt.normalize()
        clean[roles.metric] = pd.to_numeric(clean[roles.metric], errors="coerce").fillna(0)
        return cls(df=clean, roles=roles)

    @classmethod
    def from_csv(cls, path) -> "Dataset":
        return cls.from_frame(pd.read_csv(path))

    @property
    def days(self) -> pd.DatetimeIndex:
        """Every calendar day from the first row to the last."""
        return pd.date_range(self.df["_day"].min(), self.df["_day"].max(), freq="D")

    @property
    def has_times(self) -> bool:
        """True if the date column includes a time of day, not just a date."""
        return bool((self.df["_timestamp"] != self.df["_day"]).any())

    def rows(self, filters: dict[str, str] | None = None) -> pd.DataFrame:
        """Rows matching every {column: value} filter."""
        rows = self.df
        for column, value in (filters or {}).items():
            if column not in self.roles.segments:
                raise ValueError(f"'{column}' is not a breakdown column. Options: {self.roles.segments}")
            rows = rows[rows[column].astype(str) == str(value)]
        return rows

    def daily(self, metric: str = "revenue", filters: dict[str, str] | None = None) -> pd.Series:
        """The metric for each day (days with no rows count as 0)."""
        if metric not in ("revenue", "orders"):
            raise ValueError(f"daily() supports 'revenue' or 'orders', not {metric!r}")
        grouped = self.rows(filters).groupby("_day")
        values = grouped[self.roles.metric].sum() if metric == "revenue" else grouped.size()
        return values.reindex(self.days, fill_value=0).astype(float)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _period(ds: Dataset, start: str, end: str) -> pd.DatetimeIndex:
    """The days from start to end, checked against the data's date range."""
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    if days.empty or days[0] < ds.days[0] or days[-1] > ds.days[-1]:
        raise ValueError(
            f"Period {start} to {end} is outside the data "
            f"({ds.days[0].date()} to {ds.days[-1].date()})."
        )
    return days


def _actual_and_expected(series: pd.Series, period: pd.DatetimeIndex) -> tuple[float, float]:
    """Sum of actual and expected over `period`; the period is left out of the fit."""
    expected = fit_expected(series, exclude=period)
    return float(series[period].sum()), float(expected[period].sum())


def _pct(part: float, whole: float) -> float | None:
    """Percentage rounded to one decimal, or None when the base is zero."""
    return round(part / whole * 100, 1) if whole else None


def _gap_summary(actual: float, expected: float, decimals: int = 2) -> dict:
    return {
        "actual": round(actual, decimals),
        "expected": round(expected, decimals),
        "gap": round(actual - expected, decimals),
        "gap_pct": _pct(actual - expected, expected),
    }


# --------------------------------------------------------------------------
# Tool 1: find unusual periods
# --------------------------------------------------------------------------


def find_unusual_periods(ds: Dataset, metric: str = "revenue") -> dict:
    """Runs of days where the metric was far from expected (day-of-week and
    trend adjusted), most extreme first."""
    if metric == "average_order_value":
        revenue, orders = ds.daily("revenue"), ds.daily("orders")
        series = (revenue / orders.replace(0, np.nan)).fillna(revenue.sum() / orders.sum())
    else:
        series = ds.daily(metric)
    periods, _ = _find_periods(series)
    return {
        "metric": metric,
        "data_from": ds.days[0].date().isoformat(),
        "data_to": ds.days[-1].date().isoformat(),
        "periods": [p.as_dict() for p in periods],
        "note": "No unusual periods found." if not periods else None,
    }


# --------------------------------------------------------------------------
# Tool 2: compare a period to expected
# --------------------------------------------------------------------------


def compare_to_expected(ds: Dataset, metric: str, start: str, end: str) -> dict:
    """Actual vs expected for a period, with the % gap."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    period = _period(ds, start, end)
    if metric == "average_order_value":
        rev_a, rev_e = _actual_and_expected(ds.daily("revenue"), period)
        ord_a, ord_e = _actual_and_expected(ds.daily("orders"), period)
        result = _gap_summary(rev_a / ord_a if ord_a else 0, rev_e / ord_e if ord_e else 0)
    else:
        actual, expected = _actual_and_expected(ds.daily(metric), period)
        result = _gap_summary(actual, expected, decimals=2 if metric == "revenue" else 1)
    return {"metric": metric, "start": start, "end": end, **result}


# --------------------------------------------------------------------------
# Tool 3: orders x average order value
# --------------------------------------------------------------------------


def decompose(ds: Dataset, start: str, end: str) -> dict:
    """Split the revenue gap into 'fewer (or more) orders' and 'smaller (or
    larger) orders'.

    revenue = orders x average order value, so
        gap = (actual orders - expected orders) x expected AOV      <- order count effect
            + actual orders x (actual AOV - expected AOV)            <- order value effect
    The two effects add up exactly to the revenue gap.
    """
    period = _period(ds, start, end)
    rev_a, rev_e = _actual_and_expected(ds.daily("revenue"), period)
    ord_a, ord_e = _actual_and_expected(ds.daily("orders"), period)
    aov_a = rev_a / ord_a if ord_a else 0.0
    aov_e = rev_e / ord_e if ord_e else 0.0

    count_effect = (ord_a - ord_e) * aov_e
    value_effect = ord_a * (aov_a - aov_e)
    gap = rev_a - rev_e
    main = "order count" if abs(count_effect) >= abs(value_effect) else "order value"
    return {
        "start": start,
        "end": end,
        "revenue": _gap_summary(rev_a, rev_e),
        "orders": _gap_summary(ord_a, ord_e, decimals=1),
        "average_order_value": _gap_summary(aov_a, aov_e),
        "order_count_effect": round(count_effect, 2),
        "order_value_effect": round(value_effect, 2),
        "order_count_share_of_gap_pct": _pct(count_effect, gap),
        "order_value_share_of_gap_pct": _pct(value_effect, gap),
        "main_cause": main,
    }


# --------------------------------------------------------------------------
# Tool 4: which segments the gap came from
# --------------------------------------------------------------------------


def segment_gap(
    ds: Dataset,
    start: str,
    end: str,
    dimensions: list[str] | None = None,
    include_combinations: bool = True,
) -> dict:
    """Where did the revenue gap come from?

    Each segment's expected revenue = total expected revenue x that segment's
    normal share of revenue (from the weeks just before the period). Its gap is
    actual - expected, and `share_of_gap_pct` is its part of the total gap.

    Also checks pairs of dimensions (e.g. device + payment method), because a
    problem often hits one specific combination. `top_driver` is the most
    specific segment that accounts for most of the gap.
    """
    dimensions = dimensions or ds.roles.segments
    unknown = [d for d in dimensions if d not in ds.roles.segments]
    if unknown:
        raise ValueError(f"Unknown breakdown columns {unknown}. Options: {ds.roles.segments}")

    period = _period(ds, start, end)
    total_actual, total_expected = _actual_and_expected(ds.daily("revenue"), period)
    total_gap = total_actual - total_expected

    metric = ds.roles.metric
    lookback = ds.df[(ds.df["_day"] < period[0]) & (ds.df["_day"] >= period[0] - pd.Timedelta(days=MIX_LOOKBACK_DAYS))]
    in_period = ds.df[ds.df["_day"].isin(period)]
    lookback_total = lookback[metric].sum()

    def summarise(group_columns: list[str]) -> list[dict]:
        normal_share = lookback.groupby(group_columns)[metric].sum() / lookback_total
        actual = in_period.groupby(group_columns)[metric].sum()
        rows = []
        for key, share in normal_share.items():
            key = key if isinstance(key, tuple) else (key,)
            expected = total_expected * share
            got = float(actual.get(key if len(key) > 1 else key[0], 0.0))
            rows.append(
                {
                    "segment": {c: str(v) for c, v in zip(group_columns, key)},
                    "normal_share_of_revenue_pct": round(share * 100, 1),
                    **_gap_summary(got, expected),
                    "share_of_gap_pct": _pct(got - expected, total_gap),
                }
            )
        # Biggest contributors to the gap (in the gap's direction) first.
        sign = -1 if total_gap < 0 else 1
        return sorted(rows, key=lambda r: sign * -r["gap"])

    by_dimension = {d: summarise([d]) for d in dimensions}
    combos = []
    if include_combinations:
        for pair in combinations(dimensions, 2):
            combos += [r for r in summarise(list(pair)) if r["normal_share_of_revenue_pct"] >= MIN_COMBO_SHARE * 100]
        sign = -1 if total_gap < 0 else 1
        combos = sorted(combos, key=lambda r: sign * -r["gap"])[:5]

    return {
        "start": start,
        "end": end,
        "total": _gap_summary(total_actual, total_expected),
        "by_dimension": by_dimension,
        "top_combinations": combos,
        "top_driver": _top_driver(by_dimension, combos, total_actual, total_expected),
    }


def _top_driver(by_dimension: dict, combos: list, total_actual: float, total_expected: float) -> dict | None:
    """Most specific segment that explains most of the gap.

    Candidates must account for at least MIN_DRIVER_SHARE of the gap. Among
    those, pick the one whose own % change is most extreme: "mobile + Shop Pay
    fell 85%" beats "mobile fell 40%" when both explain the same dollars.
    """
    total_gap = total_actual - total_expected
    if total_gap == 0:
        return None
    candidates = [r for rows in by_dimension.values() for r in rows] + combos
    candidates = [r for r in candidates if (r["share_of_gap_pct"] or 0) >= MIN_DRIVER_SHARE * 100]
    if not candidates:
        return None
    sign = -1 if total_gap < 0 else 1
    best = max(candidates, key=lambda r: sign * (r["gap_pct"] or 0))
    rest_expected = total_expected - best["expected"]
    rest_gap = (total_actual - best["actual"]) - rest_expected
    return {
        **best,
        "rest_of_business_gap_pct": _pct(rest_gap, rest_expected),
    }


# --------------------------------------------------------------------------
# Tool 5: drill into one slice
# --------------------------------------------------------------------------


def drill_down(
    ds: Dataset, filters: dict[str, str], start: str, end: str, metric: str = "revenue"
) -> dict:
    """The metric per day for one slice (e.g. device=Mobile, payment_method=Shop Pay),
    against its own expected line, and when the change started."""
    if metric not in ("revenue", "orders"):
        raise ValueError("drill_down supports 'revenue' or 'orders'")
    period = _period(ds, start, end)
    series = ds.daily(metric, filters)
    expected = fit_expected(series, exclude=period)

    # Show the week before the period too, so the change is visible.
    shown = pd.date_range(max(period[0] - pd.Timedelta(days=7), ds.days[0]), period[-1], freq="D")
    decimals = 2 if metric == "revenue" else 1
    daily = [
        {"date": d.date().isoformat(), "actual": round(series[d], decimals), "expected": round(expected[d], decimals)}
        for d in shown
    ]

    periods, _ = _find_periods(series, max_periods=1)
    started = periods[0].start.date().isoformat() if periods else None
    result = {
        "filters": filters,
        "metric": metric,
        "start": start,
        "end": end,
        **_gap_summary(float(series[period].sum()), float(expected[period].sum()), decimals),
        "change_started_on": started,
        "change_started_around": None,
        "daily": daily,
    }
    if started and ds.has_times:
        result["change_started_around"] = _estimate_start_time(ds, filters, pd.Timestamp(started), period)
    return result


def _estimate_start_time(ds: Dataset, filters: dict, start_day: pd.Timestamp, period: pd.DatetimeIndex) -> str | None:
    """Estimate the hour the slice's order rate changed.

    Looks at hourly order counts over the day before and the day the change
    was detected, compared with the slice's normal count for each hour. Tries
    every possible switch hour and keeps the one that best fits "normal before,
    a different rate after" (a Poisson likelihood).
    """
    rows = ds.rows(filters)
    normal_rows = rows[~rows["_day"].isin(period) & (rows["_day"] < start_day - pd.Timedelta(days=1))]
    n_normal_days = normal_rows["_day"].nunique()
    if n_normal_days < 7:
        return None
    per_hour = normal_rows.groupby(normal_rows["_timestamp"].dt.hour).size().reindex(range(24), fill_value=0) / n_normal_days
    expected = np.tile(per_hour.to_numpy(), 2) + 1e-9  # two days, hour by hour

    window_start = start_day - pd.Timedelta(days=1)
    window = rows[(rows["_timestamp"] >= window_start) & (rows["_timestamp"] < start_day + pd.Timedelta(days=1))]
    slot = ((window["_timestamp"] - window_start) / pd.Timedelta(hours=1)).astype(int)
    observed = np.bincount(slot, minlength=48)[:48].astype(float)

    def poisson_loglik(obs, exp):
        return float(np.sum(obs * np.log(exp) - exp))

    best_hour, best_ll = None, -np.inf
    for h in range(1, 48):
        after_ratio = observed[h:].sum() / expected[h:].sum()
        ll = poisson_loglik(observed[:h], expected[:h]) + poisson_loglik(observed[h:], expected[h:] * max(after_ratio, 1e-6))
        if ll > best_ll:
            best_hour, best_ll = h, ll
    when = window_start + pd.Timedelta(hours=best_hour)
    return when.strftime("%Y-%m-%d %H:00")


# --------------------------------------------------------------------------
# Tool 6: same period last year
# --------------------------------------------------------------------------


def same_period_last_year(ds: Dataset, metric: str, start: str, end: str) -> dict:
    """Compare the period with the same weekdays 52 weeks earlier.

    Returns status "not enough history" if the data doesn't go back that far.
    """
    if metric not in ("revenue", "orders"):
        raise ValueError("same_period_last_year supports 'revenue' or 'orders'")
    period = _period(ds, start, end)
    last_year = period - pd.Timedelta(weeks=52)
    if last_year[0] < ds.days[0]:
        return {
            "metric": metric,
            "status": "not enough history",
            "history_starts": ds.days[0].date().isoformat(),
            "needed_from": last_year[0].date().isoformat(),
        }
    series = ds.daily(metric)
    now, before = float(series[period].sum()), float(series[last_year].sum())
    return {
        "metric": metric,
        "status": "ok",
        "this_year": round(now, 2),
        "last_year": round(before, 2),
        "last_year_start": last_year[0].date().isoformat(),
        "last_year_end": last_year[-1].date().isoformat(),
        "change_pct": _pct(now - before, before),
    }
