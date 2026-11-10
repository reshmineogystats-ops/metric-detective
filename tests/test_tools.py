import numpy as np
import pandas as pd
import pytest

from analysis.tools import (
    Dataset,
    compare_to_expected,
    decompose,
    drill_down,
    find_unusual_periods,
    same_period_last_year,
    segment_gap,
)

START, END = "2026-11-03", "2026-11-08"
BROKEN = {"device": "Mobile", "payment_method": "Shop Pay"}


# ---- Milestone 1 "done when": the headline checks -------------------------


def test_flags_nov_3_to_8(sample):
    result = find_unusual_periods(sample, "revenue")
    top = result["periods"][0]
    assert (top["start"], top["end"]) == (START, END)
    assert top["direction"] == "below"
    assert -28 < top["gap_pct"] < -18


def test_mobile_shop_pay_is_the_top_driver(sample):
    driver = segment_gap(sample, START, END)["top_driver"]
    assert driver["segment"] == BROKEN
    assert driver["gap_pct"] < -60
    assert abs(driver["rest_of_business_gap_pct"]) < 10  # everything else looks normal


# ---- find_unusual_periods ---------------------------------------------------


def test_clean_data_has_no_unusual_period(clean_sample):
    assert find_unusual_periods(clean_sample, "revenue")["periods"] == []


def test_order_count_also_flags_the_drop(sample):
    top = find_unusual_periods(sample, "orders")["periods"][0]
    assert (top["start"], top["end"]) == (START, END)


# ---- compare_to_expected ----------------------------------------------------


def test_compare_to_expected_adds_up(sample):
    r = compare_to_expected(sample, "revenue", START, END)
    assert r["gap"] == pytest.approx(r["actual"] - r["expected"], abs=0.02)
    assert r["actual"] == pytest.approx(sample.daily("revenue")[START:END].sum(), abs=0.01)


def test_compare_average_order_value_is_roughly_unchanged(sample):
    r = compare_to_expected(sample, "average_order_value", START, END)
    assert abs(r["gap_pct"]) < 10  # the problem is fewer orders, not smaller ones


def test_rejects_dates_outside_the_data(sample):
    with pytest.raises(ValueError):
        compare_to_expected(sample, "revenue", "2026-12-01", "2026-12-07")


# ---- decompose --------------------------------------------------------------


def test_decompose_effects_sum_to_gap(sample):
    r = decompose(sample, START, END)
    assert r["order_count_effect"] + r["order_value_effect"] == pytest.approx(r["revenue"]["gap"], abs=0.05)
    assert r["main_cause"] == "order count"


def test_decompose_detects_smaller_orders():
    days = pd.date_range("2026-01-05", periods=56, freq="D")
    rows = [{"date": d, "total": 50.0 if d < days[-7] else 30.0} for d in days for _ in range(20)]
    r = decompose(Dataset.from_frame(pd.DataFrame(rows)), str(days[-7].date()), str(days[-1].date()))
    assert r["main_cause"] == "order value"
    assert r["orders"]["gap"] == pytest.approx(0, abs=0.5)


# ---- segment_gap ------------------------------------------------------------


def test_segment_shares_sum_to_100_per_dimension(sample):
    r = segment_gap(sample, START, END, include_combinations=False)
    for dimension, rows in r["by_dimension"].items():
        assert sum(row["share_of_gap_pct"] for row in rows) == pytest.approx(100, abs=0.5), dimension
        assert sum(row["normal_share_of_revenue_pct"] for row in rows) == pytest.approx(100, abs=0.5)


def test_segment_gap_ranks_mobile_first_on_device(sample):
    rows = segment_gap(sample, START, END, dimensions=["device"])["by_dimension"]["device"]
    assert rows[0]["segment"] == {"device": "Mobile"}


def test_segment_gap_rejects_unknown_dimension(sample):
    with pytest.raises(ValueError):
        segment_gap(sample, START, END, dimensions=["customer_email"])


# ---- drill_down -------------------------------------------------------------


def test_drill_down_finds_start_day_and_hour(sample):
    r = drill_down(sample, BROKEN, START, END, metric="orders")
    assert r["change_started_on"] == START
    assert r["change_started_around"] in ("2026-11-03 09:00", "2026-11-03 10:00", "2026-11-03 11:00")
    assert r["gap_pct"] < -60
    assert len(r["daily"]) == 13  # six days plus the week before


def test_drill_down_on_healthy_slice_shows_no_change(sample):
    r = drill_down(sample, {"device": "Desktop"}, START, END)
    assert r["change_started_on"] is None
    assert abs(r["gap_pct"]) < 15


def test_drill_down_cannot_filter_on_personal_column(sample):
    with pytest.raises(ValueError):
        drill_down(sample, {"customer_email": "customer1@example.com"}, START, END)


# ---- same_period_last_year --------------------------------------------------


def test_last_year_not_enough_history(sample):
    r = same_period_last_year(sample, "revenue", START, END)
    assert r["status"] == "not enough history"
    assert r["history_starts"] == "2026-09-14"


def test_last_year_with_enough_history():
    days = pd.date_range("2025-01-06", "2026-01-11", freq="D")
    rows = [{"date": d, "total": 100.0 if d.year == 2025 else 120.0} for d in days]
    r = same_period_last_year(Dataset.from_frame(pd.DataFrame(rows)), "revenue", "2026-01-05", "2026-01-11")
    assert r["status"] == "ok"
    assert r["last_year_start"] == "2025-01-06"
    assert r["change_pct"] == pytest.approx(20.0)
