"""Shared test setup: the committed demo file and a no-problem twin."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.tools import Dataset  # noqa: E402
import make_sample_data  # noqa: E402


@pytest.fixture(scope="session")
def sample() -> Dataset:
    """The demo shop with the planted mobile + Shop Pay checkout problem."""
    return Dataset.from_csv(ROOT / "data" / "sample_shop.csv")


@pytest.fixture(scope="session")
def clean_sample() -> Dataset:
    """Same shop, same seed, but no planted problem."""
    return Dataset.from_frame(make_sample_data.make_orders(with_incident=False))


def daily_series(values, start="2026-01-05") -> pd.Series:
    """Helper: a daily series starting on a Monday."""
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"), dtype=float)
