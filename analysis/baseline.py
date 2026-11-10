"""What should each day have looked like? And which days didn't?

The "expected" value for a day comes from a simple model fitted on normal days:

    expected = typical level for that day of the week + a straight-line trend

(an ordinary least-squares fit, done with numpy). Days that are part of a
suspected problem are left out of the fit, so a drop can't drag down its own
baseline.

To find unusual periods we look at runs of days, not single days. A 20% dip on
one day is often noise; the same dip six days in a row is not. For every run of
consecutive days we add up the gaps (actual - expected) and turn that into a
robust z-score. The run with the most extreme score is the unusual period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# A run of days counts as unusual when its score is beyond this. It is much
# higher than the textbook 3 because we test hundreds of possible runs, and
# daily revenue has heavier tails than a bell curve. Set from simulations:
# on 30 clean copies of the demo shop, scores almost always stayed below 5.5;
# with the planted checkout problem they were usually 7-12. Milestone 4's
# evals re-check this.
SCORE_THRESHOLD = 5.5
HIGH_CONFIDENCE_SCORE = 7.0
MAX_RUN_DAYS = 14
MIN_TRAINING_DAYS = 14


@dataclass
class UnusualPeriod:
    """A run of consecutive days far from expected."""

    start: pd.Timestamp
    end: pd.Timestamp
    actual: float
    expected: float
    score: float  # robust z-score of the run; negative = below expected

    @property
    def gap(self) -> float:
        return self.actual - self.expected

    @property
    def gap_pct(self) -> float:
        return self.gap / self.expected if self.expected else float("nan")

    @property
    def confidence(self) -> str:
        """'high' when far past the bar, otherwise 'medium'."""
        return "high" if abs(self.score) >= HIGH_CONFIDENCE_SCORE else "medium"

    @property
    def direction(self) -> str:
        return "below" if self.gap < 0 else "above"

    def as_dict(self) -> dict:
        return {
            "start": self.start.date().isoformat(),
            "end": self.end.date().isoformat(),
            "days": (self.end - self.start).days + 1,
            "actual": round(self.actual, 2),
            "expected": round(self.expected, 2),
            "gap": round(self.gap, 2),
            "gap_pct": round(self.gap_pct * 100, 1),
            "direction": self.direction,
            "score": round(self.score, 1),
            "confidence": self.confidence,
        }


def _design_matrix(dates: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """One column per day of the week (Mon..Sun) plus a trend column in weeks."""
    day_of_week = np.zeros((len(dates), 7))
    day_of_week[np.arange(len(dates)), dates.day_of_week] = 1.0
    weeks = ((dates - origin).days.to_numpy() / 7.0).reshape(-1, 1)
    return np.hstack([day_of_week, weeks])


def fit_expected(series: pd.Series, exclude: pd.DatetimeIndex | None = None) -> pd.Series:
    """Expected value for every day in `series` (a daily series indexed by date).

    Days in `exclude` are not used for fitting but still get an expected value.
    With too little history, falls back to the day-of-week average with no trend.
    """
    series = series.astype(float)
    dates = pd.DatetimeIndex(series.index)
    train = ~dates.isin(exclude) if exclude is not None else np.ones(len(dates), bool)
    X = _design_matrix(dates, dates.min())

    if train.sum() < MIN_TRAINING_DAYS:
        X = X[:, :7]  # not enough data to estimate a trend reliably
    coef, *_ = np.linalg.lstsq(X[train], series.to_numpy()[train], rcond=None)
    expected = X @ coef
    return pd.Series(np.clip(expected, 0, None), index=series.index, name="expected")


def _robust_sigma(residuals: np.ndarray) -> float:
    """Typical size of a normal day's gap.

    Uses the larger of two estimates: the median absolute deviation (robust to
    outliers we haven't excluded yet) and the plain standard deviation. Taking
    the larger one errs toward fewer false alarms.
    """
    mad_sigma = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
    sigma = max(mad_sigma, float(np.std(residuals)))
    return sigma if sigma > 0 else 1.0


def _best_run(residuals: np.ndarray, sigma: float, blocked: np.ndarray) -> tuple[int, int, float]:
    """Find the run of days [i, j] with the most extreme summed gap.

    Returns (start index, end index, score). Days in `blocked` (already part of
    another period) can't be included.
    """
    best = (0, 0, 0.0)
    n = len(residuals)
    for i in range(n):
        total = 0.0
        for j in range(i, min(n, i + MAX_RUN_DAYS)):
            if blocked[j]:
                break
            total += residuals[j]
            score = total / (sigma * np.sqrt(j - i + 1))
            if abs(score) > abs(best[2]):
                best = (i, j, score)
    return best


def find_unusual_periods(
    series: pd.Series, max_periods: int = 3, threshold: float = SCORE_THRESHOLD
) -> tuple[list[UnusualPeriod], pd.Series]:
    """Find runs of days far from expected, most extreme first.

    Returns (periods, expected) where `expected` is fitted with every unusual
    period left out.
    """
    series = series.astype(float)
    dates = pd.DatetimeIndex(series.index)
    found: list[tuple[int, int]] = []

    for _ in range(max_periods):
        candidate: tuple[int, int] | None = None
        # Refit a few times: once we suspect a period, leave it out of the
        # baseline and look again, until the period stops changing.
        for _refine in range(4):
            excluded = [(i, j) for i, j in found] + ([candidate] if candidate else [])
            exclude_idx = _indices(dates, excluded)
            expected = fit_expected(series, exclude=exclude_idx)
            residuals = (series - expected).to_numpy()
            normal_days = ~dates.isin(exclude_idx)
            sigma = _robust_sigma(residuals[normal_days])
            blocked = dates.isin(_indices(dates, found))
            i, j, score = _best_run(residuals, sigma, blocked)
            if candidate == (i, j):
                break
            candidate = (i, j)
        if abs(score) < threshold:
            break
        found.append(candidate)

    expected = fit_expected(series, exclude=_indices(dates, found))
    periods = [
        UnusualPeriod(
            start=dates[i],
            end=dates[j],
            actual=float(series.iloc[i : j + 1].sum()),
            expected=float(expected.iloc[i : j + 1].sum()),
            score=_score(series, expected, dates, found, i, j),
        )
        for i, j in found
    ]
    return periods, expected


def _indices(dates: pd.DatetimeIndex, runs: list[tuple[int, int]]) -> pd.DatetimeIndex:
    """All dates covered by a list of (start index, end index) runs."""
    picked = [dates[i : j + 1] for i, j in runs]
    return picked[0].append(picked[1:]) if picked else pd.DatetimeIndex([])


def _score(series, expected, dates, found, i, j) -> float:
    """Final score of run [i, j] against the final baseline."""
    residuals = (series - expected).to_numpy()
    normal = ~dates.isin(_indices(dates, found))
    sigma = _robust_sigma(residuals[normal])
    return float(residuals[i : j + 1].sum() / (sigma * np.sqrt(j - i + 1)))
