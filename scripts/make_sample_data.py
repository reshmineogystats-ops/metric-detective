"""Generate the demo dataset for a fictional store, Juniper Home Goods.

Eight weeks of orders, Mon Sep 14 - Sun Nov 8, 2026, with one planted problem:
from 10 a.m. on Tue Nov 3, mobile shoppers paying with Shop Pay mostly stop
completing orders (about 11 a day falls to 1-2 a day). Everything else stays
normal. That is what a broken mobile checkout looks like in an orders export.

The random seed is fixed, so running this script always produces the same file.

Usage:
    python scripts/make_sample_data.py            # writes data/sample_shop.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 56
START = pd.Timestamp("2026-09-14")  # a Monday
END = pd.Timestamp("2026-11-08")  # a Sunday
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "sample_shop.csv"

# Typical day: about 34 orders. Weekends are a bit busier.
BASE_ORDERS_PER_DAY = 31.0
WEEKLY_GROWTH = 0.01  # +1% a week, so the baseline has to handle a trend
DAY_OF_WEEK_FACTOR = {0: 0.97, 1: 0.95, 2: 0.96, 3: 1.00, 4: 1.04, 5: 1.10, 6: 0.98}
DAILY_NOISE = 0.06  # day-to-day wobble in order count (6%)

# Order value: lognormal with a mean of about $48.
MEAN_ORDER_VALUE = 48.0
ORDER_VALUE_SPREAD = 0.38

DEVICES = {"Mobile": 0.56, "Desktop": 0.37, "Tablet": 0.07}
# Payment method depends on device: mobile shoppers lean on Shop Pay.
PAYMENT_BY_DEVICE = {
    "Mobile": {"Shop Pay": 0.58, "Credit card": 0.20, "Apple Pay": 0.14, "PayPal": 0.08},
    "Desktop": {"Credit card": 0.50, "PayPal": 0.25, "Shop Pay": 0.20, "Apple Pay": 0.05},
    "Tablet": {"Credit card": 0.40, "Shop Pay": 0.35, "PayPal": 0.15, "Apple Pay": 0.10},
}
REGIONS = {"West": 0.34, "Northeast": 0.24, "South": 0.26, "Midwest": 0.16}
CHANNELS = {"Online store": 0.86, "Instagram shop": 0.09, "Pinterest": 0.05}

# Share of the day's orders placed in each hour (quiet overnight, busy evenings).
HOURLY_WEIGHTS = np.array(
    [1, 0.6, 0.4, 0.3, 0.3, 0.5, 1, 2, 3, 4, 4.5, 4.5,
     4.5, 4.5, 4.2, 4, 4, 4.5, 5, 5.5, 5.5, 4.5, 3, 2]
)
HOURLY_WEIGHTS = HOURLY_WEIGHTS / HOURLY_WEIGHTS.sum()

# The planted problem.
INCIDENT_START = pd.Timestamp("2026-11-03 10:00")
INCIDENT_KEEP_RATE = 0.13  # only ~13% of mobile + Shop Pay orders still go through


def _pick(rng: np.random.Generator, options: dict[str, float], size: int) -> np.ndarray:
    """Randomly choose `size` values from {option: probability}."""
    return rng.choice(list(options), size=size, p=list(options.values()))


def make_orders(seed: int = SEED, with_incident: bool = True) -> pd.DataFrame:
    """Build the orders table. Set with_incident=False for a no-problem version."""
    rng = np.random.default_rng(seed)
    rows = []
    order_number = 100_000

    for day in pd.date_range(START, END, freq="D"):
        weeks_in = (day - START).days / 7
        expected = (
            BASE_ORDERS_PER_DAY
            * DAY_OF_WEEK_FACTOR[day.day_of_week]
            * (1 + WEEKLY_GROWTH) ** weeks_in
        )
        n_orders = max(0, round(expected * (1 + rng.normal(0, DAILY_NOISE))))

        devices = _pick(rng, DEVICES, n_orders)
        hours = rng.choice(24, size=n_orders, p=HOURLY_WEIGHTS)
        minutes = rng.integers(0, 60, size=n_orders)
        # Lognormal mean = exp(mu + sigma^2 / 2), so solve for mu.
        mu = np.log(MEAN_ORDER_VALUE) - ORDER_VALUE_SPREAD**2 / 2
        totals = rng.lognormal(mu, ORDER_VALUE_SPREAD, size=n_orders)

        for i in range(n_orders):
            device = devices[i]
            payment = _pick(rng, PAYMENT_BY_DEVICE[device], 1)[0]
            placed_at = day + pd.Timedelta(hours=int(hours[i]), minutes=int(minutes[i]))

            # The broken checkout: most mobile + Shop Pay orders never happen.
            if (
                with_incident
                and placed_at >= INCIDENT_START
                and device == "Mobile"
                and payment == "Shop Pay"
                and rng.random() > INCIDENT_KEEP_RATE
            ):
                continue

            order_number += 1
            rows.append(
                {
                    "order_date": placed_at,
                    "order_id": f"JHG-{order_number}",
                    "total": round(float(totals[i]), 2),
                    "device": device,
                    "payment_method": payment,
                    "region": _pick(rng, REGIONS, 1)[0],
                    "sales_channel": _pick(rng, CHANNELS, 1)[0],
                    # Fake emails: present so the app can show it skips personal info.
                    "customer_email": f"customer{rng.integers(1, 1400)}@example.com",
                }
            )

    orders = pd.DataFrame(rows).sort_values("order_date").reset_index(drop=True)
    orders["order_date"] = orders["order_date"].dt.strftime("%Y-%m-%d %H:%M")
    return orders


def main() -> None:
    orders = make_orders()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    orders.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(orders):,} orders to {OUTPUT}")
    print(f"Average order value: ${orders['total'].mean():.2f}")


if __name__ == "__main__":
    main()
