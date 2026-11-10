"""Work out what each column in an uploaded spreadsheet is for.

Users upload exports from Shopify, Stripe, Square or Google Sheets, so column
names vary. This module guesses each column's role from its name and values:

- date      the column that says when each row happened
- metric    a number worth adding up (revenue, total, amount ...)
- segment   a category to break results down by (device, region ...)
- id        a unique code per row (order ID) - not useful to analyse
- personal  information about a person (email, phone, name, address).
            Personal columns are never analysed and never sent to the AI.

The user can override any guess in the app.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

# Name fragments that suggest a column holds personal information.
PERSONAL_NAME_HINTS = (
    "email", "e-mail", "phone", "mobile_number", "first_name", "last_name",
    "full_name", "customer_name", "name", "address", "street", "zip", "postcode",
    "postal", "ip_address", "birth", "ssn",
)
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_PATTERN = re.compile(r"^\+?[\d\s().-]{7,}$")

# Name fragments that suggest the main money metric, best match first.
METRIC_NAME_HINTS = ("revenue", "total", "sales", "amount", "gross", "net", "price", "value")
DATE_NAME_HINTS = ("date", "time", "created", "placed", "day", "timestamp")

# A text column with at most this many distinct values counts as a segment.
MAX_SEGMENT_VALUES = 30


@dataclass
class ColumnRoles:
    """The detected role of every column in a table."""

    date: str | None = None
    metric: str | None = None  # the main number to explain, e.g. revenue
    other_metrics: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    personal: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Plain dict version, handy for showing in the app or sending to the AI."""
        return {
            "date": self.date,
            "metric": self.metric,
            "other_metrics": self.other_metrics,
            "segments": self.segments,
            "ids": self.ids,
            "personal_skipped": self.personal,
            "ignored": self.ignored,
        }


def _looks_personal(name: str, values: pd.Series) -> bool:
    """True if the column name or its values look like personal information."""
    lowered = name.lower()
    # "name" alone is too broad ("product_name"), so require a person-like word with it.
    for hint in PERSONAL_NAME_HINTS:
        if hint == "name":
            if lowered in ("name", "customer", "customer_name", "billing_name", "shipping_name"):
                return True
        elif hint in lowered:
            return True

    sample = values.dropna().astype(str).head(50)
    if len(sample) == 0:
        return False
    if sample.str.match(EMAIL_PATTERN).mean() > 0.5:
        return True
    if (
        not pd.api.types.is_numeric_dtype(values)
        and sample.str.match(PHONE_PATTERN).mean() > 0.8
        # Dates like "2026-01-07" also match the phone pattern, so rule them out.
        and not _looks_like_date(name, values)
    ):
        return True
    return False


def _looks_like_date(name: str, values: pd.Series) -> bool:
    """True if most values parse as dates (and the column isn't just numbers)."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return True
    if pd.api.types.is_numeric_dtype(values):
        return False
    sample = values.dropna().astype(str).head(200)
    if len(sample) == 0:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    share_parsed = parsed.notna().mean()
    has_date_name = any(h in name.lower() for h in DATE_NAME_HINTS)
    return share_parsed > 0.9 or (has_date_name and share_parsed > 0.6)


def _looks_like_id(name: str, values: pd.Series) -> bool:
    """True for columns like order_id: (almost) every value is different."""
    lowered = name.lower()
    unique_share = values.nunique(dropna=True) / max(len(values.dropna()), 1)
    named_like_id = lowered == "id" or lowered.endswith("_id") or lowered.endswith(" id") or "number" in lowered
    return (named_like_id and unique_share > 0.5) or (
        not pd.api.types.is_numeric_dtype(values) and unique_share > 0.95 and len(values) > 20
    )


def _pick_main_metric(numeric_columns: list[str]) -> str | None:
    """Choose the most revenue-like numeric column."""
    for hint in METRIC_NAME_HINTS:
        for column in numeric_columns:
            if hint in column.lower():
                return column
    return numeric_columns[0] if numeric_columns else None


def detect_columns(df: pd.DataFrame) -> ColumnRoles:
    """Guess the role of every column in `df`.

    Checks run in order of how much harm a wrong guess would do: personal
    information first (so it is never analysed), then dates, IDs, numbers and
    categories.
    """
    roles = ColumnRoles()
    numeric_columns: list[str] = []

    for column in df.columns:
        values = df[column]
        if _looks_personal(column, values):
            roles.personal.append(column)
        elif roles.date is None and _looks_like_date(column, values):
            roles.date = column
        elif _looks_like_id(column, values):
            roles.ids.append(column)
        elif pd.api.types.is_numeric_dtype(values):
            numeric_columns.append(column)
        elif values.nunique(dropna=True) <= MAX_SEGMENT_VALUES:
            roles.segments.append(column)
        else:
            roles.ignored.append(column)  # free text, too many distinct values

    roles.metric = _pick_main_metric(numeric_columns)
    roles.other_metrics = [c for c in numeric_columns if c != roles.metric]
    return roles


def safe_summary(df: pd.DataFrame, roles: ColumnRoles, max_examples: int = 5) -> dict:
    """What the AI is allowed to see about the file: column names, roles and a
    few example values. Personal and ID columns get no examples, and raw rows
    are never included.
    """
    hidden = set(roles.personal) | set(roles.ids)
    columns = {}
    for column in df.columns:
        if column in hidden:
            columns[column] = {"examples": "hidden"}
        else:
            examples = df[column].dropna().astype(str).unique()[:max_examples].tolist()
            columns[column] = {"examples": examples}
    return {"row_count": len(df), "roles": roles.as_dict(), "columns": columns}
