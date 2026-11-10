"""The tools the AI can call: their descriptions (what the AI reads), how to
run them, and a one-line summary of each result (what the user sees while the
investigation runs).

Every tool also takes a `reason` (why the AI is running it, in a few words).
That becomes the step title in the "Investigating..." view, so the user can
follow the AI's thinking.

The summaries are written by code from the tool output, not by the AI, so the
progress view can't show a made-up number either.
"""

from __future__ import annotations

import json
from typing import Any

from analysis import tools as analysis_tools
from analysis.tools import Dataset

REASON = {
    "type": "string",
    "description": "Why you are running this step, in under 10 words. Shown to the user as the step title.",
}
DATE = {"type": "string", "description": "Date as YYYY-MM-DD (inclusive)."}
METRIC = {
    "type": "string",
    "enum": ["revenue", "orders", "average_order_value"],
    "description": "revenue = sum of the main money column; orders = number of rows; average_order_value = revenue / orders.",
}
FILTERS = {
    "type": "array",
    "description": "Slice to look at, e.g. [{\"column\": \"device\", \"value\": \"Mobile\"}]. Columns must be breakdown columns.",
    "items": {
        "type": "object",
        "properties": {"column": {"type": "string"}, "value": {"type": "string"}},
        "required": ["column", "value"],
        "additionalProperties": False,
    },
}


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {"reason": REASON, **properties},
        "required": ["reason", *required],
        "additionalProperties": False,
    }


ANALYSIS_TOOLS: list[dict] = [
    {
        "name": "find_unusual_periods",
        "description": (
            "Find runs of days where a metric was far from expected (adjusted for day of week and trend). "
            "Use first to see whether and when something unusual happened. Returns each period's dates, "
            "actual, expected, gap, gap_pct and a confidence label."
        ),
        "input_schema": _schema({"metric": METRIC}, ["metric"]),
    },
    {
        "name": "compare_to_expected",
        "description": "Actual vs expected for a metric over a date range, with the gap and % gap.",
        "input_schema": _schema({"metric": METRIC, "start": DATE, "end": DATE}, ["metric", "start", "end"]),
    },
    {
        "name": "decompose",
        "description": (
            "Split a revenue gap into an order-count effect (fewer or more orders) and an order-value effect "
            "(smaller or larger orders). The two effects add up to the revenue gap."
        ),
        "input_schema": _schema({"start": DATE, "end": DATE}, ["start", "end"]),
    },
    {
        "name": "segment_gap",
        "description": (
            "Where did the revenue gap come from? For each breakdown column (and pairs of columns), each value's "
            "actual vs expected and its share of the total gap. `top_driver` is the most specific segment that "
            "explains most of the gap, plus how the rest of the business did."
        ),
        "input_schema": _schema(
            {
                "start": DATE,
                "end": DATE,
                "dimensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Breakdown columns to check. Omit or pass [] for all of them.",
                },
            },
            ["start", "end"],
        ),
    },
    {
        "name": "drill_down",
        "description": (
            "One slice (e.g. device = Mobile and payment_method = Shop Pay) day by day against its own expected "
            "line, with the day the change started and, if the file has times, the estimated hour."
        ),
        "input_schema": _schema(
            {"filters": FILTERS, "start": DATE, "end": DATE, "metric": {**METRIC, "enum": ["revenue", "orders"]}},
            ["filters", "start", "end", "metric"],
        ),
    },
    {
        "name": "same_period_last_year",
        "description": (
            "Compare a period with the same weekdays 52 weeks earlier, to check for seasonality. "
            "Says 'not enough history' if the data doesn't go back that far."
        ),
        "input_schema": _schema(
            {"metric": {**METRIC, "enum": ["revenue", "orders"]}, "start": DATE, "end": DATE},
            ["metric", "start", "end"],
        ),
    },
]

SUBMIT_ANSWER = {
    "name": "submit_answer",
    "description": (
        "Submit your final answer. Call this exactly once, when you are done investigating. "
        "Every number you write must come from a tool result above."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "headline": {"type": "string", "description": "One sentence: what happened and why."},
            "explanation": {"type": "string", "description": "Plain-English explanation, under 120 words."},
            "drivers": {
                "type": "array",
                "description": "The segments that caused the change, biggest first.",
                "items": {
                    "type": "object",
                    "properties": {"segment": {"type": "string"}, "impact": {"type": "string"}},
                    "required": ["segment", "impact"],
                    "additionalProperties": False,
                },
            },
            "ruled_out": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explanations you checked and ruled out, each with the evidence.",
            },
            "confidence": {
                "type": "object",
                "properties": {"level": {"type": "string", "enum": ["high", "medium", "low"]}, "reason": {"type": "string"}},
                "required": ["level", "reason"],
                "additionalProperties": False,
            },
            "next_checks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-3 concrete things the user should check or do next.",
            },
            "numbers_used": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Every number in your answer, each with the tool it came from, e.g. '-23.0% (find_unusual_periods)'.",
            },
        },
        "required": ["headline", "explanation", "drivers", "ruled_out", "confidence", "next_checks", "numbers_used"],
        "additionalProperties": False,
    },
}

SUBMIT_UPDATE = {
    "name": "submit_update",
    "description": "Submit the drafted Monday update. Every number must come from a tool result above.",
    "input_schema": {
        "type": "object",
        "properties": {
            "update": {
                "type": "string",
                "description": "4-6 sentences: what moved, why, what we're doing, what we're watching.",
            },
            "numbers_used": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["update", "numbers_used"],
        "additionalProperties": False,
    },
}

ALL_TOOLS = ANALYSIS_TOOLS + [SUBMIT_ANSWER, SUBMIT_UPDATE]
FINAL_TOOLS = {"submit_answer", "submit_update"}


# --------------------------------------------------------------------------
# Running a tool
# --------------------------------------------------------------------------


def run_tool(ds: Dataset, name: str, tool_input: dict[str, Any]) -> dict:
    """Run one analysis tool with the AI's arguments. Raises ValueError on bad input."""
    args = {k: v for k, v in tool_input.items() if k != "reason"}
    if name == "find_unusual_periods":
        return analysis_tools.find_unusual_periods(ds, args["metric"])
    if name == "compare_to_expected":
        return analysis_tools.compare_to_expected(ds, args["metric"], args["start"], args["end"])
    if name == "decompose":
        return analysis_tools.decompose(ds, args["start"], args["end"])
    if name == "segment_gap":
        return analysis_tools.segment_gap(ds, args["start"], args["end"], dimensions=args.get("dimensions") or None)
    if name == "drill_down":
        filters = {f["column"]: f["value"] for f in args["filters"]}
        return analysis_tools.drill_down(ds, filters, args["start"], args["end"], metric=args.get("metric", "revenue"))
    if name == "same_period_last_year":
        return analysis_tools.same_period_last_year(ds, args["metric"], args["start"], args["end"])
    raise ValueError(f"Unknown tool: {name}")


def to_json(result: dict) -> str:
    """Compact JSON for the AI to read."""
    return json.dumps(result, separators=(",", ":"), default=str)


# --------------------------------------------------------------------------
# One-line summaries for the progress view
# --------------------------------------------------------------------------


def _money(x: float) -> str:
    return f"${abs(x):,.0f}"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.1f}%"


def _segment_name(segment: dict) -> str:
    return " + ".join(segment.values())


def summarize(name: str, result: dict) -> str:
    """One short line describing a tool result, for the user."""
    try:
        if name == "find_unusual_periods":
            periods = result["periods"]
            if not periods:
                return f"No unusual periods in {result['metric']}"
            p = periods[0]
            more = f" (+{len(periods) - 1} more)" if len(periods) > 1 else ""
            return f"{p['start']} to {p['end']}: {result['metric']} {_pct(p['gap_pct'])} vs expected, {p['confidence']} confidence{more}"
        if name == "compare_to_expected":
            return f"{result['metric']}: {result['actual']:,} vs {result['expected']:,} expected ({_pct(result['gap_pct'])})"
        if name == "decompose":
            return (
                f"Main cause: {result['main_cause']}. Orders {_pct(result['orders']['gap_pct'])}, "
                f"average order value {_pct(result['average_order_value']['gap_pct'])}"
            )
        if name == "segment_gap":
            driver = result["top_driver"]
            if not driver:
                return "No single segment explains most of the gap"
            return (
                f"Top driver: {_segment_name(driver['segment'])} ({_pct(driver['gap_pct'])}), "
                f"rest of business {_pct(driver['rest_of_business_gap_pct'])}"
            )
        if name == "drill_down":
            when = result.get("change_started_around") or result.get("change_started_on") or "no clear start"
            return f"{_segment_name(result['filters'])}: {_pct(result['gap_pct'])} vs expected; change started {when}"
        if name == "same_period_last_year":
            if result["status"] != "ok":
                return f"Not enough history (data starts {result['history_starts']})"
            return f"vs same period last year: {_pct(result['change_pct'])}"
    except (KeyError, TypeError):
        pass
    return "Done"
