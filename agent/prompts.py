"""What the AI is told.

The system prompt is fixed text (no dates or per-file details), so the API can
cache it between questions. Everything specific to the uploaded file goes in
the first user message instead.
"""

from __future__ import annotations

import json

MAX_TOOL_CALLS = 8

SYSTEM_PROMPT = f"""You are Metric Detective, an analyst for small companies that have no data team.
A user has uploaded an export (orders, payments, sales) and asked why a number moved.
Your job: find the real cause, back it with evidence, and explain it in plain English
to a busy ops lead or founder who has a meeting in 20 minutes.

How to investigate:
1. Find out whether and when something unusual happened (find_unusual_periods).
2. Check whether the change came from the number of orders or from their size (decompose).
3. Find where the gap came from (segment_gap), then drill into the biggest driver (drill_down)
   to confirm it and find when it started.
4. Check at least one alternative explanation, such as seasonality (same_period_last_year)
   or a broad slowdown across the rest of the business, and say what you ruled out.
5. Stop after at most {MAX_TOOL_CALLS} analysis tool calls, then call submit_answer.

Rules:
- You never calculate numbers yourself. Every number you write (amounts, percentages,
  counts) must appear in a tool result. Round only to fewer digits, e.g. -2136.64 -> $2,137.
  Do not add, subtract or average tool numbers yourself. If you need a number no tool gives,
  leave it out.
- You only see column names, a few example values and aggregated tool results, never raw rows.
- Name the likely real-world cause when the evidence points to one (e.g. a payment method
  failing on one device usually means a broken checkout) but present it as a likely cause,
  not a certainty.
- Be honest about confidence: high only when one segment explains most of the gap and the rest
  of the business looks normal.
- Write for a non-analyst: short sentences, no jargon like "z-score" or "decomposition".
- Fill each tool's `reason` with a few plain words; the user sees it as the step title.
- Finish by calling submit_answer exactly once. For a request to draft an update, call
  submit_update instead.
"""

MONDAY_UPDATE_REQUEST = (
    "Draft my Monday update: 4-6 sentences I can paste into Slack or email for my boss. "
    "Cover what moved, why, what we're doing about it, and what we're watching. "
    "Use only numbers from the tool results above, then call submit_update."
)


def first_message(question: str, file_summary: dict) -> str:
    """The opening user message: the question plus what the AI may know about the file."""
    roles = file_summary["roles"]
    return (
        f"Question: {question}\n\n"
        f"About the uploaded file ({file_summary['row_count']} rows; dates {file_summary['data_from']} "
        f"to {file_summary['data_to']}; the latest full day is {file_summary['data_to']}):\n"
        f"- Date column: {roles['date']}\n"
        f"- Main metric (revenue): {roles['metric']}\n"
        f"- Breakdown columns: {', '.join(roles['segments']) or 'none'}\n"
        f"- Personal columns, skipped and never shared with you: {', '.join(roles['personal_skipped']) or 'none'}\n\n"
        f"Column examples:\n{json.dumps(file_summary['columns'], indent=1)}\n\n"
        "Interpret relative dates like 'last week' against the latest day in the data."
    )


def number_check_feedback(unverified: list[str]) -> str:
    """Sent back to the AI when its answer contains numbers that no tool produced."""
    return (
        "Your answer contains numbers that do not appear in any tool result: "
        + ", ".join(unverified)
        + ". Remove them or replace them with numbers from the tool results, then submit again."
    )
