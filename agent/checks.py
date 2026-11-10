"""Number check: every number in the AI's answer must come from a tool result.

This is the guardrail behind "every number is traceable". After the AI submits
an answer, we pull every number out of its text and look for a matching number
in the tool outputs it received.

What counts as a match:
- The same value, ignoring sign ("down $2,137" matches a gap of -2136.64).
- Rounded to the precision the AI wrote: "23%" matches -23.0, "$2,137" matches
  2136.64, "$2,100" matches 2136.64 (rounded to the nearest hundred), "$2.1k"
  matches 2136.64. Rounding is capped at 3% so "about $10,000" can't pass for 9,294.

What is not checked (it isn't data):
- Dates and times ("Nov 3", "2026-11-03", "10:00", "10 a.m.").
- Small whole numbers up to 12 ("2 checks", "the last 6 days").
- Numbers that appear in the user's own question.
"""

from __future__ import annotations

import json
import re
from typing import Any

MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
# Things that look like numbers but are dates or times. Removed before checking.
DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b"),  # 2026-11-03, 2026-11-03 10:00
    re.compile(rf"\b{MONTHS}\s+\d{{1,2}}(?:\s*[-–]\s*\d{{1,2}})?(?:,?\s+\d{{4}})?\b", re.I),  # Nov 3, Nov 3-8, Nov 3, 2026
    re.compile(rf"\b\d{{1,2}}\s+{MONTHS}\b", re.I),  # 3 Nov
    re.compile(r"\b\d{1,2}:\d{2}\b"),  # 10:00
    re.compile(r"\b\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)", re.I),  # 10 a.m., 10am
    re.compile(r"\b(?:19|20)\d{2}\b"),  # years
]
# A number, with optional thousands separators, decimals and a k/m suffix.
NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*([kKmM])?(?![\w])")
SMALL_INT_LIMIT = 12
MAX_ROUNDING = 0.03  # rounding may change a value by at most 3%


def numbers_in_text(text: str) -> list[tuple[str, float, float]]:
    """Find numbers in text that should be checked.

    Returns (as written, value, tolerance) for each. The tolerance is how far
    the true value may be from what was written, given the precision used.
    """
    for pattern in DATE_PATTERNS:
        text = pattern.sub(" ", text)

    found = []
    for match in NUMBER.finditer(text):
        whole, decimals, suffix = match.group(1), match.group(2) or "", match.group(3) or ""
        value = float(whole.replace(",", "") + decimals)
        written = match.group(0).strip()

        if decimals:
            # "78.5" was rounded to one decimal place, so it may be off by 0.05.
            tolerance = 0.5 * 10 ** -(len(decimals) - 1)
        else:
            # "2,100" is rounded to the nearest hundred; "2,137" to the nearest one.
            digits = whole.replace(",", "")
            trailing_zeros = len(digits) - len(digits.rstrip("0"))
            tolerance = max(min(0.5 * 10**trailing_zeros, MAX_ROUNDING * value), 0.5)

        multiplier = {"k": 1_000, "m": 1_000_000}.get(suffix.lower(), 1)
        value, tolerance = value * multiplier, tolerance * multiplier

        if not suffix and not decimals and value <= SMALL_INT_LIMIT:
            continue
        found.append((written, value, tolerance))
    return found


def numbers_in_results(results: list[Any]) -> list[float]:
    """Every number in the tool outputs (recursively, including inside text)."""
    values: list[float] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            values.append(abs(float(obj)))
        elif isinstance(obj, str):
            values.extend(value for _, value, _ in numbers_in_text(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(results)
    return values


def answer_text(answer: dict) -> str:
    """The parts of an answer the user reads (numbers_used is the AI's own list, so skip it)."""
    readable = {k: v for k, v in answer.items() if k != "numbers_used"}
    return json.dumps(readable, ensure_ascii=False)


def unverified_numbers(answer: dict, tool_results: list[Any], question: str = "") -> list[str]:
    """Numbers in the answer that don't match any tool result. Empty list = passed."""
    known = numbers_in_results(tool_results)
    in_question = {value for _, value, _ in numbers_in_text(question)}

    problems = []
    for written, value, tolerance in numbers_in_text(answer_text(answer)):
        if value in in_question:
            continue
        if not any(abs(value - k) <= tolerance + 1e-9 for k in known):
            problems.append(written)
    # Keep order, drop duplicates.
    return list(dict.fromkeys(problems))
