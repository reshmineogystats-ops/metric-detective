"""Agent loop tests, with a scripted fake model: no API key, no cost.

The fake plays the AI's part (which tool to call next, what answer to give);
the real analysis tools, number check and loop run as they would live.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from agent.checks import numbers_in_text, unverified_numbers
from agent.loop import DEFAULT_MODEL, Investigation
from analysis.tools import Dataset
from tests.conftest import ROOT

START, END = "2026-11-03", "2026-11-08"


# ---- a fake model ------------------------------------------------------------


def tool(name, **inputs):
    return SimpleNamespace(type="tool_use", id=f"toolu_{name}_{id(inputs)}", name=name, input={"reason": f"run {name}", **inputs})


def text(t):
    return SimpleNamespace(type="text", text=t)


def reply(*blocks, stop_reason=None):
    has_tool = any(b.type == "tool_use" for b in blocks)
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason or ("tool_use" if has_tool else "end_turn"),
        usage=SimpleNamespace(input_tokens=1000, output_tokens=100, cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


class FakeClient:
    """Returns scripted replies in order and records every request it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        # Snapshot the request as JSON, the way it would go over the wire.
        self.requests.append(json.loads(json.dumps(kwargs, default=lambda o: vars(o))))
        return self.replies.pop(0)


GOOD_ANSWER = {
    "headline": "Sales were 23% below expected from Nov 3 because mobile Shop Pay orders almost stopped.",
    "explanation": "Revenue was $2,137 short of expected over Nov 3-8. Mobile shoppers paying with Shop Pay "
    "fell 78.5% while the rest of the business was up 4.1%, which usually means a broken checkout.",
    "drivers": [{"segment": "Mobile + Shop Pay", "impact": "-78.5% vs expected"}],
    "ruled_out": ["Seasonality: not enough history to compare with last year.", "Smaller orders: order value was normal."],
    "confidence": {"level": "high", "reason": "One segment explains the whole gap."},
    "next_checks": ["Test a mobile Shop Pay checkout.", "Check app or theme changes on Nov 3."],
    "numbers_used": ["-23.0% (find_unusual_periods)", "2136.64 (find_unusual_periods)"],
}

INVESTIGATION = [
    reply(tool("find_unusual_periods", metric="revenue")),
    reply(tool("decompose", start=START, end=END), tool("segment_gap", start=START, end=END)),
    reply(
        tool(
            "drill_down",
            filters=[{"column": "device", "value": "Mobile"}, {"column": "payment_method", "value": "Shop Pay"}],
            start=START,
            end=END,
            metric="orders",
        )
    ),
    reply(tool("same_period_last_year", metric="revenue", start=START, end=END)),
]


@pytest.fixture
def raw():
    return pd.read_csv(ROOT / "data" / "sample_shop.csv")


def make_investigation(raw, replies, **kwargs):
    client = FakeClient(replies)
    steps = []
    inv = Investigation(Dataset.from_frame(raw), raw_frame=raw, client=client, on_step=steps.append, **kwargs)
    return inv, client, steps


# ---- the loop ----------------------------------------------------------------


def test_full_investigation_passes_number_check(raw):
    inv, client, steps = make_investigation(raw, INVESTIGATION + [reply(tool("submit_answer", **GOOD_ANSWER))])
    result = inv.ask("Why did sales drop last week?")

    assert result.kind == "answer"
    assert result.unverified == [] and not result.retried
    assert [s.tool for s in steps] == ["find_unusual_periods", "decompose", "segment_gap", "drill_down", "same_period_last_year"]
    assert "2026-11-03 to 2026-11-08" in steps[0].summary
    assert "Mobile + Shop Pay" in steps[2].summary
    assert len(client.requests) == 5


def test_every_tool_call_gets_its_result_back(raw):
    inv, client, _ = make_investigation(raw, INVESTIGATION + [reply(tool("submit_answer", **GOOD_ANSWER))])
    inv.ask("Why did sales drop last week?")
    second_request = client.requests[2]["messages"]
    results = second_request[-1]["content"]
    asked = [b["id"] for b in second_request[-2]["content"] if b["type"] == "tool_use"]
    assert [r["tool_use_id"] for r in results] == asked  # both parallel calls answered in one message


def test_made_up_number_is_sent_back_once(raw):
    bad = {**GOOD_ANSWER, "explanation": GOOD_ANSWER["explanation"] + " That is $9,999 lost."}
    inv, client, _ = make_investigation(
        raw, INVESTIGATION + [reply(tool("submit_answer", **bad)), reply(tool("submit_answer", **GOOD_ANSWER))]
    )
    result = inv.ask("Why did sales drop last week?")
    assert result.retried and result.unverified == []
    feedback = client.requests[-1]["messages"][-1]["content"][-1]
    assert feedback["is_error"] and "9,999" in feedback["content"]


def test_second_failure_keeps_answer_with_warning(raw):
    bad = {**GOOD_ANSWER, "headline": "Revenue fell $9,999."}
    inv, _, _ = make_investigation(
        raw, INVESTIGATION + [reply(tool("submit_answer", **bad)), reply(tool("submit_answer", **bad))]
    )
    result = inv.ask("Why did sales drop last week?")
    assert result.unverified == ["9,999"]
    assert "9,999" in result.warning


def test_stops_running_tools_after_the_limit(raw):
    too_many = [reply(tool("find_unusual_periods", metric="revenue")) for _ in range(10)]
    # The answer quotes -78.5%, which only segment_gap produces. That tool never ran
    # here, so the number check sends the answer back once, then flags it.
    submit = reply(tool("submit_answer", **GOOD_ANSWER))
    inv, client, steps = make_investigation(raw, too_many + [submit, submit], max_tool_calls=3)
    result = inv.ask("Why did sales drop last week?")
    assert len(steps) == 3
    limit_messages = [r for req in client.requests for r in req["messages"][-1]["content"] if isinstance(r, dict)]
    assert any("Limit of 3" in str(r.get("content", "")) for r in limit_messages)
    assert result.retried and "78.5" in result.unverified


def test_bad_tool_input_becomes_an_error_result(raw):
    replies = [
        reply(tool("segment_gap", start=START, end=END, dimensions=["customer_email"])),
        *INVESTIGATION,
        reply(tool("submit_answer", **GOOD_ANSWER)),
    ]
    inv, client, steps = make_investigation(raw, replies, max_tool_calls=8)
    result = inv.ask("Why did sales drop last week?")
    assert steps[0].is_error
    error_result = client.requests[1]["messages"][-1]["content"][0]
    assert error_result["is_error"] and "Unknown breakdown columns" in error_result["content"]
    assert result.kind == "answer"


def test_plain_text_reply_is_nudged_to_submit(raw):
    replies = INVESTIGATION + [reply(text("Mobile checkout broke.")), reply(tool("submit_answer", **GOOD_ANSWER))]
    inv, client, _ = make_investigation(raw, replies)
    assert inv.ask("Why did sales drop last week?").kind == "answer"
    assert "submit_answer" in client.requests[-1]["messages"][-1]["content"]


def test_monday_update_follow_up_reuses_context(raw):
    update = {
        "update": "Revenue was 23% below expected Nov 3-8 because mobile Shop Pay orders fell 78.5%. "
        "We're testing the mobile checkout today and watching orders hourly.",
        "numbers_used": ["-23.0", "-78.5"],
    }
    replies = INVESTIGATION + [reply(tool("submit_answer", **GOOD_ANSWER)), reply(tool("submit_update", **update))]
    inv, client, _ = make_investigation(raw, replies)
    inv.ask("Why did sales drop last week?")
    result = inv.draft_update()
    assert result.kind == "update" and result.unverified == []
    last_messages = client.requests[-1]["messages"]
    assert "Draft my Monday update" in last_messages[-1]["content"]
    assert len(last_messages) > 10  # the whole investigation came along


def test_request_settings(raw):
    inv, client, _ = make_investigation(raw, INVESTIGATION + [reply(tool("submit_answer", **GOOD_ANSWER))])
    inv.ask("Why did sales drop last week?")
    request = client.requests[0]
    assert request["model"] == DEFAULT_MODEL == "claude-haiku-5-5"
    assert request["output_config"] == {"effort": "medium"}
    assert "temperature" not in request
    assert {t["name"] for t in request["tools"]} >= {"find_unusual_periods", "submit_answer", "submit_update"}


# ---- privacy -----------------------------------------------------------------


def test_no_personal_data_or_raw_rows_reach_the_model(raw):
    replies = INVESTIGATION + [reply(tool("submit_answer", **GOOD_ANSWER))]
    inv, client, _ = make_investigation(raw, replies)
    inv.ask("Why did sales drop last week?")
    everything_sent = json.dumps(client.requests)
    assert "@example.com" not in everything_sent  # no customer emails
    assert "JHG-" not in everything_sent  # no order IDs
    assert "customer_email" in everything_sent  # the AI is told the column was skipped, by name only


# ---- number check details ----------------------------------------------------


@pytest.mark.parametrize(
    "written, known, ok",
    [
        ("$2,137", [-2136.64], True),
        ("$2.1k", [-2136.64], True),
        ("$2,100", [-2136.64], True),
        ("23%", [-23.0], True),
        ("$2,000", [-2136.64], False),  # rounded too far
        ("78.4%", [-78.5], False),
        ("$10,000", [9294.03], False),
    ],
)
def test_number_matching(written, known, ok):
    assert (unverified_numbers({"headline": f"Down {written}."}, known) == []) is ok


def test_dates_times_and_small_counts_are_not_checked():
    assert numbers_in_text("Nov 3-8, 2026 from 10 a.m. (10:00 on 2026-11-03), 2 checks over 6 days") == []


def test_numbers_from_the_question_are_allowed():
    assert unverified_numbers({"headline": "Here is what happened in the last 30 days."}, [], "What changed in the last 30 days?") == []
