# AGENTS.md — handoff for coding agents (Codex, etc.)

This repo was started with Claude Code. Read this file first, then **`CLAUDE.md`**, which is the
full product spec (it is named for Claude Code, but it applies to any agent working here).
If the two ever disagree, `CLAUDE.md` wins for product decisions; this file wins for current status.

## Who you're working with

- Reshmi Neogy, product manager (UCLA Anderson MBA), building a public PM portfolio. Knows Python
  and SQL basics; not a professional engineer.
- This repo is linked from job applications. A clean README, small commits with clear messages
  and passing tests matter as much as features.
- **Before writing code, explain the plan in plain language and wait for a go-ahead.** After,
  summarize what changed.
- Ask before adding a large dependency, changing the architecture, or switching AI providers.
- Commit after each working step. Never commit secrets. Don't push without asking.

## What the product is (one paragraph)

**Metric Detective**: upload a sales/orders export (Shopify, Stripe, Google Sheets…), ask
"Why did sales drop last week?", and an AI agent investigates, shows its work, and drafts a
ready-to-send update for your boss. For ops leads and founders at 10–200 person companies with
no data analyst. **Core rule: code does the math, the AI decides where to look and explains.**
The model never calculates numbers; it calls tested Python tools and every number in its answer
must trace back to a tool output (enforced by a number check).

## Current status (as of 2026-11-10)

| Milestone | State |
|---|---|
| 1. Analysis tools + sample data + tests | ✅ Done |
| 2. Agent loop as a command-line script (`ask.py`) | ✅ Built, ✅ tested offline with a fake model, ❌ **never run against a real model** (no API key yet) |
| 3. Streamlit app | Not started |
| 4. Evals (20 planted cases + 3 controls) | Not started |
| 5. Deploy to Streamlit Community Cloud | Not started |
| 6. v2 from tester feedback | Not started |

- 46 tests pass: `.venv/bin/python -m pytest`
- GitHub: `github.com/reshmineogystats-ops/metric-detective` (branch `main`). The Milestone 2
  commits may not be pushed yet — check `git status -sb`.
- No `.env` / API key exists yet. The user does not want to buy API credit right now.

## Setup

```bash
python3 -m venv .venv                      # Python 3.11+; developed on 3.15 (macOS)
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                 # 46 tests, no API key needed
.venv/bin/python scripts/make_sample_data.py   # regenerates data/sample_shop.csv (fixed seed 56)
```

## Map of the code

```
analysis/                 pure Python, no AI
  columns.py              detect date / metric / breakdown / ID / personal columns; safe_summary()
  baseline.py             expected value per day (day of week + trend); find runs of unusual days
  tools.py                Dataset class + 6 tools: find_unusual_periods, compare_to_expected,
                          decompose, segment_gap, drill_down, same_period_last_year
agent/
  tool_specs.py           tool schemas the model reads (Anthropic format: name/description/input_schema),
                          run_tool() dispatcher, summarize() one-line step text written by code
  prompts.py              SYSTEM_PROMPT, first_message(), Monday update request, number-check feedback
  checks.py               number check: unverified_numbers(answer, tool_outputs, question)
  loop.py                 Investigation class: ask(), draft_update(); the tool-calling loop
ask.py                    CLI: python ask.py "Why did sales drop last week?" [--update] [--file x.csv]
scripts/make_sample_data.py   demo store "Juniper Home Goods"
tests/                    test_columns, test_baseline, test_tools, test_agent (scripted FakeClient)
```

**What the demo data should produce** (tests assert these): unusual period Nov 3–8, 2026,
revenue $7,157 vs $9,294 expected (−23.0%); cause = fewer orders (AOV normal); top driver
device=Mobile + payment_method=Shop Pay at −78.5%, rest of business +4.1%; change started
2026-11-03 around 10:00; last-year comparison = "not enough history".

## How the agent loop works (agent/loop.py)

1. First user message = question + file summary (column names, roles, ≤5 example values per
   non-personal column). Raw rows are never sent.
2. Model calls analysis tools (each takes a `reason` string → shown as the step title). Our code
   runs them and returns compact JSON. Max 8 analysis calls; after that tools return an error.
3. Model finishes by calling the `submit_answer` tool (fields: headline, explanation, drivers,
   ruled_out, confidence{level,reason}, next_checks, numbers_used) — or `submit_update` for the
   Monday update follow-up.
4. Number check: if any number in the answer isn't in a tool output (sign ignored, rounding to the
   written precision allowed, capped at 3%; dates/times/ints ≤12/numbers in the question exempt),
   the answer is sent back once as an error tool_result; a second failure is kept with a warning.
5. The whole conversation is kept, so follow-ups reuse context.

The **only** code that talks to an AI provider is `Investigation._client()` and
`Investigation._call_model()` in `agent/loop.py`, plus the response shape the loop reads:
`response.content` (blocks with `.type` "text" / "tool_use", `.id`, `.name`, `.input`),
`response.stop_reason`, `response.usage`. Tests replace the client with `FakeClient` in
`tests/test_agent.py`, so the loop logic is provider-independent in practice.

Current settings: Anthropic Python SDK (`anthropic>=2.8`), model `claude-haiku-5-5`, effort
`medium`, key from `ANTHROPIC_API_KEY` in `.env` (git-ignored; template in `.env.example`).

## Things learned the hard way (don't undo these)

- **False alarms:** the unusual-period scan tests hundreds of day-runs, so the threshold is 5.5
  (high confidence ≥ 7), set from 30 clean simulated datasets. A lower bar flagged random wobbles.
- **Dates looked like phone numbers** (`2025-01-07` matched the phone regex) and got hidden as
  personal info. Fixed in `columns.py`; a test covers it.
- Segment expected values = segment's revenue share over the 28 days before the period × total
  expected revenue. `top_driver` = most specific segment (single or pair of columns) explaining
  ≥60% of the gap.
- Personal columns are dropped inside `Dataset.from_frame`, so no tool can read them. A test checks
  no email or order ID appears in anything sent to the model — keep that test passing.

## Open decisions / likely next steps (ask the user which)

1. **Run Milestone 2 live.** Needs an API key. Options the user is weighing:
   - Get an Anthropic key ($5 credit, monthly limit set) — no code changes.
   - **Switch the agent to OpenAI** (e.g. if they'd rather use OpenAI credit). If asked: add a
     provider adapter rather than rewriting the loop — convert `ALL_TOOLS` to OpenAI function-tool
     format, call the OpenAI API in `_call_model`, and map the response back into the
     content/tool_use/stop_reason shape the loop reads (or refactor the loop around a small
     provider interface). Keep the number check, 8-call cap, privacy rules and all tests. Note:
     a ChatGPT subscription does not include API access; OpenAI API usage is billed separately.
2. **No-AI fallback mode:** a template that writes a readable answer from the tool outputs when
   there is no key / the API fails. Useful for the public demo; not a replacement for the agent.
3. **Milestone 3: Streamlit app** following the user flow in `CLAUDE.md` (home → "here's how I
   read your file" → overview chart → live investigation steps → answer → follow-ups), using
   `Investigation(on_step=...)` to stream steps.

## Conventions

- Type hints and docstrings; comments written for a reader who isn't an engineer.
- Commit messages: imperative subject line ("Add …"), short body saying what and why.
- Update `CLAUDE.md` (decisions log + status) and this file when decisions or status change.
- `.env`, uploaded data and `.venv/` are git-ignored. `data/sample_shop.csv` is committed.
