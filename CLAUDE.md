# Metric Detective — project brief

Read this first in every session. It is the product spec for this repo.

## About me and how to work with me
- I'm a product manager (UCLA Anderson MBA). I know Python and SQL basics but I'm not a professional engineer.
- Explain what you're about to do in plain language before doing it, and summarize what changed after.
- Keep code simple, readable and commented. Small steps I can review. Ask before adding a large dependency or changing the architecture below.
- This is a portfolio project: a clean README, clear commit messages and working tests matter as much as features.
- Commit after each working step. Never commit secrets.

## What we're building
**One line:** turn Monday's "why are we down?" into a verified answer and a ready-to-send update in five minutes, from any spreadsheet export.

**User:** the person who owns the weekly numbers at a 10–200 person company with no data analyst (ops lead, chief of staff, founder, growth PM at an online brand or early-stage startup).

**Job to be done:** when a key number moves before a leadership meeting or weekly update, I want to find the real cause and explain it with evidence, so I can act fast and be credible in front of my boss.

**Before → after:** 1–3 hours across Shopify, Stripe, Google Analytics and pivot tables, ending in a guess → upload an export, ask a question, get a cause backed by numbers plus a ready-to-paste update and next actions.

**Why not Shopify Sidekick / Triple Whale / Tableau Pulse / Amplitude?** Those explain changes only inside their own product and stop at the insight. This tool:
1. Works from exports of any tool with no setup, and can combine two files (e.g., orders + ad spend).
2. Shows its work: every number is computed by code and traceable; the answer lists what it ruled out and a confidence level.
3. Ends in communication and action: drafts the weekly update and the note to whoever needs to fix it.

**Not in v1:** dashboards, real-time monitoring, marketing attribution, user accounts.

## Core design principle
**Code does the math. The AI decides where to look and explains.** The language model never calculates numbers itself. It calls analysis tools (plain Python functions) and writes the answer only from their outputs.

## Architecture
```
app.py                  Streamlit UI
analysis/               pure Python analysis tools (pandas, numpy), no AI
  columns.py            detect date / metric / segment / personal-info columns
  baseline.py           expected value per day (day-of-week + trend), flag unusual periods
  tools.py              compare_to_expected, decompose, segment_gap, drill_down, same_period_last_year
agent/
  loop.py               tool-calling loop with the Anthropic Messages API
  prompts.py            system prompt and answer format
  checks.py             number check: every number in the answer must come from a tool output
feedback/               thumbs up/down log (question + rating only, never data)
scripts/make_sample_data.py   generates the demo shop data
evals/                  planted-anomaly test sets and scoring
tests/                  pytest unit tests for analysis/
```

### Analysis tools (the agent can call these)
- `find_unusual_periods(metric)` — expected value per day adjusted for day of week and trend; flag days or runs of days far from expected (robust z-score). Return actual, expected, gap and dates.
- `compare_to_expected(metric, start, end)` — actual vs expected for a period, with % gap.
- `decompose(start, end)` — split the revenue gap into number of orders × average order value.
- `segment_gap(dimensions, start, end)` — each segment value's share of the gap vs expected, per dimension, plus pairs of dimensions (e.g. device + payment method). Returns a `top_driver`: the most specific segment that accounts for at least 60% of the gap, and how the rest of the business did.
- `drill_down(filters, start, end)` — the metric per day for a filtered slice (e.g., device=Mobile, payment=Shop Pay), with the day the change started and, when the file has times, the estimated hour.
- `same_period_last_year(metric, start, end)` — returns "not enough history" if the data doesn't go back that far.
- Stretch: `join_second_file(...)` — e.g., ad spend by day and channel, to test causes outside the orders data.

### Agent loop
- Model name comes from an environment variable (default: a small, low-cost Claude model, e.g. `claude-haiku-5-5`).
- System prompt: plan the investigation, call tools, drill into the biggest driver, check one alternative explanation, stop after at most 8 tool calls.
- Final answer is structured JSON: `headline`, `explanation` (under 120 words), `drivers`, `ruled_out`, `confidence` (high / medium / low with a reason), `next_checks` (2–3), `numbers_used`.
- Stream each step to the UI as it happens (step title, tool name, one-line result).
- Follow-ups reuse the same context. Built-in follow-up: **"Draft my Monday update"** → 4–6 sentences: what moved, why, what we're doing, what we're watching.

### Guardrails
- **Number check:** every number in the answer must appear in tool outputs (allow rounding). If not, retry once, then show the answer with a warning.
- **Privacy:** the model sees column names, at most 5 example values per non-personal column, and aggregated tool outputs. Never raw rows. Columns that look personal (email, phone, name, address) are skipped and never sent. Say this on the upload screen.
- **Cost control:** API key from `.env` locally and Streamlit secrets when deployed; a per-browser limit of 10 questions a day; cache answers for the sample dataset.

## Data
- **Demo data:** `scripts/make_sample_data.py` generates `data/sample_shop.csv` for a fictional store, Juniper Home Goods: 8 weeks of orders (Sep 14 – Nov 8, 2026), about 1,800 orders, average order around $48. Columns: `order_date`, `order_id`, `total`, `device`, `payment_method`, `region`, `sales_channel`, `customer_email` (fake). Planted problem: from 10 a.m. on Tue Nov 3, mobile + Shop Pay orders drop from about 11 a day to 1–2 a day; everything else stays normal. Fixed seed (56). Actual generated file: 1,747 orders, average $47.80. Flagged period Nov 3–8: $7,157 vs $9,294 expected (−23%). Full week Nov 2–8: $8,800 vs $10,857 expected (−19%). Mobile + Shop Pay: −78.5%; rest of business +4.1%.
- **Evals:** generate 20 planted-anomaly datasets with known causes (different dimensions, sizes, start days; order-count drops vs order-value drops; one-segment vs two-segment combinations) plus 3 control datasets with no real anomaly.
- **Optional real-world sample:** the Olist Brazilian e-commerce public dataset (CC BY-NC-SA 4.0; credit Olist in the README).

## User flow (matches the clickable prototype)
1. Home: speaks to the Monday-meeting moment. Upload CSV, or "Try sample data". Privacy note.
2. "Here's how I read your file": detected column roles (date, revenue, order count, breakdowns, skipped personal info); user can change any.
3. Overview: daily chart with expected line and flagged unusual periods; headline number and % vs expected; ask box with 3 suggested questions.
4. Investigating: steps appear one by one (title, tool, result).
5. Answer: headline, explanation, small chart of the driving segment, "where the gap came from" bars by dimension, what I ruled out, confidence, what to check next, money at stake, "how I got this" (expandable), thumbs up/down.
6. Follow-ups, including "Draft my Monday update", "When exactly did it start?", "Was this the same last year?".

Prototype (for my reference): https://claude.ai/artifact/8HasyGthHLi9K33CptyRFq

## Milestones
1. **Analysis tools + sample data + tests.** Done when, on the sample data, the tools flag Nov 3–8 and rank mobile + Shop Pay as the top driver, and `pytest` passes.
2. **Agent loop as a command-line script:** `python ask.py "Why did sales drop last week?"` prints the steps and a structured answer that passes the number check.
3. **Streamlit app** following the user flow above. Run locally.
4. **Evals:** run the 20 planted cases + 3 controls; save a results table (cause found, start date right, false alarms) and add it to the README.
5. **Deploy** to Streamlit Community Cloud (key in secrets). README: problem, who it's for, demo GIF, results, "why not Sidekick?", how-it-works diagram, what's next.
6. **v2 from user feedback** after 10–20 testers.

## Decisions log
- **Unusual-period detection (Milestone 1):** scans every run of 1–14 consecutive days and scores the summed gap as a robust z-score. Bar set at 5.5 (high confidence at 7+), from simulation: on 30 clean copies of the demo shop, the top score was almost always below 5.5; with the planted problem it was usually 7–12. Re-check in Milestone 4 evals.
- **Segment expected values** use each segment's share of revenue over the 28 days before the period, times total expected revenue.
- **Personal-info detection** checks values too (emails, phone numbers), not just column names, and never mistakes date values for phone numbers.

## Status
- Milestone 1: done (27 tests passing).

## Success metrics to track
Answers rated helpful per week (north star); time to explanation vs the user's usual way; accuracy on planted cases; % of new users who upload their own data and ask a question; % of answers whose update gets copied; % of users who return the next week.

## Conventions
- Python 3.11+ (developed on 3.15, in a `.venv` virtual environment). Dependencies in `requirements.txt` (pandas, numpy, pytest now; streamlit, anthropic, python-dotenv added in later milestones). statsmodels was dropped: the baseline is a small least-squares fit in numpy.
- `.env` and any data files users upload are in `.gitignore`. Never print, log or commit API keys.
- Type hints and docstrings on analysis functions. Unit tests for every analysis tool.
- Keep this file up to date when decisions change.
