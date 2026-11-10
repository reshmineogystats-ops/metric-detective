# Metric Detective

*An AI analyst that explains why your numbers moved.*

**Turn Monday's "why are we down?" into a verified answer and a ready-to-send update in five minutes, from any spreadsheet export.**

For the person who owns the weekly numbers at a 10–200 person company with no data analyst: an ops lead, chief of staff or founder. Upload an export from Shopify, Stripe, Square or Google Sheets, ask "Why did sales drop last week?", and an AI agent investigates, shows its work and drafts the update for your boss.

> 🚧 In progress. Milestones 1–2 of 6 are built: the analysis engine and the AI agent (command line). The app comes next.

## The core idea: code does the math, the AI decides where to look

Language models are good at planning an investigation and explaining results, but they can make up numbers. So in this tool the AI never calculates anything. It calls analysis tools, plain tested Python functions, and writes its answer only from their outputs. Every number in an answer can be traced to a tool result.

## How the agent works

```
"Why did sales drop last week?"
   → AI picks a tool ─→ Python runs the analysis ─→ result back to the AI   (repeat, at most 8 times)
   → AI submits a structured answer
   → number check: is every number in the answer in a tool result?   no → sent back once to fix
   → headline · explanation · drivers · what was ruled out · confidence · what to check next
```

Guardrails built in:
- **Number check.** Every amount and percentage in the answer must match a tool output (rounding allowed). Made-up numbers are sent back to the AI once, then flagged to the user.
- **Privacy.** The AI sees column names, up to five example values per column and aggregated results. Never raw rows, and never columns that look personal (emails, phones, names, addresses). A test checks that no customer email or order ID appears in anything sent to the model.
- **Cost.** A small, low-cost model (Claude Haiku 5.5) and a cap of 8 analysis steps per question. Each run prints its token use and cost.

## What the analysis engine does today

On the demo store, **Juniper Home Goods** (8 weeks of fictional orders with one planted problem: mobile checkout breaks for Shop Pay at 10 a.m. on Tue Nov 3):

| Question | Tool | Result |
|---|---|---|
| Was anything unusual? | `find_unusual_periods` | Nov 3–8: revenue $7,157 vs $9,294 expected (**−23%**), high confidence |
| Fewer orders or smaller ones? | `decompose` | Fewer orders: 150 vs 202 expected. Average order value is normal. |
| Where did the gap come from? | `segment_gap` | **Mobile + Shop Pay: −78.5%**; the rest of the business +4.1% |
| When did it start? | `drill_down` | Nov 3, **around 10:00** |
| Same time last year? | `same_period_last_year` | Not enough history (says so instead of guessing) |

Personal information never leaves the file: columns that look like emails, phone numbers, names or addresses are detected (by name *and* by value) and dropped before any analysis.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                     # 46 tests, no API key needed
cp .env.example .env                           # then paste your Anthropic API key into .env
.venv/bin/python ask.py "Why did sales drop last week?" --update
```

`ask.py` uses the demo store by default; pass `--file your_export.csv` to use your own data. `scripts/make_sample_data.py` regenerates the demo file.

## Project layout

```
analysis/columns.py    detect date / metric / breakdown / personal columns
analysis/baseline.py   expected value per day (day of week + trend), unusual periods
analysis/tools.py      the six analysis tools
agent/                 the AI loop, prompts, tool descriptions and number check
ask.py                 command-line entry point
scripts/               demo data generator
tests/                 unit tests for every tool
```

## Roadmap

1. ✅ Analysis tools, sample data and tests
2. ✅ AI agent as a command-line script, with a check that every number comes from a tool (live run next)
3. Streamlit app: upload → "here's how I read your file" → ask → watch it investigate → answer
4. Evals: 20 planted problems + 3 clean controls, with results published here
5. Public demo on Streamlit Community Cloud
6. v2 from tester feedback
