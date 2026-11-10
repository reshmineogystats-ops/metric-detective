# Why Did It Move?

**Turn Monday's "why are we down?" into a verified answer and a ready-to-send update in five minutes, from any spreadsheet export.**

For the person who owns the weekly numbers at a 10–200 person company with no data analyst: an ops lead, chief of staff or founder. Upload an export from Shopify, Stripe, Square or Google Sheets, ask "Why did sales drop last week?", and an AI agent investigates, shows its work and drafts the update for your boss.

> 🚧 In progress. Milestone 1 of 6 (analysis engine) is done. The AI agent and the app come next.

## The core idea: code does the math, the AI decides where to look

Language models are good at planning an investigation and explaining results, but they can make up numbers. So in this tool the AI never calculates anything. It calls analysis tools, plain tested Python functions, and writes its answer only from their outputs. Every number in an answer can be traced to a tool result.

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
.venv/bin/python scripts/make_sample_data.py   # regenerates data/sample_shop.csv
.venv/bin/python -m pytest                     # 27 tests
```

## Project layout

```
analysis/columns.py    detect date / metric / breakdown / personal columns
analysis/baseline.py   expected value per day (day of week + trend), unusual periods
analysis/tools.py      the six tools the AI agent will call
scripts/               demo data generator
tests/                 unit tests for every tool
```

## Roadmap

1. ✅ Analysis tools, sample data and tests
2. AI agent as a command-line script, with a check that every number comes from a tool
3. Streamlit app: upload → "here's how I read your file" → ask → watch it investigate → answer
4. Evals: 20 planted problems + 3 clean controls, with results published here
5. Public demo on Streamlit Community Cloud
6. v2 from tester feedback
