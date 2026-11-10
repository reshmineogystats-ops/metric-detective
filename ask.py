"""Ask Metric Detective a question from the command line.

Usage:
    python ask.py "Why did sales drop last week?"
    python ask.py "Why did sales drop last week?" --update     # also draft the Monday update
    python ask.py "What happened in October?" --file my_export.csv

Needs an Anthropic API key in a `.env` file (ANTHROPIC_API_KEY=...) or in your
environment. The key is never printed or logged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from agent.loop import AgentError, Investigation, Step
from analysis.tools import Dataset

SAMPLE = Path(__file__).resolve().parent / "data" / "sample_shop.csv"
# Price per million tokens for the default model (Claude Haiku 5.5, prompts up to 100K tokens).
PRICE_PER_M = {"claude-haiku-5-5": (0.10, 0.50)}


def print_step(step: Step) -> None:
    mark = "!" if step.is_error else "✓"
    print(f"  {mark} {step.number}. {step.title}")
    print(f"       {step.tool} → {step.summary}", flush=True)


def print_answer(answer: dict) -> None:
    print(f"\n{answer['headline']}\n")
    print(answer["explanation"])
    if answer.get("drivers"):
        print("\nWhere it came from:")
        for d in answer["drivers"]:
            print(f"  • {d['segment']}: {d['impact']}")
    if answer.get("ruled_out"):
        print("\nRuled out:")
        for r in answer["ruled_out"]:
            print(f"  • {r}")
    c = answer["confidence"]
    print(f"\nConfidence: {c['level']} - {c['reason']}")
    if answer.get("next_checks"):
        print("\nWhat to check next:")
        for n in answer["next_checks"]:
            print(f"  • {n}")


def print_cost(inv: Investigation) -> None:
    u = inv.usage
    line = f"\n[{u.model_calls} model calls, {u.input_tokens:,} input + {u.output_tokens:,} output tokens"
    if u.cache_read_tokens:
        line += f", {u.cache_read_tokens:,} read from cache"
    price = PRICE_PER_M.get(inv.model)
    if price:
        cost = (u.input_tokens + u.cache_write_tokens * 1.25 + u.cache_read_tokens * 0.1) * price[0] / 1e6
        cost += u.output_tokens * price[1] / 1e6
        line += f", about ${cost:.4f}"
    print(line + f", model {inv.model}]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask why a metric moved.")
    parser.add_argument("question", help='e.g. "Why did sales drop last week?"')
    parser.add_argument("--file", type=Path, default=SAMPLE, help="CSV export (default: the demo shop)")
    parser.add_argument("--update", action="store_true", help="also draft the Monday update")
    args = parser.parse_args()

    load_dotenv()
    raw = pd.read_csv(args.file)
    inv = Investigation(Dataset.from_frame(raw), raw_frame=raw, on_step=print_step)

    print(f"Question: {args.question}\nInvestigating...")
    try:
        result = inv.ask(args.question)
        print_answer(result.content)
        if result.warning:
            print(f"\n⚠ {result.warning}")
        else:
            print("\n✓ Number check passed: every number above came from the analysis.")

        if args.update:
            print("\n--- Monday update " + "-" * 40)
            update = inv.draft_update()
            print(update.content["update"])
            if update.warning:
                print(f"\n⚠ {update.warning}")
    except AgentError as error:
        print(f"\nCouldn't finish: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # API problems: missing key, network, rate limits
        name = type(error).__name__
        if "Authentication" in name or "api_key" in str(error).lower():
            print(
                "\nNo valid Anthropic API key found. Add a line ANTHROPIC_API_KEY=... to a .env file "
                "in this folder (it is git-ignored), then try again.",
                file=sys.stderr,
            )
            return 1
        raise
    finally:
        if inv.usage.model_calls:
            print_cost(inv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
