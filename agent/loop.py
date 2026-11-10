"""The investigation loop: the AI picks a tool, our code runs it, repeat.

    question -> AI -> tool call -> Python runs the analysis -> result back to AI
             -> ... (at most 8 analysis calls) ... -> AI calls submit_answer
             -> number check -> answer (or one retry if a number can't be traced)

The AI only ever receives the file summary (column names, a few example
values) and the tool results. Raw rows never leave this process.

One Investigation keeps its conversation, so follow-up questions ("Draft my
Monday update") build on what was already found.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from agent import prompts
from agent.checks import unverified_numbers
from agent.tool_specs import ALL_TOOLS, FINAL_TOOLS, run_tool, summarize, to_json
from analysis.columns import safe_summary
from analysis.tools import Dataset

DEFAULT_MODEL = "claude-haiku-5-5"  # small and low-cost; override with METRIC_DETECTIVE_MODEL
DEFAULT_EFFORT = "medium"  # how hard the model thinks; override with METRIC_DETECTIVE_EFFORT
MAX_TURNS = 16  # safety stop for the whole loop, in model calls
MAX_TOKENS = 16000


class AgentError(RuntimeError):
    """The investigation could not finish (refusal, cut-off answer, or no answer)."""


@dataclass
class Step:
    """One analysis step, as shown in the "Investigating..." view."""

    number: int
    title: str  # the AI's short reason, e.g. "Check if fewer orders or smaller ones"
    tool: str
    summary: str  # one line written by code from the tool output
    is_error: bool = False


@dataclass
class Result:
    """What one question produced."""

    kind: str  # "answer" (submit_answer) or "update" (submit_update)
    content: dict  # the submitted answer or update
    steps: list[Step]
    unverified: list[str] = field(default_factory=list)  # numbers the check couldn't trace
    retried: bool = False  # True if the number check sent the answer back once

    @property
    def warning(self) -> str | None:
        if not self.unverified:
            return None
        return "Some numbers could not be traced to the analysis: " + ", ".join(self.unverified)


@dataclass
class Usage:
    """Token counts across every model call, for cost tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model_calls: int = 0

    def add(self, usage: Any) -> None:
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.model_calls += 1


class Investigation:
    """A conversation about one dataset."""

    def __init__(
        self,
        ds: Dataset,
        raw_frame=None,
        client: Any = None,
        model: str | None = None,
        on_step: Callable[[Step], None] | None = None,
        max_tool_calls: int = prompts.MAX_TOOL_CALLS,
    ):
        self.ds = ds
        self.client = client
        self.model = model or os.environ.get("METRIC_DETECTIVE_MODEL", DEFAULT_MODEL)
        self.effort = os.environ.get("METRIC_DETECTIVE_EFFORT", DEFAULT_EFFORT)
        self.on_step = on_step or (lambda step: None)
        self.max_tool_calls = max_tool_calls

        # What the AI may know about the file: names, roles, a few example values.
        frame = raw_frame if raw_frame is not None else ds.df.drop(columns=["_timestamp", "_day"])
        self.file_summary = safe_summary(frame, ds.roles)
        self.file_summary["data_from"] = ds.days[0].date().isoformat()
        self.file_summary["data_to"] = ds.days[-1].date().isoformat()

        self.messages: list[dict] = []
        self.tool_outputs: list[dict] = []  # everything the tools returned, for the number check
        self.questions: list[str] = []
        self.steps: list[Step] = []
        self.usage = Usage()

    # ---- public API ----------------------------------------------------

    def ask(self, question: str) -> Result:
        """Investigate a question. The first question includes the file summary."""
        self.questions.append(question)
        text = prompts.first_message(question, self.file_summary) if not self.messages else question
        self.messages.append({"role": "user", "content": text})
        return self._run()

    def draft_update(self) -> Result:
        """The built-in follow-up: a ready-to-paste Monday update."""
        if not self.messages:
            raise AgentError("Ask a question first; the update builds on the investigation.")
        self.messages.append({"role": "user", "content": prompts.MONDAY_UPDATE_REQUEST})
        return self._run()

    # ---- the loop ------------------------------------------------------

    def _client(self):
        if self.client is None:
            import anthropic  # imported here so tests and analysis work without it

            self.client = anthropic.Anthropic()
        return self.client

    def _call_model(self):
        response = self._client().messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=prompts.SYSTEM_PROMPT,
            tools=ALL_TOOLS,
            messages=self.messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},  # reuse the unchanged start of the conversation
        )
        self.usage.add(response.usage)
        return response

    def _run(self) -> Result:
        steps_before = len(self.steps)
        retried = False

        for _turn in range(MAX_TURNS):
            response = self._call_model()
            # Keep the whole reply (including any thinking blocks) in the history.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                raise AgentError("The model declined to answer this question.")
            if response.stop_reason == "max_tokens":
                raise AgentError("The model's reply was cut off before it finished.")

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                # It replied in plain text; ask for the structured answer.
                self.messages.append(
                    {"role": "user", "content": "Please call submit_answer (or submit_update) with your final answer."}
                )
                continue

            results = []
            final_block = None
            for block in tool_uses:
                if block.name in FINAL_TOOLS:
                    final_block = block
                else:
                    results.append(self._run_analysis_tool(block))

            if self.analysis_calls >= self.max_tool_calls and final_block is None:
                results.append(
                    {"type": "text", "text": f"You have used all {self.max_tool_calls} analysis tool calls. Call submit_answer now."}
                )

            if final_block is not None:
                content = dict(final_block.input)
                unverified = unverified_numbers(content, self.tool_outputs, " ".join(self.questions))
                if unverified and not retried:
                    retried = True
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": final_block.id,
                            "content": prompts.number_check_feedback(unverified),
                            "is_error": True,
                        }
                    )
                    self.messages.append({"role": "user", "content": results})
                    continue
                # Accepted (or second failure: keep it, with a warning).
                results.append({"type": "tool_result", "tool_use_id": final_block.id, "content": "Received."})
                self.messages.append({"role": "user", "content": results})
                return Result(
                    kind="update" if final_block.name == "submit_update" else "answer",
                    content=content,
                    steps=self.steps[steps_before:],
                    unverified=unverified,
                    retried=retried,
                )

            self.messages.append({"role": "user", "content": results})

        raise AgentError(f"No answer after {MAX_TURNS} model calls.")

    @property
    def analysis_calls(self) -> int:
        return len(self.steps)

    def _run_analysis_tool(self, block) -> dict:
        """Run one analysis tool call and record it as a step."""
        reason = str(block.input.get("reason", "")).strip() or block.name.replace("_", " ")
        if self.analysis_calls >= self.max_tool_calls:
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"Limit of {self.max_tool_calls} analysis calls reached. Call submit_answer now.",
                "is_error": True,
            }
        try:
            output = run_tool(self.ds, block.name, dict(block.input))
        except (ValueError, KeyError, TypeError) as error:
            step = Step(len(self.steps) + 1, reason, block.name, f"Couldn't run: {error}", is_error=True)
            self.steps.append(step)
            self.on_step(step)
            return {"type": "tool_result", "tool_use_id": block.id, "content": f"Error: {error}", "is_error": True}

        self.tool_outputs.append(output)
        step = Step(len(self.steps) + 1, reason, block.name, summarize(block.name, output))
        self.steps.append(step)
        self.on_step(step)
        return {"type": "tool_result", "tool_use_id": block.id, "content": to_json(output)}
