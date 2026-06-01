"""
Agent loop for Claude function calling.

Responsibilities (mirrors assignment requirements):
- LLM-driven tool selection: we never branch on user text to pick a tool;
  the model decides via its `tool_use` blocks.
- Graceful degradation: a tool that returns {ok: False, error: "..."} is
  sent back to the model as a normal tool_result with is_error=True. The
  model decides what to do with it. A tool that *raises* (e.g. broken
  third-party library) is caught here and converted to the same shape —
  the loop never crashes.
- Bounded steps: `max_steps` prevents runaway looping when the model gets
  stuck calling the same tool repeatedly.
- Reproducible: temperature=0 by default.

The loop returns an `AgentTrace` rich enough for the eval harness to
compute per-prompt: tool sequence, accuracy of selection, latency,
whether the agent abstained (final_text without any tool calls).
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import anthropic

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_STEPS = 5


@dataclass
class ToolCallRecord:
    """One tool invocation by the agent."""

    name: str
    input: dict
    result: Any
    ok: bool
    error: Optional[str]
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "input": self.input,
            "result": self.result,
            "ok": self.ok,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class AgentTrace:
    """Everything the eval harness needs to grade a single agent run."""

    final_text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    steps: int = 0
    total_latency_ms: float = 0.0
    stop_reason: Optional[str] = None
    error: Optional[str] = None  # set on hard error (e.g., max_steps hit)

    def to_dict(self) -> dict:
        return {
            "final_text": self.final_text,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "steps": self.steps,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "stop_reason": self.stop_reason,
            "error": self.error,
        }


def _stringify_tool_result(res: dict) -> str:
    """Serialize a tool's {ok, result|error} dict to a string for Claude."""
    try:
        return json.dumps(res, default=str)
    except Exception:
        return str(res)


def run_agent(
    user_message: str,
    *,
    system_prompt: str,
    tool_specs: list[dict],
    tool_funcs: dict[str, Any],
    client: anthropic.Anthropic,
    model: str = DEFAULT_MODEL,
    max_steps: int = DEFAULT_MAX_STEPS,
    temperature: float = 0.0,
) -> AgentTrace:
    """
    Run the agent on a single user message.

    Args:
        user_message: the prompt to send.
        system_prompt: system instructions (we use this to A/B test policies).
        tool_specs: list of {name, description, input_schema} dicts sent to Claude.
        tool_funcs: dispatch table {tool_name: callable returning {ok, result|error}}.
        client: anthropic.Anthropic instance.
        model: which model to use.
        max_steps: hard cap on agent <-> tool round-trips.
        temperature: 0 for reproducibility in eval; raise for exploration.

    Returns:
        AgentTrace with final answer, full tool sequence, and timing.
    """
    trace = AgentTrace(final_text="")
    messages: list[dict] = [{"role": "user", "content": user_message}]
    start = time.perf_counter()

    try:
        loop_completed_naturally = False
        for step in range(1, max_steps + 1):
            trace.steps = step

            response = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=temperature,
                system=system_prompt,
                tools=tool_specs,
                messages=messages,
            )
            trace.stop_reason = response.stop_reason

            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses:
                # Model produced its final answer — collect any text blocks.
                for block in response.content:
                    if block.type == "text":
                        trace.final_text = block.text
                loop_completed_naturally = True
                break

            # Append the assistant's full response (text + tool_use blocks).
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call; collect tool_result blocks for the next turn.
            tool_results = []
            for tu in tool_uses:
                t0 = time.perf_counter()
                func = tool_funcs.get(tu.name)

                if func is None:
                    res = {"ok": False, "error": f"unknown tool: {tu.name}"}
                else:
                    try:
                        res = func(**dict(tu.input))
                    except Exception as e:
                        # Belt-and-suspenders: each tool already wraps in
                        # try/except. This is the safety net for anything
                        # that slipped through.
                        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}

                latency_ms = (time.perf_counter() - t0) * 1000.0

                trace.tool_calls.append(
                    ToolCallRecord(
                        name=tu.name,
                        input=dict(tu.input),
                        result=res.get("result") if res.get("ok") else None,
                        ok=bool(res.get("ok")),
                        error=res.get("error"),
                        latency_ms=latency_ms,
                    )
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": _stringify_tool_result(res),
                        "is_error": not bool(res.get("ok")),
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        if not loop_completed_naturally:
            trace.error = f"max_steps ({max_steps}) reached without final answer"

    except Exception as e:
        trace.error = f"agent loop crashed: {type(e).__name__}: {e}"
    finally:
        trace.total_latency_ms = (time.perf_counter() - start) * 1000.0

    return trace
