"""
Interactive REPL for trying the agent. Run:

    python -m agent.cli                 # use prompt A
    python -m agent.cli --prompt B      # use prompt B

Type a message and press Enter. The CLI prints every tool call the
agent makes (with latency and ok/error), the final answer, and run
metadata. Useful for sanity checks and qualitative iteration.

For batch evaluation, use the eval harness (Task #4).
"""

import argparse
import os

import anthropic
from dotenv import load_dotenv

from agent.loop import run_agent
from agent.prompts import PROMPTS
from tools import TOOL_FUNCS, TOOL_SPECS


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent REPL")
    parser.add_argument(
        "--prompt",
        choices=list(PROMPTS),
        default="A",
        help="Which system prompt to use (default A).",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set. See .env.example.")
    client = anthropic.Anthropic(api_key=api_key)

    print(f"Agent REPL  (system prompt: {args.prompt})")
    print("Type a message, or 'quit' to exit.\n")

    while True:
        try:
            msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg.lower() in ("quit", "exit"):
            break

        trace = run_agent(
            user_message=msg,
            system_prompt=PROMPTS[args.prompt],
            tool_specs=TOOL_SPECS,
            tool_funcs=TOOL_FUNCS,
            client=client,
        )

        for tc in trace.tool_calls:
            status = "ok" if tc.ok else f"ERROR ({tc.error})"
            print(f"  [tool] {tc.name}({tc.input}) -> {status}  ({tc.latency_ms:.0f}ms)")

        if trace.error:
            print(f"  [error] {trace.error}")

        print(trace.final_text or "(no final text)")
        print(
            f"  [steps={trace.steps}, "
            f"total={trace.total_latency_ms:.0f}ms, "
            f"stop={trace.stop_reason}]\n"
        )


if __name__ == "__main__":
    main()
