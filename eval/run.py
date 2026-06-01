"""
Eval harness - runs the agent on a labelled prompt set and reports the
three metrics the assignment asks for:

  1. Tool-selection accuracy (happy-path + ambiguous categories)
  2. Out-of-scope abstention rate
  3. End-to-end latency (p50 / p95)

Usage:
    python -m eval.run --prompt-set prompts/eval.csv --system-prompt A
    python -m eval.run --prompt-set prompts/eval.csv --system-prompt B
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import statistics
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from agent.loop import run_agent
from agent.prompts import PROMPTS
from tools import TOOL_FUNCS, TOOL_SPECS
from tools.notes_store import STATE_FILE as NOTES_STATE_FILE

CATEGORIES = {"happy_path", "ambiguous", "out_of_scope"}
EXPECTED_COLUMNS = {"id", "category", "prompt", "expected", "notes"}


def load_prompts(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = EXPECTED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: missing columns: {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            row = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            if not row.get("id") or not row.get("prompt"):
                if not any(row.values()):
                    continue
                raise SystemExit(f"{path}:{line_no}: missing id or prompt")
            if row["category"] not in CATEGORIES:
                raise SystemExit(
                    f"{path}:{line_no}: bad category {row['category']!r}"
                )
            if row["category"] != "out_of_scope" and not row["expected"]:
                raise SystemExit(
                    f"{path}:{line_no}: 'expected' tool name required for "
                    f"category {row['category']!r}"
                )
            rows.append(row)
    return rows


def grade(row: dict, trace_dict: dict) -> dict:
    tool_sequence = [tc["name"] for tc in trace_dict["tool_calls"]]
    abstained = len(tool_sequence) == 0
    if row["category"] == "out_of_scope":
        correct = abstained
    else:
        correct = row["expected"] in tool_sequence
    return {
        "id": row["id"],
        "category": row["category"],
        "prompt": row["prompt"],
        "expected": row.get("expected") or None,
        "notes": row.get("notes") or "",
        "first_tool": tool_sequence[0] if tool_sequence else None,
        "tool_sequence": tool_sequence,
        "abstained": abstained,
        "correct": correct,
        "final_text": trace_dict["final_text"],
        "steps": trace_dict["steps"],
        "total_latency_ms": trace_dict["total_latency_ms"],
        "trace_error": trace_dict["error"],
        "trace": trace_dict,
    }


def _percentile(xs, p):
    if not xs:
        return None
    xs_sorted = sorted(xs)
    k = int(round(p / 100 * (len(xs_sorted) - 1)))
    return xs_sorted[k]


def _acc(rows):
    if not rows:
        return None
    return sum(1 for r in rows if r["correct"]) / len(rows)


def summarize(graded):
    by_cat = {c: [g for g in graded if g["category"] == c] for c in CATEGORIES}
    latencies = [g["total_latency_ms"] for g in graded]
    return {
        "n": len(graded),
        "by_category_n": {c: len(by_cat[c]) for c in CATEGORIES},
        "happy_path_accuracy": _acc(by_cat["happy_path"]),
        "ambiguous_preferred_rate": _acc(by_cat["ambiguous"]),
        "out_of_scope_abstention_rate": _acc(by_cat["out_of_scope"]),
        "overall_correct_rate": _acc(graded),
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "max": max(latencies) if latencies else None,
            "mean": round(statistics.mean(latencies), 1) if latencies else None,
        },
        "trace_errors": sum(1 for g in graded if g["trace_error"]),
        "tool_failures": sum(
            1 for g in graded for tc in g["trace"]["tool_calls"] if not tc["ok"]
        ),
    }


def _fmt_pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def print_summary(summary, system_prompt):
    print()
    print("=" * 64)
    print(f"Summary - system prompt {system_prompt} - n={summary['n']}")
    print("=" * 64)
    n_by_cat = summary["by_category_n"]
    print(f"  Happy-path accuracy        : {_fmt_pct(summary['happy_path_accuracy']):>5}   (n={n_by_cat['happy_path']})")
    print(f"  Ambiguous preferred rate   : {_fmt_pct(summary['ambiguous_preferred_rate']):>5}   (n={n_by_cat['ambiguous']})")
    print(f"  Out-of-scope abstention    : {_fmt_pct(summary['out_of_scope_abstention_rate']):>5}   (n={n_by_cat['out_of_scope']})")
    print(f"  Overall correct            : {_fmt_pct(summary['overall_correct_rate']):>5}")
    lat = summary["latency_ms"]
    if lat["p50"] is not None:
        print(f"  Latency p50 / p95 / max    : {lat['p50']:.0f} / {lat['p95']:.0f} / {lat['max']:.0f} ms")
    print(f"  Trace errors / tool failures: {summary['trace_errors']} / {summary['tool_failures']}")


def main():
    parser = argparse.ArgumentParser(description="LEC agent eval harness")
    parser.add_argument("--prompt-set", default="prompts/eval.csv")
    parser.add_argument("--system-prompt", choices=list(PROMPTS), default="A")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--no-reset-state", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set (see .env.example).")
    client = anthropic.Anthropic(api_key=api_key)

    rows = load_prompts(Path(args.prompt_set))
    if not rows:
        raise SystemExit(f"No prompts loaded from {args.prompt_set}")

    if args.start:
        rows = rows[args.start:]
    if args.limit:
        rows = rows[: args.limit]

    if not args.no_reset_state:
        NOTES_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        NOTES_STATE_FILE.write_text(
            json.dumps({"next_id": 1, "notes": []}, indent=2),
            encoding="utf-8",
        )
        print(f"Reset notes state file: {NOTES_STATE_FILE}")

    print(f"Running {len(rows)} prompts  |  system prompt {args.system_prompt}  |  prompt set: {args.prompt_set}")
    graded: list[dict] = []
    for i, row in enumerate(rows, start=1):
        snippet = row["prompt"][:60].replace("\n", " ")
        print(f"  [{i:>2}/{len(rows)}] {row['id']:>4}  {row['category']:<14}  {snippet}")
        sys.stdout.flush()
        trace = run_agent(
            user_message=row["prompt"],
            system_prompt=PROMPTS[args.system_prompt],
            tool_specs=TOOL_SPECS,
            tool_funcs=TOOL_FUNCS,
            client=client,
            max_steps=args.max_steps,
        )
        graded.append(grade(row, trace.to_dict()))

    summary = summarize(graded)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"{ts}_prompt{args.system_prompt}.json"
    payload = {
        "meta": {
            "timestamp": ts,
            "system_prompt": args.system_prompt,
            "prompt_set": str(args.prompt_set),
            "n_prompts": len(rows),
        },
        "summary": summary,
        "results": graded,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print_summary(summary, args.system_prompt)
    print()
    print(f"Saved raw results: {out_path}")


if __name__ == "__main__":
    main()
