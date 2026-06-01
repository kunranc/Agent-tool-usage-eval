"""
Compare two eval run JSONs (typically promptA vs promptB) and print a diff.

Usage:
    python scripts/compare.py runs/<promptA>.json runs/<promptB>.json
"""

import json
import sys
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt_pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def fmt_ms(x):
    return "n/a" if x is None else f"{x:.0f}"


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)

    a, b = load(sys.argv[1]), load(sys.argv[2])
    label_a = a["meta"]["system_prompt"]
    label_b = b["meta"]["system_prompt"]
    sa, sb = a["summary"], b["summary"]

    print()
    print(f"{'Metric':<32}{label_a:>10}{label_b:>10}{'delta':>10}")
    print("-" * 62)

    def row_pct(name, va, vb):
        delta = "n/a" if va is None or vb is None else f"{(va-vb)*100:+.0f}pp"
        print(f"{name:<32}{fmt_pct(va):>10}{fmt_pct(vb):>10}{delta:>10}")

    def row_ms(name, va, vb):
        delta = "n/a" if va is None or vb is None else f"{va-vb:+.0f}"
        print(f"{name:<32}{fmt_ms(va):>10}{fmt_ms(vb):>10}{delta:>10}")

    row_pct("Happy-path accuracy", sa["happy_path_accuracy"], sb["happy_path_accuracy"])
    row_pct("Ambiguous preferred rate", sa["ambiguous_preferred_rate"], sb["ambiguous_preferred_rate"])
    row_pct("Out-of-scope abstention", sa["out_of_scope_abstention_rate"], sb["out_of_scope_abstention_rate"])
    row_pct("Overall correct", sa["overall_correct_rate"], sb["overall_correct_rate"])

    la, lb = sa["latency_ms"], sb["latency_ms"]
    row_ms("Latency p50 (ms)", la["p50"], lb["p50"])
    row_ms("Latency p95 (ms)", la["p95"], lb["p95"])

    print()
    print(f"Trace errors:  {label_a}={sa['trace_errors']}   {label_b}={sb['trace_errors']}")
    print(f"Tool failures: {label_a}={sa['tool_failures']}   {label_b}={sb['tool_failures']}")

    # Per-prompt disagreement
    disagree = []
    for ra, rb in zip(a["results"], b["results"]):
        if ra["id"] != rb["id"]:
            print(f"WARNING: id mismatch {ra['id']} vs {rb['id']}")
            continue
        if ra["correct"] != rb["correct"] or ra["tool_sequence"] != rb["tool_sequence"]:
            disagree.append((ra, rb))

    if disagree:
        print()
        print(f"--- Per-prompt disagreements ({len(disagree)}) ---")
        for ra, rb in disagree:
            seq_a = ",".join(ra["tool_sequence"]) or "ABSTAIN"
            seq_b = ",".join(rb["tool_sequence"]) or "ABSTAIN"
            print(f"  {ra['id']} [{ra['category']}] expected={ra['expected']}")
            print(f"    {label_a}: {seq_a:<25} correct={ra['correct']}")
            print(f"    {label_b}: {seq_b:<25} correct={rb['correct']}")
    else:
        print()
        print("No per-prompt disagreements.")


if __name__ == "__main__":
    main()
