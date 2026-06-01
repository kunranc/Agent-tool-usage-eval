# LEC AI — Assignment 2: Agent + Adversarial Eval

A 3-tool LLM agent (Claude Haiku 4.5) with an evaluation harness that grades tool-selection behaviour under two system-prompt policies.

## Summary

| Metric                                   | A (eager) — **shipped** | B (cautious)   |
| ---------------------------------------- | ----------------------- | -------------- |
| Happy-path accuracy                      | **100%** (10/10)        | 90% (9/10)     |
| Ambiguous preferred rate                 | **100%** (5/5)          | 80% (4/5)      |
| Out-of-scope abstention                  | 100% (5/5)              | 100% (5/5)     |
| Overall correct                          | **100%** (20/20)        | 90% (18/20)    |
| Latency p50 / p95                        | 2385 / 6921 ms          | 2148 / 5665 ms |
| Trace errors / tool failures (main eval) | 0 / 0                   | 0 / 0          |

I ran the eval 3 times for A and 3 times for B at `temperature=0`, giving n=20 prompts × 3 runs = 60 individual prompt executions per policy. Correctness is bit-deterministic across runs; latency varies with network.

**Picked A. Named failure mode: model-knowledge bypass on well-known facts** (for B, its "answer from your own knowledge" rule fires and the tool call is skipped, B answers from its training data instead).

## 1. Why I picked Assignment 2

I picked 2 over Assignment 1 (retrieval) and Assignment 3 (context compression) for two reasons.

First, I wanted to design a tool-using agent end-to-end, including its harness and evaluation.

Second, it complements what I'd built before. I'd previously wrote programs that applies to fully deterministic systems. Assignment 2 lets me apply the same discipline to an LLM-driven system, where the behaviour isn't deterministic by default, to explore evaluation and different failure modes.

## 2. Decisions and alternatives ruled out

The brief suggested Gemini 2.0 Flash for the agent runtime. I tried it first. A fresh GCP project, free-tier model, returned `429 RESOURCE_EXHAUSTED, quota: 0` on the first call. With a short deadline I switched to Claude Haiku 4.5 instead. Total runtime cost across all eval runs and iteration ended up under $2.

A handful of other decisions made along the way:

| Decision                                                                                  | Alternative                                                     | Why                                                                                                                                                                                                                         |
| ----------------------------------------------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Claude Haiku 4.5 as the agent's runtime model                                             | Claude Sonnet / Opus, or a larger frontier model                | Tool-calling on short prompts is well within Haiku's range. Going larger would multiply latency and cost without changing the tool-selection behaviour.                                                                     |
| `temperature=0`, `max_steps=5`                                                            | Higher temperature with sampling, longer chains                 | `temperature=0` tells the model to always pick the highest-probability next token, making outputs near-deterministic across runs. That keeps the eval reproducible. The queries are not expected to need more than 3 steps. |
| Wikipedia REST API                                                                        | Generic web search                                              | Better reproducibility. Narrower domain forces a harder out-of-scope test.                                                                                                                                                  |
| JSON file persistence for notes                                                           | SQLite                                                          | Brief allows either. JSON is easier to inspect and reset between runs.                                                                                                                                                      |
| `notes_store` as a single function, action parameter with values: `add`, `list`, `delete` | Three separate tools: `add_note` / `list_notes` / `delete_note` | Brief asks for 3 *meaningfully different* tools. Padding to 5 with `add_note` / `list_notes` / `delete_note` would just be three variants of the same store, violating what the brief suggests.                             |
| Failure injection via BREAK token in note text                                            | Test-mode flag                                                  | Keeps the failure path in production code, not behind a flag. The graceful-degradation behaviour I ship is the same one under test.                                                                                         |

## 3. Architecture

```
   user prompt
        │
        ▼
  ┌─────────────────┐    tool_use     ┌──────────────┐
  │  agent.loop     │ ───────────────▶│   tools/     │
  │  (Claude Haiku) │                 │              │
  │  max_steps=5    │ ◀───────────────│  calculator  │  (stateless)
  │  temperature=0  │   tool_result   │  notes_store │  (stateful → state/notes.json)
  └─────────────────┘                 │  wiki_search │  (stateless, urllib)
        │                             └──────────────┘
        ▼                                    │
   AgentTrace                                │ BREAK token in note text
   (full sequence,                           │ raises RuntimeError → caught
    latencies, errors)                       ▼
        │                            graceful degradation:
        ▼                            agent loop sees {ok:False, error}
   eval/run.py                       and keeps going
        │
        ▼
   runs/<ts>_prompt<A|B>.json
```

### Three tools (`tools/`)

I picked three tools that exercise the three different shapes of behaviour the brief asks for: a stateless transform, a stateful store, and an external lookup.

1. **`calculator(expression)`** — the stateless transform. Evaluates one arithmetic expression. I used Python's `ast` module to whitelist operators (`+ - * / // % **`, parentheses, unary minus) so the tool can never execute arbitrary code`.
2. **`notes_store(action, text?, note_id?)`** — the stateful one. Persists to `state/notes.json` between calls and between sessions. Three actions: `add`, `list`, `delete`. I added a deliberate failure hook here for the graceful-degradation test: any note text containing `BREAK` raises `RuntimeError` (see §2 for why I chose this approach over a test-mode flag).
3. **`wikipedia_search(query, max_results=5)`** — the external lookup. Hits Wikipedia's REST API directly through stdlib `urllib`. Returns the top hits as `{title, snippet, url}`.

Every tool body is wrapped in `try/except` and returns either `{ok: True, result}` or `{ok: False, error}`. The agent never sees an exception, it sees a structured value it can reason about.

### Agent loop (`agent/loop.py`, ~150 lines)

The loop itself is small. Four things make it work:

- **LLM-driven tool selection.** I never branch on user text to pick a tool. The model gets the three tool specs and emits `tool_use` blocks that decide what to call.
- **Graceful degradation.** Two layers of safety: each tool already normalises its own errors, and the loop wraps every call in `try/except` to catch anything that slipped through. The loop itself never crashes, it produces an error trace instead.
- **Bounded steps.** Capped at `max_steps=5`. If the agent ever gets stuck in a tool-call loop, we record `trace_error` and grade the run as failed rather than spin forever.
- **Reproducible.** `temperature=0` and a structured `AgentTrace` return value (tool sequence, per-call latencies, final text, stop reason) so the eval grader has everything it needs.

## 4. The 20 frozen prompts (`prompts/eval.csv`)

Per the brief: 10 happy-path / 5 ambiguous / 5 out-of-scope. Frozen before the first eval run; not edited between iterations.

| Category                            | n   | Examples (id → preferred tool)                                                                                                                                                                |
| ----------------------------------- | --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Happy-path                          | 10  | h01 "What is 17 * 23?" → calculator; h05 "List all notes" → notes_store; h09 "What is photosynthesis?" → wikipedia_search                                                                     |
| Ambiguous (preferred tool labelled) | 5   | a01 "Remember my dog's name is Max" → notes_store (eager saves, cautious clarifies); a05 "Year of the French Revolution?" → wikipedia_search (eager tools, cautious answers 1789 from memory) |
| Out-of-scope (no tool should fire)  | 5   | o03 "Order me a pizza" → abstain (no transaction tool; trap: eager A might save-as-note); o05 "What's happening in the news today?" → abstain (real-time, wiki excludes)                      |

Distribution within happy-path: 3 calc / 3 notes (add → list → delete, in order so state flows) / 4 wiki.

## 5. Eval methodology

For each prompt: send to Claude with the system prompt + 3 tool specs, loop until `end_turn` or `max_steps=5`, record the full `AgentTrace`. Grade:

- **happy-path / ambiguous**: correct if `expected` tool name appears anywhere in the tool sequence.
- **out-of-scope**: correct if the tool sequence is empty (abstained).

State (`state/notes.json`) is wiped at the start of each eval run for determinism. Full per-prompt traces are saved to `runs/<timestamp>_prompt<A|B>.json` so reviewers can audit every tool call without re-running.

## 6. System prompts A and B — what they share, where they diverge

Same wording skeleton, three deliberate differences on a single dimension (tool eagerness):

|                  | A (eager)                                              | B (cautious)                                                                              |
| ---------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| Default Behavior | "Lean toward using a tool whenever one plausibly fits" | "Only call a tool when its purpose CLEARLY matches and you're confident in the arguments" |
| Fallback         | "If no tool fits, say so honestly"                     | "Prefer answering from your own knowledge or explain why outside scope"                   |
| Failure recovery | "If a tool errors, try a different tool or retry"      | "If a tool errors, do not retry blindly — explain what happened"                          |

## 7. Results

See Summary for the canonical table.

**Determinism (variance check).** A produced identical tool sequences across all 3 runs on all 20 prompts. B produced identical sequences on 19/20 — only h10 ("overview of Kyoto") occasionally double-called Wikipedia (both correct). B's two failures are stable: h09 and a05 fail on every single run.

**A vs B winner: I shipped A.** A wins overall correctness (100% vs 90%) without sacrificing abstention (both 100%). B's losses aren't factually wrong answers, they're cases where B answered from training-data knowledge instead of calling Wikipedia. The grader measures *tool selection*, so B is penalised.

## 8. Named failure mode — model-knowledge bypass on well-known facts

**A note on what I expected vs what I found.** Before running the eval, I expected A and B to diverge mostly on the *out-of-scope* prompts. I assumed eager A would over-tool ("save 'order pizza' as a note") while cautious B would abstain. Both turned out to abstain perfectly (100% on out-of-scope, all 3 runs each).

The actual A/B split came from somewhere I hadn't predicted: **B skipping Wikipedia on two specific encyclopedic prompts** because the answers are already in the model's training data. That's the finding below.

**The finding.** System prompt B's *"answer from your own knowledge"* rule fires whenever a prompt's answer is well-attested in Haiku's training data, even when the grader defines "correct" as *tool was called*. B then produces an accurate reply with an empty tool sequence and is marked wrong.

**Falsifiable evidence (reproducible on every run):**

| Prompt                                                | A's tool sequence            | B's tool sequence                     | Both factually correct? |
| ----------------------------------------------------- | ---------------------------- | ------------------------------------- | ----------------------- |
| h09 *"What is photosynthesis?"*                       | `["wikipedia_search"]`,      | `[]`, self-authored biology paragraph | yes                     |
| a05 *"In what year did the French Revolution begin?"* | `["wikipedia_search"]`       | `[]`, says "1789" from memory         | yes                     |

These are the only two failures in B's 60 prompt × run pairs and they occur on the same two prompts every single run.

**What I deliberately did not do.** Tighten B's wording to force it to call Wikipedia on well-known facts. That would manually wipe away B's inclination to "answer from your own knowledge".

## 9. Bonus finding — A vs B diverge on failure recovery

Tested via `prompts/failure.csv` (2 prompts that trigger the BREAK injection in `notes_store`). Both A and B record `0 trace errors / 2 tool failures` — graceful degradation works in both.

However, I discovered their actual replies diverged.

On prompt f02 — *"save a reminder: tomorrow morning BREAK call the dentist"* — A and B did completely different things:

|             | A (eager)                                                                              | B (cautious)                                                                                                                                                         |
| ----------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tool calls  | 2: initial failed; auto-retried with `BREAK` stripped, succeeded                       | 1: initial failed; no retry                                                                                                                                          |
| Final reply | *"Done! Saved as 'Tomorrow morning - call the dentist'"* — silently rewrote user input | *"I encountered an error. Would you like me to save it with: 'Tomorrow morning — call the dentist' / 'Tomorrow morning: call the dentist' / etc.?"* — offered choice |

**A treated the failure as a problem to solve on the user's behalf; B treated it as a moment to ask the user.** The bonus finding is that failure paths are where system-prompt design has the most leverage, not happy-path tool selection.

## 10. What I'd do with another week

1. **Larger prompt set.** 20 prompts I wrote myself can't catch all failure modes. I'd have a separate Claude instance generate variants of real user queries to test different tool-selection ambiguities.
2. **A near-duplicate 4th tool** (e.g., a `weather` tool whose description nearly overlaps `wikipedia_search`) to test what failure modes would occur when functions overlap.
3. **Deeper study of the bonus finding.** I'd vary the failure type (malformed output, etc.) and watch whether the same A and B divergence shows up. If it does, the finding becomes a real claim. If it only shows up on input-rejection errors like BREAK, that's also worth knowing.

## 11. Reproduce

```bash
git clone <repo>
cd LEC-AI
cp .env.example .env       # add ANTHROPIC_API_KEY
make install                # pip install -r requirements.txt
make test                   # 11 unit tests
make eval                   # A on prompts/eval.csv → runs/*_promptA.json
make eval-promptB           # B for comparison → runs/*_promptB.json
python -m eval.run --prompt-set prompts/failure.csv --system-prompt A   # graceful degradation check
```

All numbers in this README come from JSONs in `runs/` (committed). Reviewers can `cat runs/*_promptA.json | jq .summary` to verify without an API key, or use the included comparison script:

```bash
python scripts/compare.py runs/<promptA>.json runs/<promptB>.json
```

It prints a side-by-side metric diff plus the per-prompt disagreements — the h09 and a05 rows that produce the named failure mode.
