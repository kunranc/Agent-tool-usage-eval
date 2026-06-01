"""
Notes store tool — STATEFUL. Persists across turns via a JSON file.

Single entry point `notes_store(action, ...)` with three actions:
- "add"    -> requires `text`; assigns an auto-incrementing integer id
- "list"   -> returns all notes (no args)
- "delete" -> requires `note_id`

Failure injection hook (verified by `prompts/failure.csv`):
if a note's text contains the literal string 'BREAK', we raise RuntimeError.
The agent loop catches this and continues with the error as data, not a crash —
that's the graceful-degradation requirement from the brief.
"""

import json
import os
from pathlib import Path
from typing import Optional

# State file is under ./state/ by default. Tests can override LEC_STATE_DIR.
STATE_DIR = Path(os.environ.get("LEC_STATE_DIR", "state"))
STATE_FILE = STATE_DIR / "notes.json"


def _load() -> dict:
    if not STATE_FILE.exists():
        return {"next_id": 1, "notes": []}
    with STATE_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def _save(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def notes_store(
    action: str,
    text: Optional[str] = None,
    note_id: Optional[int] = None,
) -> dict:
    """Run the notes store. Returns {ok, result|error}."""
    try:
        if action == "add":
            if not text:
                return {"ok": False, "error": "action 'add' requires 'text'"}
            if "BREAK" in text:
                # Deliberate failure hook for graceful-degradation testing.
                raise RuntimeError("simulated tool failure (BREAK token)")
            state = _load()
            note = {"id": state["next_id"], "text": text}
            state["notes"].append(note)
            state["next_id"] += 1
            _save(state)
            return {"ok": True, "result": note}

        if action == "list":
            return {"ok": True, "result": _load()["notes"]}

        if action == "delete":
            if note_id is None:
                return {"ok": False, "error": "action 'delete' requires 'note_id'"}
            state = _load()
            before = len(state["notes"])
            state["notes"] = [n for n in state["notes"] if n["id"] != note_id]
            if len(state["notes"]) == before:
                return {"ok": False, "error": f"no note with id {note_id}"}
            _save(state)
            return {"ok": True, "result": {"deleted_id": note_id}}

        return {"ok": False, "error": f"unknown action: {action!r}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


SPEC = {
    "name": "notes_store",
    "description": (
        "A personal notes store that PERSISTS across turns and across "
        "program runs. Use this to remember information the user wants "
        "saved, retrieve a list of saved notes, or delete one. "
        "Each note has an integer id (auto-assigned) and a text body. "
        "Do NOT use this for general fact lookup (use web_search) or "
        "for arithmetic (use calculator). Do NOT invent ids — list "
        "first to see real ids before deleting."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "delete"],
                "description": (
                    "Which operation to run. "
                    "'add' saves a new note (requires text). "
                    "'list' returns all saved notes. "
                    "'delete' removes a note by id (requires note_id)."
                ),
            },
            "text": {
                "type": "string",
                "description": "Note body. Required for action='add'.",
            },
            "note_id": {
                "type": "integer",
                "description": "Id of an existing note. Required for action='delete'.",
            },
        },
        "required": ["action"],
    },
}
