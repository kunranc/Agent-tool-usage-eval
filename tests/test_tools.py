"""
Sanity tests for the three tools. Hermetic where possible —
notes_store uses a tmp dir so we don't touch the real state file.
"""

import tools.notes_store as ns_mod  # the MODULE, not the re-exported function
from tools.calculator import calculator
from tools.notes_store import notes_store
from tools.wikipedia_search import wikipedia_search


# --- calculator -------------------------------------------------------------
def test_calc_basic():
    r = calculator("2 + 3 * 4")
    assert r["ok"] and r["result"] == 14


def test_calc_parens_and_division():
    r = calculator("(100 - 17) / 4")
    assert r["ok"] and r["result"] == 20.75


def test_calc_power():
    r = calculator("2 ** 10")
    assert r["ok"] and r["result"] == 1024


def test_calc_rejects_arbitrary_code():
    r = calculator("__import__('os').system('echo pwned')")
    assert not r["ok"]


def test_calc_rejects_garbage():
    r = calculator("two plus two")
    assert not r["ok"]


# --- notes_store ------------------------------------------------------------
def _redirect_state(tmp_path, monkeypatch):
    monkeypatch.setattr(ns_mod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(ns_mod, "STATE_FILE", tmp_path / "notes.json")


def test_notes_add_list_delete(tmp_path, monkeypatch):
    _redirect_state(tmp_path, monkeypatch)

    r = notes_store(action="add", text="buy milk")
    assert r["ok"] and r["result"]["id"] == 1

    r = notes_store(action="add", text="call mom")
    assert r["ok"] and r["result"]["id"] == 2

    r = notes_store(action="list")
    assert r["ok"] and len(r["result"]) == 2

    r = notes_store(action="delete", note_id=1)
    assert r["ok"]

    r = notes_store(action="list")
    assert r["ok"] and len(r["result"]) == 1 and r["result"][0]["id"] == 2


def test_notes_delete_missing(tmp_path, monkeypatch):
    _redirect_state(tmp_path, monkeypatch)
    r = notes_store(action="delete", note_id=999)
    assert not r["ok"] and "no note" in r["error"]


def test_notes_add_requires_text(tmp_path, monkeypatch):
    _redirect_state(tmp_path, monkeypatch)
    r = notes_store(action="add")
    assert not r["ok"]


def test_notes_failure_injection(tmp_path, monkeypatch):
    """The agent should see this error string and keep going, not crash."""
    _redirect_state(tmp_path, monkeypatch)
    r = notes_store(action="add", text="this contains BREAK token")
    assert not r["ok"]
    assert "simulated" in r["error"]


def test_notes_unknown_action(tmp_path, monkeypatch):
    _redirect_state(tmp_path, monkeypatch)
    r = notes_store(action="frobnicate")
    assert not r["ok"]


# --- wikipedia_search -------------------------------------------------------
def test_wikipedia_search_smoke():
    """Soft smoke test — verify shape; network may legitimately fail."""
    r = wikipedia_search("Alan Turing", max_results=3)
    assert "ok" in r
    if r["ok"]:
        assert isinstance(r["result"], list)
        assert len(r["result"]) > 0
        for item in r["result"]:
            assert {"title", "snippet", "url"} <= item.keys()
            assert "wikipedia.org/wiki/" in item["url"]
    else:
        # External tool may legitimately fail (rate limit, no network).
        # Graceful degradation: error string, not a crash.
        assert "error" in r
