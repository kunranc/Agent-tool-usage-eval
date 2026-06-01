"""
Tool registry. The agent imports `TOOL_SPECS` (sent to the LLM) and
`TOOL_FUNCS` (used to dispatch tool calls) from here.

The aliases below are deliberately underscore-prefixed so the public
package attributes `tools.calculator`, `tools.notes_store`, and
`tools.wikipedia_search` continue to refer to the SUBMODULES (not the
re-exported functions). This matters for tests that need to monkeypatch
module-level constants like STATE_DIR.
"""

from .calculator import calculator as _calculator, SPEC as _CALC_SPEC
from .notes_store import notes_store as _notes_store, SPEC as _NOTES_SPEC
from .wikipedia_search import wikipedia_search as _wikipedia_search, SPEC as _WIKI_SPEC

TOOL_SPECS = [_CALC_SPEC, _NOTES_SPEC, _WIKI_SPEC]

TOOL_FUNCS = {
    "calculator": _calculator,
    "notes_store": _notes_store,
    "wikipedia_search": _wikipedia_search,
}

__all__ = ["TOOL_SPECS", "TOOL_FUNCS"]
