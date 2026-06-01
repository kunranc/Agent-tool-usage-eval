"""
Wikipedia search tool — stateless external API call to the Wikipedia REST API.

Chosen over a generic web search because:
- No rate limits in the volumes we hit, so the eval is reproducible.
- Narrower domain forces the agent to abstain when a question is not
  Wikipedia-shaped (current events, opinions, product reviews, etc.) —
  this makes the out-of-scope abstention test harder and more informative.
- Stdlib-only (urllib) — no third-party dependency to break.
"""

import json
import re
import urllib.parse
import urllib.request

_BASE = "https://en.wikipedia.org"
_USER_AGENT = "LEC-Agent/0.1 (assignment; contact via Anthropic console)"

# Wikipedia's search snippet contains <span class="searchmatch">...</span>
# highlighting — strip those before showing to the LLM.
_HTML_RE = re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    return _HTML_RE.sub("", s).strip()


def wikipedia_search(query: str, max_results: int = 5) -> dict:
    """Search English Wikipedia. Returns {ok, result|error}."""
    try:
        limit = min(max(int(max_results), 1), 10)
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
            "utf8": 1,
        }
        url = f"{_BASE}/w/api.php?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        hits = data.get("query", {}).get("search", [])
        results = [
            {
                "title": h.get("title"),
                "snippet": _strip_html(h.get("snippet", "")),
                "url": f"{_BASE}/wiki/"
                + urllib.parse.quote((h.get("title") or "").replace(" ", "_")),
            }
            for h in hits
        ]
        return {"ok": True, "result": results}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


SPEC = {
    "name": "wikipedia_search",
    "description": (
        "Search English Wikipedia for encyclopedic information and return "
        "the top results as a list of {title, snippet, url}. "
        "Use for: established facts, definitions, biographies, historical "
        "events, scientific concepts, named entities (people, places, "
        "organisations) that would have a Wikipedia entry. "
        "Do NOT use for: arithmetic (use calculator), retrieving the user's "
        "saved notes (use notes_store), breaking news in the last few days, "
        "product reviews, personal opinions, real-time data (stock prices, "
        "weather), or anything Wikipedia would not cover. If the question is "
        "not Wikipedia-shaped, abstain rather than searching."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Wikipedia search query, usually a topic name "
                    "(e.g. 'Alan Turing', 'photosynthesis', 'Treaty of Versailles')."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "How many results to return (default 5, max 10).",
            },
        },
        "required": ["query"],
    },
}
