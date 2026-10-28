"""web_search tool: DuckDuckGo, top 3 snippets."""

from __future__ import annotations

import logging
import time
from typing import Any

from limi.tools.base import Tool, object_schema

log = logging.getLogger("limi.search")

MAX_RESULTS = 3
SNIPPET_CHARS = 300
ATTEMPTS = 2  # one retry, DuckDuckGo rate-limits now and then
RETRY_DELAY_S = 1.5


def _search_class() -> Any:
    """Prefer the renamed 'ddgs' package, fall back to 'duckduckgo_search'."""
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS
        return DDGS
    except ImportError:
        return None


def web_search(query: str) -> str:
    """Search the web and return up to 3 'title: snippet' lines. Needs no config."""
    query = str(query).strip()
    if not query:
        return "error: empty search query"
    ddgs_cls = _search_class()
    if ddgs_cls is None:
        log.error("no search package installed, run: pip install ddgs")
        return "error: search package not installed (pip install ddgs)"
    log.info("search: %r", query)
    last_error = ""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            results = list(ddgs_cls().text(query, max_results=MAX_RESULTS))
        except Exception as exc:  # network errors, rate limits...
            last_error = str(exc)
            log.warning("search attempt %d failed: %s", attempt, exc)
            if attempt < ATTEMPTS:
                time.sleep(RETRY_DELAY_S)
            continue
        if not results:
            log.info("search: no results for %r", query)
            return "no results found"
        log.info("search: %d results for %r", len(results), query)
        lines = []
        for i, item in enumerate(results[:MAX_RESULTS], 1):
            body = str(item.get("body", ""))[:SNIPPET_CHARS]
            lines.append(f"{i}. {item.get('title', '')}: {body}")
        return "\n".join(lines)
    return f"search failed: {last_error}"


TOOL = Tool(
    name="web_search",
    description="Search the web. Use it when you don't know the answer or the question needs fresh data.",
    parameters=object_schema({"query": {"type": "string", "description": "search query"}}, ["query"]),
    func=web_search,
)
