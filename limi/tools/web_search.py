"""web_search tool: DuckDuckGo, top 3 snippets."""

from __future__ import annotations

from limi.tools.base import Tool, object_schema

MAX_RESULTS = 3
SNIPPET_CHARS = 300


def web_search(query: str) -> str:
    """Search the web and return up to 3 'title: snippet' lines. Needs no config."""
    query = str(query).strip()
    if not query:
        return "error: empty search query"
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        try:
            from ddgs import DDGS  # type: ignore[no-redef]  # the package's newer name
        except ImportError:
            return "error: duckduckgo_search is not installed"
    try:
        results = list(DDGS().text(query, max_results=MAX_RESULTS))
    except Exception as exc:  # network errors, rate limits...
        return f"search failed: {exc}"
    if not results:
        return "no results found"
    lines = []
    for i, item in enumerate(results[:MAX_RESULTS], 1):
        body = str(item.get("body", ""))[:SNIPPET_CHARS]
        lines.append(f"{i}. {item.get('title', '')}: {body}")
    return "\n".join(lines)


TOOL = Tool(
    name="web_search",
    description="Search the web. Use it when you don't know the answer or the question needs fresh data.",
    parameters=object_schema({"query": {"type": "string", "description": "search query"}}, ["query"]),
    func=web_search,
)
