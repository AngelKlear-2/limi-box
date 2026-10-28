"""Tool registry and dispatcher."""

from __future__ import annotations

import logging
from typing import Any, Iterable

from limi.memory.store import MemoryStore
from limi.music.base import MusicProvider
from limi.tools.base import Tool, ToolResult

log = logging.getLogger("limi.tools")


class ToolRegistry:
    """Holds tools by name. `dispatch` never raises: problems come back as text."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def dispatch(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"error: unknown tool '{name}'. Available tools: {', '.join(self._tools) or 'none'}")
        args = arguments or {}
        props = tool.parameters.get("properties", {})
        missing = [k for k in tool.parameters.get("required", []) if k not in args]
        if missing:
            return ToolResult(f"error: tool '{name}' is missing arguments: {', '.join(missing)}")
        log.info("tool %s %s", name, args)
        try:
            out = tool.func(**{k: v for k, v in args.items() if k in props})
        except Exception as exc:  # a broken tool must not break the conversation
            log.exception("tool %s failed", name)
            return ToolResult(f"error: tool '{name}' failed: {exc}")
        result = out if isinstance(out, ToolResult) else ToolResult(str(out))
        log.info("tool %s -> %s", name, result.text[:120].replace("\n", " | "))
        return result


def build_registry(store: MemoryStore, music: MusicProvider) -> ToolRegistry:
    """Registry with every built-in tool."""
    from limi.tools import memory_tools, music_tools, time_tool, web_search

    registry = ToolRegistry()
    registry.register(web_search.TOOL)
    registry.register(time_tool.TOOL)
    for tool in memory_tools.make_memory_tools(store):
        registry.register(tool)
    for tool in music_tools.make_music_tools(music):
        registry.register(tool)
    return registry
