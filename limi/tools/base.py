"""Tool definition types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

FLAG_CONFIRM_FORGET = "confirm_forget"  # the app must ask the user before wiping memory


@dataclass(frozen=True)
class ToolResult:
    """What a tool hands back: text for the LLM, plus an optional flag for the app."""

    text: str
    flag: str | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema of the arguments
    func: Callable[..., "str | ToolResult"]

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def object_schema(properties: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    """JSON schema for a tool's arguments object."""
    return {"type": "object", "properties": properties or {}, "required": required or []}
