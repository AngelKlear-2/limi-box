"""Memory tools: remember a fact, request a (confirmed) wipe."""

from __future__ import annotations

from limi.memory.store import MemoryStore
from limi.tools.base import FLAG_CONFIRM_FORGET, Tool, ToolResult, object_schema


def make_memory_tools(store: MemoryStore) -> list[Tool]:
    """Build the memory tools bound to a store. Needs no config."""

    def memory_remember(fact: str) -> str:
        return "saved" if store.add_fact(str(fact)) else "already known or empty, nothing saved"

    def memory_forget_request() -> ToolResult:
        # Deliberately deletes nothing: the dialog asks the user to confirm first.
        return ToolResult("Confirmation requested. Nothing has been deleted.", flag=FLAG_CONFIRM_FORGET)

    return [
        Tool(
            name="memory_remember",
            description=(
                "Save a lasting fact about the user (name, age, interests, preferences) "
                "so you remember it in later conversations."
            ),
            parameters=object_schema(
                {"fact": {"type": "string", "description": "one short fact, e.g. 'Her name is Anna'"}}, ["fact"]
            ),
            func=memory_remember,
        ),
        Tool(
            name="memory_forget_request",
            description=(
                "Call when the user asks you to forget them or delete the dialog. "
                "It does not delete anything: the app asks the user to confirm."
            ),
            parameters=object_schema(),
            func=memory_forget_request,
        ),
    ]
