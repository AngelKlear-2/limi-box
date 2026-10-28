"""LLM provider interface and the shared response types."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ProviderUnavailable(RuntimeError):
    """A provider cannot start (missing package, missing model file...).

    The message is written for the end user and printed as is.
    """


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    mood: str = "neutral"
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: str = ""  # what the model actually said, kept for the tool loop


def render_tools_block(tools: list[dict[str, Any]]) -> str:
    """Describe the tools in plain text; tool calls travel inside the JSON reply.

    Using the JSON reply for tool calls keeps every backend identical,
    including local models that have no native function calling.
    """
    if not tools:
        return 'No tools are available right now. Always set "tool_call" to null.'
    lines = [
        "TOOLS. To use a tool instead of answering, reply with this JSON and nothing else:",
        '{"text": "", "mood": "thinking", "tool_call": {"name": "<tool name>", "arguments": {...}}}',
        "Call at most one tool per reply. You will receive the result, then answer normally.",
        "Available tools:",
    ]
    for tool in tools:
        schema = json.dumps(tool.get("parameters", {}), ensure_ascii=False)
        lines.append(f"- {tool['name']}: {tool['description']} Arguments (JSON schema): {schema}")
    return "\n".join(lines)


def inject_tools(messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return a copy of messages with the tools block appended to the system message."""
    block = render_tools_block(tools)
    out = [dict(m) for m in messages]
    if out and out[0]["role"] == "system":
        out[0]["content"] = f"{out[0]['content']}\n\n{block}"
    else:
        out.insert(0, {"role": "system", "content": block})
    return out


class LLMProvider(ABC):
    """Backends only implement `_complete` (messages in, raw text out).

    `chat` adds the tools block and parses the JSON reply, so every backend
    returns the same LLMResponse.
    """

    @abstractmethod
    def _complete(self, messages: list[dict[str, str]]) -> str:
        """Send chat messages (system/user/assistant) and return the raw reply text."""

    def chat(self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        from limi.llm.parser import parse_llm_output  # local import: parser needs this module

        raw = self._complete(inject_tools(messages, tools or []))
        return parse_llm_output(raw)
