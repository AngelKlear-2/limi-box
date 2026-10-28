"""Robust parser for the model's JSON reply, plus text cleanup for TTS."""

from __future__ import annotations

import json
import re
from typing import Any, Iterator

from limi.llm.base import LLMResponse, ToolCall
from limi.state import MOOD_NAMES

_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # pictographs, emoticons, transport, flags, cards
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U00002B00-\U00002BFF"  # stars, arrows
    "\U000023E9-\U000023FF"  # clocks and media symbols
    "\U0000FE0F\U0000200D"  # variation selector, zero width joiner
    "]+"
)
_TEXT_FIELD = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)')
_REPLY_KEYS = ("text", "mood", "tool_call", "tool_calls")


def clean_text(text: str) -> str:
    """Strip markdown and emojis so the TTS reads plain words."""
    t = re.sub(r"```.*?```", " ", text, flags=re.S)  # code is not read aloud
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)
    t = re.sub(r"(?m)^\s*>\s?", "", t)
    t = re.sub(r"(?m)^\s*(?:[-*+•]|\d+[.)])\s+", "", t)
    t = re.sub(r"(\*\*|__)(.+?)\1", r"\2", t, flags=re.S)
    t = t.replace("*", "")
    t = _EMOJI.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def _json_objects(s: str) -> Iterator[str]:
    """Yield each balanced {...} substring of s, respecting JSON strings."""
    i = 0
    while i < len(s):
        if s[i] != "{":
            i += 1
            continue
        depth, in_str, esc = 0, False, False
        for j in range(i, len(s)):
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield s[i : j + 1]
                    i = j
                    break
        i += 1


def _load_reply(raw: str) -> dict[str, Any] | None:
    """Find the reply object in raw output: plain, fenced, or buried in prose."""
    stripped = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.S | re.I)
    candidates = [stripped] + ([fenced.group(1).strip()] if fenced else [])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except ValueError:
            continue
        if isinstance(obj, dict) and any(k in obj for k in _REPLY_KEYS):
            return obj
    for cand in candidates:
        for chunk in _json_objects(cand):
            try:
                obj = json.loads(chunk)
            except ValueError:
                continue
            if isinstance(obj, dict) and any(k in obj for k in _REPLY_KEYS):
                return obj
    return None


def _salvage_text(raw: str) -> str | None:
    """Pull the "text" value out of truncated JSON so we never read braces aloud."""
    m = _TEXT_FIELD.search(raw)
    if not m:
        return None
    try:
        return json.loads(f'"{m.group(1)}"')
    except ValueError:
        return m.group(1)


def _tool_calls(value: Any) -> list[ToolCall]:
    if value in (None, "", "null", False):
        return []
    items = value if isinstance(value, list) else [value]
    calls: list[ToolCall] = []
    for item in items:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("tool") or item.get("function")
        if not isinstance(name, str) or not name.strip():
            continue
        args = item.get("arguments", item.get("args", item.get("parameters", {})))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        calls.append(ToolCall(name=name.strip(), arguments=args if isinstance(args, dict) else {}))
    return calls


def parse_llm_output(raw: str) -> LLMResponse:
    """Turn raw model output into an LLMResponse. Never raises.

    Broken or missing JSON falls back to: text = the raw output, mood = neutral.
    """
    raw = raw or ""
    reply = _load_reply(raw)
    if reply is None:
        salvaged = _salvage_text(raw) if raw.lstrip().startswith(("{", "```")) else None
        return LLMResponse(text=clean_text(salvaged if salvaged is not None else raw), raw=raw)

    text = reply.get("text")
    mood = str(reply.get("mood", "")).strip().lower()
    return LLMResponse(
        text=clean_text(text) if isinstance(text, str) else "",
        mood=mood if mood in MOOD_NAMES else "neutral",
        tool_calls=_tool_calls(reply.get("tool_call", reply.get("tool_calls"))),
        raw=raw,
    )
