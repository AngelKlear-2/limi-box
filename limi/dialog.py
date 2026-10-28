"""Conversation logic: LLM + tool loop, history, and the forget-confirmation flow.

Audio-free on purpose: text in, DialogReply out, so it is easy to test.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from limi.config import Config, resolve_path
from limi.llm.base import LLMProvider, LLMResponse
from limi.memory.store import MemoryStore
from limi.text import detect_language, lang_code, tokens
from limi.tools.base import FLAG_CONFIRM_FORGET
from limi.tools.registry import ToolRegistry

log = logging.getLogger("limi.dialog")

DEFAULT_PROMPT = (
    "You are {name}, a short, friendly, slightly witty voice assistant. "
    "Answer in 1 to 3 plain sentences in the user's language. "
    'Reply ONLY with JSON: {"text": "...", "mood": "happy|sad|angry|neutral|thinking", "tool_call": null}'
)

_STRINGS = {
    "en": {
        "ask": "Do you want me to delete the dialog and forget everything about you? Say yes or no.",
        "done": "Done, I've forgotten everything.",
        "keep": "Okay, I'll keep it.",
        "fail": "Sorry, I couldn't come up with an answer.",
    },
    "ru": {
        "ask": "Удалить диалог и забыть всё о вас? Скажите да или нет.",
        "done": "Готово, я всё забыла.",
        "keep": "Хорошо, оставлю как есть.",
        "fail": "Извините, не получилось найти ответ.",
    },
}

_FORGET = re.compile(
    r"\b(forget (me|everything|all about me|what you know)"
    r"|delete (the |our |my )?(dialog|dialogue|conversation|chat|history|memory)"
    r"|(wipe|clear|erase) (your |the |all )?memory"
    r"|забудь (меня|все|обо мне|все обо мне)"
    r"|удали(ть)? (наш |мой |весь |всю )?(диалог|разговор|историю|переписку|память)"
    r"|(очисти|сотри|стери)( свою| всю| твою)? память)\b"
)
_YES = {"yes", "yeah", "yep", "yup", "sure", "да", "ага", "конечно", "удаляй", "удали"}
_NO = {"no", "nope", "nah", "don", "cancel", "stop", "нет", "неа", "не", "отмена", "стоп", "оставь"}


def _t(lang: str, key: str) -> str:
    return _STRINGS.get(lang, _STRINGS["en"])[key]


def is_forget_request(text: str) -> bool:
    """Keyword fallback for 'forget me' / 'delete the dialog' (EN and RU)."""
    return _FORGET.search(" ".join(tokens(text))) is not None


def classify_confirmation(text: str) -> str:
    """'yes', 'no' or 'other'. Only a short, unambiguous yes counts as yes."""
    words = tokens(text)
    if not words:
        return "other"
    has_no = any(w in _NO for w in words)
    if words[0] in _YES and len(words) <= 4 and not has_no:
        return "yes"
    if has_no and words[0] in _NO:
        return "no"
    return "other"


@dataclass
class DialogReply:
    text: str
    mood: str = "neutral"
    language: str = "ru"
    sound: str | None = None  # event sound to play before speaking (confirm, done)
    awaiting_confirm: bool = False


class Dialog:
    """Holds the conversation. Needs the Config, an LLM, a tool registry and the memory store."""

    def __init__(
        self,
        cfg: Config,
        llm: LLMProvider,
        tools: ToolRegistry,
        store: MemoryStore,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._llm = llm
        self._tools = tools
        self._store = store
        self._clock = clock
        self._default_lang = lang_code(cfg.language)
        self._prompt_path: Path = resolve_path(cfg.llm.system_prompt_path)
        self._awaiting = False
        self._deadline = 0.0
        self._warned_prompt = False

    # confirmation state -------------------------------------------------------

    @property
    def awaiting_confirm(self) -> bool:
        """True while a forget confirmation is pending and not timed out."""
        if self._awaiting and self._clock() > self._deadline:
            log.info("forget confirmation timed out, memory kept")
            self._awaiting = False
        return self._awaiting

    def confirm_remaining(self) -> float:
        return max(0.0, self._deadline - self._clock()) if self.awaiting_confirm else 0.0

    def restart_confirm_timer(self) -> None:
        """Start the timeout window now (called after the question was spoken)."""
        self._deadline = self._clock() + self._cfg.behavior.confirm_timeout_s

    def cancel_confirm(self) -> None:
        self._awaiting = False

    # main entry ---------------------------------------------------------------

    def process(self, user_text: str) -> DialogReply:
        """Handle one user utterance and return what to say."""
        text = user_text.strip()
        lang = detect_language(text, self._default_lang)

        if self.awaiting_confirm:
            verdict = classify_confirmation(text)
            if verdict == "yes":
                return self._confirm_wipe(lang)
            self.cancel_confirm()  # no / other words: memory stays
            if verdict == "no":
                return DialogReply(_t(lang, "keep"), language=lang)
            # other words: confirmation is cancelled, then the words are handled normally

        if is_forget_request(text):
            return self._ask_confirmation(lang)
        return self._answer(text, lang)

    # forget flow --------------------------------------------------------------

    def _ask_confirmation(self, lang: str) -> DialogReply:
        self._awaiting = True
        self.restart_confirm_timer()
        return DialogReply(_t(lang, "ask"), language=lang, sound="confirm", awaiting_confirm=True)

    def _confirm_wipe(self, lang: str) -> DialogReply:
        self._store.wipe()
        self._awaiting = False
        log.info("memory wiped on user confirmation")
        return DialogReply(_t(lang, "done"), language=lang, sound="done")

    # LLM loop -----------------------------------------------------------------

    def _system_prompt(self) -> str:
        try:
            base = self._prompt_path.read_text(encoding="utf-8")
        except OSError:
            if not self._warned_prompt:
                log.warning("system prompt not found at %s, using the built-in one", self._prompt_path)
                self._warned_prompt = True
            base = DEFAULT_PROMPT
        base = base.replace("{name}", self._cfg.name)
        facts = self._store.facts_block()
        return f"{base}\n\n{facts}" if facts else base

    def _history(self) -> list[dict[str, str]]:
        history = self._store.recent_messages(self._cfg.llm.history_len)
        while history and history[0]["role"] != "user":  # some APIs need a user turn first
            history.pop(0)
        return history

    def _answer(self, text: str, lang: str) -> DialogReply:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            *self._history(),
            {"role": "user", "content": text},
        ]
        schemas = self._tools.schemas()
        resp = self._llm.chat(messages, schemas)
        rounds = 0
        while resp.tool_calls:
            if rounds >= self._cfg.llm.max_tool_rounds:
                resp = self._force_final(messages, resp)
                break
            rounds += 1
            flags = self._run_tools(messages, resp)
            if FLAG_CONFIRM_FORGET in flags:
                return self._ask_confirmation(lang)
            resp = self._llm.chat(messages, schemas)

        spoken = resp.text.strip()
        if not spoken or resp.tool_calls:
            spoken = _t(lang, "fail")
        self._store.add_message("user", text)
        self._store.add_message("assistant", spoken)
        return DialogReply(spoken, mood=resp.mood, language=lang)

    def _run_tools(self, messages: list[dict[str, str]], resp: LLMResponse) -> set[str]:
        """Execute the requested tools, append the exchange to messages, return flags."""
        messages.append({"role": "assistant", "content": resp.raw or _call_json(resp)})
        flags: set[str] = set()
        for call in resp.tool_calls:
            log.info("tool call: %s", call.name)
            result = self._tools.dispatch(call.name, call.arguments)
            if result.flag:
                flags.add(result.flag)
            messages.append(
                {
                    "role": "user",
                    "content": f"[tool result: {call.name}]\n{result.text}\n"
                    "Now answer the user with the JSON reply format.",
                }
            )
        return flags

    def _force_final(self, messages: list[dict[str, str]], resp: LLMResponse) -> LLMResponse:
        """Tool round limit reached: ask for a final answer with no tools offered."""
        log.info("tool round limit reached, forcing a final answer")
        messages.append({"role": "assistant", "content": resp.raw or _call_json(resp)})
        messages.append(
            {
                "role": "user",
                "content": "Tool limit reached. Answer now with what you have. Set tool_call to null.",
            }
        )
        return self._llm.chat(messages, [])


def _call_json(resp: LLMResponse) -> str:
    """Rebuild the model's tool-call JSON when the raw text is not available."""
    call = resp.tool_calls[0]
    return json.dumps(
        {"text": "", "mood": "thinking", "tool_call": {"name": call.name, "arguments": call.arguments}},
        ensure_ascii=False,
    )
