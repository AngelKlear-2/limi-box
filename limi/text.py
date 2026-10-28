"""Small text helpers shared by the dialog, TTS and wake word matching."""

from __future__ import annotations

import re

_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"\w+", re.UNICODE)


def lang_code(language: str, fallback: str = "ru") -> str:
    """Turn a config language ('ru-RU', 'en_US', 'auto') into a short code."""
    lang = (language or "").strip().lower()
    if not lang or lang == "auto":
        return fallback
    return re.split(r"[-_]", lang)[0]


def detect_language(text: str, default: str = "ru") -> str:
    """Cheap script-based guess: 'ru' for Cyrillic, 'en' for Latin, else default."""
    cyr = len(_CYRILLIC.findall(text))
    lat = len(_LATIN.findall(text))
    if cyr == 0 and lat == 0:
        return default
    return "ru" if cyr >= lat else "en"


def normalize(text: str) -> str:
    """Lowercase and fold ё to е so matching is forgiving."""
    return text.lower().replace("ё", "е")


def tokens(text: str) -> list[str]:
    """Split text into normalized word tokens (punctuation dropped)."""
    return _WORD.findall(normalize(text))
