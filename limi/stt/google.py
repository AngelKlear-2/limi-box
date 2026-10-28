"""Google Web Speech through the SpeechRecognition package (no key, for development)."""

from __future__ import annotations

import logging

from limi.llm.base import ProviderUnavailable
from limi.stt.base import SAMPLE_RATE, STTProvider

log = logging.getLogger("limi.stt")

_AUTO_LOCALES = {"ru": "ru-RU", "en": "en-US"}


class GoogleSTT(STTProvider):
    """Needs config: the top-level language (e.g. ru-RU) and stt.auto_languages.

    Online and unofficial, meant for development only. Google's free endpoint
    takes one language per call, so with language: auto it uses the first
    entry of stt.auto_languages.
    """

    def __init__(self, auto_languages: list[str]) -> None:
        try:
            import speech_recognition as sr
        except ImportError as exc:
            raise ProviderUnavailable("SpeechRecognition is missing. Run: pip install SpeechRecognition") from exc
        self._sr = sr
        self._recognizer = sr.Recognizer()
        first = (auto_languages or ["ru"])[0].lower()
        self._auto_locale = _AUTO_LOCALES.get(first, first)

    def transcribe(self, audio_bytes: bytes, language: str) -> str:
        lang = (language or "").strip()
        if not lang or lang.lower() == "auto":
            lang = self._auto_locale
        audio = self._sr.AudioData(audio_bytes, SAMPLE_RATE, 2)
        try:
            return str(self._recognizer.recognize_google(audio, language=lang)).strip()
        except self._sr.UnknownValueError:
            return ""
