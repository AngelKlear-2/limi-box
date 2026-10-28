"""Local speech recognition with faster-whisper."""

from __future__ import annotations

import logging

from limi.llm.base import ProviderUnavailable
from limi.stt.base import STTProvider, pcm16_to_float32, whisper_language

log = logging.getLogger("limi.stt")


class WhisperLocalSTT(STTProvider):
    """Needs config: stt.whisper_model (tiny/base/small/...), stt.whisper_device,
    stt.whisper_compute_type, stt.auto_languages, and the top-level language.

    The model is downloaded by faster-whisper on first start. With
    language: auto, detection is limited to stt.auto_languages (default ru, en).
    """

    def __init__(self, model_size: str, device: str, compute_type: str, auto_languages: list[str]) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ProviderUnavailable("faster-whisper is missing. Run: pip install faster-whisper") from exc
        log.info("Loading Whisper model '%s' (first run downloads it)...", model_size)
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._auto = [a.lower() for a in auto_languages] or ["ru", "en"]

    def transcribe(self, audio_bytes: bytes, language: str) -> str:
        audio = pcm16_to_float32(audio_bytes)
        if audio.size == 0:
            return ""
        lang = whisper_language(language)
        text, detected = self._run(audio, lang)
        if lang is None and detected not in self._auto:
            # Whisper guessed a language we do not use: retry in the first allowed one.
            text, _ = self._run(audio, self._auto[0])
        return text

    def _run(self, audio, lang: str | None) -> tuple[str, str]:
        segments, info = self._model.transcribe(audio, language=lang, beam_size=1, vad_filter=False)
        return " ".join(seg.text.strip() for seg in segments).strip(), info.language
