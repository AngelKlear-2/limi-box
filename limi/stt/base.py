"""STT provider interface.

Audio contract: raw PCM16, mono, 16 kHz (what the microphone module produces).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

SAMPLE_RATE = 16000


class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, language: str) -> str:
        """Return the recognised text ('' if nothing).

        `language` is a config value such as 'ru-RU', 'en-US' or 'auto'.
        """


def whisper_language(language: str) -> str | None:
    """'ru-RU' -> 'ru'; 'auto' or empty -> None (let the engine detect)."""
    lang = (language or "").strip().lower()
    if not lang or lang == "auto":
        return None
    return lang.replace("_", "-").split("-")[0]


def pcm16_to_float32(audio_bytes: bytes) -> np.ndarray:
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
