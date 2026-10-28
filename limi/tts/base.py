"""TTS provider interface and WAV helpers."""

from __future__ import annotations

import io
import wave
from abc import ABC, abstractmethod

import numpy as np


class TTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str, language: str) -> bytes:
        """Return a complete WAV file (PCM16) with the spoken text.

        `language` is a short code such as 'ru' or 'en'.
        """


def pcm16_to_wav(samples: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap int16 samples in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return buf.getvalue()
