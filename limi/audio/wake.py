"""Wake word detection: openWakeWord, or STT-on-text matching as the fallback."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from limi.audio.mic import Microphone
from limi.llm.base import ProviderUnavailable
from limi.stt.base import STTProvider
from limi.text import tokens

log = logging.getLogger("limi.wake")


@dataclass(frozen=True)
class WakeEvent:
    remainder: str = ""  # words spoken after the wake word, e.g. "Limi, what time is it"


def match_wake_word(text: str, wake_words: list[str], min_ratio: float = 0.75, max_pos: int = 4) -> str | None:
    """Find a wake word near the start of text (case-insensitive, a bit fuzzy).

    Returns the words after it ('' if none), or None when no wake word is there.
    """
    words = tokens(text)
    for wake in wake_words:
        target = tokens(wake)
        n = len(target)
        if n == 0:
            continue
        for i in range(min(max_pos, len(words) - n + 1)):
            window = words[i : i + n]
            if all(SequenceMatcher(None, a, b).ratio() >= min_ratio for a, b in zip(window, target)):
                return " ".join(words[i + n :])
    return None


class WakeDetector(ABC):
    @abstractmethod
    def wait(self, mic: Microphone, stop: threading.Event, timeout: float | None = None) -> WakeEvent | None:
        """Block until the wake word is heard.

        Returns None when `stop` is set or `timeout` seconds pass first.
        """


class SttWakeDetector(WakeDetector):
    """Transcribes short utterances and looks for a wake word in the text.

    Needs config: wake_words, language, wake.stt_match_ratio. Simple but heavier
    than openWakeWord: every loud sound costs one STT call.
    """

    def __init__(
        self, stt: STTProvider, wake_words: list[str], language: str, min_ratio: float, max_phrase_s: float = 6.0
    ) -> None:
        self._stt = stt
        self._wake_words = wake_words
        self._language = language
        self._min_ratio = min_ratio
        self._max_phrase_s = max_phrase_s

    def wait(self, mic: Microphone, stop: threading.Event, timeout: float | None = None) -> WakeEvent | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not stop.is_set():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return None
            pcm = mic.record_utterance(remaining, self._max_phrase_s, stop)
            if pcm is None:
                continue
            try:
                text = self._stt.transcribe(pcm, self._language)
            except Exception:
                log.exception("wake STT failed")
                continue
            rest = match_wake_word(text, self._wake_words, self._min_ratio)
            if rest is not None:
                return WakeEvent(rest)
        return None


class OpenWakeWordDetector(WakeDetector):
    """openWakeWord frame-by-frame detection.

    Needs config: wake.openwakeword_model (path to a custom 'limi' .onnx model,
    or a built-in model name) and wake.threshold.
    """

    CHUNK = 1280  # 80 ms at 16 kHz, what openWakeWord expects

    def __init__(self, model: str, threshold: float) -> None:
        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise ProviderUnavailable(
                "openWakeWord is not installed. Run: pip install openwakeword, "
                "or set wake.backend: stt in config.yaml."
            ) from exc
        self._model = Model(wakeword_models=[model], inference_framework="onnx")
        self._threshold = threshold

    def wait(self, mic: Microphone, stop: threading.Event, timeout: float | None = None) -> WakeEvent | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        buf = np.empty(0, dtype=np.int16)
        while not stop.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                return None
            frame = mic.read_frame(0.1)
            if frame is None:
                continue
            buf = np.concatenate([buf, np.frombuffer(frame, dtype=np.int16)])
            while len(buf) >= self.CHUNK:
                chunk, buf = buf[: self.CHUNK], buf[self.CHUNK :]
                scores = self._model.predict(chunk)
                if scores and max(scores.values()) >= self._threshold:
                    if hasattr(self._model, "reset"):
                        self._model.reset()
                    return WakeEvent()
        return None
