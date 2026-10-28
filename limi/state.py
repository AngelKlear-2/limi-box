"""Shared state object. The face reads it; the assistant loop writes it.

This is the only module that holds mutable shared state. The face knows
nothing about STT, TTS or the LLM: it only looks at what is stored here.
"""

from __future__ import annotations

import threading
import time
from enum import Enum


class State(str, Enum):
    SLEEP = "sleep"
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    AWAIT_CONFIRM = "await_confirm"  # waiting for yes/no, face looks like LISTENING


class Mood(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    THINKING = "thinking"

    @classmethod
    def parse(cls, value: str | None) -> "Mood":
        """Map any string to a Mood; unknown values become NEUTRAL."""
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.NEUTRAL


MOOD_NAMES: tuple[str, ...] = tuple(m.value for m in Mood)


class SharedState:
    """Thread-safe holder for state, mood, audio amplitude and the quit flag."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = State.IDLE
        self._mood = Mood.NEUTRAL
        self._amplitude: float | None = None  # 0..1 while TTS plays, else None
        self._last_interaction = time.monotonic()
        self.quit = threading.Event()

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    @state.setter
    def state(self, value: State) -> None:
        with self._lock:
            self._state = value

    @property
    def mood(self) -> Mood:
        with self._lock:
            return self._mood

    @mood.setter
    def mood(self, value: Mood) -> None:
        with self._lock:
            self._mood = value

    @property
    def amplitude(self) -> float | None:
        with self._lock:
            return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float | None) -> None:
        with self._lock:
            self._amplitude = value

    def set_amplitude(self, value: float) -> None:
        """Callback form of the amplitude setter, used by the audio player."""
        self.amplitude = value

    def snapshot(self) -> tuple[State, Mood, float | None]:
        """Read state, mood and amplitude together, consistently."""
        with self._lock:
            return self._state, self._mood, self._amplitude

    def touch(self) -> None:
        """Mark 'now' as the last moment of user interaction."""
        with self._lock:
            self._last_interaction = time.monotonic()

    def idle_for(self) -> float:
        """Seconds since the last interaction."""
        with self._lock:
            return time.monotonic() - self._last_interaction
