"""Sound effects and (interruptible) TTS playback through sounddevice."""

from __future__ import annotations

import io
import logging
import threading
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from limi.audio.mic import parse_device

log = logging.getLogger("limi.player")

LEVEL_FULL_SCALE = 8000.0  # RMS that counts as 'mouth fully open'


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """WAV bytes -> (int16 samples shaped (n, channels), sample rate).

    Supports 8, 16 and 32 bit integer PCM.
    """
    with wave.open(io.BytesIO(data), "rb") as wf:
        channels, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if width == 2:
        samples = np.frombuffer(raw, dtype=np.int16)
    elif width == 1:
        samples = ((np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8).astype(np.int16)
    elif width == 4:
        samples = (np.frombuffer(raw, dtype=np.int32) >> 16).astype(np.int16)
    else:
        raise ValueError(f"unsupported WAV sample width: {width * 8} bit")
    usable = len(samples) - len(samples) % channels
    return samples[:usable].reshape(-1, channels), rate


class Playback:
    """One running playback. `stop()` cuts it off; `done` is set when it ends."""

    def __init__(
        self,
        samples: np.ndarray,
        rate: int,
        device: int | str | None,
        level_cb: Callable[[float], None] | None,
    ) -> None:
        import sounddevice as sd

        self._sd = sd
        self._samples = samples
        self._pos = 0
        self._level_cb = level_cb
        self._stop = threading.Event()
        self.done = threading.Event()
        self._stream = sd.OutputStream(
            samplerate=rate,
            channels=samples.shape[1],
            dtype="int16",
            device=device,
            callback=self._callback,
            finished_callback=self._finished,
        )
        self._stream.start()

    def _callback(self, outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        if self._stop.is_set():
            outdata.fill(0)
            raise self._sd.CallbackStop
        chunk = self._samples[self._pos : self._pos + frames]
        n = len(chunk)
        outdata[:n] = chunk
        if n < frames:
            outdata[n:] = 0
        self._pos += n
        if self._level_cb is not None and n:
            level = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            self._level_cb(min(1.0, level / LEVEL_FULL_SCALE))
        if self._pos >= len(self._samples):
            raise self._sd.CallbackStop

    def _finished(self) -> None:
        self.done.set()
        # Streams cannot be closed from their own callback thread.
        threading.Thread(target=self._close, daemon=True).start()

    def _close(self) -> None:
        try:
            self._stream.close()
        except Exception:
            log.debug("closing output stream failed", exc_info=True)

    def stop(self) -> None:
        self._stop.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self.done.wait(timeout)


class Player:
    """Plays event sounds by fixed name and TTS audio.

    Needs config: sounds_dir and audio.output_device. A missing sound file is
    skipped silently (logged at debug level), never an error.
    """

    def __init__(self, sounds_dir: Path, output_device: str = "") -> None:
        self._dir = sounds_dir
        self._device = parse_device(output_device)
        self._cache: dict[str, tuple[np.ndarray, int] | None] = {}
        self._active: set[Playback] = set()
        self._lock = threading.Lock()

    def _load(self, name: str) -> tuple[np.ndarray, int] | None:
        if name not in self._cache:
            path = self._dir / f"{name}.wav"
            clip = None
            if path.is_file():
                try:
                    clip = decode_wav(path.read_bytes())
                except Exception as exc:
                    log.warning("cannot read sound %s: %s", path, exc)
            else:
                log.debug("sound file missing, skipped: %s", path)
            self._cache[name] = clip
        return self._cache[name]

    def play_event(self, name: str, block: bool = False) -> bool:
        """Play sounds/<name>.wav. Returns True if it started, False if skipped."""
        clip = self._load(name)
        if clip is None:
            return False
        try:
            playback = self._start(clip[0], clip[1], None)
        except Exception as exc:
            log.warning("cannot play sound '%s': %s", name, exc)
            return False
        if block:
            playback.wait()
        return True

    def play_wav(self, wav_bytes: bytes, level_cb: Callable[[float], None] | None = None) -> Playback:
        """Start playing WAV bytes without blocking. `level_cb` gets 0..1 loudness."""
        samples, rate = decode_wav(wav_bytes)
        return self._start(samples, rate, level_cb)

    def stop(self) -> None:
        """Stop everything that is playing."""
        with self._lock:
            active = list(self._active)
        for playback in active:
            playback.stop()

    def _start(self, samples: np.ndarray, rate: int, level_cb: Callable[[float], None] | None) -> Playback:
        playback = Playback(samples, rate, self._device, level_cb)
        with self._lock:
            self._active.add(playback)
        threading.Thread(target=self._forget_when_done, args=(playback,), daemon=True).start()
        return playback

    def _forget_when_done(self, playback: Playback) -> None:
        playback.wait()
        with self._lock:
            self._active.discard(playback)
