"""Microphone capture and an energy-based voice activity detector (VAD)."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque

import numpy as np

log = logging.getLogger("limi.mic")

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples = 960 bytes of PCM16


def parse_device(value: str) -> int | str | None:
    """Config device string -> sounddevice argument (index, name or default)."""
    value = value.strip()
    if not value:
        return None
    return int(value) if value.isdigit() else value


def rms(frame: bytes) -> float:
    """Root-mean-square level of a PCM16 frame (0 to 32767)."""
    samples = np.frombuffer(frame, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


class UtteranceDetector:
    """Energy VAD state machine. Feed 30 ms PCM16 frames in order.

    Speech starts after `start_frames` loud frames in a row and ends after
    `end_silence_s` of quiet. The threshold follows the room noise floor but
    never drops below `min_rms`. A short pre-roll keeps the first syllable.
    """

    def __init__(
        self,
        min_rms: float = 250.0,
        end_silence_s: float = 0.8,
        start_frames: int = 3,
        preroll_s: float = 0.3,
        noise_mult: float = 3.0,
    ) -> None:
        self._min_rms = min_rms
        self._noise_mult = noise_mult
        self._start_frames = start_frames
        self._end_frames = max(1, int(end_silence_s * 1000 / FRAME_MS))
        self._preroll: deque[bytes] = deque(maxlen=max(start_frames, int(preroll_s * 1000 / FRAME_MS)))
        self._noise: float | None = None
        self._speech: list[bytes] = []
        self._loud = 0
        self._quiet = 0
        self.speaking = False
        self.done = False

    def threshold(self) -> float:
        floor = 0.0 if self._noise is None else self._noise * self._noise_mult
        return max(self._min_rms, floor)

    def feed(self, frame: bytes) -> None:
        level = rms(frame)
        threshold = self.threshold()
        if not self.speaking:
            if level < threshold:
                self._noise = level if self._noise is None else 0.95 * self._noise + 0.05 * level
            self._preroll.append(frame)
            self._loud = self._loud + 1 if level >= threshold else 0
            if self._loud >= self._start_frames:
                self.speaking = True
                self._speech = list(self._preroll)
            return
        self._speech.append(frame)
        self._quiet = 0 if level >= threshold else self._quiet + 1
        if self._quiet >= self._end_frames:
            self.done = True

    def audio(self) -> bytes:
        """PCM16 mono 16 kHz audio of the utterance so far."""
        return b"".join(self._speech)


class Microphone:
    """Always-on input stream; consumers pull frames from a queue.

    Needs config: audio.input_device, audio.vad_min_rms, audio.end_silence_s.
    Only one consumer should read at a time (idle wake detection, or the
    wake monitor during playback, or an utterance recording).
    """

    def __init__(self, device: str = "", vad_min_rms: float = 250.0, end_silence_s: float = 0.8) -> None:
        self._device = parse_device(device)
        self._vad_min_rms = vad_min_rms
        self._end_silence_s = end_silence_s
        self._frames: queue.Queue[bytes] = queue.Queue(maxsize=200)  # about 6 s of audio
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            device=self._device,
            callback=self._on_audio,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _on_audio(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        data = indata.tobytes()
        try:
            self._frames.put_nowait(data)
        except queue.Full:  # consumer is busy: drop the oldest frame
            try:
                self._frames.get_nowait()
                self._frames.put_nowait(data)
            except (queue.Empty, queue.Full):
                pass

    def read_frame(self, timeout: float = 0.1) -> bytes | None:
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush(self) -> None:
        """Drop buffered audio, e.g. our own beep or speech."""
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def record_utterance(
        self, start_timeout_s: float | None, max_s: float, stop: threading.Event | None = None
    ) -> bytes | None:
        """Record one utterance.

        Returns PCM16 16 kHz mono bytes, or None if no speech began within
        start_timeout_s (None = wait forever) or `stop` was set.
        """
        det = UtteranceDetector(min_rms=self._vad_min_rms, end_silence_s=self._end_silence_s)
        t0 = time.monotonic()
        speech_started_at: float | None = None
        while True:
            if stop is not None and stop.is_set():
                return None
            frame = self.read_frame(0.1)
            now = time.monotonic()
            if frame is not None:
                det.feed(frame)
                if det.done:
                    return det.audio()
            if det.speaking:
                speech_started_at = speech_started_at or now
                if now - speech_started_at >= max_s:
                    return det.audio()
            elif start_timeout_s is not None and now - t0 >= start_timeout_s:
                return None
