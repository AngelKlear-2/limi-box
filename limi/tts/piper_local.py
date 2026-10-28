"""Local TTS with Piper (works on Raspberry Pi)."""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path

from limi.llm.base import ProviderUnavailable
from limi.tts.base import TTSProvider

log = logging.getLogger("limi.tts")


class PiperLocalTTS(TTSProvider):
    """Needs config: tts.piper_voice, the path to a Piper .onnx voice file.

    The matching <voice>.onnx.json must sit next to it. The voice decides the
    language, so the `language` argument is ignored.
    """

    def __init__(self, voice_path: Path) -> None:
        if not voice_path.is_file():
            raise ProviderUnavailable(
                f"Piper voice not found. Expected a file at:\n  {voice_path}\n"
                "Download a voice (.onnx and .onnx.json) there, or set tts.provider: gtts."
            )
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise ProviderUnavailable("piper-tts is missing. Run: pip install piper-tts") from exc
        self._voice = PiperVoice.load(str(voice_path))

    def synthesize(self, text: str, language: str) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            if hasattr(self._voice, "synthesize_wav"):  # piper-tts 1.3+
                self._voice.synthesize_wav(text, wf)
            else:  # piper-tts 1.2
                self._voice.synthesize(text, wf)
        return buf.getvalue()
