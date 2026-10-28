"""Local Russian TTS with xVibePocketTTS (GenVoice, MIT code).

Needs config: tts.xvibe_voice. A built-in voice name (male_deep, female_silky,
male_authoritative) or a path to a clean 1-30 s WAV for voice cloning.
Install once (Python 3.10-3.13, pulls PyTorch, CPU is fine):
    pip install git+https://github.com/GenVoice/xVibePocketTTS
The model (genvoice/xVibePocketTTS) is downloaded from Hugging Face on first start.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from limi.llm.base import ProviderUnavailable
from limi.tts.base import TTSProvider, pcm16_to_wav

log = logging.getLogger("limi.tts")

BUILTIN_VOICES = ("male_deep", "female_silky", "male_authoritative")


class XVibeLocalTTS(TTSProvider):
    def __init__(self, voice: str) -> None:
        try:
            from xvibe_pocket_tts import RussianTTS
        except ImportError as exc:
            raise ProviderUnavailable(
                "xVibePocketTTS is not installed. Run: "
                "pip install git+https://github.com/GenVoice/xVibePocketTTS "
                "or choose tts.provider: piper_local."
            ) from exc
        log.info("Loading xVibePocketTTS (first run downloads the model)...")
        self._tts = RussianTTS(device="cpu")
        self._voice: str | Path = voice if voice in BUILTIN_VOICES else Path(voice).expanduser()

    def synthesize(self, text: str, language: str) -> bytes:
        audio = self._tts.generate(text, voice=self._voice)
        samples = (np.clip(audio.numpy(), -1.0, 1.0) * 32767).astype(np.int16)
        return pcm16_to_wav(samples, int(self._tts.sample_rate))
