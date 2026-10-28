"""Online TTS fallback with gTTS (development only)."""

from __future__ import annotations

import io

import numpy as np

from limi.llm.base import ProviderUnavailable
from limi.tts.base import TTSProvider, pcm16_to_wav


class GTTSProvider(TTSProvider):
    """Needs no config beyond internet access; the language comes from the reply text.

    gTTS returns MP3. It is decoded to WAV with pygame.mixer, so ffmpeg is not needed.
    """

    def __init__(self) -> None:
        try:
            from gtts import gTTS
            import pygame
        except ImportError as exc:
            raise ProviderUnavailable("gTTS or pygame is missing. Run: pip install gTTS pygame") from exc
        self._gtts = gTTS
        self._pygame = pygame

    def synthesize(self, text: str, language: str) -> bytes:
        mp3 = io.BytesIO()
        self._gtts(text=text, lang=language or "en").write_to_fp(mp3)
        mp3.seek(0)
        return self._decode_mp3(mp3)

    def _decode_mp3(self, mp3: io.BytesIO) -> bytes:
        pg = self._pygame
        if not pg.mixer.get_init():
            pg.mixer.init(frequency=24000, size=-16, channels=1)
        freq, _, channels = pg.mixer.get_init()
        sound = pg.mixer.Sound(file=mp3)
        samples = pg.sndarray.array(sound)  # shape (n,) or (n, channels), int16
        return pcm16_to_wav(np.ascontiguousarray(samples), freq, channels)
