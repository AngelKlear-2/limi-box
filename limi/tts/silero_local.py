"""Optional Russian TTS with Silero (needs PyTorch)."""

from __future__ import annotations

import logging

import numpy as np

from limi.llm.base import ProviderUnavailable
from limi.tts.base import TTSProvider, pcm16_to_wav

log = logging.getLogger("limi.tts")

SAMPLE_RATE = 24000


class SileroLocalTTS(TTSProvider):
    """Needs config: tts.silero_speaker (aidar, baya, kseniya, xenia, eugene).

    Russian only. Requires `pip install torch` (heavy, optional). The model is
    fetched through torch.hub on first start.
    """

    def __init__(self, speaker: str) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ProviderUnavailable(
                "Silero TTS needs PyTorch (optional, heavy). Run: pip install torch, "
                "or choose tts.provider: piper_local."
            ) from exc
        log.info("Loading Silero TTS (first run downloads the model)...")
        self._torch = torch
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models", model="silero_tts", language="ru", speaker="v4_ru"
        )
        self._speaker = speaker

    def synthesize(self, text: str, language: str) -> bytes:
        audio = self._model.apply_tts(text=text, speaker=self._speaker, sample_rate=SAMPLE_RATE)
        samples = (np.clip(audio.numpy(), -1.0, 1.0) * 32767).astype(np.int16)
        return pcm16_to_wav(samples, SAMPLE_RATE)
