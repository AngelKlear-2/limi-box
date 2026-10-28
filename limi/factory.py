"""Build providers from config. Swapping an engine is a config change only."""

from __future__ import annotations

import logging

from limi.audio.wake import OpenWakeWordDetector, SttWakeDetector, WakeDetector
from limi.config import Config, resolve_path
from limi.llm.base import LLMProvider, ProviderUnavailable
from limi.stt.base import STTProvider
from limi.tts.base import TTSProvider

log = logging.getLogger("limi.factory")


def build_stt(cfg: Config) -> STTProvider:
    if cfg.stt.provider == "google":
        from limi.stt.google import GoogleSTT

        return GoogleSTT(cfg.stt.auto_languages)
    from limi.stt.whisper_local import WhisperLocalSTT

    return WhisperLocalSTT(
        cfg.stt.whisper_model, cfg.stt.whisper_device, cfg.stt.whisper_compute_type, cfg.stt.auto_languages
    )


def build_tts(cfg: Config) -> TTSProvider:
    if cfg.tts.provider == "gtts":
        from limi.tts.gtts import GTTSProvider

        return GTTSProvider()
    if cfg.tts.provider == "xvibe_local":
        from limi.tts.xvibe_local import XVibeLocalTTS

        return XVibeLocalTTS(cfg.tts.xvibe_voice)
    if cfg.tts.provider == "silero_local":
        from limi.tts.silero_local import SileroLocalTTS

        return SileroLocalTTS(cfg.tts.silero_speaker)
    from limi.tts.piper_local import PiperLocalTTS

    return PiperLocalTTS(resolve_path(cfg.tts.piper_voice))


def build_llm(cfg: Config) -> LLMProvider:
    llm = cfg.llm
    if llm.provider == "api":
        from limi.llm.openai_api import OpenAICompatLLM

        return OpenAICompatLLM(llm.api.base_url, llm.api.model, llm.api.api_key, llm.temperature)
    if llm.provider == "anthropic":
        from limi.llm.anthropic_api import AnthropicLLM

        return AnthropicLLM(llm.anthropic.model, llm.anthropic.api_key, llm.temperature, llm.max_tokens)
    from limi.llm.local_gguf import LocalGGUFLLM

    return LocalGGUFLLM(
        resolve_path(llm.local_gguf.model_path),
        llm.local_gguf.n_ctx,
        llm.local_gguf.n_gpu_layers,
        llm.temperature,
        llm.max_tokens,
    )


def build_wake(cfg: Config, stt: STTProvider) -> WakeDetector:
    """Pick the wake word backend: openwakeword, stt, or auto (oww if usable)."""
    backend = cfg.wake.backend
    if backend in ("auto", "openwakeword") and cfg.wake.openwakeword_model:
        try:
            return OpenWakeWordDetector(cfg.wake.openwakeword_model, cfg.wake.threshold)
        except ProviderUnavailable:
            if backend == "openwakeword":
                raise
            log.warning("openWakeWord unavailable, falling back to STT wake word matching")
    elif backend == "openwakeword":
        raise ProviderUnavailable(
            "wake.backend is 'openwakeword' but wake.openwakeword_model is empty. "
            "Set a model path/name or use wake.backend: stt."
        )
    return SttWakeDetector(stt, cfg.wake_words, cfg.language, cfg.wake.stt_match_ratio)
