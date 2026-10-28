import io
import sys
import types
import unittest
import wave
from unittest import mock

import numpy as np

from limi.config import Config, validate
from limi.llm.base import ProviderUnavailable
from limi.tts.xvibe_local import XVibeLocalTTS


class FakeTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def numpy(self) -> np.ndarray:
        return self._arr


class FakeRussianTTS:
    """Stands in for xvibe_pocket_tts.RussianTTS so tests need no torch."""

    last_voice = None

    def __init__(self, **_: object) -> None:
        self.sample_rate = 24000

    def generate(self, text: str, voice: object = "male_deep", **_: object) -> FakeTensor:
        FakeRussianTTS.last_voice = voice
        return FakeTensor(np.linspace(-2.0, 2.0, 240, dtype=np.float32))  # exceeds [-1, 1] on purpose


def fake_module() -> types.ModuleType:
    mod = types.ModuleType("xvibe_pocket_tts")
    mod.RussianTTS = FakeRussianTTS  # type: ignore[attr-defined]
    return mod


class XVibeTTSTests(unittest.TestCase):
    def test_synthesize_returns_clipped_pcm16_wav_at_model_rate(self) -> None:
        with mock.patch.dict(sys.modules, {"xvibe_pocket_tts": fake_module()}):
            wav = XVibeLocalTTS("male_deep").synthesize("Привет", "ru")
        with wave.open(io.BytesIO(wav)) as wf:
            self.assertEqual((wf.getnchannels(), wf.getsampwidth(), wf.getframerate()), (1, 2, 24000))
            frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        self.assertEqual(frames.max(), 32767)
        self.assertEqual(frames.min(), -32767)

    def test_builtin_voice_name_is_passed_through(self) -> None:
        with mock.patch.dict(sys.modules, {"xvibe_pocket_tts": fake_module()}):
            XVibeLocalTTS("female_silky").synthesize("да", "ru")
        self.assertEqual(FakeRussianTTS.last_voice, "female_silky")

    def test_wav_path_voice_is_passed_as_path(self) -> None:
        with mock.patch.dict(sys.modules, {"xvibe_pocket_tts": fake_module()}):
            XVibeLocalTTS("samples/friend.wav").synthesize("да", "ru")
        self.assertEqual(str(FakeRussianTTS.last_voice).replace("\\", "/"), "samples/friend.wav")

    def test_missing_package_gives_clear_message(self) -> None:
        with mock.patch.dict(sys.modules, {"xvibe_pocket_tts": None}):
            with self.assertRaises(ProviderUnavailable) as ctx:
                XVibeLocalTTS("male_deep")
        self.assertIn("pip install git+https://github.com/GenVoice/xVibePocketTTS", str(ctx.exception))

    def test_default_config_uses_xvibe_and_validates_builtin_voice(self) -> None:
        cfg = Config()
        self.assertEqual(cfg.tts.provider, "xvibe_local")
        self.assertEqual(cfg.tts.xvibe_voice, "male_deep")
        self.assertFalse([p for p in validate(cfg) if "xvibe" in p])

    def test_missing_wav_voice_is_reported(self) -> None:
        cfg = Config()
        cfg.tts.xvibe_voice = "does/not/exist.wav"
        self.assertTrue([p for p in validate(cfg) if "xvibe_voice" in p])


if __name__ == "__main__":
    unittest.main()
