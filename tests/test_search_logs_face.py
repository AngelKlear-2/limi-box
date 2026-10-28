import io
import logging
import sys
import types
import unittest
from unittest import mock

from limi.face.renderer import FaceModel
from limi.logutil import LogFormatter
from limi.state import Mood, State
from limi.tools.web_search import web_search


class FakeDDGS:
    calls = 0
    fail_times = 0

    def __init__(self) -> None:
        pass

    def text(self, query: str, max_results: int) -> list[dict[str, str]]:
        FakeDDGS.calls += 1
        if FakeDDGS.calls <= FakeDDGS.fail_times:
            raise RuntimeError("ratelimit")
        return [{"title": "Кобиджон Алиев", "body": "Пример описания"}]


class SearchTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeDDGS.calls = 0
        FakeDDGS.fail_times = 0
        self.mod = types.ModuleType("ddgs")
        self.mod.DDGS = FakeDDGS  # type: ignore[attr-defined]

    def test_retries_once_after_a_failure(self) -> None:
        FakeDDGS.fail_times = 1
        with mock.patch.dict(sys.modules, {"ddgs": self.mod}), mock.patch("limi.tools.web_search.time.sleep"):
            out = web_search("Кобиджон Алиев")
        self.assertEqual(FakeDDGS.calls, 2)
        self.assertIn("Кобиджон Алиев", out)

    def test_reports_error_after_all_attempts(self) -> None:
        FakeDDGS.fail_times = 99
        with mock.patch.dict(sys.modules, {"ddgs": self.mod}), mock.patch("limi.tools.web_search.time.sleep"):
            out = web_search("x")
        self.assertTrue(out.startswith("search failed: ratelimit"))

    def test_missing_package_is_explained(self) -> None:
        with mock.patch.dict(sys.modules, {"ddgs": None, "duckduckgo_search": None}):
            out = web_search("x")
        self.assertIn("pip install ddgs", out)


class LogFormatTests(unittest.TestCase):
    def make(self, level: int, msg: str, color: bool) -> str:
        record = logging.LogRecord("limi", level, __file__, 1, msg, None, None)
        return LogFormatter(color).format(record)

    def test_plain_format_matches_requested_shape(self) -> None:
        line = self.make(logging.INFO, "hello", color=False)
        self.assertRegex(line, r"^\[INFO -- \d{2}:\d{2}:\d{2}, \d{2}\.\d{2}\.\d{4}\] hello$")

    def test_levels_are_colored(self) -> None:
        self.assertIn("\x1b[92m[INFO", self.make(logging.INFO, "a", color=True))
        self.assertIn("\x1b[93m[WARN", self.make(logging.WARNING, "a", color=True))
        self.assertIn("\x1b[91m[ERR", self.make(logging.ERROR, "a", color=True))


class FaceTests(unittest.TestCase):
    def test_ears_perk_up_when_listening_and_fold_when_asleep(self) -> None:
        m = FaceModel()
        self.assertEqual(m._targets(State.LISTENING, Mood.NEUTRAL, None, 0.0)["ears"], 1.3)
        self.assertEqual(m._targets(State.SLEEP, Mood.NEUTRAL, None, 0.0)["ears"], 0.4)

    def test_brows_are_no_longer_drawn(self) -> None:
        from limi.face.renderer import FaceRenderer
        self.assertFalse(hasattr(FaceRenderer, "_draw_brows"))


if __name__ == "__main__":
    unittest.main()
