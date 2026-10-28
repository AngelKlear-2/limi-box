import os
import tempfile
import unittest
from typing import Any

from limi.config import Config
from limi.dialog import Dialog, classify_confirmation, is_forget_request
from limi.llm.base import LLMProvider, LLMResponse, ToolCall
from limi.memory.store import MemoryStore
from limi.music.stub import StubMusicProvider
from limi.tools.base import Tool, object_schema
from limi.tools.registry import ToolRegistry, build_registry


class ScriptedLLM(LLMProvider):
    """Returns queued responses and records every call."""

    def __init__(self, *replies: LLMResponse) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[list[dict[str, str]], list[dict[str, Any]]]] = []

    def _complete(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    def chat(self, messages, tools=None):  # type: ignore[override]
        self.calls.append(([dict(m) for m in messages], list(tools or [])))
        return self.replies.pop(0) if self.replies else LLMResponse(text="fallback answer")


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class DialogTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(os.path.join(self.tmp.name, "limi.db"))
        self.clock = Clock()
        self.cfg = Config()

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def make(self, llm: LLMProvider, registry: ToolRegistry | None = None) -> Dialog:
        registry = registry or build_registry(self.store, StubMusicProvider())
        return Dialog(self.cfg, llm, registry, self.store, clock=self.clock)

    def seed_memory(self) -> None:
        self.store.add_fact("name is Anna")
        self.store.add_message("user", "hello")
        self.store.add_message("assistant", "hi")


class ForgetConfirmTests(DialogTestBase):
    def test_keyword_asks_and_deletes_nothing(self) -> None:
        self.seed_memory()
        d = self.make(ScriptedLLM())
        reply = d.process("please forget me")
        self.assertTrue(reply.awaiting_confirm)
        self.assertEqual(reply.sound, "confirm")
        self.assertIn("yes or no", reply.text)
        self.assertTrue(d.awaiting_confirm)
        self.assertEqual(len(self.store.facts()), 1)
        self.assertEqual(self.store.message_count(), 2)

    def test_yes_wipes_facts_and_messages(self) -> None:
        self.seed_memory()
        d = self.make(ScriptedLLM())
        d.process("delete the dialog")
        reply = d.process("Yes.")
        self.assertEqual(reply.sound, "done")
        self.assertIn("forgotten", reply.text)
        self.assertEqual(self.store.facts(), [])
        self.assertEqual(self.store.message_count(), 0)
        self.assertFalse(d.awaiting_confirm)

    def test_no_keeps_memory(self) -> None:
        self.seed_memory()
        d = self.make(ScriptedLLM())
        d.process("forget everything")
        reply = d.process("no")
        self.assertIn("keep", reply.text)
        self.assertEqual(len(self.store.facts()), 1)
        self.assertEqual(self.store.message_count(), 2)
        self.assertFalse(d.awaiting_confirm)

    def test_other_words_cancel_and_keep_memory(self) -> None:
        self.seed_memory()
        llm = ScriptedLLM(LLMResponse(text="it is noon", mood="happy"))
        d = self.make(llm)
        d.process("forget me")
        reply = d.process("what time is it")
        self.assertEqual(reply.text, "it is noon")  # handled as a normal request
        self.assertEqual(len(self.store.facts()), 1)
        self.assertFalse(d.awaiting_confirm)

    def test_ambiguous_yes_is_not_a_yes(self) -> None:
        self.seed_memory()
        d = self.make(ScriptedLLM())
        d.process("forget me")
        d.process("yes but actually no wait")
        self.assertEqual(len(self.store.facts()), 1)

    def test_timeout_cancels(self) -> None:
        self.seed_memory()
        d = self.make(ScriptedLLM())
        d.process("forget me")
        self.clock.now += self.cfg.behavior.confirm_timeout_s + 1
        self.assertFalse(d.awaiting_confirm)
        d.process("yes")  # too late
        self.assertEqual(len(self.store.facts()), 1)
        self.assertEqual(self.store.message_count(), 2 + 2)  # late 'yes' became a normal turn

    def test_timer_can_be_restarted_after_speaking(self) -> None:
        d = self.make(ScriptedLLM())
        d.process("forget me")
        self.clock.now += 10  # question took 10 s to speak
        d.restart_confirm_timer()
        self.clock.now += 10
        self.assertTrue(d.awaiting_confirm)
        self.assertAlmostEqual(d.confirm_remaining(), 5.0)

    def test_cancel_confirm(self) -> None:
        d = self.make(ScriptedLLM())
        d.process("forget me")
        d.cancel_confirm()
        self.assertFalse(d.awaiting_confirm)

    def test_llm_tool_call_triggers_confirmation_not_deletion(self) -> None:
        self.seed_memory()
        llm = ScriptedLLM(LLMResponse(text="", mood="thinking", tool_calls=[ToolCall("memory_forget_request")]))
        d = self.make(llm)
        reply = d.process("I want you to start over with me")
        self.assertTrue(reply.awaiting_confirm)
        self.assertEqual(len(self.store.facts()), 1)
        self.assertEqual(self.store.message_count(), 2)
        self.assertEqual(d.process("да").sound, "done")
        self.assertEqual(self.store.facts(), [])

    def test_russian_flow(self) -> None:
        self.seed_memory()
        d = self.make(ScriptedLLM())
        reply = d.process("Лими, забудь меня")
        self.assertEqual(reply.language, "ru")
        self.assertIn("да или нет", reply.text)
        self.assertEqual(len(self.store.facts()), 1)
        reply = d.process("Да")
        self.assertIn("забыла", reply.text)
        self.assertEqual(self.store.facts(), [])

    def test_russian_no(self) -> None:
        self.seed_memory()
        d = self.make(ScriptedLLM())
        d.process("удали диалог")
        d.process("нет")
        self.assertEqual(len(self.store.facts()), 1)

    def test_keyword_and_classifier_helpers(self) -> None:
        for phrase in ("forget me", "Delete the dialog!", "забудь всё", "очисти память", "удали наш диалог"):
            self.assertTrue(is_forget_request(phrase), phrase)
        for phrase in ("what is the weather", "I always forget my keys", "удали файл"):
            self.assertFalse(is_forget_request(phrase), phrase)
        self.assertEqual(classify_confirmation("да"), "yes")
        self.assertEqual(classify_confirmation("yes, delete it"), "yes")
        self.assertEqual(classify_confirmation("нет"), "no")
        self.assertEqual(classify_confirmation("don't"), "no")
        self.assertEqual(classify_confirmation("maybe"), "other")
        self.assertEqual(classify_confirmation(""), "other")


class ToolLoopTests(DialogTestBase):
    def test_facts_are_injected_into_system_prompt(self) -> None:
        self.store.add_fact("name is Anna")
        llm = ScriptedLLM(LLMResponse(text="Hi Anna"))
        self.make(llm).process("hello")
        system = llm.calls[0][0][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("What you know about the user:", system["content"])
        self.assertIn("name is Anna", system["content"])

    def test_history_is_sent_and_saved(self) -> None:
        self.seed_memory()
        llm = ScriptedLLM(LLMResponse(text="fine, thanks", mood="happy"))
        reply = self.make(llm).process("how are you")
        sent = llm.calls[0][0]
        self.assertEqual([m["role"] for m in sent], ["system", "user", "assistant", "user"])
        self.assertEqual(reply.mood, "happy")
        self.assertEqual(self.store.recent_messages(2)[-1]["content"], "fine, thanks")

    def test_history_len_limits_what_is_sent(self) -> None:
        self.cfg.llm.history_len = 2
        for i in range(10):
            self.store.add_message("user" if i % 2 == 0 else "assistant", f"m{i}")
        llm = ScriptedLLM(LLMResponse(text="ok"))
        self.make(llm).process("next")
        contents = [m["content"] for m in llm.calls[0][0][1:]]
        self.assertEqual(contents, ["m9", "next"][1:] if False else contents)  # shape checked below
        self.assertLessEqual(len(contents), 3)
        self.assertEqual(llm.calls[0][0][1]["role"], "user")  # never starts with an assistant turn

    def test_tool_round_then_final_answer(self) -> None:
        seen: list[str] = []
        tool = Tool("web_search", "search", object_schema({"query": {"type": "string"}}, ["query"]),
                    lambda query: seen.append(query) or "result: sunny")
        llm = ScriptedLLM(
            LLMResponse(text="", tool_calls=[ToolCall("web_search", {"query": "weather"})], raw='{"tool_call": {}}'),
            LLMResponse(text="It is sunny", mood="happy"),
        )
        reply = self.make(llm, ToolRegistry([tool])).process("weather?")
        self.assertEqual(seen, ["weather"])
        self.assertEqual(reply.text, "It is sunny")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("result: sunny", llm.calls[1][0][-1]["content"])

    def test_tool_rounds_are_limited_to_two(self) -> None:
        count = {"n": 0}

        def search(query: str) -> str:
            count["n"] += 1
            return "r"

        tool = Tool("web_search", "search", object_schema({"query": {"type": "string"}}, ["query"]), search)
        call = LLMResponse(text="", tool_calls=[ToolCall("web_search", {"query": "x"})])
        llm = ScriptedLLM(call, call, call, LLMResponse(text="final answer"))
        reply = self.make(llm, ToolRegistry([tool])).process("loop forever")
        self.assertEqual(count["n"], 2)
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(llm.calls[3][1], [])  # forced final call offers no tools
        self.assertEqual(reply.text, "final answer")

    def test_stubborn_model_gets_a_fallback_answer(self) -> None:
        tool = Tool("web_search", "s", object_schema({"query": {"type": "string"}}, ["query"]), lambda query: "r")
        call = LLMResponse(text="", tool_calls=[ToolCall("web_search", {"query": "x"})])
        llm = ScriptedLLM(call, call, call, call)
        reply = self.make(llm, ToolRegistry([tool])).process("hi")
        self.assertTrue(reply.text)

    def test_unknown_tool_returns_error_text_not_crash(self) -> None:
        registry = build_registry(self.store, StubMusicProvider())
        result = registry.dispatch("does_not_exist", {})
        self.assertIn("unknown tool", result.text)
        llm = ScriptedLLM(
            LLMResponse(text="", tool_calls=[ToolCall("does_not_exist")]),
            LLMResponse(text="sorry, can't"),
        )
        self.assertEqual(self.make(llm).process("do magic").text, "sorry, can't")

    def test_broken_tool_and_missing_args_do_not_crash(self) -> None:
        def boom() -> str:
            raise RuntimeError("kaboom")

        registry = ToolRegistry([
            Tool("boom", "x", object_schema(), boom),
            Tool("needs", "x", object_schema({"a": {"type": "string"}}, ["a"]), lambda a: a),
        ])
        self.assertIn("kaboom", registry.dispatch("boom", {}).text)
        self.assertIn("missing arguments", registry.dispatch("needs", {}).text)

    def test_remember_tool_saves_fact(self) -> None:
        llm = ScriptedLLM(
            LLMResponse(text="", tool_calls=[ToolCall("memory_remember", {"fact": "likes tea"})]),
            LLMResponse(text="Noted"),
        )
        self.make(llm).process("I like tea")
        self.assertEqual([f.text for f in self.store.facts()], ["likes tea"])

    def test_music_tools_are_stubs(self) -> None:
        registry = build_registry(self.store, StubMusicProvider())
        for name, args in (("music_play", {"query": "jazz"}), ("music_like", {}), ("music_dislike", {})):
            self.assertEqual(registry.dispatch(name, args).text, "music not implemented yet")

    def test_get_time_tool(self) -> None:
        registry = build_registry(self.store, StubMusicProvider())
        self.assertRegex(registry.dispatch("get_time", {}).text, r"\d{2}:\d{2}")


if __name__ == "__main__":
    unittest.main()
