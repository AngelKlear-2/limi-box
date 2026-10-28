import unittest

from limi.llm.parser import clean_text, parse_llm_output


class ParserTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        r = parse_llm_output('{"text": "Hello there", "mood": "happy", "tool_call": null}')
        self.assertEqual(r.text, "Hello there")
        self.assertEqual(r.mood, "happy")
        self.assertEqual(r.tool_calls, [])

    def test_fenced_json(self) -> None:
        r = parse_llm_output('```json\n{"text": "Hi", "mood": "sad", "tool_call": null}\n```')
        self.assertEqual((r.text, r.mood), ("Hi", "sad"))

    def test_json_buried_in_prose(self) -> None:
        r = parse_llm_output('Sure! {"text": "Ok {braces} inside", "mood": "neutral"} hope that helps')
        self.assertEqual(r.text, "Ok {braces} inside")

    def test_plain_text_falls_back_to_raw_and_neutral(self) -> None:
        r = parse_llm_output("I am not JSON at all")
        self.assertEqual(r.text, "I am not JSON at all")
        self.assertEqual(r.mood, "neutral")
        self.assertEqual(r.tool_calls, [])

    def test_truncated_json_never_reads_braces_aloud(self) -> None:
        r = parse_llm_output('{"text": "Half an ans", "mood": ')
        self.assertEqual(r.text, "Half an ans")
        self.assertEqual(r.mood, "neutral")

    def test_empty_output(self) -> None:
        r = parse_llm_output("")
        self.assertEqual((r.text, r.mood), ("", "neutral"))

    def test_unknown_mood_becomes_neutral(self) -> None:
        self.assertEqual(parse_llm_output('{"text": "x", "mood": "ecstatic"}').mood, "neutral")

    def test_mood_is_case_insensitive(self) -> None:
        self.assertEqual(parse_llm_output('{"text": "x", "mood": "ANGRY"}').mood, "angry")

    def test_tool_call_object(self) -> None:
        r = parse_llm_output('{"text": "", "mood": "thinking", "tool_call": {"name": "web_search", "arguments": {"query": "rain"}}}')
        self.assertEqual(len(r.tool_calls), 1)
        self.assertEqual(r.tool_calls[0].name, "web_search")
        self.assertEqual(r.tool_calls[0].arguments, {"query": "rain"})

    def test_tool_call_arguments_as_string_and_list(self) -> None:
        r = parse_llm_output('{"text": "", "tool_call": [{"name": "a", "arguments": "{\\"x\\": 1}"}, {"name": "b"}]}')
        self.assertEqual([c.name for c in r.tool_calls], ["a", "b"])
        self.assertEqual(r.tool_calls[0].arguments, {"x": 1})
        self.assertEqual(r.tool_calls[1].arguments, {})

    def test_bad_tool_call_is_ignored(self) -> None:
        r = parse_llm_output('{"text": "ok", "tool_call": {"arguments": {}}}')
        self.assertEqual(r.tool_calls, [])
        self.assertEqual(r.text, "ok")

    def test_json_without_reply_keys_is_treated_as_text(self) -> None:
        raw = '{"foo": 1}'
        self.assertEqual(parse_llm_output(raw).text, raw)

    def test_text_is_cleaned_inside_json(self) -> None:
        r = parse_llm_output('{"text": "**Sure** 😀 here you go", "mood": "happy"}')
        self.assertEqual(r.text, "Sure here you go")

    def test_clean_text_markdown_and_emoji(self) -> None:
        raw = "# Title\n- **bold** and `code` 🎉\n1. [link](http://a.b) ok ✨"
        self.assertEqual(clean_text(raw), "Title bold and code link ok")

    def test_clean_text_drops_code_blocks_and_keeps_cyrillic(self) -> None:
        self.assertEqual(clean_text("Привет! ```print(1)``` Как дела?"), "Привет! Как дела?")


if __name__ == "__main__":
    unittest.main()
