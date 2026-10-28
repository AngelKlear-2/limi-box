import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from limi.memory.__main__ import main as memory_cli
from limi.memory.store import MAX_STORED_MESSAGES, MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "sub", "limi.db")  # parent dir is created
        self.store = MemoryStore(self.path)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_add_and_list_facts(self) -> None:
        self.assertTrue(self.store.add_fact("Her name is Anna"))
        self.assertTrue(self.store.add_fact("likes jazz"))
        self.assertEqual([f.text for f in self.store.facts()], ["Her name is Anna", "likes jazz"])

    def test_duplicate_and_empty_facts_are_rejected(self) -> None:
        self.assertTrue(self.store.add_fact("likes jazz"))
        self.assertFalse(self.store.add_fact("  Likes   JAZZ "))
        self.assertFalse(self.store.add_fact("   "))
        self.assertEqual(len(self.store.facts()), 1)

    def test_facts_block(self) -> None:
        self.assertEqual(self.store.facts_block(), "")
        self.store.add_fact("name is Sam")
        self.assertEqual(self.store.facts_block(), "What you know about the user:\n- name is Sam")

    def test_recent_messages_are_the_last_n_oldest_first(self) -> None:
        for i in range(5):
            self.store.add_message("user" if i % 2 == 0 else "assistant", f"m{i}")
        recent = self.store.recent_messages(3)
        self.assertEqual([m["content"] for m in recent], ["m2", "m3", "m4"])
        self.assertEqual(recent[0]["role"], "user")
        self.assertEqual(self.store.recent_messages(0), [])

    def test_history_is_rolling(self) -> None:
        for i in range(MAX_STORED_MESSAGES + 10):
            self.store.add_message("user", str(i))
        self.assertEqual(self.store.message_count(), MAX_STORED_MESSAGES)
        self.assertEqual(self.store.recent_messages(1)[0]["content"], str(MAX_STORED_MESSAGES + 9))

    def test_wipe_clears_everything(self) -> None:
        self.store.add_fact("a fact")
        self.store.add_message("user", "hi")
        self.store.wipe()
        self.assertEqual(self.store.facts(), [])
        self.assertEqual(self.store.message_count(), 0)

    def test_data_persists_across_reopen(self) -> None:
        self.store.add_fact("persistent")
        self.store.close()
        self.store = MemoryStore(self.path)
        self.assertEqual([f.text for f in self.store.facts()], ["persistent"])


class MemoryCliTests(unittest.TestCase):
    def test_list_and_wipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "limi.db")
            store = MemoryStore(db)
            store.add_fact("cli fact")
            store.close()

            out = StringIO()
            with redirect_stdout(out):
                self.assertEqual(memory_cli(["list", "--db", db]), 0)
            self.assertIn("cli fact", out.getvalue())

            with redirect_stdout(StringIO()):
                self.assertEqual(memory_cli(["wipe", "--db", db, "--yes"]), 0)
            out = StringIO()
            with redirect_stdout(out):
                memory_cli(["list", "--db", db])
            self.assertIn("No facts stored", out.getvalue())


if __name__ == "__main__":
    unittest.main()
