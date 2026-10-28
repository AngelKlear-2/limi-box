import os
import unittest
from unittest import mock

from limi.net import configure_proxy, normalize_proxy


class ProxyTests(unittest.TestCase):
    def test_scheme_is_added_when_missing(self) -> None:
        self.assertEqual(normalize_proxy("127.0.0.1:10809"), "http://127.0.0.1:10809")
        self.assertEqual(normalize_proxy(" socks5://127.0.0.1:10808/ "), "socks5://127.0.0.1:10808")
        self.assertEqual(normalize_proxy(""), "")

    def test_configure_sets_all_vars_and_bypasses_local(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configure_proxy("127.0.0.1:10809"), "http://127.0.0.1:10809")
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://127.0.0.1:10809")
            self.assertEqual(os.environ["https_proxy"], "http://127.0.0.1:10809")
            self.assertIn("localhost", os.environ["NO_PROXY"])

    def test_scheme_less_shell_variable_is_fixed_without_config(self) -> None:
        with mock.patch.dict(os.environ, {"HTTPS_PROXY": "127.0.0.1:10808"}, clear=True):
            configure_proxy(None)
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://127.0.0.1:10808")

    def test_env_overrides_config(self) -> None:
        with mock.patch.dict(os.environ, {"LIMI_PROXY": "127.0.0.1:10809"}, clear=True):
            self.assertEqual(configure_proxy("127.0.0.1:1"), "http://127.0.0.1:10809")


if __name__ == "__main__":
    unittest.main()
