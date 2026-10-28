"""Proxy support for VPN clients such as Happ, v2rayN or Clash.

httpx (used by openai, anthropic and the search package) reads HTTP_PROXY,
HTTPS_PROXY and ALL_PROXY. A value without a scheme ('127.0.0.1:10809') makes it
fail with 'Unknown scheme', so every value gets 'http://' added when it is missing.
Local addresses bypass the proxy so a local LLM server (Ollama, LM Studio) still works.
"""

from __future__ import annotations

import os

PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
LOCAL_BYPASS = "localhost,127.0.0.1,::1"


def normalize_proxy(value: str) -> str:
    """'127.0.0.1:10809' -> 'http://127.0.0.1:10809'. Values with a scheme are kept."""
    value = value.strip()
    if not value:
        return ""
    if "://" not in value:
        value = "http://" + value
    return value.rstrip("/")


def configure_proxy(proxy: str | None) -> str:
    """Apply the proxy to the process environment. Returns the normalized value or ''."""
    value = normalize_proxy(os.environ.get("LIMI_PROXY") or proxy or "")
    if value:
        for var in PROXY_VARS:
            os.environ[var] = value
            os.environ[var.lower()] = value
        bypass = LOCAL_BYPASS
        if os.environ.get("NO_PROXY"):
            bypass = os.environ["NO_PROXY"] + "," + LOCAL_BYPASS
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = bypass
        return value
    # no proxy configured: still fix any scheme-less proxy variables already set in the shell
    for var in PROXY_VARS + tuple(v.lower() for v in PROXY_VARS):
        if os.environ.get(var):
            os.environ[var] = normalize_proxy(os.environ[var])
    return ""
