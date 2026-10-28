"""Logging setup with secret redaction."""

from __future__ import annotations

import logging
import os
import sys
import time

# ANSI colors per level: green INFO, yellow WARN, red ERR (as requested)
LEVELS: dict[int, tuple[str, str]] = {
    logging.DEBUG: ("DEBUG", "\x1b[90m"),
    logging.INFO: ("INFO", "\x1b[92m"),
    logging.WARNING: ("WARN", "\x1b[93m"),
    logging.ERROR: ("ERR", "\x1b[91m"),
    logging.CRITICAL: ("CRIT", "\x1b[95m"),
}
RESET = "\x1b[0m"


class LogFormatter(logging.Formatter):
    """Writes '[INFO -- 13:42:50, 28.10.2026] message', colored when the stream is a terminal."""

    def __init__(self, color: bool) -> None:
        super().__init__("%(message)s")
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        label, code = LEVELS.get(record.levelno, (record.levelname, ""))
        stamp = time.strftime("%H:%M:%S, %d.%m.%Y", time.localtime(record.created))
        line = f"[{label} -- {stamp}] {super().format(record)}"
        return f"{code}{line}{RESET}" if self._color and code else line


def mask_secret(value: str) -> str:
    """Show just enough of a secret to recognise it: 'sk-***ab'."""
    if not value:
        return "(unset)"
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}***{value[-2:]}"


class RedactFilter(logging.Filter):
    """Replaces any registered secret found in a log line with '***'."""

    def __init__(self) -> None:
        super().__init__()
        self._secrets: set[str] = set()

    def add(self, *secrets: str) -> None:
        self._secrets.update(s for s in secrets if s)

    def filter(self, record: logging.LogRecord) -> bool:
        if self._secrets:
            message = record.getMessage()
            for secret in self._secrets:
                message = message.replace(secret, "***")
            record.msg, record.args = message, ()
        return True


def setup_logging(level: int = logging.INFO) -> RedactFilter:
    """Configure root logging and return the filter so secrets can be added later."""
    color = sys.stderr.isatty() and not os.environ.get("NO_COLOR")
    if color and os.name == "nt":
        os.system("")  # enables ANSI escape codes in the Windows console
    handler = logging.StreamHandler()
    handler.setFormatter(LogFormatter(color))
    redactor = RedactFilter()
    handler.addFilter(redactor)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    return redactor
