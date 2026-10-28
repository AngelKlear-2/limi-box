"""Logging setup with secret redaction."""

from __future__ import annotations

import logging


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
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    redactor = RedactFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(redactor)
    return redactor
