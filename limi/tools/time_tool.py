"""get_time tool."""

from __future__ import annotations

from datetime import datetime

from limi.tools.base import Tool, object_schema


def get_time() -> str:
    """Local date and time of the machine Limi runs on. Needs no config."""
    now = datetime.now().astimezone()
    return now.strftime("%A, %d %B %Y, %H:%M (UTC%z)")


TOOL = Tool(
    name="get_time",
    description="Get the current local time and date.",
    parameters=object_schema(),
    func=get_time,
)
