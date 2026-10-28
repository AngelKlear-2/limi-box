"""Music tools. STUBS ONLY for now."""

from __future__ import annotations

from limi.music.base import MusicProvider
from limi.tools.base import Tool, object_schema

NOT_IMPLEMENTED = "music not implemented yet"


def make_music_tools(music: MusicProvider) -> list[Tool]:
    """Music tools bound to a provider (the stub today).

    TODO(next step): when a real MusicProvider exists, return its real status
    from these functions instead of NOT_IMPLEMENTED, and add music_stop.
    """

    def music_play(query: str) -> str:
        music.play(str(query))
        return NOT_IMPLEMENTED

    def music_like() -> str:
        music.like()
        return NOT_IMPLEMENTED

    def music_dislike() -> str:
        music.dislike()
        return NOT_IMPLEMENTED

    return [
        Tool(
            "music_play",
            "Play music by song, artist or mood (not available yet).",
            object_schema({"query": {"type": "string", "description": "what to play"}}, ["query"]),
            music_play,
        ),
        Tool("music_like", "Like the current track (not available yet).", object_schema(), music_like),
        Tool("music_dislike", "Dislike the current track (not available yet).", object_schema(), music_dislike),
    ]
