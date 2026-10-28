"""Placeholder music provider: logs calls and does nothing."""

from __future__ import annotations

import logging

from limi.music.base import MusicProvider

log = logging.getLogger("limi.music")


class StubMusicProvider(MusicProvider):
    """Needs no config.

    TODO(next step): replace this with the real stream provider (direct mp3
    URL via mpv/VLC/pygame, or similar) and select it in the music factory.
    """

    def play(self, url_or_query: str) -> None:
        log.info("music.play(%r) ignored: music not implemented yet", url_or_query)

    def stop(self) -> None:
        log.info("music.stop() ignored: music not implemented yet")

    def like(self) -> None:
        log.info("music.like() ignored: music not implemented yet")

    def dislike(self) -> None:
        log.info("music.dislike() ignored: music not implemented yet")
