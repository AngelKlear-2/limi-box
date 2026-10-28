"""Music provider interface.

TODO(next step): the real stream provider plugs in here. Likely a direct
mp3/stream URL played through mpv, VLC or pygame. Nothing is implemented yet,
see stub.py for the placeholder that is wired in today.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class MusicProvider(ABC):
    @abstractmethod
    def play(self, url_or_query: str) -> None:
        """Start playing a stream URL or search for the query."""

    @abstractmethod
    def stop(self) -> None:
        """Stop playback."""

    @abstractmethod
    def like(self) -> None:
        """Mark the current track as liked."""

    @abstractmethod
    def dislike(self) -> None:
        """Mark the current track as disliked."""
