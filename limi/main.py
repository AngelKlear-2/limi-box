"""Entry point and main loop: python -m limi.main"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time

from limi.audio.mic import Microphone
from limi.audio.player import Player
from limi.audio.wake import WakeDetector, WakeEvent
from limi.config import Config, ConfigError, load_config, resolve_path
from limi.dialog import Dialog, DialogReply
from limi.llm.base import ProviderUnavailable
from limi.logutil import setup_logging
from limi.net import configure_proxy
from limi.state import Mood, SharedState, State
from limi.stt.base import STTProvider
from limi.text import detect_language, lang_code
from limi.tts.base import TTSProvider

log = logging.getLogger("limi")

DEMO_STATES = (State.SLEEP, State.IDLE, State.LISTENING, State.THINKING, State.SPEAKING)


class Assistant:
    """The voice loop: wake word -> listen -> STT -> LLM -> TTS -> back to idle."""

    def __init__(
        self,
        cfg: Config,
        shared: SharedState,
        mic: Microphone,
        wake: WakeDetector,
        stt: STTProvider,
        tts: TTSProvider,
        player: Player,
        dialog: Dialog,
    ) -> None:
        self.cfg, self.shared = cfg, shared
        self.mic, self.wake, self.stt, self.tts = mic, wake, stt, tts
        self.player, self.dialog = player, dialog

    def run(self) -> None:
        """Run until shared.quit is set. Errors in a turn never end the loop."""
        pending: WakeEvent | None = None
        while not self.shared.quit.is_set():
            try:
                event = pending or self._wait_for_wake()
                pending = None
                if event is not None:
                    pending = self._turn(event)
            except Exception:
                log.exception("turn failed")
                pending = None
                self._on_error()
                time.sleep(0.5)

    # idle ---------------------------------------------------------------------

    def _wait_for_wake(self) -> WakeEvent | None:
        while not self.shared.quit.is_set():
            event = self.wake.wait(self.mic, self.shared.quit, timeout=1.0)
            if event is not None:
                return event
            asleep = self.shared.state is State.SLEEP
            if not asleep and self.shared.idle_for() >= self.cfg.behavior.sleep_after_s:
                self._go_to_sleep()
        return None

    def _go_to_sleep(self) -> None:
        self.player.play_event("sleep")
        self.shared.mood = Mood.NEUTRAL
        self.shared.state = State.SLEEP

    def _to_idle(self) -> None:
        self.shared.mood = Mood.NEUTRAL
        self.shared.amplitude = None
        self.shared.state = State.IDLE
        self.shared.touch()

    def _on_error(self) -> None:
        try:
            self.player.stop()
            self.player.play_event("error", block=True)
        except Exception:
            log.debug("could not play error sound", exc_info=True)
        self._to_idle()

    # one conversation turn ----------------------------------------------------

    def _turn(self, event: WakeEvent) -> WakeEvent | None:
        """Handle one wake-to-idle turn. Returns a new WakeEvent if speech was interrupted."""
        s = self.shared
        s.touch()
        self.dialog.cancel_confirm()  # a fresh wake word abandons any pending confirmation
        s.mood = Mood.NEUTRAL
        s.state = State.LISTENING
        self.player.play_event("wake", block=True)

        text = event.remainder.strip() or self._hear(self.cfg.behavior.listen_timeout_s)
        if not text:
            self._to_idle()  # nothing said: back to waiting for the wake word
            return None

        self.player.play_event("ack")
        s.state, s.mood = State.THINKING, Mood.THINKING
        interrupted = self._speak(self.dialog.process(text))
        while interrupted is None and self.dialog.awaiting_confirm:
            self.dialog.restart_confirm_timer()
            s.state, s.mood = State.AWAIT_CONFIRM, Mood.NEUTRAL
            answer = self._hear(self.dialog.confirm_remaining())
            if not answer:
                self.dialog.cancel_confirm()  # silence: nothing is deleted
                break
            s.state, s.mood = State.THINKING, Mood.THINKING
            interrupted = self._speak(self.dialog.process(answer))
        if interrupted is None:
            self._to_idle()
        return interrupted

    def _hear(self, timeout_s: float) -> str | None:
        """Record one utterance and transcribe it. None if nobody spoke."""
        self.mic.flush()  # drop our own beeps and speech
        pcm = self.mic.record_utterance(timeout_s, self.cfg.behavior.max_utterance_s, self.shared.quit)
        if pcm is None:
            return None
        self.shared.state = State.THINKING
        text = self.stt.transcribe(pcm, self.cfg.language).strip()
        log.info("heard: %r", text)
        return text or None

    def _speak(self, reply: DialogReply) -> WakeEvent | None:
        """Speak a reply. Returns a WakeEvent if the wake word interrupted it."""
        s = self.shared
        mood = Mood.parse(reply.mood)
        if mood is not s.mood and mood in (Mood.HAPPY, Mood.SAD, Mood.ANGRY):
            self.player.play_event(f"mood_{mood.value}")
        s.mood = mood
        if reply.sound:
            self.player.play_event(reply.sound, block=True)
        if not reply.text.strip():
            return None
        lang = reply.language or detect_language(reply.text, lang_code(self.cfg.language))
        wav = self.tts.synthesize(reply.text, lang)
        s.state = State.SPEAKING
        return self._play_speech(wav)

    def _play_speech(self, wav: bytes) -> WakeEvent | None:
        playback = self.player.play_wav(wav, level_cb=self.shared.set_amplitude)
        interrupted: list[WakeEvent] = []
        stop_watch = threading.Event()

        def watch() -> None:
            event = self.wake.wait(self.mic, stop_watch, None)
            if event is not None:
                interrupted.append(event)
                playback.stop()

        watcher = None
        if self.cfg.behavior.interrupt_on_wake:
            self.mic.flush()
            watcher = threading.Thread(target=watch, name="wake-monitor", daemon=True)
            watcher.start()
        try:
            while not playback.wait(0.05):
                if self.shared.quit.is_set():
                    playback.stop()
        finally:
            stop_watch.set()
            if watcher is not None:
                watcher.join(timeout=10)
                if watcher.is_alive():
                    log.warning("wake monitor is still busy; continuing")
            self.shared.amplitude = None
        return interrupted[0] if interrupted else None


# --- startup -----------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m limi.main", description="Limi voice assistant")
    p.add_argument("--config", help="path to config.yaml (default: project root)")
    p.add_argument("--no-face", action="store_true", help="run without the pygame window")
    p.add_argument("--demo-face", action="store_true", help="cycle every state and mood, no audio needed")
    p.add_argument("--demo-seconds", type=float, default=1.0, help="seconds per state/mood in the demo")
    p.add_argument("--demo-once", action="store_true", help="exit after one full demo cycle")
    p.add_argument("--debug", action="store_true", help="verbose logging")
    return p.parse_args(argv)


def _print_problems(problems: list[str]) -> None:
    print("Limi cannot start. Please fix:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)


def run_face_demo(args: argparse.Namespace) -> int:
    """Cycle all 5 states x 5 moods on screen. Needs pygame and a display only."""
    from limi.face.renderer import FaceRenderer

    try:
        display = load_config(args.config, validate_config=False).display
        size = (display.width, display.height, display.fullscreen)
    except ConfigError:
        size = (480, 320, False)
    shared = SharedState()
    renderer = FaceRenderer(shared, *size)
    try:
        renderer.open()
    except Exception as exc:
        print(f"Cannot open the face window: {exc}\nInstall pygame (pip install pygame) and make sure a display "
              "is available (or set SDL_VIDEODRIVER=dummy for a headless test).", file=sys.stderr)
        return 2

    combos = [(st, mood) for st in DEMO_STATES for mood in Mood]
    start = time.monotonic()
    shown = -1

    def tick(now: float) -> bool:
        nonlocal shown
        idx = int((now - start) / max(args.demo_seconds, 0.05))
        if args.demo_once and idx >= len(combos):
            return False
        state, mood = combos[idx % len(combos)]
        shared.state, shared.mood = state, mood
        shared.amplitude = abs(math.sin(now * 7)) * 0.8 if state is State.SPEAKING else None
        if idx != shown:
            shown = idx
            renderer.caption = f"{state.value} / {mood.value}"
            print(f"[{idx % len(combos) + 1:>2}/{len(combos)}] {state.value:<9} {mood.value}")
        return True

    try:
        renderer.run(tick)
    except KeyboardInterrupt:
        pass
    return 0


def build_assistant(cfg: Config, shared: SharedState) -> Assistant:
    """Create every component. Raises ProviderUnavailable with a readable message."""
    from limi.factory import build_llm, build_stt, build_tts, build_wake
    from limi.memory.store import MemoryStore
    from limi.music.stub import StubMusicProvider
    from limi.tools.registry import build_registry

    llm = build_llm(cfg)
    stt = build_stt(cfg)
    tts = build_tts(cfg)
    wake = build_wake(cfg, stt)
    store = MemoryStore(resolve_path(cfg.memory.db_path))
    registry = build_registry(store, StubMusicProvider())
    dialog = Dialog(cfg, llm, registry, store)
    player = Player(resolve_path(cfg.sounds_dir), cfg.audio.output_device)
    mic = Microphone(cfg.audio.input_device, cfg.audio.vad_min_rms, cfg.audio.end_silence_s)
    return Assistant(cfg, shared, mic, wake, stt, tts, player, dialog)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    redactor = setup_logging(logging.DEBUG if args.debug else logging.INFO)
    configure_proxy(None)  # fix scheme-less proxy env vars before any client is created

    if args.demo_face:
        return run_face_demo(args)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        _print_problems(exc.problems)
        return 2
    redactor.add(*cfg.secrets())
    proxy = configure_proxy(cfg.network.proxy)
    if proxy:
        log.info("Using proxy %s for network calls", proxy)
    log.info("Starting %s: %s", cfg.name, cfg.describe())

    shared = SharedState()
    try:
        assistant = build_assistant(cfg, shared)
        assistant.mic.start()
    except ProviderUnavailable as exc:
        _print_problems([str(exc)])
        return 2
    except ImportError as exc:
        _print_problems([f"Missing Python package: {exc}. Run: pip install -r requirements.txt"])
        return 2
    except Exception as exc:  # e.g. no microphone
        log.debug("startup failure", exc_info=True)
        _print_problems([f"Startup failed: {exc}"])
        return 2

    renderer = None
    if not args.no_face:
        try:
            from limi.face.renderer import FaceRenderer

            renderer = FaceRenderer(shared, cfg.display.width, cfg.display.height, cfg.display.fullscreen)
            renderer.open()
        except Exception as exc:
            log.warning("No face window (%s); running without it. Use --no-face to silence this.", exc)
            renderer = None

    worker = threading.Thread(target=assistant.run, name="assistant", daemon=True)
    try:
        log.info("Say '%s' to wake me. Ctrl+C to quit.", cfg.wake_words[0])
        worker.start()
        if renderer is not None:
            renderer.run()  # blocks in the main thread until the window closes
        else:
            while worker.is_alive():
                worker.join(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        shared.quit.set()
        worker.join(timeout=5)
        assistant.mic.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
