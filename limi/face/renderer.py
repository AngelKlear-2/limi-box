"""Procedural pygame face.

FaceModel is pure math (no pygame) and turns (state, mood, amplitude, time)
into smooth drawing parameters. FaceRenderer draws them. The face only reads
SharedState; it knows nothing about STT, TTS or the LLM.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, fields, replace
from typing import Callable

from limi.state import Mood, SharedState, State

BLINK_SECONDS = 0.16
BLINK_GAP = (2.0, 5.0)  # random seconds between blinks
LOOK_GAP = (1.2, 3.5)  # random seconds between glances in IDLE
DEFAULT_TAU = 0.067  # ~95% of the way in 200 ms
TAU = {"mouth_open": 0.04, "pupil_x": 0.08, "pupil_y": 0.08}


@dataclass
class FaceParams:
    eye_open: float = 1.0  # 0 closed .. 1 normal .. 1.25 wide
    eye_w: float = 1.0
    eye_y: float = 0.0  # +1 = eyes lower on the face
    pupil_x: float = 0.0  # -1..1
    pupil_y: float = 0.0
    brow_tilt: float = 0.0  # +1 inner ends up (sad), -1 inner ends down (angry)
    brow_raise: float = 0.0
    brow_asym: float = 0.0  # raises the left brow only (thinking)
    mouth_curve: float = 0.0  # +1 smile, -1 frown
    mouth_open: float = 0.0  # 0..1
    mouth_w: float = 1.0
    ring: float = 0.0  # listening ring visibility
    dots: float = 0.0  # thinking dots visibility
    zzz: float = 0.0  # sleep letters visibility
    ears: float = 1.0  # cat ear height: perked up when listening, flat when asleep


class FaceModel:
    """Smooths face parameters toward targets set by state and mood."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()
        self.params = FaceParams()
        self.blinks = 0
        self._blink = 1.0  # multiplier on eye_open, 1 = fully open
        self._blink_start: float | None = None
        self._next_blink: float | None = None
        self._next_look: float | None = None
        self._look = (0.0, 0.0)

    def update(self, dt: float, state: State, mood: Mood, amplitude: float | None, t: float) -> None:
        """Advance the animation by dt seconds; t is a monotonically rising clock."""
        for name, goal in self._targets(state, mood, amplitude, t).items():
            current = getattr(self.params, name)
            k = 1.0 - math.exp(-max(dt, 0.0) / TAU.get(name, DEFAULT_TAU))
            setattr(self.params, name, current + (goal - current) * k)
        self._update_blink(state, t)

    def render_params(self) -> FaceParams:
        """Current params with the blink applied."""
        return replace(self.params, eye_open=self.params.eye_open * self._blink)

    # targets ----------------------------------------------------------------

    def _targets(self, state: State, mood: Mood, amp: float | None, t: float) -> dict[str, float]:
        g = {f.name: getattr(FaceParams(), f.name) for f in fields(FaceParams)}
        g["mouth_curve"] = 0.1  # neutral face has a faint smile

        if state is State.SLEEP:
            g.update(eye_open=0.0, mouth_w=0.5, zzz=1.0, ears=0.4)
        elif state in (State.LISTENING, State.AWAIT_CONFIRM):
            g.update(eye_open=1.25, eye_w=1.15, ring=1.0, ears=1.3)
        elif state is State.THINKING:
            g.update(eye_open=0.95, dots=1.0, pupil_x=0.7 * math.sin(t * 2.2), pupil_y=-0.4)
        elif state is State.SPEAKING:
            g["mouth_open"] = self._mouth_open(amp, t)
        else:  # IDLE
            self._update_look(t)
            g.update(pupil_x=self._look[0], pupil_y=self._look[1])

        if mood is Mood.HAPPY:
            g.update(mouth_curve=0.85, brow_raise=0.3)
            g["eye_open"] *= 0.85
        elif mood is Mood.SAD:
            g.update(mouth_curve=-0.75, brow_tilt=0.8, eye_y=0.3)
            g["mouth_w"] *= 0.7
            g["eye_open"] *= 0.8
        elif mood is Mood.ANGRY:
            g.update(mouth_curve=0.0, brow_tilt=-0.9, brow_raise=-0.2)
            g["mouth_w"] *= 0.8
            g["eye_open"] *= 0.75
        elif mood is Mood.THINKING:
            g.update(mouth_curve=-0.1, brow_asym=0.7)
            g["mouth_w"] *= 0.5
            if state in (State.IDLE, State.SPEAKING):
                g["pupil_x"] = max(-1.0, g["pupil_x"] - 0.4)
                g["pupil_y"] = max(-1.0, g["pupil_y"] - 0.3)
        return g

    @staticmethod
    def _mouth_open(amp: float | None, t: float) -> float:
        if amp is None:  # no audio level available: simple talking animation
            return 0.15 + 0.55 * abs(math.sin(t * 9.0))
        return min(1.0, max(0.0, amp) * 1.2)

    def _update_look(self, t: float) -> None:
        if self._next_look is None or t >= self._next_look:
            if self._rng.random() < 0.3:
                self._look = (0.0, 0.0)
            else:
                self._look = (self._rng.uniform(-0.8, 0.8), self._rng.uniform(-0.4, 0.4))
            self._next_look = t + self._rng.uniform(*LOOK_GAP)

    def _update_blink(self, state: State, t: float) -> None:
        if self._next_blink is None:
            self._next_blink = t + self._rng.uniform(*BLINK_GAP)
        if self._blink_start is None:
            self._blink = 1.0
            if t >= self._next_blink and state is not State.SLEEP:
                self._blink_start = t
                self.blinks += 1
            return
        p = (t - self._blink_start) / BLINK_SECONDS
        if p >= 1.0:
            self._blink_start = None
            self._blink = 1.0
            self._next_blink = t + self._rng.uniform(*BLINK_GAP)
        else:
            self._blink = 0.5 * (1.0 + math.cos(2.0 * math.pi * p))  # 1 -> 0 -> 1


# --- drawing -----------------------------------------------------------------

BG = (10, 12, 20)
EYE = (225, 238, 255)
PUPIL = (18, 24, 40)
LINE = (150, 200, 255)
MOUTH_IN = (70, 20, 35)
ACCENT = (120, 220, 255)
EAR_OUT = (60, 70, 100)
EAR_IN = (255, 140, 180)
EYE_X = (0.30, 0.70)
EYE_Y = 0.40


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], k: float) -> tuple[int, int, int]:
    k = max(0.0, min(1.0, k))
    return (int(a[0] + (b[0] - a[0]) * k), int(a[1] + (b[1] - a[1]) * k), int(a[2] + (b[2] - a[2]) * k))


class FaceRenderer:
    """Draws the face in a pygame window. Run it in the main thread.

    Needs config: display.width, display.height, display.fullscreen.
    pygame is imported lazily so the rest of Limi works without it.
    """

    def __init__(
        self, shared: SharedState, width: int = 480, height: int = 320, fullscreen: bool = False, fps: int = 30
    ) -> None:
        self._shared = shared
        self._w, self._h = width, height
        self._fullscreen = fullscreen
        self._fps = fps
        self._model = FaceModel()
        self.caption = ""  # optional small text in the corner (used by the demo)
        self._pg = None
        self._screen = None
        self._fonts: list = []
        self._small_font = None

    def open(self) -> None:
        """Create the window. Raises if pygame is missing or there is no display."""
        import pygame

        self._pg = pygame
        pygame.display.init()
        pygame.font.init()
        flags = pygame.FULLSCREEN if self._fullscreen else 0
        self._screen = pygame.display.set_mode((self._w, self._h), flags)
        pygame.display.set_caption("Limi")
        pygame.mouse.set_visible(not self._fullscreen)
        self._fonts = [pygame.font.Font(None, int(self._h * s)) for s in (0.10, 0.14, 0.18)]
        self._small_font = pygame.font.Font(None, max(14, int(self._h * 0.06)))

    def run(self, tick: Callable[[float], bool] | None = None) -> None:
        """Main loop. `tick(now)` runs every frame; returning False ends the loop.

        Closing the window, Esc or Q set shared.quit.
        """
        pg = self._pg
        if pg is None:
            self.open()
            pg = self._pg
        clock = pg.time.Clock()
        last = time.monotonic()
        while not self._shared.quit.is_set():
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key in (pg.K_ESCAPE, pg.K_q)):
                    self._shared.quit.set()
            now = time.monotonic()
            dt = min(now - last, 0.1)
            last = now
            if tick is not None and tick(now) is False:
                break
            state, mood, amp = self._shared.snapshot()
            self._model.update(dt, state, mood, amp, now)
            self._draw(self._model.render_params(), now)
            clock.tick(self._fps)
        self.close()

    def close(self) -> None:
        if self._pg is not None:
            self._pg.display.quit()
            self._pg = None

    # drawing helpers ------------------------------------------------------

    def _draw(self, p: FaceParams, t: float) -> None:
        pg, s = self._pg, self._screen
        w, h = s.get_size()
        s.fill(BG)
        self._draw_ears(p, w, h)
        self._draw_eyes(p, w, h)
        self._draw_mouth(p, w, h)
        self._draw_ring(p, w, h, t)
        self._draw_dots(p, w, h, t)
        self._draw_zzz(p, w, h, t)
        if self.caption and self._small_font is not None:
            s.blit(self._small_font.render(self.caption, True, (90, 100, 130)), (8, 6))
        pg.display.flip()

    def _draw_eyes(self, p: FaceParams, w: int, h: int) -> None:
        pg, s = self._pg, self._screen
        ew = 0.17 * w * p.eye_w
        full_h = 0.24 * h
        eh = full_h * max(p.eye_open, 0.0)
        cy = (EYE_Y + 0.04 * p.eye_y) * h
        for fx in EYE_X:
            cx = fx * w
            if eh < 4:  # closed: a gentle downward curve
                rect = pg.Rect(0, 0, max(2, int(ew)), max(2, int(full_h * 0.5)))
                rect.center = (int(cx), int(cy - full_h * 0.1))
                pg.draw.arc(s, LINE, rect, math.pi, 2 * math.pi, max(3, int(0.012 * w)))
                continue
            rect = pg.Rect(0, 0, int(ew), int(eh))
            rect.center = (int(cx), int(cy))
            pg.draw.ellipse(s, EYE, rect)
            radius = max(2, int(min(ew, eh) * 0.30))
            px = cx + p.pupil_x * max(0.0, ew / 2 - radius) * 0.9
            py = cy + p.pupil_y * max(0.0, eh / 2 - radius) * 0.9
            pg.draw.circle(s, PUPIL, (int(px), int(py)), radius)

    def _draw_ears(self, p: FaceParams, w: int, h: int) -> None:
        """Cat ears on top of the head, one triangle per side with a pink inner part."""
        pg, s = self._pg, self._screen
        base_y = 0.27 * h
        top_y = base_y - 0.19 * h * max(p.ears, 0.05)
        for fx in EYE_X:
            sign = -1 if fx < 0.5 else 1  # ears point outward
            cx = fx * w
            outer = [(cx + sign * 0.11 * w, base_y), (cx + sign * 0.05 * w, top_y), (cx - sign * 0.05 * w, base_y)]
            pg.draw.polygon(s, EAR_OUT, outer)
            pg.draw.polygon(s, LINE, outer, max(2, int(0.008 * w)))
            gx = sum(x for x, _ in outer) / 3
            gy = sum(y for _, y in outer) / 3
            inner = [(gx + 0.55 * (x - gx), gy + 0.55 * (y - gy)) for x, y in outer]
            pg.draw.polygon(s, EAR_IN, inner)

    def _draw_mouth(self, p: FaceParams, w: int, h: int) -> None:
        pg, s = self._pg, self._screen
        cx, cy = w / 2, 0.74 * h
        half = 0.16 * w * p.mouth_w
        curve = p.mouth_curve * 0.07 * h
        opening = p.mouth_open * 0.13 * h
        n = 24
        top, bottom = [], []
        for i in range(n + 1):
            x = cx - half + 2 * half * i / n
            # float drift can push |u| a hair past 1; clamp so (1-u*u)**0.6 never sees a negative base
            u = max(-1.0, min(1.0, (x - cx) / half)) if half else 0.0
            y = cy + curve * (1 - u * u)
            top.append((x, y))
            bottom.append((x, y + opening * (1 - u * u) ** 0.6))
        thick = max(3, int(0.010 * w))
        if opening > 2:
            pg.draw.polygon(s, MOUTH_IN, top + bottom[::-1])
            pg.draw.lines(s, LINE, False, bottom, thick)
        pg.draw.lines(s, LINE, False, top, thick)

    def _draw_ring(self, p: FaceParams, w: int, h: int, t: float) -> None:
        if p.ring < 0.05:
            return
        radius = 0.04 * h * (1 + 0.2 * math.sin(t * 5))
        self._pg.draw.circle(
            self._screen, _mix(BG, ACCENT, p.ring), (w // 2, int(0.10 * h)), max(2, int(radius)), 3
        )

    def _draw_dots(self, p: FaceParams, w: int, h: int, t: float) -> None:
        if p.dots < 0.05:
            return
        color = _mix(BG, ACCENT, p.dots)
        for i in range(3):
            bounce = max(0.0, math.sin(t * 5 - i * 0.9))
            pos = (int(w / 2 + (i - 1) * 0.06 * w), int(0.91 * h - bounce * 0.03 * h))
            self._pg.draw.circle(self._screen, color, pos, max(2, int(0.012 * w)))

    def _draw_zzz(self, p: FaceParams, w: int, h: int, t: float) -> None:
        if p.zzz < 0.05 or not self._fonts:
            return
        color = _mix(BG, ACCENT, p.zzz)
        for i, font in enumerate(self._fonts):
            bob = 0.012 * h * math.sin(t * 2 + i)
            surf = font.render("z", True, color)
            self._screen.blit(surf, (int(0.68 * w + i * 0.06 * w), int(0.27 * h - i * 0.08 * h + bob)))
