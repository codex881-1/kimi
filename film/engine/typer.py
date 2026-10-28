"""Typewriter text: human rhythm, per-character ink-in, cursor, and a sound event per key."""
from __future__ import annotations

import numpy as np
import skia

from .core import col, draw_text, font, shape, expo_out, clamp01, Layer
from .sfx import sfx


class Typer:
    def __init__(self, text, t0, face="mono", size=30, cps=17.0, seed=1, tracking=0.0, sound=True,
                 gain_db=0.0, pan=0.0, line_h=None, typewriter=False, cursor=True, cursor_hold=0.9,
                 key_kind="key"):
        self.text, self.t0, self.face, self.size = text, t0, face, size
        self.tracking, self.typewriter, self.cursor = tracking, typewriter, cursor
        self.line_h = line_h or size * 1.32
        rng = np.random.default_rng(seed)
        times, t, burst = [], t0, 0
        for i, ch in enumerate(text):
            times.append(t)
            dt = (1.0 / cps) * rng.lognormal(0.0, 0.30)
            if burst > 0:
                dt *= 0.72
                burst -= 1
            elif rng.random() < 0.12:
                burst = int(rng.integers(2, 5))
            if ch == " ":
                dt *= 1.25
            elif ch in ",:;":
                dt *= 2.0
            elif ch in ".!?\u2014":
                dt *= 2.6
            elif ch == "\n":
                dt *= 3.0
            t += dt
        self.times = np.array(times)
        self.t_end = t
        self.cursor_hold = cursor_hold
        self.lines = text.split("\n")
        # per-char typewriter variation (density + baseline)
        n = len(text)
        self.dens = 0.80 + 0.20 * rng.random(n) if typewriter else np.ones(n)
        self.jy = rng.normal(0, 0.55, n) if typewriter else np.zeros(n)
        self.rot = rng.normal(0, 0.5, n) if typewriter else np.zeros(n)
        if sound:
            for i, ch in enumerate(text):
                if ch == "\n":
                    continue
                sfx(times[i], key_kind, gain_db=gain_db + (-2.5 if ch == " " else 0.0), pan=pan,
                    space=(ch == " "), seed=int(seed * 1000 + i))

    def visible_count(self, t):
        return int(np.searchsorted(self.times, t, side="right"))

    def draw(self, c: skia.Canvas, t, x, y, color, alpha=1.0, align="left", glow=None):
        if t < self.t0 - 0.01 or alpha <= 0.003:
            return
        n_vis = self.visible_count(t)
        idx = 0
        cx_end, cy_end = x, y
        for li, line in enumerate(self.lines):
            ly = y + li * self.line_h
            xs, wline = shape(self.face, self.size, line, self.tracking)
            lx = x - (wline / 2 if align == "center" else wline if align == "right" else 0)
            f = font(self.face, self.size)
            for ci, ch in enumerate(line):
                gi = idx + ci
                if gi >= n_vis:
                    break
                if ch == " ":
                    continue
                age = t - self.times[gi]
                p = expo_out(age / 0.09)
                a = alpha * self.dens[gi] * clamp01(age / 0.035 + 0.15)
                paint = skia.Paint(Color4f=col(color, a), AntiAlias=True)
                px = lx + xs[ci]
                py = ly + self.jy[gi] + (1 - p) * self.size * 0.10
                if self.rot[gi] or p < 1:
                    c.save()
                    c.translate(px, py)
                    c.rotate(float(self.rot[gi]))
                    s = 1 + 0.18 * (1 - p)
                    c.scale(s, s)
                    c.drawString(ch, 0, 0, f, paint)
                    c.restore()
                else:
                    c.drawString(ch, px, py, f, paint)
            if n_vis > idx:
                k = min(len(line), n_vis - idx)
                cx_end = lx + (xs[k] if k < len(line) else wline + self.size * 0.08)
                cy_end = ly
            idx += len(line) + 1
        if self.cursor:
            done = n_vis >= len(self.text)
            if not done or t < self.t_end + self.cursor_hold:
                blink = 1.0 if not done else (1.0 if (t * 2.2) % 1 < 0.55 else 0.0)
                if blink:
                    w = self.size * 0.56
                    c.drawRect(skia.Rect.MakeXYWH(cx_end + 2, cy_end - self.size * 0.78, w, self.size * 0.95),
                               skia.Paint(Color4f=col(color, alpha * 0.85), AntiAlias=True))
