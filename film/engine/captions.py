"""Kinetic captions locked to Kokoro word timestamps.

Spec grammar per group: words in reading order; *word* = key (heavy sans), plain = lead
(serif italic); optional {accent} suffix on a key word, e.g. *China*{red}.
"""
from __future__ import annotations

import json
import re

import numpy as np
import skia

from .core import (ROOT, W, col, mix, draw_text, text_width, expo_out, smooth, clamp01, Layer, PAL)
from .sfx import sfx

KEY_FACE, KEY_SIZE = "sans_black", 84
LEAD_FACE, LEAD_SIZE = "serif_i", 60
MAX_W = 930
GAP = 0.22  # space as fraction of size

THEMES = {
    "night": {"lead": (0.64, 0.69, 0.74), "key": (0.95, 0.96, 0.97), "shadow": (0.0, 0.0, 0.0)},
    "paper": {"lead": (0.43, 0.41, 0.38), "key": (0.08, 0.085, 0.095), "shadow": (0.93, 0.90, 0.84)},
}
ACCENTS = {"red": PAL["red"], "amber": PAL["amber"], "ice": PAL["ice"]}


def _norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


class Word:
    def __init__(self, text, key, accent, t0, t1):
        self.text, self.key, self.accent, self.t0, self.t1 = text, key, accent, t0, t1
        self.face = KEY_FACE if key else LEAD_FACE
        self.size = KEY_SIZE if key else LEAD_SIZE
        self.track = -0.025 if key else 0.0
        self.w = text_width(self.face, self.size, text, self.track)


class Group:
    def __init__(self, words, t_in):
        self.words, self.t_in = words, t_in
        self.t_out = None
        self._layout()

    def _layout(self):
        rows, cur, cur_w, prev_key = [], [], 0.0, None
        for w in self.words:
            gap = (GAP * (w.size if cur else 0))
            # break before the first key word when lead text precedes it (reference-style hierarchy)
            brk = cur and ((w.key and prev_key is False and not any(x.key for x in cur)) or cur_w + gap + w.w > MAX_W)
            if brk:
                rows.append(cur)
                cur, cur_w = [], 0.0
                gap = 0
            cur.append(w)
            cur_w += gap + w.w
            prev_key = w.key
        if cur:
            rows.append(cur)
        self.rows = []
        y = 0.0
        for r in rows:
            big = any(w.key for w in r)
            asc = (KEY_SIZE if big else LEAD_SIZE) * 0.80
            y += asc
            tot = sum(w.w for w in r) + sum(GAP * w.size for w in r[1:])
            x = (W - tot) / 2
            pos = []
            for i, w in enumerate(r):
                if i:
                    x += GAP * w.size
                pos.append((w, x, y))
                x += w.w
            self.rows.append(pos)
            y += (KEY_SIZE if big else LEAD_SIZE) * 0.34
        self.height = y


class Captions:
    def __init__(self, spec: dict, timing_path=None, hold=1.1):
        tl = json.loads((timing_path or (ROOT / "vo" / "vo_timing.json")).read_text())
        self.groups: list[Group] = []
        for line in tl["lines"]:
            vo_words = [w for w in line["words"] if _norm(w["w"])]
            gspecs = spec.get(line["id"])
            if not gspecs:
                continue
            k = 0
            rest = None  # unconsumed remainder of a multi-word VO token ('Lop Nur')
            line_groups = []
            for gs in gspecs:
                words = []
                for tok in gs.split():
                    acc = None
                    m = re.search(r"\{(\w+)\}$", tok)
                    if m:
                        acc = m.group(1)
                        tok = tok[:m.start()]
                    key = tok.startswith("*")
                    txt = tok.replace("*", "")
                    target = _norm(txt)
                    while k < len(vo_words):
                        cur = rest if rest is not None else _norm(vo_words[k]["w"])
                        if cur.startswith(target):
                            break
                        k += 1
                        rest = None
                    if k >= len(vo_words):
                        raise ValueError(f"caption word '{txt}' not found in {line['id']}")
                    vw = vo_words[k]
                    cur = rest if rest is not None else _norm(vw["w"])
                    rest = cur[len(target):] or None
                    if rest is None:
                        k += 1
                    words.append(Word(txt, key, acc, vw["t0"], vw["t1"]))
                g = Group(words, words[0].t0)
                line_groups.append(g)
            for a, b in zip(line_groups, line_groups[1:]):
                a.t_out = b.t_in - 0.24
            line_groups[-1].t_out = line["start"] + line["dur"] + hold
            self.groups.extend(line_groups)
        # a group may not outlive the next line's first group
        for a, b in zip(self.groups, self.groups[1:]):
            a.t_out = min(a.t_out, b.t_in - 0.24)
        for g in self.groups:
            n_key = sum(w.key for w in g.words)
            sfx(g.t_in - 0.035, "cream", gain_db=-1.0 if n_key else -4.0, weight=min(1.0, 0.4 + 0.2 * n_key),
                seed=int(g.t_in * 1000))

    def draw(self, c: skia.Canvas, t, y_top=330, theme=0.0, alpha=1.0, accent_ok=True):
        """theme: 0 = night palette, 1 = paper palette (blendable)."""
        th = {k: mix(THEMES["night"][k], THEMES["paper"][k], theme) for k in ("lead", "key", "shadow")}
        for g in self.groups:
            if t < g.t_in - 0.1 or t > g.t_out + 0.35:
                continue
            out = clamp01((t - g.t_out) / 0.2)
            if out >= 1:
                continue
            for row in g.rows:
                for w, x, y in row:
                    t_on = w.t0 - 0.07
                    if t < t_on:
                        continue
                    p = expo_out((t - t_on) / 0.42)
                    a = alpha * smooth((t - t_on) / 0.16) * (1 - smooth(out))
                    if a <= 0.003:
                        continue
                    blur = 7.0 * (1 - p) + 6.0 * out
                    dy = 18 * (1 - p) - 16 * expo_out(out)
                    s = 1 + (0.05 if w.key else 0.02) * (1 - p)
                    color = th["key"] if w.key else th["lead"]
                    if w.accent and accent_ok:
                        color = mix(color, ACCENTS[w.accent], 0.9)
                    yy = y_top + y + dy
                    cx = x + w.w / 2
                    # legibility shadow (soft, wide)
                    pad = 70
                    bounds = skia.Rect.MakeLTRB(cx - w.w * s / 2 - pad, yy - w.size - pad, cx + w.w * s / 2 + pad,
                                                yy + w.size * 0.4 + pad)
                    sh_a = a * 0.55 * (1 - theme)
                    if sh_a > 0.01:
                      with Layer(c, alpha=sh_a, blur=16, bounds=bounds):
                        c.save()
                        c.translate(cx, yy)
                        c.scale(s, s)
                        draw_text(c, w.text, -w.w / 2, 0, w.face, w.size, skia.Paint(Color4f=col(th["shadow"]), AntiAlias=True), w.track)
                        c.restore()
                    with Layer(c, alpha=a, blur=blur if blur > 0.3 else 0.0, bounds=bounds):
                        c.save()
                        c.translate(cx, yy)
                        c.scale(s, s)
                        draw_text(c, w.text, -w.w / 2, 0, w.face, w.size, skia.Paint(Color4f=col(color), AntiAlias=True), w.track)
                        c.restore()
