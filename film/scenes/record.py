"""S13: In 1978, India's prime minister admitted it in parliament."""
from __future__ import annotations

import math

import numpy as np
import skia

from engine.core import W, H, col, paint_fill, paint_stroke, Layer, expo_out, smooth, smoother, ramp, draw_text
from engine.kit import path_from, smooth_path, partial, label
from engine.typer import Typer
from engine.sfx import sfx
from scenes.paper import PaperScene, INK, PENCIL, RED, ink


class Record1978(PaperScene):
    name = "record"

    def __init__(self, t0, t1):
        super().__init__(t0, t1)
        tw = dict(face="type", size=27, typewriter=True, cursor=False)
        self.ty_h1 = Typer("LOK SABHA DEBATES", 55.75, face="type_b", size=34, cps=18, seed=131, gain_db=-1,
                           typewriter=True, cursor=False)
        self.ty_date = Typer("17 APRIL 1978", 56.35, face="type_b", size=46, cps=12, seed=132, gain_db=0,
                             typewriter=True, cursor=False)
        self.ty_st = Typer("STATEMENT BY THE PRIME MINISTER", 57.15, cps=30, seed=133, gain_db=-6, **tw)
        self.ty_pm = Typer("SHRI MORARJI DESAI", 57.65, face="type_b", size=27, cps=24, seed=134, gain_db=-5,
                           typewriter=True, cursor=False)
        self.ty_re = Typer("Re: Reported planting of nuclear\ndevice by the CIA in the Nanda Devi", 58.15, cps=34,
                           seed=135, gain_db=-7, line_h=40, **tw)
        sfx(t0 - 0.05, "paper_slide", gain_db=-3)
        sfx(58.45, "marker", gain_db=-8, dur=0.5)
        sfx(58.95, "paper_slide", gain_db=-6)
        sfx(59.25, "stamp_soft", gain_db=-6)

    def draw(self, c, t):
        self.sheet(c, t)
        u = self.local(t)
        k = smooth(ramp(u, 0, self.t1 - self.t0))
        c.translate(W / 2, 1100)
        c.scale(1 + 0.05 * k, 1 + 0.05 * k)
        c.translate(-W / 2, -1100)
        with Layer(c, blend=skia.BlendMode.kMultiply):
            x0 = 110
            c.drawLine(x0, 650, W - x0, 650, ink(0.8, 2.4))
            c.drawLine(x0, 658, W - x0, 658, ink(0.5, 1.0))
            self.ty_h1.draw(c, t, x0, 720, INK, 1.0)
            label(c, "PARLIAMENT OF INDIA", W - x0, 720, face="type", size=20, color=PENCIL,
                  alpha=smooth(ramp(t, 55.9, 56.2)), tracking=0.1, align="right")
            self.ty_date.draw(c, t, x0, 820, INK, 1.0)
            c.drawLine(x0, 860, W - x0, 860, ink(0.35, 1.0))
            self.ty_st.draw(c, t, x0, 930, INK, 1.0)
            self.ty_pm.draw(c, t, x0, 976, INK, 1.0)
            self.ty_re.draw(c, t, x0, 1060, INK, 1.0)
            # greeked body text
            ga = smooth(ramp(t, 58.3, 58.9))
            rng = np.random.default_rng(3)
            for i in range(9):
                y = 1170 + i * 40
                L = (W - 2 * x0) * (0.55 + 0.45 * rng.random()) if i % 4 != 3 else (W - 2 * x0) * 0.4
                xx = x0
                while xx < x0 + L * ga:
                    wl = rng.uniform(30, 110)
                    c.drawLine(xx, y, min(xx + wl, x0 + L), y,
                               paint_stroke(col(INK, 0.22), 9, cap=skia.Paint.kButt_Cap))
                    xx += wl + 14
            # red pencil: underline the subject, circle the date
            f = smoother(ramp(t, 58.45, 58.95))
            if f > 0:
                ul = smooth_path([(x0 + 250, 1072), (x0 + 520, 1076), (x0 + 820, 1071)])
                c.drawPath(partial(ul, f), paint_stroke(col(RED, 0.9), 3.4))
                ang = np.linspace(0, 2 * math.pi * 1.08, 70)
                cx_, cy_ = x0 + 175, 805
                ring = path_from([(cx_ + math.cos(a) * (210 + 6 * math.sin(a * 3)), cy_ + math.sin(a) * 58) for a in ang])
                c.drawPath(partial(ring, smoother(ramp(t, 58.6, 59.1))), paint_stroke(col(RED, 0.85), 3.0))
        self.draw_clip(c, t)

    def draw_clip(self, c, t):
        e = expo_out(ramp(t, 58.95, 59.45))
        if e <= 0:
            return
        x, y = W / 2 + 40, 1440 + (1 - e) * 700
        c.save()
        c.translate(x, y)
        c.rotate(-4 + 2 * (1 - e))
        w, h = 820, 330
        with Layer(c, alpha=0.4 * e, blur=18):
            c.drawRect(skia.Rect.MakeXYWH(-w / 2 + 10, -h / 2 + 16, w, h), paint_fill(col((0, 0, 0))))
        c.drawRect(skia.Rect.MakeXYWH(-w / 2, -h / 2, w, h), paint_fill(col((0.93, 0.91, 0.86))))
        with Layer(c, blend=skia.BlendMode.kMultiply):
            draw_text(c, "THE NEW YORK TIMES  \u00b7  APRIL 18, 1978", -w / 2 + 34, -h / 2 + 46, "news_r", 20,
                      paint_fill(col(INK, 0.7)), 0.08)
            c.drawLine(-w / 2 + 34, -h / 2 + 62, w / 2 - 34, -h / 2 + 62, ink(0.6, 1.2))
            draw_text(c, "Desai Says U.S.-Indian Team", -w / 2 + 34, -h / 2 + 128, "news", 50, paint_fill(col(INK)), -0.01)
            draw_text(c, "Lost Atomic Spy Gear", -w / 2 + 34, -h / 2 + 186, "news", 50, paint_fill(col(INK)), -0.01)
            rng = np.random.default_rng(9)
            for ci in range(3):
                cx0 = -w / 2 + 34 + ci * 255
                for i in range(4):
                    yy = -h / 2 + 228 + i * 22
                    c.drawLine(cx0, yy, cx0 + rng.uniform(150, 230), yy,
                               paint_stroke(col(INK, 0.25), 7, cap=skia.Paint.kButt_Cap))
        c.restore()
