"""S14: the ice that feeds the Ganges - meltwater path from the Nanda Devi glacier to the plains."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import skia

from engine.core import W, H, col, paint_fill, paint_stroke, Layer, smooth, smoother, ramp, lerp, glide, expo_inout
from engine.kit import Scene, path_from, partial, glow_dot, glow_stroke, brackets, point_at
from engine.typer import Typer
from engine.sfx import sfx
from engine import fx
from scenes.mapscene import MapCam, region, rivers, graticule, ND

# hand-traced upper course: Nanda Devi glacier -> Rishi Ganga -> Dhauliganga -> Joshimath
RISHI = [(79.94, 30.385), (79.88, 30.40), (79.82, 30.42), (79.77, 30.45), (79.73, 30.49), (79.69, 30.52),
         (79.63, 30.55), (79.57, 30.56)]


@lru_cache(None)
def ganges_path():
    main = max(rivers()["Ganges"], key=len)
    arr = np.array(main)
    jm = np.array([79.565, 30.555])
    i0 = int(np.argmin(np.hypot(*(arr - jm).T)))
    end = int(np.argmin(np.abs(arr[:, 0] - 86.2)))   # stop past Patna, before the delta
    return np.vstack([np.array(RISHI), arr[i0 + 1:end]])


class Ganges(Scene):
    theme = "night"
    name = "ganges"

    def __init__(self, t0, t1):
        super().__init__(t0, t1)
        self.d, self.img = region()
        gp = ganges_path()
        pts = [self.d.px(lo, la) for lo, la in gp]
        self.gx = np.array([p[0] for p in pts])
        self.gy = np.array([p[1] for p in pts])
        self.cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(self.gx), np.diff(self.gy)))])
        self.nd = self.d.px(*ND)
        self.ty_rg = Typer("RISHI GANGA", 61.0, size=20, cps=24, seed=141, gain_db=-7, cursor=False)
        self.ty_g = Typer("GANGES", 64.88, face="sans_bold", size=46, cps=10, seed=142, tracking=0.35, gain_db=-1,
                          cursor=False)
        self.cities = [("HARIDWAR", (78.16, 29.95), 62.4), ("KANPUR", (80.33, 26.45), 63.3),
                       ("VARANASI", (83.01, 25.32), 63.9), ("PATNA", (85.14, 25.61), 64.4)]
        self.city_ty = [Typer(n, tt, size=18, cps=26, seed=150 + i, gain_db=-10, cursor=False)
                        for i, (n, _, tt) in enumerate(self.cities)]
        sfx(t0, "whoosh_soft", gain_db=-8, dur=1.2)
        sfx(60.6, "water", gain_db=-9, dur=5.2)
        sfx(65.5, "whoosh_deep", gain_db=-4, dur=0.9)

    def flow_len(self, t):
        return self.cum[-1] * smoother(ramp(t, 60.55, 64.9)) ** 1.15

    def camera(self, t):
        ndx, ndy = self.nd
        L = self.flow_len(t)
        i = min(int(np.searchsorted(self.cum, L)), len(self.gx) - 1)
        hx, hy = self.gx[i], self.gy[i]
        k = glide(ramp(t, 60.3, 64.6))
        tx = lerp(ndx, (ndx + hx) / 2, k)
        ty = lerp(ndy, (ndy + hy) / 2, k)
        sc = math.exp(lerp(math.log(4.2), math.log(1.0), k))
        r = expo_inout(ramp(t, 65.3, 66.05))
        tx, ty = lerp(tx, ndx, r), lerp(ty, ndy, r)
        sc = math.exp(lerp(math.log(sc), math.log(9.0), r))
        return MapCam(tx, ty, sc, -62, 0, cy=H * 0.56)

    def draw(self, c, t):
        u = self.local(t)
        mc = self.camera(t)
        fin = smooth(ramp(u, 0, 0.4))
        with Layer(c, alpha=fin):
            c.drawRect(skia.Rect.MakeWH(W, H), paint_fill(col("night0")))
            mc.draw_image(c, self.img)
            graticule(c, mc, self.d, 0.7)
            sx, sy = mc.p(self.gx, self.gy)
            full = path_from(list(zip(sx, sy)))
            c.drawPath(full, paint_stroke(col((0.35, 0.5, 0.62), 0.35), 1.4))
            f = self.flow_len(t) / self.cum[-1]
            if f > 0:
                glow_stroke(c, partial(full, f), "ice", 2.6, 0.95, glow=9, glow_alpha=0.85)
                for kk in range(40):
                    fp = (((t - 60.5) * 0.10 + kk / 40) % 1.0) * f
                    if fp <= 0.002:
                        continue
                    (px, py), _ = point_at(full, fp)
                    c.drawCircle(px, py, 2.6, paint_fill(col((0.85, 0.95, 1.0), 0.9)))
            ndx, ndy = mc.p(*self.nd)
            glow_dot(c, float(ndx), float(ndy), 5, "amber", 0.95 + 0.05 * math.sin(t * 5), 7)
            rx, ry = mc.p(*self.d.px(79.78, 30.47))
            self.ty_rg.draw(c, t, float(rx) + 26, float(ry) - 20, "ice", 0.9 * (1 - smooth(ramp(t, 62.8, 63.3))))
            for (name, ll, tt), ty_ in zip(self.cities, self.city_ty):
                if t < tt - 0.05:
                    continue
                cx_, cy_ = mc.p(*self.d.px(*ll))
                a = smooth(ramp(t, tt, tt + 0.2)) * (1 - smooth(ramp(t, 65.2, 65.5)))
                c.drawCircle(float(cx_), float(cy_), 4, paint_fill(col("snow", a)))
                ty_.draw(c, t, float(cx_) + 14, float(cy_) + 6, "snow", 0.85 * a)
            gxp, gyp = mc.p(*self.d.px(82.4, 25.9))
            self.ty_g.draw(c, t, float(gxp), float(gyp) + 90, "ice", 1 - smooth(ramp(t, 65.2, 65.5)), align="center")
            brackets(c, 54, 250, W - 108, H - 560, "steel", 0.3 * fin)

    def overlay(self, img, t):
        if t > 65.4:
            k = smooth(ramp(t, 65.4, 66.0))
            x, y = self.camera(t).p(*self.nd)
            img = fx.radial_blur(img, float(x), float(y), 0.11 * k, 8)
        return img

    def post(self, t):
        return dict(bloom=0.6, bloom_thr=0.5, halation=0.16, vignette=0.34, grain=0.034, lift=0.01)
