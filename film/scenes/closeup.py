"""S10-S11a crevice close-up (tied down / spring / gone) and S12 search grid."""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np
import skia

from engine.core import (W, H, PAL, col, mix, paint_fill, paint_stroke, Layer, expo_out, expo_inout, smooth,
                         smoother, clamp01, ramp, lerp, glide, fbm, fbm_np, hash01, spring, draw_text, text_width,
                         image_from_float)
from engine.kit import (Scene, CACHE, SAMP, path_from, smooth_path, partial, dashed, label, glow_dot, glow_stroke,
                        brackets, point_at)
from engine.typer import Typer
from engine.sfx import sfx
from engine import terrain as T
from engine import fx
from scenes.ridge import Snow, route, camps, frame
from scenes.mapscene import MapCam

ND = (79.9708, 30.3758)


class SplitFlap:
    """Mechanical date display: each character flips through to its target."""

    def __init__(self, a, b, t0, x, y, size=58, gain_db=-6):
        self.a, self.b, self.t0, self.x, self.y, self.size = a, b, t0, x, y, size
        self.cw = size * 0.78
        self.flips = []
        alpha = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        for i, (ca, cb) in enumerate(zip(a, b)):
            if ca == cb:
                self.flips.append([])
                continue
            ia, ib = alpha.index(ca), alpha.index(cb)
            n = (ib - ia) % len(alpha)
            n = min(n, 7) if n > 7 else n
            seq = [alpha[(ia + k) % len(alpha)] for k in range(1, n)] + [cb]
            times = [t0 + i * 0.06 + k * 0.075 for k in range(len(seq))]
            self.flips.append(list(zip(times, seq)))
            for tt in times:
                sfx(tt, "flap", gain_db=gain_db - 3 + 2 * (tt == times[-1]), seed=int(tt * 1000) + i, pan=(i - 4) * 0.06)

    def char_at(self, i, t):
        ch = self.a[i]
        for tt, c in self.flips[i]:
            if t >= tt:
                ch = c
        return ch

    def draw(self, c, t, alpha=1.0):
        n = len(self.a)
        x0 = self.x - n * (self.cw + 6) / 2
        for i in range(n):
            x = x0 + i * (self.cw + 6)
            if self.a[i] == " ":
                continue
            r = skia.Rect.MakeXYWH(x, self.y - self.size * 0.9, self.cw, self.size * 1.2)
            c.drawRoundRect(r, 6, 6, paint_fill(col((0.07, 0.08, 0.095), 0.92 * alpha)))
            c.drawRoundRect(r, 6, 6, paint_stroke(col("steel", 0.35 * alpha), 1.0))
            ch = self.char_at(i, t)
            # flip motion: squash right after each flip
            sq = 1.0
            for tt, _ in self.flips[i]:
                if 0 <= t - tt < 0.07:
                    sq = 0.35 + 0.65 * ((t - tt) / 0.07)
            c.save()
            c.translate(x + self.cw / 2, self.y - self.size * 0.3)
            c.scale(1, sq)
            draw_text(c, ch, 0, self.size * 0.36, "mono_b", self.size, paint_fill(col("snow", alpha)), 0, "center")
            c.restore()
            c.drawLine(x, self.y - self.size * 0.3, x + self.cw, self.y - self.size * 0.3,
                       paint_stroke(col((0, 0, 0), 0.8 * alpha), 2))


class Crevice(Scene):
    theme = "night"
    name = "crevice"

    def __init__(self, t0, t1):
        super().__init__(t0, t1)
        self.snow = Snow(n=260, seed=21)
        rng = np.random.default_rng(11)
        # rock massif made of faceted blocks (right) and the crack running diagonally
        self.blocks = []
        for k in range(28):
            cx, cy = rng.uniform(80, 1100), rng.uniform(900, 1900)
            r = rng.uniform(140, 320)
            nv = int(rng.integers(5, 8))
            ang0 = rng.uniform(0, 6.28)
            pts = [(cx + math.cos(ang0 + j * 2 * math.pi / nv) * r * rng.uniform(0.7, 1.15),
                    cy + math.sin(ang0 + j * 2 * math.pi / nv) * r * rng.uniform(0.6, 1.0)) for j in range(nv)]
            self.blocks.append((pts, rng.uniform(0, 1)))
        self.crack = [(420, 1010), (520, 1090), (575, 1180), (640, 1260), (690, 1380), (760, 1480)]
        self.pkg = (590, 1150)
        self.flap = SplitFlap("OCT 1965", "MAY 1966", 48.0, W / 2, 1520, size=60)
        self.ty_last = Typer("LAST KNOWN POSITION", 49.85, size=21, cps=26, seed=111, gain_db=-5)
        self.ty_tied = Typer("CACHED NEAR CAMP IV", 45.0, size=21, cps=26, seed=112, gain_db=-6)
        sfx(44.78, "rope", gain_db=-6, dur=1.0)
        sfx(45.80, "knot", gain_db=-5)
        for k in range(6):
            sfx(46.55 + k * 0.17, "step_snow", gain_db=-10 - k * 1.2, seed=k, pan=-0.2 - k * 0.05)
        sfx(47.9, "time_whoosh", gain_db=-6, dur=1.0)
        sfx(49.70, "absence", gain_db=-5, dur=1.6)

    def light(self, t):
        return smooth(ramp(t, 48.0, 48.9))   # 0 = storm night, 1 = cold spring dawn

    def draw(self, c, t):
        u = self.local(t)
        L = self.light(t)
        k = smooth(ramp(u, 0, self.t1 - self.t0))
        # camera: slow push toward the package
        s = 1.75 + 0.18 * k
        c.translate(W / 2, 1120)
        c.scale(s, s)
        c.translate(-self.pkg[0], -self.pkg[1] + 20 * k)
        sky = mix((0.03, 0.04, 0.055), (0.20, 0.25, 0.31), L)
        sky2 = mix((0.07, 0.09, 0.11), (0.42, 0.47, 0.52), L)
        p = skia.Paint()
        p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(0, H)],
                    [col(sky).toColor(), col(sky2).toColor()]))
        c.drawRect(skia.Rect.MakeLTRB(-400, -400, W + 400, H + 400), p)
        # rock blocks
        for pts, v in self.blocks:
            path = path_from(pts, close=True)
            tone = mix((0.05, 0.06, 0.075), (0.16, 0.18, 0.21), v * 0.6 + 0.3 * L)
            c.drawPath(path, paint_fill(col(tone)))
            c.save()
            c.clipPath(path, skia.ClipOp.kIntersect, True)
            for j in range(-20, 20):
                off = j * 16
                c.drawLine(pts[0][0] - 400, pts[0][1] + off - 140, pts[0][0] + 400, pts[0][1] + off + 140,
                           paint_stroke(col((0.35, 0.40, 0.46), 0.10 + 0.05 * v), 1.0))
            c.restore()
            # snow caps on upper edges
            top = sorted(pts, key=lambda q: q[1])[:2]
            c.drawLine(top[0][0], top[0][1], top[1][0], top[1][1], paint_stroke(col("snow", 0.55 + 0.3 * L), 5))
        # crack (deep shadow wedge)
        cr = self.crack
        left = [(x - 34 + 8 * math.sin(i), y) for i, (x, y) in enumerate(cr)]
        right = [(x + 34 + 6 * math.cos(i), y + 6) for i, (x, y) in enumerate(cr)][::-1]
        crack = path_from(left + right, close=True)
        c.drawPath(crack, paint_fill(col((0.005, 0.007, 0.01))))
        c.drawPath(path_from(left), paint_stroke(col("snow", 0.5 + 0.3 * L), 2.0))
        gone = smooth(ramp(t, 49.55, 49.95))
        px, py = self.pkg
        if gone < 1:
            with Layer(c, alpha=1 - gone):
                self.draw_package(c, t, px, py)
                self.draw_rope(c, t, px, py)
        if gone > 0:
            # empty crack: ghost outline where it was
            r = skia.Rect.MakeXYWH(px - 80, py - 62, 160, 124)
            c.drawRoundRect(r, 8, 8, dashed(paint_stroke(col("amber", 0.9 * gone), 1.8), 7, 7, -t * 20))
            self.ty_last.draw(c, t, px + 100, py - 70, "amber", gone)
        # footprints leaving, downhill to the left
        fa = 1 - 0.85 * L
        for j in range(6):
            tt = 46.55 + j * 0.17
            if t < tt:
                break
            a = smooth(ramp(t, tt, tt + 0.1)) * fa
            fx_, fy_ = px - 120 - j * 70, py + 120 + j * 62
            for side in (-1, 1):
                c.save()
                c.translate(fx_ + side * 12, fy_ + side * 6)
                c.rotate(-40)
                c.drawOval(skia.Rect.MakeXYWH(-7, -13, 14, 26), paint_fill(col((0.02, 0.03, 0.04), 0.7 * a)))
                c.restore()
        # snow (storm fades into spring)
        st = (1 - L) * (1 - smooth(ramp(t, 46.8, 47.6)) * 0.5)
        self.snow.draw(c, t, 0.8 * st, wind=(-1.0, 0.55), t_start=44.0)
        if L < 1:
            c.drawRect(skia.Rect.MakeLTRB(-400, -400, W + 400, H + 400),
                       paint_fill(col((0.6, 0.66, 0.72), 0.18 * (1 - L) * (1 - smooth(ramp(t, 44.6, 45.4))) + 0.06 * (1 - L))))

    def draw_package(self, c, t, px, py):
        steel = col((0.68, 0.74, 0.80))
        # two radios + generator, lashed together
        for k, (dx, dy) in enumerate(((-60, -30), (-60, 18))):
            r = skia.Rect.MakeXYWH(px + dx - 20, py + dy - 22, 84, 44)
            c.drawRect(r, paint_fill(col((0.10, 0.12, 0.14))))
            c.drawRect(r, paint_stroke(steel, 1.6))
            for j in range(3):
                c.drawCircle(px + dx + j * 20, py + dy, 5, paint_stroke(steel, 1.2))
        gx, gy = px + 58, py - 4
        c.drawRect(skia.Rect.MakeXYWH(gx - 26, gy - 46, 52, 92), paint_fill(col((0.09, 0.10, 0.11))))
        for j in range(5):
            c.drawLine(gx - 26 + j * 13, gy - 46, gx - 26 + j * 13, gy + 46, paint_stroke(steel, 1.2))
        c.drawRect(skia.Rect.MakeXYWH(gx - 26, gy - 46, 52, 92), paint_stroke(steel, 1.6))
        beat = 0.8 + 0.2 * math.sin(t * 4.8)
        glow_dot(c, gx, gy, 6, "amber", 0.9 * beat, 8)
        # folded antenna strapped on top
        c.drawLine(px - 90, py - 58, px + 70, py - 66, paint_stroke(steel, 2.4))
        for j in range(5):
            xx = px - 80 + j * 34
            c.drawLine(xx, py - 72, xx + 4, py - 50, paint_stroke(steel, 1.6))

    def draw_rope(self, c, t, px, py):
        f = smoother(ramp(t, 44.75, 45.75))
        if f <= 0:
            return
        pit = [(px - 150, py - 110), (px + 150, py - 120), (px + 130, py + 110)]
        rope_c = col((0.88, 0.55, 0.30))
        for i, (ax, ay) in enumerate(pit):
            # piton
            c.drawPath(path_from([(ax - 8, ay - 4), (ax + 8, ay - 4), (ax, ay + 18)], close=True),
                       paint_fill(col("steel")))
            c.drawCircle(ax, ay - 6, 5, paint_stroke(col("steel"), 2))
        tight = spring(max(0.0, t - 45.78), 0.4, 18) if t > 45.78 else 0.0
        sag = 60 * (1 - tight) + 6
        segs = [((pit[0]), (px + 60, py + 55)), ((pit[1]), (px - 80, py + 50)), ((pit[2]), (px - 70, py - 60))]
        for i, (a, b) in enumerate(segs):
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + sag
            pth = smooth_path([a, (mx, my), b])
            ff = clamp01(f * 3 - i)
            c.drawPath(partial(pth, ff), paint_stroke(rope_c, 4.0))
            c.drawPath(partial(pth, ff), dashed(paint_stroke(col((0.5, 0.28, 0.12), 0.8), 4.0, cap=skia.Paint.kButt_Cap), 3, 5))
        if t > 45.78:
            k = expo_out(ramp(t, 45.78, 46.0))
            c.drawCircle(px + 4, py + 6, 9 * k, paint_fill(rope_c))
        pass

    def overlay(self, img, t):
        return img

    def post(self, t):
        L = self.light(t)
        return dict(bloom=0.5, bloom_thr=0.6, halation=0.14, vignette=0.4 - 0.1 * L, grain=0.038,
                    lift=0.012 + 0.02 * L, contrast=1.02)

    def draw_after(self, c, t):
        pass


class CreviceHUD(Scene):
    """split-flap date drawn unscaled over the crevice camera move"""
    theme = "night"
    name = "crevice_hud"

    def __init__(self, crevice):
        super().__init__(crevice.t0, crevice.t1)
        self.cv = crevice

    def draw(self, c, t):
        a = smooth(ramp(t, 47.75, 47.95)) * (1 - smooth(ramp(t, 49.4, 49.7)))
        if a > 0:
            self.cv.flap.draw(c, t, a)

    def post(self, t):
        return self.cv.post(t)


@lru_cache(None)
def nanda_relief():
    d = T.dem("nanda_z13")
    p = CACHE / "relief_nanda_z13.png"
    if p.exists():
        rel = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32) / 65535.0
    else:
        rel = T.relief_night(d, zf=1.3)
        cv2.imwrite(str(p), (rel[..., ::-1] * 65535).astype(np.uint16))
    return d, image_from_float(rel)


class Search(Scene):
    """S12: They searched for years with radiation detectors. Nothing."""
    theme = "night"
    name = "search"

    def __init__(self, t0, t1):
        super().__init__(t0, t1)
        self.d, self.img = nanda_relief()
        pts, z = route()
        c4 = pts[camps()[-1][1]]
        fr = frame()
        lon, lat = fr.to_lonlat(c4[0], c4[1])
        self.c4px = self.d.px(float(lon), float(lat))
        # search grid laid downslope (south-west) of Camp IV
        self.cells = []
        mpp = self.d.m_per_px(ND[1])
        cell = 820 / mpp
        ox, oy = self.c4px
        for i in range(-1, 4):
            for j in range(-2, 3):
                cx = ox - i * cell * 0.62 + j * cell * 0.78
                cy = oy + i * cell * 0.78 + j * cell * 0.62
                self.cells.append((cx, cy, cell))
        rng = np.random.default_rng(4)
        order = np.argsort([math.hypot(cx - ox, cy - oy) + rng.uniform(0, cell) for cx, cy, _ in self.cells])
        self.order = {int(k): r for r, k in enumerate(order)}
        self.t_nothing = 54.53
        self.ty_det = Typer("NEUTRON DETECTOR", 52.10, size=21, cps=24, seed=121, gain_db=-5)
        self.ty_y = [Typer(y, 52.45 + 0.42 * k, face="mono_b", size=30, cps=10, seed=122 + k, tracking=0.1, gain_db=-3,
                           cursor=False) for k, y in enumerate(("1966", "1967", "1968"))]
        self.ty_no = Typer("NO TRACE", 54.6, face="mono_b", size=64, cps=12, seed=126, tracking=0.3, gain_db=-1,
                           cursor=False)
        rng = np.random.default_rng(8)
        for k in range(9):
            sfx(52.0 + rng.uniform(0, 2.4), "geiger", gain_db=-14, seed=k, pan=rng.uniform(-0.4, 0.4))
        sfx(52.0, "sonar", gain_db=-11, dur=2.4)
        sfx(self.t_nothing, "nothing", gain_db=-3)

    def cam(self, t):
        u = self.local(t)
        k = smooth(ramp(u, 0, self.t1 - self.t0))
        ox, oy = self.c4px
        return MapCam(ox - 40, oy + 90, 1.05 + 0.15 * k, 215 + 10 * k, tilt=22, cy=H * 0.55)

    def draw(self, c, t):
        u = self.local(t)
        mc = self.cam(t)
        fin = smooth(ramp(u, 0, 0.30))
        with Layer(c, alpha=fin):
            c.drawRect(skia.Rect.MakeWH(W, H), paint_fill(col("night0")))
            mc.draw_image(c, self.img)
            # grid cells
            n = len(self.cells)
            prog = ramp(t, 52.0, 54.4) * n
            nothing = smooth(ramp(t, self.t_nothing, self.t_nothing + 0.25))
            for idx, (cx, cy, cs) in enumerate(self.cells):
                r = self.order[idx]
                if r > prog + 1:
                    continue
                age = prog - r
                q = cs / 2 * 0.92
                ang = math.radians(38)
                corners = []
                for sx_, sy_ in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    dx, dy = sx_ * q, sy_ * q
                    corners.append((cx + dx * math.cos(ang) - dy * math.sin(ang), cy + dx * math.sin(ang) + dy * math.cos(ang)))
                X, Y = zip(*corners)
                px, py = mc.p(np.array(X), np.array(Y))
                pth = path_from(list(zip(px, py)), close=True)
                flash = math.exp(-max(age, 0) * 2.2)
                base = mix("ice", "red", nothing)
                c.drawPath(pth, paint_fill(col(base, 0.10 * flash + 0.05 + 0.10 * nothing * (1 - smooth(ramp(t, 54.8, 55.4))))))
                c.drawPath(pth, paint_stroke(col(base, 0.35 + 0.5 * flash), 1.4))
                if age > 0.3:
                    mx, my = px.mean(), py.mean()
                    draw_text(c, "0", float(mx), float(my) + 7, "mono", 18, paint_fill(col(base, 0.6)), 0, "center")
            # sweep from Camp IV
            sx, sy = mc.p(*self.c4px)
            sx, sy = float(sx), float(sy)
            ang = (t - 52.0) * 2.6
            if t < self.t_nothing:
                p = skia.Paint(AntiAlias=True)
                p.setShader(skia.GradientShader.MakeSweep(sx, sy, [col("ice", 0.0).toColor(), col("ice", 0.0).toColor(),
                                                                  col("ice", 0.28).toColor()], [0.0, 0.82, 1.0]))
                c.save()
                c.translate(sx, sy)
                c.rotate(math.degrees(ang))
                c.translate(-sx, -sy)
                c.drawCircle(sx, sy, 560, p)
                c.restore()
            c.drawCircle(sx, sy, 7, paint_stroke(col("amber"), 1.8))
            label(c, "C-IV", sx + 14, sy - 12, face="mono_b", size=19, color="snow", alpha=0.9)
            # readout panel
            pa = smooth(ramp(t, 52.0, 52.3))
            x0, y0 = 92, 1380
            self.ty_det.draw(c, t, x0, y0, "steel", pa)
            cnt = "0000"
            draw_text(c, f"COUNT  {cnt}", x0, y0 + 40, "mono_b", 30, paint_fill(col(mix("snow", "red", nothing), pa)), 0.05)
            for k, ty_ in enumerate(self.ty_y):
                ty_.draw(c, t, W - 92 - (2 - k) * 118, y0 + 40, "snow", pa, align="right")
            na = smooth(ramp(t, 54.55, 54.75))
            if na > 0:
                c.drawRect(skia.Rect.MakeXYWH(W / 2 - 250, 1090 - 62, 500, 88), paint_fill(col((0.02, 0.025, 0.03), 0.75 * na)))
                c.drawRect(skia.Rect.MakeXYWH(W / 2 - 250, 1090 - 62, 500, 88), paint_stroke(col("red", 0.9 * na), 2))
            self.ty_no.draw(c, t, W / 2 + 10, 1090, "red", 1.0, align="center")
            brackets(c, 54, 250, W - 108, H - 560, "steel", 0.3 * fin)
        # the avalanche dust clears to reveal the search
        dust = 1 - smooth(ramp(u, 0.1, 0.75))
        if dust > 0:
            c.drawRect(skia.Rect.MakeWH(W, H), paint_fill(col((0.70, 0.75, 0.80), 0.8 * dust)))

    def post(self, t):
        return dict(bloom=0.5, bloom_thr=0.58, halation=0.12, vignette=0.36, grain=0.036, lift=0.012)

    def overlay(self, img, t):
        if t < self.t0 + 0.4:
            k = 1 - smooth(ramp(t, self.t0, self.t0 + 0.4))
            img = fx.radial_blur(img, W / 2, H / 2, 0.08 * k, 6)
        return img
