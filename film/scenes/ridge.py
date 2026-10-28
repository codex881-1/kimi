"""Night ridge-line terrain of the Nanda Devi massif (DEM z12, real heights).

S6  'chose Nanda Devi'      - scan-in build, agencies, altitude ruler
S9  'carried it up'          - route + camps, then the blizzard
S11 'swept down the mountain'- slope collapse"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import skia

from engine.core import (W, H, PAL, col, mix, paint_fill, paint_stroke, Layer, expo_out, expo_inout, smooth,
                         smoother, clamp01, ramp, lerp, draw_text, glide, silk, fbm, fbm_np, hash01, cubic_inout)
from engine.kit import (Scene, path_from, smooth_path, partial, dashed, label, leader, glow_dot, glow_stroke,
                        brackets, point_at)
from engine.typer import Typer
from engine.sfx import sfx
from engine import terrain as T
from engine import fx

ND = (79.9708, 30.3758)
SUMMIT_Z = 7816.0


@lru_cache(None)
def frame():
    return T.LocalFrame(T.dem("nanda_z12"), *ND)


@lru_cache(None)
def route():
    """Schematic line of ascent from the south-west (Sanctuary side) to the summit.
    The exact 1965 line is not documented; this follows the south-west approach."""
    fr = frame()
    ctrl = np.array([(-4700.0, -5900.0), (-3700.0, -4500.0), (-2500.0, -3000.0), (-1500.0, -1750.0),
                     (-700.0, -800.0), (-200.0, -220.0), (0.0, 0.0)])
    pts = []
    for i in range(len(ctrl) - 1):
        p0 = ctrl[max(i - 1, 0)]
        p1, p2 = ctrl[i], ctrl[i + 1]
        p3 = ctrl[min(i + 2, len(ctrl) - 1)]
        for u in np.linspace(0, 1, 60, endpoint=False):
            u2, u3 = u * u, u * u * u
            pts.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * u3))
    pts.append(ctrl[-1])
    pts = np.array(pts)
    z = fr.height(pts[:, 0], pts[:, 1]).astype(np.float64)
    return pts, z


def camps():
    pts, z = route()
    out = []
    for name, target in (("BASE", None), ("C-I", 5600), ("C-II", 6250), ("C-III", 6900), ("C-IV", SUMMIT_Z - 300)):
        i = 0 if target is None else int(np.argmax(z >= target))
        out.append((name, i))
    return out


class RidgeView:
    def __init__(self):
        self.fr = frame()

    def camera(self, dist, yaw, alt, pitch, fov, look=(0.0, 0.0), cy=H * 0.60, look_z=None):
        pos = (look[0] - dist * math.sin(math.radians(yaw)), look[1] - dist * math.cos(math.radians(yaw)), alt)
        if look_z is not None:
            pitch = -math.degrees(math.atan2(alt - look_z, dist))
        return T.RidgeCamera(pos, yaw, pitch, fov, W, H, cy=cy)

    def draw(self, c, cam, yaw, reveal=1.0, alpha=1.0, zmod=None, jitter=0.0, t=0.0, d_far=34000, n=190,
             snow_tone=1.0, line_w=1.0):
        lines = T.ridge_lines(self.fr, cam, yaw, n_slices=n, d_near=900, d_far=d_far, half_width=24000,
                              samples=1100, x_step=2.0, zmod=zmod)
        if not lines:
            return
        dmax = max(l[5] for l in lines)
        for dist_, xs, ys, zs, vis, along in reversed(lines):
            rel = along / dmax
            if rel > reveal:
                continue
            front = clamp01(1 - (reveal - rel) / 0.06) if reveal < 1 else 0.0
            fade = clamp01(1 - rel) ** 0.8
            idx = np.flatnonzero(vis)
            if len(idx) < 2:
                continue
            splits = np.flatnonzero(np.diff(idx) > 1) + 1
            yj = ys
            if jitter > 0:
                yj = ys + jitter * fbm_np(xs * 0.01 + along * 0.001 + t * 6.0, 2.2)
            for run in np.split(idx, splits):
                if len(run) < 2:
                    continue
                zmean = float(zs[run].mean())
                snow = clamp01((zmean - 4300) / 2600) * snow_tone
                tone = mix((0.36, 0.44, 0.52), (0.92, 0.95, 0.97), snow)
                a = alpha * (0.22 + 0.78 * fade) * (0.55 + 0.45 * snow)
                w = line_w * (0.8 + 1.1 * fade)
                if front > 0:
                    tone = mix(tone, (0.75, 0.92, 1.0), front)
                    a = min(1.0, a + 0.6 * front)
                    w += 1.2 * front
                pth = path_from(list(zip(xs[run], yj[run])))
                c.drawPath(pth, paint_stroke(col(tone, a), w))

    def project(self, cam, e, n, z):
        sx, sy, dz = cam.project(np.asarray(e), np.asarray(n), np.asarray(z))
        return sx, sy


def sky(c, top=(0.010, 0.014, 0.020), mid=(0.025, 0.035, 0.048), low=(0.05, 0.066, 0.085)):
    p = skia.Paint()
    p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(0, H)],
                [col(top).toColor(), col(mid).toColor(), col(low).toColor()], [0.0, 0.5, 1.0]))
    c.drawRect(skia.Rect.MakeWH(W, H), p)


class Stars:
    def __init__(self, n=220, seed=9):
        r = np.random.default_rng(seed)
        self.s = [(r.uniform(0, W), r.uniform(0, H * 0.55), r.uniform(0.4, 1.3), r.uniform(0, 6.28)) for _ in range(n)]

    def draw(self, c, t, a):
        for x, y, rr, ph in self.s:
            c.drawCircle(x, y, rr, paint_fill(col("snow", a * (0.3 + 0.25 * math.sin(t * 1.1 + ph)))))


class NandaIntro(Scene):
    """S6: The CIA and Indian intelligence chose Nanda Devi. Nearly eight kilometers up."""
    theme = "night"
    name = "ridge_intro"

    def __init__(self, t0, t1):
        super().__init__(t0, t1)
        self.rv = RidgeView()
        self.stars = Stars()
        self.ty_cia = Typer("CIA", 23.86, face="mono_b", size=26, cps=10, seed=61, tracking=0.25, gain_db=-2, pan=-0.3,
                            cursor=False)
        self.ty_ib = Typer("INTELLIGENCE BUREAU \u00b7 INDIA", 24.74, face="mono_b", size=22, cps=26, seed=62,
                           tracking=0.12, gain_db=-3, pan=0.3, cursor=False)
        self.ty_nd = Typer("NANDA DEVI", 25.95, face="sans_bold", size=40, cps=13, seed=63, tracking=0.2, gain_db=-1,
                           cursor=False)
        self.ty_co = Typer("30.376\u00b0N  79.971\u00b0E", 26.6, size=20, cps=26, seed=64, gain_db=-7)
        sfx(23.5, "scan", gain_db=-7, dur=1.4)
        sfx(25.9, "lock", gain_db=-8)
        for k in range(12):
            sfx(27.25 + k * 0.095, "tick", gain_db=-14 + k * 0.4, seed=k)

    def cam(self, t):
        u = self.local(t)
        k = smooth(ramp(u, 0, self.t1 - self.t0))
        yaw = 24 + 9 * k
        return self.rv.camera(lerp(21000, 16500, k), yaw, lerp(9800, 9300, k), 0, 44, look=(-600.0, -900.0),
                              cy=H * 0.60, look_z=6200), yaw

    def draw(self, c, t):
        u = self.local(t)
        cam, yaw = self.cam(t)
        fin = smooth(ramp(u, 0.0, 0.35))
        with Layer(c, alpha=fin):
            sky(c)
            self.stars.draw(c, t, 0.8)
            reveal = smoother(ramp(t, 23.55, 24.9))
            self.rv.draw(c, cam, yaw, reveal=0.02 + 0.98 * reveal, t=t)
            sx, sy = self.rv.project(cam, 0.0, 0.0, SUMMIT_Z)
            sx, sy = float(sx), float(sy)
            # summit lock-on
            la = smooth(ramp(t, 25.75, 26.1))
            if la > 0:
                r = 26 + 30 * (1 - expo_out(ramp(t, 25.75, 26.3)))
                p = paint_stroke(col("snow", 0.9 * la), 1.6)
                for kx, ky in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                    c.drawLine(sx + kx * r, sy + ky * r, sx + kx * (r - 10), sy + ky * r, p)
                    c.drawLine(sx + kx * r, sy + ky * r, sx + kx * r, sy + ky * (r - 10), p)
                c.drawCircle(sx, sy, 3, paint_fill(col("snow", la)))
                c.drawLine(sx, sy - r - 4, sx, sy - 150, paint_stroke(col("snow", 0.6 * la), 1.2))
                self.ty_nd.draw(c, t, sx, sy - 168, "snow", 1.0, align="center")
                self.ty_co.draw(c, t, sx, sy - 140 + 0, "steel", 0.9, align="center")
            # altitude ruler (right edge)
            ra = smooth(ramp(t, 27.1, 27.4))
            if ra > 0:
                x = W - 96
                _, y0 = self.rv.project(cam, 0.0, 0.0, 0.0)
                y0 = float(min(y0, H - 260))
                y8 = sy - (y0 - sy) * (8000 - SUMMIT_Z) / SUMMIT_Z
                # metres -> screen along the ruler (linear between sea level mark and summit height)
                def yy(m):
                    return lerp(y0, sy, m / SUMMIT_Z)
                c.drawLine(x, yy(0), x, yy(8000), paint_stroke(col("steel", 0.7 * ra), 1.4))
                for km in range(0, 9):
                    ty = yy(km * 1000)
                    c.drawLine(x - (14 if km % 2 == 0 else 8), ty, x, ty, paint_stroke(col("steel", 0.7 * ra), 1.4))
                    if km % 2 == 0:
                        label(c, f"{km}", x + 12, ty + 7, size=18, color="steel", alpha=0.8 * ra)
                label(c, "KM", x + 6, yy(8000) - 18, size=16, color="steel", alpha=0.7 * ra)
                m = 7816 * expo_out(ramp(t, 27.25, 28.45))
                my = yy(m)
                c.drawLine(sx + 30, sy, x - 16, sy, dashed(paint_stroke(col("snow", 0.55 * ra * smooth(ramp(t, 28.0, 28.4))), 1.2), 4, 6))
                c.drawPath(path_from([(x - 4, my), (x - 22, my - 9), (x - 22, my + 9)], close=True),
                           paint_fill(col("snow", ra)))
                label(c, f"{m:,.0f} m", x - 30, my - 16, face="mono_b", size=26, color="snow", alpha=ra, align="right")
            # agencies: dossier tags along the bottom
            for ty_, xx, t_on in ((self.ty_cia, 92, 23.8), (self.ty_ib, 92, 24.7)):
                a = smooth(ramp(t, t_on, t_on + 0.25))
                if a > 0:
                    yb = 1440 if ty_ is self.ty_cia else 1490
                    c.drawRect(skia.Rect.MakeXYWH(xx - 14, yb - 30, 6, 38), paint_fill(col("red" if ty_ is self.ty_cia else "amber", a)))
                    ty_.draw(c, t, xx + 4, yb, "snow", a)
            brackets(c, 54, 250, W - 108, H - 560, "steel", 0.3 * fin)

    def post(self, t):
        return dict(bloom=0.5, bloom_thr=0.6, halation=0.12, vignette=0.36, grain=0.036, lift=0.01)

    def overlay(self, img, t):
        if t < self.t0 + 0.35:
            k = 1 - smooth(ramp(t, self.t0, self.t0 + 0.35))
            img = fx.radial_blur(img, W / 2, H * 0.6, 0.12 * k, 8)
        return img


class Snow:
    """Wind-driven snow in three depth layers, drawn as motion-blurred streaks."""

    def __init__(self, n=900, seed=4):
        r = np.random.default_rng(seed)
        self.p = r.uniform(0, 1, (n, 2))
        self.z = r.uniform(0.15, 1.0, n)
        self.ph = r.uniform(0, 6.28, n)

    def draw(self, c, t, amount, wind=(-1.0, 0.45), t_start=0.0):
        if amount <= 0.01:
            return
        n = int(len(self.z) * clamp01(amount))
        tt = t - t_start
        vx, vy = wind
        for i in range(n):
            z = self.z[i]
            sp = 900 * z + 200
            x = (self.p[i, 0] * (W + 400) + vx * sp * tt + 30 * math.sin(tt * 2 + self.ph[i])) % (W + 400) - 200
            y = (self.p[i, 1] * (H + 400) + vy * sp * tt) % (H + 400) - 200
            L = 10 + 40 * z
            w = 0.8 + 2.6 * z
            a = (0.25 + 0.6 * z) * amount
            c.drawLine(x, y, x - vx * L, y - vy * L, paint_stroke(col("snow", a), w))


class Climb(Scene):
    """S9: In October 1965, climbers carried it up. Near the top, a blizzard hit."""
    theme = "night"
    name = "climb"

    def __init__(self, t0, t1):
        super().__init__(t0, t1)
        self.rv = RidgeView()
        self.snow = Snow()
        self.stars = Stars(seed=12)
        self.pts, self.z = route()
        self.camps = camps()
        self.t_route0, self.t_route1 = 40.95, 42.5
        self.t_bliz = 43.42
        self.ty_date = Typer("OCTOBER 1965", 39.3, face="mono_b", size=26, cps=16, seed=71, tracking=0.2, gain_db=-2)
        self.ty_team = Typer("LED BY M.S. KOHLI", 39.95, size=21, cps=24, seed=72, gain_db=-5)
        self.camp_typers = []
        for k, (name, i) in enumerate(self.camps):
            tc = lerp(self.t_route0, self.t_route1, i / len(self.pts)) + 0.05
            self.camp_typers.append((Typer(name, tc, face="mono_b", size=19, cps=22, seed=73 + k, tracking=0.1,
                                           gain_db=-7, cursor=False), i, tc))
        sfx(self.t_route0, "pencil", gain_db=-16, dur=1.5)
        sfx(self.t_bliz - 0.35, "blizzard", gain_db=-2, dur=1.6 + (t1 - self.t_bliz))

    def cam(self, t):
        u = self.local(t)
        k = smooth(ramp(u, 0, self.t1 - self.t0))
        push = glide(ramp(t, 42.4, 44.2))
        yaw = 38 + 6 * k
        look = (lerp(-2600, -900, push), lerp(-3200, -1100, push))
        lz = lerp(5600, 7000, push)
        return self.rv.camera(lerp(15500, 14000, k) - 4500 * push, yaw, lerp(10200, 9900, k), 0, 46,
                              look=look, cy=H * 0.60, look_z=lz), yaw

    def storm(self, t):
        return smooth(ramp(t, self.t_bliz - 0.25, self.t_bliz + 0.6))

    def draw(self, c, t):
        u = self.local(t)
        cam, yaw = self.cam(t)
        st = self.storm(t)
        fin = smooth(ramp(u, 0.0, 0.3))
        with Layer(c, alpha=fin):
            sky(c)
            self.stars.draw(c, t, 0.8 * (1 - st))
            self.rv.draw(c, cam, yaw, t=t, jitter=3.5 * st, alpha=1 - 0.45 * st)
            # route + camps
            f = smoother(ramp(t, self.t_route0, self.t_route1))
            if f > 0:
                P = self.pts
                Z = self.z + 25
                sx, sy = self.rv.project(cam, P[:, 0], P[:, 1], Z)
                path = path_from(list(zip(sx, sy)))
                c.drawPath(partial(path, f), dashed(paint_stroke(col("snow", 0.9 * (1 - 0.5 * st)), 2.2), 2, 7))
                # device being carried up (amber), stalls at Camp IV
                i4 = self.camps[-1][1]
                fd = min(f, i4 / len(P))
                (dx, dy), _ = point_at(path, fd)
                flick = 1 - 0.35 * st * (0.5 + 0.5 * math.sin(t * 23))
                glow_dot(c, dx, dy, 4.5, "amber", 1.0 * flick, 6)
                for ty_, i, tc in self.camp_typers:
                    if t < tc - 0.05:
                        continue
                    a = smooth(ramp(t, tc, tc + 0.2)) * (1 - 0.6 * st)
                    cx_, cy_ = float(sx[i]), float(sy[i])
                    c.drawCircle(cx_, cy_, 4.2, paint_fill(col("snow", a)))
                    c.drawLine(cx_, cy_, cx_ - 70, cy_, paint_stroke(col("snow", 0.5 * a), 1.0))
                    ty_.draw(c, t, cx_ - 78, cy_ + 6, "snow", a, align="right")
            # whiteout + snow
            if st > 0:
                p = skia.Paint()
                p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(0, H)],
                            [col((0.55, 0.62, 0.68), 0.35 * st).toColor(), col((0.75, 0.80, 0.85), 0.55 * st).toColor()]))
                c.drawRect(skia.Rect.MakeWH(W, H), p)
                self.snow.draw(c, t, st, t_start=self.t_bliz - 0.3)
                # gust bands
                for k in range(3):
                    gy = (H * (0.3 + 0.25 * k) + 140 * math.sin(t * 0.9 + k)) % H
                    with Layer(c, alpha=0.18 * st, blur=40):
                        c.drawRect(skia.Rect.MakeXYWH(-50, gy, W + 100, 120), paint_fill(col((0.85, 0.9, 0.95))))
            if f > 0:
                label(c, "ROUTE (SCHEMATIC)", W - 92, 1478, size=17, color="steel", alpha=0.7 * (1 - 0.6 * st),
                      align="right")
            # HUD
            ha = smooth(ramp(t, 39.2, 39.5)) * (1 - 0.6 * st)
            self.ty_date.draw(c, t, 92, 1440, "snow", ha)
            self.ty_team.draw(c, t, 92, 1478, "steel", ha)
            brackets(c, 54, 250, W - 108, H - 560, "steel", 0.3 * fin)

    def post(self, t):
        st = self.storm(t)
        return dict(bloom=0.5 + 0.3 * st, bloom_thr=0.6, halation=0.1, vignette=0.36 - 0.1 * st, grain=0.036 + 0.02 * st,
                    lift=0.01 + 0.03 * st, contrast=1.0 - 0.12 * st)

    def mblur(self, t):
        return 0.0


class Slide(Scene):
    """S11b: swept down the mountain - the slope under Camp IV gives way."""
    theme = "night"
    name = "slide"

    def __init__(self, t0, t1, t_go=50.25):
        super().__init__(t0, t1)
        self.rv = RidgeView()
        self.pts, self.z = route()
        self.c4 = self.pts[camps()[-1][1]]
        self.t_go = t_go
        self.ty_q = Typer("?", t_go + 0.75, face="sans_bold", size=60, cps=4, seed=81, gain_db=-4, cursor=False)
        sfx(t_go - 0.05, "avalanche", gain_db=0.0, dur=2.6)

    def zmod(self, t):
        u = max(0.0, t - self.t_go)
        if u <= 0:
            return None
        ce, cn = self.c4
        # flow direction: downslope toward the south-west sanctuary
        dv = np.array([-0.62, -0.78])

        def f(e, n, z):
            de, dn = e - ce, n - cn
            along = de * dv[0] + dn * dv[1]
            across = -de * dv[1] + dn * dv[0]
            front = 3200 * expo_out(u / 1.6)
            width = 700 + 0.25 * np.maximum(along, 0)
            m = (along > -200) & (along < front) & (np.abs(across) < width)
            w = np.exp(-(across / width) ** 2 * 2.5) * np.clip((front - along) / 400, 0, 1) * np.clip((along + 200) / 300, 0, 1)
            churn = fbm_np(along * 0.004 + u * 3.0 + across * 0.002, 4.1) * 120
            return z - m * w * (420 + churn * 2) * min(1.0, u / 0.5)
        return f

    def draw(self, c, t):
        u = self.local(t)
        k = smooth(ramp(u, 0, self.t1 - self.t0))
        yaw = 30
        shake = 0.0
        if t > self.t_go:
            shake = 10 * math.exp(-(t - self.t_go) * 0.9) * min(1, (t - self.t_go) * 4)
        ce, cn = self.c4
        cam = self.rv.camera(11000 - 700 * k, yaw, 9600, 0, 48, look=(ce - 1500.0, cn - 1900.0), cy=H * 0.52,
                             look_z=6000)
        cam.cx += shake * fbm(t * 11, 1.0)
        cam.cy += shake * fbm(t * 11, 2.0)
        fin = smooth(ramp(u, 0.0, 0.2))
        with Layer(c, alpha=fin):
            sky(c)
            self.rv.draw(c, cam, yaw, t=t, zmod=self.zmod(t), jitter=1.5 * smooth(ramp(t, self.t_go, self.t_go + 0.4)))
            sx, sy = self.rv.project(cam, self.c4[0], self.c4[1], float(frame().height(np.array([self.c4[0]]), np.array([self.c4[1]]))[0]) + 30)
            sx, sy = float(sx), float(sy)
            c.drawCircle(sx, sy, 6, paint_stroke(col("amber", 0.9), 1.6))
            label(c, "C-IV", sx + 14, sy - 10, face="mono_b", size=19, color="snow", alpha=0.9)
            # trajectory arrow into the unknown
            f = expo_out(ramp(t, self.t_go + 0.15, self.t_go + 1.1))
            if f > 0:
                ex, ey = sx - 300, sy + 420
                pth = smooth_path([(sx, sy), (sx - 80, sy + 160), (sx - 200, sy + 300), (ex, ey)])
                c.drawPath(partial(pth, f), dashed(paint_stroke(col("amber", 0.9), 2.0), 6, 8))
                self.ty_q.draw(c, t, ex - 10, ey + 70, "amber", 1.0, align="center")
            # powder cloud: thousands of grains streaming downslope, plus a soft body
            if t > self.t_go:
                uu = t - self.t_go
                rng = np.random.default_rng(3)
                n = 1400
                a0 = rng.uniform(0, 1, n)
                spd = rng.uniform(0.5, 1.0, n)
                spread = rng.normal(0, 1, n)
                d = np.clip(uu * 1.1 * spd - a0 * 0.35, 0, None)
                px = sx - d * 300 + spread * (20 + d * 110)
                py = sy + 20 + d * 520 + np.abs(spread) * d * 30 - 60 * d * d
                ok = d > 0
                body = clamp01(uu * 1.5) * (1 - smooth(ramp(uu, 1.4, 2.0)) * 0.3)
                for k in range(3):
                    cxk = sx - (0.25 + 0.3 * k) * min(uu, 1.6) * 300
                    cyk = sy + (0.25 + 0.3 * k) * min(uu, 1.6) * 520
                    r = 60 + 90 * k * min(1, uu)
                    pp = skia.Paint(AntiAlias=True)
                    pp.setShader(skia.GradientShader.MakeRadial(skia.Point(cxk, cyk), r,
                                 [col((0.75, 0.8, 0.86), 0.22 * body).toColor(), col((0.75, 0.8, 0.86), 0).toColor()]))
                    c.drawCircle(cxk, cyk, r, pp)
                pts = [skia.Point(float(x), float(y)) for x, y, o in zip(px, py, ok) if o]
                if pts:
                    pnt = skia.Paint(AntiAlias=True, Color4f=col((0.88, 0.92, 0.96), 0.55 * body))
                    pnt.setStrokeWidth(2.2)
                    pnt.setStrokeCap(skia.Paint.kRound_Cap)
                    c.drawPoints(skia.Canvas.kPoints_PointMode, pts, pnt)
            brackets(c, 54, 250, W - 108, H - 560, "steel", 0.3 * fin)
        # dust bloom that carries us into the search
        wo = smooth(ramp(t, self.t1 - 0.35, self.t1))
        if wo > 0:
            c.drawRect(skia.Rect.MakeWH(W, H), paint_fill(col((0.70, 0.75, 0.80), 0.85 * wo)))

    def post(self, t):
        return dict(bloom=0.55, bloom_thr=0.6, vignette=0.36, grain=0.04, lift=0.015)
