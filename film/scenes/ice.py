"""S1 / S15 - the heart in the ice.

A real west-east cross-section through Nanda Devi (DEM z12), vertically exaggerated,
with fractal ice strata so one continuous camera move can travel from the plutonium
capsules (centimetres) to the whole massif (kilometres)."""
from __future__ import annotations

import math

import numpy as np
import skia

from engine.core import (W, H, PAL, col, mix, paint_fill, paint_stroke, Layer, expo_out, expo_inout, smooth,
                         smoother, clamp01, ramp, fbm, fbm_np, hash01, lerp, draw_text, quart_inout, silk, glide)
from engine.kit import (Scene, path_from, glow_dot, label, leader, dashed, partial, smooth_path)
from engine.typer import Typer
from engine.sfx import sfx
from engine import terrain as T

ND = (79.9708, 30.3758)
EXAG = 1.9
BASE_Z = 3000.0


def heartbeat(t, period=1.3, phase=0.0):
    """lub-dub envelope in [0, 1]."""
    u = ((t + phase) % period) / period
    a = math.exp(-((u - 0.02) / 0.035) ** 2)
    b = 0.6 * math.exp(-((u - 0.2) / 0.045) ** 2)
    return a + b


class IceSection:
    """Geometry + drawing of the cross-section; camera = (cx, cy, s) in display metres."""

    def __init__(self):
        d = T.dem("nanda_z12")
        self.xs = np.linspace(-11000, 11000, 2200)
        kx = 111320 * math.cos(math.radians(ND[1]))
        z = d.sample(ND[0] + self.xs / kx, np.full_like(self.xs, ND[1]))
        z = np.convolve(np.pad(z, 3, mode="edge"), np.ones(7) / 7, "valid")
        self.ztrue = z
        self.zd = BASE_Z + (z - BASE_Z) * EXAG          # display height (m)
        i = int(np.argmax(self.zd))
        self.summit = (float(self.xs[i]), float(self.zd[i]))
        # core: west flank, buried in the ice cap (illustrative - true location unknown)
        cx = self.summit[0] - 1350.0
        self.core = (cx, float(np.interp(cx, self.xs, self.zd)) - 70.0)
        rng = np.random.default_rng(3)
        self.bubbles = [(self.core[0] + rng.normal(0, 0.9), self.core[1] + rng.normal(0, 0.7),
                         rng.uniform(0.004, 0.028), rng.uniform(0, 10)) for _ in range(110)]
        self.cracks = []
        for k in range(16):
            a = rng.uniform(0, 2 * math.pi)
            r0 = rng.uniform(0.45, 0.7)
            pts = [(self.core[0] + math.cos(a) * r0, self.core[1] + math.sin(a) * r0)]
            L = rng.uniform(0.4, 1.8)
            for j in range(7):
                a += rng.normal(0, 0.35)
                step = L / 7
                pts.append((pts[-1][0] + math.cos(a) * step, pts[-1][1] + math.sin(a) * step))
            self.cracks.append(pts)
        self.stars = [(rng.uniform(0, W), rng.uniform(0, H * 0.75), rng.uniform(0.4, 1.4), rng.uniform(0, 6.28))
                      for _ in range(260)]

    def surf(self, x):
        return np.interp(x, self.xs, self.zd)

    def ice_thick(self, x):
        zt = np.interp(x, self.xs, self.ztrue)
        return 160.0 * EXAG * np.clip((zt - 5000) / 900, 0, 1)

    # world (display m) -> screen
    @staticmethod
    def to_screen(cam, x, y):
        cx, cy, s = cam
        return W / 2 + (np.asarray(x) - cx) * s, H / 2 - (np.asarray(y) - cy) * s

    def draw(self, c, t, cam, glow_k=1.0, labels=None, alpha=1.0, star_alpha=1.0):
        cx, cy, s = cam
        xl, xr = cx - W / 2 / s - 2 / s, cx + W / 2 / s + 2 / s
        yt, yb = cy + H / 2 / s, cy - H / 2 / s
        # sky: deep night with a cold glow low over the range
        p = skia.Paint()
        p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(0, H)],
                    [col((0.010, 0.013, 0.018)).toColor(), col((0.030, 0.042, 0.056)).toColor(),
                     col((0.055, 0.075, 0.095)).toColor()], [0.0, 0.55, 1.0]))
        c.drawRect(skia.Rect.MakeWH(W, H), p)
        sky_vis = clamp01((0.25 - s) / 0.2)
        if sky_vis > 0 and star_alpha > 0:
            for (x, y, r, ph) in self.stars:
                tw = 0.55 + 0.45 * math.sin(t * 1.3 + ph)
                c.drawCircle(x, y, r, paint_fill(col("snow", 0.35 * tw * sky_vis * star_alpha)))
        n = 420
        X = np.linspace(xl, xr, n)
        Y = self.surf(X)
        sx, sy = self.to_screen(cam, X, Y)
        sy = np.clip(sy, -4000, H + 4000)
        interior = path_from(list(zip(sx, sy)) + [(sx[-1], H + 10), (sx[0], H + 10)], close=True)
        p = skia.Paint(AntiAlias=True)
        top_y = float(sy.min())
        p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, top_y), skia.Point(0, max(top_y + 900, H))],
                    [col((0.075, 0.090, 0.108)).toColor(), col((0.030, 0.036, 0.044)).toColor()]))
        c.drawPath(interior, p)
        c.save()
        c.clipPath(interior, skia.ClipOp.kIntersect, True)
        # rock bedding: tilted metamorphic layers at three scales
        ang = math.radians(-17)
        ca_, sa_ = math.cos(ang), math.sin(ang)
        for level, sp in enumerate((2.5, 20.0, 160.0)):
            px_sp = sp * s
            a = clamp01((px_sp - 4) / 8) * clamp01((700 - px_sp) / 400)
            if a <= 0.01:
                continue
            # lines u = x*sin + y*cos = const, in world; iterate over visible u range
            corners = [(xl, yt), (xr, yt), (xl, yb), (xr, yb)]
            us = [cx_ * sa_ + cy_ * ca_ for cx_, cy_ in corners]
            k0, k1 = int(min(us) // sp), int(max(us) // sp) + 1
            if k1 - k0 > 220:
                continue
            L = np.linspace(-1, 1, 90) * (abs(xr - xl) + abs(yt - yb))
            mid_x, mid_y = (xl + xr) / 2, (yt + yb) / 2
            for k in range(k0, k1):
                u0 = k * sp
                d0 = u0 - (mid_x * sa_ + mid_y * ca_)
                bx_, by_ = mid_x + sa_ * d0, mid_y + ca_ * d0
                wx = bx_ + ca_ * L
                wy = by_ - sa_ * L + fbm_np(L / (sp * 6) + k * 0.71, level * 7.7) * sp * 0.3
                lx, ly = self.to_screen(cam, wx, wy)
                hv = hash01(k, level, 3.3)
                c.drawPath(path_from(list(zip(lx, ly))),
                           paint_stroke(col((0.30, 0.33, 0.37), a * (0.10 + 0.16 * hv) * alpha), 0.9 + 1.4 * (hv > 0.8)))
        # ice cap band
        th = self.ice_thick(X)
        _, sy2 = self.to_screen(cam, X, Y - th)
        sy2 = np.clip(sy2, -4000, H + 4000)
        band = path_from(list(zip(sx, sy)) + list(zip(sx[::-1], sy2[::-1])), close=True)
        p = skia.Paint(AntiAlias=True)
        p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, top_y), skia.Point(0, top_y + max(60, 300 * s))],
                    [col((0.16, 0.23, 0.29), 0.95).toColor(), col((0.075, 0.12, 0.16), 0.92).toColor()]))
        c.drawPath(band, p)
        # ice strata at several scales (annual layers, with dark dust bands)
        for level, sp in enumerate((0.07, 0.5, 4.0, 32.0, 240.0)):
            px_sp = sp * s
            a = clamp01((px_sp - 5) / 10) * clamp01((900 - px_sp) / 500)
            if a <= 0.01:
                continue
            d0 = max(0.0, (self.surf(cx) - yt) - 50 * sp)
            d1 = (self.surf(cx) - yb) + 50 * sp
            k0, k1 = int(d0 // sp), int(d1 // sp) + 1
            if k1 - k0 > 260:
                k0 = max(k0, int((self.surf(cx) - cy) // sp) - 130)
                k1 = k0 + 260
            Xl = np.linspace(xl, xr, 160)
            Ys = self.surf(Xl)
            thl = self.ice_thick(Xl)
            for k in range(max(k0, 1), k1):
                dep = k * sp
                wav = fbm_np(Xl / (sp * 9) + k * 1.37, level * 13.1 + k * 0.17)
                yy = Ys - dep + wav * sp * 0.35
                in_ice = dep < thl
                if not in_ice.any():
                    continue
                lx, ly = self.to_screen(cam, Xl, yy)
                if ly.max() < -10 or ly.min() > H + 10:
                    continue
                hv = hash01(k, level)
                frac = float(np.mean(in_ice))
                if hv > 0.88:   # dust / ash band
                    c.drawPath(path_from(list(zip(lx, ly))),
                               paint_stroke(col((0.03, 0.05, 0.07), a * 0.55 * frac * alpha), min(sp * s * 0.35, 9)))
                else:
                    tone = (0.50, 0.66, 0.78) if hv > 0.25 else (0.36, 0.48, 0.58)
                    c.drawPath(path_from(list(zip(lx, ly))),
                               paint_stroke(col(tone, a * (0.22 + 0.30 * hv) * frac * alpha), 0.9 + 0.8 * hv))
        # snow crust just under the surface
        crust = path_from(list(zip(sx, sy + 3)))
        with Layer(c, alpha=0.35 * alpha, blur=6):
            c.drawPath(crust, paint_stroke(col((0.75, 0.82, 0.88)), 10))
        # meltwater cavity + cracks + bubbles around the core
        ccx, ccy = self.to_screen(cam, *self.core)
        rcav = 0.62 * s
        if rcav > 2:
            with Layer(c, alpha=0.9 * alpha, blur=max(1.0, rcav * 0.12)):
                c.drawCircle(ccx, ccy, rcav, paint_fill(col((0.02, 0.035, 0.05))))
        crack_a = clamp01((s - 60) / 120)
        if crack_a > 0:
            for pts in self.cracks:
                X_, Y_ = zip(*pts)
                qx, qy = self.to_screen(cam, X_, Y_)
                c.drawPath(path_from(list(zip(qx, qy))), paint_stroke(col((0.55, 0.72, 0.85), 0.35 * crack_a), 1.2))
        bub_a = clamp01((s - 80) / 140)
        if bub_a > 0:
            for (bx, by, br, ph) in self.bubbles:
                yy = by + 0.012 * t + 0.01 * math.sin(t * 0.7 + ph)
                qx, qy = self.to_screen(cam, bx + 0.006 * math.sin(t * 0.5 + ph), yy)
                rr = br * s
                if rr < 0.6 or qx < -20 or qx > W + 20 or qy < -20 or qy > H + 20:
                    continue
                c.drawCircle(float(qx), float(qy), rr, paint_stroke(col((0.65, 0.8, 0.9), 0.45 * bub_a), max(0.8, rr * 0.18)))
                c.drawCircle(float(qx - rr * 0.35), float(qy - rr * 0.35), rr * 0.22, paint_fill(col("snow", 0.5 * bub_a)))
        c.restore()
        # rim light along the surface
        rim = path_from(list(zip(sx, sy)))
        with Layer(c, alpha=0.5 * alpha, blur=5, blend=skia.BlendMode.kPlus):
            c.drawPath(rim, paint_stroke(col((0.55, 0.65, 0.75)), 3))
        c.drawPath(rim, paint_stroke(col((0.86, 0.9, 0.94), 0.9 * alpha), 1.6))
        self.draw_core(c, t, cam, glow_k)
        return ccx, ccy

    def draw_core(self, c, t, cam, k):
        s = cam[2]
        x, y = self.to_screen(cam, *self.core)
        x, y = float(x), float(y)
        beat = heartbeat(t)
        inten = k * (0.78 + 0.22 * beat)
        # light spill into the ice
        spill = max(150.0, 2.8 * s)
        p = skia.Paint(AntiAlias=True)
        a = col("amber")
        p.setShader(skia.GradientShader.MakeRadial(skia.Point(x, y), spill,
                    [skia.Color4f(a.fR, a.fG, a.fB, 0.20 * inten).toColor(),
                     skia.Color4f(a.fR, a.fG, a.fB, 0.05 * inten).toColor(),
                     skia.Color4f(a.fR, a.fG, a.fB, 0).toColor()], [0, 0.4, 1]))
        p.setBlendMode(skia.BlendMode.kPlus)
        c.drawCircle(x, y, spill, p)
        R = 0.25 * s  # generator body radius on screen
        if R > 7:
            det = clamp01((R - 7) / 20)
            c.save()
            c.translate(x, y)
            c.rotate(-8 + 2 * math.sin(t * 0.3))
            steel = col((0.62, 0.68, 0.74), det)
            # six radiator fins
            for i in range(6):
                c.save()
                c.rotate(i * 60 + 30)
                c.drawRect(skia.Rect.MakeLTRB(R * 0.92, -R * 0.07, R * 1.95, R * 0.07),
                           paint_fill(col((0.10, 0.12, 0.14), det)))
                c.drawRect(skia.Rect.MakeLTRB(R * 0.92, -R * 0.07, R * 1.95, R * 0.07), paint_stroke(steel, 1.3))
                c.restore()
            c.drawCircle(0, 0, R, paint_fill(col((0.09, 0.10, 0.11), det)))
            c.drawCircle(0, 0, R, paint_stroke(steel, 1.6))
            c.drawCircle(0, 0, R * 0.80, paint_stroke(col((0.62, 0.68, 0.74), 0.5 * det), 1.0))
            # thermocouple ring (zig-zag)
            zz = []
            for i in range(97):
                ang = i / 96 * 2 * math.pi
                rr = R * (0.62 + (0.05 if i % 2 else -0.05))
                zz.append((math.cos(ang) * rr, math.sin(ang) * rr))
            c.drawPath(path_from(zz), paint_stroke(col((0.85, 0.6, 0.35), 0.7 * det), 1.0))
            # graphite block + seven capsules (hex)
            hot = col("amber")
            pp = skia.Paint(AntiAlias=True)
            pp.setShader(skia.GradientShader.MakeRadial(skia.Point(0, 0), R * 0.48,
                         [col((1.0, 0.75, 0.42), det * inten).toColor(), col((0.55, 0.25, 0.08), det * inten * 0.9).toColor()]))
            c.drawCircle(0, 0, R * 0.46, pp)
            cap_r = R * 0.105
            for i in range(7):
                if i == 0:
                    px, py = 0.0, 0.0
                else:
                    ang = (i - 1) * math.pi / 3
                    px, py = math.cos(ang) * R * 0.25, math.sin(ang) * R * 0.25
                glow_dot(c, px, py, cap_r, "core", intensity=det * inten, halo=2.4)
            c.restore()
        glow_dot(c, x, y, max(3.0, min(R * 0.5, 9.0)), "amber", intensity=inten * (1 - clamp01((R - 30) / 60) * 0.6),
                 halo=7.0)


class IceScene(Scene):
    theme = "night"
    name = "ice"

    def __init__(self, t0, t1, mode="open", vo=None):
        super().__init__(t0, t1)
        self.mode = mode
        self.sec = IceSection()
        sx, sy = self.sec.summit
        self.full_cam = (sx - 700.0, sy - 1750.0, 0.095)
        i = int(np.searchsorted(self.sec.xs, self.sec.core[0]))
        dz = (self.sec.zd[i + 6] - self.sec.zd[i - 6]) / (self.sec.xs[i + 6] - self.sec.xs[i - 6])
        self.slope_deg = math.degrees(math.atan(dz))
        if mode == "open":
            self.ty_gen = Typer("SNAP-19C  GENERATOR", t0 + 1.05, size=24, cps=24, seed=11, gain_db=-3, pan=0.25)
            self.ty_pu = Typer("7 \u00d7 PLUTONIUM-238 CAPSULES", t0 + 1.75, size=24, cps=24, seed=12, gain_db=-3, pan=0.25)
            self.ty_nd = Typer("NANDA DEVI  7,816 m", 4.95, size=26, cps=20, seed=13, gain_db=-2, pan=0.1)
            self.ty_loc = Typer("LOCATION: UNKNOWN", 6.10, size=24, cps=18, seed=14, gain_db=-2, pan=-0.1)
            for k in range(6):
                sfx(t0 + 0.35 + k * 1.3, "heart", gain_db=-6 - k * 1.5)
            sfx(t0 + 1.2, "deep_air", gain_db=-8, dur=4.6)       # long pull-back breath
        else:
            self.ty_h = Typer("Pu-238  HALF-LIFE  87.7 YEARS", t0 + 0.55, size=26, cps=26, seed=21, gain_db=-4, pan=-0.2)
            self.ty_p = Typer("HEAT LEFT TODAY  \u2248 62%", t0 + 1.55, face="mono_b", size=30, cps=22, seed=22, gain_db=-3, pan=-0.2)
            for k in range(4):
                sfx(t0 + 0.4 + k * 1.3, "heart", gain_db=-5 - k * 2)

    def core_screen(self, t):
        cam = self.cam_open(t)
        x, y = self.sec.to_screen(cam, *self.sec.core)
        return float(x), float(y)

    def overlay(self, img, t):
        if self.mode == "open" and t > 7.1:
            from engine import fx
            k = smooth(ramp(t, 7.1, 7.75))
            x, y = self.core_screen(t)
            img = fx.radial_blur(img, x, y, 0.10 * k, 8)
        return img

    def cam_open(self, t):
        u = self.local(t)
        k = glide(ramp(u, 1.15, 5.35))
        s0, s1 = 520.0, self.full_cam[2]
        s = math.exp(lerp(math.log(s0), math.log(s1), k))
        # keep the core pinned to a screen point that travels from centre to its final spot
        fx_, fy_ = self.sec.to_screen(self.full_cam, *self.sec.core)
        P = (lerp(W / 2, float(fx_), smooth(k)), lerp(H * 0.56, float(fy_), smooth(k)))
        cx = self.sec.core[0] - (P[0] - W / 2) / s
        cy = self.sec.core[1] + (P[1] - H / 2) / s
        # late gentle push
        push = 1 + 0.035 * smooth(ramp(u, 5.3, 7.2)) + 1.6 * expo_inout(ramp(u, 7.15, 7.75)) ** 2
        s *= push
        cx = self.sec.core[0] - (P[0] - W / 2) / s
        cy = self.sec.core[1] + (P[1] - H / 2) / s
        return (cx, cy, s)

    def cam_end(self, t):
        u = self.local(t)
        k = expo_inout(ramp(u, 0.0, 3.6) * 0.92)
        s0, s1 = self.full_cam[2] * 1.05, 34.0
        s = math.exp(lerp(math.log(s0), math.log(s1), k))
        fx_, fy_ = self.sec.to_screen(self.full_cam, *self.sec.core)
        P = (lerp(float(fx_), W / 2, smooth(k)), lerp(float(fy_), H * 0.6, smooth(k)))
        cx = self.sec.core[0] - (P[0] - W / 2) / s
        cy = self.sec.core[1] + (P[1] - H / 2) / s
        return (cx, cy, s)

    def draw(self, c, t):
        u = self.local(t)
        if self.mode == "open":
            cam = self.cam_open(t)
            fade_in = smooth(ramp(u, 0.0, 0.9)) * (1 - smooth(ramp(u, 7.5, 7.9)))
            glow_k = smooth(ramp(u, 0.05, 0.6))
            k = glide(ramp(u, 1.15, 5.35))
            roll = self.slope_deg * (1 - smoother(ramp(u, 0.9, 4.6)))
            px_, py_ = self.sec.to_screen(cam, *self.sec.core)
            with Layer(c, alpha=fade_in):
                c.save()
                c.translate(float(px_), float(py_))
                c.rotate(roll)
                c.translate(-float(px_), -float(py_))
                x, y = self.sec.draw(c, t, cam, glow_k)
                c.restore()
                # macro labels
                la = smooth(ramp(u, 1.0, 1.3)) * (1 - smooth(ramp(u, 2.6, 3.2)))
                if la > 0:
                    R = 0.25 * cam[2]
                    ax, ay = x + R * 0.72, y - R * 0.72
                    bx, by = x + R * 1.25 + 40, y - R * 1.25 - 120
                    leader(c, ax, ay, bx, by, "snow", 0.8 * la, 1.4, 3.5, ramp(u, 0.95, 1.25))
                    self.ty_gen.draw(c, t, bx + 10, by - 6, "snow", la)
                    self.ty_pu.draw(c, t, bx + 10, by + 30, "amber", la)
                # summit label
                lb = smooth(ramp(u, 4.85, 5.2))
                if lb > 0:
                    sx, sy = self.sec.to_screen(cam, *self.sec.summit)
                    leader(c, float(sx), float(sy) - 10, float(sx), float(sy) - 130, "snow", 0.75 * lb, 1.3, 3, ramp(u, 4.85, 5.1))
                    self.ty_nd.draw(c, t, float(sx), float(sy) - 146, "snow", lb, align="center")
                # reticle: never found
                lc = smooth(ramp(u, 5.95, 6.3))
                if lc > 0:
                    r = 54 + 10 * (1 - expo_out(ramp(u, 5.95, 6.6)))
                    p = dashed(paint_stroke(col("amber", 0.85 * lc), 1.6), 7, 7, -u * 30)
                    c.save()
                    c.translate(x, y)
                    c.rotate(u * 22)
                    c.drawCircle(0, 0, r, p)
                    c.restore()
                    for a_ in range(4):
                        ang = a_ * math.pi / 2 + 0.785
                        c.drawLine(x + math.cos(ang) * (r + 6), y + math.sin(ang) * (r + 6),
                                   x + math.cos(ang) * (r + 18), y + math.sin(ang) * (r + 18),
                                   paint_stroke(col("amber", 0.85 * lc), 1.6))
                    self.ty_loc.draw(c, t, x, y + r + 52, "amber", lc, align="center")
        else:
            cam = self.cam_end(t)
            fin = smooth(ramp(u, 0.0, 0.5))
            out = smooth(ramp(t, self.t1 - 1.6, self.t1 - 0.15))
            with Layer(c, alpha=fin * (1 - out * 0.85)):
                x, y = self.sec.draw(c, t, cam, 1.0 + 0.35 * smooth(ramp(u, 1.6, 2.2)), star_alpha=1.0)
            # decay chart (top-right, under captions)
            ca = smooth(ramp(u, 0.4, 0.8)) * (1 - out)
            if ca > 0:
                self.draw_decay(c, t, 120, 1330, 520, 190, ca)
            if out > 0:
                # the core lingers a beat longer than the world
                core_a = 1 - smooth(ramp(t, self.t1 - 0.9, self.t1 - 0.05))
                glow_dot(c, x, y, 7, "amber", intensity=core_a * (0.8 + 0.2 * heartbeat(t)), halo=8)

    def draw_decay(self, c, t, x, y, w, h, a):
        u = self.local(t)
        ax = paint_stroke(col("steel", 0.6 * a), 1.2)
        c.drawLine(x, y + h, x + w, y + h, ax)
        c.drawLine(x, y, x, y + h, ax)
        yrs = np.linspace(1965, 2120, 140)
        frac = 0.5 ** ((yrs - 1965) / 87.7)
        px = x + (yrs - 1965) / (2120 - 1965) * w
        py = y + h - frac * h
        curve = path_from(list(zip(px, py)))
        f = smooth(ramp(u, 0.5, 1.6))
        c.drawPath(partial(curve, f), paint_stroke(col("amber", 0.95 * a), 2.0))
        x26 = x + (2026 - 1965) / (2120 - 1965) * w
        y26 = y + h - 0.617 * h
        m = smooth(ramp(u, 1.45, 1.7))
        if m > 0:
            c.drawLine(x26, y + h, x26, y26, dashed(paint_stroke(col("snow", 0.6 * a * m), 1.2), 4, 4))
            glow_dot(c, x26, y26, 4, "amber", intensity=a * m, halo=4)
            label(c, "1965", x, y + h + 32, size=21, color="steel", alpha=0.8 * a)
            label(c, "2026", x26, y + h + 32, size=21, color="snow", alpha=0.9 * a * m, align="center")
        self.ty_h.draw(c, t, x, y - 64, "steel", a)
        self.ty_p.draw(c, t, x, y - 22, "amber", a)

    def post(self, t):
        u = self.local(t)
        if self.mode == "open":
            return dict(bloom=0.75, bloom_thr=0.5, halation=0.22, vignette=0.38, grain=0.036, lift=0.01,
                        exposure=1.0)
        return dict(bloom=0.8, bloom_thr=0.5, halation=0.24, vignette=0.42, grain=0.036, lift=0.008,
                    fade=1 - smooth(ramp(t, self.t1 - 0.5, self.t1)) * 0.0)
