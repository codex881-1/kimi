"""Paper-world shots: line-of-sight explainer (S5), device blueprint (S7-S8), 1978 record (S13)."""
from __future__ import annotations

import math

import numpy as np
import skia

from engine.core import (W, H, PAL, col, mix, paint_fill, paint_stroke, Layer, expo_out, expo_inout, smooth,
                         smoother, clamp01, ramp, lerp, draw_text, text_width, glide, silk, fbm, spring, back_out,
                         cubic_inout, quart_inout)
from engine.kit import (Scene, draw_paper, path_from, smooth_path, partial, point_at, dashed, label, leader, glow_dot,
                        path_length)
from engine.typer import Typer
from engine.sfx import sfx
from engine import terrain as T
from engine import fx

INK = PAL["ink"]
PENCIL = PAL["pencil"]
SIG = (0.16, 0.42, 0.56)      # signal ink on paper
RED = PAL["red"]
ND = (79.9708, 30.3758)
LOP = (89.80, 40.80)


def ink(a=1.0, w=2.0):
    return paint_stroke(col(INK, a), w)


def hatch(c, clip_path, spacing=7.0, angle=-35.0, color=INK, alpha=0.5, width=1.0, bounds=None):
    """Engraving-style hatch fill clipped to a path."""
    c.save()
    c.clipPath(clip_path, skia.ClipOp.kIntersect, True)
    b = bounds or clip_path.getBounds()
    cx, cy = b.centerX(), b.centerY()
    L = math.hypot(b.width(), b.height()) / 2 + 10
    a = math.radians(angle)
    dx, dy = math.cos(a), math.sin(a)
    nx, ny = -dy, dx
    p = paint_stroke(col(color, alpha), width, cap=skia.Paint.kButt_Cap)
    k = -L
    while k <= L:
        ox, oy = cx + nx * k, cy + ny * k
        c.drawLine(ox - dx * L, oy - dy * L, ox + dx * L, oy + dy * L, p)
        k += spacing
    c.restore()


class PaperScene(Scene):
    theme = "paper"
    slide_in = 0.42

    def theme_value(self, t):
        return smooth(ramp(t, self.t0, self.t0 + 0.12))

    def sheet(self, c, t, drift_scale=1.0):
        """Paper slides up over the previous shot like a sheet put on the desk."""
        u = self.local(t)
        k = expo_out(clamp01(u / self.slide_in))
        oy = (1 - k) * H * 1.05
        rot = (1 - spring(u, 0.6, 11)) * 3.0
        # contact shadow on the shot underneath
        if k < 1:
            with Layer(c, alpha=0.55 * k, blur=40):
                c.drawRect(skia.Rect.MakeLTRB(-40, oy - 30, W + 40, H + 200), paint_fill(col((0, 0, 0))))
        c.translate(W / 2, H / 2 + oy)
        c.rotate(rot)
        c.translate(-W / 2, -H / 2)
        dx, dy = 6 * fbm(t * 0.2, 4.4), 6 * fbm(t * 0.2, 9.1)
        draw_paper(c, t, cam_x=dx, cam_y=dy, scale=1.0)
        return k

    def post(self, t):
        return dict(bloom=0.08, bloom_thr=0.9, halation=0.0, chroma=0.55, vignette=0.24, grain=0.028, warmth=0.012)


class LineOfSight(PaperScene):
    name = "los"

    def __init__(self, t0, t1):
        super().__init__(t0, t1)
        d = T.dem("region_z7")
        # path: Indian plains (behind Nanda Devi on the same axis) -> Nanda Devi -> Lop Nur
        back = 0.20
        P0 = (ND[0] - (LOP[0] - ND[0]) * back, ND[1] - (LOP[1] - ND[1]) * back)
        dist, z, _, _ = T.profile(d, P0, LOP, n=700)
        z = np.convolve(np.pad(np.maximum(z, 0), 2, mode="edge"), np.ones(5) / 5, "valid")
        self.D = dist[-1]
        self.dist, self.z = dist, z
        self.d_nd = float(dist[np.argmin(np.abs(dist - dist[-1] * back / (1 + back)))])
        i_nd = int(np.argmax(np.where(np.abs(dist - self.d_nd) < 60, z, 0)))
        self.d_nd, self.z_nd = float(dist[i_nd]), float(z[i_nd])
        # earth arc (exaggerated curvature) + exaggerated relief
        self.cx, self.R = W / 2, 1500.0
        self.cy = 1470 + self.R
        self.th0, self.th1 = -math.asin(482 / self.R), math.asin(482 / self.R)
        self.k = 430 / 8000.0
        # missile flight above the Tarim, ending at Lop Nur
        self.t_ray1, self.t_hit = 18.98, 20.10
        self.t_climb0, self.t_climb1 = 21.45, 22.75
        self.t_ray2 = 22.62
        M = self.missile(0.55)
        L1 = self.ground(0.0, 40)
        self.hit = self.first_hit(M, L1)
        self.ty_title = Typer("LINE OF SIGHT", t0 + 0.45, face="type_b", size=36, cps=16, seed=51, gain_db=-2,
                              typewriter=True, cursor=False)
        self.ty_sub = Typer("SCHEMATIC \u00b7 NOT TO SCALE", t0 + 1.25, face="type", size=21, cps=22, seed=52,
                            gain_db=-6, typewriter=True, cursor=False)
        self.ty_blk = Typer("BLOCKED", self.t_hit + 0.08, face="mono_b", size=24, cps=16, seed=53, gain_db=-2,
                            tracking=0.2, cursor=False)
        self.ty_clr = Typer("CLEAR LINE OF SIGHT", self.t_ray2 + 0.42, face="mono_b", size=22, cps=24, seed=54,
                            gain_db=-3, tracking=0.12, cursor=False)
        sfx(t0 - 0.05, "paper_slide", gain_db=-3)
        sfx(t0 + 0.6, "pencil", gain_db=-12, dur=1.5)
        sfx(self.t_ray1, "zap_soft", gain_db=-9, dur=0.9)
        sfx(self.t_hit, "thud_deny", gain_db=-4)
        sfx(self.t_climb0, "rise", gain_db=-10, dur=1.3)
        sfx(self.t_ray2, "zap_soft", gain_db=-8, dur=0.6)
        sfx(self.t_ray2 + 0.38, "lock", gain_db=-6)
        for k in range(7):
            sfx(18.7 + k * 0.42, "blip", gain_db=-17, pan=0.35, freq=1300 + 120 * (k % 2))

    # geometry ---------------------------------------------------------
    def polar(self, dkm, hm):
        th = self.th0 + (dkm / self.D) * (self.th1 - self.th0)
        r = self.R + hm * self.k
        return self.cx + r * math.sin(th), self.cy - r * math.cos(th)

    def ground(self, dkm, extra=0.0):
        return self.polar(dkm, float(np.interp(dkm, self.dist, self.z)) + extra)

    def missile(self, u):
        d1 = self.D
        alt = 7000 + 7600 * math.sin(math.pi * clamp01(u * 0.85 + 0.08))
        return self.polar(lerp(d1 + 30, d1 - 600, u), alt)

    def first_hit(self, A, B, n=900):
        for i in range(n + 1):
            f = i / n
            x, y = A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f
            r = math.hypot(x - self.cx, y - self.cy)
            th = math.atan2(x - self.cx, self.cy - y)
            dkm = (th - self.th0) / (self.th1 - self.th0) * self.D
            if 0 <= dkm <= self.D:
                h = float(np.interp(dkm, self.dist, self.z))
                if r < self.R + h * self.k:
                    return (x, y), f
        return None, 1.0

    def terrain_path(self):
        pts = [self.polar(dk, h) for dk, h in zip(self.dist, self.z)]
        top = path_from(pts)
        a0, a1 = self.th0, self.th1
        arc = [(self.cx + self.R * math.sin(a), self.cy - self.R * math.cos(a)) for a in np.linspace(a1, a0, 120)]
        solid = path_from(pts + arc, close=True)
        return top, solid

    def draw(self, c, t):
        k_in = self.sheet(c, t)
        u = self.local(t)
        # whole diagram slowly breathes in; at the end we dive into the summit
        sx_, sy_ = self.ground(self.d_nd)
        dive = expo_inout(ramp(t, self.t1 - 0.62, self.t1)) ** 1.5
        sc = 1.0 + 0.035 * smooth(ramp(u, 0.3, 5.5)) + 5.0 * dive
        c.translate(sx_, sy_)
        c.scale(sc, sc)
        c.translate(-sx_, -sy_)
        with Layer(c, blend=skia.BlendMode.kMultiply):
            self._diagram(c, t)

    def _diagram(self, c, t):
        u = self.local(t)
        f_draw = smoother(ramp(t, self.t0 + 0.35, self.t0 + 1.6))
        top, solid = self.terrain_path()
        # Earth arc
        arc = path_from([(self.cx + self.R * math.sin(a), self.cy - self.R * math.cos(a))
                         for a in np.linspace(self.th0 - 0.03, self.th1 + 0.03, 160)])
        c.drawPath(partial(arc, f_draw), ink(0.85, 2.0))
        # terrain: hatch (revealed left->right) + outline
        b = solid.getBounds()
        c.save()
        c.clipRect(skia.Rect.MakeLTRB(b.left() - 2, b.top() - 2, b.left() + (b.width() + 4) * f_draw, b.bottom() + 4))
        hatch(c, solid, 6.0, -38, INK, 0.42, 1.0)
        c.restore()
        c.drawPath(partial(top, f_draw), ink(0.95, 2.4))
        # underground radial ticks (Earth body)
        for a in np.linspace(self.th0, self.th1, 25):
            x0, y0 = self.cx + (self.R - 6) * math.sin(a), self.cy - (self.R - 6) * math.cos(a)
            x1, y1 = self.cx + (self.R - 22) * math.sin(a), self.cy - (self.R - 22) * math.cos(a)
            c.drawLine(x0, y0, x1, y1, ink(0.35 * f_draw, 1.2))
        # labels
        la = lambda t0_: smooth(ramp(t, t0_, t0_ + 0.3))
        x, y = self.ground(0)
        label(c, "INDIA", x + 6, y + 50, face="mono", size=20, color=INK, alpha=0.8 * la(18.2), tracking=0.2)
        nx, ny = self.ground(self.d_nd)
        a_nd = la(18.5)
        c.drawLine(nx, ny - 12, nx - 40, ny - 120, ink(0.7 * a_nd, 1.3))
        label(c, "NANDA DEVI", nx - 44, ny - 132, face="mono_b", size=20, color=INK, alpha=0.9 * a_nd, tracking=0.16)
        px, py = self.ground(self.D * 0.30)
        label(c, "TIBETAN PLATEAU", px, py - 22, face="mono", size=18, color=INK, alpha=0.75 * la(18.6), tracking=0.2,
              align="center")
        lx, ly = self.ground(self.D)
        label(c, "LOP NUR", lx - 6, ly + 50, face="mono_b", size=20, color=RED, alpha=0.9 * la(18.8), tracking=0.2,
              align="right")
        # missile
        ma = smooth(ramp(t, 18.45, 18.8))
        if ma > 0:
            us = np.linspace(0, 1, 80)
            traj = path_from([self.missile(v) for v in us])
            c.drawPath(traj, dashed(ink(0.45 * ma, 1.4), 3, 7))
            mu = 0.55
            M = self.missile(mu)
            # telemetry arcs (concentric, emitted continuously)
            for kk in range(8):
                ph = ((t - 18.5) * 0.9 + kk / 8) % 1.0
                rr = 18 + ph * 210
                aa = ma * (1 - ph) ** 1.4 * 0.65
                c.drawCircle(M[0], M[1], rr, paint_stroke(col(SIG, aa), 1.6))
            c.drawCircle(M[0], M[1], 9, paint_fill(col(INK, ma)))
            c.drawCircle(M[0], M[1], 3.5, paint_fill(col(PAL["paper"], ma)))
            label(c, "MISSILE", M[0] - 20, M[1] - 24, face="mono_b", size=19, color=INK, alpha=0.85 * ma,
                  tracking=0.16, align="right")
            # listener on the plains + ray 1
            L1 = self.ground(0.0, 40)
            self.antenna(c, L1[0], L1[1], ma)
            f1 = expo_out(ramp(t, self.t_ray1, self.t_hit) * 1.0)
            if f1 > 0 and self.hit[0] is not None:
                hx, hy = self.hit[0]
                ex, ey = lerp(M[0], hx, f1), lerp(M[1], hy, f1)
                c.drawLine(M[0], M[1], ex, ey, paint_stroke(col(SIG, 0.95), 2.6))
                if t >= self.t_hit:
                    # beyond the wall: the signal never arrives
                    c.drawLine(hx, hy, L1[0], L1[1] - 30, dashed(paint_stroke(col(RED, 0.55), 1.6), 5, 7))
                    e = expo_out(ramp(t, self.t_hit, self.t_hit + 0.25))
                    s = 15 * e
                    c.drawLine(hx - s, hy - s, hx + s, hy + s, paint_stroke(col(RED), 3.6))
                    c.drawLine(hx - s, hy + s, hx + s, hy - s, paint_stroke(col(RED), 3.6))
                    c.drawCircle(hx, hy, 28 * e, paint_stroke(col(RED, 0.5 * (1 - e) + 0.15), 1.4))
                    bw = 150
                    c.drawRect(skia.Rect.MakeXYWH(hx + 26, hy + 30, bw, 36), paint_fill(col(PAL["paper"], 0.92)))
                    self.ty_blk.draw(c, t, hx + 34, hy + 58, RED, 1.0)
            # climber-antenna: up the slope to the summit, with altitude readout
            if t >= self.t_climb0 - 0.2:
                kc = smoother(ramp(t, self.t_climb0, self.t_climb1))
                dk = lerp(0.0, self.d_nd, kc)
                hh = float(np.interp(dk, self.dist, self.z))
                if kc >= 1:
                    hh = self.z_nd
                ax, ay = self.ground(dk, 40)
                self.antenna(c, ax, ay, 1.0, scale=1.25)
                # trail
                trail = path_from([self.ground(v, 20) for v in np.linspace(0, dk, 60)])
                c.drawPath(trail, dashed(paint_stroke(col(INK, 0.6), 1.6), 2, 5))
                alt = hh if kc < 1 else 7816
                aa = smooth(ramp(t, self.t_climb0 - 0.2, self.t_climb0 + 0.1))
                label(c, "LISTENING POST ALTITUDE", 86, 800, face="mono", size=18, color=PENCIL, alpha=aa, tracking=0.12)
                label(c, f"{alt:,.0f} m", 86, 852, face="mono_b", size=40, color=INK, alpha=aa)
                f2 = expo_out(ramp(t, self.t_ray2, self.t_ray2 + 0.42))
                if f2 > 0:
                    S = self.ground(self.d_nd, 260)
                    ex, ey = lerp(M[0], S[0], f2), lerp(M[1], S[1], f2)
                    with Layer(c, alpha=0.35, blur=3):
                        c.drawLine(M[0], M[1], ex, ey, paint_stroke(col(SIG), 6))
                    c.drawLine(M[0], M[1], ex, ey, paint_stroke(col(SIG), 3.0))
                    if f2 >= 1:
                        self.ty_clr.draw(c, t, (M[0] + S[0]) / 2 + 26, (M[1] + S[1]) / 2 - 18, SIG, 1.0)
        # title block
        self.ty_title.draw(c, t, 86, 676, INK, 1.0)
        self.ty_sub.draw(c, t, 86, 712, PENCIL, 1.0)
        c.drawLine(86, 730, 86 + 420 * smooth(ramp(t, self.t0 + 0.5, self.t0 + 1.3)), 730, ink(0.6, 1.2))

    def antenna(self, c, x, y, a, scale=1.0):
        s = scale
        p = paint_stroke(col(INK, a), 2.2)
        c.drawLine(x, y + 4, x, y - 34 * s, p)
        c.drawLine(x - 12 * s, y - 22 * s, x + 12 * s, y - 22 * s, p)
        c.drawLine(x - 8 * s, y - 30 * s, x + 8 * s, y - 30 * s, p)
        c.drawCircle(x, y - 36 * s, 3.2 * s, paint_fill(col(INK, a)))

    def overlay(self, img, t):
        if t > self.t1 - 0.5:
            k = smooth(ramp(t, self.t1 - 0.5, self.t1))
            sx_, sy_ = self.ground(self.d_nd)
            img = fx.radial_blur(img, sx_, sy_, 0.12 * k, 8)
        return img
