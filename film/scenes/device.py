"""S7-S8: the device as an exploded patent drawing (paper world)."""
from __future__ import annotations

import math

import numpy as np
import skia

from engine.core import (W, H, PAL, col, mix, paint_fill, paint_stroke, Layer, expo_out, expo_inout, smooth,
                         smoother, clamp01, ramp, lerp, glide, fbm, spring, back_out, cubic_inout)
from engine.kit import (path_from, smooth_path, partial, dashed, label, glow_dot, point_at)
from engine.typer import Typer
from engine.sfx import sfx
from scenes.paper import PaperScene, INK, PENCIL, SIG, RED, ink, hatch

AMB = PAL["amber"]
PAPER = PAL["paper"]


class Iso:
    """Orthographic 3D -> 2D with yaw (about z-up) and pitch."""

    def __init__(self, yaw, pitch, scale, cx, cy):
        self.cy_, self.sy_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        self.cp, self.sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        self.s, self.cx, self.cy = scale, cx, cy

    def p(self, x, y, z):
        xr = x * self.cy_ - y * self.sy_
        yr = x * self.sy_ + y * self.cy_
        # view looks along +yr pitched DOWN: higher z is nearer the viewer
        depth = yr * self.cp - z * self.sp
        up = z * self.cp + yr * self.sp
        return self.cx + xr * self.s, self.cy - up * self.s, depth

    def n_dot(self, n):
        """facing factor of a world normal toward the viewer (>0 visible)."""
        nx, ny, nz = n
        yr = nx * self.sy_ + ny * self.cy_
        return -(yr * self.cp - nz * self.sp)


LIGHT = np.array([-0.5, -0.6, 0.8])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


class Face:
    __slots__ = ("pts", "n", "kind", "comp")

    def __init__(self, pts, n, kind, comp):
        self.pts, self.n, self.kind, self.comp = pts, n, kind, comp


def box(c3, size, comp, kind="box"):
    cx, cy, cz = c3
    sx, sy, sz = (v / 2 for v in size)
    V = [(cx + a * sx, cy + b * sy, cz + d * sz) for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)]
    def v(a, b, d):
        return V[((a > 0) << 2) | ((b > 0) << 1) | (d > 0)]
    F = []
    for axis in range(3):
        for sgn in (-1, 1):
            n = [0, 0, 0]
            n[axis] = sgn
            if axis == 0:
                q = [v(sgn, -1, -1), v(sgn, 1, -1), v(sgn, 1, 1), v(sgn, -1, 1)]
            elif axis == 1:
                q = [v(-1, sgn, -1), v(1, sgn, -1), v(1, sgn, 1), v(-1, sgn, 1)]
            else:
                q = [v(-1, -1, sgn), v(1, -1, sgn), v(1, 1, sgn), v(-1, 1, sgn)]
            F.append(Face(q, tuple(n), kind, comp))
    return F


def cylinder(c3, r, h, comp, n=36, kind="cyl"):
    cx, cy, cz = c3
    F = []
    angs = np.linspace(0, 2 * math.pi, n + 1)
    for i in range(n):
        a0, a1 = angs[i], angs[i + 1]
        am = (a0 + a1) / 2
        q = [(cx + r * math.cos(a0), cy + r * math.sin(a0), cz), (cx + r * math.cos(a1), cy + r * math.sin(a1), cz),
             (cx + r * math.cos(a1), cy + r * math.sin(a1), cz + h), (cx + r * math.cos(a0), cy + r * math.sin(a0), cz + h)]
        F.append(Face(q, (math.cos(am), math.sin(am), 0.0), kind + "_side", comp))
    top = [(cx + r * math.cos(a), cy + r * math.sin(a), cz + h) for a in angs[:-1]]
    bot = [(cx + r * math.cos(a), cy + r * math.sin(a), cz) for a in angs[:-1]]
    F.append(Face(top, (0, 0, 1), kind + "_cap", comp))
    F.append(Face(bot[::-1], (0, 0, -1), kind + "_cap", comp))
    return F


class Device(PaperScene):
    name = "device"

    def __init__(self, t0, t1):
        super().__init__(t0, t1)
        self.ty_title = Typer("THE DEVICE", 29.30, face="type_b", size=36, cps=14, seed=91, gain_db=-1,
                              typewriter=True, cursor=False)
        self.ty_sub = Typer("RECONSTRUCTED FROM PUBLISHED ACCOUNTS", 29.85, face="type", size=19, cps=34, seed=92,
                            gain_db=-8, typewriter=True, cursor=False)
        self.ty_ant = Typer("ANTENNA", 30.35, face="mono_b", size=21, cps=18, seed=93, tracking=0.14, gain_db=-4, cursor=False)
        self.ty_rad = Typer("TRANSCEIVER \u00d7 2", 31.10, face="mono_b", size=21, cps=20, seed=94, tracking=0.1, gain_db=-4, cursor=False)
        self.ty_gen = Typer("SNAP-19C GENERATOR", 31.95, face="mono_b", size=21, cps=22, seed=95, tracking=0.1, gain_db=-4, cursor=False)
        self.ty_pu = Typer("7 \u00d7 PLUTONIUM-238", 32.80, face="mono_b", size=21, cps=20, seed=96, tracking=0.1, gain_db=-3, cursor=False)
        self.ty_kg = Typer("TOTAL \u2248 57 kg", 33.35, face="type_b", size=24, cps=16, seed=97, gain_db=-5,
                           typewriter=True, cursor=False)
        self.ty_bat = Typer("BATTERY", 34.45, face="mono_b", size=20, cps=16, seed=98, tracking=0.14, gain_db=-5, cursor=False)
        self.ty_heat = Typer("HEAT", 37.15, face="mono_b", size=22, cps=12, seed=99, tracking=0.2, gain_db=-4, cursor=False)
        self.ty_tc = Typer("THERMOCOUPLES", 37.45, face="mono_b", size=19, cps=26, seed=100, tracking=0.1, gain_db=-6, cursor=False)
        self.ty_el = Typer("ELECTRICITY", 37.85, face="mono_b", size=22, cps=20, seed=101, tracking=0.16, gain_db=-3, cursor=False)
        sfx(t0 - 0.05, "paper_slide", gain_db=-3)
        sfx(30.3, "pencil", gain_db=-14, dur=0.6)
        sfx(31.0, "pencil", gain_db=-14, dur=0.7)
        sfx(31.95, "pencil", gain_db=-14, dur=0.7)
        sfx(32.35, "capsules", gain_db=-6)
        sfx(34.5, "freeze", gain_db=-7, dur=0.9)
        sfx(34.95, "power_down", gain_db=-8)
        sfx(35.5, "warm_swell", gain_db=-7, dur=1.6)
        sfx(36.9, "whoosh_soft", gain_db=-12, dur=0.8)
        sfx(37.85, "electric", gain_db=-10, dur=1.2)
        for k in range(5):
            sfx(38.0 + k * 0.22, "blip", gain_db=-16, freq=1500, pan=-0.3)

    slide_in = 0.46

    def sheet(self, c, t, drift_scale=1.0):
        u = self.local(t)
        k = expo_out(clamp01(u / self.slide_in))
        ox = (1 - k) * W * 1.1
        rot = (1 - spring(u, 0.6, 11)) * -2.5
        if k < 1:
            with Layer(c, alpha=0.5 * k, blur=40):
                c.drawRect(skia.Rect.MakeLTRB(ox - 40, -40, W + 300, H + 40), paint_fill(col((0, 0, 0))))
        c.translate(W / 2 + ox, H / 2)
        c.rotate(rot)
        c.translate(-W / 2, -H / 2)
        from engine.kit import draw_paper
        draw_paper(c, t, cam_x=6 * fbm(t * 0.2, 4.4), cam_y=6 * fbm(t * 0.2, 9.1))
        return k

    # -------------------------------------------------------------- geometry
    def faces(self, t):
        F = []
        # transceivers
        F += box((-30, 6, 10), (34, 22, 20), "rad1")
        F += box((-30, 6, 36), (34, 22, 20), "rad2")
        # generator body + fins
        F += cylinder((22, 4, 0), 9.5, 30, "gen")
        for k in range(6):
            a = math.radians(30 + k * 60)
            cxk, cyk = 22 + math.cos(a) * 15.5, 4 + math.sin(a) * 15.5
            # radial fin approximated by a rotated thin box: build it then rotate in-plane
            fin = box((0, 0, 15), (8, 1.2, 26), "gen", kind="fin")
            for f in fin:
                f.pts = [(22 + (x * math.cos(a) - y * math.sin(a)) + math.cos(a) * 13.0,
                          4 + (x * math.sin(a) + y * math.cos(a)) + math.sin(a) * 13.0, z) for x, y, z in f.pts]
                nx, ny, nz = f.n
                f.n = (nx * math.cos(a) - ny * math.sin(a), nx * math.sin(a) + ny * math.cos(a), nz)
            F += fin
        # antenna (packed for the climb): mast lying above the radios, elements across it
        F += box((-14, 6, 62), (72, 1.8, 1.8), "ant", kind="elem")
        for k in range(5):
            xk = -44 + k * 14
            L = 26 - 3 * k
            F += box((xk, 6, 62), (1.4, L, 1.4), "ant", kind="elem")
        return F

    def capsule_pos(self, t):
        """7 capsules lifted out of the generator into a hex cluster."""
        e = expo_out(ramp(t, 32.35, 33.1))
        out = []
        for i in range(7):
            if i == 0:
                dx, dy = 0.0, 0.0
            else:
                a = (i - 1) * math.pi / 3
                dx, dy = math.cos(a) * 4.2, math.sin(a) * 4.2
            out.append((22 + dx * (0.35 + 0.65 * e), 4 + dy * (0.35 + 0.65 * e), 22 + 30 * e))
        return out, e

    def draw(self, c, t):
        self.sheet(c, t)
        u = self.local(t)
        yaw = 34 + 8 * smooth(ramp(u, 0, self.t1 - self.t0)) + 2 * math.sin(t * 0.6)
        zoom = 1.0 + 0.03 * smooth(ramp(u, 0, 6))
        iso = Iso(yaw, 26, 7.9 * zoom, W / 2 + 10, 1240)
        dim = smooth(ramp(t, 36.75, 37.2))
        prog = {
            "ant": smoother(ramp(t, 30.25, 30.95)),
            "rad1": smoother(ramp(t, 30.95, 31.45)),
            "rad2": smoother(ramp(t, 31.15, 31.65)),
            "gen": smoother(ramp(t, 31.9, 32.6)),
        }
        with Layer(c, blend=skia.BlendMode.kMultiply):
            self.draw_title(c, t)
        with Layer(c, blend=skia.BlendMode.kMultiply, alpha=1 - 0.7 * dim, blur=3.0 * dim):
            self.draw_model(c, t, iso, prog)
            self.draw_labels(c, t, iso, prog)
        with Layer(c, blend=skia.BlendMode.kMultiply):
            self.draw_battery(c, t)
        with Layer(c, alpha=1 - 0.6 * dim):
            self.draw_capsules(c, t, iso)
        self.draw_inset(c, t, iso)
        # final dive into the warm core: the sheet goes dark around the glow
        k = smooth(ramp(t, self.t1 - 0.55, self.t1))
        if k > 0:
            gx, gy, _ = iso.p(22, 4, 55)
            p = skia.Paint()
            p.setShader(skia.GradientShader.MakeRadial(skia.Point(gx, gy), 900 * (1 - 0.8 * k) + 40,
                        [col((0, 0, 0), 0.0).toColor(), col((0.01, 0.012, 0.016), k).toColor()], [0.0, 0.35]))
            c.drawRect(skia.Rect.MakeWH(W, H), p)

    def draw_title(self, c, t):
        self.ty_title.draw(c, t, 86, 1560, INK, 1.0)
        self.ty_sub.draw(c, t, 86, 1594, PENCIL, 1.0)
        c.drawLine(86, 1530, 86 + 560 * smooth(ramp(t, 29.5, 30.2)), 1530, ink(0.6, 1.2))

    def draw_model(self, c, t, iso, prog):
        F = [f for f in self.faces(t) if prog.get(f.comp, 0) > 0]
        proj = []
        for f in F:
            vis = iso.n_dot(f.n)
            if vis <= 0.001:
                continue
            P = [iso.p(*q) for q in f.pts]
            depth = sum(p[2] for p in P) / len(P)
            proj.append((depth, f, P, vis))
        proj.sort(key=lambda r: -r[0])
        for depth, f, P, vis in proj:
            pr = prog[f.comp]
            pts2 = [(p[0], p[1]) for p in P]
            path = path_from(pts2, close=True)
            lit = float(np.dot(np.array(f.n), LIGHT))
            fill_a = smooth(ramp(pr, 0.55, 1.0))
            if fill_a > 0:
                tone = mix(PAPER, (0.80, 0.76, 0.68), clamp01(0.55 - lit * 0.6))
                c.drawPath(path, paint_fill(col(tone, fill_a)))
                if lit < -0.05 and f.kind not in ("elem",):
                    hatch(c, path, 4.5, -60, INK, 0.30 * fill_a, 0.8)
            edge_w = 1.5 if f.kind in ("box", "fin") else 1.2
            if f.kind.endswith("_side"):
                shade = clamp01(0.5 - lit * 0.7)
                c.drawPath(path, paint_fill(col(mix(PAPER, (0.62, 0.58, 0.50), shade), fill_a)))
                c.drawPath(partial(path_from([pts2[0], pts2[3]]), pr), ink((0.12 + 0.55 * shade) * fill_a + 0.05, 1.0))
                if vis < 0.22:
                    c.drawPath(partial(path_from([pts2[0], pts2[3]]), pr), ink(0.9, 1.6))
                continue
            c.drawPath(partial(path, pr), ink(0.92, edge_w))
        # dials on the radios (front faces)
        for comp, zc in (("rad1", 10), ("rad2", 36)):
            pr = prog[comp]
            if pr < 0.8:
                continue
            a = smooth(ramp(pr, 0.8, 1.0))
            for k, xo in enumerate((-9, 0, 9)):
                x, y, _ = iso.p(-30 + xo, 6 - 11.2, zc + 3)
                c.drawCircle(x, y, 7 if k < 2 else 4, ink(0.85 * a, 1.3))
                x2, y2, _ = iso.p(-30 + xo, 6 - 11.2, zc - 5)
                c.drawLine(x2 - 5, y2, x2 + 5, y2, ink(0.7 * a, 1.2))

    def draw_capsules(self, c, t, iso):
        caps, e = self.capsule_pos(t)
        if t < 32.3:
            return
        warm = 0.65 + 0.35 * smooth(ramp(t, 35.4, 36.0))
        for (x, y, z) in caps:
            # capsule body
            x0, y0, _ = iso.p(x, y, z)
            x1, y1, _ = iso.p(x, y, z + 6)
            with Layer(c, blend=skia.BlendMode.kMultiply):
                c.drawLine(x0, y0, x1, y1, paint_stroke(col(INK, 0.85), 11))
            c.drawLine(x0, y0, x1, y1, paint_stroke(col((1.0, 0.72, 0.36), 0.95 * e), 7))
            glow_dot(c, (x0 + x1) / 2, (y0 + y1) / 2, 4.5, AMB, intensity=1.0 * e * warm, halo=9)
        if e > 0:
            gx, gy, _ = iso.p(22, 4, 30)
            cx_, cy_, _ = iso.p(22, 4, 52)
            with Layer(c, blend=skia.BlendMode.kMultiply):
                c.drawLine(gx, gy, cx_, cy_ + 30, dashed(ink(0.5 * e, 1.2), 3, 5))
        # heat shimmer above the cluster once plutonium "doesn't die"
        hs = smooth(ramp(t, 35.45, 35.9)) * (1 - smooth(ramp(t, 36.8, 37.1)))
        if hs > 0:
            cx_, cy_, _ = iso.p(22, 4, 58)
            for k in range(3):
                pts = [(cx_ - 24 + 24 * k + 7 * math.sin(v * 0.06 + t * 5 + k), cy_ - v) for v in np.linspace(10, 110, 30)]
                c.drawPath(path_from(pts), paint_stroke(col(AMB, 0.65 * hs * (1 - k * 0.15)), 2.0))

    def draw_labels(self, c, t, iso, prog):
        def lab(ty_, anchor3, dx, dy, color=INK, key=None):
            pr = prog.get(key, 1.0) if key else 1.0
            if t < ty_.t0 - 0.02:
                return
            ax, ay, _ = iso.p(*anchor3)
            bx, by = ax + dx, ay + dy
            f = expo_out(ramp(t, ty_.t0 - 0.12, ty_.t0 + 0.15))
            c.drawLine(ax, ay, lerp(ax, bx, f), lerp(ay, by, f), ink(0.75, 1.2))
            c.drawCircle(ax, ay, 3.2, paint_fill(col(INK, f)))
            c.drawLine(bx, by, bx + (14 if dx > 0 else -14) * f, by, ink(0.75, 1.2))
            ty_.draw(c, t, bx + (20 if dx > 0 else -20), by + 7, color, 1.0, align="left" if dx > 0 else "right")
        lab(self.ty_ant, (-50, 6, 62), -40, -70, key="ant")
        lab(self.ty_rad, (-30, -5, 0), 30, 150, key="rad1")
        lab(self.ty_gen, (31, 4, 12), 70, 120, key="gen")
        caps, e = self.capsule_pos(t)
        if e > 0.2:
            lab(self.ty_pu, (26, 4, 55), 90, -70, color=(0.62, 0.32, 0.08))
        self.ty_kg.draw(c, t, W - 86, 1560, INK, 1.0, align="right")

    def draw_battery(self, c, t):
        a = smooth(ramp(t, 34.0, 34.3)) * (1 - smooth(ramp(t, 36.7, 37.0)))
        if a <= 0:
            return
        x, y, w, h = 610, 1270, 64, 128
        c.drawRoundRect(skia.Rect.MakeXYWH(x, y, w, h), 8, 8, ink(0.9 * a, 2.2))
        c.drawRect(skia.Rect.MakeXYWH(x + w / 2 - 14, y - 10, 28, 10), paint_fill(col(INK, 0.9 * a)))
        lvl = 1 - smoother(ramp(t, 34.5, 35.15))
        for k in range(4):
            on = lvl > (k + 0.5) / 4
            yy = y + h - 12 - (k + 1) * 32
            c.drawRect(skia.Rect.MakeXYWH(x + 10, yy, w - 20, 24),
                       paint_fill(col(INK if on else PAPER, (0.75 if on else 0.0) * a)))
        # frost crystals creeping over it
        fr = smooth(ramp(t, 34.4, 35.2))
        if fr > 0:
            rng = np.random.default_rng(7)
            for k in range(14):
                sx_, sy_ = x + rng.uniform(0, w), y + rng.uniform(0, h)
                L = (10 + rng.uniform(0, 18)) * fr
                for j in range(6):
                    ang = j * math.pi / 3 + k
                    ex, ey = sx_ + math.cos(ang) * L, sy_ + math.sin(ang) * L
                    c.drawLine(sx_, sy_, ex, ey, paint_stroke(col(SIG, 0.75 * a * fr), 1.2))
                    c.drawLine(ex - math.cos(ang + 0.6) * L * 0.3, ey - math.sin(ang + 0.6) * L * 0.3, ex, ey,
                               paint_stroke(col(SIG, 0.5 * a * fr), 1.0))
        if lvl <= 0.01:
            e = expo_out(ramp(t, 35.1, 35.35))
            cx_, cy_ = x + w / 2, y + h / 2
            c.drawLine(cx_ - 30 * e, cy_ - 30 * e, cx_ + 30 * e, cy_ + 30 * e, paint_stroke(col(RED, a), 4))
            c.drawLine(cx_ - 30 * e, cy_ + 30 * e, cx_ + 30 * e, cy_ - 30 * e, paint_stroke(col(RED, a), 4))
        self.ty_bat.draw(c, t, x + w / 2, y + h + 38, INK, a, align="center")

    def draw_inset(self, c, t, iso):
        """Magnifier: generator cross-section - heat -> thermocouples -> electricity."""
        a = smooth(ramp(t, 36.75, 37.1)) * (1 - smooth(ramp(t, 38.55, 38.85)))
        if a <= 0:
            return
        gx, gy, _ = iso.p(22, 4, 15)
        R = 250 * expo_out(ramp(t, 36.75, 37.25))
        cx_, cy_ = W / 2 + 170, 1120
        with Layer(c, alpha=a):
            # tether from generator to the lens
            c.drawLine(gx, gy, cx_ - R * 0.6, cy_ + R * 0.8, ink(0.5, 1.2))
            c.drawCircle(cx_ + 6, cy_ + 10, R, paint_fill(col((0, 0, 0), 0.18)))
            c.drawCircle(cx_, cy_, R, paint_fill(col((0.96, 0.94, 0.89))))
            c.save()
            c.clipPath(skia.Path().addCircle(cx_, cy_, R), skia.ClipOp.kIntersect, True)
            s = R / 250
            # fins + body
            for k in range(6):
                ang = math.radians(30 + k * 60)
                c.save()
                c.translate(cx_, cy_)
                c.rotate(math.degrees(ang))
                c.drawRect(skia.Rect.MakeLTRB(118 * s, -9 * s, 236 * s, 9 * s), ink(0.9, 1.6))
                c.restore()
            c.drawCircle(cx_, cy_, 120 * s, ink(0.95, 2.4))
            # thermocouple ring
            zz = []
            for i in range(121):
                ang = i / 120 * 2 * math.pi
                rr = (78 + (6 if i % 2 else -6)) * s
                zz.append((cx_ + math.cos(ang) * rr, cy_ + math.sin(ang) * rr))
            tc = smooth(ramp(t, 37.4, 37.7))
            c.drawPath(path_from(zz), paint_stroke(col(mix(INK, (0.85, 0.45, 0.15), tc), 0.95), 1.8))
            # heat source + capsules
            p = skia.Paint(AntiAlias=True)
            p.setShader(skia.GradientShader.MakeRadial(skia.Point(cx_, cy_), 56 * s,
                        [col((1.0, 0.72, 0.38)).toColor(), col((0.92, 0.55, 0.22)).toColor()]))
            c.drawCircle(cx_, cy_, 52 * s, p)
            for i in range(7):
                if i == 0:
                    px, py = cx_, cy_
                else:
                    ang = (i - 1) * math.pi / 3
                    px, py = cx_ + math.cos(ang) * 28 * s, cy_ + math.sin(ang) * 28 * s
                c.drawCircle(px, py, 11 * s, paint_fill(col((1.0, 0.93, 0.78))))
            # heat arrows: capsules -> ring
            ha = smooth(ramp(t, 37.1, 37.4))
            for k in range(8):
                ang = k * math.pi / 4 + 0.2
                ph = ((t * 1.4) + k * 0.13) % 1.0
                r0, r1 = 55 * s, 74 * s
                rr = lerp(r0, r1, ph)
                c.drawLine(cx_ + math.cos(ang) * r0, cy_ + math.sin(ang) * r0, cx_ + math.cos(ang) * rr,
                           cy_ + math.sin(ang) * rr, paint_stroke(col(AMB, ha * (1 - ph)), 3.0))
            c.restore()
            c.drawCircle(cx_, cy_, R, ink(0.95, 2.6))
            # wire out of the lens to the radios with current dots
            ea = smooth(ramp(t, 37.8, 38.1))
            if ea > 0:
                rx, ry, _ = iso.p(-30, 6, 46)
                wire = smooth_path([(cx_ - R * 0.95, cy_ - R * 0.2), (cx_ - R - 120, cy_ - 140), (rx + 60, ry - 40), (rx, ry)])
                c.drawPath(partial(wire, ea), paint_stroke(col(SIG, 0.95), 2.4))
                for k in range(8):
                    f = ((t - 37.8) * 0.9 + k / 8) % 1.0
                    if f > ea:
                        continue
                    (px, py), _ = point_at(wire, f)
                    c.drawCircle(px, py, 4.2, paint_fill(col(SIG)))
                # antenna emits
                ax, ay, _ = iso.p(-30, 6, 108)
                for k in range(3):
                    ph = ((t - 38.0) * 1.2 + k / 3) % 1.0
                    if t < 38.0:
                        break
                    c.drawCircle(ax, ay, 14 + ph * 90, paint_stroke(col(SIG, 0.8 * (1 - ph)), 1.8))
            self.ty_heat.draw(c, t, cx_, cy_ + R + 44, (0.72, 0.38, 0.08), 1.0, align="center")
            self.ty_tc.draw(c, t, cx_, cy_ + R + 76, INK, 1.0, align="center")
            self.ty_el.draw(c, t, cx_ - R - 20, cy_ - R + 6, SIG, 1.0, align="right")

    def theme_value(self, t):
        return smooth(ramp(t, self.t0, self.t0 + 0.12)) * (1 - smooth(ramp(t, self.t1 - 0.4, self.t1 - 0.1)))

    def post(self, t):
        k = smooth(ramp(t, self.t1 - 0.5, self.t1))
        return dict(bloom=0.10 + 0.5 * k, bloom_thr=0.85 - 0.3 * k, halation=0.1 * k, chroma=0.55, vignette=0.24,
                    grain=0.028, warmth=0.012)
