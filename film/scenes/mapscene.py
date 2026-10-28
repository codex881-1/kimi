"""Strategic night map on real relief (DEM z7): a 3D map camera (heading / tilt / zoom)
renders the relief through a homography; markers, labels, arcs and rings live in map space."""
from __future__ import annotations

import json
import math
from functools import lru_cache

import cv2
import numpy as np
import skia

from engine.core import (ROOT, W, H, PAL, col, mix, paint_fill, paint_stroke, Layer, expo_out, expo_inout, smooth,
                         smoother, clamp01, ramp, lerp, draw_text, glide, silk, fbm, quart_inout, cubic_inout,
                         image_from_float, text_width)
from engine.kit import (Scene, SAMP, CACHE, path_from, smooth_path, glow_dot, glow_stroke, label, leader, dashed,
                        partial, brackets, point_at)
from engine.typer import Typer
from engine.sfx import sfx
from engine import terrain as T
from engine import fx

ND = (79.9708, 30.3758)
LOP = (89.80, 40.80)          # Lop Nur test area (approx. ground zero of test 596)
JQ = (100.29, 40.96)          # Shuangchengzi / Jiuquan launch area (approx.)


@lru_cache(None)
def region():
    d = T.dem("region_z7")
    p = CACHE / "relief_night_z7.png"
    if p.exists():
        rel = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32) / 65535.0
    else:
        rel = T.relief_night(d)
        # faint dune grain in the deserts so the Tarim basin is not a void
        h = np.maximum(d.h, 0)
        rng = np.random.default_rng(5)
        n = cv2.GaussianBlur(rng.standard_normal(h.shape).astype(np.float32), (0, 0), 1.2)
        n = cv2.GaussianBlur(n, (0, 0), 0.5) * 3.0
        yy, xx = np.mgrid[0:h.shape[0], 0:h.shape[1]].astype(np.float32)
        dunes = np.sin(xx * 0.55 + yy * 0.18 + n * 2.0) * 0.5 + 0.5
        low = np.clip((1600 - h) / 600, 0, 1) * (h > 1)
        rel = rel + (dunes * 0.018 + n * 0.006)[..., None] * low[..., None] * np.array([1.0, 0.95, 0.85], np.float32)
        # oceans (Bay of Bengal) - deep ink
        sea = (d.h <= 0)[..., None]
        rel = np.where(sea, np.array([0.012, 0.018, 0.026], np.float32), rel)
        rel = np.clip(rel, 0, 1).astype(np.float32)
        cv2.imwrite(str(p), (rel[..., ::-1] * 65535).astype(np.uint16))
    return d, image_from_float(rel)


@lru_cache(None)
def rivers():
    d = json.loads((ROOT / "data" / "vec" / "rivers.geojson").read_text())
    out = {}

    def flat(g):
        if not g:
            return []
        if isinstance(g[0], (int, float)):
            return [g]
        r = []
        for x in g:
            r += flat(x)
        return r

    for f in d["features"]:
        n = f["properties"].get("name") or ""
        if n in ("Ganges", "Yamuna", "Ghäghara", "Gandak", "Indus", "Sutlej", "Brahmaputra", "Yarlung"):
            g = f["geometry"]
            parts = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
            out.setdefault(n, []).extend(parts)
    return out


class MapCam:
    """Camera over the map plane. Map px (x east, y south) -> world (e=x, n=-y, z=up)."""

    def __init__(self, tx, ty, scale, heading, tilt=0.0, fov=38.0, shake=(0.0, 0.0), cx=W / 2, cy=H / 2):
        self.f = (W / 2) / math.tan(math.radians(fov) / 2)
        dist = self.f / scale
        self.scale = scale
        th = math.radians(tilt)
        hd = math.radians(heading)
        fwd_h = np.array([math.sin(hd), math.cos(hd)])
        tgt = np.array([tx, -ty, 0.0])
        pos = tgt + np.array([-fwd_h[0] * dist * math.sin(th), -fwd_h[1] * dist * math.sin(th), dist * math.cos(th)])
        self.cam = T.RidgeCamera(pos, heading, -(90 - tilt) + 1e-4, fov, W, H, cx=cx + shake[0], cy=cy + shake[1])

    def p(self, x, y, z=0.0):
        sx, sy, dz = self.cam.project(np.asarray(x, np.float64), -np.asarray(y, np.float64), np.asarray(z, np.float64))
        return sx, sy

    def depth(self, x, y, z=0.0):
        return self.cam.project(np.asarray(x, np.float64), -np.asarray(y, np.float64), np.asarray(z, np.float64))[2]

    def draw_image(self, c, img, rect=None, alpha=1.0):
        x0, y0, x1, y1 = rect or (0, 0, img.width(), img.height())
        src = [skia.Point(x0, y0), skia.Point(x1, y0), skia.Point(x1, y1), skia.Point(x0, y1)]
        dx, dy = self.p([x0, x1, x1, x0], [y0, y0, y1, y1])
        dst = [skia.Point(float(a), float(b)) for a, b in zip(dx, dy)]
        m = skia.Matrix()
        if not m.setPolyToPoly(src, dst):
            return
        c.save()
        c.concat(m)
        p = skia.Paint()
        p.setAlphaf(alpha)
        c.drawImageRect(img, skia.Rect.MakeLTRB(x0, y0, x1, y1), skia.Rect.MakeLTRB(x0, y0, x1, y1), SAMP, p)
        c.restore()


def geo_label(c, mc, d, lonlat, text, size=30, color="steel", alpha=1.0, tracking=0.32, face="sans_med", dy=0):
    x, y = d.px(*lonlat)
    sx, sy = mc.p(x, y)
    draw_text(c, text, float(sx), float(sy) + dy, face, size, paint_fill(col(color, alpha)), tracking, "center")


def graticule(c, mc, d, alpha):
    if alpha <= 0.01:
        return
    p = paint_stroke(col("steel", 0.10 * alpha), 1.0)
    for lat in range(25, 46, 5):
        lons = np.linspace(72, 104, 60)
        xs, ys = zip(*[d.px(lo, lat) for lo in lons])
        sx, sy = mc.p(np.array(xs), np.array(ys))
        c.drawPath(path_from(list(zip(sx, sy))), p)
    for lon in range(75, 104, 5):
        lats = np.linspace(22.5, 46, 60)
        xs, ys = zip(*[d.px(lon, la) for la in lats])
        sx, sy = mc.p(np.array(xs), np.array(ys))
        c.drawPath(path_from(list(zip(sx, sy))), p)


class MapChina(Scene):
    """S2-S4: 'built to spy on China' -> Lop Nur 1964 -> missile telemetry."""
    theme = "night"
    name = "map"

    def __init__(self, t0, t1, start_pt=(W / 2, H * 0.56)):
        super().__init__(t0, t1)
        self.start_pt = start_pt
        self.d, self.img = region()
        d = self.d
        self.a = d.px(*ND)
        self.b = d.px(*LOP)
        self.j = d.px(*JQ)
        ax, ay = self.a
        bx, by = self.b
        self.heading = math.degrees(math.atan2(bx - ax, -(by - ay)))
        self.t_bomb = 12.73
        self.ty_india = Typer("INDIA", 8.05, face="sans_med", size=30, cps=14, seed=31, tracking=0.35, gain_db=-4,
                              pan=-0.2, cursor=False)
        self.ty_china = Typer("CHINA", 8.90, face="sans_bold", size=40, cps=12, seed=32, tracking=0.38, gain_db=-1,
                              pan=0.15, cursor=False)
        self.ty_cn = Typer("\u4e2d\u56fd", 9.35, face="cjk", size=34, cps=6, seed=33, gain_db=-3, pan=0.15, cursor=False)
        self.ty_date = Typer("16 OCT 1964  \u00b7  TEST 596", 10.45, size=24, cps=20, seed=34, gain_db=-3, pan=0.2)
        self.ty_lop = Typer("LOP NUR", 13.42, face="sans_bold", size=34, cps=13, seed=35, tracking=0.3, gain_db=-2,
                            pan=0.2, cursor=False)
        self.ty_lop_cn = Typer("\u7f57\u5e03\u6cca", 13.95, face="cjk", size=28, cps=7, seed=36, gain_db=-4, pan=0.2,
                               cursor=False)
        self.ty_msl = Typer("MISSILE TESTS", 14.55, size=24, cps=20, seed=37, gain_db=-3, pan=0.35)
        self.ty_tel = Typer("RADIO TELEMETRY", 16.62, size=24, cps=22, seed=38, gain_db=-3, pan=0.3)
        sfx(7.45, "whoosh_soft", gain_db=-6, dur=1.0)
        sfx(8.0, "line_draw", gain_db=-14, dur=1.6)
        sfx(self.t_bomb - 0.06, "nuke", gain_db=0.0)
        sfx(10.0, "whoosh_soft", gain_db=-10, dur=1.8)
        sfx(14.4, "whoosh_soft", gain_db=-8, dur=1.6)
        sfx(14.75, "missile", gain_db=-9, dur=1.9)
        self.ring_times = [14.9 + 0.30 * k for k in range(10)]
        for k, rt in enumerate(self.ring_times):
            sfx(rt, "blip", gain_db=-13 - (k % 3) * 1.5, pan=0.3, freq=1450 + 160 * (k % 3))

    # ------------------------------------------------------------ camera
    def camera(self, t):
        ax, ay = self.a
        bx, by = self.b
        jx, jy = self.j
        mx, my = (ax + bx) / 2, (ay + by) / 2
        # key poses
        k1 = glide(ramp(t, 7.45, 9.95))          # pull out to the whole axis
        k2 = glide(ramp(t, 9.95, 12.35))         # fly to Lop Nur
        k3 = glide(ramp(t, 14.3, 16.9))          # tilt + reveal launch site
        k4 = glide(ramp(t, 16.6, 17.9))          # look back south
        tx = lerp(ax, mx - 40, k1)
        ty = lerp(ay, my, k1)
        sc = math.exp(lerp(math.log(5.0), math.log(0.66), k1))
        hd = self.heading + 9 * (1 - k1)
        tx, ty = lerp(tx, bx, k2), lerp(ty, by + 40, k2)
        sc = math.exp(lerp(math.log(sc), math.log(1.55), k2))
        # slow push during the blast
        sc *= 1 + 0.05 * smooth(ramp(t, 12.3, 14.3))
        ex, ey = (bx * 0.55 + jx * 0.45), (by * 0.55 + jy * 0.45) + 150
        tx, ty = lerp(tx, ex, k3), lerp(ty, ey, k3)
        sc = math.exp(lerp(math.log(sc), math.log(0.80), k3))
        tilt = 46 * k3
        hd = lerp(hd, self.heading - 18, k3)
        tx, ty = lerp(tx, lerp(bx, ax, 0.32), k4), lerp(ty, lerp(by, ay, 0.32), k4)
        sc = math.exp(lerp(math.log(sc), math.log(0.62), k4))
        # shake after the blast
        sh = 0.0
        if t > self.t_bomb:
            e = math.exp(-(t - self.t_bomb) * 3.2)
            sh = 22 * e
        shake = (sh * fbm(t * 9, 3.1), sh * fbm(t * 9, 7.7))
        # the amber dot starts exactly where the ice core was (match cut), then eases to centre
        m0 = 1 - smooth(ramp(t, 7.45, 9.2))
        ccx = lerp(W / 2, self.start_pt[0], m0)
        ccy = lerp(H * 0.56, self.start_pt[1], m0)
        return MapCam(tx, ty, sc, hd, tilt, shake=shake, cx=ccx, cy=ccy)

    def missile_pos(self, u):
        """u in [0,1] along the Jiuquan -> Lop Nur arc (map px, z up)."""
        jx, jy = self.j
        bx, by = self.b
        x = lerp(jx, bx, u)
        y = lerp(jy, by, u)
        z = 470 * 4 * u * (1 - u)
        return x, y, z

    def draw(self, c, t):
        d = self.d
        mc = self.camera(t)
        fade_in = smooth(ramp(t, self.t0, self.t0 + 0.55))
        with Layer(c, alpha=fade_in):
            c.clear(col("night0"))
            mc.draw_image(c, self.img)
            graticule(c, mc, d, 1.0)
            self.draw_rivers(c, mc, 0.5)
            fog = glide(ramp(t, 14.3, 16.9))
            if fog > 0:
                p = skia.Paint()
                p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(0, H * 0.62)],
                            [col("night0", 1.0 * fog).toColor(), col("night0", 0.85 * fog).toColor(),
                             col("night0", 0.0).toColor()], [0.0, 0.45, 1.0]))
                c.drawRect(skia.Rect.MakeWH(W, H), p)
            # region labels
            la = smooth(ramp(t, 8.0, 8.6))
            geo_label(c, mc, d, (85.0, 33.4), "TIBETAN PLATEAU", 22, "steel", 0.55 * la, 0.42, "mono")
            geo_label(c, mc, d, (84.0, 39.6), "XINJIANG", 22, "steel", 0.55 * smooth(ramp(t, 9.0, 9.6)), 0.42, "mono")
            # axis line ND -> LOP
            ax, ay = self.a
            bx, by = self.b
            f = smooth(ramp(t, 8.0, 9.8))
            if f > 0:
                xs = np.linspace(ax, bx, 80)
                ys = np.linspace(ay, by, 80)
                sx, sy = mc.p(xs, ys)
                path = path_from(list(zip(sx, sy)))
                glow_stroke(c, partial(path, f), "snow", 1.4, 0.5, glow=5, glow_alpha=0.35)
                c.drawPath(partial(path, f), dashed(paint_stroke(col("snow", 0.95), 2.2), 4, 9, -t * 18))
                # distance tag
                if t > 9.2 and t < 14.2:
                    km = 1455 * smooth(ramp(t, 9.2, 9.9))
                    mx_, my_ = mc.p((ax + bx) / 2, (ay + by) / 2)
                    a_ = smooth(ramp(t, 9.2, 9.5)) * (1 - smooth(ramp(t, 11.0, 11.6)))
                    label(c, f"{km:,.0f} KM", float(mx_) - 22, float(my_) + 8, size=22, color="snow", alpha=0.9 * a_, align="right")
            # Nanda Devi marker
            sx, sy = mc.p(ax, ay)
            glow_dot(c, float(sx), float(sy), 4.5, "amber", 1.0, 6)
            self.ty_india.draw(c, t, float(sx), float(sy) + 64, "steel", 0.9, align="center")
            # CHINA label, placed on the plateau north of the border
            cx_, cy_ = mc.p(*d.px(88.6, 35.6))
            self.ty_china.draw(c, t, float(cx_), float(cy_), "red", 0.95, align="center")
            self.ty_cn.draw(c, t, float(cx_), float(cy_) + 46, "red", 0.75, align="center")
            # Lop Nur
            self.draw_lop(c, mc, t)
            # missile arc + telemetry
            self.draw_missile(c, mc, t)
            # HUD: live coordinate readout
            lon, lat = d.lonlat(*self.cam_target(t))
            hud_a = 0.55 * smooth(ramp(t, 7.8, 8.4))
            label(c, f"{lat:05.2f}\u00b0N  {lon:06.2f}\u00b0E", 70, H - 330, size=19, color="steel", alpha=hud_a)
            label(c, "RELIEF: SRTM / AWS TERRAIN", W - 70, H - 330, size=19, color="steel", alpha=hud_a * 0.8,
                  align="right")
            brackets(c, 54, 250, W - 108, H - 560, "steel", 0.35 * hud_a / 0.55)
        # seismograph strip
        self.draw_seismo(c, t)

    def cam_target(self, t):
        ax, ay = self.a
        bx, by = self.b
        k1 = glide(ramp(t, 7.45, 9.95))
        k2 = glide(ramp(t, 9.95, 12.35))
        return lerp(lerp(ax, (ax + bx) / 2, k1), bx, k2), lerp(lerp(ay, (ay + by) / 2, k1), by, k2)

    def draw_rivers(self, c, mc, a):
        d = self.d
        for name, parts in rivers().items():
            for part in parts:
                pts = np.array(part)
                xs, ys = zip(*[d.px(lo, la) for lo, la in pts])
                sx, sy = mc.p(np.array(xs), np.array(ys))
                c.drawPath(path_from(list(zip(sx, sy))), paint_stroke(col((0.35, 0.5, 0.62), 0.28 * a), 1.2))

    def draw_lop(self, c, mc, t):
        bx, by = self.b
        sx, sy = mc.p(bx, by)
        sx, sy = float(sx), float(sy)
        pre = smooth(ramp(t, 10.2, 10.6))
        if t < self.t_bomb:
            # target reticle waiting
            if pre > 0:
                r = 38 + 6 * math.sin(t * 4)
                c.drawCircle(sx, sy, r, paint_stroke(col("red", 0.8 * pre), 1.5))
                c.drawCircle(sx, sy, 3.5, paint_fill(col("red", pre)))
                for k in range(4):
                    a_ = k * math.pi / 2
                    c.drawLine(sx + math.cos(a_) * (r - 8), sy + math.sin(a_) * (r - 8), sx + math.cos(a_) * (r + 10),
                               sy + math.sin(a_) * (r + 10), paint_stroke(col("red", 0.8 * pre), 1.5))
        else:
            u = t - self.t_bomb
            # flash core
            fl = math.exp(-u * 2.4)
            glow_dot(c, sx, sy, 10 + 30 * fl, (1.0, 0.86, 0.66), 1.6 * fl + 0.35, 9)
            # light rays
            if fl > 0.05:
                with Layer(c, alpha=fl, blend=skia.BlendMode.kPlus):
                    for k in range(14):
                        ang = k / 14 * 2 * math.pi + 0.2
                        L = 520 * (0.6 + 0.4 * math.sin(k * 7.3)) * (0.4 + 0.6 * expo_out(u / 0.4))
                        p = skia.Paint(AntiAlias=True)
                        p.setShader(skia.GradientShader.MakeLinear(
                            [skia.Point(sx, sy), skia.Point(sx + math.cos(ang) * L, sy + math.sin(ang) * L)],
                            [col((1, 0.8, 0.6), 0.35).toColor(), col((1, 0.5, 0.3), 0.0).toColor()]))
                        p.setStrokeWidth(3 + 4 * fl)
                        p.setStyle(skia.Paint.kStroke_Style)
                        c.drawLine(sx, sy, sx + math.cos(ang) * L, sy + math.sin(ang) * L, p)
            # expanding pressure rings in the map plane
            for k, delay in enumerate((0.0, 0.18, 0.42)):
                uu = u - delay
                if uu <= 0:
                    continue
                R = 820 * expo_out(uu / 2.4) * (1 - 0.18 * k)
                a_ = math.exp(-uu * 1.1) * (0.9 - 0.25 * k)
                n = 120
                ang = np.linspace(0, 2 * math.pi, n)
                rx = bx + np.cos(ang) * R / max(mc.scale, 1e-3)
                ry = by + np.sin(ang) * R / max(mc.scale, 1e-3)
                px, py = mc.p(rx, ry)
                ring = path_from(list(zip(px, py)), close=True)
                glow_stroke(c, ring, "red", 2.2 - 0.5 * k, a_, glow=10, glow_alpha=0.7)
            # ground-zero mark
            c.drawCircle(sx, sy, 5, paint_fill(col("red")))
        late = 1 - smooth(ramp(t, 14.2, 14.7))
        self.ty_date.draw(c, t, sx + 58, sy - 64, "red", 0.95 * pre * late)
        self.ty_lop.draw(c, t, sx + 58, sy + 16, "snow", 1.0)
        self.ty_lop_cn.draw(c, t, sx + 58, sy + 58, "steel", 0.85 * late)

    def draw_missile(self, c, mc, t):
        t_l, t_e = 14.65, 16.55
        if t < t_l - 0.2:
            return
        jx, jy = self.j
        sx, sy = mc.p(jx, jy)
        a0 = smooth(ramp(t, t_l - 0.2, t_l + 0.2))
        c.drawCircle(float(sx), float(sy), 4, paint_fill(col("ice", a0)))
        # full arc ghost
        us = np.linspace(0, 1, 90)
        P = [self.missile_pos(u) for u in us]
        px, py = mc.p(*map(np.array, zip(*P)))
        ghost = path_from(list(zip(px, py)))
        c.drawPath(ghost, dashed(paint_stroke(col("ice", 0.25 * a0), 1.2), 4, 8))
        # ground track
        gx, gy = mc.p(np.array([p[0] for p in P]), np.array([p[1] for p in P]))
        c.drawPath(path_from(list(zip(gx, gy))), dashed(paint_stroke(col("steel", 0.25 * a0), 1.0), 2, 6))
        u = smoother(ramp(t, t_l, t_e))
        # trail
        if u > 0:
            trail = partial(ghost, max(0, u - 0.35), u)
            glow_stroke(c, trail, "ice", 3.0, 1.0, glow=9, glow_alpha=0.9)
            mx_, my_, mz_ = self.missile_pos(u)
            qx, qy = mc.p(mx_, my_, mz_)
            qx, qy = float(qx), float(qy)
            # telemetry rings (radio pulses)
            for rt in self.ring_times:
                uu = t - rt
                if uu < 0 or uu > 2.2:
                    continue
                pos_u = smoother(ramp(rt, t_l, t_e))
                ox, oy, oz = self.missile_pos(pos_u)
                rx, ry = mc.p(ox, oy, oz)
                R = 30 + 520 * expo_out(uu / 2.2)
                a_ = (1 - uu / 2.2) ** 1.6 * 0.7
                c.drawCircle(float(rx), float(ry), R, paint_stroke(col("ice", a_), 1.4))
            if u < 1:
                glow_dot(c, qx, qy, 4.5, "ice", 1.0, 6)
        self.ty_msl.draw(c, t, float(sx) - 30, float(sy) + 58, "ice", 0.95 * a0, align="right")
        if t > 16.5:
            ex, ey = mc.p(*self.b)
            mx_, my_, mz_ = self.missile_pos(1.0)
            self.ty_tel.draw(c, t, float(ex) - 40, float(ey) + 64, "ice", 0.95, align="right")

    def draw_seismo(self, c, t):
        a = smooth(ramp(t, 10.6, 11.0)) * (1 - smooth(ramp(t, 14.3, 14.8)))
        if a <= 0:
            return
        y0 = H - 410
        x0, x1 = 120, W - 120
        n = 300
        xs = np.linspace(x0, x1, n)
        # scrolling trace: time maps right-to-left, newest sample at the right edge
        tt = t - (x1 - xs) / (x1 - x0) * 2.4
        amp = np.array([3 * fbm(v * 22, 1.3) for v in tt])
        hit = tt > self.t_bomb
        dt = np.where(hit, tt - self.t_bomb, 0)
        big = hit * 120 * np.exp(-dt * 2.2) * np.sin(dt * 55) * np.minimum(1, dt * 30)
        ys = y0 + amp + big
        c.drawLine(x0, y0, x1, y0, paint_stroke(col("steel", 0.2 * a), 1.0))
        glow_stroke(c, path_from(list(zip(xs, ys))), "red" if t > self.t_bomb else "steel", 1.6, 0.85 * a, glow=5,
                    glow_alpha=0.5)
        label(c, "SEISMIC", x0, y0 - 140, size=17, color="steel", alpha=0.7 * a)

    def overlay(self, img, t):
        if t < 7.95:
            k = 1 - smooth(ramp(t, 7.45, 7.95))
            img = fx.radial_blur(img, self.start_pt[0], self.start_pt[1], 0.10 * k, 8)
        if self.t_bomb <= t < self.t_bomb + 2.4:
            u = t - self.t_bomb
            mc = self.camera(t)
            sx, sy = mc.p(*self.b)
            img = fx.shockwave(img, float(sx), float(sy), 1300 * expo_out(u / 2.2), width=70,
                               amp=26 * math.exp(-u * 1.4))
        return img

    def post(self, t):
        p = dict(bloom=0.62, bloom_thr=0.5, halation=0.2, vignette=0.34, grain=0.034)
        if t >= self.t_bomb - 0.02:
            u = t - self.t_bomb
            p["exposure"] = 1 + 1.4 * math.exp(-u * 6) + 0.15 * math.exp(-u * 1.2)
            p["bloom"] = 0.62 + 1.2 * math.exp(-u * 3)
            p["chroma"] = 0.9 + 3.0 * math.exp(-u * 3)
        return p

    def mblur(self, t):
        # fast camera legs get real motion blur
        if 10.2 < t < 12.1 or 14.5 < t < 16.6:
            return 0.5
        return 0.0
