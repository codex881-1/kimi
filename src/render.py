#!/usr/bin/env python3
"""SENIOR BOWL — deterministic 1080x1920 motion film renderer."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import skia
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
W, H = 1080, 1920
DURATION = 68.0
IVORY = "#EEE8DA"
MUTED = "#89908B"
RED = "#F05244"
AMBER = "#C9A876"
INK = "#101313"


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ramp(t, start, length):
    return clamp((t - start) / length)


def ease(x):
    x = clamp(x)
    return 1 - (1 - x) ** 4


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def lerp(a, b, x):
    return a + (b - a) * x


def color(value, alpha=1.0):
    if isinstance(value, int):
        c = skia.Color4f(value)
        c.fA *= alpha
        return c
    value = value.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return skia.Color4f(r, g, b, alpha)


def paint(fill=IVORY, alpha=1.0, stroke=0, aa=True):
    p = skia.Paint()
    p.setAntiAlias(aa)
    p.setColor(color(fill, alpha))
    if stroke:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
        p.setStrokeJoin(skia.Paint.kRound_Join)
        p.setStrokeCap(skia.Paint.kRound_Cap)
    return p


def rect(x, y, w, h):
    return skia.Rect.MakeXYWH(x, y, w, h)


def path_line(points, close=False):
    p = skia.Path()
    if not points:
        return p
    p.moveTo(*points[0])
    for pt in points[1:]:
        p.lineTo(*pt)
    if close:
        p.close()
    return p


def cubic(p0, p1, p2, p3, n=80):
    pts = []
    for u in np.linspace(0, 1, n):
        v = 1 - u
        pts.append((v**3*p0[0] + 3*v*v*u*p1[0] + 3*v*u*u*p2[0] + u**3*p3[0],
                    v**3*p0[1] + 3*v*v*u*p1[1] + 3*v*u*u*p2[1] + u**3*p3[1]))
    return pts


class Film:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.surface = skia.Surface.MakeRaster(skia.ImageInfo.Make(
            int(W * scale), int(H * scale), skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType))
        self.canvas = self.surface.getCanvas()
        self.fonts = {}
        for weight in (400, 500, 600, 700):
            tf = skia.Typeface.MakeFromFile(str(ROOT / f"assets/fonts/SpaceGrotesk-{weight}.ttf"))
            for size in (18, 22, 24, 28, 32, 36, 42, 48, 56, 64, 72, 84, 96, 108, 124, 144, 176, 240):
                self.fonts[(weight, size)] = skia.Font(tf, size)
        for weight in ("Regular", "Medium"):
            tf = skia.Typeface.MakeFromFile(str(ROOT / f"assets/fonts/IBMPlexMono-{weight}.ttf"))
            for size in (16, 18, 20, 22, 24, 28, 32, 36, 42, 48, 56, 64):
                self.fonts[(f"mono{weight}", size)] = skia.Font(tf, size)
        cjk = skia.Typeface.MakeFromFile("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
        self.fonts[("cjk", 32)] = skia.Font(cjk, 32)
        self.fonts[("cjk", 42)] = skia.Font(cjk, 42)
        self.noise = self._make_noise()
        self.haze = self._make_haze()
        self.vignette = self._make_vignette()
        self.geo = json.loads((ROOT / "assets/data/ne_50m_admin_0_countries.geojson").read_text())

    def _make_noise(self):
        images = []
        rng = np.random.default_rng(210521)
        for _ in range(10):
            n = rng.integers(0, 256, (H // 3, W // 3), dtype=np.uint8)
            a = np.full_like(n, 16)
            rgba = np.stack([n, n, n, a], axis=-1)
            images.append(skia.Image.fromarray(np.ascontiguousarray(rgba), skia.kRGBA_8888_ColorType))
        return images

    def _make_haze(self):
        rng = np.random.default_rng(1969)
        result = []
        for j in range(3):
            acc = np.zeros((H // 2, W // 2), dtype=np.float32)
            for k, size in enumerate((5, 11, 23, 47)):
                small = rng.random((size, size), dtype=np.float32)
                im = Image.fromarray(np.uint8(small * 255)).resize((W // 2, H // 2), Image.Resampling.BICUBIC)
                acc += np.asarray(im, dtype=np.float32) / 255 * (0.48 ** k)
            acc = (acc - acc.min()) / (acc.max() - acc.min())
            alpha = np.uint8(clamp(0) + np.clip((acc - .36) * 34, 0, 20))
            rgba = np.zeros((H // 2, W // 2, 4), dtype=np.uint8)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = 97 + j * 8, 121 + j * 4, 120
            rgba[..., 3] = alpha
            result.append(skia.Image.fromarray(np.ascontiguousarray(rgba), skia.kRGBA_8888_ColorType))
        return result

    def _make_vignette(self):
        yy, xx = np.mgrid[-1:1:complex(H // 2), -1:1:complex(W // 2)]
        rr = np.sqrt((xx * .84) ** 2 + (yy * .63) ** 2)
        alpha = np.uint8(np.clip((rr - .36) * 104, 0, 118))
        rgba = np.zeros((H // 2, W // 2, 4), dtype=np.uint8)
        rgba[..., 3] = alpha
        return skia.Image.fromarray(np.ascontiguousarray(rgba), skia.kRGBA_8888_ColorType)

    def font(self, weight=500, size=36):
        key=(weight,size)
        if key not in self.fonts:
            tf=skia.Typeface.MakeFromFile(str(ROOT / f"assets/fonts/SpaceGrotesk-{weight}.ttf"))
            self.fonts[key]=skia.Font(tf,size)
        return self.fonts[key]

    def text(self, s, x, y, size=36, weight=500, fill=IVORY, alpha=1, anchor="left", tracking=0):
        f = self.font(weight, size)
        p = paint(fill, alpha)
        width = f.measureText(s) + max(0, len(s) - 1) * tracking
        if anchor == "center":
            x -= width / 2
        elif anchor == "right":
            x -= width
        if tracking == 0:
            self.canvas.drawString(s, x, y, f, p)
        else:
            for ch in s:
                self.canvas.drawString(ch, x, y, f, p)
                x += f.measureText(ch) + tracking
        return width

    def mono(self, s, x, y, size=20, fill=MUTED, alpha=1, anchor="left", tracking=1.2, weight="Regular"):
        key=(f"mono{weight}",size)
        if key not in self.fonts:
            tf=skia.Typeface.MakeFromFile(str(ROOT / f"assets/fonts/IBMPlexMono-{weight}.ttf"))
            self.fonts[key]=skia.Font(tf,size)
        f=self.fonts[key]
        p = paint(fill, alpha)
        width = f.measureText(s) + max(0, len(s) - 1) * tracking
        if anchor == "center":
            x -= width / 2
        elif anchor == "right":
            x -= width
        for ch in s:
            self.canvas.drawString(ch, x, y, f, p)
            x += f.measureText(ch) + tracking
        return width

    def typed(self, s, x, y, start, t, cps=15, **kwargs):
        n = int(clamp((t - start) * cps, 0, len(s)))
        self.mono(s[:n], x, y, **kwargs)
        if 0 < n < len(s) and int(t * 3) % 2 == 0:
            f = self.fonts[("monoRegular", kwargs.get("size", 20))]
            w = f.measureText(s[:n]) + n * kwargs.get("tracking", 1.2)
            self.canvas.drawRect(rect(x + w + 4, y - kwargs.get("size", 20), 3, kwargs.get("size", 20) + 3), paint(RED))

    def reveal(self, s, x, y, start, t, duration=.75, direction=28, **kwargs):
        q = ease(ramp(t, start, duration))
        if q <= 0:
            return
        self.canvas.save()
        self.canvas.clipRect(rect(0, y - kwargs.get("size", 36) * 1.35 - 40, W, kwargs.get("size", 36) * 1.8 + 80))
        self.text(s, x, y + direction * (1 - q), alpha=q, **kwargs)
        self.canvas.restore()

    def line(self, pts, fill=MUTED, alpha=1, width=1, dash=None):
        p = paint(fill, alpha, width)
        if dash:
            p.setPathEffect(skia.DashPathEffect.Make(dash, 0))
        self.canvas.drawPath(path_line(pts), p)

    def background(self, t, accent=False, paper=False):
        c = self.canvas
        if paper:
            c.drawColor(color("#DCD5C7"))
            shader = skia.GradientShader.MakeLinear([(0, 0), (W, H)], [int(color("#F0EADD")), int(color("#CFC7B8"))], [0, 1])
        else:
            c.drawColor(color("#07090A"))
            shader = skia.GradientShader.MakeLinear([(0, 0), (W, H)], [int(color("#152023")), int(color("#060708")), int(color("#090B0C"))], [0, .56, 1])
        p = skia.Paint()
        p.setShader(shader)
        c.drawRect(rect(0, 0, W, H), p)
        if not paper:
            j = int(t / 8) % 3
            dx = 24 * math.sin(t * .07 + j)
            dy = 35 * math.cos(t * .05 + j)
            p = paint(IVORY, .75)
            p.setBlendMode(skia.BlendMode.kScreen)
            c.drawImageRect(self.haze[j], rect(-70 + dx, -60 + dy, W + 140, H + 120), paint=p)
            self.grid(t)

    def grid(self, t, alpha=.12):
        for x in range(60, W, 80):
            self.line([(x, 0), (x, H)], "#7B918E", alpha, .55)
        for y in range(40, H, 80):
            self.line([(0, y), (W, y)], "#7B918E", alpha, .55)
        for x in range(60, W, 80):
            for y in range(40, H, 80):
                if (x // 80 + y // 80) % 3 == 0:
                    self.canvas.drawCircle(x, y, 1.4, paint("#B3C1BA", alpha * 1.8))

    def post(self, t, paper=False):
        c = self.canvas
        idx = int(t * 24) % len(self.noise)
        p = skia.Paint()
        p.setBlendMode(skia.BlendMode.kSoftLight if paper else skia.BlendMode.kOverlay)
        c.drawImageRect(self.noise[idx], rect(0, 0, W, H), paint=p)
        if not paper:
            c.drawImageRect(self.vignette, rect(0, 0, W, H), paint=skia.Paint())

    def chrome(self, chapter, t, light=False):
        fg = INK if light else MUTED
        a = ease(ramp(t, 0, .8))
        self.canvas.drawRect(rect(82, 73, 9, 9), paint(RED, a))
        self.mono("PUBLIC RECORD / RECONSTRUCTION", 106, 83, 16, fg, a, tracking=1.4)
        self.mono("SENIOR BOWL", 996, 83, 16, fg, a, "right", 1.8)
        self.line([(82, 112), (998, 112)], fg, a * .28, .8)
        self.mono(chapter, 82, 1828, 16, fg, a, tracking=1.5)
        self.mono("1969—1971", 998, 1828, 16, fg, a, "right", 1.5)
        self.canvas.drawRect(rect(82, 1858, 916, 2), paint(fg, a * .15))
        self.canvas.drawRect(rect(82, 1858, 916 * clamp(t / DURATION), 2), paint(RED, a * .8))

    def project(self, v, center, scale, angle=0, pitch=.18, roll=0):
        x, y, z = v
        cy, sy = math.cos(roll), math.sin(roll)
        x, z = x * cy + z * sy, -x * sy + z * cy
        cp, sp = math.cos(pitch), math.sin(pitch)
        y, z = y * cp - z * sp, y * sp + z * cp
        ca, sa = math.cos(angle), math.sin(angle)
        return center[0] + scale * (x * ca - y * sa), center[1] + scale * (x * sa + y * ca), z

    def drone(self, center, scale=1, angle=0, pitch=.18, roll=0, alpha=1, wreck=False, booster=False):
        outline = [(0,-315,8),(28,-255,18),(45,-135,30),(61,-45,42),(218,103,2),(207,142,0),
                   (78,111,23),(62,231,27),(35,275,12),(0,291,3),(-35,275,12),(-62,231,27),
                   (-78,111,23),(-207,142,0),(-218,103,2),(-61,-45,42),(-45,-135,30),(-28,-255,18)]
        center_v = (0, 24, 53)
        pr = lambda v: self.project(v, center, scale, angle, pitch, roll)
        faces = []
        for i in range(len(outline)):
            if wreck and i in (2, 3, 10, 14):
                continue
            a, b, c = outline[i], outline[(i + 1) % len(outline)], center_v
            pa, pb, pc = pr(a), pr(b), pr(c)
            ux, uy, uz = b[0]-a[0], b[1]-a[1], b[2]-a[2]
            vx, vy, vz = c[0]-a[0], c[1]-a[1], c[2]-a[2]
            nx, ny, nz = uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx
            norm = math.sqrt(nx*nx+ny*ny+nz*nz) or 1
            nz = abs(nz/norm)
            nx = nx/norm
            lum = clamp(.38 + .43*nz - .14*nx, .25, .88)
            base = np.array([112, 122, 119]) * lum
            col = "#%02x%02x%02x" % tuple(np.uint8(base))
            depth = (pa[2] + pb[2] + pc[2]) / 3
            faces.append((depth, path_line([(pa[0],pa[1]),(pb[0],pb[1]),(pc[0],pc[1])], True), col))
        for _, p, col in sorted(faces, key=lambda z: z[0]):
            self.canvas.drawPath(p, paint(col, alpha))
            self.canvas.drawPath(p, paint("#A5AEA6", alpha * .24, .75))
        # Panel seams and the dorsal ramjet intake.
        for pts in [[(0,-285,15),(0,-40,55),(0,35,56),(0,242,33)],
                    [(-48,-22,45),(-145,95,18),(-186,116,7)],
                    [(48,-22,45),(145,95,18),(186,116,7)],
                    [(-56,143,32),(0,155,54),(56,143,32)]]:
            pp = [pr(v) for v in pts]
            self.line([(v[0],v[1]) for v in pp], "#C1C8BC", alpha * .27, .9)
        ring = []
        for a in np.linspace(0, math.tau, 50):
            ring.append(pr((43*math.cos(a), 111 + 61*math.sin(a), 53)))
        self.canvas.drawPath(path_line([(p[0],p[1]) for p in ring], True), paint("#090D0E", alpha))
        self.canvas.drawPath(path_line([(p[0],p[1]) for p in ring], True), paint("#99A69F", alpha * .55, 2))
        cone = pr((0, 111, 72))
        self.canvas.drawCircle(cone[0], cone[1], 13 * scale, paint("#5E6A66", alpha))
        self.canvas.drawCircle(cone[0]-3*scale, cone[1]-4*scale, 4 * scale, paint("#B3B9AA", alpha * .55))
        if booster:
            bp = pr((0, 320, 2))
            self.canvas.drawCircle(bp[0], bp[1], 34 * scale, paint("#242C2D", alpha))

    def b52(self, x, y, s=1, alpha=1):
        c = self.canvas
        c.save(); c.translate(x, y); c.scale(s, s)
        body = [(0,-350),(24,-300),(30,180),(75,265),(72,282),(24,254),(14,340),(-14,340),(-24,254),(-72,282),(-75,265),(-30,180),(-24,-300)]
        wing = [(-24,-120),(-355,110),(-350,139),(-30,42),(30,42),(350,139),(355,110),(24,-120)]
        tail = [(-20,190),(-145,264),(-140,281),(-20,248),(20,248),(140,281),(145,264),(20,190)]
        for pts in (wing, tail, body):
            c.drawPath(path_line(pts, True), paint("#4A5554", alpha))
            c.drawPath(path_line(pts, True), paint("#9AA59C", alpha * .35, 1.2))
        for side in (-1, 1):
            for i in range(4):
                ex = side * (93 + i * 54)
                ey = 8 + i * 24
                c.drawOval(rect(ex-13, ey-30, 26, 60), paint("#161D1E", alpha))
                c.drawCircle(ex, ey-16, 5, paint("#B8C0B5", alpha * .5))
        c.restore()

    def c130(self, x, y, s=1, alpha=1):
        c = self.canvas
        c.save(); c.translate(x, y); c.scale(s, s)
        body = [(-250,-15),(-225,-42),(95,-42),(190,-22),(236,0),(190,24),(-195,28),(-250,12)]
        c.drawPath(path_line(body, True), paint("#66716C", alpha))
        c.drawPath(path_line([(-45,-25),(15,-150),(70,-147),(30,-22)], True), paint("#515D59", alpha))
        c.drawPath(path_line([(-190,-25),(-235,-135),(-202,-135),(-137,-25)], True), paint("#59645F", alpha))
        c.drawPath(path_line([(-175,0),(-255,-58),(-245,-66),(-125,-4)], True), paint("#59645F", alpha))
        for ex, ey in ((-5,-88),(45,-91),(-5,30),(45,28)):
            c.drawOval(rect(ex-18,ey-12,55,25),paint("#222B2B",alpha))
            c.drawCircle(ex+29,ey,30,paint("#B8C3B8",alpha*.12,1.2))
            c.drawLine(ex+29,ey-27,ex+29,ey+27,paint("#B8C3B8",alpha*.38,1))
        c.drawPath(path_line([(135,-26),(170,-18),(181,-7),(145,-8)],True),paint("#A7BBB5",alpha*.6))
        c.restore()

    def target(self, x, y, r, t, alpha=1):
        q = (t * .65) % 1
        self.canvas.drawCircle(x, y, r * (1 + q * .8), paint(RED, alpha * (1-q) * .38, 1.4))
        self.canvas.drawCircle(x, y, r, paint(RED, alpha * .7, 1.3))
        self.canvas.drawCircle(x, y, 4.5, paint(RED, alpha))
        for a in (0, math.pi/2, math.pi, 3*math.pi/2):
            self.line([(x+math.cos(a)*(r+7),y+math.sin(a)*(r+7)),
                       (x+math.cos(a)*(r+22),y+math.sin(a)*(r+22))],RED,alpha*.8,1.2)

    def scene_intro(self, t):
        self.background(t)
        self.chrome("00 / THE MACHINE", t)
        self.typed("CHINA / 1969", 82, 210, .25, t, 14, size=24, fill=AMBER, tracking=2.8)
        q = ease(ramp(t, .1, 1.5))
        self.canvas.save()
        self.canvas.translate(0, 70 * (1-q))
        self.drone((548, 805+7*math.sin(t*.45)), 1.38+.08*ramp(t,0,7), -.16+.025*t, .31+.025*math.sin(t*.4), -.06, q)
        self.canvas.restore()
        scan_y = lerp(330, 1290, smooth(ramp(t, .35, 2.4)))
        if t < 3.2:
            p = paint(RED, .6 * (1-ramp(t,2.5,.7)))
            p.setImageFilter(skia.ImageFilters.Blur(8,8))
            self.canvas.drawRect(rect(110,scan_y,860,2),p)
            self.canvas.drawRect(rect(150,scan_y,780,1),paint(RED,.8))
        self.reveal("A BLACK ARROW", 82, 1395, 1.0, t, size=84, weight=600)
        self.reveal("NO PILOT.", 82, 1500, 2.05, t, size=108, weight=700, fill=RED)
        self.reveal("JUST A CAMERA. AND A MEMORY.", 86, 1580, 3.1, t, size=28, weight=500, fill=MUTED)
        self.mono("LOCKHEED  D-21B", 86, 1710, 20, IVORY, ramp(t,3.6,.8), tracking=2.2)

    def scene_launch(self, t):
        self.background(t)
        self.chrome("01 / AIR LAUNCH", t)
        self.typed("D-21B / SENIOR BOWL",82,210,7.0,t,17,size=24,fill=AMBER,tracking=2)
        sep = ease(ramp(t,8.65,2.0))
        self.b52(505,655,1.12+.08*sep,1-ramp(t,9.1,1.0))
        dx,dy = -186*sep+220*math.sin(math.pi*sep),-70*sep
        drone_scale=.42+.78*sep
        drone_x,drone_y=726+dx,790+dy
        self.drone((drone_x,drone_y),drone_scale,-.52+.2*sep,.2,0,1,booster=t<10.7)
        flame = ramp(t,9.55,.15)*(1-ramp(t,11.3,.5))
        if flame:
            x,y=drone_x,drone_y+300*drone_scale
            p=skia.Paint(); p.setShader(skia.GradientShader.MakeRadial((x,y),150,
                [int(color("#FFF2C2",flame)),int(color("#F26A3C",flame*.8)),int(color("#F05244",0))],[0,.25,1]))
            self.canvas.drawCircle(x,y,150,p)
            for i in range(14):
                a=i*2.399; length=70+(i%5)*24
                self.line([(x,y),(x+math.cos(a)*length,y+math.sin(a)*length)],"#F5B36B",flame*.34,1)
        for i in range(22):
            xx=(i*197+int(t*900))%1350-140
            yy=350+(i*113)%980
            self.line([(xx,yy),(xx+120+flame*180,yy-48)],"#9CAEA7",flame*.12,1)
        self.mono("B-52H / CARRIER AIRCRAFT",160,1110,18,MUTED,ramp(t,7.4,.8))
        self.line([(160,1085),(345,905)],MUTED,.4,1)
        self.reveal("PAST MACH THREE.",82,1375,10.1,t,size=76,weight=600)
        self.canvas.save(); self.canvas.clipRect(rect(70,1400,930,260))
        self.text("3.3",72,1588,size=240,weight=600,fill=IVORY,alpha=ease(ramp(t,10.2,1.1)))
        self.canvas.restore()
        self.mono("MACH",640,1495,24,RED,ramp(t,10.5,.6),tracking=4)
        self.mono("90,000+ FT",640,1550,24,IVORY,ramp(t,10.8,.6),tracking=2)
        self.mono("SOLID ROCKET BOOSTER",640,1610,16,MUTED,ramp(t,11.1,.6),tracking=1.5)

    def geo_project(self, lon, lat):
        return 110+(lon-68)*14.8, 1320-(lat-10)*18.2

    def geo_path(self, coords):
        pts=[]
        for ring in coords:
            pp=[self.geo_project(*xy) for xy in ring]
            if len(pp)>2:
                pts.append(pp)
        p=skia.Path()
        for pp in pts:
            p.addPath(path_line(pp,True))
        return p

    def scene_map(self,t):
        self.background(t)
        self.chrome("02 / AUTONOMOUS ROUTE",t)
        self.typed("TARGET / LOP NOR",82,210,14.6,t,16,size=24,fill=AMBER,tracking=2.2)
        reveal=ease(ramp(t,14.5,1.3))
        china=None
        for feature in self.geo["features"]:
            name=feature["properties"].get("ADMIN","")
            g=feature["geometry"]
            if not g: continue
            polys=g["coordinates"] if g["type"]=="MultiPolygon" else [g["coordinates"]]
            for poly in polys:
                p=self.geo_path(poly)
                if name=="China":
                    china=p
                else:
                    self.canvas.drawPath(p,paint("#111718",reveal*.6))
                    self.canvas.drawPath(p,paint("#50605D",reveal*.16,.7))
        if china:
            self.canvas.drawPath(china,paint("#293936",reveal))
            self.canvas.drawPath(china,paint("#A9B9AE",reveal*.85,1.8))
            self.canvas.save(); self.canvas.clipPath(china)
            for lon in np.arange(72,137,1.35):
                for lat in np.arange(17,55,1.15):
                    x,y=self.geo_project(lon,lat)
                    self.canvas.drawCircle(x,y,1.6,paint("#9BB3A7",reveal*.36))
            self.canvas.restore()
        lx,ly=self.geo_project(90.5,40.5)
        self.target(lx,ly,31,t,reveal)
        self.line([(lx+42,ly-15),(lx+170,ly-105),(850,ly-105)],RED,reveal*.7,1)
        self.text("罗布泊",850,ly-130,size=32,weight="cjk",fill=IVORY,alpha=reveal,anchor="center")
        self.mono("NUCLEAR TEST SITE",850,ly-73,16,RED,reveal,"center",1.5)
        route=cubic((1110,1320),(920,1160),(740,1090),(lx,ly),100)
        n=int(len(route)*ease(ramp(t,15.4,3.0)))
        if n>1:
            self.line(route[:n],RED,.95,2.4)
            for j in range(0,n,12):
                self.canvas.drawCircle(route[j][0],route[j][1],2.5,paint(RED,.6))
            x,y=route[n-1]
            self.drone((x,y),.13,-.6,.12,0)
        ret=cubic((lx,ly),(lx+190,ly-120),(820,780),(1100,660),80)
        self.line(ret,"#8D9D96",.3,1.2,[7,11])
        self.reveal("NO JOYSTICK.",82,1500,18.1,t,size=76,weight=600)
        self.reveal("NO SECOND CHANCES.",82,1590,19.0,t,size=52,weight=500,fill=IVORY)
        self.mono("ROUTE SCHEMATIC / INERTIAL GUIDANCE",84,1690,17,MUTED,ramp(t,19.4,.7),tracking=1.4)

    def scene_camera(self,t):
        self.background(t)
        self.chrome("03 / OPTICAL PACKAGE",t)
        self.mono("THE EVIDENCE / ON FILM",82,210,22,IVORY,ramp(t,21.6,.8),tracking=2.2)
        self.drone((540,760),1.25+.035*ramp(t,21.5,6),0,.08,0)
        # Exposed camera bay over the metal skin.
        q=ease(ramp(t,21.7,.8))
        rr=skia.RRect.MakeRectXY(rect(355,630,370,360),22,22)
        self.canvas.drawRRect(rr,paint("#0A1011",q*.95))
        self.canvas.drawRRect(rr,paint(AMBER,q*.65,1.5))
        for yy in (698,790,882):
            self.line([(375,yy),(705,yy)],AMBER,q*.15,.8)
        # Lens glass and aperture.
        p=skia.Paint(); p.setShader(skia.GradientShader.MakeRadial((540,760),112,
            [int(color("#A9D3CA",q)),int(color("#305455",q)),int(color("#071011",q))],[0,.45,1]))
        self.canvas.drawCircle(540,760,104,p)
        self.canvas.drawCircle(540,760,104,paint("#B1C2B5",q*.55,2))
        for i in range(7):
            a=i*math.tau/7+t*.04
            pts=[(540+math.cos(a)*34,760+math.sin(a)*34),
                 (540+math.cos(a+.6)*88,760+math.sin(a+.6)*88),
                 (540+math.cos(a+1.25)*72,760+math.sin(a+1.25)*72)]
            self.canvas.drawPath(path_line(pts,True),paint("#152E30",q*.7))
        self.canvas.drawCircle(520,728,13,paint("#D8EEE0",q*.45))
        self.mono("HIGH-RESOLUTION CAMERA",540,970,16,AMBER,q,"center",1.4)
        # Film pod / twin spool cutaway.
        podq=ease(ramp(t,23.5,.8))
        self.canvas.drawRRect(skia.RRect.MakeRectXY(rect(418,1070,244,150),70,70),paint("#A18C67",podq))
        self.canvas.drawRRect(skia.RRect.MakeRectXY(rect(429,1081,222,128),60,60),paint("#272A26",podq))
        for x in (490,590):
            self.canvas.drawCircle(x,1145,43,paint("#C5B58F",podq*.85,2))
            self.canvas.drawCircle(x,1145,10,paint("#C5B58F",podq))
            for a in np.linspace(0,math.tau,7)[:-1]:
                self.canvas.drawCircle(x+28*math.cos(a+t),1145+28*math.sin(a+t),4,paint("#C5B58F",podq*.5))
        self.mono("RECOVERY POD",540,1260,17,AMBER,podq,"center",2)
        scan=ease(ramp(t,22.0,1.2))*(1-ramp(t,26.5,.7))
        self.canvas.drawPath(path_line([(475,850),(170,1445),(910,1445),(605,850)],True),paint("#78A29A",scan*.07))
        for i in range(11):
            y=1320+i*22
            pts=[]
            for x in range(190,901,24):
                pts.append((x,y+22*math.sin(x*.012+i*.7)))
            self.line(pts,"#76938A",scan*.24,1)
        flash=1-ramp(t,22.0,.18)
        if t>=22 and flash:
            self.canvas.drawCircle(540,760,140,paint(IVORY,flash*.35))
        self.reveal("FILM IN. THEN HOME.",82,1580,24.5,t,size=64,weight=600)
        self.mono("TARGET → TURN → PACIFIC",84,1670,18,MUTED,ramp(t,25.2,.7),tracking=2)

    def parachute(self,x,y,s=1,open_amount=1,alpha=1,failed=False):
        c=self.canvas
        c.save(); c.translate(x,y); c.scale(s,s)
        k=open_amount
        if failed:
            canopy=[(-105*k,0),(-75*k,-65*k),(-15*k,-38*k),(25*k,-92*k),(105*k,0),(45*k,-12*k),(-45*k,-5*k)]
        else:
            canopy=[(-120*k,0),(-110*k,-70*k),(-58*k,-120*k),(0,-135*k),(58*k,-120*k),(110*k,-70*k),(120*k,0),
                    (80*k,-18*k),(40*k,0),(0,-18*k),(-40*k,0),(-80*k,-18*k)]
        p=skia.Path(); p.moveTo(*canopy[0])
        # Curved canopy segments.
        p.cubicTo(-110*k,-85*k,-60*k,-135*k,0,-135*k)
        p.cubicTo(60*k,-135*k,110*k,-85*k,120*k,0)
        p.cubicTo(80*k,-18*k,40*k,18*k,0,-18*k)
        p.cubicTo(-40*k,18*k,-80*k,-18*k,-120*k,0)
        c.drawPath(p,paint(IVORY,alpha*.9))
        c.drawPath(p,paint("#A89F8E",alpha*.8,1.2))
        for xx in (-100,-48,0,48,100):
            c.drawLine(xx*k,-8,0,210,paint("#C4BCAA",alpha*.65,1))
        c.restore()

    def pod(self,x,y,s=1,alpha=1):
        c=self.canvas;c.save();c.translate(x,y);c.scale(s,s)
        c.drawRRect(skia.RRect.MakeRectXY(rect(-37,-55,74,110),32,32),paint(AMBER,alpha))
        c.drawRRect(skia.RRect.MakeRectXY(rect(-28,-45,56,90),25,25),paint("#343B37",alpha))
        c.drawCircle(0,-4,17,paint("#B5AA8C",alpha*.7,2))
        c.restore()

    def scene_recovery(self,t):
        self.background(t)
        self.chrome("04 / MID-AIR RECOVERY",t)
        self.mono("THE PLAN / THREE STEPS",82,210,22,IVORY,ramp(t,27.6,.7),tracking=2.2)
        # Atmospheric perspective and ocean far below.
        for i in range(7):
            y=440+i*175+18*math.sin(t*.2+i)
            self.line([(0,y),(W,y-60)],"#92AAA5",.045+i*.008,26)
        self.canvas.drawRect(rect(0,1510,W,410),paint("#071416",.7))
        for i in range(17):
            y=1530+i*22
            pts=[(x,y+5*math.sin(x*.017+t*.8+i)) for x in range(-20,1120,35)]
            self.line(pts,"#6D9690",.1,1)
        self.drone((690,390),.45,.45,.28,.1,ramp(t,27.5,.7))
        ej=ease(ramp(t,28.7,1.4)); chute=ease(ramp(t,30.0,.8))
        px,py=lerp(690,585,ej),lerp(450,870,ej)
        if t>=28.7:
            self.pod(px,py,.8)
        if chute:
            self.parachute(px,py-155*chute,.85,chute)
            self.line([(px,py-50),(px,py-155*chute)],"#C4BCAA",.65,1)
        catch=ease(ramp(t,31.8,2.0))
        cx=lerp(-300,365,catch)
        self.c130(cx,1070,.95)
        # Capture envelope, intentionally schematic rather than a literal net.
        self.line([(cx+225,1030),(px-95,py-70),(px+95,py-70),(cx+225,1110)],AMBER,catch*.45,1.2,[7,10])
        self.target(px,py,58,t,catch*.7)
        stages=[("01","EJECT",28.5),("02","DEPLOY",30.0),("03","CATCH",31.8)]
        for i,(num,label,st) in enumerate(stages):
            y=1260+i*105
            a=ease(ramp(t,st,.5))
            self.mono(num,84,y,20,RED,a,tracking=1.5)
            self.text(label,140,y,size=36,weight=600,alpha=a)
            self.canvas.drawRect(rect(140,y+22,270*a,2),paint(RED,a*.65))
        self.mono("JC-130B / MID-AIR RECOVERY",998,1635,17,MUTED,ramp(t,32,.7),"right",1.3)
        self.reveal("THE DRONE WAS EXPENDABLE.",82,1740,33.7,t,size=40,weight=500,fill=IVORY)

    def scene_plan(self,t):
        self.canvas.drawColor(color("#050606"))
        q=ease(ramp(t,36.05,.38))
        self.mono("FINE ON PAPER",540,790,20,RED,q,"center",3)
        self.text("THAT WAS",540,950,size=108,weight=600,anchor="center",alpha=q)
        self.text("THE PLAN.",540,1070,size=108,weight=600,anchor="center",alpha=q,fill=RED)
        self.canvas.drawRect(rect(435,1155,210*q,3),paint(IVORY,q*.6))
        self.post(t)

    def failure_diagram(self,idx,local):
        cx,cy=540,820
        if idx==0:
            self.canvas.drawCircle(cx,cy,160,paint("#1D2928",1))
            self.canvas.drawCircle(cx,cy,160,paint(MUTED,.35,1.2))
            for r in (45,95,140): self.canvas.drawCircle(cx,cy,r,paint(MUTED,.15,1))
            self.target(cx+50,cy-40,25,local,1)
            route=cubic((260,1110),(330,850),(610,950),(cx+50,cy-40),70)
            route+=cubic((cx+50,cy-40),(cx+160,cy-160),(770,520),(920,420),50)
            n=int(len(route)*ease(ramp(local,.2,1.7)))
            self.line(route[:n],RED,.9,3)
            self.mono("TURN POINT MISSED",540,1100,18,RED,1,"center",2)
        elif idx==1:
            collapse=smooth(ramp(local,.65,1.2))
            self.parachute(540,590,1.15,1-collapse*.8,1,True)
            y=lerp(770,1080,ramp(local,.7,1.4)**2)
            self.pod(540,y,1.1)
            for xx in (440,490,590,640):
                self.line([(xx,620),(540,y-55)],MUTED,.5,1)
            self.mono("CANOPY FAILURE",540,1170,18,RED,1,"center",2)
        elif idx==2:
            for i in range(12):
                y=670+i*37
                pts=[(x,y+10*math.sin(x*.014+i+local*2)) for x in range(180,901,25)]
                self.line(pts,"#718F89",.25,1.2)
            drop=ease(ramp(local,.25,1.0)); sink=ramp(local,1.2,1.5)
            self.pod(540,lerp(470,740,drop)+sink*180,1.05,1-sink*.6)
            for r in (45,105,180):
                self.canvas.drawOval(rect(540-r*(1+sink),690-r*.22*(1+sink),r*2*(1+sink),r*.44*(1+sink)),paint("#9BB5AC",(1-sink)*.4,1.4))
            self.mono("POD LOST AT SEA",540,1170,18,RED,1,"center",2)
        else:
            for i in range(8):
                pts=[]
                for a in np.linspace(0,math.tau,100):
                    r=75+i*34+13*math.sin(a*5+i)
                    pts.append((540+r*math.cos(a),790+r*.58*math.sin(a)))
                self.line(pts,"#819C94",.38,1.2)
            q=ease(ramp(local,.1,1.2))
            self.drone((lerp(260,540,q),lerp(480,800,q)),.38,local*2.4,.4,.2,1-ramp(local,1.3,.3))
            if local>1.25:
                self.target(540,800,40+80*ease(ramp(local,1.25,1)),local,1)
                for i in range(9):
                    a=i*2.4; rr=35+(i%4)*22
                    self.line([(540,800),(540+math.cos(a)*rr,800+math.sin(a)*rr)],RED,.6,2)
            self.mono("LOST OVER CHINA",540,1170,18,RED,1,"center",2)

    def scene_failures(self,t):
        local_all=t-38
        idx=min(3,int(local_all/3.25)); local=local_all-idx*3.25
        self.background(t)
        self.chrome("05 / MISSION LEDGER",t)
        dates=["09 NOV 1969","20 DEC 1970","04 MAR 1971","20 MAR 1971"]
        titles=["MISSED THE TURN.","PARACHUTE FAILED.","LOST AT SEA.","FELL INSIDE CHINA."]
        self.typed(f"MISSION 0{idx+1} / {dates[idx]}",82,210,38+idx*3.25,t,20,size=22,fill=AMBER,tracking=1.6)
        q=ease(ramp(local,.05,.45))
        self.canvas.drawCircle(820,520,150,paint("#121C1D",q))
        self.failure_diagram(idx,local)
        self.reveal(titles[idx],82,1400,38+idx*3.25+.25,t,size=64,weight=600,fill=IVORY)
        reasons=["GUIDANCE / NO RECOVERY","FILM POD / DESTROYED","RECOVERY / FAILED","AIRFRAME / UNRECOVERED"]
        self.mono(reasons[idx],84,1470,17,RED,q,tracking=1.5)
        for i in range(4):
            x=84+i*230
            active=i==idx; done=i<idx
            self.mono(f"0{i+1}",x,1635,22,RED if active else MUTED,1 if active or done else .3)
            self.canvas.drawRect(rect(x,1660,188,3),paint(RED if active or done else MUTED,1 if active else .35))
            if done:
                self.line([(x,1700),(x+12,1712),(x+32,1684)],RED,.8,2)
        self.mono("FOUR OPERATIONAL FLIGHTS",84,1770,16,MUTED,.8,tracking=1.4)

    def scene_zero(self,t):
        self.background(t)
        self.chrome("06 / THE RESULT",t)
        self.typed("FOUR MISSIONS",82,210,51.2,t,18,size=22,fill=AMBER,tracking=2)
        # Blank film runs continuously behind the tally.
        offset=(t-51)*150
        for col in (120,810):
            self.canvas.save(); self.canvas.clipRect(rect(col-70,300,210,1270))
            for j in range(-2,9):
                y=j*210+offset%210
                self.canvas.drawRRect(skia.RRect.MakeRectXY(rect(col,y,150,176),5,5),paint("#83928C",.09,1.2))
                for sy in (y+14,y+142):
                    for sx in (col-13,col+145):
                        self.canvas.drawRect(rect(sx,sy,18,18),paint("#87938C",.22))
            self.canvas.restore()
        self.text("4",70,710,size=240,weight=600,fill=IVORY,alpha=ease(ramp(t,51.1,.8)))
        self.mono("MISSIONS",96,790,22,MUTED,1,tracking=4)
        self.text("0",565,1210,size=240,weight=600,fill=RED,alpha=ease(ramp(t,52.0,.8)))
        self.mono("USABLE FRAMES",590,1290,22,IVORY,1,tracking=3)
        self.line([(80,965),(1000,965)],MUTED,.35,1)
        self.reveal("NO USABLE PICTURES.",82,1500,53.2,t,size=64,weight=600)
        self.reveal("PROGRAM ENDED / 1971",84,1590,54.4,t,size=28,weight=500,fill=MUTED)

    def scene_museum(self,t):
        self.background(t)
        self.chrome("07 / THE ARTIFACT",t)
        self.typed("ARTIFACT / D-21B",82,210,57.1,t,16,size=22,fill=AMBER,tracking=2)
        # Gallery spotlight.
        p=skia.Paint();p.setShader(skia.GradientShader.MakeRadial((540,760),750,
            [int(color("#D9CCAA",.16)),int(color("#64766C",.04)),int(color("#000000",0))],[0,.6,1]))
        self.canvas.drawRect(rect(0,150,W,1430),p)
        q=ease(ramp(t,57.2,1.4))
        self.canvas.drawRect(rect(190,1225,700,55),paint("#151B1A",q))
        self.canvas.drawRect(rect(230,1280,620,190),paint("#0A0E0E",q))
        self.drone((540,830),1.13+.035*ramp(t,57,8),-.12+.012*(t-57),.48,.04,q,wreck=True)
        # Detached fragments in the display.
        for x,y,a in ((265,1030,-.5),(780,980,.7),(350,520,.2)):
            pts=[(x-45,y-20),(x+60,y),(x+25,y+35),(x-20,y+15)]
            self.canvas.drawPath(path_line(pts,True),paint("#46514C",q))
            self.canvas.drawPath(path_line(pts,True),paint("#A4A994",q*.25,1))
        # Glass case and grazing reflections.
        self.canvas.drawRect(rect(145,350,790,880),paint("#9EB4A8",q*.22,1.4))
        self.line([(145,350),(250,1230)],"#C9D4C5",q*.1,1)
        self.line([(700,350),(930,1230)],"#DDE4D5",q*.09,22)
        self.line([(780,350),(935,830)],"#DDE4D5",q*.13,3)
        self.mono("PUBLIC DISPLAY / CHINA",540,1370,20,IVORY,q,"center",2)
        self.text("公开展出",540,1430,size=32,weight="cjk",fill=AMBER,alpha=q,anchor="center")
        self.reveal("HIDING IN PLAIN SIGHT.",82,1595,60.6,t,size=64,weight=600)
        self.mono("A SECRET MISSION / LEFT AS WRECKAGE",84,1680,17,MUTED,ramp(t,61.5,.7),tracking=1.5)

    def scene_final(self,t):
        self.background(t,paper=True)
        self.chrome("08 / PUBLIC RECORD",t,light=True)
        self.typed("SENIOR BOWL",82,245,65.0,t,17,size=28,fill=INK,tracking=4,weight="Medium")
        self.canvas.drawRect(rect(82,310,916,2),paint(INK,.25))
        self.reveal("FOUR FLIGHTS.",82,760,65.35,t,size=108,weight=700,fill=INK,direction=40)
        self.reveal("ZERO FILM.",82,900,65.7,t,size=124,weight=700,fill=RED,direction=40)
        self.mono("LOCKHEED D-21B / 1969—1971",86,1020,20,INK,ramp(t,66,.6),tracking=2)
        self.canvas.save();self.canvas.translate(835,1190);self.canvas.rotate(-8)
        qq=ease(ramp(t,66.0,.55))
        self.canvas.drawRRect(skia.RRect.MakeRectXY(rect(-130,-46,260,92),5,5),paint(RED,qq*.8,4))
        self.mono("DECLASSIFIED",0,12,24,RED,qq,"center",2,weight="Medium")
        self.canvas.restore()
        self.mono("BASED ON DECLASSIFIED + PUBLIC RECORDS",82,1705,16,INK,ramp(t,66.2,.7),tracking=1.3)
        self.post(t,paper=True)

    def render(self,t):
        c=self.canvas
        c.save();c.scale(self.scale,self.scale);c.clear(skia.ColorTRANSPARENT)
        if t<7:self.scene_intro(t)
        elif t<14.5:self.scene_launch(t)
        elif t<21.5:self.scene_map(t)
        elif t<27.5:self.scene_camera(t)
        elif t<36:self.scene_recovery(t)
        elif t<38:self.scene_plan(t)
        elif t<51:self.scene_failures(t)
        elif t<57:self.scene_zero(t)
        elif t<65:self.scene_museum(t)
        else:self.scene_final(t)
        if t<38 or t>=51:
            if not (t>=36 and t<38) and not t>=65:
                self.post(t)
        if t<.45:
            c.drawRect(rect(0,0,W,H),paint("#000000",1-smooth(ramp(t,0,.45))))
        c.restore()
        return self.surface.makeImageSnapshot()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--time",type=float)
    ap.add_argument("--stills",type=str)
    ap.add_argument("--video",action="store_true")
    ap.add_argument("--output",default="build/frame.png")
    ap.add_argument("--scale",type=float,default=1)
    ap.add_argument("--fps",type=int,default=60)
    ap.add_argument("--start",type=float,default=0)
    ap.add_argument("--end",type=float,default=DURATION)
    args=ap.parse_args()
    film=Film(args.scale)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if args.stills:
        for t in map(float,args.stills.split(",")):
            image=film.render(t)
            path=out.with_name(f"{out.stem}_{t:05.2f}.png")
            image.save(str(path),skia.kPNG)
            print(path,flush=True)
        return
    if args.time is not None:
        film.render(args.time).save(str(out),skia.kPNG)
        print(out)
        return
    if args.video:
        w,h=int(W*args.scale),int(H*args.scale)
        cmd=["ffmpeg","-hide_banner","-loglevel","warning","-y","-f","rawvideo","-pix_fmt","rgba",
             "-s",f"{w}x{h}","-r",str(args.fps),"-i","-","-an","-c:v","libx264","-preset","medium",
             "-crf","16","-pix_fmt","yuv420p","-movflags","+faststart",str(out)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        first=int(round(args.start*args.fps));last=int(round(args.end*args.fps))
        try:
            for n in range(first,last):
                image=film.render(n/args.fps)
                proc.stdin.write(image.toarray().tobytes())
                if (n-first)%60==0:
                    print(f"frame {n-first}/{last-first}  t={n/args.fps:.2f}",flush=True)
        finally:
            if proc.stdin:proc.stdin.close()
        code=proc.wait()
        if code:raise SystemExit(code)


if __name__=="__main__":
    main()
