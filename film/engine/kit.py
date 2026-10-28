"""Shared scene toolkit: base class, paper texture, HUD furniture, draw-on paths, glows."""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np
import skia

from .core import (ROOT, W, H, PAL, col, mix, draw_text, text_width, paint_fill, paint_stroke, Layer,
                   image_from_float, expo_out, smooth, clamp01, ramp)

CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
SAMP = skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear)
SAMP_CUBIC = skia.SamplingOptions(skia.CubicResampler.Mitchell())


class Scene:
    theme = "night"      # caption palette
    cap_y = 330          # caption top
    name = "scene"

    def __init__(self, t0, t1):
        self.t0, self.t1 = t0, t1

    def active(self, t):
        return self.t0 <= t < self.t1

    def local(self, t):
        return t - self.t0

    def draw(self, c: skia.Canvas, t):
        raise NotImplementedError

    def overlay(self, img, t):
        return img

    def post(self, t):
        return {}

    def mblur(self, t):
        return 0.0


# ---------------------------------------------------------------- textures
@lru_cache(None)
def paper_texture(w=1400, h=2400, seed=7):
    p = CACHE / f"paper_{w}x{h}_{seed}.png"
    if p.exists():
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32) / 65535.0
        return a
    rng = np.random.default_rng(seed)
    base = np.array(PAL["paper"], np.float32)
    # low-frequency mottling and faint stains
    n = np.zeros((h, w), np.float32)
    for s, a in ((300, 0.010), (110, 0.007), (30, 0.006), (6, 0.006)):
        z = rng.standard_normal((h // s + 3, w // s + 3)).astype(np.float32)
        n += cv2.resize(z, (w, h), interpolation=cv2.INTER_CUBIC) * a
    # fibres
    fib = np.zeros((h, w), np.float32)
    for _ in range(5200):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, math.pi)
        L = rng.uniform(6, 34)
        bend = rng.normal(0, 0.25)
        pts = []
        for k in range(8):
            u = k / 7
            a_ = ang + bend * (u - 0.5)
            pts.append((x + math.cos(a_) * L * (u - 0.5), y + math.sin(a_) * L * (u - 0.5)))
        cv2.polylines(fib, [np.array(pts, np.int32)], False, float(rng.uniform(0.3, 1.0)), 1, cv2.LINE_AA)
    fib = cv2.GaussianBlur(fib, (0, 0), 0.6)
    fine = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.7) * 0.012
    lum = 1 + n - fib * 0.035 + fine
    img = base[None, None, :] * lum[..., None]
    # warm edge darkening (aged sheet)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    e = np.minimum(np.minimum(xx, w - xx), np.minimum(yy, h - yy)) / 180
    img *= (0.93 + 0.07 * np.clip(e, 0, 1))[..., None]
    img = np.clip(img, 0, 1).astype(np.float32)
    cv2.imwrite(str(p), (img[..., ::-1] * 65535).astype(np.uint16))
    return img


@lru_cache(None)
def paper_image():
    return image_from_float(paper_texture())


def draw_paper(c, t, cam_x=0.0, cam_y=0.0, scale=1.0, alpha=1.0):
    img = paper_image()
    c.save()
    c.translate(W / 2, H / 2)
    c.scale(scale, scale)
    c.translate(-img.width() / 2 - cam_x, -img.height() / 2 - cam_y)
    p = skia.Paint()
    p.setAlphaf(alpha)
    c.drawImage(img, 0, 0, SAMP, p)
    c.restore()


def night_sky(c, top=PAL["night0"], bottom=PAL["night2"], y0=0, y1=H):
    p = skia.Paint()
    p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, y0), skia.Point(0, y1)],
                                               [col(top).toColor(), col(bottom).toColor()]))
    c.drawRect(skia.Rect.MakeWH(W, H), p)


# ---------------------------------------------------------------- paths
def path_from(pts, close=False):
    p = skia.Path()
    if len(pts) == 0:
        return p
    p.moveTo(float(pts[0][0]), float(pts[0][1]))
    for x, y in pts[1:]:
        p.lineTo(float(x), float(y))
    if close:
        p.close()
    return p


def smooth_path(pts):
    """Catmull-Rom through points -> cubic path."""
    p = skia.Path()
    pts = [tuple(map(float, q)) for q in pts]
    if len(pts) < 3:
        return path_from(pts)
    p.moveTo(*pts[0])
    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i else pts[i]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        p.cubicTo(*c1, *c2, *p2)
    return p


def path_length(path):
    pm = skia.PathMeasure(path, False)
    L = 0.0
    while True:
        L += pm.getLength()
        if not pm.nextContour():
            break
    return L


def partial(path, f0, f1=None):
    """Sub-path covering fraction [f0, f1] (or [0, f0]) of total length, across contours."""
    if f1 is None:
        f0, f1 = 0.0, f0
    f0, f1 = clamp01(f0), clamp01(f1)
    out = skia.Path()
    if f1 <= f0:
        return out
    total = path_length(path)
    a, b = f0 * total, f1 * total
    pm = skia.PathMeasure(path, False)
    acc = 0.0
    while True:
        L = pm.getLength()
        s, e = max(a - acc, 0), min(b - acc, L)
        if e > s:
            pm.getSegment(s, e, out, True)
        acc += L
        if acc >= b or not pm.nextContour():
            break
    return out


def point_at(path, f):
    pm = skia.PathMeasure(path, False)
    L = pm.getLength()
    pos, tan = pm.getPosTan(clamp01(f) * L)
    return (pos.x(), pos.y()), (tan.x(), tan.y())


def dashed(paint, on=10, off=8, phase=0.0):
    p = skia.Paint(paint)
    p.setPathEffect(skia.DashPathEffect.Make([on, off], phase))
    return p


# ---------------------------------------------------------------- glows
def glow_dot(c, x, y, r, color, intensity=1.0, halo=6.0):
    """Light source: hot core + soft halo (bloom in post finishes it)."""
    if intensity <= 0.001:
        return
    cc = col(color)
    p = skia.Paint(AntiAlias=True)
    p.setShader(skia.GradientShader.MakeRadial(
        skia.Point(x, y), r * halo,
        [skia.Color4f(cc.fR, cc.fG, cc.fB, 0.55 * intensity).toColor(),
         skia.Color4f(cc.fR, cc.fG, cc.fB, 0.18 * intensity).toColor(),
         skia.Color4f(cc.fR, cc.fG, cc.fB, 0.0).toColor()],
        [0.0, 0.35, 1.0]))
    p.setBlendMode(skia.BlendMode.kPlus)
    c.drawCircle(x, y, r * halo, p)
    core = mix(color, (1, 1, 1), 0.55)
    c.drawCircle(x, y, r, paint_fill(col(core, min(1, intensity))))


def glow_stroke(c, path, color, width, alpha=1.0, glow=8.0, glow_alpha=0.6):
    with Layer(c, alpha=alpha * glow_alpha, blur=glow, blend=skia.BlendMode.kPlus):
        c.drawPath(path, paint_stroke(col(color), width * 2.2))
    c.drawPath(path, paint_stroke(col(color, alpha), width))


# ---------------------------------------------------------------- HUD furniture
def label(c, text, x, y, face="mono", size=24, color="snow", alpha=1.0, tracking=0.12, align="left"):
    return draw_text(c, text, x, y, face, size, paint_fill(col(color, alpha)), tracking, align)


def leader(c, ax, ay, bx, by, color, alpha=1.0, w=1.6, dot=4.0, f=1.0):
    """Leader line from anchor (a) with dot to label point (b), drawn on with fraction f."""
    if f <= 0:
        return
    mx, my = ax + (bx - ax) * min(1, f), ay + (by - ay) * min(1, f)
    c.drawLine(ax, ay, mx, my, paint_stroke(col(color, alpha), w))
    c.drawCircle(ax, ay, dot * min(1, f * 3), paint_fill(col(color, alpha)))


def brackets(c, x, y, w, h, color, alpha=1.0, L=26, sw=2.0):
    p = paint_stroke(col(color, alpha), sw, cap=skia.Paint.kSquare_Cap)
    for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        c.drawLine(px, py, px + dx * L, py, p)
        c.drawLine(px, py, px, py + dy * L, p)


def frame_hud(c, t, color="steel", alpha=0.5, text_l="", text_r=""):
    """Thin viewfinder furniture around the frame edge (kept out of IG UI zones)."""
    m = 54
    brackets(c, m, 250, W - 2 * m, H - 250 - 300, col(color).__class__ and color, alpha, L=34, sw=1.6)
    if text_l:
        label(c, text_l, m + 4, 236, size=19, color=color, alpha=alpha * 1.2)
    if text_r:
        label(c, text_r, W - m - 4, 236, size=19, color=color, alpha=alpha * 1.2, align="right")


def ticker(value, digits=0):
    return f"{value:,.{digits}f}"
