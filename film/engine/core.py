"""Core of the film engine: constants, palette, easing, deterministic noise, fonts,
HarfBuzz-shaped text, and skia <-> numpy helpers."""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import skia
import uharfbuzz as hb

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "assets" / "fonts"
W, H, FPS = 1080, 1920, 30

# ---------------------------------------------------------------- palette (sRGB 0..1)
def _hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

PAL = {
    "ink": _hex("#15171B"),
    "ink2": _hex("#2B2E33"),
    "paper": _hex("#ECE5D6"),
    "paper2": _hex("#DED4C0"),
    "pencil": _hex("#8C877C"),
    "night0": _hex("#06080B"),
    "night1": _hex("#0D1218"),
    "night2": _hex("#18212B"),
    "snow": _hex("#E8EDF1"),
    "steel": _hex("#93A3B3"),
    "ice": _hex("#8FD3FF"),
    "red": _hex("#E5452F"),
    "amber": _hex("#FFB04A"),
    "core": _hex("#FFE9C2"),
}


def col(name_or_rgb, a=1.0):
    c = PAL[name_or_rgb] if isinstance(name_or_rgb, str) else name_or_rgb
    return skia.Color4f(c[0], c[1], c[2], float(np.clip(a, 0, 1)))


def mix(a, b, t):
    a = PAL[a] if isinstance(a, str) else a
    b = PAL[b] if isinstance(b, str) else b
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


# ---------------------------------------------------------------- easing
def clamp01(x):
    return 0.0 if x <= 0 else 1.0 if x >= 1 else x


def ramp(t, t0, t1):
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return clamp01((t - t0) / (t1 - t0))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = clamp01(x)
    return x * x * x * (x * (6 * x - 15) + 10)


def expo_out(x):
    x = clamp01(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def expo_in(x):
    x = clamp01(x)
    return 0.0 if x <= 0 else 2 ** (10 * x - 10)


def expo_inout(x):
    x = clamp01(x)
    if x <= 0 or x >= 1:
        return x
    return 2 ** (20 * x - 10) / 2 if x < 0.5 else (2 - 2 ** (-20 * x + 10)) / 2


def cubic_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def cubic_inout(x):
    x = clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def quint_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 5


def quart_inout(x):
    x = clamp01(x)
    return 8 * x ** 4 if x < 0.5 else 1 - (-2 * x + 2) ** 4 / 2


def back_out(x, s=1.4):
    x = clamp01(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def spring(x, zeta=0.55, omega=14.0):
    """Damped spring step response, x in seconds since trigger. Settles to 1."""
    if x <= 0:
        return 0.0
    if zeta >= 1:
        return 1 - math.exp(-omega * x) * (1 + omega * x)
    wd = omega * math.sqrt(1 - zeta * zeta)
    return 1 - math.exp(-zeta * omega * x) * (math.cos(wd * x) + zeta / math.sqrt(1 - zeta * zeta) * math.sin(wd * x))


def lerp(a, b, t):
    return a + (b - a) * t


def bezier_ease(x1, y1, x2, y2):
    """CSS-style cubic-bezier timing function."""
    def f(x):
        x = clamp01(x)
        lo, hi = 0.0, 1.0
        for _ in range(30):
            m = (lo + hi) / 2
            bx = 3 * (1 - m) ** 2 * m * x1 + 3 * (1 - m) * m * m * x2 + m ** 3
            if bx < x:
                lo = m
            else:
                hi = m
        m = (lo + hi) / 2
        return 3 * (1 - m) ** 2 * m * y1 + 3 * (1 - m) * m * m * y2 + m ** 3
    return f


# A long, silky deceleration used for most camera moves.
silk = bezier_ease(0.16, 1.0, 0.3, 1.0)
glide = bezier_ease(0.65, 0.0, 0.35, 1.0)


# ---------------------------------------------------------------- deterministic noise
def hash01(*xs):
    h = 2166136261
    for x in xs:
        for b in np.float64(x).tobytes():
            h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h / 0xFFFFFFFF


def vnoise(t, seed=0.0):
    """Smooth 1D value noise in [-1, 1]."""
    i = math.floor(t)
    f = t - i
    a = hash01(i, seed) * 2 - 1
    b = hash01(i + 1, seed) * 2 - 1
    u = f * f * (3 - 2 * f)
    return a + (b - a) * u


def fbm(t, seed=0.0, octaves=3):
    s, amp, fr, norm = 0.0, 1.0, 1.0, 0.0
    for o in range(octaves):
        s += amp * vnoise(t * fr, seed + o * 17.3)
        norm += amp
        amp *= 0.5
        fr *= 2.03
    return s / norm


def _hash_np(i, seed):
    x = np.sin(i * 127.1 + seed * 311.7) * 43758.5453123
    return x - np.floor(x)


def fbm_np(x, seed=0.0, octaves=3):
    """Vectorised smooth value-noise fbm in [-1, 1]."""
    x = np.asarray(x, np.float64)
    s = np.zeros_like(x)
    amp, fr, norm = 1.0, 1.0, 0.0
    for o in range(octaves):
        xx = x * fr
        i = np.floor(xx)
        f = xx - i
        u = f * f * (3 - 2 * f)
        a = _hash_np(i, seed + o * 17.3) * 2 - 1
        b = _hash_np(i + 1, seed + o * 17.3) * 2 - 1
        s += amp * (a + (b - a) * u)
        norm += amp
        amp *= 0.5
        fr *= 2.03
    return s / norm


def drift(t, seed, amp=1.0, speed=0.35):
    """Organic 2D camera drift (handheld breathing)."""
    return amp * fbm(t * speed, seed), amp * fbm(t * speed, seed + 101.7)


# ---------------------------------------------------------------- fonts & text
def _tag(s):
    return int.from_bytes(s.encode(), "big")


@lru_cache(None)
def typeface(file, **axes):
    tf = skia.Typeface.MakeFromFile(str(FONTS / file))
    if axes:
        coords = [skia.FontArguments.VariationPosition.Coordinate(_tag(k), float(v)) for k, v in axes.items()]
        args = skia.FontArguments()
        args.setVariationDesignPosition(
            skia.FontArguments.VariationPosition(skia.FontArguments.VariationPosition.Coordinates(coords)))
        tf = tf.makeClone(args)
    return tf


FACES = {
    "serif_i": ("InstrumentSerif-Italic.ttf", {}),
    "serif": ("InstrumentSerif-Regular.ttf", {}),
    "sans_black": ("Inter[opsz,wght].ttf", {"wght": 800, "opsz": 32}),
    "sans_bold": ("Inter[opsz,wght].ttf", {"wght": 650, "opsz": 28}),
    "sans_med": ("Inter[opsz,wght].ttf", {"wght": 500, "opsz": 20}),
    "sans": ("Inter[opsz,wght].ttf", {"wght": 420, "opsz": 14}),
    "type": ("CourierPrime-Regular.ttf", {}),
    "type_b": ("CourierPrime-Bold.ttf", {}),
    "mono": ("JetBrainsMono[wght].ttf", {"wght": 450}),
    "mono_b": ("JetBrainsMono[wght].ttf", {"wght": 700}),
    "plex": ("IBMPlexMono-Medium.ttf", {}),
    "news": ("OldStandard-Bold.ttf", {}),
    "news_r": ("OldStandard-Regular.ttf", {}),
    "cjk": ("NotoSerifSC[wght].ttf", {"wght": 600}),
}


@lru_cache(None)
def face(name):
    f, ax = FACES[name]
    return typeface(f, **ax)


@lru_cache(None)
def _hb_font(name):
    f, ax = FACES[name]
    blob = hb.Blob.from_file_path(str(FONTS / f))
    fc = hb.Face(blob)
    font = hb.Font(fc)
    if ax:
        font.set_variations({k: float(v) for k, v in ax.items()})
    return font, fc.upem


def font(name, size):
    f = skia.Font(face(name), size)
    f.setSubpixel(True)
    f.setEdging(skia.Font.Edging.kSubpixelAntiAlias)
    f.setHinting(skia.FontHinting.kNone)
    f.setLinearMetrics(True)
    return f


@lru_cache(4096)
def shape(name, size, text, tracking=0.0):
    """Kerned per-character x offsets (one per char) and total advance.
    tracking is in em units (0.02 = +2%)."""
    hbf, upem = _hb_font(name)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hbf, buf, {"kern": True, "liga": False})
    scale = size / upem
    xs = [0.0] * len(text)
    x = 0.0
    seen = set()
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        c = info.cluster
        if c not in seen and c < len(text):
            xs[c] = x + pos.x_offset * scale
            seen.add(c)
        x += pos.x_advance * scale + tracking * size
    # chars inside multi-char clusters inherit previous position
    for i in range(len(text)):
        if i not in seen:
            xs[i] = xs[i - 1] if i else 0.0
    return tuple(xs), x - (tracking * size if text else 0)


def text_width(name, size, text, tracking=0.0):
    return shape(name, size, text, tracking)[1]


def draw_text(canvas, text, x, y, name, size, paint, tracking=0.0, align="left"):
    xs, w = shape(name, size, text, tracking)
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    blob = skia.TextBlob.MakeFromPosTextH(text, [x + v for v in xs], y, font(name, size))
    if blob is not None:
        canvas.drawTextBlob(blob, 0, 0, paint)
    return w


def paint_fill(c, aa=True, blend=None):
    p = skia.Paint(Color4f=c, AntiAlias=aa)
    if blend is not None:
        p.setBlendMode(blend)
    return p


def paint_stroke(c, w, cap=skia.Paint.kRound_Cap, join=skia.Paint.kRound_Join):
    p = skia.Paint(Color4f=c, AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=w)
    p.setStrokeCap(cap)
    p.setStrokeJoin(join)
    return p


# ---------------------------------------------------------------- surfaces
def new_surface(w=W, h=H):
    info = skia.ImageInfo.Make(w, h, skia.ColorType.kRGBA_8888_ColorType, skia.AlphaType.kPremul_AlphaType)
    return skia.Surface.MakeRaster(info)


def new_surface_f16(w=W, h=H):
    info = skia.ImageInfo.Make(w, h, skia.ColorType.kRGBA_F16_ColorType, skia.AlphaType.kPremul_AlphaType)
    return skia.Surface.MakeRaster(info)


def surface_to_float(surf):
    img = surf.makeImageSnapshot()
    if img.colorType() == skia.ColorType.kRGBA_F16_ColorType:
        a = img.toarray(colorType=skia.ColorType.kRGBA_F32_ColorType)
        return a[..., :3].astype(np.float32)
    a = img.toarray()
    return a[..., :3].astype(np.float32) / 255.0


def image_from_float(arr):
    """float32 HxWx3 or HxWx4 (0..1) -> skia.Image"""
    a = np.clip(arr, 0, 1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,), a.dtype)], 2)
    u8 = (a * 255 + 0.5).astype(np.uint8)
    return skia.Image.fromarray(np.ascontiguousarray(u8), colorType=skia.ColorType.kRGBA_8888_ColorType)


def image_from_u8(u8):
    if u8.shape[2] == 3:
        u8 = np.concatenate([u8, np.full(u8.shape[:2] + (1,), 255, np.uint8)], 2)
    return skia.Image.fromarray(np.ascontiguousarray(u8), colorType=skia.ColorType.kRGBA_8888_ColorType)


def blur_filter(sigma):
    if sigma <= 0.05:
        return None
    return skia.ImageFilters.Blur(sigma, sigma, skia.TileMode.kDecal)


class Layer:
    """Context manager: draw into an isolated layer with opacity / blur / blend."""

    def __init__(self, canvas, alpha=1.0, blur=0.0, blend=None, bounds=None):
        self.c, self.a, self.b, self.blend, self.bounds = canvas, alpha, blur, blend, bounds

    def __enter__(self):
        p = skia.Paint()
        p.setAlphaf(float(np.clip(self.a, 0, 1)))
        f = blur_filter(self.b)
        if f is not None:
            p.setImageFilter(f)
        if self.blend is not None:
            p.setBlendMode(self.blend)
        if self.bounds is not None:
            self.c.saveLayer(self.bounds, p)
        else:
            self.c.saveLayer(None, p)
        return self.c

    def __exit__(self, *a):
        self.c.restore()
