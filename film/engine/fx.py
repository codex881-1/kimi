"""Film-look post-processing on float32 RGB frames (0..1), plus shared VFX helpers."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .core import W, H

cv2.setNumThreads(1)

_rng = np.random.default_rng(1965)
_GRAIN = None
_VIG = None


def _grain_bank():
    global _GRAIN
    if _GRAIN is None:
        bank = []
        for i in range(12):
            g = _rng.standard_normal((H + 64, W + 64)).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), 0.65)
            g2 = cv2.GaussianBlur(_rng.standard_normal((H // 2 + 32, W // 2 + 32)).astype(np.float32), (0, 0), 0.8)
            g2 = cv2.resize(g2, (W + 64, H + 64), interpolation=cv2.INTER_LINEAR)
            g = g / g.std() * 0.8 + g2 / g2.std() * 0.45
            bank.append(g.astype(np.float32))
        _GRAIN = bank
    return _GRAIN


def _vignette():
    global _VIG
    if _VIG is None:
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        nx = (x - W / 2) / (W / 2)
        ny = (y - H / 2) / (H / 2)
        r = np.sqrt(nx * nx * 1.0 + ny * ny * 0.72)
        _VIG = np.clip(r, 0, 1.6).astype(np.float32)
    return _VIG


def _bright(small, threshold):
    lum = small.max(axis=2)
    k = np.clip((lum - threshold) / (1 - threshold + 1e-6), 0, 1) ** 1.5
    return small * k[..., None]


def bloom_layer(small4, threshold=0.62, strength=0.55, tint=(1.0, 0.92, 0.82), radius=1.0):
    """Bloom computed entirely at quarter resolution; returns a quarter-res additive layer."""
    bright = _bright(small4, threshold)
    acc = np.zeros_like(bright)
    for s_, wgt in ((2.5, 0.45), (8, 0.35), (22, 0.30), (50, 0.22)):
        acc += cv2.GaussianBlur(bright, (0, 0), s_ * radius) * wgt
    return acc * (np.asarray(tint, np.float32) * strength)


def halation_layer(small4, strength=0.18):
    lum = small4.max(axis=2)
    k = np.clip((lum - 0.78) / 0.22, 0, 1)
    k = cv2.GaussianBlur(k, (0, 0), 2.5)
    return k[..., None] * (np.array([1.0, 0.42, 0.18], np.float32) * strength)


def chroma(img, amount=1.0):
    if amount <= 0.01:
        return img
    out = img.copy()
    for ch, s in ((0, 1 + 0.0016 * amount), (2, 1 - 0.0016 * amount)):
        M = np.array([[s, 0, (1 - s) * W / 2], [0, s, (1 - s) * H / 2]], np.float32)
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


from numba import njit, prange  # noqa: E402


@njit(parallel=True, fastmath=True, cache=True)
def _finish_kernel(img, add, vig, grain, out, exposure, vamt, contrast, sat, wr, wg, wb, lift, gamt, fade, seed):
    Hh, Ww = img.shape[0], img.shape[1]
    for y in prange(Hh):
        for x in range(Ww):
            vm = 1.0 - vamt * vig[y, x]
            r = (img[y, x, 0] * exposure + add[y, x, 0]) * vm
            g = (img[y, x, 1] * exposure + add[y, x, 1]) * vm
            b = (img[y, x, 2] * exposure + add[y, x, 2]) * vm
            if contrast != 1.0:
                r = 0.5 + (r - 0.5) * contrast
                g = 0.5 + (g - 0.5) * contrast
                b = 0.5 + (b - 0.5) * contrast
            if sat != 1.0:
                l = 0.2126 * r + 0.7152 * g + 0.0722 * b
                r = l + (r - l) * sat
                g = l + (g - l) * sat
                b = l + (b - l) * sat
            r = lift + r * wr * (1.0 - lift)
            g = lift + g * wg * (1.0 - lift)
            b = lift + b * wb * (1.0 - lift)
            # soft highlight shoulder (rational, cheap)
            if r > 0.85:
                d = (r - 0.85) / 0.15
                r = 0.85 + 0.15 * d / (1.0 + d)
            if g > 0.85:
                d = (g - 0.85) / 0.15
                g = 0.85 + 0.15 * d / (1.0 + d)
            if b > 0.85:
                d = (b - 0.85) / 0.15
                b = 0.85 + 0.15 * d / (1.0 + d)
            l = (r + g + b) / 3.0
            gr = grain[y, x] * gamt * (0.35 + 1.4 * l * (1.0 - l))
            h1 = ((x * 73856093) ^ (y * 19349663) ^ seed) & 1023
            h2 = ((x * 83492791) ^ (y * 26544357) ^ (seed * 7)) & 1023
            dth = (h1 - h2) / 1023.0 / 255.0
            r = min(max((r + gr) * fade + dth, 0.0), 1.0)
            g = min(max((g + gr) * fade + dth, 0.0), 1.0)
            b = min(max((b + gr) * fade + dth, 0.0), 1.0)
            out[y, x, 0] = np.uint8(r * 255.0 + 0.5)
            out[y, x, 1] = np.uint8(g * 255.0 + 0.5)
            out[y, x, 2] = np.uint8(b * 255.0 + 0.5)


_VIGC = {}


def _vig_curve():
    if "v" not in _VIGC:
        _VIGC["v"] = (np.clip(_vignette() - 0.35, 0, 1.2) ** 1.6).astype(np.float32)
    return _VIGC["v"]


def finish_u8(img, frame_idx, p):
    """Fused finishing pass -> uint8 RGB."""
    img = np.ascontiguousarray(img, np.float32)
    exposure = float(p.get("exposure", 1.0))
    small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA) * exposure
    add_s = np.zeros_like(small)
    if p.get("bloom", 0) > 0:
        add_s += bloom_layer(small, p.get("bloom_thr", 0.62), p["bloom"], p.get("bloom_tint", (1.0, 0.92, 0.82)))
    if p.get("halation", 0) > 0:
        add_s += halation_layer(small, p["halation"])
    add = np.ascontiguousarray(cv2.resize(add_s, (W, H), interpolation=cv2.INTER_LINEAR))
    img = np.ascontiguousarray(chroma(img, p.get("chroma", 0.8)))
    bank = _grain_bank()
    r = np.random.default_rng(frame_idx * 7919 + 13)
    gi = bank[frame_idx % len(bank)]
    ox, oy = int(r.integers(0, 64)), int(r.integers(0, 64))
    gr = np.ascontiguousarray(gi[oy:oy + H, ox:ox + W])
    warm = float(p.get("warmth", 0.0))
    out = np.empty((H, W, 3), np.uint8)
    _finish_kernel(img, add, _vig_curve(), gr, out, exposure, float(p.get("vignette", 0.22)),
                   float(p.get("contrast", 1.0)), float(p.get("sat", 1.0)), 1 + warm, 1 + warm * 0.25, 1 - warm,
                   float(p.get("lift", 0.0)), float(p.get("grain", 0.03)), float(p.get("fade", 1.0)),
                   int((frame_idx * 2654435761) % 2147483647))
    return out


# ---------------------------------------------------------------- displacement helpers
_GRID = None


def _grid():
    global _GRID
    if _GRID is None:
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        _GRID = (x, y)
    return _GRID


def shockwave(img, cx, cy, radius, width=60, amp=18):
    """Refractive ring distortion (pressure front) centred on (cx, cy)."""
    if radius <= 0 or amp <= 0.05:
        return img
    gx, gy = _grid()
    dx, dy = gx - cx, gy - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-3
    k = (r - radius) / width
    prof = np.exp(-k * k * 2.2) * np.sin(k * 2.6) * amp
    mx = gx - dx / r * prof
    my = gy - dy / r * prof
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def radial_blur(img, cx, cy, strength=0.04, steps=8):
    """Zoom blur toward (cx, cy) - used for zoom-through transitions."""
    if strength <= 0.002:
        return img
    acc = np.zeros_like(img)
    for i in range(steps):
        s = 1 + strength * i / (steps - 1)
        M = np.array([[s, 0, (1 - s) * cx], [0, s, (1 - s) * cy]], np.float32)
        acc += cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / steps


def directional_blur(img, angle_deg, length):
    if length < 1.5:
        return img
    n = int(min(64, max(3, length)))
    k = np.zeros((n, n), np.float32)
    c = (n - 1) / 2
    a = math.radians(angle_deg)
    for i in range(n * 3):
        t = i / (n * 3 - 1) - 0.5
        x, y = int(round(c + math.cos(a) * t * (n - 1))), int(round(c + math.sin(a) * t * (n - 1)))
        k[y, x] = 1
    k /= k.sum()
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)
