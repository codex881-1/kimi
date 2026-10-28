"""Real-terrain components built from AWS Terrarium DEM mosaics.

- DEM: geo <-> pixel mapping (web mercator), bilinear sampling, local metric frame
- relief_night(): multi-directional hillshade + ambient occlusion, graded for the night map
- RidgeCamera / ridge_lines(): world-fixed terrain scanlines with floating-horizon occlusion
- profile(): elevation profile along a path (great-circle approximated in lon/lat)
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from .core import ROOT

DEMDIR = ROOT / "data" / "dem"
R_EARTH = 6371000.0


def _merc_y(lat):
    return math.asinh(math.tan(math.radians(lat)))


class DEM:
    def __init__(self, name):
        d = np.load(DEMDIR / f"{name}.npz")
        self.h = d["h"].astype(np.float32)
        self.z = int(d["z"])
        self.lon0, self.lon1 = float(d["lon0"]), float(d["lon1"])
        self.lat0, self.lat1 = float(d["lat0"]), float(d["lat1"])
        self.H, self.W = self.h.shape
        self.my0, self.my1 = _merc_y(self.lat1), _merc_y(self.lat0)  # top, bottom

    def px(self, lon, lat):
        x = (lon - self.lon0) / (self.lon1 - self.lon0) * self.W
        y = (self.my0 - _merc_y(lat)) / (self.my0 - self.my1) * self.H
        return x, y

    def lonlat(self, x, y):
        lon = self.lon0 + x / self.W * (self.lon1 - self.lon0)
        my = self.my0 - y / self.H * (self.my0 - self.my1)
        return lon, math.degrees(math.atan(math.sinh(my)))

    def m_per_px(self, lat):
        return 2 * math.pi * R_EARTH * math.cos(math.radians(lat)) / (256 * 2 ** self.z)

    def sample_px(self, x, y):
        return map_coordinates(self.h, [np.asarray(y, np.float64), np.asarray(x, np.float64)], order=1,
                               mode="nearest").astype(np.float32)

    def sample(self, lon, lat):
        lon, lat = np.asarray(lon, np.float64), np.asarray(lat, np.float64)
        x = (lon - self.lon0) / (self.lon1 - self.lon0) * self.W
        y = (self.my0 - np.arcsinh(np.tan(np.radians(lat)))) / (self.my0 - self.my1) * self.H
        return self.sample_px(x, y)


@lru_cache(None)
def dem(name):
    return DEM(name)


# ---------------------------------------------------------------- relief shading
def hillshade(h, cell, az_deg, alt_deg, zf=1.0):
    gy, gx = np.gradient(h * zf, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = math.radians(az_deg), math.radians(alt_deg)
    s = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect - math.pi / 2)
    return np.clip(s, 0, 1).astype(np.float32)


def relief_layers(d: DEM, zf=2.2):
    """Return (shade, ao, elev_norm, slope) float32 arrays for a DEM."""
    h = np.maximum(d.h, 0)
    mid_lat = (d.lat0 + d.lat1) / 2
    cell = d.m_per_px(mid_lat)
    hs = (0.40 * hillshade(h, cell, 315, 38, zf) + 0.25 * hillshade(h, cell, 270, 45, zf)
          + 0.20 * hillshade(h, cell, 360, 50, zf) + 0.15 * hillshade(h, cell, 225, 40, zf))
    # ambient occlusion proxy: how far a point sits below its neighbourhood
    hb = cv2.GaussianBlur(h, (0, 0), 6)
    hb2 = cv2.GaussianBlur(h, (0, 0), 24)
    ao = np.clip(0.5 + (h - hb) / 900 + (h - hb2) / 3000, 0, 1).astype(np.float32)
    gy, gx = np.gradient(h, cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    return hs, ao, h, slope


def relief_night(d: DEM, zf=2.2):
    """Night-graded relief: lowlands sink to blue-black, plateau in cold slate, snow catches moonlight."""
    hs, ao, h, slope = relief_layers(d, zf)
    e = np.clip(h / 7000.0, 0, 1)
    snow = np.clip((h - 4700) / 1300, 0, 1) * np.clip(1.2 - slope / 55, 0, 1)
    light = np.clip(hs * 0.9 + (ao - 0.5) * 0.9, 0, 1.2)
    base = np.stack([0.030 + 0.10 * e, 0.040 + 0.115 * e, 0.055 + 0.13 * e], -1)
    lit = base * (0.25 + 1.35 * light[..., None])
    snowc = np.array([0.80, 0.86, 0.92], np.float32)
    lit = lit * (1 - 0.75 * snow[..., None]) + snowc * (light[..., None] ** 1.3) * 0.75 * snow[..., None]
    # cool atmospheric haze in the deep basins
    low = np.clip(1 - h / 1500, 0, 1)[..., None]
    lit = lit * (1 - 0.35 * low) + np.array([0.020, 0.028, 0.038]) * 0.35 * low
    return np.clip(lit, 0, 1).astype(np.float32)


def relief_paper(d: DEM, zf=2.0):
    """Engraved-paper relief (for the analysis world): warm paper, ink shadows."""
    hs, ao, h, slope = relief_layers(d, zf)
    light = np.clip(hs * 0.85 + (ao - 0.5) * 0.8 + 0.15, 0, 1)
    paper = np.array([0.925, 0.898, 0.839], np.float32)
    ink = np.array([0.20, 0.19, 0.18], np.float32)
    shade = (1 - light) ** 1.6
    out = paper * (1 - 0.85 * shade[..., None]) + ink * 0.85 * shade[..., None] * 0.0
    return np.clip(out, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- local metric frame
class LocalFrame:
    """East/North metres around an origin (lon0, lat0) on a DEM."""

    def __init__(self, d: DEM, lon0, lat0):
        self.d, self.lon0, self.lat0 = d, lon0, lat0
        self.kx = 111320.0 * math.cos(math.radians(lat0))
        self.ky = 110540.0

    def to_lonlat(self, e, n):
        return self.lon0 + np.asarray(e) / self.kx, self.lat0 + np.asarray(n) / self.ky

    def from_lonlat(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def height(self, e, n):
        lon, lat = self.to_lonlat(e, n)
        return self.d.sample(lon, lat)


class RidgeCamera:
    """Pinhole camera in the local frame. yaw: heading in degrees (0 = north, 90 = east)."""

    def __init__(self, pos, yaw, pitch, fov_deg, w, h, cx=None, cy=None):
        self.pos = np.asarray(pos, np.float64)
        self.yaw, self.pitch = math.radians(yaw), math.radians(pitch)
        self.w, self.h = w, h
        self.f = (w / 2) / math.tan(math.radians(fov_deg) / 2)
        self.cx = w / 2 if cx is None else cx
        self.cy = h / 2 if cy is None else cy
        sy, cyw = math.sin(self.yaw), math.cos(self.yaw)
        sp, cp = math.sin(self.pitch), math.cos(self.pitch)
        self.fwd = np.array([sy * cp, cyw * cp, sp])
        self.right = np.array([cyw, -sy, 0.0])
        self.up = np.cross(self.right, self.fwd)

    def project(self, e, n, z):
        e, n, z = np.broadcast_arrays(np.asarray(e, np.float64), np.asarray(n, np.float64), np.asarray(z, np.float64))
        p = np.stack([e - self.pos[0], n - self.pos[1], z - self.pos[2]], -1)
        dz = p @ self.fwd
        dx = p @ self.right
        dy = p @ self.up
        dzs = np.maximum(dz, 1e-3)
        return self.cx + self.f * dx / dzs, self.cy - self.f * dy / dzs, dz


def ridge_lines(frame: LocalFrame, cam: RidgeCamera, axis_deg, n_slices=170, d_near=1500, d_far=26000,
                half_width=26000, samples=900, origin=(0.0, 0.0), x_step=1.0, spacing_pow=1.35, zmod=None):
    """World-fixed scanlines perpendicular to axis_deg, ordered front-to-back from the camera.
    Returns list of (dist, xs, ys, zs_world, visible_mask) with ys sampled per screen column."""
    ax = math.radians(axis_deg)
    fwd = np.array([math.sin(ax), math.cos(ax)])
    lat = np.array([math.cos(ax), -math.sin(ax)])
    o = np.asarray(origin, np.float64)
    cam_along = (cam.pos[:2] - o) @ fwd
    # slice offsets along axis, denser near the camera
    u = np.linspace(0, 1, n_slices) ** spacing_pow
    offs = cam_along + d_near + u * (d_far - d_near)
    # snap to a world grid so lines stay glued to the terrain as the camera moves
    step = (d_far - d_near) / n_slices
    offs = np.unique(np.round(offs / step * 2) / 2 * step)
    cols = np.arange(0, cam.w + x_step, x_step, dtype=np.float64)
    horizon = np.full(cols.shape, np.inf)
    out = []
    lt = np.linspace(-half_width, half_width, samples)
    for off in offs:
        base = o + fwd * off
        e = base[0] + lat[0] * lt
        n = base[1] + lat[1] * lt
        z = frame.height(e, n).astype(np.float64)
        if zmod is not None:
            z = zmod(e, n, z)
        sx, sy, dz = cam.project(e, n, z)
        ok = dz > 50
        if ok.sum() < 3:
            continue
        sx, sy, zz = sx[ok], sy[ok], z[ok]
        order = np.argsort(sx)
        sx, sy, zz = sx[order], sy[order], zz[order]
        inside = (cols >= sx[0]) & (cols <= sx[-1])
        ys = np.interp(cols, sx, sy)
        zs = np.interp(cols, sx, zz)
        vis = inside & (ys < horizon - 0.25)
        horizon = np.where(inside, np.minimum(horizon, ys), horizon)
        out.append((float(np.mean(dz)), cols, ys, zs, vis, float(off - cam_along)))
    return out


def profile(d: DEM, lonlat_a, lonlat_b, n=1200):
    """Elevation along the straight lon/lat path; returns (dist_km, elev_m, lons, lats)."""
    lons = np.linspace(lonlat_a[0], lonlat_b[0], n)
    lats = np.linspace(lonlat_a[1], lonlat_b[1], n)
    z = d.sample(lons, lats)
    kx = 111.32 * np.cos(np.radians(lats))
    seg = np.hypot(np.diff(lons) * kx[:-1], np.diff(lats) * 110.54)
    dist = np.concatenate([[0], np.cumsum(seg)])
    return dist, z, lons, lats
