"""Fetch AWS Terrarium elevation tiles and mosaic them into float32 heightmaps (metres).

Outputs (data/dem/):
  region_z{z}.npz  - Himalaya -> Tarim basin -> Lop Nur, for the strategic map
  nanda_z{z}.npz   - Nanda Devi massif, for the 3D ridge-line terrain
Each npz holds: h (H, W float32), lat0, lat1, lon0, lon1, z
"""
import io
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
OUT = Path(__file__).resolve().parent.parent / "data" / "dem"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "tiles"
CACHE.mkdir(exist_ok=True)


def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x, y, z):
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def get_tile(z, x, y):
    p = CACHE / f"{z}_{x}_{y}.png"
    if not p.exists():
        for _ in range(4):
            try:
                r = requests.get(URL.format(z=z, x=x, y=y), timeout=30)
                r.raise_for_status()
                p.write_bytes(r.content)
                break
            except Exception as e:  # retry transient S3 errors
                err = e
        else:
            raise err
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
    return a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0


def mosaic(name, lon0, lat0, lon1, lat1, z):
    x0, y0 = lonlat_to_tile(lon0, lat1, z)
    x1, y1 = lonlat_to_tile(lon1, lat0, z)
    tx = range(int(x0), int(x1) + 1)
    ty = range(int(y0), int(y1) + 1)
    jobs = [(z, x, y) for y in ty for x in tx]
    with ThreadPoolExecutor(16) as ex:
        tiles = list(ex.map(lambda j: get_tile(*j), jobs))
    H, W = len(ty) * 256, len(tx) * 256
    h = np.zeros((H, W), np.float32)
    for (zz, x, y), t in zip(jobs, tiles):
        h[(y - ty[0]) * 256:(y - ty[0] + 1) * 256, (x - tx[0]) * 256:(x - tx[0] + 1) * 256] = t
    # bounds of the full tile mosaic (web-mercator rows)
    L0, T0 = tile_to_lonlat(tx[0], ty[0], z)
    L1, B1 = tile_to_lonlat(tx[-1] + 1, ty[-1] + 1, z)
    np.savez_compressed(OUT / f"{name}_z{z}.npz", h=h, lon0=L0, lon1=L1, lat0=B1, lat1=T0, z=z,
                        tx0=tx[0], ty0=ty[0])
    print(name, z, h.shape, f"lon {L0:.3f}..{L1:.3f} lat {B1:.3f}..{T0:.3f}",
          f"min {h.min():.0f} max {h.max():.0f}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["region", "nanda"]
    if "region" in which:
        mosaic("region", 72.0, 23.5, 104.0, 45.5, 7)
    if "nanda" in which:
        mosaic("nanda", 79.70, 30.18, 80.25, 30.58, 12)
        mosaic("nanda", 79.86, 30.30, 80.08, 30.45, 13)
