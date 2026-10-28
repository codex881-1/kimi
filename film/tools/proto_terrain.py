import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import skia

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.core import *  # noqa
from engine import terrain as T

OUT = Path("/tmp/verify/proto")
OUT.mkdir(parents=True, exist_ok=True)

ND = (79.9708, 30.3758)      # Nanda Devi main summit
LOP = (89.80, 40.80)         # approx. ground zero of test 596 (Lop Nur)


def map_still():
    d = T.dem("region_z7")
    t = time.time()
    rel = T.relief_night(d)
    print("relief", rel.shape, f"{time.time()-t:.1f}s")
    cv2.imwrite(str(OUT / "relief_full.jpg"), (rel[..., ::-1] * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])
    ax, ay = d.px(*ND)
    bx, by = d.px(*LOP)
    ang = math.degrees(math.atan2(bx - ax, -(by - ay)))  # bearing in pixel space
    print("bearing px", ang, "dist px", math.hypot(bx - ax, by - ay))
    img = image_from_float(rel)
    surf = new_surface()
    c = surf.getCanvas()
    c.clear(col("night0"))
    mx, my = (ax + bx) / 2, (ay + by) / 2
    c.save()
    c.translate(W / 2, H / 2 + 60)
    c.rotate(-ang)
    s = 1.15
    c.scale(s, s)
    c.translate(-mx, -my)
    c.drawImage(img, 0, 0, skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear))
    c.drawCircle(ax, ay, 6, paint_fill(col("amber")))
    c.drawCircle(bx, by, 6, paint_fill(col("red")))
    c.restore()
    surf.makeImageSnapshot().save(str(OUT / "map.png"), skia.kPNG)


def ridge_still():
    d = T.dem("nanda_z13")
    fr = T.LocalFrame(d, *ND)
    # camera south-west of the summit, looking north-east
    yaw = 35
    dist = 17000
    pos = (-dist * math.sin(math.radians(yaw)), -dist * math.cos(math.radians(yaw)), 6400)
    cam = T.RidgeCamera(pos, yaw, -2.0, 46, W, H, cy=H * 0.52)
    t = time.time()
    lines = T.ridge_lines(fr, cam, yaw, n_slices=190, d_near=2500, d_far=30000, half_width=14000, samples=1400)
    print("ridge", len(lines), f"{time.time()-t:.2f}s")
    surf = new_surface()
    c = surf.getCanvas()
    c.clear(col("night0"))
    t = time.time()
    for dist_, xs, ys, zs, vis in reversed(lines):
        fade = np.clip(1 - (dist_ - 2500) / 30000, 0.12, 1)
        # break into visible runs
        idx = np.flatnonzero(vis)
        if len(idx) == 0:
            continue
        splits = np.flatnonzero(np.diff(idx) > 1) + 1
        for run in np.split(idx, splits):
            if len(run) < 2:
                continue
            p = skia.Path()
            p.moveTo(xs[run[0]], ys[run[0]])
            for i in run[1:]:
                p.lineTo(xs[i], ys[i])
            zmean = zs[run].mean()
            snow = np.clip((zmean - 4500) / 2500, 0, 1)
            cc = mix("steel", "snow", snow)
            c.drawPath(p, paint_stroke(col(cc, 0.25 + 0.75 * fade), 1.0 + 1.2 * fade))
    print("draw", f"{time.time()-t:.2f}s")
    surf.makeImageSnapshot().save(str(OUT / "ridge.png"), skia.kPNG)


def profile_stats():
    d = T.dem("region_z7")
    dist, z, lons, lats = T.profile(d, ND, LOP)
    print("profile km", dist[-1], "max", z.max(), "argmax km", dist[z.argmax()])
    for k in range(0, len(dist), 100):
        print(f"{dist[k]:7.0f} km  {z[k]:6.0f} m")


if __name__ == "__main__":
    what = sys.argv[1:] or ["map", "ridge", "profile"]
    if "map" in what:
        map_still()
    if "ridge" in what:
        ridge_still()
    if "profile" in what:
        profile_stats()
