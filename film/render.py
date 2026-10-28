"""Frame compositor + parallel renderer.

usage:
  python render.py still 3.2 12.8 ...      -> /tmp/verify/stills/t_XX.XX.jpg
  python render.py video [t0 t1] [--fps N] -> out/video_noaudio.mp4
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np
import skia

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine.core import W, H, FPS, new_surface, surface_to_float, col  # noqa: E402
from engine import fx  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

DEFAULT_POST = {
    "night": dict(bloom=0.55, bloom_thr=0.55, halation=0.16, chroma=0.9, vignette=0.30, grain=0.034, lift=0.012,
                  contrast=1.03, warmth=0.0, sat=1.0),
    "paper": dict(bloom=0.10, bloom_thr=0.85, halation=0.0, chroma=0.6, vignette=0.20, grain=0.026, lift=0.0,
                  contrast=1.0, warmth=0.01, sat=1.0),
}


def build():
    from film import make_timeline
    return make_timeline()


def compose(scenes, captions, t, fi, mb_samples=5):
    act = [s for s in scenes if s.active(t)]
    mb = max([s.mblur(t) for s in act] + [0.0])

    def draw_scenes(tt):
        surf = new_surface()
        c = surf.getCanvas()
        c.clear(skia.Color4f(0, 0, 0, 1))
        for s in act:
            c.save()
            s.draw(c, tt)
            c.restore()
        return surface_to_float(surf)

    if mb > 0.01:
        shutter = mb / FPS
        acc = None
        for k in range(mb_samples):
            tt = t + shutter * (k / (mb_samples - 1) - 0.5)
            a = draw_scenes(tt)
            acc = a if acc is None else acc + a
        img = acc / mb_samples
    else:
        img = draw_scenes(t)
    for s in act:
        img = s.overlay(img, t)
    top = act[-1] if act else None
    theme = getattr(top, "theme", "night") if top else "night"
    theme_v = top.theme_value(t) if top is not None and hasattr(top, "theme_value") else (1.0 if theme == "paper" else 0.0)
    cap_y = getattr(top, "cap_y", 330) if top else 330
    cap_alpha = top.cap_alpha(t) if top is not None and hasattr(top, "cap_alpha") else 1.0
    if captions is not None:
        cs = new_surface()
        cc = cs.getCanvas()
        cc.clear(skia.Color4f(0, 0, 0, 0))
        captions.draw(cc, t, y_top=cap_y, theme=theme_v, alpha=cap_alpha)
        # skia's toarray() is unpremultiplied: weight colour by coverage
        rgba = cs.makeImageSnapshot().toarray().astype(np.float32) / 255.0
        a = rgba[..., 3:4]
        img = img * (1 - a) + rgba[..., :3] * a
    base = dict(DEFAULT_POST["paper" if theme_v > 0.5 else "night"])
    if top is not None:
        base.update(top.post(t))
    return fx.finish_u8(img, fi, base)


def render_stills(times):
    scenes, captions, dur = build()
    d = Path("/tmp/verify/stills")
    d.mkdir(parents=True, exist_ok=True)
    import cv2
    for t in times:
        t0 = time.time()
        fi = int(round(t * FPS))
        u8 = compose(scenes, captions, t, fi)
        p = d / f"t_{t:05.2f}.jpg"
        cv2.imwrite(str(p), u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(p, f"{time.time() - t0:.2f}s", flush=True)


def _worker(idx, f0, f1, fps, path):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    scenes, captions, dur = build()
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p",
           "-x264-params", "aq-mode=3:psy-rd=1.0,0.15:deblock=-1,-1", str(path)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t_start = time.time()
    for fi in range(f0, f1):
        t = fi / fps
        u8 = compose(scenes, captions, t, fi)
        pr.stdin.write(u8.tobytes())
        if (fi - f0) % 30 == 0:
            el = time.time() - t_start
            print(f"[w{idx}] frame {fi} ({fi - f0 + 1}/{f1 - f0}) {el / (fi - f0 + 1):.2f}s/f", flush=True)
    pr.stdin.close()
    pr.wait()


def render_video(t0=None, t1=None, fps=FPS, workers=2, name="video_noaudio.mp4"):
    scenes, captions, dur = build()
    t0 = 0.0 if t0 is None else t0
    t1 = dur if t1 is None else t1
    f0, f1 = int(round(t0 * fps)), int(round(t1 * fps))
    n = f1 - f0
    bounds = [f0 + n * k // workers for k in range(workers + 1)]
    parts = [OUT / f"part_{k}.mp4" for k in range(workers)]
    procs = [Process(target=_worker, args=(k, bounds[k], bounds[k + 1], fps, parts[k])) for k in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    lst = OUT / "parts.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(OUT / name)], check=True)
    print("wrote", OUT / name)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "still":
        render_stills([float(x) for x in sys.argv[2:]])
    elif mode == "video":
        args = [a for a in sys.argv[2:] if not a.startswith("--")]
        fps = FPS
        name = "video_noaudio.mp4"
        for a in sys.argv[2:]:
            if a.startswith("--fps="):
                fps = int(a.split("=")[1])
            if a.startswith("--name="):
                name = a.split("=")[1]
        render_video(float(args[0]) if args else None, float(args[1]) if len(args) > 1 else None, fps, name=name)
