"""Seam check: PSNR of final frames vs fresh renders at part boundaries (and neighbours)."""
import subprocess
import sys

import numpy as np

sys.path.insert(0, ".")
import render  # noqa: E402

M = sys.argv[1] if len(sys.argv) > 1 else "out/heart_in_the_ice_master.mp4"
sc, caps, d = render.build()


def frame(n):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{n / 30 + 0.001:.4f}", "-i", M, "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(1920, 1080, 3).astype(np.float32)


def psnr(a, b):
    m = ((a - b) ** 2).mean()
    return 99.0 if m == 0 else 10 * np.log10(255 ** 2 / m)


for n in (599, 600, 720, 1047):
    ref = render.compose(sc, caps, n / 30, n).astype(np.float32)
    row = {k: round(psnr(frame(k), ref), 1) for k in (n - 1, n, n + 1)}
    print(f"render of frame {n} vs final frames {row}", flush=True)
