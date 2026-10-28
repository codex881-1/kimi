"""Audio QA: spectrogram strip, short-term loudness, VO/SFX balance, and per-effect spectral checks."""
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from audio import synth as S  # noqa: E402

out = Path("/tmp/verify/audio")
out.mkdir(parents=True, exist_ok=True)
x, sr = sf.read(ROOT / "out" / "master.wav")
mono = x.mean(1)

# spectrogram (log-frequency), 20 px per second
hop, nfft = sr // 50, 4096
frames = np.lib.stride_tricks.sliding_window_view(np.pad(mono, (0, nfft)), nfft)[::hop]
spec = np.abs(np.fft.rfft(frames * np.hanning(nfft), axis=1))
freqs = np.fft.rfftfreq(nfft, 1 / sr)
logf = np.geomspace(30, 16000, 360)
idx = np.searchsorted(freqs, logf)
img = 20 * np.log10(spec[:, idx] / spec.max() + 1e-9)
img = np.clip((img + 100) / 90, 0, 1)[:, ::-1].T
img = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
for s in range(0, int(len(mono) / sr) + 1, 5):
    cv2.line(img, (s * 50, 0), (s * 50, 12), (255, 255, 255), 1)
    cv2.putText(img, str(s), (s * 50 + 2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
cv2.imwrite(str(out / "spectrogram.png"), img)

# short-term loudness per second (3 s window)
meter = pyln.Meter(sr)
st = []
for s in range(0, int(len(x) / sr) - 2):
    st.append(meter.integrated_loudness(x[s * sr:(s + 3) * sr]))
print("short-term LUFS (3s) per second:", " ".join(f"{v:.0f}" for v in st))
print("integrated", round(meter.integrated_loudness(x), 2))

# VO vs effects balance from stems
vo, _ = sf.read(ROOT / "vo" / "vo_raw.wav")
for k in ("keys", "ui", "fx"):
    y, _ = sf.read(ROOT / "out" / f"stem_{k}.wav")
    print(f"stem {k:5s} peak {20*np.log10(np.abs(y).max()+1e-9):6.1f} dBFS  rms {20*np.log10(np.sqrt((y**2).mean())+1e-9):6.1f} dBFS")


def centroid(s):
    m = s.mean(1)
    sp = np.abs(np.fft.rfft(m * np.hanning(len(m))))
    f = np.fft.rfftfreq(len(m), 1 / S.SR)
    return (sp * f).sum() / sp.sum()


for name, ev in (("key", {"seed": 3}), ("key-space", {"seed": 4, "space": True}), ("cream", {"seed": 5, "weight": 0.8}),
                 ("heart", {}), ("flap", {"seed": 2}), ("blip", {"freq": 1400})):
    s, _ = S.render_event({"kind": name.split("-")[0], "t": 0, **ev})
    print(f"{name:10s} spectral centroid {centroid(s):7.0f} Hz   len {len(s)/S.SR*1000:5.0f} ms")
