"""Objective voice-take metrics: duration, words/sec, F0 stats (gender/expressiveness),
pause structure, spectral tilt, clipping, and Whisper word accuracy vs. the script."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
lines = {l["id"]: l["text"] for l in json.loads((ROOT / "script_v1.json").read_text())}


def f0_track(x, sr, fmin=70, fmax=400, hop=0.01, win=0.04):
    """Simple normalized-autocorrelation pitch tracker (YIN-lite)."""
    h, w = int(hop * sr), int(win * sr)
    lo, hi = int(sr / fmax), int(sr / fmin)
    out = []
    for s in range(0, len(x) - w - hi, h):
        fr = x[s:s + w + hi]
        if np.sqrt(np.mean(fr[:w] ** 2)) < 0.01:
            out.append(np.nan)
            continue
        a = fr[:w]
        best, bl = 0, 0
        e0 = np.dot(a, a)
        for lag in range(lo, hi):
            b = fr[lag:lag + w]
            r = np.dot(a, b) / (np.sqrt(e0 * np.dot(b, b)) + 1e-9)
            if r > best:
                best, bl = r, lag
        out.append(sr / bl if best > 0.6 else np.nan)
    return np.array(out)


def norm_words(s):
    s = s.lower().replace("1965", "nineteen sixty five").replace("1978", "nineteen seventy eight")
    s = s.replace("c.i.a.", "cia").replace("-", " ")
    return re.findall(r"[a-z']+", s)


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0] = range(len(r) + 1)
    d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / max(1, len(r))


def analyse(paths, asr=None):
    rows = []
    for p in paths:
        x, sr = sf.read(p)
        if x.ndim > 1:
            x = x.mean(1)
        lid = re.search(r"(L\d\d)", p.name).group(1)
        dur = len(x) / sr
        nw = len(norm_words(lines[lid]))
        f0 = f0_track(x, sr)
        v = f0[~np.isnan(f0)]
        semis = 12 * np.log2(v / np.median(v)) if len(v) else np.array([0])
        # pauses: 20ms frames below -40 dBFS relative to peak RMS
        fr = int(0.02 * sr)
        rms = np.sqrt(np.convolve(x ** 2, np.ones(fr) / fr, "same"))[::fr]
        thr = rms.max() * 0.03
        sil = rms < thr
        runs, c = [], 0
        for s in sil:
            if s:
                c += 1
            else:
                if c:
                    runs.append(c * 0.02)
                c = 0
        inner = [r for r in runs[1:] if r >= 0.15]
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        fq = np.fft.rfftfreq(len(x), 1 / sr)
        hf = spec[(fq > 4000) & (fq < 8000)].sum() / spec[(fq > 100) & (fq < 4000)].sum()
        row = dict(take=p.name, dur=round(dur, 2), wps=round(nw / dur, 2), f0_med=round(float(np.median(v)), 1) if len(v) else 0,
                   f0_range_st=round(float(np.percentile(semis, 95) - np.percentile(semis, 5)), 1),
                   pauses=len(inner), peak=round(float(np.abs(x).max()), 3), hf_ratio=round(float(hf), 3))
        if asr is not None:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(16000, sr)
            x16 = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
            segs, _ = asr.transcribe(x16, language="en", beam_size=5)
            hyp = " ".join(s.text for s in segs)
            row["wer"] = round(wer(lines[lid], hyp), 3)
            row["asr"] = hyp.strip()
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    pats = sys.argv[1:]
    paths = sorted(p for pat in pats for p in (ROOT / "takes").glob(pat))
    try:
        from faster_whisper import WhisperModel
        asr = WhisperModel("small.en", device="cpu", compute_type="int8")
    except Exception as e:
        print("no asr", e)
        asr = None
    analyse(paths, asr)
