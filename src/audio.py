#!/usr/bin/env python3
"""Deterministic sound mix for the 68 s vertical documentary (no music).

Pipeline: edge-tts narration (cached) -> retimed, voiced-processed clips ->
synthesised tactile effects (keyboard, text taps, mechanical cues, air) ->
buses with sends, sidechain ducking and a safety limiter -> PCM16 wav + cues.json.

Run from the repo root:  .venv/bin/python src/audio.py [--refresh-tts] [--offline]
See docs/sound-design.md for the design notes.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import scipy.signal as sps
import soundfile as sf
from scipy.ndimage import minimum_filter1d, uniform_filter1d

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
SR = 48_000
DURATION = 68.0
NSAMP = int(round(DURATION * SR))
GLOBAL_SEED = 20240721

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets" / "audio"
CACHE_DIR = OUT_DIR / "tts_cache"
CLIP_DIR = OUT_DIR / "narration"

VOICE = "en-US-AvaMultilingualNeural"
TTS_FORMAT = "audio-24khz-96kbitrate-mono-mp3"
TTS_SR = 24_000

# Loudness / master targets
VOICE_TARGET_LUFS = -17.0      # narration-only integrated loudness (stereo)
CEILING_DBFS = -1.5            # limiter ceiling, sample peak

# Narration. `rate` is the service speaking-rate in percent (found with --fit);
# `gap` is the pause between sentences in seconds; `inner` scales comma/colon pauses.
# `starts` places each sentence separately (used when sentences must land on cues).
NARRATION = [
    dict(id="N1", start=0.80, max_end=6.80, rate=32, gap=0.20, inner=0.75,
         sentences=["Nineteen sixty-nine.", "A black arrow heads for China.", "No pilot.",
                    "Just a camera, and a route locked in its memory."]),
    dict(id="N2", start=7.35, max_end=14.20, rate=12, gap=0.26, inner=0.85,
         sentences=["The D-twenty-one.", "Built by Lockheed for the CIA.",
                    "A B-fifty-two carried it up.", "A rocket booster pushed it past Mach three."]),
    dict(id="N3", start=14.85, max_end=21.20, rate=28, gap=0.20, inner=0.55,
         sentences=["Its target: Lop Nor, China's nuclear test site.",
                    "The drone followed the route alone.", "No joystick.", "No second chances."]),
    dict(id="N4", start=21.85, max_end=27.20, rate=12, gap=0.28, inner=0.85,
         sentences=["After photographing the site, it turned for the Pacific.",
                    "The film sat inside a small recovery pod."]),
    dict(id="N5", start=27.85, max_end=35.70, rate=6, gap=0.30, inner=0.9,
         sentences=["Over the ocean, the pod ejected.",
                    "A parachute opened, and a C-one-thirty moved in to catch it.",
                    "The drone itself was expendable."]),
    dict(id="N6", start=36.45, max_end=37.80, rate=-8, gap=0.30, inner=1.0,
         sentences=["That was the plan."]),
    dict(id="N7", start=38.40, max_end=50.70, rate=4, gap=0.40, inner=0.9,
         starts=[38.40, 41.40, 44.55, 47.70],
         sentences=["The first drone missed its turn.", "The second returned, but its parachute failed.",
                    "On the third, the pod was lost at sea.", "The fourth crashed inside China."]),
    dict(id="N8", start=51.35, max_end=56.70, rate=8, gap=0.28, inner=0.85,
         sentences=["Four missions.", "No usable pictures.",
                    "In nineteen seventy-one, the program was shut down."]),
    dict(id="N9", start=57.30, max_end=64.70, rate=8, gap=0.30, inner=0.85,
         sentences=["But one wreck survived.",
                    "Decades later, China put a D-twenty-one on display: a secret mission, hiding in plain sight."]),
]
END_MARGIN = 0.05              # narration must end this much before max_end

TYPED = [
    ("T01", "CHINA / 1969", 0.25),
    ("T02", "D-21B", 7.00),
    ("T03", "LOP NOR", 14.60),
    ("T04", "MISSION 01", 38.10),
    ("T05", "MISSION 02", 41.20),
    ("T06", "MISSION 03", 44.30),
    ("T07", "MISSION 04", 47.40),
    ("T08", "FOUR MISSIONS", 51.20),
    ("T09", "ARTIFACT / D-21B", 57.10),
    ("T10", "SENIOR BOWL", 65.00),
]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-12))


def gain(d):
    return 10.0 ** (d / 20.0)


def rng_for(name: str) -> np.random.Generator:
    return np.random.default_rng([GLOBAL_SEED, zlib.crc32(name.encode())])


def tarr(n, sr=SR):
    return np.arange(n) / sr


def sec(x):
    return int(round(x * SR))


def fade_edges(x, fi=0.003, fo=0.01, sr=SR):
    x = x.copy()
    ni, no = min(len(x), int(fi * sr)), min(len(x), int(fo * sr))
    if ni > 1:
        r = 0.5 - 0.5 * np.cos(np.pi * np.arange(ni) / ni)
        x[:ni] *= r if x.ndim == 1 else r[:, None]
    if no > 1:
        r = 0.5 + 0.5 * np.cos(np.pi * np.arange(no) / no)
        x[-no:] *= r if x.ndim == 1 else r[:, None]
    return x


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def biquad(kind, f0, q=0.7071, gain_db=0.0, fs=SR):
    """RBJ cookbook biquad as one sos row."""
    f0 = min(f0, 0.45 * fs)
    w0 = 2 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    A = 10 ** (gain_db / 40)
    al = sw / (2 * q)
    sA = np.sqrt(A)
    if kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + al, -2 * cw, 1 - al]
    elif kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + al, -2 * cw, 1 - al]
    elif kind == "bp":
        b = [al, 0, -al]
        a = [1 + al, -2 * cw, 1 - al]
    elif kind == "peak":
        b = [1 + al * A, -2 * cw, 1 - al * A]
        a = [1 + al / A, -2 * cw, 1 - al / A]
    elif kind == "ls":
        b = [A * ((A + 1) - (A - 1) * cw + 2 * sA * al), 2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - 2 * sA * al)]
        a = [(A + 1) + (A - 1) * cw + 2 * sA * al, -2 * ((A - 1) + (A + 1) * cw),
             (A + 1) + (A - 1) * cw - 2 * sA * al]
    elif kind == "hs":
        b = [A * ((A + 1) + (A - 1) * cw + 2 * sA * al), -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - 2 * sA * al)]
        a = [(A + 1) - (A - 1) * cw + 2 * sA * al, 2 * ((A - 1) - (A + 1) * cw),
             (A + 1) - (A - 1) * cw - 2 * sA * al]
    else:
        raise ValueError(kind)
    b, a = np.array(b, float), np.array(a, float)
    return np.r_[b / a[0], 1.0, a[1] / a[0], a[2] / a[0]]


def apply_sos(x, sos, zero_phase=False):
    sos = np.atleast_2d(sos)
    return sps.sosfiltfilt(sos, x, axis=0) if zero_phase else sps.sosfilt(sos, x, axis=0)


def butter(x, kind, fc, order=2, zero_phase=False, fs=SR):
    sos = sps.butter(order, fc, btype=kind, fs=fs, output="sos")
    return apply_sos(x, sos, zero_phase)


def bandpass(x, lo, hi, order=2, zero_phase=False):
    return butter(x, "bandpass", [lo, min(hi, 0.45 * SR)], order, zero_phase)


def tv_filter(x, fc, kind="lp", q=0.7071, block=48):
    """Block-wise time-varying biquad; `fc` is a scalar or per-sample array."""
    n = len(x)
    fc = np.broadcast_to(np.asarray(fc, float), (n,))
    out = np.empty(n)
    zi = np.zeros((1, 2))
    for i in range(0, n, block):
        sos = biquad(kind, float(fc[i]), q)[None, :]
        out[i:i + block], zi = sps.sosfilt(sos, x[i:i + block], zi=zi)
    return out


# --------------------------------------------------------------------------- #
# Noise and curves
# --------------------------------------------------------------------------- #
def colored_noise(n, rng, alpha=1.0, sr=SR):
    """Unit-variance noise with PSD ~ f^-alpha (0 white, 1 pink, 2 brown)."""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    f[0] = f[1] if len(f) > 1 else 1.0
    X *= f ** (-alpha / 2)
    X[0] = 0
    x = np.fft.irfft(X, n)
    return x / (np.std(x) + 1e-12)


def slow_curve(n, rng, rate_hz, sr=SR):
    """Smooth zero-mean random curve (~unit std) varying at roughly `rate_hz`."""
    k = max(4, int(n / sr * rate_hz) + 4)
    pts = rng.standard_normal(k)
    xs = np.linspace(0, n - 1, k)
    y = np.interp(np.arange(n), xs, pts)
    w = max(1, int(sr / rate_hz / 2))
    y = uniform_filter1d(y, w, mode="nearest")
    return (y - y.mean()) / (y.std() + 1e-12)


def interp_curve(points, n=NSAMP, sr=SR, smooth=0.0):
    """Piecewise-linear curve through (time, value) points, optional smoothing (s)."""
    pts = np.array(points, float)
    y = np.interp(np.arange(n) / sr, pts[:, 0], pts[:, 1])
    if smooth > 0:
        y = uniform_filter1d(y, max(1, int(smooth * sr)), mode="nearest")
    return y


def exp_decay(n, tau, sr=SR):
    return np.exp(-np.arange(n) / (tau * sr))


def norm_peak(x, dbfs):
    p = np.max(np.abs(x))
    return x * (gain(dbfs) / p) if p > 0 else x


def rms_db(x):
    return 10 * np.log10(np.mean(np.square(x)) + 1e-20)


def to_stereo(x, pan=0.0):
    """Mono -> stereo, constant-power pan (centre = -3 dB per side)."""
    if x.ndim == 2:
        return x
    th = (pan + 1) * np.pi / 4
    return np.stack([x * np.cos(th), x * np.sin(th)], axis=1)


# --------------------------------------------------------------------------- #
# Loudness (ITU-R BS.1770-4) and true peak
# --------------------------------------------------------------------------- #
_K_PRE = (np.array([1.53512485958697, -2.69169618940638, 1.19839281085285]),
          np.array([1.0, -1.69065929318241, 0.73248077421585]))
_K_RLB = (np.array([1.0, -2.0, 1.0]), np.array([1.0, -1.99004745483398, 0.99007225036621]))


def k_weight(x):
    y = sps.lfilter(*_K_PRE, x, axis=0)
    return sps.lfilter(*_K_RLB, y, axis=0)


def lufs(x, sr=SR):
    """Integrated loudness with absolute (-70) and relative (-10 LU) gating."""
    if sr != 48000:
        raise ValueError("48 kHz only")
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    z = k_weight(x)
    blk, hop = int(0.4 * sr), int(0.1 * sr)
    if len(z) < blk:
        return -120.0
    cs = np.concatenate([np.zeros((1, z.shape[1])), np.cumsum(z * z, axis=0)])
    idx = np.arange(0, len(z) - blk + 1, hop)
    ms = ((cs[idx + blk] - cs[idx]) / blk).sum(axis=1)
    l = -0.691 + 10 * np.log10(ms + 1e-20)
    keep = l > -70
    if not keep.any():
        return -120.0
    rel = -0.691 + 10 * np.log10(ms[keep].mean()) - 10
    keep &= l > rel
    return float(-0.691 + 10 * np.log10(ms[keep].mean())) if keep.any() else -120.0


def true_peak_db(x):
    y = sps.resample_poly(x, 4, 1, axis=0)
    return float(db(np.max(np.abs(y))))


# --------------------------------------------------------------------------- #
# Dynamics (control-rate, vectorised where possible)
# --------------------------------------------------------------------------- #
def _ctrl_env(x, hop):
    m = len(x) // hop
    return np.abs(x[:m * hop]).reshape(m, hop).max(axis=1), m


def _upsample_ctrl(g, hop, n):
    xs = (np.arange(len(g)) + 0.5) * hop
    return np.interp(np.arange(n), xs, g)


def compressor(x, thr_db, ratio, attack, release, knee=6.0, hop=24):
    """Feed-forward peak compressor on a mono signal; returns (y, mean_gr_db)."""
    lvl, m = _ctrl_env(x, hop)
    ldb = db(lvl)
    over = ldb - thr_db
    slope = 1 - 1 / ratio
    gr = np.where(over < -knee / 2, 0.0,
                  np.where(over > knee / 2, over * slope, (over + knee / 2) ** 2 / (2 * knee) * slope))
    ca, cr = np.exp(-hop / (attack * SR)), np.exp(-hop / (release * SR))
    s = np.zeros(m)
    cur = 0.0
    for i in range(m):                      # attack when GR grows, release when it shrinks
        c = ca if gr[i] > cur else cr
        cur = c * cur + (1 - c) * gr[i]
        s[i] = cur
    g = gain(-_upsample_ctrl(s, hop, len(x)))
    return x * g, float(s.mean())


def deess(x, lo=5000, hi=9500, rel_db=4.0, max_red=4.0, hop=24):
    band = bandpass(x, lo, hi, 2)
    lvl, m = _ctrl_env(band, hop)
    ldb = db(lvl)
    ref = np.percentile(ldb[ldb > ldb.max() - 40], 75) if (ldb > ldb.max() - 40).any() else ldb.max()
    gr = np.clip((ldb - (ref + rel_db)) * 0.7, 0, max_red)
    ca, cr = np.exp(-hop / (0.001 * SR)), np.exp(-hop / (0.025 * SR))
    s = np.zeros(m)
    cur = 0.0
    for i in range(m):
        c = ca if gr[i] > cur else cr
        cur = c * cur + (1 - c) * gr[i]
        s[i] = cur
    g = gain(-_upsample_ctrl(s, hop, len(x)))
    return x - band + band * g


def safety_limiter(x, ceiling_db, look=0.025):
    """Look-ahead peak limiter, channels linked. min-filter + moving average keeps
    the gain curve <= the required gain everywhere, so no overs and no zipper."""
    ceil = gain(ceiling_db)
    p = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    r = np.minimum(1.0, ceil / np.maximum(p, 1e-9))
    w = max(3, int(look * SR))
    g = uniform_filter1d(minimum_filter1d(r, size=2 * w + 1, mode="nearest"), size=w + 1, mode="nearest")
    y = x * (g[:, None] if x.ndim == 2 else g)
    return np.clip(y, -ceil, ceil), float(db(g.min()))


# --------------------------------------------------------------------------- #
# Narration: TTS (cached) -> sentence chunks -> retimed units -> voice chain
# --------------------------------------------------------------------------- #
def _tts_key(text, rate):
    return hashlib.sha1(f"{VOICE}|{TTS_FORMAT}|{rate}|{text}".encode()).hexdigest()[:10]


async def _edge_fetch(text, rate):
    """Minimal edge-tts client: same protocol as the library, but 96 kbps output
    (the library hard-codes 48 kbps) and word boundaries."""
    import aiohttp
    import edge_tts.communicate as C
    from edge_tts.constants import SEC_MS_GEC_VERSION, WSS_HEADERS, WSS_URL
    from edge_tts.drm import DRM

    tc = C.TTSConfig(VOICE, f"{rate:+d}%", "+0%", "+0Hz", "WordBoundary")
    audio, words = bytearray(), []
    timeout = aiohttp.ClientTimeout(total=90)
    async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as s:
        url = (f"{WSS_URL}&ConnectionId={C.connect_id()}&Sec-MS-GEC={DRM.generate_sec_ms_gec()}"
               f"&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}")
        async with s.ws_connect(url, compress=15, headers=DRM.headers_with_muid(WSS_HEADERS),
                                ssl=getattr(C, "_SSL_CTX", None)) as ws:
            await ws.send_str(
                "X-Timestamp:" + C.date_to_string() + "\r\nContent-Type:application/json; charset=utf-8\r\n"
                'Path:speech.config\r\n\r\n{"context":{"synthesis":{"audio":{"metadataoptions":'
                '{"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"true"},'
                '"outputFormat":"' + TTS_FORMAT + '"}}}}\r\n')
            await ws.send_str(C.ssml_headers_plus_data(C.connect_id(), C.date_to_string(), C.mkssml(tc, text)))
            async for m in ws:
                if m.type == aiohttp.WSMsgType.TEXT:
                    raw = m.data.encode()
                    hdr, data = C.get_headers_and_data(raw, raw.find(b"\r\n\r\n"))
                    path = hdr.get(b"Path")
                    if path == b"audio.metadata":
                        for md in json.loads(data)["Metadata"]:
                            if md["Type"] == "WordBoundary":
                                d = md["Data"]
                                words.append([d["text"]["Text"], d["Offset"] / 1e7, d["Duration"] / 1e7])
                    elif path == b"turn.end":
                        break
                elif m.type == aiohttp.WSMsgType.BINARY:
                    hl = int.from_bytes(m.data[:2], "big")
                    _, data = C.get_headers_and_data(m.data, hl)
                    audio += data
                elif m.type == aiohttp.WSMsgType.ERROR:
                    raise RuntimeError(str(m.data))
    if not audio:
        raise RuntimeError("no audio received")
    return bytes(audio), words


async def _edge_fetch_public(text, rate):
    """Fallback through the public library API (48 kbps)."""
    import edge_tts
    comm = edge_tts.Communicate(text, VOICE, rate=f"{rate:+d}%", boundary="WordBoundary")
    audio, words = bytearray(), []
    async for ch in comm.stream():
        if ch["type"] == "audio":
            audio += ch["data"]
        elif ch["type"] == "WordBoundary":
            words.append([ch["text"], ch["offset"] / 1e7, ch["duration"] / 1e7])
    return bytes(audio), words


def tts_clip(cid, text, rate, refresh=False, offline=False):
    """Return (mono float64 at 24 kHz, words) using the on-disk cache when valid."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _tts_key(text, rate)
    mp3, js = CACHE_DIR / f"{cid}_{key}.mp3", CACHE_DIR / f"{cid}_{key}.json"
    if refresh or not (mp3.exists() and js.exists()):
        if offline:
            raise SystemExit(f"TTS cache miss for {cid} (rate {rate:+d}%) and --offline given")
        last = None
        for attempt in range(4):
            try:
                try:
                    data, words = asyncio.run(_edge_fetch(text, rate))
                except Exception as e:                     # protocol drift -> public API
                    print(f"  [{cid}] custom client failed ({type(e).__name__}); using library client")
                    data, words = asyncio.run(_edge_fetch_public(text, rate))
                break
            except Exception as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        else:
            raise SystemExit(f"TTS failed for {cid}: {last}")
        mp3.write_bytes(data)
        js.write_text(json.dumps(dict(voice=VOICE, format=TTS_FORMAT, rate=rate, text=text, words=words)))
    meta = json.loads(js.read_text())
    x, sr = sf.read(io.BytesIO(mp3.read_bytes()), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != TTS_SR:
        x = sps.resample_poly(x, TTS_SR, sr)
    return x, meta["words"]


def env_db(x, sr, win=0.004):
    n = max(1, int(win * sr))
    return 10 * np.log10(np.maximum(uniform_filter1d(x * x, n, mode="nearest"), 0.0) + 1e-20)


def longest_run(mask):
    if not mask.any():
        return 0, 0
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    k = int(np.argmax(e - s))
    return int(s[k]), int(e[k] - s[k])


def split_sentences(x, words, sentences, sr=TTS_SR):
    """Split a synthesised block at the silent gap between sentences.
    Returns chunks with the voiced onset/end in samples (energy based)."""
    env = env_db(x, sr)
    pk = env.max()
    thr = pk - 45.0
    counts = [len(s.split()) for s in sentences]
    if sum(counts) != len(words):
        raise SystemExit(f"word-boundary count {len(words)} != text words {sum(counts)}")
    last = np.cumsum(counts) - 1
    first = np.concatenate([[0], last[:-1] + 1])
    cuts = [0]
    for i in range(len(sentences) - 1):
        ta = words[last[i]][1] + words[last[i]][2]
        tb = words[first[i + 1]][1]
        a, b = max(0, int((ta - 0.15) * sr)), min(len(x), int((tb + 0.15) * sr))
        s0, ln = longest_run(env[a:b] < thr)
        cuts.append(a + s0 + ln // 2 if ln else (a + b) // 2)
    cuts.append(len(x))
    chunks = []
    for i, text in enumerate(sentences):
        a, b = cuts[i], cuts[i + 1]
        act = np.flatnonzero(env[a:b] > thr)
        on, off = a + int(act[0]), a + int(act[-1])
        wsel = words[first[i]:last[i] + 1]
        chunks.append(dict(text=text, a=a, b=b, on=on, off=off, words=wsel))
    return chunks, pk


def retime_chunk(x, ch, pk, inner, sr=TTS_SR, pre=0.012, post=0.045):
    """Cut a sentence out of the block, shortening its inner (comma/colon) pauses.
    Returns dict(x, dur, words) with times relative to the voiced onset."""
    env = env_db(x, sr)
    a, b = max(ch["a"], ch["on"] - int(pre * sr)), min(ch["b"], ch["off"] + int(post * sr))
    seg, e = x[a:b].copy(), env[a:b]
    on0, off0 = ch["on"] - a, ch["off"] - a
    cuts = []
    if inner < 1.0:
        quiet = e < pk - 34.0
        quiet[:on0 + int(0.06 * sr)] = False
        quiet[max(0, off0 - int(0.06 * sr)):] = False
        d = np.diff(np.concatenate([[0], quiet.astype(np.int8), [0]]))
        for s0, e0 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            ln = (e0 - s0) / sr
            if ln >= 0.14:
                new = max(0.09, ln * inner)
                cut = int((ln - new) * sr)
                c0 = int((s0 + e0) / 2 - cut / 2)
                if cut > 0.02 * sr:
                    cuts.append((c0, c0 + cut))
    parts, pos, removed = [], 0, 0
    xf = int(0.004 * sr)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(xf) / xf)
    for c0, c1 in cuts:
        p = seg[pos:c0].copy()
        p[-xf:] *= ramp[::-1]
        parts.append(p)
        pos = c1
        removed += c1 - c0
    tail = seg[pos:].copy()
    if cuts:
        tail[:xf] *= ramp
    parts.append(tail)
    out = np.concatenate(parts)

    def remap(t):                          # original block time -> chunk time after cuts
        s = int(round(t * sr)) - a
        sub = sum(min(max(s - c0, 0), c1 - c0) for c0, c1 in cuts)
        return (s - sub) / sr

    onset = remap(ch["on"] / sr)
    words = [[w[0], remap(w[1]) - onset, remap(w[1] + w[2]) - onset] for w in ch["words"]]
    return dict(x=out, pre=onset, dur=remap(ch["off"] / sr) - onset, words=words,
                removed=removed / sr, text=ch["text"])


def voice_chain(x24):
    """24 kHz mono TTS audio -> warm, even 48 kHz mono narration (no pitch/time warping)."""
    x = sps.resample_poly(x24, 2, 1, window=("kaiser", 9.0))
    x = apply_sos(x, sps.butter(2, 78, "highpass", fs=SR, output="sos"))
    x = deess(x)
    eq = np.vstack([biquad("ls", 170, 0.7, 1.6), biquad("peak", 330, 1.0, -1.0),
                    biquad("peak", 3100, 0.9, 1.3), biquad("hs", 8500, 0.7, -1.2)])
    x = apply_sos(x, eq)
    x = apply_sos(x, sps.butter(4, 11000, "lowpass", fs=SR, output="sos"))
    x, _ = compressor(x, thr_db=-16.0, ratio=2.0, attack=0.006, release=0.14, knee=8.0)
    return x


def build_unit(chunks, gaps):
    """Place retimed chunks (24 kHz) one after another. Returns (x, onset_s, spans, words)."""
    on0 = chunks[0]["pre"]
    t_on, starts = on0, []
    for k, c in enumerate(chunks):
        starts.append(t_on)
        t_on += c["dur"] + (gaps[k] if k < len(gaps) else 0.0)
    total = int((starts[-1] + chunks[-1]["dur"] + 0.1) * TTS_SR) + 1
    x = np.zeros(total)
    spans, words = [], []
    for k, c in enumerate(chunks):
        i0 = int(round((starts[k] - c["pre"]) * TTS_SR))
        n = min(len(c["x"]), total - i0)
        x[i0:i0 + n] += c["x"][:n]
        spans.append(dict(text=c["text"], on=starts[k] - on0, off=starts[k] - on0 + c["dur"]))
        words += [dict(w=w[0], on=starts[k] - on0 + w[1], off=starts[k] - on0 + w[2]) for w in c["words"]]
    return x, on0, spans, words


def narration_units(refresh=False, offline=False, verbose=True):
    """Synthesise (or load) every narration block and return processed 48 kHz units.
    Each unit: id, start (s, first audible sound), x (mono, onset at `pre` seconds),
    spans/words relative to the unit onset, loudness-matched to VOICE_TARGET_LUFS."""
    units = []
    for cfg in NARRATION:
        text = " ".join(cfg["sentences"])
        x, words = tts_clip(cfg["id"], text, cfg["rate"], refresh, offline)
        chunks, pk = split_sentences(x, words, cfg["sentences"])
        rc = [retime_chunk(x, c, pk, cfg["inner"]) for c in chunks]
        budget = cfg["max_end"] - cfg["start"] - END_MARGIN
        if "starts" in cfg:
            groups = [([r], [], s) for r, s in zip(rc, cfg["starts"])]
            letters = "abcdefgh"
        else:
            speech = sum(r["dur"] for r in rc)
            n_gap = len(rc) - 1
            g = cfg["gap"] if n_gap == 0 else min(cfg["gap"], max(0.14, (budget - speech) / n_gap))
            groups = [(rc, [g] * n_gap, cfg["start"])]
            letters = ""
        for gi, (rcs, gaps, start) in enumerate(groups):
            x24, on0, spans, wlist = build_unit(rcs, gaps)
            y = voice_chain(x24)
            m = lufs(np.stack([y, y], 1))
            y *= gain(VOICE_TARGET_LUFS - m)
            uid = cfg["id"] + (letters[gi] if letters else "")
            end = start + spans[-1]["off"]
            units.append(dict(id=uid, block=cfg["id"], start=start, end=end, max_end=cfg["max_end"],
                              x=y, pre=on0, spans=spans, words=wlist, rate=cfg["rate"],
                              text=" ".join(s["text"] for s in spans), gap=(gaps[0] if gaps else None),
                              removed=sum(r["removed"] for r in rcs)))
            if verbose:
                ok = "ok " if end <= cfg["max_end"] - END_MARGIN + 1e-6 else "OVER"
                print(f"  {uid:4s} rate {cfg['rate']:+3d}%  {start:6.2f} -> {end:6.2f} (max {cfg['max_end']:.2f}) "
                      f"{ok} gap {gaps[0] if gaps else 0:.2f}s  loud {m:5.1f} LUFS pre-norm")
    return units


# --------------------------------------------------------------------------- #
# Mixing infrastructure
# --------------------------------------------------------------------------- #
class Bus:
    """Stereo bus with optional per-event reverb sends (mono send buffers)."""

    def __init__(self, name):
        self.name = name
        self.dry = np.zeros((NSAMP, 2))
        self.sends = {}

    def add(self, sig, t, gain_db=0.0, pan=0.0, sends=None, anchor=0.0):
        """Add `sig` so that its anchor sample lands at time `t` (s)."""
        s = to_stereo(sig, pan) * gain(gain_db)
        i0 = int(round((t - anchor) * SR))
        a, b = max(0, i0), min(NSAMP, i0 + len(s))
        if b <= a:
            return
        seg = s[a - i0:b - i0]
        self.dry[a:b] += seg
        for name, send_db in (sends or {}).items():
            buf = self.sends.setdefault(name, np.zeros(NSAMP))
            buf[a:b] += seg.sum(axis=1) * 0.5 * gain(send_db)

    def render(self, irs):
        out = self.dry.copy()
        for name, buf in self.sends.items():
            ir = irs[name]
            for ch in range(2):
                out[:, ch] += sps.fftconvolve(buf, ir[:, ch])[:NSAMP]
        return out


def make_ir(name, rt, predelay, length, er_gain=0.5, er_span=0.045, n_er=9):
    """Synthetic stereo room IR. `rt` = (low, mid, high) RT60 in s. Unit energy per channel."""
    rng = rng_for("ir-" + name)
    n = int(length * SR)
    t = tarr(n)
    ir = np.zeros((n, 2))
    bands = ((60, 400, rt[0], 1.0), (400, 3000, rt[1], 1.0), (3000, 9000, rt[2], 0.55))
    for ch in range(2):
        tail = np.zeros(n)
        for lo, hi, r, w in bands:
            nz = bandpass(rng.standard_normal(n), lo, hi, 2)
            tail += w * nz / nz.std() * np.exp(-6.9078 * t / r)
        td = np.maximum(t - predelay, 0.0)
        tail *= (1 - np.exp(-td / 0.008)) * (t >= predelay)
        er = np.zeros(n)
        for k in range(n_er):
            i = int((predelay + rng.uniform(0.002, er_span)) * SR)
            er[i] += er_gain * rng.choice([-1, 1]) * np.exp(-k / 5) * rng.uniform(0.6, 1.0)
        er = butter(er, "lowpass", 5500, 2)
        ir[:, ch] = tail + er * tail.std() * 14
    ir /= np.sqrt((ir ** 2).sum(axis=0, keepdims=True))
    return ir


def build_irs():
    return dict(
        tight=make_ir("tight", (0.16, 0.12, 0.07), 0.0015, 0.30, 0.6, 0.012, 5),
        room=make_ir("room", (0.42, 0.32, 0.18), 0.006, 0.9, 0.5, 0.030, 8),
        hall=make_ir("hall", (2.3, 1.7, 0.95), 0.022, 3.4, 0.35, 0.09, 10),
    )


# --------------------------------------------------------------------------- #
# Synthesis primitives
# --------------------------------------------------------------------------- #
def ring(t, freqs, taus, amps, phase=0.0):
    """Sum of exponentially damped sinusoids (modal synthesis)."""
    f = np.asarray(freqs, float)[:, None]
    tau = np.asarray(taus, float)[:, None]
    a = np.asarray(amps, float)[:, None]
    return (a * np.exp(-t[None, :] / tau) * np.sin(2 * np.pi * f * t[None, :] + phase)).sum(axis=0)


def soft_pulse(ms):
    p = np.hanning(max(3, int(ms * 1e-3 * SR)) + 2)[1:-1]
    return p / p.sum()


def band_noise(n, rng, lo, hi, order=2, color=0.0):
    x = colored_noise(n, rng, color) if color else rng.standard_normal(n)
    x = bandpass(x, lo, hi, order)
    return x / (x.std() + 1e-12)


def ad_env(n, attack, tau):
    """Raised-cosine attack, exponential decay."""
    t = tarr(n)
    a = np.where(t < attack, 0.5 - 0.5 * np.cos(np.pi * np.minimum(t / max(attack, 1e-6), 1.0)), 1.0)
    return a * np.exp(-t / tau)


def glide_sine(n, f0, f1, tau_g, amp_env=None):
    t = tarr(n)
    f = f1 + (f0 - f1) * np.exp(-t / tau_g)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return y if amp_env is None else y * amp_env


def put(buf, sig, t0, g=1.0):
    """Add a mono signal into a mono buffer at time t0 (s); safe at the edges."""
    i0 = int(round(t0 * SR))
    a, b = max(0, i0), min(len(buf), i0 + len(sig))
    if b > a:
        buf[a:b] += sig[a - i0:b - i0] * g


def mono_buf(dur):
    return np.zeros(int(dur * SR))


def metallic(rng, f0, dur, size=1.0, bright=1.0, ratios=(1.0, 2.76, 5.4, 8.93, 13.3)):
    """Struck-metal body: free-bar ratios, damping rising with frequency."""
    n = int(dur * SR)
    t = tarr(n)
    r = np.array(ratios) * (1 + 0.012 * rng.standard_normal(len(ratios)))
    f = f0 * r
    keep = f < 0.42 * SR
    taus = 0.11 * size / (1 + 0.9 * np.arange(len(r)) ** 1.15)
    amps = (1.0 / (1 + 0.55 * np.arange(len(r)))) * np.array([1, .8, .6, .45, .3])[:len(r)] ** (1 / bright)
    y = ring(t, f[keep], taus[keep], amps[keep], rng.uniform(0, 0.3))
    return y


# --------------------------------------------------------------------------- #
# Keyboard: deep, creamy "thock" (lubed linear, foam-damped, gasket-style case)
# --------------------------------------------------------------------------- #
_ROWS = ["`1234567890-=", "qwertyuiop[]", "asdfghjkl;'", "zxcvbnm,./"]
_ROW_OFF = [0.0, 1.5, 1.8, 2.3]
KEY_X = {}
for _r, (_row, _off) in enumerate(zip(_ROWS, _ROW_OFF)):
    for _c, _ch in enumerate(_row):
        KEY_X[_ch] = _off + _c
KEY_X[" "] = 6.0
_THOCK_PEAK = None


def keystroke(rng, ch, vel=1.0, hold=0.075):
    """One key press: bottom-out thock + softer top-out tick. Returns (mono, pan)."""
    key = ch.lower()
    x = KEY_X.get(key, 6.0)
    spread = (x - 6.0) / 6.5
    space = key == " "
    pitch = (1.0 + 0.045 * spread ** 2 + 0.015 * rng.standard_normal()) * (0.74 if space else 1.0)
    stretch = 1.4 if space else 1.0
    dur = 0.42 if space else 0.34
    n = int((dur + hold) * SR)
    t = tarr(n)
    base = np.array([118.0, 176.0, 262.0, 391.0, 588.0, 902.0]) * pitch
    amps = np.array([0.55, 1.0, 0.78, 0.42, 0.20, 0.08])
    taus = np.array([0.048, 0.040, 0.030, 0.022, 0.014, 0.009]) * stretch
    body = np.zeros(n)
    for f, a, tau in zip(base, amps, taus):
        inst = f * (1 + 0.045 * np.exp(-t / 0.018))              # soft pitch settle = "thock"
        body += a * np.exp(-t / tau) * np.sin(2 * np.pi * np.cumsum(inst) / SR + rng.uniform(0, 0.4))
    body = np.convolve(body, soft_pulse(1.1 + 0.4 * (1 - vel)))[:n]
    tap = band_noise(n, rng, 650, 3200) * exp_decay(n, 0.0016) * 0.13
    cream = band_noise(n, rng, 170, 620) * ad_env(n, 0.002, 0.030 * stretch) * 0.20
    sub = glide_sine(n, 98 * pitch, 60 * pitch, 0.022) * np.exp(-t / (0.055 * stretch)) * 0.32
    down = body + tap + cream + sub
    if space:                                                     # stabiliser wire rattle
        for off, f in ((0.0006, 2100), (0.0031, 3050), (0.0074, 4300)):
            rat = ring(t[:int(0.02 * SR)], [f], [0.0035], [1.0]) * 0.05
            put(down, rat, off)
    up = ring(t, base[:4] * 1.22, taus[:4] * 0.45, amps[:4] * 0.18)
    up = np.convolve(up, soft_pulse(0.7))[:n] + band_noise(n, rng, 1800, 4200) * exp_decay(n, 0.0014) * 0.012
    y = down + np.roll(up, int(hold * SR)) * (np.arange(n) >= int(hold * SR))
    y = butter(y, "lowpass", 5200, 2)
    return y * vel, float(np.clip(0.22 * spread, -0.2, 0.2))


def thock_ref_peak():
    global _THOCK_PEAK
    if _THOCK_PEAK is None:
        y, _ = keystroke(np.random.default_rng(1), "e")
        _THOCK_PEAK = float(np.max(np.abs(y)))
    return _THOCK_PEAK


def typing_schedule(text, t0, rng, base=0.094):
    """Human-ish key-down times: jittered cadence, small hesitations at spaces."""
    out, t = [], t0
    for i, ch in enumerate(text):
        out.append((ch, t))
        dt = float(np.clip(base * (1 + 0.16 * rng.standard_normal()), 0.066, 0.145))
        if ch == " ":
            dt += 0.020
        elif ch in "/-":
            dt += 0.010
        if i == 0:
            dt += 0.015
        t += dt
    return out


def soft_tap(rng, size="M"):
    """Soft, deep 'appearance' tap for non-typed major text (rounded attack, no click)."""
    f0, tauk, n_s = dict(S=(146.0, 0.8, 0.30), M=(120.0, 1.0, 0.40), L=(96.0, 1.35, 0.62))[size]
    n = int(n_s * SR)
    t = tarr(n)
    modes = ring(t, f0 * np.array([1.0, 1.52, 2.25, 3.4]), np.array([0.075, 0.05, 0.03, 0.02]) * tauk,
                 [1.0, 0.5, 0.22, 0.08])
    modes = np.convolve(modes, soft_pulse(2.6))[:n]
    breath = band_noise(n, rng, 900, 2800) * ad_env(n, 0.016, 0.07) * 0.05
    sub = glide_sine(n, f0 * 0.8, f0 * 0.52, 0.03) * np.exp(-t / (0.09 * tauk)) * 0.30
    y = butter(modes + breath + sub, "lowpass", 3200, 2)
    return y / np.max(np.abs(y))


# <<END-OF-PART-3A>>
