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
CEILING_DBFS = -1.8            # limiter ceiling (sample peak) -> true peak stays under -1.5 dBTP

# Narration. `rate` is the service speaking-rate in percent (found with --fit);
# `gap` is the pause between sentences in seconds; `inner` scales comma/colon pauses.
# `starts` places each sentence separately (used when sentences must land on cues).
NARRATION = [
    dict(id="N1", start=0.80, max_end=6.80, rate=33, gap=0.20, inner=0.75,
         sentences=["Nineteen sixty-nine.", "A black arrow heads for China.", "No pilot.",
                    "Just a camera, and a route locked in its memory."]),
    dict(id="N2", start=7.35, max_end=14.20, rate=15, gap=0.26, inner=0.85,
         sentences=["The D-twenty-one.", "Built by Lockheed for the CIA.",
                    "A B-fifty-two carried it up.", "A rocket booster pushed it past Mach three."]),
    dict(id="N3", start=14.85, max_end=21.20, rate=27, gap=0.20, inner=0.55,
         sentences=["Its target: Lop Nor, China's nuclear test site.",
                    "The drone followed the route alone.", "No joystick.", "No second chances."]),
    dict(id="N4", start=21.85, max_end=27.20, rate=14, gap=0.28, inner=0.85,
         sentences=["After photographing the site, it turned for the Pacific.",
                    "The film sat inside a small recovery pod."]),
    dict(id="N5", start=27.85, max_end=35.70, rate=8, gap=0.30, inner=0.9,
         sentences=["Over the ocean, the pod ejected.",
                    "A parachute opened, and a C-one-thirty moved in to catch it.",
                    "The drone itself was expendable."]),
    dict(id="N6", start=36.45, max_end=37.80, rate=-10, gap=0.30, inner=1.0,
         sentences=["That was the plan."]),
    dict(id="N7", start=38.40, max_end=50.70, rate=5, gap=0.40, inner=0.9,
         starts=[38.40, 41.30, 44.45, 47.65],
         sentences=["The first drone missed its turn.", "The second returned, but its parachute failed.",
                    "On the third, the pod was lost at sea.", "The fourth crashed inside China."]),
    dict(id="N8", start=51.35, max_end=56.70, rate=10, gap=0.28, inner=0.85,
         sentences=["Four missions.", "No usable pictures.",
                    "In nineteen seventy-one, the program was shut down."]),
    dict(id="N9", start=57.30, max_end=64.70, rate=9, gap=0.30, inner=0.85,
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
    base = np.array([140.0, 212.0, 318.0, 480.0, 720.0, 1090.0, 1650.0]) * pitch
    amps = np.array([0.40, 1.0, 0.95, 0.80, 0.62, 0.36, 0.18])
    taus = np.array([0.040, 0.036, 0.029, 0.023, 0.017, 0.012, 0.008]) * stretch
    body = np.zeros(n)
    for f, a, tau in zip(base, amps, taus):
        inst = f * (1 + 0.045 * np.exp(-t / 0.018))              # soft pitch settle = "thock"
        body += a * np.exp(-t / tau) * np.sin(2 * np.pi * np.cumsum(inst) / SR + rng.uniform(0, 0.4))
    body = np.convolve(body, soft_pulse(1.1 + 0.4 * (1 - vel)))[:n]
    tap = band_noise(n, rng, 650, 3500) * exp_decay(n, 0.0018) * 0.24
    cream = band_noise(n, rng, 200, 900) * ad_env(n, 0.002, 0.028 * stretch) * 0.26
    sub = glide_sine(n, 92 * pitch, 58 * pitch, 0.022) * np.exp(-t / (0.045 * stretch)) * 0.19
    amps = amps * (1 + 0.10 * rng.standard_normal(len(amps)))
    down = body + tap + cream + sub
    if space:                                                     # stabiliser wire rattle
        for off, f in ((0.0006, 2100), (0.0031, 3050), (0.0074, 4300)):
            rat = ring(t[:int(0.02 * SR)], [f], [0.0035], [1.0]) * 0.05
            put(down, rat, off)
    up = ring(t, base[:5] * 1.22, taus[:5] * 0.45, amps[:5] * 0.18)
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
                 [1.0, 0.7, 0.42, 0.2])
    modes = np.convolve(modes, soft_pulse(2.6))[:n]
    breath = band_noise(n, rng, 900, 2800) * ad_env(n, 0.016, 0.07) * 0.07
    sub = np.tanh(1.6 * glide_sine(n, f0 * 0.8, f0 * 0.52, 0.03)) * np.exp(-t / (0.09 * tauk)) * 0.26
    y = butter(modes + breath + sub, "lowpass", 3200, 2)
    return y / np.max(np.abs(y))


# --------------------------------------------------------------------------- #
# Mechanical / diegetic cues. Each returns (stereo float array, anchor_seconds)
# where the anchor is the primary transient (or onset) of the cue.
# --------------------------------------------------------------------------- #
def _stereo(l, r=None):
    return np.stack([l, l if r is None else r], axis=1)


def _finish(y, anchor, peak=1.0, fo=0.03):
    y = fade_edges(y, 0.002, fo)
    return y / np.max(np.abs(y)) * peak, anchor


def relay_click(rng, size=1.0, hard=1.0):
    """Electromechanical relay: armature snap, contact bounce, housing knock."""
    n = int(0.22 * SR)
    t = tarr(n)
    y = ring(t, [2400, 3700, 5200], [0.0045, 0.0032, 0.0024], [1.0, 0.7, 0.4]) * 1.8 * hard
    y += band_noise(n, rng, 1800, 6500) * exp_decay(n, 0.0014) * 0.9 * hard
    for k, (dt, g) in enumerate(((0.0011, 0.45), (0.0023, 0.28), (0.0034, 0.15))):
        put(y, ring(t[:int(0.01 * SR)], [2900 + 300 * k], [0.002], [1.0]), dt, g * hard)
    y += ring(t, [210 * size ** -0.4, 480 * size ** -0.4, 960 * size ** -0.4], [0.016 * size, 0.010 * size, 0.007 * size],
              [0.5, 0.3, 0.15])
    return y


def sfx_separation(rng):
    """Pylon release: hook tension creak, steel clunk, ejector ram, wing flex, airflow."""
    dur, a = 1.9, 0.38
    n = int(dur * SR)
    m = mono_buf(dur)
    nc = int(0.36 * SR)
    tc = tarr(nc)
    stick = 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * np.cumsum(25 * (1 + 0.4 * slow_curve(nc, rng, 9))) / SR))
    put(m, band_noise(nc, rng, 230, 560) * (tc / 0.36) ** 2 * stick, 0.0, 0.16)
    for tt, g in ((0.19, 0.18), (0.30, 0.30)):
        put(m, ring(tarr(int(0.03 * SR)), [1350, 2900], [0.004, 0.0025], [1, 0.5]), tt, g)
    nb = int(0.9 * SR)
    tb = tarr(nb)
    clunk = ring(tb, [138, 331, 604, 1180, 2230], [0.075, 0.05, 0.032, 0.02, 0.01], [1, 0.7, 0.5, 0.25, 0.12])
    clunk = np.convolve(clunk, soft_pulse(0.5))[:nb]
    contact = band_noise(nb, rng, 800, 5200) * exp_decay(nb, 0.0022) * 0.5
    thud = glide_sine(nb, 78, 54, 0.05) * np.exp(-tb / 0.07) * 0.8
    flex = glide_sine(nb, 58, 50, 0.2) * np.exp(-tb / 0.28) * 0.30
    put(m, clunk + contact + thud + flex, a)
    put(m, band_noise(int(0.3 * SR), rng, 900, 5000) * ad_env(int(0.3 * SR), 0.006, 0.07) * 0.18, a + 0.015)
    nw = int(1.5 * SR)
    wenv = np.sin(np.pi * np.clip(tarr(nw) / 1.5, 0, 1)) ** 2
    wl, wr = (band_noise(nw, rng, 350, 1800, 2, 1.0) * wenv * 0.16 for _ in range(2))
    y = _stereo(m)
    i0 = int((a + 0.05) * SR)
    y[i0:i0 + nw, 0] += wl[:n - i0]
    y[i0:i0 + nw, 1] += wr[:n - i0]
    return _finish(y, a)


def sfx_booster(rng, dur=5.6):
    """Solid-fuel booster: ignition thump, rising roar with crackle and chuffing, burnout."""
    n = int(dur * SR)
    t = tarr(n)
    fc = 260 + 1900 * (1 - np.exp(-t / 0.45)) * np.exp(-np.maximum(t - 3.9, 0) / 1.4)
    parts = []
    for k in range(3):
        w = tv_filter(colored_noise(n, rng, 1.0), fc, "lp", 0.7)
        w = butter(w, "highpass", 38, 2)
        parts.append(w / w.std())
    common, nl, nr = parts
    shape = np.vstack([biquad("ls", 120, 0.7, 5.0), biquad("peak", 70, 1.0, 3.0), biquad("peak", 800, 0.8, -4.0)])
    chuff = 1 + 0.16 * np.sin(2 * np.pi * np.cumsum(21 * (1 + 0.08 * slow_curve(n, rng, 1.5))) / SR)
    k = rng.poisson(380 * dur)
    imp = np.zeros(n)
    np.add.at(imp, rng.integers(0, n, k), rng.exponential(1.0, k))
    crackle = band_noise(n, rng, 1400, 3600) * 0 + bandpass(imp, 1400, 3600, 2)
    crackle /= crackle.std()
    hiss = band_noise(n, rng, 5000, 9000)
    lvl = gain(interp_curve([(0, -60), (0.03, -40), (0.18, -18), (0.55, -6), (1.0, 0), (3.9, -0.5), (4.7, -10), (5.6, -30)],
                            n, smooth=0.03))
    chans = []
    for ind in (nl, nr):
        w = 0.8 * common + 0.6 * ind
        w = apply_sos(w, shape) * chuff + 0.07 * crackle + 0.012 * hiss
        chans.append(w * lvl)
    y = _stereo(*chans)
    nth = int(0.9 * SR)
    th = glide_sine(nth, 66, 36, 0.07) * ad_env(nth, 0.006, 0.20)
    th = (np.tanh(2.2 * th) / np.tanh(2.2)) * 1.0 + band_noise(nth, rng, 40, 600) * exp_decay(nth, 0.15) * 0.4
    y[:nth] += th[:, None] * 2.2 * np.sqrt(np.mean(y[int(1.0 * SR):int(3.5 * SR)] ** 2))
    return _finish(y, 0.0, fo=0.4)


def sfx_shutter(rng):
    """Heavy focal-plane shutter: first curtain, body knock, second curtain, film advance."""
    dur, a = 0.5, 0.03
    m = mono_buf(dur)
    t = tarr(int(0.2 * SR))
    put(m, ring(t, [1650, 3380, 5200], [0.004, 0.0028, 0.002], [1, 0.5, 0.25]), a, 0.7)
    put(m, band_noise(int(0.012 * SR), rng, 1800, 7000) * np.hanning(int(0.012 * SR)), a, 0.28)
    put(m, np.convolve(ring(t, [196, 410, 790], [0.03, 0.02, 0.012], [0.9, 0.5, 0.25]), soft_pulse(0.6))[:len(t)], a + 0.004, 1.0)
    put(m, ring(t, [2300, 4100], [0.003, 0.002], [0.6, 0.3]), a + 0.016, 0.55)
    put(m, ring(t, [210, 430], [0.02, 0.014], [0.8, 0.4]), a + 0.016, 0.45)
    nm = int(0.15 * SR)
    tm = tarr(nm)
    f = 140 + 50 * tm / 0.15
    ph = 2 * np.pi * np.cumsum(f) / SR
    motor = sum(np.sin(h * ph + rng.uniform(0, 6.28)) / h for h in range(1, 13))
    gear = band_noise(nm, rng, 900, 2600) * (0.6 + 0.4 * np.sin(2 * np.pi * 52 * tm))
    menv = np.minimum(tm / 0.015, 1) * np.minimum((0.15 - tm) / 0.04, 1)
    put(m, (motor * 0.25 + gear * 0.5) * menv, a + 0.085, 0.16)
    put(m, ring(t, [1100, 2400], [0.005, 0.003], [1, 0.5]), a + 0.245, 0.35)
    return _finish(_stereo(m), a, fo=0.05)


def sfx_optical(rng):
    """Optical window shutter closing: sliding blades, servo whine, latch."""
    dur, a = 0.6, 0.21
    m = mono_buf(dur)
    ns = int(0.2 * SR)
    ts = tarr(ns)
    slide = band_noise(ns, rng, 1200, 4500) * np.sin(np.pi * ts / 0.2) ** 2 * 0.12
    f = 700 + 350 * ts / 0.2
    ph = 2 * np.pi * np.cumsum(f) / SR
    whine = (np.sin(ph) + 0.4 * np.sin(2 * ph)) * np.sin(np.pi * ts / 0.2) ** 2 * 0.10
    put(m, slide + whine, 0.0)
    t = tarr(int(0.15 * SR))
    put(m, ring(t, [2150, 3700], [0.0035, 0.002], [0.8, 0.4]), a)
    put(m, ring(t, [310, 640], [0.014, 0.010], [0.6, 0.3]), a)
    put(m, ring(t, [2600], [0.002], [1.0]), a + 0.024, 0.22)
    return _finish(_stereo(m), a, fo=0.05)


def sfx_servo(rng, dur=0.8):
    """Small actuator move: relay tick, geared motor whine up and down, settle tick."""
    n = int(dur * SR)
    t = tarr(n)
    f = np.interp(t, [0, 0.12, 0.45, 0.62], [300, 520, 505, 300])
    f = f * (1 + 0.012 * slow_curve(n, rng, 6))
    ph = 2 * np.pi * np.cumsum(f) / SR
    tone = sum(g * np.sin(k * ph + rng.uniform(0, 6.28)) for k, g in ((1, 1.0), (2, 0.45), (3, 0.3), (4, 0.15)))
    tone *= 1 + 0.25 * np.sin(2 * np.pi * 38 * t)
    env = np.interp(t, [0, 0.04, 0.5, 0.64, 0.8], [0, 1, 0.9, 0.0, 0])
    brush = band_noise(n, rng, 500, 1800) * env * 0.10
    m = (tone * 0.22 + brush) * env
    put(m, relay_click(rng, 0.6, 0.35)[:int(0.05 * SR)], 0.0, 0.5)
    put(m, ring(tarr(int(0.05 * SR)), [1900, 3100], [0.003, 0.002], [1, 0.5]), 0.66, 0.35)
    return _finish(_stereo(m), 0.0, fo=0.08)


def sfx_pod(rng):
    """Recovery pod ejection: squib/pressure pop, hatch ring, gas hiss, tumbling away."""
    dur, a = 2.0, 0.10
    n = int(dur * SR)
    m = mono_buf(dur)
    put(m, ring(tarr(int(0.03 * SR)), [1900, 3300], [0.003, 0.002], [1, 0.5]), 0.02, 0.2)
    nb = int(0.5 * SR)
    tb = tarr(nb)
    pop = ring(tb, [92, 186, 372], [0.07, 0.05, 0.03], [1, 0.8, 0.45])
    pop = np.convolve(pop, soft_pulse(0.7))[:nb] + band_noise(nb, rng, 1500, 6000) * exp_decay(nb, 0.0012) * 0.5
    pop += ring(tb, [830, 1475, 2340, 3410], [0.09, 0.06, 0.04, 0.025], [0.18, 0.14, 0.10, 0.06])
    put(m, pop, a)
    nh = int(0.4 * SR)
    put(m, band_noise(nh, rng, 800, 4800) * ad_env(nh, 0.005, 0.09) * 0.15, a + 0.01)
    nt = int(1.7 * SR)
    tt = tarr(nt)
    frot = 1.2 + 2.0 * np.exp(-tt / 0.7)
    am = 0.55 + 0.45 * np.cos(2 * np.pi * np.cumsum(frot) / SR)
    tumble = band_noise(nt, rng, 160, 900, 2, 1.0) * am * np.exp(-tt / 0.65) * np.minimum(tt / 0.12, 1) * 0.12
    y = _stereo(m)
    pan = np.clip(0.45 * tt / 1.2, 0, 0.45)
    i0 = int((a + 0.15) * SR)
    y[i0:i0 + nt, 0] += (tumble * np.cos((pan + 1) * np.pi / 4))[:n - i0]
    y[i0:i0 + nt, 1] += (tumble * np.sin((pan + 1) * np.pi / 4))[:n - i0]
    return _finish(y, a, fo=0.2)


def flap_train(rng, times, amps, lo=500, hi=1800, tau=0.03, body=0.5):
    n = int((max(times) + 0.2) * SR)
    out = np.zeros(n)
    nf = int(0.2 * SR)
    for tt, g in zip(times, amps):
        burst = band_noise(nf, rng, lo, hi) * ad_env(nf, 0.003, tau)
        thump = glide_sine(nf, 140, 95, 0.02) * np.exp(-tarr(nf) / 0.03) * body
        put(out, burst + thump, tt, g)
    return out


def snap_cloud(rng, dur, rate_fn, lo=1200, hi=4800):
    """Poisson micro-snaps of taut fabric; rate_fn(t) in events/s."""
    ts = np.arange(0, dur, 0.001)
    rates = rate_fn(ts)
    cnt = rng.poisson(rates * 0.001)
    times = np.repeat(ts, cnt) + rng.uniform(0, 0.001, cnt.sum())
    out = mono_buf(dur + 0.05)
    ns = int(0.012 * SR)
    for tt in times:
        g = rng.exponential(0.5)
        put(out, band_noise(ns, rng, lo, hi) * exp_decay(ns, rng.uniform(0.0015, 0.004)), tt, g)
    return out


def sfx_chute(rng):
    """Parachute deployment: pilot-chute snap, line pay-out, canopy whump, settling flutter."""
    dur = 3.4
    n = int(dur * SR)
    m = mono_buf(dur)
    ns = int(0.2 * SR)
    put(m, band_noise(ns, rng, 600, 3500) * ad_env(ns, 0.002, 0.03) * 0.5
        + glide_sine(ns, 240, 190, 0.02) * np.exp(-tarr(ns) / 0.03) * 0.4, 0.0)
    nl = int(0.3 * SR)
    tl = tarr(nl)
    put(m, band_noise(nl, rng, 900, 2600) * (tl / 0.3) ** 2 * np.minimum((0.3 - tl) / 0.03, 1) * 0.12, 0.06)
    nw = int(0.8 * SR)
    tw = tarr(nw)
    wenv = ad_env(nw, 0.04, 0.18)
    whump = glide_sine(nw, 58, 74, 0.1) * wenv * 0.8 + band_noise(nw, rng, 60, 450, 2, 1.0) * wenv * 0.7
    put(m, whump, 0.26)
    put(m, band_noise(int(0.15 * SR), rng, 900, 4200) * ad_env(int(0.15 * SR), 0.003, 0.02) * 0.55, 0.28)
    snaps = snap_cloud(rng, dur - 0.3, lambda t: 130 * np.exp(-t / 0.9) + 25)
    put(m, snaps, 0.3, 0.55)
    nbil = int((dur - 0.3) * SR)
    tbil = tarr(nbil)
    bill = band_noise(nbil, rng, 220, 1100, 2, 1.0) * np.clip(0.55 + 0.25 * slow_curve(nbil, rng, 7), 0.2, 1.4) * np.exp(-tbil / 1.1)
    put(m, bill, 0.3, 0.18)
    y = _stereo(m)
    y[:, 1] = m * 0.94 + np.roll(m, 31) * 0.1
    return _finish(y, 0.0, fo=0.6)


def sfx_c130(rng, dur=3.55, t_catch=1.62):
    """C-130 approach: four 4-blade props (68 Hz blade-pass) beating slightly apart,
    with a gentle Doppler glide, opening air roar and turbine whine."""
    n = int(dur * SR)
    t = tarr(n)
    dop = 1 + 0.035 * np.tanh(-(t - t_catch) / 0.5)
    lvl = gain(interp_curve([(0, -46), (0.8, -26), (1.3, -12), (t_catch, -3), (2.4, -4.5), (dur, -8)], n, smooth=0.05))
    fc = interp_curve([(0, 700), (1.0, 1200), (t_catch, 3200), (dur, 2400)], n, smooth=0.05)
    chans = [np.zeros(n), np.zeros(n)]
    for delta, pan in zip((-0.0042, -0.0014, 0.0017, 0.0045), (-0.38, -0.13, 0.13, 0.38)):
        drift = 1 + delta + 0.0012 * slow_curve(n, rng, 0.35)
        phi = 2 * np.pi * np.cumsum(68.0 * drift * dop) / SR + rng.uniform(0, 6.28)
        w = np.zeros(n)
        for h in range(1, 19):
            w += (h ** -0.85) * (1 + 0.25 * rng.standard_normal()) * np.cos(h * phi + rng.uniform(0, 6.28))
        chop = band_noise(n, rng, 120, 1500, 2, 1.0) * (0.7 + 0.3 * np.cos(4 * phi))
        w = w + 0.55 * chop * np.std(w)
        th, ph_ = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        chans[0] += w * th
        chans[1] += w * ph_
    whine = band_noise(n, rng, 2050, 2250) * 0.03 + band_noise(n, rng, 3400, 3560) * 0.008
    prox = gain(interp_curve([(0, -30), (t_catch, 0), (dur, -6)], n, smooth=0.1))
    out = []
    for c in chans:
        c = tv_filter(c / np.std(c), fc, "lp", 0.7) + whine * prox
        out.append(c * lvl)
    y = _stereo(*out)
    return _finish(y, 0.0, fo=0.02)


def sfx_catch(rng):
    """Mid-air retrieval: cable zip, hook contact, tension thump, line twang/creak, winch."""
    dur, a = 2.2, 0.12
    m = mono_buf(dur)
    nz = int(0.12 * SR)
    tz = tarr(nz)
    put(m, band_noise(nz, rng, 1500, 6000) * (tz / 0.12) ** 2 * 0.3, 0.0)
    t = tarr(int(0.3 * SR))
    put(m, ring(t, [1450, 2900, 4600], [0.004, 0.0028, 0.002], [1, 0.6, 0.3]), a, 0.8)
    put(m, band_noise(int(0.01 * SR), rng, 1500, 6000) * np.hanning(int(0.01 * SR)), a, 0.4)
    nth = int(0.6 * SR)
    put(m, glide_sine(nth, 96, 70, 0.09) * np.exp(-tarr(nth) / 0.09) * 0.9
        + ring(tarr(nth), [182, 340], [0.07, 0.04], [0.6, 0.3]), a)
    ntw = int(1.0 * SR)
    ttw = tarr(ntw)
    twang = ring(ttw, [232, 464], [0.12, 0.07], [0.25, 0.1]) * (1 + 0.06 * np.sin(2 * np.pi * 6 * ttw))
    put(m, twang, a + 0.01)
    ncr = int(0.6 * SR)
    tcr = tarr(ncr)
    stick = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * np.cumsum(7 * (1 + 0.5 * slow_curve(ncr, rng, 5))) / SR))
    put(m, band_noise(ncr, rng, 420, 1100) * stick * np.exp(-tcr / 0.35) * 0.10, a + 0.05)
    nwn = int(1.5 * SR)
    tw = tarr(nwn)
    ph = 2 * np.pi * np.cumsum(120 * (1 + 0.02 * slow_curve(nwn, rng, 2))) / SR
    winch = sum(np.sin(h * ph + rng.uniform(0, 6.28)) / h for h in range(1, 13)) * (1 + 0.3 * np.sin(2 * np.pi * 28 * tw))
    wenv = np.minimum(tw / 0.15, 1) * np.minimum((1.5 - tw) / 0.1, 1)
    tick = np.zeros(nwn)
    tt = 0.0
    while tt < 1.5:
        put(tick, ring(tarr(int(0.012 * SR)), [1800], [0.002], [1]), tt, 0.5 * rng.uniform(0.6, 1.0))
        tt += rng.uniform(0.85, 1.15) / 28
    put(m, (winch * 0.2 + tick * 0.18) * wenv, a + 0.28, 0.2)
    return _finish(_stereo(m), a, fo=0.1)


def sfx_relay_hard(rng):
    """The 'plan' cut: a hard contactor throw with heavy housing knock."""
    y = relay_click(rng, size=1.6, hard=1.0)
    y = y + glide_sine(len(y), 120, 80, 0.02) * np.exp(-tarr(len(y)) / 0.03) * 0.35
    return _finish(_stereo(y), 0.0, fo=0.05)


def sfx_fail_guidance(rng):
    """Guidance fault: servo hunting, fault relay + end-stop knock, gyro rotor spinning down."""
    dur, a = 1.8, 0.47
    n = int(dur * SR)
    m = mono_buf(dur)
    for t0, f0, f1, g in ((0.0, 520, 760, 0.30), (0.14, 520, 760, 0.28), (0.30, 700, 430, 0.30)):
        nn = int(0.075 * SR)
        tn = tarr(nn)
        f = f0 + (f1 - f0) * tn / 0.075
        ph = 2 * np.pi * np.cumsum(f) / SR
        env = np.sin(np.pi * tn / 0.075) ** 1.2
        put(m, ((np.sin(ph) + 0.4 * np.sin(2 * ph)) * 0.6 + band_noise(nn, rng, 700, 2400) * 0.3) * env, t0, g)
    put(m, relay_click(rng, 0.9, 0.5), a)
    nth = int(0.4 * SR)
    put(m, glide_sine(nth, 80, 56, 0.04) * np.exp(-tarr(nth) / 0.05) * 0.5, a)
    ng = int((dur - a - 0.02) * SR)
    tg = tarr(ng)
    f = 230 + 410 * np.exp(-tg / 0.62)
    ph = 2 * np.pi * np.cumsum(f) / SR
    wob = 1 + (0.15 + 0.25 * tg / 1.3) * np.sin(2 * np.pi * np.cumsum(3.4 + 1.6 * tg / 1.3) / SR)
    rotor = (np.sin(ph) + 0.35 * np.sin(2 * ph + 0.7) + 0.15 * np.sin(3 * ph + 1.1)) * wob * np.exp(-tg / 0.75)
    grit = band_noise(ng, rng, 1200, 3200) * (0.5 + 0.5 * np.abs(slow_curve(ng, rng, 40))) * np.exp(-tg / 0.55) * 0.10
    put(m, (rotor * 0.30 + grit) * np.minimum(tg / 0.02, 1), a + 0.02)
    return _finish(_stereo(m), a, fo=0.2)


def sfx_fail_chute(rng):
    """Parachute failure: weak pilot pop, canopy rips and a line snaps, ragged streaming
    flutter, whistle of a fast fall."""
    dur = 1.7
    n = int(dur * SR)
    m = mono_buf(dur)
    put(m, band_noise(int(0.05 * SR), rng, 700, 3000) * exp_decay(int(0.05 * SR), 0.015) * 0.3, 0.0)
    nw = int(0.25 * SR)
    put(m, glide_sine(nw, 62, 74, 0.05) * ad_env(nw, 0.03, 0.07) * 0.35
        + band_noise(nw, rng, 60, 500, 2, 1.0) * ad_env(nw, 0.03, 0.07) * 0.3, 0.05)
    nr = int(0.24 * SR)
    tr = tarr(nr)
    rip = snap_cloud(rng, 0.24, lambda t: 400 + 1400 * t / 0.24, 1200, 6500)
    put(m, rip[:nr] * (0.4 + 0.6 * tr / 0.24) * 0.45, 0.22)
    put(m, ring(tarr(int(0.4 * SR)), [310, 690], [0.05, 0.03], [0.5, 0.25]) + 0.5 * np.pad(
        ring(tarr(int(0.02 * SR)), [2600], [0.002], [1]), (0, int(0.38 * SR))), 0.46)
    times, tt = [], 0.5
    while tt < 1.5:
        times.append(tt)
        tt += rng.uniform(0.65, 1.35) / (12 + 5 * (tt - 0.5))
    amps = [0.4 * np.exp(-(x - 0.5) / 0.5) * rng.uniform(0.4, 1.0) for x in times]
    put(m, flap_train(rng, times, amps, 450, 2200, 0.022, body=0.12), 0.0)
    nwh = int(1.1 * SR)
    twh = tarr(nwh)
    fcen = 950 + 950 * twh / 1.1
    whistle = tv_filter(rng.standard_normal(nwh), fcen, "bp", 9.0)
    whistle = whistle / (whistle.std() + 1e-12) * 0.06 * np.sin(np.pi * np.clip(twh / 1.1, 0, 1)) ** 1.5
    put(m, whistle, 0.45)
    y = _stereo(m)
    y[:, 1] = np.roll(m, 41) * 0.92
    return _finish(y, 0.0, fo=0.25)


def bubble(rng, f0):
    """Minnaert bubble: rising-frequency damped sine (Van den Doel's model)."""
    d = 0.043 * f0 + 0.0014 * f0 ** 1.5
    n = int(min(6.0 / d, 0.25) * SR) + 8
    t = tarr(n)
    f = f0 * (1 + 0.1 * d * t)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-d * t)


def sfx_fail_water(rng):
    """Pod hits the sea: slap, cavity bloop, spray, bubbles, then it goes under (low-pass)."""
    dur, a = 2.7, 0.14
    n = int(dur * SR)
    t = tarr(n)
    rush = band_noise(int(a * SR), rng, 500, 2000, 2, 1.0) * (tarr(int(a * SR)) / a) ** 2 * 0.22
    nb = int(1.0 * SR)
    tb = tarr(nb)
    slap = band_noise(nb, rng, 400, 7000, 2, 1.0) * ad_env(nb, 0.001, 0.03) * 0.9
    bloop = glide_sine(nb, 190, 68, 0.09) * np.exp(-tb / 0.14) * 0.9 + band_noise(nb, rng, 40, 800, 2, 1.0) * exp_decay(nb, 0.1) * 0.5
    ns = int(1.2 * SR)
    spray = snap_cloud(rng, 1.2, lambda x: 900 * np.exp(-x / 0.3) + 40, 1500, 6500)[:ns] * 0.25
    nbub = 36
    bub = np.zeros(n)
    tcur = 0.04
    for _ in range(nbub):
        tcur += rng.exponential(0.035 + 0.02 * tcur)
        f0 = float(np.exp(rng.uniform(np.log(650), np.log(4000))))
        put(bub, bubble(rng, f0), a + tcur, 0.22 * (900 / f0) ** 0.5 * rng.uniform(0.5, 1.0))
    wet = np.zeros(n)
    put(wet, spray, a)
    wet += bub
    cut = np.interp(t, [0, a, a + 0.15, a + 0.8, dur], [7000, 7000, 5000, 420, 300])
    wet = tv_filter(wet, cut, "lp", 0.7)
    low = np.zeros(n)
    put(low, slap * 0.8 + bloop, a)
    low = tv_filter(low, np.interp(t, [0, a, a + 0.25, a + 0.9], [8000, 8000, 1500, 600]), "lp", 0.7)
    pres = band_noise(n, rng, 35, 170, 2, 2.0) * np.minimum(np.maximum(t - a - 0.1, 0) / 0.25, 1) * np.exp(-np.maximum(t - a - 0.35, 0) / 0.55) * 0.3
    m = wet + low + pres
    put(m, rush, 0.0)
    y = _stereo(m)
    y[:, 1] = np.roll(wet, 37) + low + pres + np.pad(rush, (0, n - len(rush)))
    return _finish(y, a, fo=0.4)


def sfx_fail_impact(rng):
    """Crash on land, deliberately low and restrained: deep thud, dirt, scattered debris."""
    dur = 1.8
    n = int(dur * SR)
    t = tarr(n)
    thud = glide_sine(n, 82, 36, 0.10) * ad_env(n, 0.003, 0.13) * 1.0
    thud += ring(t, [118], [0.07], [0.5])
    dirt = band_noise(n, rng, 40, 500, 2, 1.0) * exp_decay(n, 0.10) * 0.55
    dirt += band_noise(n, rng, 120, 700, 2, 1.0) * exp_decay(n, 0.20) * 0.4
    deb = np.zeros(n)
    k = 46
    for tt in 0.02 + 0.9 * rng.beta(1, 2.2, k):
        f = float(rng.uniform(260, 2400))
        nn = int(0.06 * SR)
        put(deb, ring(tarr(nn), [f], [rng.uniform(0.008, 0.04)], [1.0]), tt, gain(rng.uniform(-30, -16)))
    rattle = np.zeros(n)
    for tt, g in ((0.04, 1.0), (0.21, 0.6)):
        put(rattle, ring(tarr(int(0.5 * SR)), [410, 590, 930], [0.16, 0.12, 0.09], [0.08, 0.06, 0.04]), tt, g)
    y = butter(thud + dirt + deb + rattle, "lowpass", 2400, 2)
    return _finish(_stereo(y), 0.0, fo=0.35)


def sfx_film(rng):
    """Blank film transport: drive-motor run, sprocket ticks, tail flapping, spool stop."""
    dur = 2.75
    n = int(dur * SR)
    t = tarr(n)
    f = np.interp(t, [0, 0.4, 1.55, 2.3, 2.5], [62, 118, 118, 55, 40])
    f = f * (1 + 0.012 * slow_curve(n, rng, 9))
    ph = 2 * np.pi * np.cumsum(f) / SR
    motor = sum(np.sin(h * ph + rng.uniform(0, 6.28)) / h ** 1.2 for h in range(1, 11))
    motor *= 0.75 + 0.25 * slow_curve(n, rng, 11)
    env = np.interp(t, [0, 0.1, 0.4, 1.55, 2.3, 2.6], [0, 0.5, 1, 1, 0.4, 0])
    rustle = band_noise(n, rng, 700, 3500) * (0.6 + 0.4 * np.abs(slow_curve(n, rng, 25)))
    m = motor * env * 0.10 + band_noise(n, rng, 300, 1400) * env * 0.05 + rustle * env * 0.06
    tt = 0.0
    while tt < 2.3:
        rate = 0.2 * np.interp(tt, [0, 0.4, 1.55, 2.3], [62, 118, 118, 55])
        put(m, ring(tarr(int(0.012 * SR)), [1600, 3100], [0.003, 0.002], [1.0, 0.5]), tt, 0.30 * rng.uniform(0.6, 1.0) * np.interp(tt, [0, .4, 1.55, 2.3], [.4, 1, 1, .5]))
        tt += 1 / rate
    tt, i = 1.62, 0
    while tt < 2.55:
        nf = int(0.08 * SR)
        flap = band_noise(nf, rng, 700, 3200) * ad_env(nf, 0.002, 0.014) + ring(tarr(nf), [190], [0.02], [0.5])
        put(m, flap, tt, 0.35 * np.exp(-i / 9))
        tt += 0.075 * 1.12 ** i
        i += 1
    put(m, ring(tarr(int(0.25 * SR)), [260, 560], [0.04, 0.025], [0.6, 0.3]), 2.55, 0.6)
    return _finish(_stereo(m), 0.0, fo=0.1)


def sfx_archive(rng):
    """Steel archive drawer pushed shut: rolling slide, heavy stop, ring, latch, rebound."""
    dur, a = 2.6, 0.42
    m = mono_buf(dur)
    nr = int(a * SR)
    tr = tarr(nr)
    rough = 0.6 + 0.4 * np.sin(2 * np.pi * np.cumsum(110 * (1 + 0.3 * slow_curve(nr, rng, 8))) / SR)
    put(m, band_noise(nr, rng, 220, 1800) * (tr / a) ** 1.5 * rough * 0.11, 0.0)
    tt = 0.02
    while tt < a - 0.01:
        put(m, ring(tarr(int(0.01 * SR)), [1500], [0.002], [1]), tt, 0.05)
        tt += 0.04 + 0.05 * tt / a
    nb = int(2.0 * SR)
    tb = tarr(nb)
    body = ring(tb, [92, 138, 205, 312], [0.16, 0.12, 0.085, 0.055], [1, 0.75, 0.5, 0.3])
    body = np.convolve(body, soft_pulse(0.9))[:nb]
    ringing = ring(tb, [545, 1180, 1790, 2920], [0.12, 0.09, 0.07, 0.05], [0.14, 0.09, 0.06, 0.035])
    burst = band_noise(nb, rng, 200, 1500) * exp_decay(nb, 0.01) * 0.5
    stop = glide_sine(nb, 70, 56, 0.04) * np.exp(-tb / 0.07) * 0.6
    put(m, body + ringing + burst + stop, a)
    put(m, ring(tarr(int(0.1 * SR)), [2200, 3500], [0.004, 0.0025], [0.45, 0.22])
        + ring(tarr(int(0.1 * SR)), [420], [0.015], [0.3]), a + 0.085)
    put(m, ring(tb[:int(0.5 * SR)], [92, 138, 205], [0.1, 0.07, 0.05], [0.25, 0.18, 0.1]), a + 0.125)
    return _finish(_stereo(m), a, fo=0.3)


def sfx_crt(rng):
    """CRT degauss: low thump, decaying mains-frequency hum with beating, faint mask ring."""
    dur = 2.4
    n = int(dur * SR)
    t = tarr(n)
    thump = glide_sine(n, 70, 52, 0.05) * ad_env(n, 0.003, 0.09) * 0.7
    hum = sum(g * np.sin(2 * np.pi * f * t) for f, g in ((60, 1.0), (120, 0.7), (180, 0.3)))
    hum = hum * np.exp(-t / 0.45) * (0.7 + 0.3 * np.cos(2 * np.pi * 4.1 * t)) * 0.5
    mask = ring(t, [560, 910], [0.35, 0.25], [0.05, 0.035])
    st = mono_buf(0.3)
    for tt in sorted(rng.uniform(0, 0.2, 6)):
        put(st, band_noise(int(0.004 * SR), rng, 2000, 7000) * exp_decay(int(0.004 * SR), 0.001), tt, 0.1)
    y = thump + hum + mask
    y[:len(st)] += st
    return _finish(_stereo(y), 0.0, fo=0.4)


def sfx_contactor_open(rng):
    """Soft breaker/contactor drop used for the 'program was shut down' beat."""
    n = int(1.0 * SR)
    t = tarr(n)
    y = relay_click(rng, 1.4, 0.35)
    y = np.pad(y, (0, n - len(y)))
    y += glide_sine(n, 100, 46, 0.25) * ad_env(n, 0.012, 0.28) * 0.35
    y += ring(t, [120], [0.5], [0.04]) * np.sin(2 * np.pi * 4 * t)
    return _finish(_stereo(y), 0.0, fo=0.25)


# --------------------------------------------------------------------------- #
# Score: where everything lands (times in seconds on the 68 s timeline)
# --------------------------------------------------------------------------- #
KEY_LEVEL_DB = -11.5           # peak of a nominal key press
TAP_LEVEL_DB = -16.0           # soft appearance taps
APPEAR = [                     # (id, label it follows, size) -> soft tap 0.16 s after the last key
    ("A01", "T01", "M"), ("A02", "T08", "M"), ("A03", "T09", "M"), ("A04", "T10", "L"),
]

CUES = [
    dict(id="separation", fn=sfx_separation, t=8.80, gain=-10.0, sends=dict(room=-13),
         desc="pylon release: hook creak, steel clunk, ejector ram, wing flex, drone falling away"),
    dict(id="booster", fn=sfx_booster, t=9.60, gain=-12.0, sends=dict(room=-24),
         desc="solid booster: ignition thump, crackling roar (deep, 40-300 Hz weighted), burnout into cruise"),
    dict(id="camera_shutter", fn=sfx_shutter, t=17.50, gain=-12.0, sends=dict(room=-18),
         desc="heavy focal-plane shutter, body knock, film advance"),
    dict(id="optical_shutter", fn=sfx_optical, t=22.00, gain=-18.0, sends=dict(room=-20),
         desc="optical window shutter closing: blades, servo whine, latch"),
    dict(id="servo_turn", fn=sfx_servo, t=23.10, gain=-23.0, sends=dict(room=-20), extra=True,
         desc="small control actuator for 'turned for the Pacific'"),
    dict(id="pod_eject", fn=sfx_pod, t=28.80, gain=-12.0, sends=dict(room=-16),
         desc="recovery pod: squib/pressure pop, hatch ring, gas hiss, tumbling away"),
    dict(id="parachute", fn=sfx_chute, t=30.20, gain=-11.0, sends=dict(room=-20),
         desc="parachute: pilot snap, line pay-out, canopy whump, fabric flutter"),
    dict(id="c130_approach", fn=sfx_c130, t=32.50, gain=-17.0, sends={},
         desc="C-130: four props beating, rising roar, turbine whine, gentle Doppler"),
    dict(id="c130_catch", fn=sfx_catch, t=34.12, gain=-13.0, sends=dict(room=-22),
         desc="mid-air retrieval: cable zip, hook contact, tension thump, line twang, winch"),
    dict(id="relay_hard", fn=sfx_relay_hard, t=36.00, gain=-7.0, sends={}, bus="sting",
         desc="hard contactor throw; everything else (air, effects, tails) is gated to silence 36.0-38.0"),
    dict(id="fail1_guidance", fn=sfx_fail_guidance, t=40.20, gain=-12.0, sends=dict(room=-20),
         desc="guidance fault: servo hunting, fault relay + end-stop, gyro rotor spinning down"),
    dict(id="fail2_parachute", fn=sfx_fail_chute, t=43.40, gain=-13.5, sends=dict(room=-22),
         desc="parachute failure: weak pop, canopy rip, line snap, ragged flutter, whistle of a fast fall"),
    dict(id="fail3_water", fn=sfx_fail_water, t=46.50, gain=-11.0, sends=dict(room=-20),
         desc="sea impact: slap, cavity bloop, spray, bubbles, then muffled as the pod goes under"),
    dict(id="fail4_impact", fn=sfx_fail_impact, t=49.60, gain=-10.0, sends=dict(room=-22),
         desc="low restrained crash: deep thud, dirt, scattered debris (no explosion)"),
    dict(id="film_transport", fn=sfx_film, t=52.20, gain=-21.0, sends=dict(room=-24),
         desc="blank film transport: motor, sprocket ticks, tail flapping, spool stop"),
    dict(id="contactor_open", fn=sfx_contactor_open, t=56.15, gain=-22.0, sends=dict(room=-20), extra=True,
         desc="soft breaker drop under 'the program was shut down'"),
    dict(id="crt_degauss", fn=sfx_crt, t=57.00, gain=-23.0, sends=dict(hall=-16), extra=True,
         desc="CRT degauss thump + decaying mains hum as the archive terminal wakes"),
    dict(id="archive_close", fn=sfx_archive, t=64.80, gain=-11.0, sends=dict(hall=-9),
         desc="steel archive drawer rolls shut: heavy stop, ring, latch, rebound"),
]

# Bed layers: absolute RMS level in dBFS over time. Air is the constant diegetic floor.
BED = dict(
    air=[(0, -90), (0.3, -90), (1.8, -43), (6.0, -43), (8.6, -41), (8.8, -37), (9.8, -41), (14, -41), (22, -43),
         (28.5, -43), (29.5, -46), (33, -50), (36, -50), (38, -50), (39.5, -48), (50.5, -48), (51.0, -56),
         (51.5, -48), (56.5, -48), (57, -50), (64.5, -50), (66.5, -62), (68, -90)],
    low=[(0, -90), (1, -90), (2, -50), (8.7, -50), (9.6, -45), (14, -45), (15, -48), (28.8, -48), (30, -54),
         (38, -54), (39, -52), (51.5, -52), (56.5, -52), (57, -48), (64.5, -48), (66, -58), (68, -90)],
    carrier=[(0, -90), (5.4, -90), (7.0, -52), (8.75, -38), (8.9, -70), (9.2, -90)],
    cruise=[(0, -90), (12.8, -90), (14.8, -52), (15.5, -46), (28.4, -46), (29.4, -60), (30, -90)],
)
GATE = (36.0, 38.0, 1.5)       # silence start, end, fade-in length
DUCK = dict(bed=5.0, sfx=2.5, keys=2.0)   # dB of sidechain reduction while the voice speaks


def unit_rms(x):
    return x / (np.sqrt(np.mean(x * x)) + 1e-12)


def build_bed():
    rng = rng_for("bed")
    n = NSAMP
    t = tarr(n)

    def curve(points):
        return gain(interp_curve(points, n, smooth=0.25))

    air = []
    for _ in range(2):
        w = bandpass(colored_noise(n, rng, 1.0), 110, 1800, 2)
        air.append(unit_rms(w * (1 + 0.2 * slow_curve(n, rng, 0.12))))
    low_c = unit_rms(butter(colored_noise(n, rng, 2.0), "lowpass", 130, 2))
    low = [unit_rms(0.85 * low_c + 0.5 * unit_rms(butter(colored_noise(n, rng, 2.0), "lowpass", 130, 2))) for _ in range(2)]
    carr = unit_rms((butter(colored_noise(n, rng, 2.0), "lowpass", 160, 2)
                     + 0.4 * bandpass(colored_noise(n, rng, 1.0), 180, 900, 2) * 0.5)
                    * (0.85 + 0.15 * np.cos(2 * np.pi * 3.7 * t)) * (0.9 + 0.1 * np.cos(2 * np.pi * 5.3 * t)))
    cru = unit_rms(bandpass(colored_noise(n, rng, 1.0), 120, 1400, 2) + 0.8 * butter(colored_noise(n, rng, 2.0), "lowpass", 200, 2))
    ca, cl, cc, cr = (curve(BED[k]) for k in ("air", "low", "carrier", "cruise"))
    out = np.zeros((n, 2))
    for ch in range(2):
        out[:, ch] = air[ch] * ca + low[ch] * cl + carr * cc + cru * cr
    return out


def duck_curve(v, depth_db, lo_db=-54.0, hi_db=-40.0, attack=0.05, release=0.40, hop=120):
    p = uniform_filter1d(v * v, int(0.03 * SR), mode="nearest")[::hop]
    a = np.clip((10 * np.log10(p + 1e-12) - lo_db) / (hi_db - lo_db), 0, 1)
    ca, cr = np.exp(-hop / (attack * SR)), np.exp(-hop / (release * SR))
    s = np.zeros(len(a))
    cur = 0.0
    for i, x in enumerate(a):
        c = ca if x > cur else cr
        cur = c * cur + (1 - c) * x
        s[i] = cur
    g = gain(-depth_db * s)
    return np.interp(np.arange(NSAMP), np.arange(len(g)) * hop + hop / 2, g)


def gate_curve(t0, t1, fade_in):
    g = np.ones(NSAMP)
    i0, i1, r = int(t0 * SR), int(t1 * SR), int(0.003 * SR)
    g[i0:i0 + r] = 0.5 + 0.5 * np.cos(np.pi * np.arange(r) / r)
    g[i0 + r:i1] = 0.0
    n_in = int(fade_in * SR)
    g[i1:i1 + n_in] = 0.5 - 0.5 * np.cos(np.pi * np.arange(n_in) / n_in)
    return g


def write_pcm16(path, x, dither_seed=None):
    if dither_seed is not None:
        r = np.random.default_rng(dither_seed)
        x = x + (r.random(x.shape) - r.random(x.shape)) / 32768.0
    q = np.round(np.clip(x, -1.0, 32766 / 32768) * 32768).astype(np.int16)
    sf.write(str(path), q, SR, subtype="PCM_16")


def build_mix(units, extras=True, verbose=True):
    irs = build_irs()
    log = dict(narration=[], typed=[], appear=[], sfx=[])
    voice, keys, sfx, sting = Bus("voice"), Bus("keys"), Bus("sfx"), Bus("sting")

    for u in units:                                   # narration
        voice.add(u["x"], u["start"], 0.0, 0.0, sends=dict(room=-19), anchor=u["pre"])

    ref = thock_ref_peak()
    typed_end = {}
    for tid, text, t0 in TYPED:                       # typed labels, one thock per character
        sched = typing_schedule(text, t0, rng_for("type-" + tid))
        sends = dict(tight=-16, room=-22)
        if t0 >= 57.0:
            sends["hall"] = -15
        klog = []
        for i, (ch, tk) in enumerate(sched):
            r = rng_for(f"key-{tid}-{i}")
            vel = float(np.clip(1 + 0.09 * r.standard_normal(), 0.8, 1.15)) * (1.06 if ch == " " else 1.0)
            y, pan = keystroke(r, ch, vel=vel)
            keys.add(y / ref, tk, KEY_LEVEL_DB, pan, sends)
            klog.append(dict(char=ch, t=round(tk, 4), pan=round(pan, 3), vel=round(vel, 3)))
        typed_end[tid] = sched[-1][1]
        log["typed"].append(dict(id=tid, text=text, start=sched[0][1], last_key=sched[-1][1],
                                 end=sched[-1][1] + 0.12, keys=klog))
    for aid, tid, size in APPEAR:                     # soft deep taps for major titles
        tt = typed_end[tid] + 0.16
        sends = dict(tight=-18, room=-20)
        if tt >= 57.0:
            sends["hall"] = -9 if size == "L" else -14
        keys.add(soft_tap(rng_for("tap-" + aid), size), tt, TAP_LEVEL_DB + (1.5 if size == "L" else 0.0), 0.0, sends)
        log["appear"].append(dict(id=aid, after=tid, t=round(tt, 4), size=size, optional=True))

    for c in CUES:                                    # mechanical cues
        if c.get("extra") and not extras:
            continue
        y, anchor = c["fn"](rng_for("cue-" + c["id"]))
        (sting if c.get("bus") == "sting" else sfx).add(y, c["t"], c["gain"], 0.0, c["sends"], anchor=anchor)
        log["sfx"].append(dict(id=c["id"], t=c["t"], start=round(c["t"] - anchor, 4),
                               end=round(c["t"] - anchor + len(y) / SR, 4), peak_dbfs=c["gain"],
                               sends=c["sends"], extra=bool(c.get("extra")), desc=c["desc"]))

    stems = dict(voice=voice.render(irs), keys=keys.render(irs), sfx=sfx.render(irs), sting=sting.render(irs))
    log["voice_correction_db"] = VOICE_TARGET_LUFS - lufs(stems["voice"])
    stems["voice"] = stems["voice"] * gain(log["voice_correction_db"])
    bed = build_bed()
    vmono = stems["voice"].mean(axis=1)
    gate = gate_curve(*GATE)
    stems["sfx"] = stems["sfx"] * (duck_curve(vmono, DUCK["sfx"]) * gate)[:, None]
    stems["keys"] = stems["keys"] * duck_curve(vmono, DUCK["keys"])[:, None]
    stems["bed"] = bed * (duck_curve(vmono, DUCK["bed"]) * gate)[:, None]
    stems["keys"] = apply_sos(stems["keys"], sps.butter(2, 40, "highpass", fs=SR, output="sos"))
    return stems, log


def master(stems, trim_db):
    """Sum stems, remove subsonics, apply the safety limiter, fade the ends."""
    hp = sps.butter(2, 28, "highpass", fs=SR, output="sos")
    g = gain(trim_db)
    voice = apply_sos(stems["voice"], hp) * g
    fx = apply_sos(stems["keys"] + stems["sfx"] + stems["sting"] + stems["bed"], hp) * g
    mix, gr = safety_limiter(voice + fx, CEILING_DBFS)
    fi, fo = int(0.004 * SR), int(1.2 * SR)
    ramp_i, ramp_o = 0.5 - 0.5 * np.cos(np.pi * np.arange(fi) / fi), 0.5 + 0.5 * np.cos(np.pi * np.arange(fo) / fo)
    for a in (mix, voice, fx):
        a[:fi] *= ramp_i[:, None]
        a[-fo:] *= ramp_o[:, None]
    return mix, voice, fx, gr


def measure(x):
    return dict(sample_peak_dbfs=round(float(db(np.max(np.abs(x)))), 2),
                true_peak_dbtp=round(true_peak_db(x), 2),
                integrated_lufs=round(lufs(x), 2),
                rms_dbfs=round(float(rms_db(x)), 2),
                dc_offset=[round(float(v), 7) for v in x.mean(axis=0)])


def fit_rates(offline=False):
    """Print the smallest integer speaking-rate (%) that fits every block's slot with its
    configured sentence gap. Uses a +0% reference synthesis cached as <id>_fit0."""
    for cfg in NARRATION:
        text = " ".join(cfg["sentences"])
        x, words = tts_clip(cfg["id"] + "_fit0", text, 0, offline=offline)
        chunks, pk = split_sentences(x, words, cfg["sentences"])
        speech0 = sum(retime_chunk(x, c, pk, cfg["inner"])["dur"] for c in chunks)
        n_gap = len(chunks) - 1
        if "starts" in cfg:
            print(f"{cfg['id']}: sentences placed individually; speech at +0% = {speech0:.2f}s (rate is a style choice)")
            continue
        budget = cfg["max_end"] - cfg["start"] - END_MARGIN - n_gap * cfg["gap"]
        rate = max(-20, int(np.ceil((speech0 / budget - 1) * 100)))
        for _ in range(8):
            x, words = tts_clip(cfg["id"], text, rate, offline=offline)
            chunks, pk = split_sentences(x, words, cfg["sentences"])
            speech = sum(retime_chunk(x, c, pk, cfg["inner"])["dur"] for c in chunks)
            if speech <= budget:
                break
            rate += 1
        print(f"{cfg['id']}: speech@0%={speech0:5.2f}s  budget={budget:5.2f}s  -> rate {rate:+d}%  (speech {speech:5.2f}s)")


def r3(v):
    return round(float(v), 3)


def cues_document(units, log, stats, extras):
    narr = []
    for u in units:
        narr.append(dict(
            id=u["id"], block=u["block"], text=u["text"], rate_pct=u["rate"], start=r3(u["start"]), end=r3(u["end"]),
            slot_end=u["max_end"], fits_slot=bool(u["end"] <= u["max_end"]),
            sentences=[dict(text=s["text"], start=r3(u["start"] + s["on"]), end=r3(u["start"] + s["off"])) for s in u["spans"]],
            words=[dict(w=w["w"], start=r3(u["start"] + w["on"]), end=r3(u["start"] + w["off"])) for w in u["words"]],
            file=f"narration/{u['id']}.wav"))
    return dict(
        schema=1, generator="src/audio.py", sample_rate=SR, channels=2, duration_s=DURATION, format="PCM_16 wav",
        voice=dict(engine="edge-tts", name=VOICE, tts_format=TTS_FORMAT,
                   note="word times come from the service boundaries (+-60 ms); sentence times from the audio"),
        files=dict(mix="mix.wav", narration_only="narration_only.wav", effects_only="effects_only.wav",
                   narration_clips="narration/<id>.wav (dry, mono, onset at `start`)"),
        narration=narr,
        typed=[dict(id=t["id"], text=t["text"], start=r3(t["start"]), last_key=r3(t["last_key"]), end=r3(t["end"]),
                    keys=t["keys"]) for t in log["typed"]],
        appear_taps=log["appear"],
        sfx=log["sfx"],
        beds={k: [[a, b] for a, b in v] for k, v in BED.items()},
        automation=dict(
            silence_gate=dict(start=GATE[0], end=GATE[1], fade_in_s=GATE[2],
                              note="air, effects and tails are muted; relay_hard, keys and voice are not"),
            sidechain_duck_db=DUCK, voice_correction_db=round(log["voice_correction_db"], 2)),
        extras_enabled=bool(extras),
        master=stats)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh-tts", action="store_true", help="re-synthesise narration even if cached")
    ap.add_argument("--offline", action="store_true", help="never call the TTS service (cache only)")
    ap.add_argument("--no-extras", action="store_true", help="drop the optional extra cues (servo turn, shutdown relay, CRT)")
    ap.add_argument("--fit", action="store_true", help="print the rate that fits each narration slot, then exit")
    ap.add_argument("--target-lufs", type=float, default=-16.0, help="integrated loudness target for mix.wav")
    ap.add_argument("--out", default=str(OUT_DIR), help="output directory")
    args = ap.parse_args(argv)

    if args.fit:
        fit_rates(args.offline)
        return 0
    out = Path(args.out)
    (out / "narration").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("narration")
    units = narration_units(args.refresh_tts, args.offline)
    bad = [u["id"] for u in units if u["end"] > u["max_end"] - END_MARGIN + 1e-6]
    if bad:
        raise SystemExit(f"narration overruns its slot: {bad}; run with --fit and update NARRATION rates")
    print("mix")
    stems, log = build_mix(units, extras=not args.no_extras)
    mix, voice, fx, _ = master(stems, 0.0)
    trim = args.target_lufs - lufs(mix)
    pk = float(db(np.max(np.abs(voice + fx))))
    trim = min(trim, CEILING_DBFS - pk - 0.0)          # keep the limiter a pure safety net
    mix, voice, fx, gr = master(stems, trim)
    stats = dict(master_trim_db=round(trim, 2), limiter_max_reduction_db=round(-gr, 3),
                 mix=measure(mix), narration_only=measure(voice), effects_only=measure(fx),
                 stereo_correlation=round(float(np.corrcoef(mix[:, 0], mix[:, 1])[0, 1]), 3))
    write_pcm16(out / "mix.wav", mix, GLOBAL_SEED)
    write_pcm16(out / "narration_only.wav", voice, GLOBAL_SEED + 1)
    write_pcm16(out / "effects_only.wav", fx, GLOBAL_SEED + 2)
    for u in units:
        write_pcm16(out / "narration" / f"{u['id']}.wav", u["x"][:, None] * min(1.0, 0.9 / np.max(np.abs(u["x"]))))
    doc = cues_document(units, log, stats, not args.no_extras)
    (out / "cues.json").write_text(json.dumps(doc, indent=2) + "\n")
    m = stats["mix"]
    print(f"mix.wav: {len(mix) / SR:.3f}s  peak {m['sample_peak_dbfs']} dBFS  true-peak {m['true_peak_dbtp']} dBTP  "
          f"{m['integrated_lufs']} LUFS  trim {trim:+.2f} dB  limiter {stats['limiter_max_reduction_db']} dB")
    print(f"done in {time.time() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
