"""Procedural sound design. Every effect is synthesized from scratch (no samples, no music).

Design rules: deep and rounded rather than bright; transients softened; all randomness seeded;
each function returns (stereo float32 [N, 2], offset_seconds) where a negative offset lets a
sound start before its sync point (e.g. the inhale before the blast)."""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

SR = 48000


# ---------------------------------------------------------------- primitives
def t_axis(dur):
    return np.arange(int(dur * SR)) / SR


def rng_for(seed):
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def lp(x, f, order=2):
    return sosfilt(butter(order, min(f, SR * 0.45), "low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, "high", fs=SR, output="sos"), x)


def bp(x, f0, f1, order=2):
    return sosfilt(butter(order, [f0, min(f1, SR * 0.45)], "band", fs=SR, output="sos"), x)


def sweep_bp(x, f_start, f_end, q=1.2, blocks=64):
    """Time-varying band-pass (block-wise) for whooshes and wind."""
    n = len(x)
    out = np.zeros(n)
    edges = np.linspace(0, n, blocks + 1).astype(int)
    zi = None
    for b in range(blocks):
        u = b / max(1, blocks - 1)
        f = f_start * (f_end / f_start) ** u if f_start > 0 and f_end > 0 else f_start + (f_end - f_start) * u
        lo, hi = f / (1 + 0.5 / q), f * (1 + 0.5 / q)
        sos = butter(2, [max(lo, 20), min(hi, SR * 0.45)], "band", fs=SR, output="sos")
        seg = x[edges[b]:edges[b + 1]]
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        y, zi = _sosfilt_zi(sos, seg, zi)
        out[edges[b]:edges[b + 1]] = y
    return out


def _sosfilt_zi(sos, x, zi):
    from scipy.signal import sosfilt as f
    y, zf = f(sos, x, zi=zi)
    return y, zf


def pink(n, rng):
    w = rng.standard_normal(n)
    # Voss-ish via filtering white noise with -3dB/oct approximation
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    from scipy.signal import lfilter
    return lfilter(b, a, w) * 6


def brown(n, rng):
    w = rng.standard_normal(n)
    y = np.cumsum(w)
    y = hp(y, 18)
    return y / (np.abs(y).max() + 1e-9)


def env_exp(n, attack, decay):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-np.maximum(t - attack, 0) / max(decay, 1e-4))


def env_ar(n, attack, release, shape=2.0):
    t = np.arange(n) / SR
    dur = n / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1) ** shape
    r = np.clip((dur - t) / max(release, 1e-4), 0, 1) ** shape
    return a * r


def chirp(n, f0, f1, curve="exp"):
    t = np.arange(n) / SR
    dur = n / SR
    if curve == "exp":
        f = f0 * (f1 / f0) ** (t / dur)
    else:
        f = f0 + (f1 - f0) * (t / dur)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph)


def damped(n, f, decay, phase=0.0):
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * f * t + phase) * np.exp(-t / decay)


def stereo(x, pan=0.0, width=0.0, rng=None):
    pan = float(np.clip(pan, -1, 1))
    l, r = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    if width > 0:
        d = int(0.0004 * SR + width * 0.0008 * SR)
        xr = np.concatenate([np.zeros(d), x[:-d]]) if d > 0 else x
        return np.stack([x * l * 1.41, (x * (1 - width) + xr * width) * r * 1.41], 1).astype(np.float32)
    return np.stack([x * l * 1.41, x * r * 1.41], 1).astype(np.float32)


def soft_sat(x, drive=1.5):
    return np.tanh(x * drive) / np.tanh(drive)


def norm(x, peak=1.0):
    m = np.abs(x).max()
    return x / m * peak if m > 0 else x


# ---------------------------------------------------------------- effects
def key(ev):
    """Creamy, deep 'thock': lubed linear switch on a dense case. Bottom-out + softer top-out."""
    r = rng_for(ev.get("seed", 0))
    space = ev.get("space", False)
    n = int(0.16 * SR)
    f1 = r.uniform(150, 170) if space else r.uniform(205, 245)
    dec = r.uniform(0.030, 0.042) if space else r.uniform(0.020, 0.030)
    body = (damped(n, f1, dec) + 0.42 * damped(n, f1 * r.uniform(2.55, 2.8), dec * 0.55, 0.3)
            + 0.14 * damped(n, f1 * r.uniform(5.6, 6.3), dec * 0.28, 0.7))
    thud = lp(r.standard_normal(n), 520) * env_exp(n, 0.0008, 0.010) * 0.55
    tick = bp(r.standard_normal(n), 1800, 4200) * env_exp(n, 0.0002, 0.0018) * 0.10
    x = body * env_exp(n, 0.0012, 1.0) + thud + tick
    # top-out on release: lighter, a touch higher, 55-80 ms later
    off = int(r.uniform(0.055, 0.080) * SR)
    up = (damped(n - off, f1 * 1.18, dec * 0.6) * 0.20 + lp(r.standard_normal(n - off), 900)
          * env_exp(n - off, 0.0005, 0.006) * 0.12)
    x[off:] += up
    x = lp(soft_sat(x * 1.2, 1.3), 4200)
    return stereo(norm(x, 0.55 * r.uniform(0.85, 1.0)), ev.get("pan", 0) + r.uniform(-0.08, 0.08)), 0.0


def cream(ev):
    """Text appearance: rounded, buttery 'bloop' - low pitch-drop sine, breathy air, tiny sub tap."""
    r = rng_for(ev.get("seed", 0))
    w = ev.get("weight", 0.6)
    n = int(0.42 * SR)
    f0 = r.uniform(300, 360)
    tone = chirp(n, f0, f0 * 0.42) * env_exp(n, 0.004, 0.075 + 0.04 * w)
    air = lp(pink(n, r), 1300) * env_ar(n, 0.03, 0.30, 1.6) * 0.10
    sub = damped(n, 62, 0.07) * env_exp(n, 0.003, 1.0) * 0.35 * w
    x = lp(tone * 0.8 + air + sub, 2600)
    return stereo(norm(x, 0.50), ev.get("pan", 0), 0.4), 0.0


def heart(ev):
    n = int(0.9 * SR)
    def thump(nn, f0):
        return chirp(nn, f0, f0 * 0.62) * env_exp(nn, 0.006, 0.075)
    x = thump(n, 58)
    off = int(0.19 * SR)
    x[off:] += 0.62 * thump(n - off, 52)
    x = lp(soft_sat(x, 1.6), 300)
    return stereo(norm(x, 0.9)), 0.0


def deep_air(ev):
    dur = ev.get("dur", 4.0)
    r = rng_for(17)
    n = int(dur * SR)
    noise = pink(n, r)
    x = sweep_bp(noise, 180, 900, q=0.9) * env_ar(n, dur * 0.45, dur * 0.5, 1.5)
    sub = chirp(n, 42, 30) * env_ar(n, dur * 0.4, dur * 0.4, 2) * 0.35
    return stereo(norm(x, 0.45) + sub, 0, 0.8), 0.0


def whoosh(ev, f0=300, f1=2400, sub=0.0):
    dur = ev.get("dur", 1.0)
    r = rng_for(int(dur * 1000) + int(f0))
    n = int(dur * SR)
    x = sweep_bp(pink(n, r), f0, f1, q=1.1) * env_ar(n, dur * 0.55, dur * 0.42, 1.8)
    x = norm(x, 0.5)
    if sub:
        x += chirp(n, 70, 35) * env_ar(n, dur * 0.5, dur * 0.4, 2) * sub
    s = stereo(x, 0, 0.9)
    # slow pan sweep across the field
    pan = np.linspace(-0.6, 0.6, n)
    s[:, 0] *= (1 - pan) * 0.85
    s[:, 1] *= (1 + pan) * 0.85
    return s, -dur * 0.45


def whoosh_soft(ev):
    return whoosh(ev, 260, 1900)


def whoosh_deep(ev):
    return whoosh(ev, 90, 700, sub=0.5)


def time_whoosh(ev):
    s, o = whoosh(ev, 2600, 300, sub=0.25)
    return s, -0.2


def line_draw(ev):
    dur = ev.get("dur", 1.5)
    r = rng_for(3)
    n = int(dur * SR)
    x = bp(r.standard_normal(n), 2500, 7000) * (0.6 + 0.4 * np.sin(np.arange(n) / SR * 2 * np.pi * 9)) * env_ar(n, 0.1, 0.3)
    return stereo(norm(x, 0.2), 0.1), 0.0


def nuke(ev):
    r = rng_for(1964)
    pre = 0.45
    dur = 6.5
    n = int((pre + dur) * SR)
    x = np.zeros(n)
    p0 = int(pre * SR)
    # inhale before the flash
    m = p0
    x[:m] += sweep_bp(pink(m, r), 300, 2500, q=1.0) * np.linspace(0, 1, m) ** 3 * 0.25
    t = t_axis(dur)
    nd = len(t)
    # crack
    crack = lp(r.standard_normal(nd), 5000) * env_exp(nd, 0.0005, 0.012) * 0.9
    # sub boom (gliding down), saturated
    boom = chirp(nd, 52, 22) * env_exp(nd, 0.01, 1.6) * 1.0
    # body rumble
    body = lp(brown(nd, r), 260) * env_exp(nd, 0.05, 2.2) * 1.4
    am = 0.75 + 0.25 * np.sin(2 * np.pi * 3.3 * t + 1) * np.sin(2 * np.pi * 0.7 * t)
    # distant crackle
    crk = np.zeros(nd)
    for _ in range(140):
        i = int(r.uniform(0.05, 2.0) ** 1.3 * SR)
        if i < nd - 2000:
            g = r.uniform(0.1, 0.5) * math.exp(-i / SR / 0.9)
            crk[i:i + 600] += bp(r.standard_normal(600), 700, 3200) * np.exp(-np.arange(600) / 90) * g
    y = crack + soft_sat(boom * 1.4, 1.8) * 0.9 + body * am + crk * 0.5
    x[p0:p0 + nd] += y
    x = lp(x, 7000)
    s = stereo(norm(x, 0.95), 0, 0.7)
    return s, -pre


def missile(ev):
    dur = ev.get("dur", 1.9)
    r = rng_for(66)
    n = int(dur * SR)
    roar = lp(brown(n, r), 500) * 0.6
    jet = sweep_bp(pink(n, r), 500, 1800, q=1.4) * 0.6
    x = (roar + jet) * env_ar(n, 0.25, dur * 0.6, 1.5)
    s = stereo(norm(x, 0.5), 0.3, 0.5)
    pan = np.linspace(0.5, -0.1, n)
    s[:, 0] *= (1 - pan)
    s[:, 1] *= (1 + pan)
    return s, 0.0


def blip(ev):
    f = ev.get("freq", 1400) * 0.68
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * f * t + 0.6 * np.sin(2 * np.pi * 37 * t)) * env_exp(n, 0.004, 0.035)
    x = lp(x, 2400) + lp(rng_for(int(f)).standard_normal(n), 3000) * env_exp(n, 0.002, 0.02) * 0.05
    return stereo(norm(x, 0.35), ev.get("pan", 0)), 0.0


def paper_slide(ev):
    r = rng_for(int(ev["t"] * 100))
    dur = 0.55
    n = int(dur * SR)
    x = bp(r.standard_normal(n), 900, 6000) * env_ar(n, 0.12, 0.30, 1.5) * (0.7 + 0.3 * lp(r.standard_normal(n), 30) * 5)
    thump = np.zeros(n)
    i = int(0.36 * SR)
    thump[i:] = damped(n - i, 85, 0.05) * 0.8 + lp(r.standard_normal(n - i), 400) * env_exp(n - i, 0.001, 0.02) * 0.4
    return stereo(norm(x * 0.5, 0.35) + thump * 0.5, 0, 0.6), -0.30


def pencil(ev):
    dur = ev.get("dur", 1.0)
    r = rng_for(int(dur * 997))
    n = int(dur * SR)
    gr = np.clip(lp(r.standard_normal(n), 45) * 4, 0, 1)
    x = bp(r.standard_normal(n), 2200, 6500) * gr * env_ar(n, 0.05, 0.15)
    return stereo(norm(x, 0.18), 0.15), 0.0


def zap_soft(ev):
    dur = ev.get("dur", 0.7)
    n = int(dur * SR)
    x = chirp(n, 820, 260) * env_exp(n, 0.01, dur * 0.35) * 0.5
    x += sweep_bp(pink(n, rng_for(5)), 3000, 600, q=1.5) * env_exp(n, 0.005, dur * 0.3) * 0.5
    return stereo(norm(lp(x, 3500), 0.4), 0.2, 0.4), 0.0


def thud_deny(ev):
    n = int(0.7 * SR)
    r = rng_for(9)
    x = chirp(n, 78, 42) * env_exp(n, 0.003, 0.12) + lp(r.standard_normal(n), 300) * env_exp(n, 0.001, 0.04) * 0.6
    x += bp(r.standard_normal(n), 300, 1200) * env_exp(n, 0.001, 0.015) * 0.3
    return stereo(norm(soft_sat(x, 1.5), 0.85)), 0.0


def rise(ev):
    dur = ev.get("dur", 1.3)
    n = int(dur * SR)
    x = sweep_bp(pink(n, rng_for(11)), 180, 1600, q=1.3) * env_ar(n, dur * 0.7, 0.2, 1.6)
    return stereo(norm(x, 0.4), -0.2, 0.6), 0.0


def lock(ev):
    n = int(0.5 * SR)
    r = rng_for(13)
    x = np.zeros(n)
    for k, off in enumerate((0, int(0.07 * SR))):
        x[off:] += bp(r.standard_normal(n - off), 1500, 3500) * env_exp(n - off, 0.0003, 0.003) * 0.5
    i = int(0.07 * SR)
    x[i:] += (np.sin(2 * np.pi * 980 * np.arange(n - i) / SR) * env_exp(n - i, 0.003, 0.09) * 0.35
              + damped(n - i, 140, 0.05) * 0.4)
    return stereo(norm(lp(x, 4500), 0.45)), 0.0


def scan(ev):
    dur = ev.get("dur", 1.4)
    n = int(dur * SR)
    x = sweep_bp(pink(n, rng_for(21)), 400, 3800, q=2.0) * env_ar(n, 0.3, 0.5)
    sh = bp(rng_for(22).standard_normal(n), 5000, 9000) * env_ar(n, 0.4, 0.5) * 0.15
    return stereo(norm(x + sh, 0.35), 0, 0.8), 0.0


def tick(ev):
    r = rng_for(ev.get("seed", 0) + 300)
    n = int(0.05 * SR)
    x = bp(r.standard_normal(n), 1800, 3600) * env_exp(n, 0.0003, 0.0025) + damped(n, 420, 0.008) * 0.4
    return stereo(norm(lp(x, 5000), 0.3), 0.35), 0.0


def capsules(ev):
    n = int(1.2 * SR)
    r = rng_for(7)
    x = chirp(n, 90, 55) * env_exp(n, 0.02, 0.18) * 0.6
    for k in range(7):
        off = int((0.05 + k * 0.055 + r.uniform(0, 0.015)) * SR)
        m = n - off
        f = r.uniform(1700, 2100)
        tk = (damped(m, f, 0.025) + 0.5 * damped(m, f * 1.53, 0.015) + 0.3 * damped(m, f * 0.51, 0.04)) * 0.22
        x[off:] += tk
    return stereo(norm(lp(x, 6000), 0.5), 0.15, 0.5), 0.0


def freeze(ev):
    dur = ev.get("dur", 0.9)
    n = int(dur * SR)
    r = rng_for(31)
    x = np.zeros(n)
    for k in range(260):
        i = int((r.uniform(0, 1) ** 0.7) * (n - 400))
        x[i:i + 300] += hp(r.standard_normal(300), 3500) * np.exp(-np.arange(300) / 25) * r.uniform(0.1, 0.6)
    hiss = hp(pink(n, r), 4000) * env_ar(n, dur * 0.6, 0.2) * 0.2
    return stereo(norm(lp(x + hiss, 9000), 0.3), -0.3, 0.7), 0.0


def power_down(ev):
    n = int(0.75 * SR)
    x = chirp(n, 420, 55) * env_exp(n, 0.01, 0.25)
    x = lp(x, 1200)
    i = int(0.62 * SR)
    x[i:] += bp(rng_for(4).standard_normal(n - i), 800, 3000) * env_exp(n - i, 0.0005, 0.004) * 0.6
    return stereo(norm(x, 0.4), -0.3), 0.0


def warm_swell(ev):
    dur = ev.get("dur", 1.6)
    n = int(dur * SR)
    r = rng_for(41)
    x = lp(brown(n, r), 240) * env_ar(n, dur * 0.4, dur * 0.55, 1.5)
    x += chirp(n, 48, 56) * env_ar(n, dur * 0.4, dur * 0.5) * 0.4
    return stereo(norm(x, 0.55), 0.1, 0.6), 0.0


def electric(ev):
    dur = ev.get("dur", 1.2)
    n = int(dur * SR)
    t = np.arange(n) / SR
    saw = 2 * ((t * 100) % 1) - 1
    hum = lp(saw, 500) * 0.35
    r = rng_for(51)
    crk = np.zeros(n)
    for _ in range(40):
        i = int(r.uniform(0, n - 500))
        crk[i:i + 400] += bp(r.standard_normal(400), 2000, 6000) * np.exp(-np.arange(400) / 40) * r.uniform(0.1, 0.4)
    x = (hum + crk * 0.5) * env_ar(n, 0.2, 0.4)
    return stereo(norm(lp(x, 5000), 0.3), -0.2, 0.5), 0.0


def wind(dur, seed, intensity=1.0, gust_rate=0.35, lo=150, hi=1100):
    n = int(dur * SR)
    r = rng_for(seed)
    t = np.arange(n) / SR
    base = pink(n, r)
    gust = 0.55 + 0.45 * np.clip(lp(r.standard_normal(n), 0.8) * 30 + 0.5, 0, 1)
    f_mod = lo + (hi - lo) * (0.5 + 0.5 * np.sin(2 * np.pi * gust_rate * t + 1.3))
    x = sweep_bp(base, lo * 1.8, hi * 0.8, q=0.8, blocks=96)
    whistle = sweep_bp(pink(n, r), 640, 880, q=8.0, blocks=96) * 0.08
    return (x + whistle) * gust * intensity


def blizzard(ev):
    dur = ev.get("dur", 2.0)
    n = int(dur * SR)
    L = wind(dur, 61, 1.0, 0.5, 200, 1600)
    R = wind(dur, 62, 1.0, 0.45, 200, 1600)
    roar = lp(brown(n, rng_for(63)), 220) * 0.6
    e = env_ar(n, 0.45, 0.8, 1.4)
    s = np.stack([(L + roar) * e, (R + roar) * e], 1)
    return (norm(s, 0.75)).astype(np.float32), -0.25


def avalanche(ev):
    dur = ev.get("dur", 2.6)
    n = int(dur * SR)
    r = rng_for(71)
    rumble = lp(brown(n, r), 140) * 1.4
    crunch = bp(r.standard_normal(n), 250, 1100) * np.clip(lp(r.standard_normal(n), 25) * 6, 0, 1) * 0.6
    sub = chirp(n, 38, 28) * 0.5
    e = env_ar(n, 0.35, dur * 0.55, 1.3)
    x = soft_sat((rumble + crunch + sub) * e, 1.6)
    return stereo(norm(x, 0.9), -0.1, 0.8), -0.1


def rope(ev):
    dur = ev.get("dur", 1.0)
    n = int(dur * SR)
    r = rng_for(81)
    fr = bp(r.standard_normal(n), 900, 3200) * np.clip(lp(r.standard_normal(n), 20) * 5 + 0.4, 0, 1) * 0.3
    creak = np.zeros(n)
    for k in range(4):
        i = int(r.uniform(0, n - 6000))
        creak[i:i + 6000] += damped(6000, r.uniform(280, 460), 0.04) * 0.4
    x = (fr + creak) * env_ar(n, 0.05, 0.3)
    return stereo(norm(lp(x, 5000), 0.35), 0.1, 0.4), 0.0


def knot(ev):
    n = int(0.5 * SR)
    r = rng_for(83)
    x = chirp(n, 95, 60) * env_exp(n, 0.002, 0.07) + damped(n, 330, 0.06) * 0.3
    x += bp(r.standard_normal(n), 600, 2500) * env_exp(n, 0.001, 0.02) * 0.4
    return stereo(norm(x, 0.6)), 0.0


def step_snow(ev):
    r = rng_for(ev.get("seed", 0) + 900)
    n = int(0.22 * SR)
    x = np.zeros(n)
    for k in range(60):
        i = int(r.uniform(0, 0.12) * SR)
        m = min(220, n - i)
        x[i:i + m] += bp(r.standard_normal(m), 900, 4500) * np.exp(-np.arange(m) / 30) * r.uniform(0.2, 1)
    x = lp(x, 6000) * env_exp(n, 0.005, 0.08)
    x += lp(r.standard_normal(n), 250) * env_exp(n, 0.002, 0.03) * 0.5
    return stereo(norm(x, 0.35), ev.get("pan", 0)), 0.0


def absence(ev):
    dur = ev.get("dur", 1.6)
    n = int(dur * SR)
    r = rng_for(91)
    x = chirp(n, 36, 44) * env_ar(n, dur * 0.3, dur * 0.65, 1.4) * 0.6
    x += sweep_bp(pink(n, r), 1600, 250, q=1.0) * env_ar(n, 0.05, dur * 0.8, 1.5) * 0.4
    return stereo(norm(x, 0.55), 0, 0.9), 0.0


def flap(ev):
    r = rng_for(ev.get("seed", 0) + 500)
    n = int(0.08 * SR)
    x = bp(r.standard_normal(n), 1400, 3800) * env_exp(n, 0.0004, 0.004) * 0.7
    x += damped(n, r.uniform(280, 340), 0.012) * 0.5
    return stereo(norm(lp(x, 5500), 0.4), ev.get("pan", 0)), 0.0


def geiger(ev):
    r = rng_for(ev.get("seed", 0) + 700)
    n = int(0.03 * SR)
    x = hp(r.standard_normal(n), 2000) * env_exp(n, 0.0001, 0.0008) + damped(n, 3200, 0.002) * 0.3
    return stereo(norm(x, 0.35), ev.get("pan", 0)), 0.0


def sonar(ev):
    dur = ev.get("dur", 2.4)
    n = int(dur * SR)
    x = lp(pink(n, rng_for(101)), 700) * env_ar(n, 0.4, 0.8) * 0.5
    t = np.arange(n) / SR
    for k, t0 in enumerate((0.0, 1.2)):
        i = int(t0 * SR)
        m = n - i
        x[i:] += np.sin(2 * np.pi * 620 * t[:m]) * env_exp(m, 0.006, 0.18) * 0.18
    return stereo(norm(lp(x, 3000), 0.35), 0, 0.7), 0.0


def nothing(ev):
    n = int(1.6 * SR)
    r = rng_for(111)
    x = chirp(n, 60, 32) * env_exp(n, 0.004, 0.35) + lp(brown(n, r), 180) * env_exp(n, 0.01, 0.5) * 0.7
    return stereo(norm(soft_sat(x, 1.4), 0.85), 0, 0.5), 0.0


def marker(ev):
    dur = ev.get("dur", 0.5)
    n = int(dur * SR)
    r = rng_for(121)
    x = bp(r.standard_normal(n), 1200, 5000) * np.clip(lp(r.standard_normal(n), 60) * 4 + 0.5, 0, 1) * env_ar(n, 0.04, 0.1)
    return stereo(norm(x, 0.2), 0.2), 0.0


def stamp_soft(ev):
    n = int(0.45 * SR)
    r = rng_for(131)
    x = damped(n, 95, 0.05) + lp(r.standard_normal(n), 900) * env_exp(n, 0.001, 0.02) * 0.6
    return stereo(norm(x, 0.55)), 0.0


def water(ev):
    dur = ev.get("dur", 5.0)
    n = int(dur * SR)
    r = rng_for(141)
    rush = lp(pink(n, r), 900) * 0.5
    bub = np.zeros(n)
    t = np.arange(2400) / SR
    for _ in range(int(dur * 28)):
        i = int(r.uniform(0, n - 2400))
        f0 = r.uniform(350, 1100)
        bub[i:i + 2400] += np.sin(2 * np.pi * (f0 + 900 * t) * t) * np.exp(-t / 0.012) * r.uniform(0.05, 0.25)
    x = (rush + lp(bub, 3000)) * env_ar(n, 0.8, 1.2)
    return stereo(norm(x, 0.4), -0.1, 0.9), 0.0


SYNTHS = {k: v for k, v in globals().items() if callable(v) and not k.startswith("_")}


def render_event(ev):
    f = SYNTHS.get(ev["kind"])
    if f is None:
        raise KeyError(f"no synth for {ev['kind']}")
    s, off = f(ev)
    return s, off
