"""Mix + master: VO chain, SFX buses with reverb sends, scene ambience beds, sidechain ducking,
loudness normalisation (-14 LUFS integrated, <= -1 dBTP).

usage: python audio/mix.py  -> out/master.wav"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pedalboard import (Pedalboard, HighpassFilter, LowpassFilter, LowShelfFilter, HighShelfFilter, PeakFilter,
                        Compressor, Reverb, Limiter, Gain)
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from audio import synth as S  # noqa: E402

SR = S.SR
BUS = {
    "keys": {"key"},
    "ui": {"cream", "blip", "tick", "flap", "lock", "geiger"},
}
BUS_GAIN_DB = {"keys": -15.5, "ui": -17.0, "fx": -11.5, "amb": -30.0}


def bus_of(kind):
    for b, kinds in BUS.items():
        if kind in kinds:
            return b
    return "fx"


def place(buf, s, t):
    i = int(round(t * SR))
    if i < 0:
        s = s[-i:]
        i = 0
    j = min(len(buf), i + len(s))
    if j > i:
        buf[i:j] += s[:j - i]


def ambience(total):
    """Noise-only beds (non-tonal), crossfaded per scene world."""
    n = int(total * SR)
    out = np.zeros((n, 2), np.float32)

    def bed(t0, t1, sig, fade=0.4):
        i0, i1 = int(t0 * SR), min(n, int(t1 * SR))
        m = i1 - i0
        x = sig[:m]
        e = np.ones(m)
        f = int(fade * SR)
        e[:f] = np.linspace(0, 1, f) ** 2
        e[-f:] = np.linspace(1, 0, f) ** 2
        out[i0:i1] += x * e[:, None]

    r = np.random.default_rng(5)

    def wind_bed(dur, seed, k=1.0, lo=120, hi=700):
        a = S.wind(dur, seed, k, 0.18, lo, hi)
        b = S.wind(dur, seed + 1, k, 0.16, lo, hi)
        return np.stack([a, b], 1).astype(np.float32)

    def ice_bed(dur, seed):
        m = int(dur * SR)
        rr = np.random.default_rng(seed)
        low = S.lp(S.brown(m, rr), 90) * 0.9
        creak = np.zeros(m)
        for _ in range(int(dur * 0.9)):
            i = int(rr.uniform(0, m - 30000))
            seg = S.bp(rr.standard_normal(30000), 120, 900) * np.exp(-np.arange(30000) / 7000) * rr.uniform(0.1, 0.35)
            creak[i:i + 30000] += seg * (0.5 + 0.5 * np.sin(np.arange(30000) / SR * 2 * np.pi * rr.uniform(9, 22)))
        x = low + creak
        return np.stack([x, np.roll(x, 480)], 1).astype(np.float32)

    def room(dur, seed, f=1800):
        m = int(dur * SR)
        rr = np.random.default_rng(seed)
        x = S.lp(S.pink(m, rr), f) * 0.25
        return np.stack([x, S.lp(S.pink(m, rr), f) * 0.25], 1).astype(np.float32)

    def hum(dur, seed):
        m = int(dur * SR)
        rr = np.random.default_rng(seed)
        x = S.lp(S.brown(m, rr), 160) * 0.6 + S.bp(S.pink(m, rr), 300, 2500) * 0.08
        return np.stack([x, np.roll(x, 300)], 1).astype(np.float32)

    bed(0.0, 7.9, ice_bed(8.0, 1) * 1.1, 0.6)
    bed(7.45, 18.0, hum(10.6, 2) * 0.9 + wind_bed(10.6, 3, 0.25), 0.5)
    bed(17.55, 23.8, room(6.3, 4), 0.3)
    bed(23.45, 29.5, wind_bed(6.1, 5, 0.9), 0.4)
    bed(29.05, 39.2, room(10.2, 6), 0.3)
    bed(38.95, 44.8, wind_bed(5.9, 7, 1.0), 0.4)
    bed(44.45, 50.3, wind_bed(5.9, 8, 0.9, 150, 900), 0.4)
    bed(50.05, 52.0, wind_bed(2.0, 9, 0.8), 0.25)
    bed(51.75, 55.8, hum(4.1, 10) * 0.8 + wind_bed(4.1, 11, 0.3), 0.3)
    bed(55.45, 60.4, room(5.0, 12), 0.3)
    bed(60.15, 66.1, wind_bed(6.0, 13, 0.45) + hum(6.0, 14) * 0.4, 0.4)
    bed(65.85, total, ice_bed(total - 65.85 + 0.1, 15) * 1.1, 0.5)
    m = np.abs(out).max()
    return out / (m + 1e-9) * 0.8


def vo_chain(x):
    board = Pedalboard([
        HighpassFilter(72),
        LowShelfFilter(cutoff_frequency_hz=170, gain_db=1.8, q=0.7),
        PeakFilter(cutoff_frequency_hz=320, gain_db=-1.5, q=1.0),      # de-box
        PeakFilter(cutoff_frequency_hz=6800, gain_db=-2.5, q=2.0),     # static de-ess
        HighShelfFilter(cutoff_frequency_hz=10500, gain_db=1.5, q=0.7),  # air
        Compressor(threshold_db=-22, ratio=2.6, attack_ms=6, release_ms=90),
        Gain(4.0),
    ])
    y = board(x.T.astype(np.float32), SR).T
    room = Pedalboard([Reverb(room_size=0.16, damping=0.7, wet_level=1.0, dry_level=0.0, width=0.6)])
    wet = room(y.T.astype(np.float32), SR).T
    return y + wet * 0.055


def lookahead_limit(x, ceiling_db=-1.2, look_ms=5.0, release_ms=90.0):
    """Peak limiter: 4x-oversampled detector, look-ahead min-filter, exponential release."""
    from scipy.ndimage import minimum_filter1d
    from numba import njit
    ceil = 10 ** (ceiling_db / 20)
    over = np.abs(resample_poly(x, 4, 1, axis=0)).max(1).reshape(-1, 4).max(1)[:len(x)]
    need = np.minimum(1.0, ceil / np.maximum(over, 1e-9))
    la = int(look_ms / 1000 * SR)
    g = minimum_filter1d(need, size=2 * la + 1, mode="nearest")

    @njit(cache=True)
    def release(g, a):
        out = np.empty_like(g)
        cur = 1.0
        for i in range(len(g)):
            v = g[i]
            cur = v if v < cur else cur + (v - cur) * a
            out[i] = cur
        return out

    g = release(g, 1 - np.exp(-1 / (release_ms / 1000 * SR)))
    g = np.convolve(g, np.ones(la) / la, "same")
    g = np.minimum(g, need)  # never let smoothing overshoot the ceiling
    return (x * g[:, None]).astype(np.float32)


def main():
    from film import make_timeline
    from engine.sfx import EVENTS
    scenes, caps, total = make_timeline()
    total_n = int(total * SR)
    buses = {b: np.zeros((total_n, 2), np.float32) for b in ("keys", "ui", "fx")}
    missing = set()
    for ev in EVENTS:
        try:
            s, off = S.render_event(ev)
        except KeyError:
            missing.add(ev["kind"])
            continue
        g = 10 ** (ev.get("gain_db", 0) / 20)
        place(buses[bus_of(ev["kind"])], s * g, ev["t"] + off)
    if missing:
        print("MISSING SYNTHS:", missing)
    print(f"{len(EVENTS)} events")

    # bus processing: keys get a small warm room, ui a soft plate, fx a larger space
    keys = Pedalboard([LowpassFilter(5200), Reverb(room_size=0.12, damping=0.8, wet_level=0.10, dry_level=1.0, width=0.5)])
    ui = Pedalboard([LowpassFilter(4200), Reverb(room_size=0.35, damping=0.6, wet_level=0.18, dry_level=1.0, width=0.9)])
    fxb = Pedalboard([Reverb(room_size=0.55, damping=0.5, wet_level=0.16, dry_level=1.0, width=1.0)])
    proc = {
        "keys": keys(buses["keys"].T, SR).T,
        "ui": ui(buses["ui"].T, SR).T,
        "fx": fxb(buses["fx"].T, SR).T,
    }
    amb = ambience(total)

    # VO
    vo, sr = sf.read(ROOT / "vo" / "vo_raw.wav", dtype="float32")
    if vo.ndim == 1:
        vo = np.stack([vo, vo], 1)
    vo = vo_chain(vo)
    vo_full = np.zeros((total_n, 2), np.float32)
    vo_full[:min(total_n, len(vo))] = vo[:total_n]

    # sidechain ducking from VO envelope
    env = np.sqrt(np.convolve(vo_full[:, 0] ** 2, np.ones(2400) / 2400, "same"))
    env = env / (env.max() + 1e-9)
    duck = 1 - 0.45 * np.clip(env * 3.0, 0, 1)
    duck = np.convolve(duck, np.ones(4800) / 4800, "same")[:, None]

    def db(x):
        return 10 ** (x / 20)

    mix = (vo_full
           + proc["keys"] * db(BUS_GAIN_DB["keys"]) * (0.75 + 0.25 * duck)
           + proc["ui"] * db(BUS_GAIN_DB["ui"]) * (0.8 + 0.2 * duck)
           + proc["fx"] * db(BUS_GAIN_DB["fx"]) * duck
           + amb * db(BUS_GAIN_DB["amb"]) * duck)
    # final fade
    f = int(0.6 * SR)
    mix[-f:] *= np.linspace(1, 0, f)[:, None] ** 2
    master = Pedalboard([Compressor(threshold_db=-14, ratio=1.8, attack_ms=20, release_ms=200)])
    mix = master(mix.T.astype(np.float32), SR).T
    meter = pyln.Meter(SR)
    for _ in range(3):
        lufs = meter.integrated_loudness(mix)
        mix = mix * 10 ** ((-14.0 - lufs) / 20)
        mix = lookahead_limit(mix, ceiling_db=-1.2)
    tp = 20 * np.log10(np.abs(resample_poly(mix, 4, 1, axis=0)).max() + 1e-12)
    lufs2 = meter.integrated_loudness(mix)
    out = ROOT / "out" / "master.wav"
    sf.write(out, mix, SR, subtype="PCM_24")
    # stems for QA
    for k in ("keys", "ui", "fx"):
        sf.write(ROOT / "out" / f"stem_{k}.wav", proc[k] * db(BUS_GAIN_DB[k]), SR, subtype="PCM_16")
    print(f"master: {out}  integrated {lufs2:.2f} LUFS  true-peak ~{tp:.2f} dBTP  dur {len(mix) / SR:.2f}s")


if __name__ == "__main__":
    main()
