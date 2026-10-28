"""Generate the narration with Kokoro (af_heart), one line at a time, keeping
Kokoro's own per-word timestamps. Then assemble with designed breaths/pauses.

Outputs: vo/lines/Lxx.wav (24 kHz), vo/vo_raw.wav (48 kHz), vo/vo_timing.json
"""
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
LINES = ROOT / "lines"
LINES.mkdir(exist_ok=True)

VOICE = "af_heart"
# (id, text with pronunciation hints, speed, pause_after_s)
SCRIPT = [
    ("L01", "In 1965, the C.I.A. lost a nuclear device on a mountain in India.", 1.0, 0.30),
    ("L02", "It has never been found.", 0.96, 0.55),
    ("L03", "And it was built to spy on China.", 1.0, 0.50),
    ("L04", "A year earlier, China had tested its first atomic bomb, at [Lop Nur](/lˈɔp nˈʊɹ/).", 1.0, 0.30),
    ("L05", "Missiles would be next, and America wanted to listen in.", 1.0, 0.45),
    ("L06", "But radio travels in straight lines. Mountains block it.", 1.0, 0.25),
    ("L07", "So you listen from as high as you can.", 1.0, 0.45),
    ("L08", "The C.I.A. and Indian intelligence chose [Nanda Devi](/nˈʌndə dˈeɪvi/). Nearly eight kilometers up.", 1.0, 0.45),
    ("L09", "The device: an antenna, two radios, and seven capsules of plutonium.", 1.0, 0.30),
    ("L10", "Up there, batteries die. Plutonium doesn't. Its heat becomes electricity.", 1.0, 0.50),
    ("L11", "In October 1965, climbers carried it up. Near the top, a blizzard hit.", 1.0, 0.30),
    ("L12", "They tied it into a crack in the rock, and turned back.", 0.97, 0.60),
    ("L13", "In spring, they returned. It was gone, swept down the mountain.", 1.0, 0.45),
    ("L14", "They searched for years with radiation detectors. Nothing.", 1.0, 0.50),
    ("L15", "In 1978, India's prime minister admitted it in parliament.", 1.0, 0.50),
    ("L16", "As far as anyone knows, the plutonium is still up there, in the ice that feeds the Ganges.", 1.0, 0.55),
    ("L17", "And sixty years later, it's still warm.", 0.94, 0.0),
]
LEAD_IN = 0.45


def clean_word(w):
    return re.sub(r"[^\w'\-.]", "", w)


def main():
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a")
    timeline, chunks = [], []
    cursor = LEAD_IN
    chunks.append(np.zeros(int(LEAD_IN * 24000), np.float32))
    for lid, text, speed, pause in SCRIPT:
        audio, words = [], []
        off = 0.0
        for res in pipe(text, voice=VOICE, speed=speed):
            a = res.audio.numpy()
            for tk in (res.tokens or []):
                if tk.start_ts is None or not clean_word(tk.text):
                    continue
                words.append({"w": tk.text, "t0": round(off + tk.start_ts, 3), "t1": round(off + tk.end_ts, 3)})
            audio.append(a)
            off += len(a) / 24000
        a = np.concatenate(audio)
        # trim leading/trailing near-silence but keep natural breath room
        env = np.abs(a) > 0.006
        i0 = max(0, np.argmax(env) - int(0.04 * 24000))
        i1 = min(len(a), len(a) - np.argmax(env[::-1]) + int(0.12 * 24000))
        a = a[i0:i1]
        shift = i0 / 24000
        for w in words:
            w["t0"] = round(w["t0"] - shift, 3)
            w["t1"] = round(w["t1"] - shift, 3)
        sf.write(LINES / f"{lid}.wav", a, 24000)
        dur = len(a) / 24000
        timeline.append({"id": lid, "text": re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text).replace("C.I.A.", "CIA"),
                         "start": round(cursor, 3), "dur": round(dur, 3),
                         "words": [{**w, "t0": round(cursor + w["t0"], 3), "t1": round(cursor + w["t1"], 3)} for w in words]})
        print(f"{lid} {dur:5.2f}s @{cursor:6.2f}  {len(words)} words")
        chunks.append(a)
        chunks.append(np.zeros(int(pause * 24000), np.float32))
        cursor += dur + pause
    full = np.concatenate(chunks)
    full48 = resample_poly(full, 2, 1).astype(np.float32)
    sf.write(ROOT / "vo_raw.wav", full48, 48000, subtype="FLOAT")
    (ROOT / "vo_timing.json").write_text(json.dumps({"total": round(len(full) / 24000, 3), "lines": timeline}, indent=1))
    print("TOTAL", round(len(full) / 24000, 2), "s")


if __name__ == "__main__":
    main()
