"""Voice A/B test: Kokoro (several female voices) vs Chatterbox on the same lines."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "takes"
OUT.mkdir(exist_ok=True)
lines = json.loads((ROOT / "script_v1.json").read_text())
TEST_IDS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["L01", "L04", "L10", "L16"]
engine = sys.argv[1]
spoken = {l["id"]: l["text"].replace("1965", "nineteen sixty-five").replace("1978", "nineteen seventy-eight")
          .replace("CIA", "C.I.A.") for l in lines}

if engine == "kokoro":
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a")
    for voice in ["af_heart", "af_bella", "af_nicole", "bf_emma"]:
        if voice.startswith("b"):
            p = KPipeline(lang_code="b")
        else:
            p = pipe
        for i in TEST_IDS:
            t = time.time()
            chunks = [a for _, _, a in p(spoken[i], voice=voice, speed=0.95)]
            wav = np.concatenate([c.numpy() for c in chunks])
            sf.write(OUT / f"ko_{voice}_{i}.wav", wav, 24000)
            print(voice, i, f"{len(wav)/24000:.2f}s audio in {time.time()-t:.1f}s", flush=True)
elif engine == "chatterbox":
    import torch
    torch.set_num_threads(2)
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device="cpu")
    for i in TEST_IDS:
        t = time.time()
        wav = model.generate(spoken[i], exaggeration=0.45, cfg_weight=0.4, temperature=0.7)
        sf.write(OUT / f"cb_default_{i}.wav", wav.squeeze(0).numpy(), model.sr)
        print("cb", i, f"{wav.shape[-1]/model.sr:.2f}s audio in {time.time()-t:.1f}s", flush=True)
