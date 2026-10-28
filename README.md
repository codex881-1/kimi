# SENIOR BOWL — Four Flights. Zero Film.

A 68-second vertical documentary about the CIA and U.S. Air Force D-21B
reconnaissance drone missions over China. All picture elements and sound effects
are generated from original code; there is no music and no stock footage.

## Final format

- 1080 × 1920, 60 fps, H.264
- 48 kHz stereo AAC
- Natural female English narration with plain-language explanations
- Runtime: 68.000 seconds

## Rebuild

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make audio
make picture
make mux
```

The finished film is written to `dist/senior-bowl.mp4`. Historical claims and
their sources are separated in `docs/research.md`; the visual and sound systems
are described in `docs/production.md` and `docs/sound-design.md`.
