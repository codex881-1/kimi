# Sound design

The soundtrack contains **no music and no stock effects**. Every effect is
synthesized by `src/audio.py`; the finished cue sheet is `assets/audio/cues.json`.

## Voice

Narration uses a natural female English neural voice, generated as separate
sentences and placed against the picture. The wording was shortened where needed
rather than forcing the opening into an unnaturally fast read. Leading and
trailing silence is removed, and the voice receives light tonal and dynamics
processing for warmth and intelligibility.

The final voice starts at 00.80 s and ends at 64.52 s. Every line fits its own
shot and clears the following line; the four failure sentences land with their
respective diagrams. The last 3.48 seconds contain only the archive sound tail.

## Creamy keyboard thock

Each typed character triggers an original, varied keypress: a damped case/body
resonance, low “cream” noise, a restrained contact transient, pitch-settling
bottom-out and a softer top-out. Space presses are lower and longer. The sound is
low-passed and spatial variation is subtle, avoiding bright clicks and
machine-gun repetition.

Typing schedules use the **full visible label** and the same character rate as
the renderer, including dates and punctuation. Major text appearances use a
separate, softer low tactile tap rather than another full typing burst.

## Mechanical and environmental effects

| Picture | Sound construction |
|---|---|
| B-52 release | Hook creak, steel contact, ejector ram and airframe flex |
| Rocket boost | Ignition body, low-weighted combustion crackle and air rush |
| Guidance and camera | Gyro/relay behavior, optical shutter and small servos |
| Pod separation | Mechanical release and changing airflow |
| Parachute | Fabric movement and suspension-line tension |
| JC-130 recovery | Propeller/airframe approach and a restrained capture cue |
| Story turn at 36 s | Hard relay followed by a deliberate effects gate |
| Four failures | Guidance mechanism, canopy collapse, water and a restrained low impact |
| Blank film and archive | Film transport, case/contact sounds and a short archive close |

Effects are sidechain-ducked under speech. Ambient air is non-melodic and remains
well below the voice. There are no risers, braams, generic whooshes, sonar pings
or lens-flare effects.

## Master

- 48 kHz, stereo, PCM source mix
- 68.000 seconds exactly
- Approximately −16.8 LUFS integrated
- True peak at or below −1.8 dBTP
- AAC 256 kbit/s in the delivered MP4
