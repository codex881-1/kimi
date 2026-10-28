# Production design

## Approach

The film uses a deterministic Skia renderer rather than browser animation or a
conventional template. A small 3D mesh and lighting projection builds the D-21B;
separate procedural assemblies build the B-52H, JC-130B, camera, film pod,
parachute, ocean, terrain and display case. Natural Earth geometry supplies the
actual outline of China. Atmospheric haze, film grain, vignetting and paper
texture are synthesized locally.

The palette is obsidian, warm ivory, muted metal and vermilion. Space Grotesk
carries the narrative headlines; IBM Plex Mono carries evidence, dates and
telemetry. Chinese labels are secondary, never a substitute for the English
explanation.

## Visual argument

| Time | Mechanism | Visual purpose |
|---|---|---|
| 00–07 | D-21B, camera and stored route | Establish the strange machine |
| 07–14.5 | B-52H release and rocket boost | Explain how it reached speed |
| 14.5–21.5 | Lop Nor and an autonomous route | Locate the target; explain guidance |
| 21.5–27.5 | Camera bay and recoverable film pod | Separate the film from the aircraft |
| 27.5–36 | Eject, deploy, catch | Make the recovery plan understandable |
| 36–38 | Deliberate visual stop | Turn the story |
| 38–51 | Four individual failure diagrams | Show exactly how the plan broke |
| 51–57 | Four missions, zero usable frames | State the operational result |
| 57–65 | Fragmented airframe in a public case | Reveal the aftermath |
| 65–68 | Archival end card | Retain the central fact |

Motion is rendered at a native 60 fps. Easing is used for physical mechanisms,
not indiscriminate bouncing. Static holds are intentional after each failure and
on the final title.

## Asset policy

All required visual assets were producible in code, so no image-generation
prompts or external reconstruction images were needed. The maps and fonts are
the only downloaded visual inputs.
