"""Sound-event registry. Visual components declare sounds at construction time so the
audio mix is sample-accurate to the picture."""
from __future__ import annotations

EVENTS: list[dict] = []


def sfx(t, kind, gain_db=0.0, pan=0.0, **params):
    EVENTS.append({"t": float(t), "kind": kind, "gain_db": float(gain_db), "pan": float(pan), **params})


def reset():
    EVENTS.clear()
