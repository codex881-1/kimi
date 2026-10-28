"""Master timeline: scenes, their windows, and the caption spec."""
from __future__ import annotations

import json

from engine.core import ROOT
from engine.captions import Captions
from engine import sfx as SFX

TIMING = json.loads((ROOT / "vo" / "vo_timing.json").read_text())
DURATION = 69.8

CAPTIONS = {
    "L01": ["In *1965*,", "the CIA lost a *nuclear* *device*", "on a mountain in *India*."],
    "L02": ["It has *never* *been* *found*."],
    "L03": ["And it was built", "to *spy* *on* *China*.{red}"],
    "L04": ["A year earlier,", "China had tested its *first* *atomic* *bomb*,", "at *Lop* *Nur*.{red}"],
    "L05": ["Missiles would be *next*,", "and America wanted to *listen* *in*."],
    "L06": ["But radio travels in *straight* *lines*.", "*Mountains* *block* *it*."],
    "L07": ["So you listen from", "*as* *high* *as* *you* *can*."],
    "L08": ["The *CIA* and *Indian* *intelligence*", "chose *Nanda* *Devi*.", "Nearly *eight* *kilometers* *up*."],
    "L09": ["The *device*:", "an *antenna*, two *radios*,", "and *seven* *capsules* of *plutonium*.{amber}"],
    "L10": ["Up there, *batteries* *die*.", "*Plutonium*{amber} *doesn't*.", "Its heat becomes *electricity*."],
    "L11": ["In *October* *1965*,", "climbers *carried* *it* *up*.", "Near the top,", "a *blizzard* *hit*."],
    "L12": ["They tied it into", "a *crack* *in* *the* *rock*,", "and *turned* *back*."],
    "L13": ["In *spring*, they returned.", "It was *gone*,", "swept down the *mountain*."],
    "L14": ["They searched for *years*", "with *radiation* *detectors*.", "*Nothing*."],
    "L15": ["In *1978*,", "India's *prime* *minister*", "admitted it in *parliament*."],
    "L16": ["As far as anyone knows,", "the plutonium is *still* *up* *there*,", "in the ice that feeds", "*the* *Ganges*."],
    "L17": ["And *sixty* *years* later,", "it's *still* *warm*.{amber}"],
}


def make_timeline():
    SFX.reset()
    from scenes.ice import IceScene
    from scenes.mapscene import MapChina
    from scenes.paper import LineOfSight
    from scenes.ridge import NandaIntro, Climb, Slide
    from scenes.device import Device
    from scenes.closeup import Crevice, CreviceHUD, Search
    from scenes.record import Record1978
    from scenes.ganges import Ganges
    ice = IceScene(0.0, 7.95, "open")
    cv = Crevice(44.45, 50.25)
    scenes = [
        ice,
        MapChina(7.45, 17.95, start_pt=ice.core_screen(7.75)),
        LineOfSight(17.55, 23.75),
        NandaIntro(23.45, 29.45),
        Device(29.05, 39.15),
        Climb(38.95, 44.75),
        cv,
        CreviceHUD(cv),
        Slide(50.05, 51.95),
        Search(51.75, 55.75),
        Record1978(55.45, 60.35),
        Ganges(60.15, 66.05),
        IceScene(65.85, DURATION, "end"),
    ]
    caps = Captions(CAPTIONS)
    return scenes, caps, DURATION
