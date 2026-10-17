"""What kind of noise (or tap) a cue is, and which effect it asks for on a given plate.

A gesture is the shape of the performance, not its words: a short "hit" ("pf.", "KA-BOOM"), a "rhythm" of
hits ("pf, pf, KA-BOOM"), a voiced "sustained" sound ("fwoooom"), a short noisy "breath" ("pfffff") or a long
noisy "hiss" ("hissss"). A sample plate maps gestures to effects (the candle: a hit blows it out, a sustained
sound flares it); any other plate uses the defaults. Samples and live takes go through the same rules.
"""

from __future__ import annotations

import numpy as np

from .audio import CueSheet

GESTURES = ("hit", "rhythm", "sustained", "breath", "hiss")
DEFAULTS = {"hit": "fireball", "rhythm": "fireball", "sustained": "flame_plume", "breath": "smoke_puff",
            "hiss": "steam_jet"}
NOISY_ZCR = 2500.0  # zero crossings per second above which a sound is a hiss or a breath, not a voice
SHORT_S, VOICED_LONG_S, NOISY_LONG_S = 0.3, 0.45, 0.8


def zero_crossing_rate(samples: np.ndarray, sr: int, intervals: list[tuple[int, int]], fps: int) -> float:
    """Zero crossings per second over the active part of the take."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    crossings, n = 0, 0
    for a, b in intervals:
        seg = x[int(a / fps * sr): int(b / fps * sr)]
        if seg.size > 1:
            crossings += int(np.count_nonzero(np.signbit(seg[1:]) != np.signbit(seg[:-1])))
            n += seg.size
    return crossings / max(1, n) * sr


def gesture(cue: CueSheet, samples: np.ndarray | None = None, sr: int = 48000, source: str = "voice") -> str:
    """The gesture of a cue. A hold has no timbre: short is a hit, long is sustained."""
    if len(cue.hits()) >= 2:
        return "rhythm"
    iv = cue.intervals()
    dur = sum(b - a for a, b in iv) / cue.fps
    if source == "hold" or samples is None or not np.any(samples):
        return "hit" if dur < 0.4 else "sustained"
    if dur < SHORT_S:
        return "hit"
    if zero_crossing_rate(samples, sr, iv, cue.fps) > NOISY_ZCR:
        return "hiss" if dur >= NOISY_LONG_S else "breath"
    return "sustained" if dur >= VOICED_LONG_S else "hit"


def effect_for(g: str, plate_meta: dict | None = None, voice_name: str | None = None) -> str:
    """The effect a gesture asks for: the plate's own map (by sample voice name, then by gesture), else the
    defaults (a sustained flame from a source that faces somewhere is a breath)."""
    pm = plate_meta or {}
    scene = pm.get("scene", {})
    for key in (voice_name, g):
        if key and key in scene:
            what = scene[key]
            return what if isinstance(what, str) else what["effect"]
    if g == "sustained" and pm.get("direction") is not None:
        return "fire_breath"
    return DEFAULTS.get(g, "fireball")
