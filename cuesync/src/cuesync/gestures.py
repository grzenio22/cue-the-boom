"""What kind of noise (or tap) a cue is, and which effect it asks for on a given plate.

A gesture is the shape of the performance, not its words: a "hit" (a short "pf.", or a "KA-BOOM" that peaks
early and dies away), a "rhythm" of hits ("pf, pf, KA-BOOM"), a voiced "sustained" sound that swells or holds
("fwoooom"), a short unvoiced "breath" ("pfffff") or a long unvoiced "hiss" ("hissss"). Voiced or not is read from pitch: the share of the sound's 40 ms frames
that have a clear period in the voice range (a "BOOM" or a "wooo" has one; a hiss or a breath has none).
A sample plate maps gestures to effects (the candle: a hit blows it out, a sustained sound flares it); any
other plate uses the defaults. Samples and live takes go through the same rules.
"""

from __future__ import annotations

import numpy as np

from .audio import CueSheet

GESTURES = ("hit", "rhythm", "sustained", "breath", "hiss")
DEFAULTS = {"hit": "fireball", "rhythm": "fireball", "sustained": "flame_plume", "breath": "smoke_puff",
            "hiss": "steam_jet"}
SHORT_S, VOICED_LONG_S, NOISY_LONG_S = 0.3, 0.45, 0.8
HOLD_LONG_S = 0.4  # a press longer than this is a hold for a flame, shorter a pop
VOICED_SHARE = 0.3  # a sound with fewer voiced frames than this is a hiss or a breath
FRAME_S = 0.04
PITCH_HZ = (70.0, 400.0)
PERIODICITY = 0.45  # normalised autocorrelation peak above which a frame has a pitch
EARLY_PEAK, FADED = 0.35, 0.85  # a boom peaks in its first 35 % and ends below 85 % of its peak


def _segments(samples: np.ndarray, sr: int, intervals: list[tuple[int, int]], fps: int, offset_s: float):
    """The recording's samples under each active interval (intervals are plate frames; sample 0 of the
    recording is at plate time ``offset_s``)."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    for a, b in intervals:
        lo = int(round((a / fps - offset_s) * sr))
        hi = int(round((b / fps - offset_s) * sr))
        lo, hi = max(0, lo), min(x.size, hi)
        if hi - lo > 1:
            yield x[lo:hi]


def voiced_share(samples: np.ndarray, sr: int, intervals: list[tuple[int, int]], fps: int, offset_s: float = 0.0) -> float:
    """Share of the active 40 ms frames (loud enough to count) that have a clear pitch."""
    n = int(FRAME_S * sr)
    lag_lo, lag_hi = int(sr / PITCH_HZ[1]), int(sr / PITCH_HZ[0])
    voiced = total = 0
    for seg in _segments(samples, sr, intervals, fps, offset_s):
        loud = np.sqrt(np.mean(seg.astype(np.float64) ** 2)) if seg.size else 0.0
        for i in range(0, seg.size - n + 1, n):
            f = seg[i:i + n].astype(np.float64)
            f = f - f.mean()
            e = float(np.dot(f, f))
            if e <= 1e-9 or np.sqrt(e / n) < 0.25 * loud:
                continue  # too quiet to judge (the tail of a sound)
            ac = np.correlate(f, f, mode="full")[n - 1:]
            peak = float(ac[lag_lo:min(lag_hi, n - 1)].max()) / e if lag_hi > lag_lo else 0.0
            total += 1
            voiced += peak > PERIODICITY
    return voiced / total if total else 0.0


def zero_crossing_rate(samples: np.ndarray, sr: int, intervals: list[tuple[int, int]], fps: int,
                       offset_s: float = 0.0) -> float:
    """Zero crossings per second over the active part of the take (for reference; gesture() uses pitch)."""
    crossings = n = 0
    for seg in _segments(samples, sr, intervals, fps, offset_s):
        crossings += int(np.count_nonzero(np.signbit(seg[1:]) != np.signbit(seg[:-1])))
        n += seg.size
    return crossings / max(1, n) * sr


def gesture(cue: CueSheet, samples: np.ndarray | None = None, sr: int = 48000, source: str = "voice") -> str:
    """The gesture of a cue. A hold has no timbre: by its presses' lengths (all long: sustained; several:
    a rhythm; one short: a hit)."""
    iv = cue.intervals()
    if source == "hold" or samples is None or not np.any(samples):
        lengths = [(b - a) / cue.fps for a, b in iv]
        if lengths and all(L >= HOLD_LONG_S for L in lengths):
            return "sustained"
        return "rhythm" if len(lengths) >= 2 else "hit"
    if len(cue.hits()) >= 2:
        return "rhythm"
    dur = sum(b - a for a, b in iv) / cue.fps
    if dur < SHORT_S:
        return "hit"
    if voiced_share(samples, sr, iv, cue.fps, cue.offset_s) < VOICED_SHARE:
        return "hiss" if dur >= NOISY_LONG_S else "breath"
    if dur < VOICED_LONG_S:
        return "hit"
    # a long voiced sound: a "KA-BOOM" peaks early and dies away (a hit); a "fwoooom" swells or holds
    env = np.asarray(cue.envelope, dtype=np.float64)[iv[0][0]: iv[-1][1]]
    if env.size >= 4:
        pk = int(np.argmax(env))
        if pk <= EARLY_PEAK * env.size and env[-1] < FADED * env[pk]:
            return "hit"
    return "sustained"


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
