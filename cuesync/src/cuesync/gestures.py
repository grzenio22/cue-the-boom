"""What kind of noise (or tap) a cue is, and which effect it asks for on a given plate.

A gesture is the shape of the performance, not its words: a "hit" (a short "pf.", or a "KA-BOOM" that peaks
early and dies away), a "rhythm" of hits ("pf, pf, KA-BOOM"), a voiced "sustained" sound that swells or holds
("fwoooom"), a short unvoiced "breath" ("pfffff") or a long unvoiced "hiss" ("hissss"). Voiced or not is read from pitch: the share of the sound's 40 ms frames
that have a clear period in the voice range (a "BOOM" or a "wooo" has one; a hiss or a breath has none).
A sample plate maps gestures to effects (the candle: a hit blows it out, a sustained sound flares it); any
other plate uses the defaults. Samples and live takes go through the same rules.

A plate with no map of its own (your own clip of a hand, a prop) goes by what the sound is instead: a boom, a
whoosh, a hiss or a puff, with whether a flame already burns there (see ``heard`` and ``effect_for_kind``).
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


OBJECTS = {  # your own clip: say what it is, and the noise does to it what it would do to a sample like it
    "candle": {"title": "a candle", "subject": "the candle",
               "scene": {"hit": "blow_out", "breath": "blow_out", "rhythm": "blow_out", "hiss": "blow_out",
                         "sustained": "wick_flare"}},
    "jack-o-lantern": {"title": "a jack-o'-lantern", "subject": "the jack-o'-lantern",
                       "scene": {"hit": "fire_burst", "rhythm": "fire_burst", "sustained": "fire_burst",
                                 "breath": "fire_burst", "hiss": "fire_burst"}},
    "mouth": {"title": "a toy's mouth", "subject": "the toy",
              "scene": {"hit": "fire_breath", "rhythm": "fire_breath", "sustained": "fire_breath",
                        "breath": "smoke_puff", "hiss": "steam_jet"}},
    "hot-drink": {"title": "a hot drink", "subject": "the hot drink",
                  "scene": {"hit": "steam_jet", "rhythm": "steam_jet", "sustained": "steam_jet", "breath": "steam_jet",
                            "hiss": "steam_jet"}},
    "hand": {"title": "your hand (fire from the fingertips)", "subject": "the outstretched hand"},  # by the sound's kind
    "flammable": {"title": "something to set alight", "subject": None,
                  "scene": {"hit": "fireball", "rhythm": "fireball", "sustained": "flame_plume", "breath": "smoke_puff",
                            "hiss": "steam_jet"}},
}


KIND_ELEMENT = {"boom": "fireball", "whoosh": "flame_plume", "hiss": "steam_jet", "puff": "smoke_puff"}  # the classic four


def heard(cue: CueSheet, samples: np.ndarray | None, sr: int = 48000, source: str = "voice") -> dict:
    """What the take sounds like, not its words: "boom" (voiced, a "KA-BOOM", a "pf, pf, KA-BOOM"), "whoosh" (a
    voiced sound that swells or holds, "fwoooOOOM"), "hiss" (long and unvoiced) or "puff" (short and unvoiced:
    "pf.", "pfffff", a blow), read from its gesture and whether it has a pitch; and, for the record, the kind
    cue_detect's acoustic heuristics give each hit. (Those alone missed too many real voices to pick with: on 19
    takes a long "BOOM" was a whoosh, every "pf" a breath it dropped, a hiss a crackle, "fwoooom" a boom.)"""
    from .cue_detect import detect_cues

    g = gesture(cue, samples, sr, source)
    detected: list[str] = []
    if source == "hold" or samples is None or not np.any(samples):
        return {"kind": None, "gesture": g, "voiced": None, "detected_kinds": detected}
    try:
        detected = list(detect_cues(np.asarray(samples, np.float32).reshape(-1), sr, len(cue.envelope), cue.fps,
                                    audio_offset_ms=-1000.0 * cue.offset_s)["kinds"])
    except ValueError:
        pass
    iv = cue.intervals()
    voiced = voiced_share(samples, sr, iv, cue.fps, cue.offset_s) >= VOICED_SHARE
    dur = sum(b - a for a, b in iv) / cue.fps
    kind = {"hiss": "hiss", "breath": "puff", "sustained": "whoosh"}.get(g)
    if kind is None:  # a hit or a rhythm: voiced, it is a boom; unvoiced, a puff (or, long, a hiss)
        kind = "boom" if voiced else "hiss" if g == "rhythm" and dur >= VOICED_LONG_S else "puff"
    return {"kind": kind, "gesture": g, "voiced": bool(voiced), "detected_kinds": detected}


def _upright(strokes: list | None, aspect: float = 1.0) -> bool:
    """Is a drawing mostly up and down (a plume), rather than across (a jet)? ``aspect``: the plate's width over
    its height (stroke points are fractions of each)."""
    dx = dy = 0.0
    for s in strokes or []:
        pts = np.asarray(s.get("points") or [], np.float64).reshape(-1, 2)
        if len(pts) > 1:
            d = np.abs(np.diff(pts, axis=0)).sum(0)
            dx, dy = dx + d[0] * aspect, dy + d[1]
    return dy > dx


def effect_for_kind(kind: str, scene: bool, lit: bool = False, strokes: list | None = None, aimed: bool = False,
                    rhythm: bool = False, aspect: float = 1.0) -> str:
    """The effect a sound asks for. A boom bursts into flame (aimed, a rhythm of them shoots out as separate jets:
    jab, jab, palm-strike); a whoosh is a jet of fire (aimed, or drawn across) or a plume (drawn upright, or a prop
    nobody aimed), and flares a flame that already burns; a hiss is steam; a puff blows a burning flame out, else
    it is a blast of air and smoke. ``lit``: a flame already burns at the source (scene.lit_at)."""
    if not scene:
        return KIND_ELEMENT.get(kind, "fireball")
    if kind == "puff":
        return "blow_out" if lit else "smoke_puff"
    if kind == "hiss":
        return "steam_jet"
    if kind == "whoosh":
        if lit:
            return "wick_flare"
        if strokes:
            return "flame_plume" if _upright(strokes, aspect) else "fire_breath"
        return "fire_breath" if aimed else "flame_plume"
    return "fire_breath" if aimed and rhythm and not strokes else "fire_burst"


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
