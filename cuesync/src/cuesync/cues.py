"""Where a shot's timing comes from: a cue sheet (onsets, the hit, the loudness envelope, when it is active).

- ``voice``: a performance recorded over the plate ("KA-BOOM", "fwoooom", "pf, pf, KA-BOOM").
- ``clip``: a sound that is already in the plate (a finger snap, a lighter's click, a door slam, a beat):
  the effect locks to it, and the shot keeps the clip's own sound.
- ``hold``: press-and-hold on the plate while it plays (silent): when you press is the hit, how long you hold
  is how long it lasts. With no cue at all, one 2-second hold from 1 s.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import CueSheet, NoOnsetError, analyze
from .media import read_audio

SOURCES = ("voice", "clip", "hold")
HOLD_LEVEL = 0.85  # a hold is a firm, steady performance
HOLD_ATTACK, HOLD_RELEASE = 2, 4  # frames
DEFAULT_HOLD = (1.0, 3.0)  # seconds


class CueError(ValueError):
    """The chosen cue can't time this shot (no sound in the clip, no hit in it, a bad hold)."""


@dataclass
class Cue:
    source: str
    cue: CueSheet
    samples: np.ndarray  # the shot's sound (the voice, the clip's own sound, or silence)
    offset_s: float  # plate time of samples[0]


def cue_from_holds(holds: list[tuple[float, float]], duration_s: float, fps: int = 24) -> CueSheet:
    """A cue sheet from press-and-hold intervals (seconds on the plate): each press is an onset, the
    envelope is firm while held, with a short attack and release. The first press is the hero."""
    clean = sorted((max(0.0, float(a)), float(b)) for a, b in holds if float(b) > float(a))
    if not clean:
        raise CueError("a hold needs at least one press with a release after it")
    n = max(int(np.ceil(max(duration_s, clean[-1][1] + 0.5) * fps)), 1)
    env = np.zeros(n)
    intervals = []
    for a, b in clean:
        fa, fb = int(round(a * fps)), max(int(round(b * fps)), int(round(a * fps)) + 1)
        intervals.append((fa, min(fb, n)))
        for f in range(fa, min(n, fb + HOLD_RELEASE)):
            rise = min(1.0, (f - fa + 1) / HOLD_ATTACK)
            fall = 1.0 if f < fb else max(0.0, 1.0 - (f - fb + 1) / HOLD_RELEASE)
            env[f] = max(env[f], HOLD_LEVEL * rise * fall)
    onsets = [a for a, _ in clean]
    peak = int(np.argmax(env))
    return CueSheet(fps=fps, offset_s=0.0, duration_s=float(n / fps), onsets_s=onsets, hero_onset_s=onsets[0],
                    envelope=env, envelope_rel=env / max(float(env.max()), 1e-9), peak_frame=peak,
                    extra={"source": "hold"}, active_intervals=intervals)


def load_cue(source: str, *, plate: Path, voice: Path | None = None, offset_s: float = 0.0,
             holds: list[tuple[float, float]] | None = None, duration_s: float = 10.0, fps: int = 24,
             sr: int = 48000) -> Cue:
    """The shot's cue from ``source`` (see the module docstring)."""
    if source not in SOURCES:
        raise CueError(f"cue must be one of {SOURCES}, got {source!r}")
    if source == "voice":
        if voice is None:
            raise CueError("a voice cue needs a recording")
        samples = read_audio(voice, sr)
        cue = analyze(samples, sr, offset_s=offset_s, fps=fps)
        return Cue("voice", _within(cue, duration_s, "your noise"), samples, offset_s)
    if source == "clip":
        try:
            samples = read_audio(plate, sr)
        except Exception as exc:
            raise CueError("your clip has no sound to lock to; perform it or press and hold instead") from exc
        samples = samples[: int(round(duration_s * sr))]  # only the part of the clip the shot uses
        if samples.size == 0 or float(np.abs(samples).max()) < 1e-4:
            raise CueError("your clip is silent; perform it or press and hold instead")
        try:
            cue = analyze(samples, sr, offset_s=0.0, fps=fps)
        except NoOnsetError as exc:
            raise CueError("no clear sound in your clip to lock to (a snap, a click, a slam)") from exc
        return Cue("clip", _within(cue, duration_s, "the sound to lock to"), samples, 0.0)
    clean = [(a, b) for a, b in (holds or [DEFAULT_HOLD]) if 0 <= a < duration_s]
    if not clean:
        raise CueError(f"every press is after the end of the shot ({duration_s:.1f} s)")
    cue = cue_from_holds([(a, min(b, duration_s)) for a, b in clean], duration_s, fps)
    return Cue("hold", cue, np.zeros(int(round(cue.duration_s * sr)), np.float32), 0.0)


def _within(cue: CueSheet, duration_s: float, what: str) -> CueSheet:
    """Refuse a cue whose hit falls outside the shot (a mistimed take would paint the effect off the clip)."""
    if not 0 <= cue.hero_onset_s < duration_s:
        where = "before the clip starts" if cue.hero_onset_s < 0 else f"after the end of the shot ({duration_s:.1f} s)"
        raise CueError(f"{what} falls {where}: make it while the clip plays")
    return cue
