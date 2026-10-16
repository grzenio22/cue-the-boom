"""Place an element on the plate.

Three modes, chosen per effect:
- "guide": element frame e lands on plate frame anchor+e, so a voice-shaped effect (a plume that grows with
  the rising "fwoooOOOM") stays in lockstep with the voice it was painted from.
- "hit": the element's perceived hit lands ``lead`` frames before the sound (light travels faster than
  sound, and a fireball that blooms after its "BOOM" reads late); the seed frames before it are hidden.
- "onset": the first visible frame lands on the anchor (optionally time-remapping the peak; legacy).
"""

from __future__ import annotations

from dataclasses import dataclass

from .score import ElementTiming


@dataclass(frozen=True)
class Alignment:
    """Plate frame p shows element frame first + round((p - offset - first) * speed), for p >= offset + first."""

    offset: int
    speed: float = 1.0
    first: int | None = None  # first element frame shown (None: the element's visible onset)


MIN_REMAP_SPAN = 6  # frames; below this a one-frame cue jitter would flip the speed between extremes
HIT_PREROLL = 2  # frames of the build-up shown before the hit


def plan_alignment(
    anchor_frame: int,
    timing: ElementTiming,
    *,
    mode: str = "onset",
    lead: int = 0,
    plate_peak_frame: int | None = None,
    max_stretch: float = 0.25,
) -> Alignment:
    if mode == "guide":
        return Alignment(offset=anchor_frame, first=0)
    if mode == "hit":
        hit = timing.hit_frame
        return Alignment(offset=anchor_frame - lead - hit, first=max(timing.visible_onset, hit - HIT_PREROLL))
    if mode != "onset":
        raise ValueError(f"unknown alignment mode {mode!r}")
    offset = anchor_frame - timing.visible_onset
    speed = 1.0
    if plate_peak_frame is not None:
        plate_span = plate_peak_frame - anchor_frame
        element_span = timing.peak - timing.visible_onset
        if plate_span >= MIN_REMAP_SPAN and element_span >= MIN_REMAP_SPAN:
            speed = min(max(element_span / plate_span, 1.0 - max_stretch), 1.0 + max_stretch)
    return Alignment(offset=offset, speed=float(speed))


def element_index_for(alignment: Alignment, timing: ElementTiming, plate_frame: int, element_len: int) -> int | None:
    first = timing.visible_onset if alignment.first is None else alignment.first
    start = alignment.offset + first
    if plate_frame < start:
        return None
    e = int(round((plate_frame - start) * alignment.speed)) + first
    return e if 0 <= e < element_len else None
