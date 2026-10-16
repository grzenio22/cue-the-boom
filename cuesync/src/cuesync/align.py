"""Place an element on the plate so its *visible* onset lands on the cue, optionally time-remapping its peak."""

from __future__ import annotations

from dataclasses import dataclass

from .score import ElementTiming


@dataclass(frozen=True)
class Alignment:
    """Plate frame p shows element frame round((p - offset - onset) * speed) + onset, for p >= offset + onset."""

    offset: int
    speed: float = 1.0


MIN_REMAP_SPAN = 6  # frames; below this a one-frame cue jitter would flip the speed between extremes


def plan_alignment(
    anchor_frame: int,
    timing: ElementTiming,
    *,
    plate_peak_frame: int | None = None,
    max_stretch: float = 0.25,
) -> Alignment:
    """Visible onset lands on the anchor. With ``plate_peak_frame`` (sustained effects only), the element is
    time-remapped so its peak approaches the loudness peak, when both spans are long enough to be meaningful."""
    offset = anchor_frame - timing.visible_onset
    speed = 1.0
    if plate_peak_frame is not None:
        plate_span = plate_peak_frame - anchor_frame
        element_span = timing.peak - timing.visible_onset
        if plate_span >= MIN_REMAP_SPAN and element_span >= MIN_REMAP_SPAN:
            speed = min(max(element_span / plate_span, 1.0 - max_stretch), 1.0 + max_stretch)
    return Alignment(offset=offset, speed=float(speed))


def element_index_for(alignment: Alignment, timing: ElementTiming, plate_frame: int, element_len: int) -> int | None:
    start = alignment.offset + timing.visible_onset
    if plate_frame < start:
        return None
    e = int(round((plate_frame - start) * alignment.speed)) + timing.visible_onset
    return e if 0 <= e < element_len else None
