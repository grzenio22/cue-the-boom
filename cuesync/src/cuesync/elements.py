"""Elements made rather than rendered: lightning, water and air are drawn from the plate, the hand's track and its
matte (procedural, on the CPU), not rendered over black by LTX-2.5 like fire, smoke and steam.

Each element's module (cuesync.lightning, cuesync.water, cuesync.air) exposes one function,

    render(plate, track, mask, timeline, form, seed=0, progress=None) -> frames

taking the plate as uint8 RGB [T, H, W, 3] (as displayed, sRGB), the hand's track (hand_track.json's dict, its emitter
already carried to where the performer tapped: cuesync.bending.anchored), the hand's matte [T, H, W] uint8 (or None),
the voice's Timeline below and the element's form, and returning the shot's frames, uint8 RGB [T, H, W, 3] sRGB. It
works in linear light (cuesync.transfer.lin / gam) and may use cuesync.fx_light for the light the element throws on the
hand and the room. cuesync.bending.make_bending_shot then shakes the camera on hits, lays the Foley under it, writes
the shot and its receipt, as for a rendered element.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FORMS = {  # element -> its forms (the hand picks one: cuesync.bending.bending_effect)
    "lightning": ("arc", "ball"),  # from the fingertips; crackling in an open palm
    "water": ("orb", "whip"),  # gathered into a ball over the palm; a stream that follows the hand
    "air": ("ball", "blast"),  # a spinning ball over the palm; a push of air out of it
}


@dataclass
class Timeline:
    """The voice, frame by frame, as the element plays it (24 fps, the plate's frames)."""

    n: int  # frames in the shot
    energy: np.ndarray  # (n,) float in [0, 1]: the voice's loudness against the take's loudest (0 silent)
    hits: list[int]  # the frames the voice strikes on (each hit's onset), ascending
    sizes: list[float]  # per hit, how big it plays (cuesync.bending.hit_looks: about 0.45 for a "pf", 1 for the loudest)
    holds: list[tuple[int, int]] = field(default_factory=list)  # (first, last + 1) frames the voice holds a sound
    tap_frame: int = 0  # the frame the performer tapped the hand on
    aim: float | None = None  # degrees clockwise from up, the studio's arrow (None: the hand's own direction)
    camera: object | None = None  # cuesync.bending.camera_path's Camera (a handheld shot), or None: locked off

    def __post_init__(self) -> None:
        self.energy = np.clip(np.asarray(self.energy, np.float32).reshape(-1), 0.0, 1.0)
        if self.energy.size != self.n:
            raise ValueError(f"energy has {self.energy.size} frames for a {self.n}-frame shot")
        if len(self.sizes) != len(self.hits):
            raise ValueError("one size per hit")

    def held_at(self, t: int) -> bool:
        """Whether the voice holds a sound on frame ``t``."""
        return any(a <= t < b for a, b in self.holds)
