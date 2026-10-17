"""In-scene guides: the effect painted into the shot itself, at its source, going where it should go.

The ainvfx-fluid IC-LoRA was trained on controls that paint the WHOLE frame in flat colour zones (six colours
picked from the clip), with the clip's real first frame as conditioning part of the time. So a scene guide is
the plate's own zones on every keyframe with the effect painted over them, and LTX-2.5 gets the real plate
frame as its first frame. The model then renders the effect in the scene: the fire lights the room and the
table, the pumpkin bursts and leaves debris, the candle's own flame flares and settles. The zones on every
keyframe also hold the camera: with the effect on black instead, the model drifts and dollies.

Every effect has a source (a mouth, a wick, the inside of a pumpkin, the coffee in a mug) and a direction.
Jets follow a bent path from the source, p(a) = s + L (u a + b a^2): straight out of the mouth, then curling
up as hot gas rises. A resting flame (a candle that is already lit) is painted before and after the effect,
so the flare grows out of it and settles back into it, and blowing it out leaves it out.

Or the performer draws where it goes: strokes on the plate are the effect's footprint at full voice, and the
voice decides when it appears, how much of the drawing it fills and how it ends (see ``Strokes``).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .audio import CueSheet, Hit
from .effects import EffectSpec
from .guide import FIRE_INNER, FIRE_OUTER, MIN_RADIUS_PX, SMOKE, STEAM, GuidePlan, _smoothed, _thinning  # noqa: F401

ZONES = 6
MARGIN_PX = 48  # a jet's tip stays this far inside the frame (it would read as cut off, not as long)
PREROLL = 17  # element frames before the first hit: the model sees the scene settled before it changes
MIN_LENGTH, MAX_LENGTH = 49, 121
WISP = (150, 150, 150)  # a thin smoke curl off a wick
ENERGY_FULL = 14.0  # performance energy of a full "KA-BOOM" (a full-loudness second is 24)
SMALL = 0.6  # a take whose biggest event is smaller than this gets the effect's "small" prompt
FIRE_COLOURS = (list(FIRE_OUTER), list(FIRE_INNER))


def scene_size(width: int, height: int) -> tuple[int, int]:
    """The render's size for a plate: LTX-2.5 needs multiples of 64 (1088x1920 for a 1080x1920 plate)."""
    return 64 * math.ceil(width / 64), 64 * math.ceil(height / 64)


# ---------------------------------------------------------------- zones

def zone_palette(frame: np.ndarray, k: int = ZONES) -> np.ndarray:
    """The k flat colours (Lab, float32) the plate is painted in: k-means on a small copy of the frame."""
    small = cv2.resize(frame, (max(16, frame.shape[1] // 8), max(16, frame.shape[0] // 8)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    k = max(1, min(k, len(np.unique(lab, axis=0))))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(7)  # the same plate always gets the same zones
    _, _, centers = cv2.kmeans(lab, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    return centers.astype(np.float32)


def paint_zones(frame: np.ndarray, palette: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """``frame`` painted over in the palette's flat colours at ``size`` (w, h), zones smoothed as in the
    LoRA's training controls (no texture, no single-pixel islands)."""
    w, h = size
    small = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    d = ((lab[:, :, None, :] - palette[None, None]) ** 2).sum(-1)
    labels = d.argmin(-1).astype(np.uint8)
    for _ in range(3):
        labels = cv2.medianBlur(labels, 7)
    rgb = cv2.cvtColor(palette.astype(np.uint8)[labels], cv2.COLOR_LAB2RGB)
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------- shapes

def unit(direction: float) -> tuple[float, float]:
    """Direction in degrees clockwise from straight up -> image-space unit vector (y points down)."""
    a = math.radians(direction)
    return math.sin(a), -math.cos(a)


def _poly(points, color) -> dict:
    return {"kind": "poly", "pts": [[round(x, 1), round(y, 1)] for x, y in points], "color": list(color)}


def _ellipse(cx, cy, rx, ry, color) -> dict:
    return {"kind": "ellipse", "cx": round(cx, 1), "cy": round(cy, 1), "rx": round(rx, 1), "ry": round(ry, 1),
            "color": list(color)}


GAS = {"smoke": SMOKE, "steam": STEAM, "wisp": WISP}
MIN_DENSITY = 0.2  # thinner than this is painted as nothing: "gone" must be unambiguous


def _gas(shape: dict, fluid: str, density: float) -> dict:
    """Mark a shape as smoke, steam or a wick's wisp of the given density (1 = solid). Its colour is resolved
    when it is drawn: a thin gas fades towards the scene under it, never towards black (a faded grey cloud
    in a bright kitchen is the kitchen's own grey, and the model keeps it)."""
    return dict(shape, color=list(GAS[fluid]), fluid=fluid, density=round(float(min(max(density, 0.0), 1.0)), 3))


def _path(s, u, length, bend, n=14):
    """Points along p(a) = s + L (u a + up * bend * a^2), a in [0, 1]."""
    return [(s[0] + length * u[0] * a, s[1] + length * (u[1] * a - bend * a * a)) for a in np.linspace(0, 1, n)]


def _tube(points, w0, w1, color, tip=0.06, wobble: int | None = None) -> dict:
    """A widening band along ``points`` (half-width w0 at the start, w1 at the end) with a short tip.
    ``wobble``: a keyframe number that ruffles both edges (+-15 %, differently each keyframe), so consecutive
    keyframes differ and the model animates the fire instead of holding a wedge."""
    left, right = [], []
    for i, (x, y) in enumerate(points):
        j0, j1 = max(0, i - 1), min(len(points) - 1, i + 1)
        dx, dy = points[j1][0] - points[j0][0], points[j1][1] - points[j0][1]
        norm = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / norm, dx / norm
        hw = w0 + (w1 - w0) * i / (len(points) - 1)
        if wobble is not None and 0 < i:
            hw *= 1.0 + 0.15 * math.sin(2.3 * i + 1.7 * wobble) * min(1.0, 3 * i / len(points))
        left.append((x + nx * hw, y + ny * hw))
        right.append((x - nx * hw, y - ny * hw))
    (x1, y1), (x0, y0) = points[-1], points[-2]
    ext = math.hypot(x1 - x0, y1 - y0) * (len(points) - 1) * tip
    d = math.hypot(x1 - x0, y1 - y0) or 1.0
    tip_pt = (x1 + (x1 - x0) / d * ext, y1 + (y1 - y0) / d * ext)
    return _poly(left + [tip_pt] + right[::-1], color)


def _flame(base, u, height, width, color) -> dict:
    """A flame standing on ``base`` along ``u``: a round bottom narrowing to a point."""
    ux, uy = u
    px, py = -uy, ux  # perpendicular
    r = width / 2
    pts = []
    for i in range(36):
        t = 2 * math.pi * i / 36
        c, s = math.cos(t), math.sin(t)
        if s >= 0:  # bottom half: a half circle of radius r around base + u r
            along, across = r - r * s, r * c
        else:  # top half: narrows to the tip at distance `height`
            k = -s
            along, across = r + (height - r) * k, r * c * (1 - k) ** 0.6
        pts.append((base[0] + ux * along + px * across, base[1] + uy * along + py * across))
    return _poly(pts, color)


def _tongues(base, u, height, width, e: int, color) -> list[dict]:
    """A real flame is not one smooth wedge: a main tongue and two shorter side tongues, leaning a little and
    changing from keyframe to keyframe (a deterministic flicker), all rooted at ``base``."""
    px, py = -u[1], u[0]
    out = [_flame(base, u, height, width, color)]
    for side, phase in ((-1, 0.0), (1, 2.1)):
        k = 0.55 + 0.18 * math.sin(1.7 * e + phase)  # how tall this side tongue is now
        lean = 0.18 * side + 0.06 * math.sin(1.3 * e + phase)
        uu = (u[0] + px * lean, u[1] + py * lean)
        n = math.hypot(*uu) or 1.0
        b = (base[0] + px * side * width * 0.22, base[1] + py * side * width * 0.22)
        out.append(_flame(b, (uu[0] / n, uu[1] / n), height * k, width * 0.6, color))
    return out


def _tag(shapes: list[dict], **flags) -> list[dict]:
    return [dict(sh, **flags) for sh in shapes]


def _fire(shape_fn, inner: float = 0.55) -> list[dict]:
    """An outer orange volume and its yellow core (the model's fire palette)."""
    return _tag([shape_fn(1.0, FIRE_OUTER), shape_fn(inner, FIRE_INNER)], fire=True)


def draw(shapes: list[dict], base: np.ndarray) -> np.ndarray:
    """Paint ``shapes`` over ``base`` in order. A polygon may have holes (a drawn ring keeps its middle)."""
    img = Image.fromarray(base.copy())
    d = ImageDraw.Draw(img)
    for s in shapes:
        if s["kind"] == "poly" and s.get("holes"):
            m = Image.new("L", img.size, 0)
            md = ImageDraw.Draw(m)
            md.polygon([tuple(p) for p in s["pts"]], fill=255)
            for hole in s["holes"]:
                md.polygon([tuple(p) for p in hole], fill=0)
            img.paste(Image.new("RGB", img.size, tuple(s["color"])), (0, 0), m)
            d = ImageDraw.Draw(img)
        elif s["kind"] == "poly":
            d.polygon([tuple(p) for p in s["pts"]], fill=tuple(s["color"]))
        else:
            d.ellipse([s["cx"] - s["rx"], s["cy"] - s["ry"], s["cx"] + s["rx"], s["cy"] + s["ry"]], fill=tuple(s["color"]))
    return np.asarray(img, dtype=np.uint8)


def is_fire(shape: dict) -> bool:
    """Fire, tagged when painted (older plans: by the fire palette)."""
    return bool(shape.get("fire", shape["color"] in FIRE_COLOURS))


def is_rest(shape: dict) -> bool:
    """The resting flame that is there anyway (a lit candle), not the effect."""
    return bool(shape.get("rest", False))


# ---------------------------------------------------------------- drawn effects

@dataclass
class Strokes:
    """What the performer drew on the plate: brush strokes, each a polyline of normalised points and a
    brush width (fraction of the frame width). The first point of the first stroke is the source; the
    drawing as a whole is the effect's footprint at full voice."""

    strokes: list[dict]

    @classmethod
    def parse(cls, data) -> "Strokes | None":
        """From JSON (a list of {"points": [[x, y], ...], "width": w}); None for no usable stroke. Raises
        ValueError for anything that isn't strokes (wrong types, non-finite numbers)."""
        if not data:
            return None
        if not isinstance(data, list):
            raise ValueError("strokes must be a list")
        out = []
        for s in data:
            if not isinstance(s, dict) or not isinstance(s.get("points", []), list):
                raise ValueError("each stroke needs a list of points")
            pts = []
            for p in s.get("points", []):
                if not isinstance(p, (list, tuple)) or len(p) != 2:
                    raise ValueError("each point is [x, y]")
                x, y = float(p[0]), float(p[1])
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError("stroke points must be finite numbers")
                pts.append((min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)))
            width = float(s.get("width", 0.04))
            if not math.isfinite(width):
                raise ValueError("a stroke's width must be a finite number")
            if pts:
                out.append({"points": pts, "width": min(max(width, 0.005), 0.3)})
        return cls(out) if out else None

    @property
    def source(self) -> tuple[float, float]:
        return self.strokes[0]["points"][0]

    @property
    def direction(self) -> float:
        """From the source to the farthest drawn point, degrees clockwise from up (0 for a dab)."""
        sx, sy = self.source
        far = max((p for s in self.strokes for p in s["points"]), key=lambda p: (p[0] - sx) ** 2 + (p[1] - sy) ** 2)
        if (far[0] - sx) ** 2 + (far[1] - sy) ** 2 < 1e-6:
            return 0.0
        return math.degrees(math.atan2(far[0] - sx, -(far[1] - sy)))

    def to_json(self) -> list[dict]:
        return [{"points": [[round(x, 4), round(y, 4)] for x, y in s["points"]], "width": round(s["width"], 4)}
                for s in self.strokes]

    def mask(self, w: int, h: int) -> np.ndarray:
        """The drawing as a filled mask at w x h (round brush, round ends)."""
        img = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(img)
        for s in self.strokes:
            r = max(2.0, s["width"] * w / 2)
            pts = [(x * w, y * h) for x, y in s["points"]]
            if len(pts) > 1:
                d.line(pts, fill=255, width=int(round(2 * r)), joint="curve")
            for x, y in (pts[0], pts[-1]) if len(pts) > 1 else pts:
                d.ellipse([x - r, y - r, x + r, y + r], fill=255)
        return np.asarray(img) > 0

    def spine(self, w: int, h: int) -> np.ndarray:
        """The strokes' centre lines, thin (where a fire's yellow core goes)."""
        img = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(img)
        for s in self.strokes:
            pts = [(x * w, y * h) for x, y in s["points"]]
            width = max(2, int(round(s["width"] * w * 0.4)))
            if len(pts) > 1:
                d.line(pts, fill=255, width=width, joint="curve")
            else:
                x, y = pts[0]
                d.ellipse([x - width, y - width, x + width, y + width], fill=255)
        return np.asarray(img) > 0


def _mask_shapes(mask: np.ndarray, color, scale: int = 2) -> list[dict]:
    """A boolean mask as polygons (found at 1/``scale`` resolution), with their holes: a drawn ring stays a
    ring and does not cover what it surrounds."""
    small = cv2.resize(mask.astype(np.uint8), (max(1, mask.shape[1] // scale), max(1, mask.shape[0] // scale)),
                       interpolation=cv2.INTER_NEAREST)
    contours, hier = cv2.findContours(small, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]

    def pts(c):
        c = cv2.approxPolyDP(c, 1.0, True)
        return [(float(p[0][0]) * scale, float(p[0][1]) * scale) for p in c] if len(c) >= 3 else None

    out = []
    for i, c in enumerate(contours):
        if hier[i][3] != -1 or cv2.contourArea(c) < 4:  # holes are handled with their outer contour
            continue
        outer = pts(c)
        if outer is None:
            continue
        shape = _poly(outer, color)
        holes = []
        j = hier[i][2]  # first child: a hole
        while j != -1:
            if cv2.contourArea(contours[j]) >= 4:
                hole = pts(contours[j])
                if hole is not None:
                    holes.append([[round(x, 1), round(y, 1)] for x, y in hole])
            j = hier[j][0]
        if holes:
            shape["holes"] = holes
        out.append(shape)
    return out


class _Drawn:
    """Paints a drawn footprint, filled as far as the voice says: the fill grows out from the source (a disc
    of radius fill x reach over the drawing), follows the behaviour's timing, and ends the behaviour's way
    (fire recedes into the source; smoke and steam thin out where they are)."""

    def __init__(self, strokes: Strokes, w: int, h: int):
        self.mask = strokes.mask(w, h)
        self.spine = strokes.spine(w, h)
        sx, sy = strokes.source
        self.s = (sx * w, sy * h)
        yy, xx = np.mgrid[:h, :w]
        self.dist = np.hypot(xx - self.s[0], yy - self.s[1]).astype(np.float32)
        self.reach = float(self.dist[self.mask].max()) if self.mask.any() else 1.0

    def fill(self, c: "_Ctx", ev: "Event", e: int) -> tuple[float, float]:
        """(how much of the drawing is filled 0..1, how thinned 0..1) at keyframe e."""
        t = e - ev.at
        if t < 0:
            return 0.0, 0.0
        b = c.eff.behaviour
        fire = c.eff.fluid == "fire"
        if b in ("burst", "puff"):
            held = 8 if ev.end is None else min(max(ev.end - ev.at, 8), 24)
            amount = ev.size * (0.65 + 0.35 * _ease(t / 12))
            if t > held:
                k = (t - held) / 20
                return (amount * max(0.0, 1 - k), 0.0) if fire else (amount, min(1.0, 0.2 + 0.7 * k))
            return amount, 0.0
        # sustained: as far as the voice is loud, relative to its loudest moment
        peak = max(float(c.env[c.active].max()) if c.active.any() else 1.0, 1e-3)
        end = c.length if ev.end is None else ev.end

        def on(i: int) -> float:
            # the loudest moment within half a keyframe step: a "pf" between two keyframes still shows its size
            win = [j for j in range(max(ev.at, i - 4), min(end, i + 4)) if c.active[j]]
            level = max(float(c.env[j]) for j in win) / peak if win else 0.3
            return max(0.25, min(1.0, level)) * _ease((i - ev.at + 8) / 10) * ev.size

        if e < end:
            return on(e), 0.0
        since = e - end
        # carry on from the fill it had when the sound stopped (a small "pf" stays small, then recedes)
        last = max(on(i) for i in range(max(ev.at, end - 8), max(ev.at + 1, end)))
        if fire:
            return max(0.0, last * (1 - since / DIE_DOWN)), 0.0
        k = since / max(8, c.length - 1 - end)
        return (last, min(1.0, 0.2 + 0.8 * k)) if k < 0.85 else (0.0, 1.0)

    def shapes(self, c: "_Ctx", ev: "Event", e: int) -> list[dict]:
        amount, thin = self.fill(c, ev, e)
        out: list[dict] = []
        t = e - ev.at
        if c.eff.behaviour == "burst" and c.eff.secondary == "smoke" and t >= 10:
            # a drawn burst leaves its smoke where the drawing was, thinning out
            held = 8 if ev.end is None else min(max(ev.end - ev.at, 8), 24)
            k = max(0.0, (t - held) / max(1, c.length - ev.at - held))
            smoke = self.mask & (self.dist <= max(0.5, amount) * self.reach * 1.1 + 1.0)
            out += [_gas(sh, "smoke", 0.8 - 0.7 * k) for sh in _mask_shapes(smoke, SMOKE)]
        end = c.length if ev.end is None else ev.end
        if c.eff.fluid == "fire" and c.eff.behaviour not in ("burst", "puff") and 0 <= e - end < 30:
            # a drawn flame that went out leaves a wisp of smoke rising from where it was
            k = (e - end) / 30
            ys, xs = np.nonzero(self.mask)
            if ys.size:
                top, cx_ = float(ys.min()), float(xs.mean())
                w_ = max(float(xs.max() - xs.min()) * 0.25, MIN_RADIUS_PX)
                out.append(_gas(_ellipse(cx_, top - w_ - 0.15 * c.h * k, w_, w_ * 1.8, SMOKE), "smoke", 0.6 - 0.45 * k))
        if amount <= 0.02:
            return out + _rest(c)  # a lit candle burns before the effect and again once it is gone
        m = self.mask & (self.dist <= amount * self.reach + 1.0)
        if not m.any():
            return out + _rest(c)
        if c.eff.fluid == "fire":
            return out + _tag(_mask_shapes(m, FIRE_OUTER) + _mask_shapes(m & self.spine, FIRE_INNER), fire=True)
        fluid = "smoke" if c.eff.fluid == "smoke" else "steam"
        return out + [_gas(sh, fluid, 1.0 - thin) for sh in _mask_shapes(m, GAS[fluid])] + _rest(c)


# ---------------------------------------------------------------- events

@dataclass
class Event:
    """One thing the effect does: a hit (a burst, a puff, a blow-out, a short breath) or the whole sustained
    performance (a long breath, a flare, a hiss). Frames are element frames (plate frame - start)."""

    at: int  # element frame of the hit / the start of the performance
    end: int | None  # element frame where it stops being painted as active (None: it outlasts the element)
    peak: float  # loudness of the attack (0..1)
    size: float  # 0.3..1: a rhythm's hits relative to its biggest; a single burst by how much went into it
    energy: float  # how much performance went into it (a full-loudness second is 24)


@dataclass
class _Ctx:
    eff: EffectSpec
    s: tuple[float, float]  # source, px
    u: tuple[float, float]  # direction, unit
    w: int
    h: int
    scale: float
    env: np.ndarray  # smoothed absolute envelope per element frame
    active: np.ndarray  # performing per element frame
    length: int
    notes: set = field(default_factory=set)  # things the painting had to change (told to the performer)

    @property
    def side(self) -> float:
        return min(self.w, self.h) * self.scale

    def reach(self, u=None) -> float:
        """How far the frame edge (minus the margin) is from the source along ``u``."""
        ux, uy = u or self.u
        lim = []
        if ux > 1e-6:
            lim.append((self.w - MARGIN_PX - self.s[0]) / ux)
        if ux < -1e-6:
            lim.append((MARGIN_PX - self.s[0]) / ux)
        if uy > 1e-6:
            lim.append((self.h - MARGIN_PX - self.s[1]) / uy)
        if uy < -1e-6:
            lim.append((MARGIN_PX - self.s[1]) / uy)
        return max(0.0, min(lim)) if lim else float(max(self.w, self.h))


def _ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def _rest(c: _Ctx) -> list[dict]:
    """The flame that is already there (a lit candle), at its resting size."""
    r = c.eff.scene.resting * min(c.w, c.h)
    if r <= 0:
        return []
    height = max(r, MIN_RADIUS_PX * 1.2)
    return [dict(_flame(c.s, c.u, height, max(height * 0.42, MIN_RADIUS_PX * 0.9), FIRE_OUTER), fire=True, rest=True)]


DIE_DOWN = 6  # frames for a flame to go once the voice stops (the model follows the guide, then the re-time
#                moves it later by the model's lead; a longer die-down read as a flame that outlives the voice)


def _blast(cx: float, cy: float, r: float, e: int, col) -> dict:
    """A ragged burst of flame: an irregular outline that licks upward and changes every keyframe (a painted
    disc renders as a round, cotton-like ball)."""
    pts = []
    for i in range(28):
        a = 2 * math.pi * i / 28
        n = 0.18 * math.sin(3 * a + 0.9 * e) + 0.12 * math.sin(5 * a - 1.3 * e + 1.0) + 0.08 * math.sin(9 * a + 2.1 * e)
        k = 0.82 + n
        dx, dy = math.sin(a), -math.cos(a)  # clockwise from up
        up = 1.25 if dy < 0 else 0.9  # flame climbs, the base stays low
        pts.append((cx + dx * r * k, cy + dy * r * k * up))
    return _poly(pts, col)


def _burst(c: _Ctx, ev: Event, e: int) -> list[dict]:
    """A fireball at the source: it pops on the hit, holds while the voice lasts (bounded), collapses into
    rising smoke that stays attached to it at first. A resting flame (the candle in a jack-o'-lantern) burns
    before it and again once the fireball is gone."""
    sc = c.eff.scene
    t = e - ev.at
    if t < 0:
        return _rest(c)
    rpk = max((sc.r_min + (sc.r_max - sc.r_min) * ev.peak) * c.side * ev.size, MIN_RADIUS_PX * 1.5)
    held = 8 if ev.end is None else min(max(ev.end - ev.at, 8), 24)
    decay = 20
    if t <= held:
        r = rpk * (0.62 + 0.38 * _ease(t / 12))
    else:
        r = rpk * max(0.0, 1.0 - (t - held) / decay)
    cx, cy = c.s[0] + c.u[0] * 0.45 * rpk, c.s[1] + c.u[1] * 0.45 * rpk - 0.25 * rpk * min(t, held + decay) / 24
    shapes: list[dict] = []
    if c.eff.secondary == "smoke" and t >= 10:
        age = (t - 10) / max(1, c.length - ev.at - 10)
        sr = rpk * (0.75 + 0.55 * age) * (0.8 + 0.4 * min(ev.energy / 24.0, 1.0)) * (0.6 + 0.4 * ev.size)
        sy = cy - 0.6 * rpk - 0.5 * c.h * age * 0.5  # rises, and may leave the top of the frame
        density = 0.85 - 0.75 * age  # thins as it rises; under MIN_DENSITY it is gone
        for j, (ox, oy, k) in enumerate(((0.0, 0.0, 1.0), (-0.45, 0.2, 0.62), (0.42, 0.12, 0.58), (0.1, -0.35, 0.5))):
            jx = ox + 0.06 * math.sin(1.1 * e + 2.0 * j)
            jy = oy + 0.05 * math.cos(0.9 * e + 1.3 * j)
            shapes.append(_gas(_ellipse(cx + jx * sr, sy + jy * sr, sr * k, sr * k * 0.85, SMOKE), "smoke", density))
    if r >= MIN_RADIUS_PX:
        shapes += _fire(lambda k, col: _blast(cx, cy, r * k, e + int(7 * k), col))
    else:
        shapes += _rest(c)
    return shapes


def _plume(c: _Ctx, ev: Event, e: int) -> list[dict]:
    """A flame standing on the source, as tall as the voice is loud, in irregular tongues; when the voice
    stops it settles back to the resting flame (a candle) or dies into a smoke wisp."""
    sc = c.eff.scene
    t = e - ev.at
    if t < 0:
        return _rest(c)
    span = sc.r_max - sc.r_min

    def height(i: int) -> float:
        level = c.env[i] if c.active[i] else 0.0
        return min((sc.r_min + span * level) * c.side * ev.size, c.reach() * 0.95)

    def flame(hgt: float) -> list[dict]:
        width = max(hgt * 0.42, MIN_RADIUS_PX)
        return _tag(_tongues(c.s, c.u, hgt, width, e, FIRE_OUTER)
                    + [_flame(c.s, c.u, hgt * 0.5, max(width * 0.5, MIN_RADIUS_PX * 0.8), FIRE_INNER)], fire=True)

    if ev.end is None or e < ev.end:
        hgt = max(height(e), MIN_RADIUS_PX * 2)
        grow = _ease((t + 8) / 12)
        return flame(MIN_RADIUS_PX * 2 + (hgt - MIN_RADIUS_PX * 2) * grow)
    since = e - ev.end
    last = max(height(i) for i in range(max(0, ev.end - 8), max(1, ev.end)))
    settle = DIE_DOWN
    if since < settle:
        hgt = last * (1 - since / settle) + (sc.resting * min(c.w, c.h)) * since / settle
        tip = (c.s[0] + c.u[0] * last * 0.9, c.s[1] + c.u[1] * last * 0.9)
        wisp = [_gas(_ellipse(tip[0], tip[1] - 40 - 120 * since / settle, last * 0.18, last * 0.3, SMOKE), "smoke",
                     0.7 - 0.45 * since / settle)]
        return wisp + (flame(hgt) if hgt >= MIN_RADIUS_PX * 1.5 else _rest(c))
    return _rest(c)


def _jet(c: _Ctx, ev: Event, e: int) -> list[dict]:
    """A jet out of the source along its direction, as long as the voice is loud, curling up as it goes
    (fire) or rising in thin wisps from across the source (steam). When the voice stops a fire jet lets go
    of the source and burns out into smoke; steam lifts off the source, thins and drifts up."""
    sc = c.eff.scene
    t = e - ev.at
    if t < 0:
        return _rest(c)
    span = sc.r_max - sc.r_min
    fire = c.eff.fluid == "fire"
    reach = c.reach()
    if reach < MIN_RADIUS_PX * 3:  # no room the way it is aimed: turn it up until it has some
        for turn in (30, 60, 90):
            a = math.radians(turn) * (-1 if c.u[0] > 0 else 1)
            u2 = (c.u[0] * math.cos(a) - c.u[1] * math.sin(a), c.u[0] * math.sin(a) + c.u[1] * math.cos(a))
            if c.reach(u2) >= MIN_RADIUS_PX * 3:
                c = dataclasses.replace(c, u=u2)
                reach = c.reach()
                c.notes.add("there was no room in the aimed direction, so it was turned upwards")
                break

    def length(i: int) -> float:
        level = c.env[i] if c.active[i] else 0.0
        return (sc.r_min + span * level) * c.side * ev.size

    def jet(L: float, start: float = 0.0, fade: float = 0.0) -> list[dict]:
        # ends well inside the frame (at the edge it reads as cut off), unless that would leave nothing
        L = min(L, 0.8 * reach) if 0.8 * reach >= MIN_RADIUS_PX * 1.5 else min(L, reach)
        if L < MIN_RADIUS_PX * 1.5:
            return []
        bend = sc.bend * abs(c.u[0])  # sideways jets curl up; an upward one just rises
        pts = _path(c.s, c.u, L, bend)
        pts = pts[int(round(start * (len(pts) - 2))):]
        w0 = max(0.04 * L, MIN_RADIUS_PX * 0.45) * (1 + 2.5 * start)
        w1 = max(0.32 * L, MIN_RADIUS_PX)
        if fire:
            core_n = max(3, int(len(pts) * 0.6))
            return _tag([_tube(pts, w0, w1, _thinning(FIRE_OUTER, fade) if fade else FIRE_OUTER, wobble=e),
                         _tube(pts[:core_n], w0 * 0.6, w1 * 0.45, _thinning(FIRE_INNER, fade) if fade else FIRE_INNER,
                               wobble=e + 5)], fire=True)
        return _steam(c, L, start, fade, e)

    if ev.end is None or e < ev.end:
        grow = _ease((t + 8) / 10)
        return jet(max(length(e), MIN_RADIUS_PX * 2) * grow)
    since = e - ev.end
    last = max(length(i) for i in range(max(0, ev.end - 8), max(1, ev.end)))
    if fire:
        out: list[dict] = []
        short = ev.end - ev.at <= 10  # a "pf": its next keyframe must be fire-free, or the breaths merge into one
        if since < DIE_DOWN and not short:  # the flame lets go of the mouth and burns out downstream
            out += jet(last, start=min(0.85, since / DIE_DOWN), fade=0.15 + 0.5 * since / DIE_DOWN)
        if since < 40:  # its smoke, where the flame was: a dark wisp rising and thinning (a round blob reads as cotton)
            k = since / 40
            mid = _path(c.s, c.u, min(last, 0.8 * reach), sc.bend * abs(c.u[0]))[9]
            rx = max(0.14 * last * (1 + 0.5 * k) * (0.5 + 0.5 * ev.size), MIN_RADIUS_PX)
            ry = rx * 2.2
            out.insert(0, _gas(_ellipse(mid[0], mid[1] - ry * 0.6 - 0.22 * c.h * k, rx, ry, SMOKE), "smoke",
                               0.55 - 0.45 * k))
        return out + _rest(c)
    tail = max(8, c.length - 1 - ev.end)
    k = since / tail
    if k >= 0.85:
        return _rest(c)
    # the source stops: what it made lifts off (a gap opens above the source) and thins
    return jet(last * (1.0 - 0.2 * k), start=min(0.6, 0.15 + 0.6 * k), fade=0.3 + 0.6 * k) + _rest(c)


def _steam(c: _Ctx, L: float, start: float, fade: float, e: int) -> list[dict]:
    """Thin wisps rising from across the source (a drink's surface), each swaying on its own, with gaps
    between them: ordinary steam, not a dry-ice column."""
    px, py = -c.u[1], c.u[0]
    half = max(MIN_RADIUS_PX, 0.07 * c.side)  # half the width of the steaming surface
    density = 0.8 * (1.0 - 0.85 * fade)  # never solid: steam is translucent
    out = []
    for j, off in enumerate((-0.6, 0.0, 0.6)):
        base = (c.s[0] + px * off * half, c.s[1] + py * off * half)
        Lj = L * (0.75 + 0.25 * math.cos(1.9 * j + 0.2 * e))  # wisps of different heights
        pts = _path(base, c.u, Lj, 0.0, n=12)
        for k, (x, y) in enumerate(pts):
            a = k / (len(pts) - 1)
            if k % 2 or a < start:
                continue
            rx = max((0.022 + 0.06 * a) * L, MIN_RADIUS_PX * 0.5)
            sway = 0.16 * L * a * math.sin(0.33 * e + 2.3 * j + 3.1 * a)
            out.append(_gas(_ellipse(x + px * sway, y + py * sway, rx, rx * 1.9, STEAM), "steam", density * (1 - 0.35 * a)))
    return out


def _puff(c: _Ctx, ev: Event, e: int) -> list[dict]:
    """A puff out of the source along its direction: pops, grows while the voice lasts, drifts and thins."""
    sc = c.eff.scene
    t = e - ev.at
    if t < 0:
        return _rest(c)
    rpk = max((sc.r_min + (sc.r_max - sc.r_min) * ev.peak) * c.side * ev.size, MIN_RADIUS_PX * 1.2)
    grow = 0.55 + 0.45 * _ease(t / 16) + 0.4 * min(ev.energy / 24.0, 1.0) * _ease((t - 16) / 48)
    r = rpk * grow
    d = min(r * 1.1 + 0.1 * c.h * t / 48, c.reach() - r)
    cx, cy = c.s[0] + c.u[0] * d, c.s[1] + c.u[1] * d - 0.05 * c.h * t / 48
    fluid = "smoke" if c.eff.fluid == "smoke" else "steam"
    k = max(0.0, (t - 24) / max(1, c.length - ev.at - 24))
    return [_gas(_ellipse(cx, cy, r, r * 0.85, GAS[fluid]), fluid, 0.9 - 0.9 * k)] + _rest(c)


def _extinguish(c: _Ctx, ev: Event, e: int) -> list[dict]:
    """Blowing out a flame: lit before the hit, out after it, a thin smoke curl rising off the wick that
    leans away from the breath and fades."""
    t = e - ev.at
    if t < 0:
        return _rest(c)
    k = t / max(1, min(60, c.length - ev.at))
    if k >= 1.0:
        return []
    lean = -c.u[0] if abs(c.u[0]) > 0.2 else 0.3  # the curl drifts away from where the breath came from
    shapes = []
    for j in range(3):
        a = (j + 1) / 3
        rise = (0.08 + 0.32 * a) * c.h * min(1.0, 0.25 + k)
        x = c.s[0] + lean * 0.12 * c.w * a * a * min(1.0, 2 * k) + 0.03 * c.w * math.sin(0.4 * e + 2 * j)
        r = MIN_RADIUS_PX * (0.9 + 1.4 * a * min(1.0, 0.5 + k))
        shapes.append(_gas(_ellipse(x, c.s[1] - rise, r, r * 1.5, WISP), "wisp", 0.85 - 0.8 * k))
    return shapes


_PAINTERS = {"burst": _burst, "plume": _plume, "jet": _jet, "puff": _puff, "extinguish": _extinguish}


# ---------------------------------------------------------------- planning

def single_size(energy: float) -> float:
    """How big a single burst or puff is, from how much went into it: a quick quiet "pf" (energy ~3) is
    about 0.45 of a full "KA-BOOM" (energy ~14 and more)."""
    return float(np.clip(math.sqrt(max(energy, 0.0) / ENERGY_FULL), 0.3, 1.0))


def events_for(cue: CueSheet, eff: EffectSpec, start: int, multi_hit: bool = True) -> list[Event]:
    """What the effect does on the element's timeline. A rhythmic take ("pf, pf, KA-BOOM") is one event per
    hit for every behaviour (short breaths, pops, flares), sized relative to its biggest hit; a single sound
    is one event, a burst or puff sized by how much performance went into it."""
    env = np.asarray(cue.envelope, dtype=np.float64)
    hits: list[Hit] = cue.hits() if multi_hit else []
    end = cue.voice_end_frame

    def peak_at(f: int) -> float:
        w = env[max(0, f): max(0, f) + 8]
        return float(w.max()) if w.size else 0.0

    if len(hits) >= 2:
        subs = [cue.for_hit(h) for h in hits]
        spans = [max(1, sum(b - a for a, b in s.intervals())) for s in subs]
        weight = [(h.strength / max(x.strength for x in hits)) ** 2 * (d / max(spans)) for h, d in zip(hits, spans)]
        out = []
        for h, s, wgt in zip(hits, subs, weight):
            e_end = s.voice_end_frame
            sub_env = np.asarray(s.envelope)
            act = [b - a for a, b in s.intervals()]
            out.append(Event(at=h.frame - start, end=None if e_end is None else e_end - start, peak=peak_at(h.frame),
                             size=float(np.clip(0.35 + 0.65 * wgt ** 0.5, 0.35, 1.0)),
                             energy=float(sub_env[max(0, h.start): max(0, h.end)].sum()) if sum(act) else 0.0))
        return out
    at = cue.frame_of(cue.hero_onset_s if eff.anchor == "hero" else cue.onsets_s[0])
    energy = float(sum(env[a:b].sum() for a, b in cue.intervals()))
    size = single_size(energy) if eff.behaviour in ("burst", "puff") else 1.0
    return [Event(at=at - start, end=None if end is None else end - start, peak=peak_at(at), size=size, energy=energy)]


def omitted_hits(cue: CueSheet, multi_hit: bool = True) -> list[int]:
    """Plate frames of hits the take has but a rhythm can't carry (more than CueSheet.hits keeps)."""
    if not multi_hit:
        return []
    kept = {h.frame for h in cue.hits()}
    return [h.frame for h in cue.hits(max_hits=99) if h.frame not in kept] if len(kept) >= 2 else []


def plan_window(cue: CueSheet, eff: EffectSpec, n_plate: int | None, multi_hit: bool = True) -> tuple[int, int]:
    """(start, length): the plate frame the render starts on and its length (8k+1, 49..121). The first event
    lands on a keyframe (start = its frame - PREROLL, PREROLL = 1 + 2*8) when the plate allows; ``n_plate``
    None means a still plate, which can start anywhere (it is held). Plans from the same events
    ``paint_scene_guide`` paints."""
    first = events_for(cue, eff, 0, multi_hit=multi_hit)[0].at
    start = first - PREROLL
    if n_plate is not None:
        start = max(0, start)
    end = cue.voice_end_frame
    needed = (MAX_LENGTH if end is None else end - start + eff.tail_frames + 8)
    length = int(min(max(8 * math.ceil((needed - 1) / 8) + 1, MIN_LENGTH), MAX_LENGTH))
    if n_plate is not None and start + length > n_plate:
        length = max(MIN_LENGTH, 8 * ((n_plate - start - 1) // 8) + 1)
        start = max(0, min(start, n_plate - length))
    return start, length


@dataclass
class ScenePlan:
    guide: GuidePlan
    start: int  # plate frame of element frame 0 (may be negative on a still plate: the plate is held)
    events: list[Event] = field(default_factory=list)
    direction: float = 0.0
    zones: np.ndarray | None = None  # the zones painted under every keyframe (first one)
    dropped: list[int] = field(default_factory=list)  # plate frames of hits the render window could not hold
    strokes: Strokes | None = None
    notes: list[str] = field(default_factory=list)  # what the painting had to change, for the performer

    @classmethod
    def from_json(cls, d: dict, frames: np.ndarray) -> "ScenePlan":
        """A plan back from its guide.json and guide video (re-timing a finished shot)."""
        return cls(guide=GuidePlan.from_json(d, frames), start=int(d["start"]),
                   events=[Event(**ev) for ev in d.get("events", [])], direction=float(d.get("direction", 0.0)),
                   dropped=[int(f) for f in d.get("dropped", [])], strokes=Strokes.parse(d.get("strokes")),
                   notes=list(d.get("notes", [])))

    def to_json(self) -> dict:
        return self.guide.to_json() | {"mode": "scene", "start": self.start, "direction": round(self.direction, 1),
                                       "events": [dataclasses.asdict(ev) for ev in self.events],
                                       "dropped": self.dropped, "notes": self.notes,
                                       "strokes": self.strokes.to_json() if self.strokes else None}


def paint_scene_guide(cue: CueSheet, eff: EffectSpec, point: tuple[float, float], plate_frames: np.ndarray, *,
                      start: int, length: int, direction: float | None = None, scale: float = 1.0,
                      size: tuple[int, int] | None = None, multi_hit: bool = True, hold: str = "pre",
                      strokes: Strokes | None = None) -> ScenePlan:
    """Paint the control video for an in-scene render. ``plate_frames`` are the plate's frames (one frame
    for a still plate); keyframe e is painted over the zones of plate frame start + e, all in the palette of
    the first, so a moving plate's zones move with it and the colours never flicker.

    Before the first hit every keyframe is held over its whole 8-frame latent (the LoRA saw holds of 1 to 8
    frames): a black latent is unconstrained, and the model fills it with the effect long before the hit.
    From the hit on, keyframes are single frames, which leaves the fire and smoke their own motion (held,
    the model copies the flat shapes). ``hold``: "pre" (that), "all" or "none". Every hit is active on at
    least one keyframe, wherever it falls between them, so a quick "pf" can't vanish.

    ``strokes``: the performer drew the effect; the drawing replaces the effect's own shape (its source and
    direction come from the first stroke) and the voice fills it.

    Smoke is painted relative to the scene's brightness: light grey reads as white cotton in a dark room."""
    if eff.scene is None:
        raise ValueError(f"effect {eff.id!r} has no scene section")
    w, h = size or scene_size(plate_frames.shape[2], plate_frames.shape[1])
    if strokes is not None:
        point, direction = strokes.source, strokes.direction
    direction = eff.scene.direction if direction is None else float(direction)
    from .guide import keyframe_indices

    if hold not in ("pre", "all", "none"):
        raise ValueError(f"hold must be 'pre', 'all' or 'none', got {hold!r}")
    keys = keyframe_indices(length, 8, with_one=True)
    events = events_for(cue, eff, start, multi_hit=multi_hit)
    dropped = [start + ev.at for ev in events if not 0 <= ev.at <= length - 2] + omitted_hits(cue, multi_hit)
    events = [ev for ev in events if 0 <= ev.at <= length - 2]
    notes: set = set()
    merged: list[Event] = []
    for ev in events:  # each hit is active on the first keyframe at or after it; two hits on one keyframe are one
        k_next = next(k for k in keys if k >= ev.at)
        if ev.end is not None and ev.end <= k_next:
            ev = dataclasses.replace(ev, end=k_next + 1)
        if merged and next(k for k in keys if k >= merged[-1].at) == k_next:
            m = merged[-1]
            merged[-1] = dataclasses.replace(m, end=None if m.end is None or ev.end is None else max(m.end, ev.end),
                                             peak=max(m.peak, ev.peak), size=max(m.size, ev.size),
                                             energy=m.energy + ev.energy)
            notes.add("two hits closer than the model can tell apart were made one")
            continue
        merged.append(ev)
    events = merged
    first_at = min((ev.at for ev in events), default=length)
    last_end = max(((ev.end if ev.end is not None else length) for ev in events), default=0) + eff.tail_frames // 2
    still = len(plate_frames) == 1
    env_full = _smoothed(np.asarray(cue.envelope, dtype=np.float64))
    env = np.zeros(length)
    active = np.zeros(length, dtype=bool)
    lo, hi = max(0, start), min(env_full.size, start + length)
    if hi > lo:
        env[lo - start: hi - start] = env_full[lo:hi]
    for a, b in cue.intervals():
        a2, b2 = max(a - start, 0), min(b - start, length)
        if b2 > a2:
            active[a2:b2] = True
    ctx = _Ctx(eff=eff, s=(float(point[0]) * w, float(point[1]) * h), u=unit(direction), w=w, h=h, scale=scale,
               env=env, active=active, length=length, notes=notes)
    palette = zone_palette(plate_frames[min(max(start, 0), len(plate_frames) - 1)])
    painter = _PAINTERS[eff.behaviour]
    # a drawing is where the effect goes; blowing a flame out has nowhere to go, the drawing only says where
    drawn = _Drawn(strokes, w, h) if strokes is not None and eff.behaviour != "extinguish" else None
    frames = np.zeros((length, h, w, 3), np.uint8)
    blobs, first_zones = [], None
    for e in keys:
        src = plate_frames[min(max(start + e, 0), len(plate_frames) - 1)]
        zones = paint_zones(src, palette, (w, h))
        if first_zones is None:
            first_zones = zones
            smoke_k = _smoke_level(zones)
        shapes: list[dict] = []
        for k, ev in enumerate(events):
            # a later event takes over the source: the earlier one's flame is not painted under it twice
            nxt = events[k + 1].at if k + 1 < len(events) else None
            if nxt is not None and e >= nxt and eff.behaviour in ("plume", "jet"):
                continue
            part = drawn.shapes(ctx, ev, e) if drawn is not None else painter(ctx, ev, e)
            shapes += [s for s in part if s not in shapes]
        if eff.behaviour == "extinguish" and events and e >= events[0].at:
            shapes = [s for s in shapes if not is_fire(s)]  # once blown out it stays out, whatever comes after
        # the resting flame goes under the effect (painted first), and only once
        rest = [s for s in shapes if is_rest(s)][:1]
        shapes = rest + [s for s in shapes if not is_rest(s)] if not any(is_fire(s) and not is_rest(s) for s in shapes) \
            else [s for s in shapes if not is_rest(s)]
        shapes = [sh for sh in shapes if sh.get("density", 1.0) >= MIN_DENSITY]
        frames[e] = draw([_resolve(sh, zones, smoke_k) for sh in shapes], zones)
        # held: the scene before the first hit, a quiet keyframe between two hits (no fire painted), so the model
        # can't fill the pause between "pf" and "pf" with fire, and the scene once the effect is over
        quiet_gap = any(a.at < e and e + 7 < b.at for a, b in zip(events, events[1:])) and all(
            is_rest(s) for s in shapes)  # truly quiet: only the flame that is there anyway (held, the model copies)
        over = not shapes and e > last_end
        if 0 < e < length - 1 and (hold == "all" or (hold == "pre" and (e + 7 < first_at or quiet_gap or over))):
            for f in range(e + 1, min(e + 8, length - 1)):
                if still:
                    frames[f] = frames[e]
                else:  # a moving plate keeps moving under a held keyframe
                    zf = paint_zones(plate_frames[min(max(start + f, 0), len(plate_frames) - 1)], palette, (w, h))
                    frames[f] = draw([_resolve(sh, zf, smoke_k) for sh in shapes], zf)
        blobs.append({"e": e, "shapes": shapes})
    voice_end = None if cue.voice_end_frame is None else cue.voice_end_frame - start
    guide = GuidePlan(frames=frames, anchor_frame=start, keyframes=keys, blobs=blobs, effect_id=eff.id, size=(w, h),
                      voice_end=voice_end if voice_end is not None and 0 <= voice_end < length else None,
                      tail="scene", canvas=True, point=(float(point[0]), float(point[1])), scale=float(scale))
    return ScenePlan(guide=guide, start=start, events=events, direction=direction, zones=first_zones, dropped=dropped,
                     strokes=strokes, notes=sorted(ctx.notes))


def _smoke_level(zones: np.ndarray) -> float:
    """How bright smoke is painted in this scene (1 = the palette's grey): about twice the scene's mean
    brightness, so a dark bedroom gets dim smoke and a bright kitchen the full grey."""
    mean = float(zones.reshape(-1, 3).mean())
    return float(np.clip((2.0 * mean + 30.0) / SMOKE[0], 0.2, 1.0))


def _under(shape: dict, zones: np.ndarray) -> np.ndarray:
    """The mean zone colour under a shape's bounding box."""
    h, w = zones.shape[:2]
    if shape["kind"] == "poly":
        xs = [p[0] for p in shape["pts"]]
        ys = [p[1] for p in shape["pts"]]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    else:
        x0, x1 = shape["cx"] - shape["rx"], shape["cx"] + shape["rx"]
        y0, y1 = shape["cy"] - shape["ry"], shape["cy"] + shape["ry"]
    x0, x1 = int(np.clip(x0, 0, w - 1)), int(np.clip(x1, 1, w))
    y0, y1 = int(np.clip(y0, 0, h - 1)), int(np.clip(y1, 1, h))
    patch = zones[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    return patch.reshape(-1, 3).mean(0) if patch.size else zones.reshape(-1, 3).mean(0)


def _resolve(shape: dict, zones: np.ndarray, smoke_k: float) -> dict:
    """A gas shape's colour: its fluid's colour (smoke dimmed to the scene's smoke level, steam never) blended
    over the zones under it by its density. Everything else is drawn as it is."""
    fluid = shape.get("fluid")
    if fluid is None:
        return shape
    base = np.asarray(GAS[fluid], np.float64) * (smoke_k if fluid != "steam" else 1.0)
    d = float(shape.get("density", 1.0))
    col = _under(shape, zones) * (1 - d) + base * d
    return dict(shape, color=[int(round(v)) for v in col])


def _ending(eff: EffectSpec, words: str) -> str:
    """How a sustained flame ends, for the performer's own words: without it the model keeps a thing that
    caught fire burning to the end of the shot, whatever the guide shows."""
    if eff.fluid != "fire" or eff.behaviour not in ("plume", "jet"):
        return ""
    if any(w in words.lower() for w in ("die", "dies", "out", "calm", "smoulder", "stops")):
        return ""  # they said how it ends
    if eff.scene and eff.scene.resting > 0:
        return ", then it calms back down to a small flame"
    return ", then the flames die down and go out, leaving a wisp of smoke"


def scene_prompt(eff: EffectSpec, subject: str | None, direction: float, hits: int = 1, size: float = 1.0,
                 describe: str | None = None) -> str:
    """The prompt for a scene render. ``describe``: the performer's own words for what happens (drawn
    effects, typed material); otherwise the effect's scene prompt with the subject, the direction, the rhythm
    (for ``hits`` >= 2) filled in, or its "small" prompt when the take's biggest event is small."""
    tail = "the light in the scene reacts to it, static camera, realistic phone video"
    if describe and describe.strip():
        words = " ".join(describe.strip().rstrip(".").split())[:200]
        return f"ainvfxfluid, {words}{_ending(eff, words)}, {tail}"
    ux, uy = unit(direction)
    towards = ("upwards" if uy < -0.7 else "downwards" if uy > 0.7
               else "to the left" if ux < 0 else "to the right")
    template = eff.scene.small if (eff.scene.small and hits < 2 and size < SMALL) else eff.scene.prompt
    rhythm = eff.scene.rhythm if hits >= 2 else eff.scene.single
    text = (template.replace("{subject}", (subject or "the object").strip()).replace("{towards}", towards)
            .replace("{rhythm}", rhythm))
    return " ".join(text.replace(" ,", ",").split())
