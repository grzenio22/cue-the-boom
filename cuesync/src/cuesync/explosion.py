"""An explosion thrown from a strike (cuesync.elements' contract). A fireball rendered ahead by LTX-2.5 + the
ainvfx-fluid IC-LoRA over black (a white-hot flash bursting into a rolling ball of flame that turns into thick black
smoke, rising and drifting) is set off ahead of the fist on the strike and from then on is fixed in the room, never
carried on the hand: the fist pulls back from it, the hand can open beneath its smoke. Everything that touches the
plate is made here on the CPU:

- the white-hot core on the strike frame, its glow, and the lens's warm veil over the whole picture
- the exposure flash (+0.45 stop near the blast) and the camera's knock and zoom punch
- the shock front: a ring racing out that bends the room behind it (red and blue a little apart) with lit dust on it
- sparks flung off the ball (blackbody-cooling comets with drag and gravity, some flying past the arm) and embers
  floating up in the smoke
- the fireball's light on the room, from its own image (its fire spread round it, tinted orange, flickering, handed
  over from the flash's and dying as it turns to soot), and on the arm a warm rim on the outline facing it, the skin
  keeping its hue, its face hardly lit (the ball is behind it)
- the ball rolling upward as it burns, its smoke climbing on out of the frame, thinning from its crown and drawing in
  from its underside
- its outline broken: the fire's thin bright limb let go and flame tongues drawn out of its rim, the smoke's outer
  third eaten away into crisp wisps that rise and curl with it
- the smoke's billows sharpened and lit from above, its soot filled in over frames as the fire dies (not in one)
- the hot air shimmering round the ball

The rendered fireball and its smoke are composited by cuesync.composite (fire as light, smoke as dense soot relit by
the room), its fire put on a camera's fire colours for its brightness (deep red, orange, yellow, white-hot), its dark
folds read as soot rather than holes, and the arm's matte in front of it all.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import cv2
import numpy as np

from . import composite as C
from . import fx_light as fx
from .transfer import gam, lin

FORMS = ("fist", "palm")  # a punch throws it ahead of the fist along the forearm; a thrust palm blooms it past the palm

SIZE = 2.0  # the fireball's radius at its fullest, hand sizes (a "pf" plays it at HIT_SIZE's smallest)...
MOST = 0.40  # ...and at most this share of the frame's short side
CLOUD_MOST = 0.75  # its smoke at its widest (swollen) reaches at most this share of the short side from the blast
# (a render's cloud grows to twice its ball and more: sized by its ball alone it buried the whole picture)
THROW = (0.8, 2.5, 0.12)  # hands ahead of the fist it goes off: the strike's peak speed (px a second) over the hand times
# the third, between the first two (a harder punch throws it further); without a measured strike, 1 hand
PALM_THROW = 0.6  # a thrust palm's blast goes off this far past the palm's middle
EDGE = 0.10  # it goes off at least this share of the frame inside its edges
EDGE_FADE = 0.08  # share of the render's short side over which its own frame edges fade out
BLAST_SHARE = 0.01  # the render's blast is its first frame lit over this share of its most lit frame's area
GROW_FRAMES, BURST_SPEED = 8, 2.0  # the render's first frames of growth play this much faster (a blast is quick)
SWELL = (0.15, 10.0)  # after its burst the ball swells by this share over about SWELL[1] frames
ROLL = (0.15, 6.0, 0.3, 12.0)  # and rolls upward: its middle rises this many of its radii, eased in over about this
# many frames, and it and its smoke climb on at this many radii a second, getting going over about this many frames,
# out of the top of the frame (the render's ball hung where it went off, a flat disc fading in place; a fireball is
# buoyant)
TOP_STRETCH = 0.2  # its top half grows this much taller than its bottom over SWELL[1] (a rising ball's crown rolls
# up and out, its underside hardly moves)
CLEAR = (40, 45)  # frames after the blast its smoke starts to clear, and how long it takes, thinnest first (a
# room-filling soot that never cleared held a flat dark sheet over the room to the end of the shot)

CORE = {0: 1.0, 1: 0.5, 2: 0.15}  # the white-hot core's strength on the strike frame and after
CORE_LIGHT = {0: 1.0, 1: 0.6, 2: 0.4, 3: 0.25, 4: 0.12, 5: 0.05}  # its light on the room, outlasting it while the
# ball grows
FLASH = {0: 0.45, 1: 0.25, 2: 0.1, 3: 0.03}  # exposure, stops: the flash (at +0.7 the whole picture washed out; an
# iris overshooting after it dipped the room's light for a frame)
LIGHT_HOLD = 0.75  # the room's light falls by at most this share of itself a frame (as the flash and the core handed
# over to the growing ball, the wall went +40, +12, +20 levels: a blink)
FLICKER = (0.12, 0.5)  # the fireball's light flickers by about this share, its noise smoothed over this many frames
# (8-12 Hz: a smooth ramp down is not how firelight behaves), as long as it is fire
FLASH_REACH, FLASH_FAR = 2.5, 0.35  # the flash falls off over this many fireball radii, to this share far from it
GROW_BLUR = {0: 0.3, 1: 0.22, 2: 0.12, 3: 0.07, 4: 0.04}  # the fireball's growth over the shutter, per frame
LIGHT_GAIN = 6.0  # the fireball lights the room from its own image: its fire's luminance spread round it
# (blast_light, LIGHT_FALLOFF), times this, on each surface's colour (fx.albedo): the wall within a radius of a ball at
# full burn lifts about +40 levels, red twice green, blue not at all, and the light dies with the fire (the room light
# of a point source a hand behind the ball lifted the wall +9 levels: the blast lit nothing)
LIGHT_FALLOFF = (0.8, 2.0)  # the spread is 1 / (1 + r / (this many radii of the ball)) ** this: a wall one radius
# past its edge takes a quarter of what lights its middle, and twice as far a third of that (spread 1/r, the far wall
# was lit nearly as much as the wall by the ball: a flat lift over the whole picture, not a pool of light round it)
LIGHT_V = (0.5, 0.7)  # its colour is a camera's fire colour (RAMP) for the fire's brightness, held between deep
# orange and orange (the render's own lemon-yellow lifted a khaki wall green; yellower, the wall went yellow-white)
LIGHT_GLOW = 0.03  # the same light scattered in the air and the lens, over everything
LIGHT_KNEE = 0.9  # linear: the lit room rolls off from here to white (a channel lit past it clipped to a flat plateau)
LIGHT_WARM = 0.5  # on a bright wall red soon reaches white and stops rising: the light's green then rises at most this
# much of its red, its blue of its green (at 0.8 a khaki wall by the ball went peach, not orange)
ARM_LIGHT = (0.05, 1.8)  # on the arm: a little on its face (the blast is behind it: the fist that hides the ball shows
# the camera its unlit front) and this much in a rim on its outline facing the fire (rim_light)
RIM_BAND = (4.0, 12.0)  # px at 1080: the rim starts this far inside the arm's outline (its soft edge is the room's)
# and dies away over this many more (a tenth of it left 30 px in)
RIM_FACE = (0.3, 0.8)  # the outline lit is the one facing the fire: its outward normal on the way to the fire over this
SKIN_GR = 0.75  # on skin the light keeps the skin's hue: green rises to at most this of red...
SKIN_BG = 0.15  # ...and blue at most this much of green's rise (lit skin's red is near white: green rising alone
# turned a fist lemon-yellow, a rubber glove)
SKIN_LIFT = (10.0, 40.0)  # display levels: the most the fire lifts the arm's face, and its rim (lit flat, a fist in
# front of the ball read cream-white)
KNOCK_PX, ZOOM = 14.0, 0.022  # the camera's knock (px at 1080) and its push from the pressure wave (share)
FW_SOFT = 6.0  # px at 1080: where the fire is, softened (a hard fire/soot boundary drew flat orange paint)
GRADE = 0.6  # how far the fire's colour goes onto a camera's fire colours for its brightness (a render's lemon-yellow
# body is no colour a flame has: over a khaki wall it read yellow-green)
DEFOCUS_PX = 1.2  # px at 1080: the blast is further away than the fist the phone focused on
SOOT_ALBEDO = 0.25  # its smoke is black: it takes this share of the room's light...
SOOT_LIT = 0.3  # ...the fireball's light among it, at this share of what lights the wall (lit by the room alone, the
# soot hugging the ball was black, a drop shadow; lit by it, it glows brown-orange round the fire, then dies with it)
SOOT_FILL = (0.16, 0.12)  # display levels (black taken off): the cloud's dark body is raised to the first inside its
# silhouette, its folds kept round it (raised to one flat level, the inside of the cloud lost every billow), and none
# of it left under the second (the composite reads that as dense: a fold under it let the wall through)
BILLOW = (0.5, 1.7, 0.6)  # the render's own light and shade on its billows, put back on the filled cloud: its
# smoke's luma over their median to this power, in [lo, hi] (a dim cloud filled to one grey read as a flat dark disc;
# clipped hard, its folds read as a camouflage print)
BILLOW_IN = 0.02  # share of the short side: kept this far inside the cloud's outline (its thin rim is the room's)
SOOT_CLOSE = 0.03  # share of the short side: gaps narrower than this inside the cloud are closed before it is filled
SOOT_RAMP = 1.0 / 8  # the filled silhouette changes by at most this a render frame (as the fire dies its body stops
# setting the cloud's threshold and gaps close: the soot snapped dense in one frame, +75% dark area)
DETAIL = (0.8, 6.0)  # the smoke's billows sharpened: its luma plus this much of what a blur of this many px at 1080
# takes off it (composited, the cloud read out of focus, a stain on the wall)
TOP_LIGHT = (0.5, 0.012)  # the room's light from above: a billow's upper faces lifted by up to this share (about
# 10-15 levels), its slopes read off the render's smoke blurred this share of the short side (lit evenly, the
# cloud's inside was flat camouflage blotches, not a volume)
# Its outline (Outline), in radii of the cloud: drawn as rendered, the fireball's rim was one continuous saturated
# orange line (it looked cut out) and its smoke's a broad, soft, round dark edge (it advertised the composite)
TONGUES = (3, 8, 0.25, 0.3)  # flame tongues: waves of this many to this many cycles a turn, run down each side from
# its top, drawn out past its edge up to this many of its radii where it faces up (a quarter of that below; at 8% all
# alike they read as a procedural fringe), streaked from its rim this much of their length in (and 12 px at 1080)
TONGUE_LEAN = 0.5  # they lean up by this much of their reach (flames rise)
TONGUE_FLICKER = (8.0, 12.0, 0.4)  # their lengths flicker at this many to this many Hz, by this share
TONGUE_GAPS = (0.65, 0.12)  # the fire's outer band (this deep, in its radii) loses this much of itself between the
# tongues (see-through)
FIRE_SHRINK = 5.0  # px at 1080: the fire's outermost edge drawn over by its colours this far in before its rim is
# broken (the render's thin bright limb stayed a continuous orange line down the ball's side; faded, a pale one)
EDGE_DESAT = 0.3  # and what it adds to the room this much of its saturation
OUTER = (0.3, 0.85, 0.7)  # the smoke's outer third (from its edge to this deep, in its radii) is eaten away, up to
# this share of it at its edge, the share rising to it as its depth to this power (feathered instead, the cloud read out
# of focus, a stain on the wall)...
WISPS = (0.035, 0.05, 0.03, 0.5)  # ...between its billows (the render's own light) and wisps of a noise this big across
# and up (shares of the short side) fixed in the cloud and rising with it, cut this sharply (a share of their range:
# edges 2-4 px across at 1080), the billows this share of what decides
LOBE = 0.02  # share of the short side: what is left thickens from its cut edge over this far, as a round lobe's path
# through it does (cut flat, it read as paper)
WISP_CURL = (0.04, 0.06, 40)  # the wisps curl: swirled up to this share of the short side, in eddies this big, over
# this many frames
TOP_THIN = (14, 50, 0.6)  # from this many frames after the blast to this many, the smoke thins from its crown, by this
# share at the top (it rises and thins from the top, not fading in place)...
BOTTOM_THIN = (6, 24, 0.75)  # ...and from its underside about twice as fast, eaten in from its edge (its lobes hung
# below it like ink drips)
OUTER_SOFT = 2.5  # px at 1080: what the smoke does to the room is softened this much in its outer third (the render
# draws a thin bright rim round its smoke: composited it drew a contour line round the cloud)
PARAMS = dataclasses.replace(  # composite.composite_frame's for a fireball rolling into black smoke (its light is ours)
    C.CompositeParams(), light_gain=0.0, fire_gain=0.75, shading_lo=0.5, shading_hi=1.35, interior_blur_frac=0.04,
    interior_lo=0.45, interior_hi=0.9, sat_lo=0.4, sat_hi=0.65)

# fire as a phone films it, by its brightness (display values): deep red at its dim edges, an orange body, yellow,
# then white-hot (a luminance-true blackbody read tan: a phone's tone curve crushes the blue a flame has)
RAMP_V = np.array([0.0, 0.15, 0.35, 0.55, 0.72, 0.86, 0.95, 1.0])
RAMP = np.array([[0, 0, 0], [92, 20, 5], [196, 62, 10], [242, 118, 18], [255, 165, 40], [255, 208, 92],
                 [255, 238, 170], [255, 250, 225]], np.float64) / 255.0

SHOCK_FRAMES = 12  # the front crosses the frame in half a second (shorter read as a cut, not a wave)
SHOCK_PX = 20.0  # px at 1080 of refraction at its strongest
SHOCK_WIDTH = 0.035  # share of the short side: the front's thickness at the start (it thickens as it travels)
SHOCK_DUST = 0.2  # opacity of the lit dust riding the front (it reads on a flat wall, where there is nothing to bend)
SHOCK_CA = 0.35  # red and blue bent this much more and less than green
SPARKS, EMBERS = 220, 45


# ------------------------------------------------------------------ where it goes off, and the render's own blast
def blast_point(track: dict, k: int, form: str = "fist") -> tuple[np.ndarray, np.ndarray, float]:
    """(point px, aim, hand size) on the strike frame ``k``. A fist: its leading landmark pushed THROW hands further
    along the forearm (wrist to middle knuckle); a palm: its middle pushed PALM_THROW. Without landmarks, the track's
    emitter and direction."""
    s = float(track["scale"][k])
    lm = (track.get("landmarks") or [None] * (k + 1))[k]
    if lm is None:
        p = np.asarray(track["xy"][k], np.float64)
        aim = np.asarray(track["dir"][k], np.float64)
        return p + (PALM_THROW if form == "palm" else 1.0) * s * aim, aim / max(np.hypot(*aim), 1e-6), s
    lm = np.asarray(lm, np.float64)[:, :2]
    aim = lm[9] - lm[0]
    aim /= max(float(np.hypot(*aim)), 1e-6)
    if form == "palm":
        return lm[[0, 5, 9, 13, 17]].mean(0) + PALM_THROW * s * aim, aim, s
    speed = max((float(e.get("peak_speed") or 0.0) for e in track.get("events", []) if abs(int(e["frame"]) - k) <= 4),
                default=0.0)
    throw = float(np.clip(THROW[2] * speed / max(s, 1.0), THROW[0], THROW[1])) if speed > 0 else 1.0
    return lm[int(np.argmax(lm @ aim))] + throw * s * aim, aim, s


def inside(p: np.ndarray, w: int, h: int) -> np.ndarray:
    """``p`` moved (as little as it takes) to EDGE of the frame inside its edges."""
    return np.array([np.clip(p[0], EDGE * w, (1 - EDGE) * w), np.clip(p[1], EDGE * h, (1 - EDGE) * h)])


def render_blast(element: np.ndarray, floor: int) -> tuple[int, np.ndarray, float]:
    """(its first lit frame, where the blast is on it, the ball's radius GROW_FRAMES later) in a render over black:
    measured, so any render of a fireball can play (its blast is not always where its guide put it)."""
    lit = element[:, ::4, ::4].max(-1).astype(np.int16) > floor + 24
    counts = lit.reshape(len(element), -1).sum(1)
    # (a spark held where the blast will be before it goes off is not the blast: a render can hold one for frames)
    on = np.nonzero(counts >= max(4, BLAST_SHARE * counts.max()))[0]
    if on.size == 0:
        raise ValueError("the explosion's render shows no blast")
    k = int(on[0])
    ys, xs = np.nonzero(lit[k])
    centre = np.array([4.0 * xs.mean(), 4.0 * ys.mean()])
    r8 = 4.0 * math.sqrt(float(counts[min(k + GROW_FRAMES, len(element) - 1)]) / math.pi)
    return k, centre, max(r8, 8.0)


def render_reach(element: np.ndarray, floor: int, centre: np.ndarray, start: int) -> float:
    """How far from ``centre`` the render's fire and smoke reach at their widest from frame ``start`` on (px; the
    98th percentile of its lit pixels' distances, so a stray wisp does not count)."""
    h, w = element.shape[1:3]
    yy, xx = np.mgrid[:h:4, :w:4]
    d = np.hypot(xx - centre[0], yy - centre[1])
    reach = 0.0
    for t in range(start, len(element), 4):
        lit = element[t, ::4, ::4].max(-1).astype(np.int16) > floor + 6
        if lit.sum() > 20:
            reach = max(reach, float(np.percentile(d[lit], 98)))
    return reach


def soot_shading_ref(element: np.ndarray, floor: int, start: int) -> float:
    """The median linear luma of the render's smoke (not its fire) from frame ``start`` on: its billows are lit and
    shadowed round this. (composite.smoke_shading_ref measures only smoke the render draws bright, and a black cloud
    is dim all through: its every billow sat at the bottom of the shading's range, a flat dark disc.)"""
    vals = []
    for f in element[start::max(1, (len(element) - start) // 12), ::4, ::4]:
        e = C._black_floor_f(f, floor)
        y = e @ C._LUMA
        m = (y > 0.04) & (C.fire_weight_map(e, PARAMS, f.shape[1], True) < 0.1)
        if m.sum() > 50:
            vals.append(C._lin(e[m]) @ C._LUMA)
    return float(np.median(np.concatenate(vals))) if vals else 0.25


def room_point(cam, k0: int, t: int, p) -> np.ndarray:
    """A point fixed in the room as seen on frame ``k0``, where it is on frame ``t`` (cam: bending.camera_path)."""
    from .bending import camera_between

    q = camera_between(cam, k0, t) @ np.array([p[0], p[1], 1.0])
    return q[:2] / q[2]


# ------------------------------------------------------------------ the shock front
def shock_ring(frame: np.ndarray, origin, aim, k: int, reach: float, side: int, seed: int,
               warm: np.ndarray, dust: np.ndarray) -> np.ndarray:
    """``frame`` (linear) seen through the blast's shock front ``k`` frames after it: a ring (a little stronger along
    the aim) that bends what is behind it outward ahead of the front and back behind it, red and blue a little apart,
    with lit dust riding it. ``warm``: the fireball's colour, which lights the ring early on."""
    h, w = frame.shape[:2]
    u = k / SHOCK_FRAMES
    R = 0.035 * side + reach * (1 - (1 - u) ** 1.6)
    strength = SHOCK_PX * side / 1080.0 * min(1.0, (k + 1) / 1.5) * (1 - u)
    width = SHOCK_WIDTH * side * (1 + 1.5 * u)
    ox, oy = float(origin[0]), float(origin[1])
    rng = np.random.default_rng([seed, 31])
    lobes = np.arange(3, 9)
    ph, amp = rng.uniform(0, 2 * np.pi, lobes.size), rng.uniform(0.5, 1.0, lobes.size) / lobes
    pad = 4 * width + strength + 4  # only the band round the front is touched
    y0, y1 = max(0, int(oy - R - pad)), min(h, int(oy + R + pad) + 1)
    x0, x1 = max(0, int(ox - R - pad)), min(w, int(ox + R + pad) + 1)
    if y1 <= y0 or x1 <= x0:
        return frame
    xs = np.arange(x0, x1, dtype=np.float32) - ox
    ys = np.arange(y0, y1, dtype=np.float32)[:, None] - oy
    dist = np.sqrt(xs * xs + ys * ys) + 1e-3
    nx, ny = xs / dist, ys / dist
    ang = np.arctan2(ys, xs)
    wob = 1 + 0.04 * (amp[:, None, None] * np.sin(lobes[:, None, None] * ang + ph[:, None, None])).sum(0)
    cone = 0.6 + 0.4 * np.clip(nx * aim[0] + ny * aim[1], 0, 1)
    z = (dist - R * wob) / width
    bend = (strength * cone * z * np.exp(0.5 * (1 - z * z))).astype(np.float32)
    out = frame.copy()
    box = (y0, y1, x0, x1)
    for c, more in enumerate((1 + SHOCK_CA, 1.0, 1 - SHOCK_CA)):
        shift = np.stack([-nx * bend * more, -ny * bend * more], -1).astype(np.float32)
        out[y0:y1, x0:x1, c] = fx._remap_box(np.ascontiguousarray(frame[..., c]), shift, box)
    edge = SHOCK_DUST * (1 - u) ** 1.3 * cone * np.exp(-0.5 * ((z - 0.8) / 0.7) ** 2) * dust[y0:y1, x0:x1]
    room = fx._lit_tone(frame)
    hot = min(1.0, 2.5 * (1 - u) ** 3)
    tone = np.maximum(room * (1 - hot) + warm * hot * 0.6, room)  # never darker than the lit room: air catching light
    out[y0:y1, x0:x1] += edge[..., None] * (tone - out[y0:y1, x0:x1])
    return out


# ------------------------------------------------------------------ sparks and embers
class Particles:
    """What the blast throws, in the room as it was on the strike frame (carried by the camera after): sparks flung off
    the fireball's surface as it grows (drag, gravity, cooling along the blackbody line, drawn as comets over the
    shutter: a bright head, a fading tail) and embers that float up on the hot gas, flicker and fall."""

    def __init__(self, origin, aim, s: float, R: float, side: int, seed: int, count: float = 1.0):
        rng = np.random.default_rng([seed, 77])
        ns, ne = int(round(SPARKS * count)), int(round(EMBERS * count))
        n = ns + ne
        self.ember = np.arange(n) >= ns
        a_aim = math.atan2(aim[1], aim[0])
        fwd = rng.random(n) < 0.55
        ang = np.where(fwd, a_aim + rng.normal(0, 0.8, n), rng.uniform(-np.pi, np.pi, n))
        dirs = np.stack([np.cos(ang), np.sin(ang)], -1)
        self.t0 = np.where(self.ember, rng.uniform(2, 12, n), rng.choice([0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0], n))
        r_then = R * np.clip(0.3 + 0.7 * self.t0 / 4.0, 0, 1)  # thrown off the ball's surface as it grows
        r0 = np.where(self.ember, R * np.sqrt(rng.uniform(0, 0.7, n)), r_then * rng.uniform(0.55, 0.95, n))
        self.p0 = np.asarray(origin, np.float64)[None] + dirs * r0[:, None]
        streak = (rng.random(n) < 0.08) & ~self.ember  # a few fast streakers
        speed = s * np.exp(rng.normal(math.log(0.5), 0.5, n))
        speed = np.where(streak, s * rng.uniform(1.4, 2.2, n), speed)
        speed = np.where(self.ember, s * rng.uniform(0.03, 0.12, n), speed)
        self.v0 = dirs * speed[:, None]
        self.v0[self.ember, 1] -= s * rng.uniform(0.08, 0.2, int(self.ember.sum()))  # carried up by the hot gas
        self.tau = np.where(self.ember, rng.uniform(12, 25, n), rng.uniform(4, 10, n))  # drag time constant, frames
        self.g = s * np.where(self.ember, rng.uniform(0.002, 0.006, n), rng.uniform(0.02, 0.04, n))
        self.life = np.where(self.ember, rng.uniform(30, 75, n),
                             np.where(streak, rng.uniform(10, 18, n), rng.uniform(5, 20, n)))
        self.T0 = np.where(self.ember, rng.uniform(1700, 2100, n), rng.uniform(2400, 3300, n))
        self.tauT = np.where(self.ember, 200.0, rng.uniform(4, 10, n))
        self.I0 = np.where(self.ember, rng.uniform(0.5, 1.2, n),
                           np.where(streak, rng.uniform(4, 7, n), rng.uniform(1.2, 3.5, n)))
        self.size = np.where(self.ember, rng.uniform(1.6, 2.8, n), np.where(streak, 2.0, rng.uniform(1.0, 1.8, n)))
        self.size *= side / 1080
        self.front = rng.random(n) < 0.3  # flying towards the camera: in front of the arm and the fireball
        self.phase = rng.uniform(0, 2 * np.pi, n)

    def at(self, i: int, ages: np.ndarray) -> np.ndarray:
        """Particle ``i``'s positions at ``ages`` frames after the strike (room coordinates on the strike frame)."""
        tau, g = self.tau[i], self.g[i]
        decay = tau * (1 - np.exp(-ages / tau))
        fall = g * (tau * ages - tau * decay)
        return self.p0[i][None] + self.v0[i][None] * decay[:, None] + np.stack([np.zeros_like(fall), fall], -1)

    def draw(self, since: float, shape, cam_M: np.ndarray, front: bool, shutter: float = 0.5) -> np.ndarray:
        """The light (linear, [h, w, 3]) of the particles on one side of the arm, ``since`` frames after the strike,
        the room carried by ``cam_M`` (3x3: the strike frame to this one)."""
        h, w = shape
        emit = np.zeros((h, w, 3), np.float32)
        K = 8
        for i in range(len(self.ember)):
            if self.front[i] != front:
                continue
            age = since - self.t0[i]
            if age < 0 or age > self.life[i]:
                continue
            P = self.at(i, np.linspace(max(0.0, age - shutter), age, K))
            P = (cam_M @ np.c_[P, np.ones(len(P))].T).T
            P = P[:, :2] / P[:, 2:3]
            length = float(np.hypot(*(P[-1] - P[0])))
            fade = min(1.0, (self.life[i] - age) / 5.0) * min(1.0, (age + 1) / 1.5)
            T = 1100 + (self.T0[i] - 1100) * math.exp(-age / self.tauT[i])
            I = self.I0[i] * ((T - 900) / (self.T0[i] - 900)) ** 3 * fade
            if self.ember[i]:
                I *= 0.65 + 0.35 * math.sin(self.phase[i] + 1.7 * age) * math.sin(0.6 * self.phase[i] + 2.9 * age)
            I *= min(1.0, 5.0 / max(length, 1.0)) ** 0.35  # its light spread along the streak (streaks stay bright)
            col = fx.kelvin_rgb(T) * I
            th = max(1, int(round(self.size[i])))
            for j, (a, b) in enumerate(zip(P[:-1], P[1:])):  # a comet: the head (now) bright, the tail fading
                kk = ((j + 1) / (K - 1)) ** 1.5
                cv2.line(emit, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b),
                         tuple(float(c) * kk for c in col), th, cv2.LINE_8)
        if emit.max() > 0:
            emit = cv2.GaussianBlur(emit, (0, 0), 0.7)
            emit = emit + 0.45 * cv2.GaussianBlur(emit, (0, 0), 2.5) + 0.2 * cv2.GaussianBlur(emit, (0, 0), 10.0)
        return emit


# ------------------------------------------------------------------ the render, made ready
def soot_shape(frame8: np.ndarray, floor: int, close: int, soft: float) -> np.ndarray:
    """(h, w) in [0, 1]: a render frame's cloud as its soot is filled: its body (not the render's faint haze round it,
    judged against its smoke, not its fire: as the fire died the cloud's threshold fell and its gaps closed at once),
    gaps narrower than ``close`` px closed, every gap open to the outside left out, the rest filled, its edge feathered
    ``soft`` px."""
    y = frame8.max(-1).astype(np.float32) - floor
    smoke = (y > 4) & (C.fire_weight_map(C._black_floor_f(frame8, floor), PARAMS, frame8.shape[1], True) < 0.1)
    some = y[smoke] if smoke.sum() > 100 else y[y > 4]
    body = float(np.percentile(some, 90)) if some.size > 100 else 0.0
    if body <= 0:
        return np.zeros(y.shape, np.float32)
    lit = (y > max(10.0, 0.3 * body)).astype(np.uint8)
    k = max(3, close | 1)
    closed = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ff = np.pad(closed, 1)  # a ring of outside all round, so the fill reaches every gap open to the outside
    mask = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 2)
    sil = cv2.erode((ff[1:-1, 1:-1] != 2).astype(np.float32), np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(sil, (0, 0), max(soft, 0.5))


class SootShapes:
    """The render's cloud as its soot is filled (soot_shape, at half the render's size), render frame by render frame
    from its blast, each frame's moving at most SOOT_RAMP towards what that frame measures: the fill grows in and lets
    go over frames. Asked for in order (a frame further on is reached through those before it)."""

    def __init__(self, el: np.ndarray, floor: int, start: int, scale: float, side: int):
        self.el, self.floor, self.e, self.sil = el, floor, start - 1, None
        self.close = int(SOOT_CLOSE * side / scale / 2)  # (sizes in the plate's px, at half the render's)
        self.soft = 0.006 * side / scale / 2

    def at(self, e: int) -> np.ndarray:
        while self.e < e:
            self.e += 1
            f = self.el[self.e]
            raw = soot_shape(cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_AREA),
                             self.floor, self.close, self.soft)
            self.sil = raw if self.sil is None else self.sil + np.clip(raw - self.sil, -SOOT_RAMP, SOOT_RAMP)
        return self.sil


def smoke_blur(a: np.ndarray, smoke: np.ndarray, sigma: float) -> np.ndarray:
    """``a`` blurred over ``sigma`` px from its smoke alone (``smoke`` in [0, 1] weighting each pixel): the fire's light
    kept out of the smoke round it (blurred with it, the soot by an ember read pale, a halo)."""
    return cv2.GaussianBlur(a * smoke, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(smoke, (0, 0), sigma), 0.05)


def fill_soot(E8: np.ndarray, floor: int, box, sil: np.ndarray, smoke: np.ndarray) -> np.ndarray:
    """The render with its cloud's dark body read as dense soot, not holes: over black the self-shadowed underside of a
    black cloud renders black, and composited it let the wall through as pale patches. Inside the cloud's filled
    silhouette ``sil`` (over ``box``; ``smoke``: where it is smoke, not fire) the render is raised by what its smoke
    round it lacks of SOOT_FILL's first level, so its folds and billows keep their light and shade round it, and to at
    least its second; by the fire, to what the fire lights round it."""
    y0, y1, x0, x1 = box
    part = E8[y0:y1, x0:x1].astype(np.float32)
    sigma = 0.02 * min(E8.shape[:2])
    level, least = (floor + v * (255 - floor) for v in SOOT_FILL)
    raised = part + np.maximum(level - smoke_blur(part.max(-1), smoke, sigma), 0.0)[..., None]
    lift = np.maximum(np.maximum(raised, 0.85 * cv2.GaussianBlur(part, (0, 0), sigma)), least)
    out = E8.copy()
    out[y0:y1, x0:x1] = np.clip(part + (np.maximum(lift, part) - part) * sil[..., None] + 0.5, 0, 255).astype(np.uint8)
    return out


def soot_detail(E8: np.ndarray, floor: int, box, smoke: np.ndarray) -> np.ndarray:
    """In place over ``box``: the render's smoke (``smoke`` in [0, 1], not its fire) with its billows sharpened,
    DETAIL: its luma plus a share of what a blur of its smoke takes off it, each pixel's colour kept."""
    y0, y1, x0, x1 = box
    part = E8[y0:y1, x0:x1].astype(np.float32) - floor
    y = part @ C._LUMA.astype(np.float32)
    sharp = DETAIL[0] * (y - smoke_blur(y, smoke, DETAIL[1] * min(E8.shape[:2]) / 1080))
    gain = np.clip(1.0 + smoke * sharp / np.maximum(y, 2.0), 0.5, 2.0)
    E8[y0:y1, x0:x1] = np.clip(floor + part * gain[..., None] + 0.5, 0, 255).astype(np.uint8)
    return E8


def soot_billows(E8: np.ndarray, floor: int, box, ref: float, smoke: np.ndarray) -> np.ndarray:
    """Over ``box``: the render's own shading of its smoke (before fill_soot) as a factor less one, round ``ref`` (its
    smoke's median linear luma), eased into BILLOW's range (clipped, its folds were flat plateaus: a camouflage print),
    and the upper faces of its billows lit from above (TOP_LIGHT); nothing where it is fire (``smoke`` in [0, 1])."""
    y0, y1, x0, x1 = box
    e = C._black_floor_f(E8[y0:y1, x0:x1], floor)
    yl = cv2.GaussianBlur((C._lin(e) @ C._LUMA).astype(np.float32), (0, 0), 1.0)
    s = BILLOW[2] * np.log(np.maximum(yl, 1e-5) / max(ref, 1e-4))
    lo, hi = math.log(BILLOW[0]), math.log(BILLOW[1])
    s = np.where(s > 0, hi * np.tanh(s / hi), lo * np.tanh(s / lo))
    sigma = TOP_LIGHT[1] * min(E8.shape[:2])
    height = smoke_blur(np.log(np.maximum(yl, 1e-4)), smoke, sigma)  # (a billow is a bright lump: its height)
    up = np.tanh(np.maximum(np.gradient(height, axis=0) * sigma, 0.0) / 0.3)  # (it rises towards the lump's middle)
    return ((np.exp(s) * (1.0 + TOP_LIGHT[0] * up) - 1.0) * smoke).astype(np.float32)


def radial_blur(E8: np.ndarray, centre, amount: float, steps: int = 7) -> np.ndarray:
    """The fireball smeared outward from ``centre`` over the shutter: averaged over scales 1 .. 1 + ``amount`` (its
    growth this frame, times the shutter). A blast that grows a third in a frame is not crisp on a phone."""
    if amount < 0.02:
        return E8
    h, w = E8.shape[:2]
    acc = np.zeros(E8.shape, np.float32)
    for k in np.linspace(1.0, 1.0 + amount, steps):
        M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), 0.0, float(1.0 / k))
        acc += cv2.warpAffine(E8, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32)
    return np.clip(acc / steps + 0.5, 0, 255).astype(np.uint8)


def fire_grade(Q: np.ndarray, E8: np.ndarray, floor: int, box) -> np.ndarray:
    """In place: the composite's fire pixels (where the render, ``E8``, is fire) put GRADE of the way onto a camera's
    fire colours for their brightness (RAMP), as much as the fire is what the pixel shows (its light's share of the
    pixel's): graded whole, the ball's thin edge, the wall behind showing through it, went a saturated orange outline.
    Only inside ``box``."""
    y0, y1, x0, x1 = box
    e = C._black_floor_f(E8[y0:y1, x0:x1], floor)
    fw = np.clip(C._blur(C.fire_weight_map(e, PARAMS, E8.shape[1], True), FW_SOFT * min(E8.shape[:2]) / 1080), 0, 1)
    q = np.clip(Q[y0:y1, x0:x1], 0, None)
    share = np.clip(PARAMS.fire_gain * (C._lin(e) @ C._LUMA) * fw / np.maximum(q @ C._LUMA, 1e-4), 0, 1)
    qd = q ** (1 / 2.2)
    v = np.clip(qd @ C._LUMA, 0, 1)
    target = np.stack([np.interp(v, RAMP_V, RAMP[:, c]) for c in range(3)], -1)
    Q[y0:y1, x0:x1] = np.clip(qd + (GRADE * fw * share)[..., None] * (target - qd), 0, None) ** 2.2
    return Q


def rim_light(m: np.ndarray, src) -> np.ndarray:
    """(h, w) in [0, 1]: how much of the fire behind the arm wraps round its outline: a thin band just inside the
    matte's edge (RIM_BAND), on the side facing the fire at ``src`` (its outward normal on the direction to it,
    RIM_FACE)."""
    h, w = m.shape
    q = 2
    mq = cv2.resize(m, (w // q, h // q), interpolation=cv2.INTER_AREA)
    px = q * 1080.0 / min(h, w)  # a half-size px, in px at 1080
    d = cv2.distanceTransform((mq > 0.5).astype(np.uint8), cv2.DIST_L2, 5) * np.float32(px)
    gy, gx = np.gradient(cv2.GaussianBlur(d, (0, 0), 1.5))
    nrm = np.hypot(gx, gy) + 1e-6
    nx, ny = -gx / nrm, -gy / nrm  # outward
    yy, xx = np.indices(d.shape).astype(np.float32)
    lx, ly = src[0] / q - xx, src[1] / q - yy
    face = (nx * lx + ny * ly) / (np.hypot(lx, ly) + 1e-6)
    band = C._smoothstep(RIM_BAND[0] - 2.0, RIM_BAND[0] + 1.0, d) \
        * np.exp(-np.maximum(d - RIM_BAND[0], 0.0) / RIM_BAND[1])
    rim = band * C._smoothstep(RIM_FACE[0], RIM_FACE[1], face) * (mq > 0.5)
    return cv2.resize(rim.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------------------ the fireball's light
def falloff(shape: tuple[int, int], r0: float, ball: float, power: float = 1.0) -> tuple[np.ndarray, float]:
    """The light's spread on a (h, w) grid: the rFFT of (1 + r / ``r0``) ** -``power`` over the grid doubled (so
    nothing wraps round the frame), and its sum over a disc of radius ``ball``: a ball that size of luminance 1
    lights its own middle 1."""
    h, w = shape
    dy = np.minimum(np.arange(2 * h), 2 * h - np.arange(2 * h)).astype(np.float64)[:, None]
    dx = np.minimum(np.arange(2 * w), 2 * w - np.arange(2 * w)).astype(np.float64)[None]
    r = np.hypot(dx, dy)
    k = (1.0 + r / max(r0, 0.5)) ** -power
    return np.fft.rfft2(k), float(k[r <= max(ball, 1.0)].sum())


def blast_light(y: np.ndarray, spread: np.ndarray, norm: float) -> np.ndarray:
    """(h, w): the light the glowing image ``y`` (linear luminance, on falloff's grid) throws round it."""
    h, w = y.shape
    lit = np.fft.irfft2(np.fft.rfft2(y.astype(np.float64), (2 * h, 2 * w)) * spread, (2 * h, 2 * w))[:h, :w]
    return np.maximum(lit / norm, 0.0).astype(np.float32)


def light_colour(fire_lin: np.ndarray) -> np.ndarray:
    """The colour (linear, luminance 1) of the light of ``fire_lin`` (the render's fire, linear): a camera's fire
    colour (RAMP) for its luminance-weighted brightness as the composite shows it, held to LIGHT_V. A hot ball lights
    the room yellow-orange, a cooling one deep orange."""
    y = fire_lin @ C._LUMA
    total = float(y.sum())
    v = LIGHT_V[1] if total <= 1e-9 else float(np.clip((float((y * y).sum()) / total * PARAMS.fire_gain) ** (1 / 2.2),
                                                       *LIGHT_V))
    c = np.array([np.interp(v, RAMP_V, RAMP[:, i]) for i in range(3)]) ** 2.2
    return (c / float(c @ C._LUMA)).astype(np.float32)


def warm_light(P: np.ndarray, lit: np.ndarray, m: np.ndarray, rim: np.ndarray) -> np.ndarray:
    """``lit`` (the room ``P`` with the fireball's light on it, linear) with what the light adds kept warm as the
    picture shows it (display values, after the shot's highlight shoulder, bending.SHOULDER). On the room: green rising
    at most LIGHT_WARM of red, blue of green. On the arm (its matte ``m``): the skin's hue kept (green up to SKIN_GR of
    red, blue rising SKIN_BG of green) and its lift held to SKIN_LIFT, its face's to the first, its ``rim``'s to the
    second."""
    from .bending import SHOULDER as S

    def shown(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, None)
        return np.where(x > S, S + (1.0 - S) * np.tanh((x - S) / (1.0 - S)), x) ** np.float32(1 / 2.2)

    g0 = shown(P)
    d = shown(lit) - g0
    red = np.maximum(d[..., 0], 0.0)
    green = np.minimum(d[..., 1], LIGHT_WARM * red)
    room = np.stack([d[..., 0], green, np.minimum(d[..., 2], LIGHT_WARM * np.maximum(green, 0.0))], -1)
    green = np.clip(SKIN_GR * (g0[..., 0] + red) - g0[..., 1], 0.0, np.maximum(d[..., 1], 0.0))
    skin = np.stack([red, green, np.clip(d[..., 2], 0.0, SKIN_BG * green)], -1)
    most = (SKIN_LIFT[0] + (SKIN_LIFT[1] - SKIN_LIFT[0]) * rim) / 255.0
    skin *= np.minimum(1.0, most / np.maximum(skin @ C._LUMA, 1e-6))[..., None]
    d = room + m[..., None] * (skin - room)
    y = np.clip(g0 + d, 0.0, 0.9999) ** np.float32(2.2)  # (and back from the shoulder)
    return np.where(y > S, S + (1.0 - S) * np.arctanh(np.clip((y - S) / (1.0 - S), 0.0, 0.9999)), y).astype(np.float32)


# ------------------------------------------------------------------ its outline
def _rank(a: np.ndarray) -> np.ndarray:
    """``a`` (float) as float32 uniform in [0, 1]: each value's rank."""
    rank = np.empty(a.size, np.float32)
    rank[np.argsort(a, axis=None)] = np.linspace(0.0, 1.0, a.size, dtype=np.float32)
    return rank.reshape(a.shape)


def _field(rng: np.random.Generator, h: int, w: int, across: float, up: float) -> np.ndarray:
    """(h, w) float32 uniform in [0, 1]: white noise blurred ``across`` and ``up`` px (by FFT, so it tiles)."""
    f = np.fft.rfft2(rng.standard_normal((h, w)))
    ky, kx = np.fft.fftfreq(h)[:, None], np.fft.rfftfreq(w)[None]
    return _rank(np.fft.irfft2(f * np.exp(-2 * np.pi ** 2 * ((kx * across) ** 2 + (ky * up) ** 2)), (h, w)))


class Outline:
    """The blast's outline broken up, frame by frame over the element's box: the fire's outermost edge let go and its
    rim pushed out by waves round it into flame tongues (in bursts, flickering, leaning up), the band between them
    thinned so they and it are partly see-through, and what it adds to the room less saturated; the smoke's outer third
    eaten away by crisp wisps fixed in the cloud (rising and curling with it), which breaks its round silhouette, the
    cloud thinning from its crown as it climbs and drawing in from its underside. Seeded per shot."""

    def __init__(self, h: int, w: int, side: int, seed: int):
        rng = np.random.default_rng([seed, 53])

        def waves(lo: int, hi: int, n: int, speed: tuple[float, float]) -> tuple[np.ndarray, ...]:
            # (cycles round, phase, curl outward, radians a frame)
            return (rng.integers(lo, hi + 1, n).astype(np.float32), rng.uniform(0, 2 * np.pi, n).astype(np.float32),
                    rng.uniform(-3.0, 3.0, n).astype(np.float32),
                    (rng.choice([-1.0, 1.0], n) * rng.uniform(*speed, n)).astype(np.float32))

        self.side = side
        self.tongues = waves(TONGUES[0], TONGUES[1], 10, (0.25, 0.8))
        self.bursts = waves(3, 8, 4, (0.05, 0.15))
        self.gaps = waves(TONGUES[0], 2 * TONGUES[1], 10, (0.3, 0.9))
        hz = 2 * np.pi / fx.FPS
        self.flicker = waves(2, 6, 4, (TONGUE_FLICKER[0] * hz, TONGUE_FLICKER[1] * hz))
        hq, wq = max(2, h // 4), max(2, w // 4)
        self.wisps = [_field(rng, hq, wq, WISPS[0] * side / 4, WISPS[1] * side / 4) for _ in range(2)]
        for i, fine in enumerate(_field(rng, hq, wq, WISPS[0] * side / 12, WISPS[1] * side / 12) for _ in range(2)):
            self.wisps[i] = _rank(0.65 * self.wisps[i] + 0.35 * fine)  # (and their ragged edges)
        psi = _field(rng, hq, wq, WISP_CURL[1] * side / 4, WISP_CURL[1] * side / 4)
        gy, gx = np.gradient(cv2.GaussianBlur(psi, (0, 0), 2.0))
        top = float(np.percentile(np.hypot(gx, gy), 99)) + 1e-9
        self.curl = (np.clip(gy / top, -1, 1).astype(np.float32), np.clip(-gx / top, -1, 1).astype(np.float32))

    @staticmethod
    def _wave(p: tuple[np.ndarray, ...], theta: np.ndarray, rho: np.ndarray, since: float) -> np.ndarray:
        """In (-1, 1): the waves ``p`` round the cloud (``theta``), curling out with ``rho`` (its radii), at ``since``."""
        m, ph, curl, speed = p
        acc = np.zeros(theta.shape, np.float32)
        for i in range(len(m)):
            acc += np.sin(m[i] * theta + ph[i] + curl[i] * rho + speed[i] * since)
        return np.tanh(acc * np.float32(math.sqrt(2.0 / len(m))))

    @staticmethod
    def _inside(part: np.ndarray, floor: int, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(element in [0, 1], its cloud's silhouette, signed px in from its edge), at ``size`` (w, h). The render's
        thin threads of haze round the cloud are not the cloud (kept as such they drew thin dark lines on the wall)."""
        e = C._black_floor_f(cv2.resize(part, size, interpolation=cv2.INTER_AREA), floor)
        inside = (C._smoothstep(0.03, 0.10, e.max(-1)) > 0.5).astype(np.uint8)
        inside = cv2.morphologyEx(inside, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        sd = cv2.distanceTransform(inside, cv2.DIST_L2, 3) - cv2.distanceTransform(1 - inside, cv2.DIST_L2, 3)
        return e, inside, sd

    def warp(self, E8: np.ndarray, floor: int, box, since: float, rise: float, also: list[np.ndarray]):
        """In place over ``box``: ``E8`` (uint8, plate size) with its outline broken, and each map in ``also`` (float,
        box size) moved with it. Returns what finish needs (quarter-size maps), or None for no cloud yet."""
        y0, y1, x0, x1 = box
        part = E8[y0:y1, x0:x1]
        hb, wb = part.shape[:2]
        hq, wq = max(2, hb // 4), max(2, wb // 4)
        e, inside, sd = self._inside(part, floor, (wq, hq))
        if inside.sum() < 12:
            return None
        rc = math.sqrt(float(inside.sum()) / math.pi)  # its radius, quarter px
        cy = float(np.nonzero(inside)[0].mean())  # its middle's row
        yy, xx = np.mgrid[:hq, :wq].astype(np.float32)
        fire = np.clip(C.fire_weight_map(e, PARAMS, max(1, E8.shape[1] * wq // wb), True), 0.0, 1.0)
        near = np.clip(2.0 * cv2.GaussianBlur(fire, (0, 0), max(1.0, 0.1 * rc)), 0.0, 1.0)  # (its dim rim too)
        up = lambda a: cv2.resize(a.astype(np.float32), (wb, hb), interpolation=cv2.INTER_LINEAR)  # noqa: E731
        X, Y = np.meshgrid(np.arange(wb, dtype=np.float32), np.arange(hb, dtype=np.float32))
        fx_, fy_ = wb / wq, hb / hq
        mx, my, band = X, Y, np.zeros((hq, wq), np.float32)
        rim = band
        moved = part.astype(np.float32)
        own = cv2.morphologyEx((fire > 0.4).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if own.sum() >= 12:  # its fire's own outline (a band of soot can lie round the ball, past it)...
            outside = 1.0 - C._smoothstep(0.15 * rc, 0.3 * rc, sd)  # ...where it is the cloud's (not round every
            # patch of fire inside the smoke: each drew a dark contour)
            rf = math.sqrt(float(own.sum()) / math.pi)
            fys, fxs = np.nonzero(own)
            ft = np.arctan2(yy - float(fys.mean()), xx - float(fxs.mean()))
            frho = np.hypot(yy - float(fys.mean()), xx - float(fxs.mean())) / rf
            sdf = cv2.distanceTransform(own, cv2.DIST_L2, 3) - cv2.distanceTransform(1 - own, cv2.DIST_L2, 3)
            gy, gx = np.gradient(cv2.GaussianBlur(own.astype(np.float32), (0, 0), max(1.0, 0.15 * rf)))
            n = np.hypot(gx, gy) + 1e-6
            limb = FIRE_SHRINK * self.side / 1080 / 4  # quarter px
            cover = limb * (1.0 - C._smoothstep(limb, 2.0 * limb, sdf)) * (sdf > -1) * outside  # (its outermost px
            # drawn from those just inside them)
            moved = cv2.remap(moved, X + up(gx / n * cover * fx_), Y + up(gy / n * cover * fy_), cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
            # flame tongues drawn out of its rim past its edge (the ball inside untouched): its colours just inside
            # the edge streaked out along its normal, leaning up, fading to their tips; travelling up its sides to
            # the top, in bursts with none between, their lengths flickering, longest where it faces up (flames rise)
            top = np.abs(np.angle(np.exp(1j * (ft + np.pi / 2)))).astype(np.float32)  # (0 at its top, pi at its foot)
            length = rf * TONGUES[2] * np.maximum(self._wave(self.tongues, top, frho, since), 0.0) * outside \
                * C._smoothstep(-0.4, 0.6, self._wave(self.bursts, ft, frho, since)) \
                * (1.0 + TONGUE_FLICKER[2] * self._wave(self.flicker, ft, frho, since)) \
                * (0.25 + 0.75 * np.clip(gy / n, 0.0, 1.0))
            out = np.maximum(-sdf, 0.0)  # quarter px past its edge
            reach = np.clip(1.0 - out / np.maximum(length, 1e-3), 0.0, 1.0) * (length > 0.5) * (sdf < 0)
            back = (out + TONGUES[3] * length + 3.0) * (reach > 0)  # (quarter px in from here: the rim it is drawn
            # from)
            mx = X + up(gx / n * back * fx_)
            my = Y + up((gy / n * back + TONGUE_LEAN * out * (reach > 0)) * fy_)  # (from below: up)
            drawn = cv2.remap(moved, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            moved = np.maximum(moved, floor + (drawn - floor) * up(C._smoothstep(0.0, 0.7, reach))[..., None])
            band = (1.0 - C._smoothstep(0.0, TONGUE_GAPS[1] * rf, sdf)) * (sdf >= 0) * outside  # its outer band...
            rim = band * (1.0 - C._smoothstep(-0.2, 0.6, self._wave(self.gaps, ft, frho, since)))  # (...its gaps)
        E8[y0:y1, x0:x1] = np.clip(floor + (moved - floor) * up(1.0 - TONGUE_GAPS[0] * rim)[..., None] + 0.5, 0,
                                   255).astype(np.uint8)
        for a in also:
            a[:] = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        e2, inside2, sd2 = self._inside(E8[y0:y1, x0:x1], floor, (wq, hq))
        # the wisps: fixed in the cloud as it rises (the second peeling off upward faster), curling as it lives
        swirl = np.float32(WISP_CURL[0] * self.side / 4 * C._smoothstep(0.0, WISP_CURL[2], since))
        wisp = np.zeros((hq, wq), np.float32)
        for field, share, faster in zip(self.wisps, (0.6, 0.4), (1.0, 1.5)):
            fxq = ((xx * (wb / wq) + x0) / 4).astype(np.float32)
            fyq = ((yy * (hb / hq) + y0 + faster * rise) / 4).astype(np.float32)
            cu = cv2.remap(self.curl[0], fxq, fyq, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            cv = cv2.remap(self.curl[1], fxq, fyq, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            wisp += share * cv2.remap(field, fxq + swirl * cu, fyq + swirl * cv, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_WRAP)
        wisp = np.clip(0.5 + 1.39 * (wisp - 0.5), 0.0, 1.0)  # (the two fields' mix spread back over [0, 1])
        lum = e2.max(-1)  # (its billows: the render's light on them, over that of its brightest tenth)
        lum = np.clip(lum / max(float(np.percentile(lum[inside2 > 0], 90)) if inside2.any() else 0.0, 1e-3), 0.0, 1.0)
        wisp = WISPS[3] * lum + (1.0 - WISPS[3]) * wisp
        outer = np.clip(1.0 - sd2 / (OUTER[0] * rc), 0.0, 1.0)  # (1 at its edge, 0 at the outer third's inner side)
        under = np.clip((yy - cy) / rc, 0, 1) * (1.0 - C._smoothstep(0.0, 0.6 * rc, sd2))  # (its underside, from out)
        eat = OUTER[1] * outer ** OUTER[2] + (1.0 - C._smoothstep(-0.15 * rc, 0.0, sd2)) \
            + BOTTOM_THIN[2] * C._smoothstep(BOTTOM_THIN[0], BOTTOM_THIN[1], since) * under  # (its haze past it goes)
        thin = 1.0 - TOP_THIN[2] * C._smoothstep(TOP_THIN[0], TOP_THIN[1], since) * np.clip((cy - yy) / rc, 0, 1)
        own_fire = np.clip(2.0 * cv2.GaussianBlur(fire, (0, 0), max(1.0, 0.04 * rc)), 0.0, 1.0)  # (the fire: its
        # soot round it is eaten too, or it hung by the ball as a blurred dark crescent, a drop shadow)
        return near, EDGE_DESAT * band, wisp, eat, thin, (1.0 - near) * outer, own_fire

    @staticmethod
    def density(maps, size: tuple[int, int], side: int) -> np.ndarray:
        """(h, w) at ``size`` (w, h): how much of the composite's soot each pixel keeps: its outer third cut sharply at
        full size, thickening in from each cut edge (LOBE), the fire whole."""
        _, _, wisp, eat, thin, _, own = maps
        up = lambda a, how=cv2.INTER_LINEAR: cv2.resize(a.astype(np.float32), size, interpolation=how)  # noqa: E731
        t = up(eat)
        kept = C._smoothstep(t - WISPS[2], t + WISPS[2], up(wisp, cv2.INTER_CUBIC))
        depth = cv2.distanceTransform((kept > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        kept *= np.sqrt(np.clip((depth + 1.0) / (LOBE * side), 0.0, 1.0)) * up(thin)
        return 1.0 - (1.0 - up(own)) * (1.0 - kept)

    def finish(self, Q: np.ndarray, P: np.ndarray, box, maps) -> None:
        """In place over ``box``: the composite ``Q`` (linear) with warp's maps put on it against the room ``P``: soot
        never lighter than the room behind it, a little softened in its outer third (the render draws a thin bright
        rim round its smoke: composited it drew a contour line round the cloud), its density where it is eaten and
        thinned, and on the fire's rim what the fire adds less saturated (the room behind it keeps its colour)."""
        near, desat, _, _, _, outer, _ = maps
        y0, y1, x0, x1 = box
        size = (x1 - x0, y1 - y0)
        up = lambda a: cv2.resize(a.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)[..., None]  # noqa: E731
        q, p = Q[y0:y1, x0:x1], P[y0:y1, x0:x1]
        base = np.maximum(p, 0.02)
        rel = (q - p - (1.0 - up(near)) * np.maximum(q - p, 0.0)) / base  # (what it does to the room, as a share
        # of the room: softened as a difference, the dark it laid on a white switch plate spread black onto the wall)
        soft = up(outer)
        rel = rel + soft * (cv2.GaussianBlur(rel, (0, 0), OUTER_SOFT * self.side / 1080) - rel)
        d = base * rel * self.density(maps, size, self.side)[..., None]
        d += up(desat) * ((d @ C._LUMA)[..., None] - d)
        Q[y0:y1, x0:x1] = np.maximum(p + d, 0.0)


def shoulder_added(P: np.ndarray, room: np.ndarray) -> np.ndarray:
    """``P`` (linear) with what the blast added to the ``room`` (the plate, linear) rolled off over the shot's
    highlight shoulder (bending.SHOULDER, its tanh), each channel; what it darkened and the room's own highlights as
    they were (rolled off again, the phone's highlights, the skin's red among them, dimmed for as long as the blast
    lasted)."""
    from .bending import SHOULDER as S

    def curve(x: np.ndarray) -> np.ndarray:
        return np.where(x > S, S + (1.0 - S) * np.tanh((x - S) / (1.0 - S)), x)

    return np.where(P > room, room + curve(P) - curve(room), P).astype(np.float32)


def library_element() -> np.ndarray:
    """The explosion rendered ahead (the made-ahead library's explosion/element.mp4)."""
    from .bending import library_dir
    from .media import read_video

    lib = library_dir()
    path = None if lib is None else Path(lib) / "explosion" / "element.mp4"
    if path is None or not path.exists():
        raise FileNotFoundError("no explosion in the made-ahead library (samples/library/bending/explosion)")
    return read_video(path).frames


# ------------------------------------------------------------------ the shot
def render(plate: np.ndarray, track: dict, mask: np.ndarray | None, timeline, form: str = "fist", seed: int = 0,
           progress=None, element: np.ndarray | None = None) -> np.ndarray:
    """The explosion on the plate (uint8 RGB [T, H, W, 3]), set off on the loudest hit (as big as it is loud),
    shaken and punched by its blast. ``element``: the fireball rendered over black (default: the library's)."""
    say = progress or (lambda stage, frac, msg: None)
    n, h, w = plate.shape[:3]
    side = min(h, w)
    if not timeline.hits:
        return plate.copy()
    hero = int(np.argmax(timeline.sizes))
    k0, size = int(np.clip(timeline.hits[hero], 0, n - 1)), float(timeline.sizes[hero])
    origin, aim, s = blast_point(track, k0, "palm" if form == "palm" else "fist")
    origin = inside(origin, w, h)
    cam = timeline.camera
    el = library_element() if element is None else element
    floor = C.estimate_black_level(el)
    from .bending import border_fade

    # the render's own frame edges fade out (its smoke rising past the top of the render drew a straight cut across
    # the room once carried there)
    edge = border_fade(el.shape[1], el.shape[2], EDGE_FADE)[..., None]
    el = (floor + (el.astype(np.float32) - floor) * edge).clip(0, 255).astype(np.uint8)
    e_ign, el_centre, el_r8 = render_blast(el, floor)
    R = min(SIZE * s * size, MOST * side)
    k_el = min(R / el_r8, CLOUD_MOST * side / max(render_reach(el, floor, el_centre, e_ign) * (1 + SWELL[0]), 1.0))
    R = k_el * el_r8
    smoke_ref = C.smoke_reference_luma(el, PARAMS, floor, True)
    shading_ref = C.smoke_shading_ref(el, PARAMS, floor, True, smoke_ref)
    billow_ref = soot_shading_ref(el, floor, e_ign + 3 * GROW_FRAMES)
    env_rgb = C.environment_colour(plate[0], el, floor, PARAMS)
    amb = C.ambient_map(plate[k0], PARAMS)
    plate_key = float((plate[k0, ::8, ::8].astype(np.float32) / 255.0 @ C._LUMA).mean())
    some = np.stack([lin(plate[i]) for i in np.linspace(0, n - 1, 8).astype(int)])
    white, tint = fx.plate_white(some), fx.plate_tint(some)
    del some
    rng = np.random.default_rng(seed)
    dust = 0.35 + 0.65 * np.clip(cv2.resize(rng.random((h // 20, w // 20)).astype(np.float32), (w, h),
                                            interpolation=cv2.INTER_CUBIC), 0, 1)
    parts = Particles(origin, aim, s, R, side, seed, count=0.4 + 0.6 * size)
    reach = 1.05 * max(math.hypot(origin[0] - x, origin[1] - y) for x in (0, w) for y in (0, h))

    def frame_of(since: int) -> int:  # the render's growth played BURST_SPEED times faster, then on in real time
        g = GROW_FRAMES / BURST_SPEED
        return e_ign + (int(round(since * BURST_SPEED)) if since <= g else int(round(GROW_FRAMES + (since - g))))

    def lift(t: int) -> tuple[float, float]:
        """(frames since its burst, px its middle has risen by then): it rolls up as it burns, its smoke climbs on."""
        since = max(0.0, (t - k0) - GROW_FRAMES / BURST_SPEED)  # (after its burst, as the render plays in real time)
        climb = since - ROLL[3] * (1.0 - math.exp(-since / ROLL[3]))
        return since, R * (ROLL[0] * (1.0 - math.exp(-since / ROLL[1])) + ROLL[2] * climb / fx.FPS)

    xs_all = np.arange(w, dtype=np.float32)
    shapes = SootShapes(el, floor, e_ign, k_el, side)

    def element_on(t: int) -> tuple[np.ndarray, np.ndarray] | None:
        """The render on frame ``t`` placed in the room, and its cloud's filled silhouette (SootShapes) with it."""
        e = frame_of(t - k0)
        if t < k0 or e >= len(el):
            return None
        since, up = lift(t)
        grow = 1.0 - math.exp(-since / SWELL[1])
        k = k_el * (1.0 + SWELL[0] * grow)  # the hot cloud swells as it burns out, and rises
        o = (origin[0], origin[1] - up)
        S = np.array([[k, 0, o[0] - k * el_centre[0]], [0, k, o[1] - k * el_centre[1]], [0, 0, 1]])
        Ai = cv2.invertAffineTransform((_between(cam, k0, t) @ S)[:2])
        mid = float(room_point(cam, k0, t, o)[1])
        v = np.arange(h, dtype=np.float32) - mid  # its rows over its middle are stretched up, the rest kept
        rows = mid + v / (1.0 + TOP_STRETCH * grow / (1.0 + np.exp(np.clip(v / (0.25 * R), -50, 50))))
        mx = (Ai[0, 0] * xs_all[None] + Ai[0, 1] * rows[:, None] + Ai[0, 2]).astype(np.float32)
        my = (Ai[1, 0] * xs_all[None] + Ai[1, 1] * rows[:, None] + Ai[1, 2]).astype(np.float32)
        sil = cv2.remap(shapes.at(e), (mx - 0.5) / 2, (my - 0.5) / 2, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # (from half the render's size)
        return cv2.remap(el[e], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), sil

    out = plate.copy()
    last = min(n, k0 + max(SHOCK_FRAMES, int(np.max(parts.t0 + parts.life)) + 1,
                           int(math.ceil(GROW_FRAMES / BURST_SPEED + len(el) - e_ign - GROW_FRAMES)) + 1))
    q = 8
    spread, spread_norm = falloff((h // q, w // q), LIGHT_FALLOFF[0] * R / q, R / q, LIGHT_FALLOFF[1])
    outline = Outline(h, w, side, seed)
    wobble = np.random.default_rng([seed, 89]).standard_normal((1, n + 8))
    wobble = cv2.GaussianBlur(wobble, (0, 0), FLICKER[1])[0, 4:-4]
    flicker = 1.0 + FLICKER[0] * np.clip(wobble / max(float(wobble.std()), 1e-6), -1.5, 1.5)
    held = held_colour = src = None  # (the light on the frame before, its colour, and where its fire's middle was)
    for t in range(k0, last):
        since = t - k0
        P = lin(plate[t]).astype(np.float32)
        room = P.copy()
        m = np.zeros((h, w), np.float32) if mask is None else mask[min(t, len(mask) - 1)].astype(np.float32) / 255.0
        m = np.minimum(m, cv2.GaussianBlur(m, (0, 0), 1.5))
        o_t = room_point(cam, k0, t, origin)
        placed = element_on(t)
        E8 = box = billow = rim_maps = None
        if placed is not None:
            E8, sil = placed
            E8 = radial_blur(E8, o_t, GROW_BLUR.get(since, 0.0))
            E8 = cv2.GaussianBlur(E8, (0, 0), DEFOCUS_PX * side / 1080)
            ys_, xs_ = np.nonzero(E8[::4, ::4].max(-1) > floor + 2)
            if ys_.size:
                box = (max(0, 4 * ys_.min() - 24), min(h, 4 * ys_.max() + 24), max(0, 4 * xs_.min() - 24),
                       min(w, 4 * xs_.max() + 24))
                y0, y1, x0, x1 = box
                smoke = 1.0 - np.clip(C.fire_weight_map(C._black_floor_f(E8[y0:y1, x0:x1], floor), PARAMS, w, True),
                                      0.0, 1.0)
                E8 = soot_detail(E8, floor, box, smoke)
                billow = soot_billows(E8, floor, box, billow_ref, smoke)
                E8 = fill_soot(E8, floor, box, sil[y0:y1, x0:x1], smoke)
                r_in = max(3, int(BILLOW_IN * side) | 1)
                inner = cv2.erode(sil[y0:y1, x0:x1], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_in, r_in)))
                billow = billow * cv2.GaussianBlur(inner, (0, 0), r_in / 2)
                rim_maps = outline.warp(E8, floor, box, since, lift(t)[1], [billow])
        # the blast's light on the room and the arm, from its own image: the render's fire, and the white-hot core's
        # light on its first frames, spread round it and tinted to its colour
        fire_small, colour, amb_t = None, fx.kelvin_rgb(4500.0), amb
        if E8 is not None:
            e = C._black_floor_f(cv2.resize(E8, (w // q, h // q), interpolation=cv2.INTER_AREA), floor)
            fire_small = C._lin(e) * C.fire_weight_map(e, PARAMS, w // q, True)[..., None]
            colour = light_colour(fire_small)
        core_l = CORE_LIGHT.get(since, 0.0) * size
        if core_l > 0:
            rc = (0.35 + 0.3 * since) * s
            yy, xx = np.mgrid[:h // q, :w // q].astype(np.float32) * q
            core_s = (3.0 * core_l * np.exp(-((xx - o_t[0]) ** 2 + (yy - o_t[1]) ** 2) / (2 * rc * rc)))[..., None] \
                * fx.kelvin_rgb(4500.0)
            f_y = 0.0 if fire_small is None else float(fire_small.sum())
            c_y = float(core_s.sum())
            colour = (colour * f_y + fx.kelvin_rgb(4500.0) * c_y) / (f_y + c_y)
            fire_small = core_s if fire_small is None else fire_small + core_s
        lit = None
        if fire_small is not None and fire_small.max() > 0:
            y_small = fire_small @ C._LUMA
            lit = blast_light(y_small, spread, spread_norm)
            ys_, xs_ = np.indices(y_small.shape)
            src = (q * float((xs_ * y_small).sum() / y_small.sum()), q * float((ys_ * y_small).sum() / y_small.sum()))
            held_colour = colour
        if held is not None:  # (the room's light falls by at most LIGHT_HOLD a frame)
            lit = LIGHT_HOLD * held if lit is None else np.maximum(lit, LIGHT_HOLD * held)
        if lit is not None and lit.max() > 1e-4:
            held, colour = lit, held_colour
            L = cv2.resize(lit, (w, h)) * np.float32(LIGHT_GAIN * (1.0 + (flicker[t] - 1.0) * min(1.0, since / 3.0)))
            rim = rim_light(m, src) if mask is not None else np.zeros((h, w), np.float32)  # (from the fire's middle)
            on = (1.0 - m) + m * (ARM_LIGHT[0] + ARM_LIGHT[1] * rim)
            shone = P + (fx.albedo(P, white, tint) * (L * on)[..., None] + np.float32(LIGHT_GLOW) * L[..., None]) \
                * colour
            hot = shone > LIGHT_KNEE  # (the composite takes the plate in 8 bits: past white it would clip flat)
            shone[hot] = LIGHT_KNEE + (1.0 - LIGHT_KNEE) * np.tanh((shone[hot] - LIGHT_KNEE) / (1.0 - LIGHT_KNEE))
            P = warm_light(P, shone, m, rim)
            amb_t = amb + np.float32(SOOT_LIT) * L[..., None] * colour
        # the fireball and its smoke, behind the arm
        if E8 is not None:
            comp = C.composite_frame(gam(P), E8, PARAMS, holdout_row=None, emission=tuple(o_t), fire_energy_ref=None,
                                     plate_key=plate_key, grain_sigma=1.5, black_floor=floor, fire_effect=True,
                                     dense_smoke=True, smoke_ref=smoke_ref, env_rgb=env_rgb, seed=seed + t,
                                     ambient=amb_t, shading_ref=shading_ref, smoke_albedo=SOOT_ALBEDO,
                                     g_smoke=1.0 - _ease((since - CLEAR[0]) / CLEAR[1]), thin_soot=True)
            Q = lin(comp)
            if box is not None:
                Q = fire_grade(Q, E8, floor, box)
            if billow is not None:  # (as the smoke clears, its billows go with it)
                y0, y1, x0, x1 = box
                lit = Q[y0:y1, x0:x1] * (1.0 + (1.0 - _ease((since - CLEAR[0]) / CLEAR[1])) * billow)[..., None]
                Q[y0:y1, x0:x1] = np.minimum(lit, np.maximum(Q[y0:y1, x0:x1], P[y0:y1, x0:x1]))  # (soot is never
                # lit brighter than the room behind it)
            if rim_maps is not None:
                outline.finish(Q, P, box, rim_maps)
            P = Q * (1 - m[..., None]) + P * m[..., None]
            if since >= 1:  # the hot air round and above the fireball shimmers
                hot = np.clip(fx._lum(lin(cv2.resize(E8, (w // 4, h // 4), interpolation=cv2.INTER_AREA))) * 4.0, 0, 1)
                P = fx.heat_haze(P[None], cv2.resize(hot, (w, h))[None], seed=seed, frame0=t)[0]
        core_i = CORE.get(since, 0.0) * size
        if core_i > 0:  # the white-hot core and its glow behind the arm; the lens's veil over everything
            rc = (0.35 + 0.3 * since) * s
            yy, xx = np.mgrid[:h, :w].astype(np.float32)
            r2 = (xx - o_t[0]) ** 2 + (yy - o_t[1]) ** 2
            core = (2.5 * core_i * np.exp(-r2 / (2 * rc * rc)))[..., None] * fx.kelvin_rgb(5200.0)
            glow = (0.18 * core_i * np.exp(-r2 / (2 * (3.0 * rc) ** 2)))[..., None] * fx.kelvin_rgb(3200.0)
            P = P + core * (1 - m[..., None]) + glow * (1 - 0.6 * m[..., None]) + np.float32(0.015 * core_i) \
                * fx.kelvin_rgb(3000.0)
        if since < SHOCK_FRAMES:
            P = shock_ring(P, o_t, aim, since, reach * (0.5 + 0.5 * size), side, seed, fx.kelvin_rgb(2600.0) * 0.6, dust)
        M = _between(cam, k0, t)
        cover = np.zeros((h, w), np.float32) if E8 is None else \
            np.clip((E8.max(-1).astype(np.float32) / 255.0 - 0.06) / 0.3, 0, 1) * 0.9
        for front in (False, True):  # behind the arm and inside the fireball's body, or flying towards the camera
            emit = parts.draw(since, (h, w), M, front)
            if emit.max() > 0:
                P = P + (emit if front else emit * ((1 - m) * (1 - cover))[..., None])
        stops = FLASH.get(since, 0.0) * size
        if stops > 0:  # the blast's flash lights what is near it most (a uniform wash greyed the hand to a ghost)
            yy, xx = np.mgrid[:h:4, :w:4].astype(np.float32)
            near = np.exp(-((xx - o_t[0]) ** 2 + (yy - o_t[1]) ** 2) / (2 * (FLASH_REACH * max(R, s)) ** 2))
            k = 1.0 + (2.0 ** stops - 1.0) * (FLASH_FAR + (1.0 - FLASH_FAR) * near)
            P = P * cv2.resize(k.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        out[t] = gam(shoulder_added(P, room))
        say("compositing", 0.7 + 0.14 * (t - k0 + 1) / max(1, last - k0), f"The explosion: frame {t + 1} of {last}")
    out = fx.shake(out, [k0], [size], px=KNOCK_PX * h / 1080, seed=seed)  # the camera knocked...
    for t in range(k0, min(n, k0 + 8)):  # ...and pushed by the pressure wave
        z = ZOOM * size * math.exp(-(((t - k0) - 2) / 1.6) ** 2)
        if z > 1e-4:
            o_t = room_point(cam, k0, t, origin)
            Mz = cv2.getRotationMatrix2D((float(o_t[0]), float(o_t[1])), 0.0, 1.0 + z)
            out[t] = cv2.warpAffine(out[t], Mz, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def _ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _between(cam, a: int, b: int) -> np.ndarray:
    from .bending import camera_between

    return camera_between(cam, a, b)
