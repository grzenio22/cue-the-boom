"""An explosion thrown from a strike (cuesync.elements' contract). A fireball rendered ahead by LTX-2.5 + the
ainvfx-fluid IC-LoRA over black (a white-hot flash bursting into a rolling ball of flame that turns into thick black
smoke, rising and drifting) is set off ahead of the fist on the strike and from then on is fixed in the room, never
carried on the hand: the fist pulls back from it, the hand can open beneath its smoke. Everything that touches the
plate is made here on the CPU:

- the white-hot core on the strike frame, its glow, and the lens's warm veil over the whole picture
- the exposure flash (+0.7 stop, the iris overshooting after) and the camera's knock and zoom punch
- the shock front: a ring racing out that bends the room behind it (red and blue a little apart) with lit dust on it
- sparks flung off the ball (blackbody-cooling comets with drag and gravity, some flying past the arm) and embers
  floating up in the smoke
- the blast's light on the room and on the arm, wrapping round the arm's outline on the side facing it
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
THROW = (0.8, 2.5, 0.12)  # hands ahead of the fist it goes off: the strike's peak speed (px a second) over the hand times
# the third, between the first two (a harder punch throws it further); without a measured strike, 1 hand
PALM_THROW = 0.6  # a thrust palm's blast goes off this far past the palm's middle
EDGE = 0.10  # it goes off at least this share of the frame inside its edges
GROW_FRAMES, BURST_SPEED = 8, 2.0  # the render's first frames of growth play this much faster (a blast is quick)

CORE = {0: 1.0, 1: 0.5, 2: 0.15}  # the white-hot core's strength on the strike frame and after
FLASH = {0: 0.7, 1: 0.4, 2: 0.18, 3: 0.05, 4: -0.08}  # exposure, stops: the flash, and the iris overshooting after it
GROW_BLUR = {0: 0.3, 1: 0.22, 2: 0.12, 3: 0.07, 4: 0.04}  # the fireball's growth over the shutter, per frame
RIM_GAIN = 3.0  # the blast wrapping round the arm's outline, against its light on the hand
RIM_EDGE = 0.10  # hand sizes: how far in from the arm's outline the rim reaches
KNOCK_PX, ZOOM = 14.0, 0.022  # the camera's knock (px at 1080) and its push from the pressure wave (share)
FW_SOFT = 6.0  # px at 1080: where the fire is, softened (a hard fire/soot boundary drew flat orange paint)
GRADE = 0.6  # how far the fire's colour goes onto a camera's fire colours for its brightness (a render's lemon-yellow
# body is no colour a flame has: over a khaki wall it read yellow-green)
DEFOCUS_PX = 1.2  # px at 1080: the blast is further away than the fist the phone focused on
SOOT_ALBEDO = 0.25  # its smoke is black: it takes this share of the room's light
SOOT_FILL = 0.13  # display level (black taken off) the cloud's darkest folds are raised to inside its silhouette
SOOT_CLOSE = 0.03  # share of the short side: gaps narrower than this inside the cloud are closed before it is filled
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
    on = np.nonzero(counts >= 4)[0]
    if on.size == 0:
        raise ValueError("the explosion's render shows no blast")
    k = int(on[0])
    ys, xs = np.nonzero(lit[k])
    centre = np.array([4.0 * xs.mean(), 4.0 * ys.mean()])
    r8 = 4.0 * math.sqrt(float(counts[min(k + GROW_FRAMES, len(element) - 1)]) / math.pi)
    return k, centre, max(r8, 8.0)


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
def fill_soot(E8: np.ndarray, floor: int, box) -> np.ndarray:
    """The render with its cloud's darkest folds read as dense soot, not holes: over black the self-shadowed underside
    of a black cloud renders black, and composited it let the wall through as pale patches. Inside the cloud's filled
    silhouette nothing is darker than SOOT_FILL; the silhouette's edge stays soft."""
    y0, y1, x0, x1 = box
    part = E8[y0:y1, x0:x1]
    y = part.max(-1).astype(np.float32) - floor
    body = float(np.percentile(y[y > 4], 90)) if (y > 4).sum() > 100 else 0.0
    if body <= 0:
        return E8
    lit = (y > max(10.0, 0.3 * body)).astype(np.uint8)  # the cloud's body, not the render's faint haze round it
    k = max(3, int(SOOT_CLOSE * min(E8.shape[:2])) | 1)
    closed = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ff = np.pad(closed, 1)  # a ring of outside all round, so the fill reaches every gap open to the outside
    mask = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 2)
    sil = cv2.erode((ff[1:-1, 1:-1] != 2).astype(np.float32), np.ones((5, 5), np.uint8))
    sil = cv2.GaussianBlur(sil, (0, 0), 0.006 * min(E8.shape[:2]))
    around = cv2.GaussianBlur(part.astype(np.float32), (0, 0), 0.02 * min(E8.shape[:2]))  # a fold takes the smoke
    lift = np.maximum(0.85 * around, floor + SOOT_FILL * (255 - floor)) * sil[..., None]  # round it: billows kept
    out = E8.copy()
    out[y0:y1, x0:x1] = np.maximum(part.astype(np.float32), lift).astype(np.uint8)
    return out


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


def rim_light(m: np.ndarray, src, s: float) -> np.ndarray:
    """(h, w) in [0, 1]: how much of the blast behind the arm wraps round its outline: a band inside the matte's edge,
    on the side facing the blast (outward normal . direction to it)."""
    h, w = m.shape
    q = 2
    mq = cv2.resize(m, (w // q, h // q), interpolation=cv2.INTER_AREA)
    d = cv2.distanceTransform((mq > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    gy, gx = np.gradient(cv2.GaussianBlur(d, (0, 0), 1.5))
    nrm = np.hypot(gx, gy) + 1e-6
    nx, ny = -gx / nrm, -gy / nrm  # outward
    yy, xx = np.indices(d.shape).astype(np.float32)
    lx, ly = src[0] / q - xx, src[1] / q - yy
    ll = np.hypot(lx, ly) + 1e-6
    face = np.clip((nx * lx + ny * ly) / ll, 0, 1)
    band = np.exp(-d / max(RIM_EDGE * s / q, 1.0)) * (mq > 0.5)
    return cv2.resize((band * (0.25 + 0.75 * face) * face ** 0.5).astype(np.float32), (w, h),
                      interpolation=cv2.INTER_LINEAR)


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
    e_ign, el_centre, el_r8 = render_blast(el, floor)
    R = min(SIZE * s * size, MOST * side)
    k_el = R / el_r8
    smoke_ref = C.smoke_reference_luma(el, PARAMS, floor, True)
    shading_ref = C.smoke_shading_ref(el, PARAMS, floor, True, smoke_ref)
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

    def element_on(t: int) -> np.ndarray | None:
        e = frame_of(t - k0)
        if t < k0 or e >= len(el):
            return None
        S = np.array([[k_el, 0, origin[0] - k_el * el_centre[0]], [0, k_el, origin[1] - k_el * el_centre[1]], [0, 0, 1]])
        A = (_between(cam, k0, t) @ S)[:2]
        return cv2.warpAffine(el[e], A.astype(np.float32), (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

    out = plate.copy()
    last = min(n, k0 + max(SHOCK_FRAMES, int(np.max(parts.t0 + parts.life)) + 1,
                           int(math.ceil(GROW_FRAMES / BURST_SPEED + len(el) - e_ign - GROW_FRAMES)) + 1))
    q = 8
    for t in range(k0, last):
        since = t - k0
        P = lin(plate[t]).astype(np.float32)
        m = np.zeros((h, w), np.float32) if mask is None else mask[min(t, len(mask) - 1)].astype(np.float32) / 255.0
        m = np.minimum(m, cv2.GaussianBlur(m, (0, 0), 1.5))
        o_t = room_point(cam, k0, t, origin)
        E8 = element_on(t)
        box = None
        if E8 is not None:
            E8 = radial_blur(E8, o_t, GROW_BLUR.get(since, 0.0))
            E8 = cv2.GaussianBlur(E8, (0, 0), DEFOCUS_PX * side / 1080)
            ys_, xs_ = np.nonzero(E8[::4, ::4].max(-1) > floor + 2)
            if ys_.size:
                box = (max(0, 4 * ys_.min() - 24), min(h, 4 * ys_.max() + 24), max(0, 4 * xs_.min() - 24),
                       min(w, 4 * xs_.max() + 24))
                E8 = fill_soot(E8, floor, box)
        # the blast's light on the room and the arm: from the render's fire, and the white-hot core on its first frames
        fire_small = None
        if E8 is not None:
            e = C._black_floor_f(cv2.resize(E8, (w // q, h // q), interpolation=cv2.INTER_AREA), floor)
            fire_small = C._lin(e) * C.fire_weight_map(e, PARAMS, w // q, True)[..., None]
        core_i = CORE.get(since, 0.0) * size
        if core_i > 0:
            rc = (0.35 + 0.3 * since) * s
            yy, xx = np.mgrid[:h // q, :w // q].astype(np.float32) * q
            core_s = (3.0 * core_i * np.exp(-((xx - o_t[0]) ** 2 + (yy - o_t[1]) ** 2) / (2 * rc * rc)))[..., None] \
                * fx.kelvin_rgb(4500.0)
            fire_small = core_s if fire_small is None else fire_small + core_s
        if fire_small is not None and fire_small.max() > 0:
            hand_f, room_f, _ = fx.light_fields(fire_small[None].astype(np.float32), np.array([s]), (h, w))
            _, radius, colour = fx.element_light(fire_small[None].astype(np.float32), "fire", size=(h, w))
            room = cv2.resize(room_f[0], (w, h)) * np.float32(fx.ROOM_GAIN * 1.3)
            handl = cv2.resize(hand_f[0], (w, h)) * np.float32(fx.HAND_GAIN)
            rim = rim_light(m, o_t, s)
            light = np.minimum(room * (1 - m) + handl * m * (0.22 + RIM_GAIN * rim) + room * m * 0.5, 3.0)
            P = P + fx.albedo(P, white, tint) * light[..., None] * colour[0]
            P = P + cv2.resize(cv2.GaussianBlur(fire_small, (0, 0), max(1.0, float(radius[0]) / q)), (w, h)) * 0.05
        # the fireball and its smoke, behind the arm
        if E8 is not None:
            comp = C.composite_frame(gam(P), E8, PARAMS, holdout_row=None, emission=tuple(o_t), fire_energy_ref=None,
                                     plate_key=plate_key, grain_sigma=1.5, black_floor=floor, fire_effect=True,
                                     dense_smoke=True, smoke_ref=smoke_ref, env_rgb=env_rgb, seed=seed + t,
                                     ambient=amb, shading_ref=shading_ref, smoke_albedo=SOOT_ALBEDO)
            Q = lin(comp)
            if box is not None:
                Q = fire_grade(Q, E8, floor, box)
            P = Q * (1 - m[..., None]) + P * m[..., None]
            if since >= 1:  # the hot air round and above the fireball shimmers
                hot = np.clip(fx._lum(lin(cv2.resize(E8, (w // 4, h // 4), interpolation=cv2.INTER_AREA))) * 4.0, 0, 1)
                P = fx.heat_haze(P[None], cv2.resize(hot, (w, h))[None], seed=seed, frame0=t)[0]
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
        stops = FLASH.get(since, 0.0) * (size if FLASH.get(since, 0.0) > 0 else 1.0)
        if stops:
            P = P * np.float32(2.0 ** stops)
        from .bending import shoulder

        shoulder(P)
        out[t] = gam(P)
        say("compositing", 0.7 + 0.14 * (t - k0 + 1) / max(1, last - k0), f"The explosion: frame {t + 1} of {last}")
    out = fx.shake(out, [k0], [size], px=KNOCK_PX * h / 1080, seed=seed)  # the camera knocked...
    for t in range(k0, min(n, k0 + 8)):  # ...and pushed by the pressure wave
        z = ZOOM * size * math.exp(-(((t - k0) - 2) / 1.6) ** 2)
        if z > 1e-4:
            o_t = room_point(cam, k0, t, origin)
            Mz = cv2.getRotationMatrix2D((float(o_t[0]), float(o_t[1])), 0.0, 1.0 + z)
            out[t] = cv2.warpAffine(out[t], Mz, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def _between(cam, a: int, b: int) -> np.ndarray:
    from .bending import camera_between

    return camera_between(cam, a, b)
