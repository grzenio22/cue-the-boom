"""Air, made rather than rendered (cuesync.elements): a spinning ball of air over the open palm (``ball``), or a push
of air out of it (``blast``).

Air is invisible. What shows is what it bends (the plate seen through it, shifted where the air is denser or swirls),
the light it gathers and spreads (a shadowgraph: denser air focuses the room's light into a bright band with a darker
one beside it, as the shadow of a candle's hot air does on a wall) and what it carries (dust, grit and faint streaks of
moving air catching the room's light). All of it is drawn from the plate itself, the hand's track and its matte, in
linear light, with numpy and OpenCV, every random part from ``seed``: the same inputs give the same bytes. Air throws
no light of its own, so nothing here lights the hand or the room; what it carries takes the colour of the room's lit
surfaces (fx_light._lit_tone), and the plate's own grain where it covers the plate.

- ``ball``: a sphere of air standing on the palm, held by the voice. What is behind it is magnified a little (a weak
  lens), squeezed at its rim and twisted round its spin axis by lobes that turn with it; a shadowgraph band runs round
  its rim; a faint milky shell is densest at its limb, where the eye grazes the most of it; streaks of air orbit it on
  orbits leaning off its axis, so they wrap a ball (round one axis alone they stacked into a jar of rings), the far
  ones behind the fingers; dust swirls inside, the room's dust is drawn in round its equator, and a few bigger motes
  glint as they turn. It gathers in from wide as the voice starts, grows and spins faster the longer and louder the
  voice holds, rides out a breath, and bursts outward when the voice lets go.
- ``blast``: a push of air out of the palm on each of the voice's hits. A pressure front runs out across the room
  (what is seen through it bulges outward, its edge is a shadowgraph band), a turbulent wake behind it smears what it
  passes, a lumpy cloud of dusty air thrown off the palm lags the front, dust and flakes fly out of the palm in 3D
  (towards the lens they grow and go soft), the room's own dust is shoved outward as the front passes it, and all of
  it then slows, swirls and settles; what goes behind the palm's plane is hidden by the hand. What was thrown stays in
  the room: on a handheld shot it moves with the room, not with the screen. A push at the lens reaches the phone: the
  picture swells and smears outward for a moment.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import fx_light as fx
from .elements import FORMS, Timeline
from .transfer import gam, lin

_BALL, _BLAST, _ROOM, _GRAIN = 11, 12, 13, 14  # mixed into each seed, so one seed gives each part its own randomness

# ---- the hand -------------------------------------------------------------------------------------------------------
TURN_EDGE = 0.3  # a palm whose normal faces the lens less than this (its z) is close to edge-on: there its side is
# taken from the frame before rather than from which way it faces (each frame on its own flipped the ball across the
# palm as a hand turned edge-on)
SMOOTH_XY, SMOOTH_NORMAL, SMOOTH_SIZE = 0.8, 2.0, 2.0  # frames: the anchor, the palm's normal and the hand's size are
# smoothed over time this much, both ways (the landmarks' normal jitters by a few degrees a frame, and a ball a hand
# off the palm shook with it; a lagging filter left it sliding behind a moving palm)
AIM_ACROSS = 0.7  # with the studio's arrow drawn, the push goes at least this much across the picture (the arrow is
# in the picture; a palm facing the lens would otherwise send it at the camera whatever was drawn)
FEED = 0.35  # the voice feeds the element while it holds a sound, or while it is louder than this (of the take's
# loudest: a phone's room tone and breaths read 0.15-0.34 of a "ffff" and fed a ball from the first frame)

# ---- the ball -------------------------------------------------------------------------------------------------------
BALL_R = 0.72  # hand sizes: the ball's radius once formed (about as wide as the spread hand)
BALL_GROW, BALL_GROW_S = 0.3, 3.0  # ... and it grows this much more over this many seconds of held voice
BALL_BREATH = 0.06  # ... and swells this much more at the voice's loudest
BALL_GAP = 0.15  # hand sizes between the palm and the ball's underside (along the palm's normal)
BALL_FORM, BALL_RELEASE = 12, 9  # frames it gathers over (half a second), and bursts and thins out over
BALL_LINGER = 6  # frames it stays after the voice last feeds it, before it bursts
BALL_BRIDGE = 30  # frames: a silence no longer than this (a breath between two "ffff"s) keeps the ball, thinned out
# and slowing (BALL_IDLE); a longer one lets it go, and the next sound gathers a new one
BALL_IDLE = 0.6  # between two sounds the ball thins to this much of itself
BALL_MIN = 22  # frames it lives at least: a short "pf" still shows a ball forming, spinning and bursting
BALL_SPIN = 0.9  # revolutions a second once formed on a quiet voice ...
BALL_SPIN_VOICE, BALL_SPIN_HELD = 1.0, 0.6  # ... this much faster at the voice's loudest, and this much after
# BALL_GROW_S seconds held (about 2.5 a second at the most: faster smeared every streak into a ring)
BALL_LENS = 0.16  # the plate behind its middle is magnified this much (dense spinning air, a weak lens)
BALL_RIM = 0.05  # radii: at its rim what is behind is pulled in from this much further out (the rim draws the ball's
# outline across the fingers seen through it)
BALL_TWIST = 0.17  # rad: how far the shell twists what is behind it round the axis, at its densest
BALL_TURB = 3.2  # px at 1080: the co-rotating shimmer
BALL_CA = 0.06  # colour fringe of the bend (0.22 drew red and cyan outlines round the fingers)
BALL_SHADOW = 0.12  # the shadowgraph: the rim focuses the room's light into a band this much brighter, with a darker
# band inside it (without it, on a plain wall the ball was a few faint streaks: refraction shows only what has texture)
BALL_MIST = 0.2  # opacity of the milky shell at its limb (0.3 and more read as a fogged glass bowl)
BALL_BACK = 0.45  # the far side shows this much (seen through the near side), and not through the fingers
BALL_CAUSTIC = 0.06  # the palm under the ball takes its light focused into moving patches, this much brighter / darker
BALL_STREAKS, BALL_INFLOW = 46, 8  # streaks orbiting the ball; long arms drawing the room's air in round its equator
STREAK_TILT = 0.6  # rad: the streaks' orbits lean off the spin axis by up to this
STREAK_ALPHA = 0.5  # peak opacity of the brightest streak (at 0.3 the ball vanished at phone size on a plain wall)
STREAK_PX = (1.3, 2.8)  # px at 1080: a fine streak's softness, and a broad wisp's
STREAK_STEP = 0.7  # px between the points a streak is drawn from (further apart a soft streak turned to beads)
BALL_DUST, ROOM_DUST, MOTES = 300, 180, 26  # dust inside; the room's dust drawn in; bigger motes that glint
DUST_ALPHA, MOTE_ALPHA = 0.5, 0.85  # peak opacity of a bright dust speck, and of a mote at its glint
BALL_PAD = 2.9  # radii: the ball's air (the inflow arms, the dust it draws in) reaches this far from its middle

# ---- the blast ------------------------------------------------------------------------------------------------------
BLAST_FRAMES = 12  # the front crosses the room in half a second
FRONT_FROM, FRONT_TO = 0.06, 0.95  # shares of the short side: the front's radius first and last (more towards the lens)
BUBBLE_PX = 46.0  # px at 1080: how far the pressure bubble pushes what is seen through it outward, at its strongest
BUBBLE_END = 0.3  # ... and the share of that it still has as it leaves the frame (it bulged the switch plate visibly)
FRONT_W = 0.05  # share of the short side: the front's thickness (it thickens as it goes)
SHADOWGRAPH = 0.12  # the front gathers and spreads the light behind it: a bright band, dark either side, this deep (at
# 0.06 the front was invisible on a plain wall)
WAKE_PX = 5.0  # px at 1080: the turbulent wake's radial smear behind the front
BLAST_CA = 0.05  # (0.12 split an orange fringe off the fingertips while the bubble was small and strong)
BUBBLE_CLEAR = (0.6, 0.9)  # hand sizes: the bubble bends nothing until its front is this far out, and is whole this
# much further on (bent while it was still over the hand, the fingers wobbled on the hit like a glitch)
CLOUD_N, CLOUD_CLUMPS = 4000, 9  # parcels of dusty air thrown out with the push, most of them in this many clumps (a
# puff is lumpy; a texture laid on a ring round the palm read as a faint smudge, not as air going somewhere)
CLOUD_ALPHA = 0.42  # opacity the dusty air saturates at, where it is thickest
CLOUD_THICK = 0.6  # ... and how fast it gets there: the opacity of a cloud spread evenly over the frame's short side
# squared is about this much of CLOUD_ALPHA
CLOUD_DEPTH = 0.3  # the dusty air keeps this share of its speed towards the lens (all of it carried the puff off
# the bottom of the picture: off the lens's axis, what comes at the lens slides outward)
CLOUD_Q = 8  # the dusty air is drawn this many times coarser than the frame, then blurred and drawn at full size
CLOUD_SOFT = (16.0, 4.0)  # px at 1080: how soft the dusty air is as it leaves the palm, and how much softer a frame
# (it diffuses as it slows; kept sharp, its parcels showed as blobs)
CLOUD_FADE = (6, 22)  # frames: the dusty air thins out from this many frames after the push, over this many more
DUST_N, FLAKES = 1400, 60  # dust and grit flung out of the palm; bigger flakes that tumble
ROOM_N = 1500  # the room's own dust, still and unseen until the front shoves it
ROOM_PUSH = 0.06  # share of the short side a frame: how fast the front throws the room's dust it passes
ROOM_SEEN = 0.45  # thrown, it shows at least this much as it slows and hangs (seen only while fast, the room's dust
# flashed past and was gone before the eye found it)
DUST_LIFE = (22, 30)  # frames: thrown dust hangs this long after the push, then is gone over the next this many
FOCAL = 1.0  # focal length in short sides (a phone's main camera: about 53 degrees across the short side)
BUFFET = 0.018  # a push straight at the lens swells the picture this much when the gust reaches the phone
FIELD_Q = 2  # the bend is worked out on a grid this many times coarser and drawn at full size (it is smooth)


# ---- shared pieces --------------------------------------------------------------------------------------------------

def _smooth(x):
    """Smoothstep of x clipped to [0, 1] (numpy array or float)."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _splat(buf: np.ndarray, x: np.ndarray, y: np.ndarray, v: np.ndarray) -> None:
    """Adds weights ``v`` at sub-pixel points (x, y) into ``buf`` (h, w), bilinearly, in place."""
    h, w = buf.shape
    x0, y0 = np.floor(x), np.floor(y)
    fx_, fy_ = x - x0, y - y0
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
    idx, wt = [], []
    for dx, dy, f in ((0, 0, (1 - fx_) * (1 - fy_)), (1, 0, fx_ * (1 - fy_)), (0, 1, (1 - fx_) * fy_), (1, 1, fx_ * fy_)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        idx.append(yi[ok] * w + xi[ok])
        wt.append((v * f)[ok])
    idx_ = np.concatenate(idx)
    if idx_.size:
        buf += np.bincount(idx_, weights=np.concatenate(wt), minlength=h * w).astype(np.float32).reshape(h, w)


def _line_blur(buf: np.ndarray, sigma: float) -> np.ndarray:
    """A splatted line (unit weight a pixel along it) blurred to ``sigma`` px and scaled so its middle reads its
    weight: a soft line, as bright however soft."""
    return cv2.GaussianBlur(buf, (0, 0), sigma) * np.float32(math.sqrt(2 * math.pi) * sigma)


def _tile_noise(rng: np.random.Generator, h: int, w: int, sx: float, sy: float, ch: int = 1) -> np.ndarray:
    """Smooth unit noise (h, w, ch) that wraps round both ways, so it can be sampled round and round."""
    n = rng.standard_normal((h, w, ch)).astype(np.float32)
    py, px = int(3 * sy) + 1, int(3 * sx) + 1
    n = cv2.GaussianBlur(np.pad(n, ((py, py), (px, px), (0, 0)), mode="wrap"), (0, 0), sigmaX=sx, sigmaY=sy)
    n = n.reshape(h + 2 * py, w + 2 * px, ch)[py:-py, px:-px]
    return np.ascontiguousarray(n / max(float(n.std()), 1e-6)).squeeze()


class _Curl:
    """A smooth divergence-free 2D flow (a few travelling waves of a stream function): the eddies that swirl dust as
    it slows, the same for the same seed."""

    def __init__(self, rng: np.random.Generator, cell: float, n: int = 7, drift: float = 0.08):
        ang = rng.uniform(0, 2 * np.pi, n)
        kk = 2 * np.pi / (cell * rng.uniform(0.6, 1.5, n))
        self.kx, self.ky = kk * np.cos(ang), kk * np.sin(ang)
        self.ph = rng.uniform(0, 2 * np.pi, n)
        self.om = rng.uniform(-drift, drift, n) * 2 * np.pi
        self.a = 1.0 / (kk * math.sqrt(n / 2))  # about unit speed

    def __call__(self, x: np.ndarray, y: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        arg = self.kx[:, None] * x[None] + self.ky[:, None] * y[None] + (self.om * t + self.ph)[:, None]
        c = self.a[:, None] * np.cos(arg)
        return (c * self.ky[:, None]).sum(0), -(c * self.kx[:, None]).sum(0)


def _remap_rgb(P: np.ndarray, dx: np.ndarray, dy: np.ndarray, box: tuple[int, int, int, int], ca: float) -> None:
    """In place: ``P`` inside ``box`` (y0, y1, x0, x1) resampled from (x + dx, y + dy), red bent (1 + ca) as far and
    blue (1 - ca): the slight colour fringe of a bend. Past the frame's edge it reads the edge."""
    y0, y1, x0, x1 = box
    gx = np.arange(x0, x1, dtype=np.float32)[None, :]
    gy = np.arange(y0, y1, dtype=np.float32)[:, None]
    src = P.copy()
    for c, kk in enumerate((1 + ca, 1.0, 1 - ca)):
        P[y0:y1, x0:x1, c] = cv2.remap(src[..., c], gx + np.float32(kk) * dx, gy + np.float32(kk) * dy,
                                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _upsample(field: np.ndarray, bh: int, bw: int) -> np.ndarray:
    """A field worked out on the coarse grid of a box (its cells' middles), drawn at the box's full size."""
    return cv2.resize(field, (bw, bh), interpolation=cv2.INTER_LINEAR)


def _coarse_grid(box: tuple[int, int, int, int], q: int) -> tuple[np.ndarray, np.ndarray]:
    """The middles of a box's q x q cells, in frame px: (x (1, bw/q), y (bh/q, 1)), as _upsample reads them."""
    y0, y1, x0, x1 = box
    bw, bh = max(1, -(-(x1 - x0) // q)), max(1, -(-(y1 - y0) // q))
    sx, sy = (x1 - x0) / bw, (y1 - y0) / bh
    return ((x0 + (np.arange(bw, dtype=np.float32) + 0.5) * sx - 0.5)[None, :],
            (y0 + (np.arange(bh, dtype=np.float32) + 0.5) * sy - 0.5)[:, None])


_STEP_LUT = (2.2 * np.maximum(np.linspace(0, 1, 1024) ** 2, 1e-4) ** (-1 / 2.2)).astype(np.float32)  # one 8-bit
# gamma step at linear luminance y, as a share of y, read at sqrt(y) (a power per pixel was a third of a frame's time)


def _over(P: np.ndarray, alpha: np.ndarray, colour: np.ndarray, noise: np.ndarray | None, code: float) -> None:
    """In place: ``colour`` laid over ``P`` (h, w, 3) at ``alpha`` (h, w), with the plate's grain in the share of it
    it covers (``noise``: unit grain the size of ``P``; ``code``: one unit of it in 8-bit gamma steps): the made layer is
    no cleaner than the footage under it."""
    a = np.clip(alpha, 0.0, 0.92)[..., None]
    P *= 1 - a
    P += a * colour
    if noise is not None and code > 0:
        i = (np.sqrt(np.clip(P @ fx._LUMA, 0.0, 1.0)) * np.float32(1023)).astype(np.int32)
        P *= 1 + (a[..., 0] * noise * (np.float32(code) * _STEP_LUT[i]))[..., None]


def _grain(seed: int, salt: int, t: int, h: int, w: int) -> np.ndarray:
    return fx._soft_noise(fx._rng(seed, _GRAIN, salt, t), h, w)


def _clip_box(cx: float, cy: float, rx: float, ry: float, w: int, h: int) -> tuple[int, int, int, int] | None:
    y0, y1 = max(0, int(math.floor(cy - ry))), min(h, int(math.ceil(cy + ry)) + 1)
    x0, x1 = max(0, int(math.floor(cx - rx))), min(w, int(math.ceil(cx + rx)) + 1)
    return None if y1 - y0 < 2 or x1 - x0 < 2 else (y0, y1, x0, x1)


def _gauss_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """``x`` (n, ...) smoothed along time by a Gaussian of ``sigma`` frames, both ways, its ends held."""
    if sigma <= 0 or len(x) < 2:
        return x.copy()
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return np.stack([np.tensordot(k, pad[i: i + 2 * r + 1], axes=(0, 0)) for i in range(len(x))])


def palm_normals(track: dict, n: int, w: int, h: int) -> np.ndarray:
    """(n, 3) unit normal of the palm in the camera's axes (x right, y down, z towards the lens), turned to face the
    lens: the air comes out of the palm we film. From the 3D landmarks (MediaPipe's z grows away from the lens); a
    frame without them takes the nearest frame's; a palm close to edge-on keeps the side it had the frame before
    (TURN_EDGE). A track without landmarks leans its own direction towards the lens. The track's handedness is not
    used: MediaPipe calls an upturned palm the other hand, which flips the palm's side."""
    kx, ky = w / float(track.get("W", w)), h / float(track.get("H", h))
    lms = track.get("landmarks") or []
    out = np.full((n, 3), np.nan)
    for t in range(min(n, len(lms))):
        L = lms[t]
        if L is None or np.asarray(L).shape[-1] < 3:
            continue
        L = np.asarray(L, np.float64)[:, :3] * (kx, ky, math.sqrt(kx * ky))
        nn = np.cross(L[5] - L[0], L[17] - L[0]) * (1.0, 1.0, -1.0)
        norm = float(np.linalg.norm(nn))
        if norm > 1e-6:
            out[t] = nn / norm if nn[2] >= 0 else -nn / norm
    have = np.flatnonzero(~np.isnan(out[:, 0]))
    if have.size == 0:
        _, u, _ = fx.track_arrays(track, n, w, h)
        nn = np.column_stack([0.6 * u, np.full(n, 0.8)])
        return nn / np.linalg.norm(nn, axis=1, keepdims=True)
    near = have[np.abs(np.arange(n)[:, None] - have[None, :]).argmin(1)]
    out = out[near]
    for t in range(1, n):  # edge-on, the side it faced is the one it faced before
        if abs(out[t, 2]) < TURN_EDGE and out[t] @ out[t - 1] < 0:
            out[t] = -out[t]
    out = _gauss_smooth(out, SMOOTH_NORMAL)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def voice_level(timeline: Timeline) -> np.ndarray:
    """(n,) the voice's loudness as the air follows it: up at once, down over about half a second."""
    e = timeline.energy
    out = np.zeros(timeline.n, np.float32)
    prev = 0.0
    for t in range(timeline.n):
        prev += (0.6 if e[t] > prev else 0.08) * (float(e[t]) - prev)
        out[t] = prev
    return out


def fed(timeline: Timeline, level: np.ndarray) -> np.ndarray:
    """(n,) bool: frames the voice feeds the element (it holds a sound, or it is louder than FEED)."""
    held = np.zeros(timeline.n, bool)
    for a, b in timeline.holds:
        held[max(0, a): max(0, min(timeline.n, b))] = True
    return held | (level > FEED)


def _tone(frame_u8: np.ndarray) -> np.ndarray:
    """The colour of the room's lit surfaces on this frame (linear): fx_light._lit_tone of a quarter-size copy, made
    linear only at that size (it reads a mean of the brightest tenth, which needs no more)."""
    h, w = frame_u8.shape[:2]
    small = cv2.resize(frame_u8, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    return fx._lit_tone(lin(small))


# ---- the ball -------------------------------------------------------------------------------------------------------

def ball_lives(timeline: Timeline, level: np.ndarray) -> list[tuple[int, int]]:
    """[(first, release)] frames of each ball the voice makes: one gathers on a sound (a held sound, a hit, a frame
    the voice feeds it; on the tap without a voice), rides out silences of up to BALL_BRIDGE frames, and lets go
    BALL_LINGER frames after the voice last fed it (BALL_MIN frames after it started at the soonest)."""
    n = timeline.n
    sound = fed(timeline, level)
    for f in timeline.hits:
        if 0 <= f < n:
            sound[f] = True
    frames = np.flatnonzero(sound)
    if frames.size == 0:
        return [(int(timeline.tap_frame), int(timeline.tap_frame) + BALL_MIN)] if 0 <= timeline.tap_frame < n else []
    cut = np.flatnonzero(np.diff(frames) > BALL_BRIDGE)
    lives = []
    for a, b in zip(np.r_[0, cut + 1], np.r_[cut, frames.size - 1]):
        t0 = int(frames[a])
        t_rel = max(int(frames[b]) + 1 + BALL_LINGER, t0 + BALL_MIN)
        if lives and t0 < lives[-1][1] + BALL_RELEASE:  # the last one has not burst yet: it carries on
            lives[-1] = (lives[-1][0], max(lives[-1][1], t_rel))
        else:
            lives.append((t0, t_rel))
    return lives


class _Ball:
    """One ball of air: its seeded streaks, dust and textures, and the state it carries from frame to frame (how far
    it has spun, its axes)."""

    def __init__(self, seed: int, salt: int):
        rng = fx._rng(seed, _BALL, salt)
        N = BALL_STREAKS + BALL_INFLOW
        self.inflow = np.arange(N) >= BALL_STREAKS
        tilt = np.where(self.inflow, rng.uniform(0, 0.15, N), rng.uniform(0, 1, N) ** 0.7 * STREAK_TILT)
        az = rng.uniform(0, 2 * np.pi, N)
        ct, st, ca, sa = np.cos(tilt), np.sin(tilt), np.cos(az), np.sin(az)
        # each streak's own orbit: (u, v) span its plane, w is its axis, leaning off the ball's by its tilt
        self.u = np.column_stack([ct * ca, ct * sa, -st])
        self.v = np.column_stack([-sa, ca, np.zeros(N)])
        self.w = np.column_stack([st * ca, st * sa, ct])
        bands = rng.uniform(-0.8, 0.8, 6)  # streaks ride a few layers of flow, not a scribble
        lat = np.clip(bands[rng.integers(0, 6, N)] + rng.normal(0, 0.12, N), -1.15, 1.15)
        self.lat = np.where(self.inflow, rng.normal(0, 0.1, N), lat)
        self.rho = np.where(self.inflow, 1.0, rng.uniform(0.86, 1.03, N))
        self.flare = np.where(self.inflow, 1.5, 0.0)  # an inflow arm runs from the ball out to 2.5 radii
        self.phase = rng.uniform(0, 2 * np.pi, N)
        self.span = np.where(self.inflow, rng.uniform(2.0, 3.4, N), rng.uniform(1.1, 3.0, N))
        self.speed = np.where(self.inflow, 0.8, rng.uniform(0.85, 1.2, N))
        self.helix = rng.uniform(-0.18, 0.18, N)
        self.bright = np.where(self.inflow, 0.45 * rng.uniform(0.5, 1.0, N), rng.uniform(0.3, 1.0, N) ** 1.3)
        self.soft = (rng.random(N) < 0.45) | self.inflow  # broad soft wisps; the rest fine streaks
        self.wob = rng.uniform(0, 2 * np.pi, N)
        self.gaps = rng.uniform(1.0, 2.6, N)  # each streak comes and goes along its length this many times (whole,
        # each was a wire drawn round the ball)
        self.gph = rng.uniform(0, 2 * np.pi, N)
        m = BALL_DUST  # dust inside, orbiting the axis, faster further in (a vortex)
        self.d_lat = np.arcsin(rng.uniform(-0.95, 0.95, m))
        self.d_rho = rng.uniform(0.2, 1.05, m) ** 0.6
        self.d_phase = rng.uniform(0, 2 * np.pi, m)
        self.d_glint = rng.uniform(0, 2 * np.pi, m)
        self.d_gf = rng.uniform(0.2, 0.9, m)
        self.d_bright = rng.uniform(0.25, 1.0, m) ** 2
        self.d_dark = rng.random(m) < 0.12  # specks of grit, darker than the wall
        q = ROOM_DUST  # the room's dust, drawn in and round as the ball feeds on the air
        self.r_rho = rng.uniform(1.15, 2.7, q)
        self.r_phase = rng.uniform(0, 2 * np.pi, q)
        self.r_lat = rng.normal(0, 0.2, q)
        self.r_bright = rng.uniform(0.2, 0.9, q) ** 2
        k = MOTES  # bigger flecks riding the shell, flashing as they turn to the light
        self.m_lat = np.arcsin(rng.uniform(-0.8, 0.8, k))
        self.m_rho = rng.uniform(0.7, 1.0, k)
        self.m_phase = rng.uniform(0, 2 * np.pi, k)
        self.m_tum = rng.uniform(0, 2 * np.pi, k)
        self.m_tf = rng.uniform(0.15, 0.45, k)
        self.m_size = rng.uniform(1.4, 2.6, k)
        self.noise = _tile_noise(rng, 256, 256, 6.0, 6.0, 3)  # the co-rotating shimmer, and the caustics
        self.mist = _tile_noise(rng, 160, 640, 16.0, 1.8)  # lat x lon: long along the flow, thin across it
        self.arcs = _tile_noise(rng, 1, 512, 18.0, 0.5).reshape(1, -1)  # where round the rim its band is strong
        self.spin = 0.0
        self.axes: np.ndarray | None = None  # rows e1, e2, a: carried frame to frame

    def turn_axes(self, a: np.ndarray) -> np.ndarray:
        """The ball's axes for spin axis ``a``: the last frame's carried over to it (parallel transport), so the
        streaks' orbits never jump as the palm turns."""
        a = a / np.linalg.norm(a)
        if self.axes is None:
            ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            e1 = np.cross(a, ref)
        else:
            e1 = self.axes[0] - (self.axes[0] @ a) * a
            if np.linalg.norm(e1) < 1e-6:
                e1 = self.axes[1] - (self.axes[1] @ a) * a
        e1 = e1 / np.linalg.norm(e1)
        self.axes = np.stack([e1, np.cross(a, e1), a])
        return self.axes

    def _mist(self, sx, sy, sign, B):
        """The milky shell's density on the near (sign +1) or far (-1) side, fixed in the spinning air; ``sx``, ``sy``
        in radii from the middle."""
        z = np.sqrt(np.clip(1 - sx * sx - sy * sy, 0, 1)) * sign
        l1 = sx * B[0, 0] + sy * B[0, 1] + z * B[0, 2]
        l2 = sx * B[1, 0] + sy * B[1, 1] + z * B[1, 2]
        l3 = sx * B[2, 0] + sy * B[2, 1] + z * B[2, 2]
        lon = (np.arctan2(l2, l1) - self.spin) % (2 * np.pi)
        lat = np.arcsin(np.clip(l3, -1, 1))
        u = (lon * (640 / (2 * np.pi))).astype(np.float32)
        v = ((lat + np.pi / 2) * (159 / np.pi)).astype(np.float32)
        m = cv2.remap(self.mist, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        dens = np.clip(0.45 + 0.55 * m, 0, None) ** 2
        polar = np.clip(1.3 - np.abs(l3) ** 2 * 1.1, 0.2, 1)  # thinner round the poles: it spins round its axis
        limb = 0.22 + 0.78 * (1 - np.abs(z)) ** 1.6  # the eye grazes more air at the edge
        return dens * polar * limb

    def points(self, c, R, B, form, rel, vis, omega, t, k):
        """Everything the ball carries, as points: (x, y, depth, weight, kind) with kind 0 a fine streak, 1 a broad
        wisp, 2 lit dust, 3 dark grit, 4 a mote; the far side's points (depth < 0) carry BALL_BACK of their weight."""
        spread = 1 + 1.4 * (1 - form) ** 2  # forming: it gathers in from wide
        # streaks: each a comet of moving air along its orbit, its head a half frame on (the shutter)
        L = self.span * R * (self.rho + 0.5 * self.flare) * spread * (1 + rel)
        S = int(np.clip(L.max() / (STREAK_STEP * k), 16, 2400))
        s = np.linspace(0, 1, S)[None, :]
        head = self.phase + self.speed * (self.spin + 0.5 * omega)
        span = (self.span * (0.4 + 0.6 * form) * (1 - 0.6 * rel))[:, None]
        phi = head[:, None] - s * span
        lat = self.lat[:, None] + self.helix[:, None] * s + 0.1 * np.sin(0.2 * self.spin + self.wob)[:, None]
        rho = R * spread * (self.rho[:, None] + self.flare[:, None] * s ** 1.2) * (1 + 0.5 * rel * (1 + s))
        cl, sl = np.cos(lat), np.sin(lat)
        loc = rho[..., None] * (cl[..., None] * (np.cos(phi)[..., None] * self.u[:, None] + np.sin(phi)[..., None]
                                                 * self.v[:, None]) + sl[..., None] * self.w[:, None])
        pts = loc @ B  # (N, S, 3) in the camera's axes
        tail = np.where(self.inflow[:, None], 0.7 * np.sin(np.pi * s) ** 1.2 * (1.0 - rel) * (0.5 * (1 - form) + 0.6),
                        (1 - s) ** 1.2 * _smooth(s / 0.08 + 0.15))
        tail = tail * np.clip(0.55 + 0.6 * np.sin(2 * np.pi * self.gaps[:, None] * s + self.gph[:, None]
                                                  + 0.15 * self.spin), 0.06, 1.0)  # in wisps, coming and going
        ds = (span[:, 0] * rho.mean(1) / (S - 1))[:, None]  # px of streak each point stands for
        wt = self.bright[:, None] * tail * vis * ds
        kind = np.broadcast_to(self.soft[:, None].astype(np.int8), wt.shape)
        xs, ys, zs, ws, ks = [pts[..., 0].ravel()], [pts[..., 1].ravel()], [pts[..., 2].ravel()], [wt.ravel()], [
            kind.ravel()]
        # dust inside, smeared over the shutter's half frame
        sh = np.linspace(0, 1, 5)[None]
        om_d = 1.2 / np.sqrt(self.d_rho) * omega
        ph = self.d_phase[:, None] + (1.2 / np.sqrt(self.d_rho) * self.spin)[:, None] + sh * 0.5 * om_d[:, None]
        la = (self.d_lat + 0.1 * np.sin(0.3 * self.spin + self.d_glint))[:, None]
        rr = (self.d_rho * R * spread * (1 + 0.8 * rel))[:, None]
        p = (rr * np.cos(la))[..., None] * (np.cos(ph)[..., None] * B[0] + np.sin(ph)[..., None] * B[1]) + (
            rr * np.sin(la))[..., None] * B[2]
        gl = 0.3 + 0.7 * np.abs(np.cos(self.d_glint + self.d_gf * t)) ** 6
        wd = np.repeat((self.d_bright * gl * vis * 1.6 / 5)[:, None], 5, 1)
        kd = np.repeat(np.where(self.d_dark, 3, 2).astype(np.int8)[:, None], 5, 1)
        xs.append(p[..., 0].ravel()), ys.append(p[..., 1].ravel()), zs.append(p[..., 2].ravel())
        ws.append(wd.ravel()), ks.append(kd.ravel())
        # the room's dust drawn in: spiralling slowly in round the equator, more coming from outside
        prog = (self.r_phase * 0.37 + 0.02 * self.spin) % 1.0
        rr2 = R * (1.0 + (self.r_rho - 1.0) * (1 - prog))
        ph2 = self.r_phase[:, None] + 0.55 * self.spin * (R / rr2)[:, None] + sh * 0.3 * omega
        p = rr2[:, None, None] * (np.cos(ph2)[..., None] * B[0] + np.sin(ph2)[..., None] * B[1]) + (
            self.r_lat * R)[:, None, None] * B[2]
        wr = np.repeat((self.r_bright * np.sin(np.pi * prog) * vis * (1 - rel) * 1.4 / 5)[:, None], 5, 1)
        xs.append(p[..., 0].ravel()), ys.append(p[..., 1].ravel()), zs.append(p[..., 2].ravel())
        ws.append(wr.ravel()), ks.append(np.full(wr.size, 2, np.int8))
        # motes on the shell, flashing as each turns its face to the room's light
        ph3 = self.m_phase[:, None] + 1.05 * self.spin + sh * 0.5 * omega
        rr3 = (self.m_rho * R * spread * (1 + 0.9 * rel))[:, None]
        p = (rr3 * np.cos(self.m_lat)[:, None])[..., None] * (np.cos(ph3)[..., None] * B[0] + np.sin(ph3)[..., None]
                                                               * B[1]) + (rr3 * np.sin(self.m_lat)[:, None])[..., None] * B[2]
        glint = 0.25 + 0.75 * np.abs(np.cos(self.m_tum + self.m_tf * t)) ** 8
        wm = np.repeat((glint * vis * (self.m_size / 2.6) ** 2 / 5)[:, None], 5, 1)
        xs.append(p[..., 0].ravel()), ys.append(p[..., 1].ravel()), zs.append(p[..., 2].ravel())
        ws.append(wm.ravel()), ks.append(np.full(wm.size, 4, np.int8))
        x, y, z, wgt, kd_ = (np.concatenate(v) for v in (xs, ys, zs, ws, ks))
        wgt = np.where(z < 0, wgt * BALL_BACK, wgt)
        return c[0] + x, c[1] + y, z, wgt, kd_

    def draw(self, P: np.ndarray, off: tuple[int, int], c: np.ndarray, R: float, a: np.ndarray, form: float,
             rel: float, idle: float, omega: float, t: int, hand: np.ndarray | None, tone: np.ndarray,
             noise: np.ndarray | None, code: float, k: float) -> None:
        """Draws the ball on ``P`` (a linear crop of the frame whose top-left is ``off`` = (x, y) in frame px) in
        place: centre ``c`` (frame px), radius ``R`` (px), spin axis ``a``; ``form`` 0..1 how gathered, ``rel`` 0..1
        how far it has burst, ``idle`` how much of itself it is between two sounds, ``omega`` its spin this frame
        (rad), ``hand`` the matte over the crop (0..1)."""
        self.spin += omega
        B = self.turn_axes(a)
        h, w = P.shape[:2]
        ox, oy = off
        cx, cy, R = float(c[0] - ox), float(c[1] - oy), float(R)  # (plain floats keep the fields float32)
        vis = float(form * (1 - rel) ** 1.5 * idle)
        if vis <= 1e-3:
            return
        # which way the near side's air crosses the picture: the plate behind is dragged the same way
        ph = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        pt = np.cos(ph)[:, None] * B[0] + np.sin(ph)[:, None] * B[1]
        vel = -np.sin(ph)[:, None] * B[0] + np.cos(ph)[:, None] * B[1]
        near = pt[:, 2] > 0
        turn = float(np.sign(np.mean(pt[near, 0] * vel[near, 1] - pt[near, 1] * vel[near, 0])) or 1.0) if near.any() \
            else 1.0

        # ---- the bend, the shadowgraph, the caustics and the mist: on a coarse grid round the ball
        box = _clip_box(cx, cy, 1.7 * R, 1.7 * R, w, h)
        if box is not None:
            y0, y1, x0, x1 = box
            gx, gy = _coarse_grid(box, FIELD_Q)
            dx, dy = gx - cx, gy - cy
            phi = np.arctan2(dy, dx)
            wob = 1 + 0.035 * np.sin(3 * phi - 0.4 * self.spin + 1.3) + 0.025 * np.sin(5 * phi + 0.7 * self.spin)
            r = np.hypot(dx, dy) / (R * wob)
            z = np.sqrt(np.clip(1 - r * r, 0, 1))
            shell = (z - np.sqrt(np.clip(0.36 - r * r, 0, 1))) / 0.8  # the chord through the shell round a 0.6 core
            draft = np.where(r > 1, np.exp(-((r - 1) / 0.25) ** 2), 0.0)  # the swirl drags the air round it a little
            sw = (np.clip(shell, 0, 1) + 0.45 * draft) * vis
            lobes = 0.6 + 0.4 * np.cos(3 * (phi - turn * self.spin) + 1.2 * np.sin(2 * phi - 0.7 * turn * self.spin))
            theta = turn * BALL_TWIST * sw * lobes
            rim = np.exp(-((r - 0.97) / 0.06) ** 2)
            m = vis * (BALL_LENS * z - BALL_RIM * rim)  # the middle magnified, the rim pulled in from outside it
            sx, sy = dx * (1 - m), dy * (1 - m)
            ct, st = np.cos(-theta), np.sin(-theta)  # read from behind the flow: the plate is dragged with the spin
            rx, ry = sx * ct - sy * st, sx * st + sy * ct
            cs, sn = math.cos(-turn * self.spin), math.sin(-turn * self.spin)
            u = (dx * cs - dy * sn) / (0.5 * R) * 20 + 128
            v = (dx * sn + dy * cs) / (0.5 * R) * 20 + 128
            nz = cv2.remap(self.noise, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
            tb = BALL_TURB * k * sw
            bh, bw = y1 - y0, x1 - x0
            ddx = _upsample((rx + nz[..., 0] * tb - dx).astype(np.float32), bh, bw)
            ddy = _upsample((ry + nz[..., 1] * tb - dy).astype(np.float32), bh, bw)
            _remap_rgb(P, ddx, ddy, box, BALL_CA)
            # the shadowgraph: a bright band on the rim, broken into arcs that turn with the spin, darker inside it
            arc = cv2.remap(self.arcs, (((phi - turn * self.spin * 0.6) % (2 * np.pi)) * (512 / (2 * np.pi)))
                            .astype(np.float32), np.zeros_like(phi, np.float32), cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_WRAP)
            band = np.exp(-((r - 1.0) / 0.035) ** 2) * np.clip(0.75 + 0.45 * arc, 0.2, 1.6) \
                - 0.55 * np.exp(-((r - 0.88) / 0.07) ** 2)
            gain_ = BALL_SHADOW * vis * band
            if hand is not None:  # the palm under it takes the light the spinning air focuses, in moving patches
                hc = cv2.resize(hand[y0:y1, x0:x1], (gx.shape[1], gy.shape[0]), interpolation=cv2.INTER_AREA)
                gain_ = gain_ + BALL_CAUSTIC * vis * hc * (np.clip(nz[..., 2], 0, None) ** 2 - 0.4) * \
                    _smooth((1.15 - r) / 0.3)
            P[y0:y1, x0:x1] *= 1 + _upsample(gain_.astype(np.float32), bh, bw)[..., None]
            edge = _smooth((1 - r) / 0.12 + 0.25)  # a soft edge, not a rim
            mist_b = self._mist(dx / (R * wob), dy / (R * wob), -1, B) * edge * BALL_BACK
            if hand is not None:  # the far side of the shell passes behind the fingers
                mist_b = mist_b * (1 - 0.9 * hc)
            mist = BALL_MIST * vis * (self._mist(dx / (R * wob), dy / (R * wob), 1, B) * edge + mist_b)
            mist = cv2.GaussianBlur(_upsample(mist.astype(np.float32), bh, bw), (0, 0), 1.2 * k)
            col = np.minimum(tone * 1.2, 0.95).astype(np.float32)
            _over(P[y0:y1, x0:x1], mist, col, None if noise is None else noise[y0:y1, x0:x1], code)

        # ---- the streaks, dust and motes it carries
        x, y, z, wgt, kind = self.points(c, R, B, form, rel, vis, omega, t, k)
        x, y = x - ox, y - oy
        if hand is not None:  # the far side's points pass behind the fingers
            xi, yi = np.clip(x.astype(np.int64), 0, w - 1), np.clip(y.astype(np.int64), 0, h - 1)
            wgt = np.where(z < 0, wgt * (1 - 0.9 * hand[yi, xi]), wgt)
        bufs = [np.zeros((h, w), np.float32) for _ in range(5)]
        for j, buf in enumerate(bufs):
            sel = kind == j
            _splat(buf, x[sel], y[sel], wgt[sel])
        lit = _line_blur(bufs[0], STREAK_PX[0] * k) * STREAK_ALPHA + _line_blur(bufs[1], STREAK_PX[1] * k) * (
            0.8 * STREAK_ALPHA)
        dust_sig, mote_sig = 0.9 * k, 1.6 * k  # a point of weight 1, blurred, peaks at 1 / (2 pi sigma^2)
        lit += cv2.GaussianBlur(bufs[2], (0, 0), dust_sig) * np.float32(DUST_ALPHA * 2 * math.pi * dust_sig ** 2 / 1.6)
        lit += cv2.GaussianBlur(bufs[4], (0, 0), mote_sig) * np.float32(MOTE_ALPHA * 2 * math.pi * mote_sig ** 2)
        col_s = np.minimum(np.maximum(tone * 1.25, tone.max() * 0.55), 0.95).astype(np.float32)
        _over(P, lit, col_s, noise, code)
        grit = cv2.GaussianBlur(bufs[3], (0, 0), dust_sig) * np.float32(DUST_ALPHA * 2 * math.pi * dust_sig ** 2 / 1.6)
        _over(P, grit, (tone * 0.25).astype(np.float32), None, code)


def _ball_shot(plate: np.ndarray, out: np.ndarray, track: dict, mask: np.ndarray | None, tl: Timeline, seed: int,
               code: float, say) -> None:
    n, h, w = plate.shape[:3]
    level = voice_level(tl)
    lives = ball_lives(tl, level)
    xy, _, size = fx.track_arrays(track, n, w, h)
    xy, size = _gauss_smooth(xy, SMOOTH_XY), _gauss_smooth(size, SMOOTH_SIZE)
    normal = palm_normals(track, n, w, h)
    feed = fed(tl, level)
    k = min(h, w) / 1080.0
    total = sum(min(n, b + BALL_RELEASE) - a for a, b in lives)
    done = 0
    for j, (t0, t_rel) in enumerate(lives):
        ball = _Ball(seed, j)
        held, idle, tone_s = 0, 1.0, None
        for t in range(t0, min(n, t_rel + BALL_RELEASE)):
            held += int(feed[t])
            idle += (0.35 if feed[t] else 0.06) * (float(feed[t]) - idle)  # thins out in a silence, back at once
            grow = min(1.0, held / (BALL_GROW_S * fx.FPS))
            form = float(_smooth((t - t0 + 1) / BALL_FORM))
            rel = float(np.clip((t - t_rel + 1) / BALL_RELEASE, 0, 1))
            e = float(level[t])
            R0 = BALL_R * size[t] * (1 + BALL_GROW * grow) * (1 + BALL_BREATH * e)
            R = R0 * (0.3 + 0.7 * form) * (1 + 1.0 * rel ** 0.7)
            c = xy[t] + normal[t, :2] * (R0 + BALL_GAP * size[t])
            omega = 2 * np.pi * BALL_SPIN * (0.5 + 0.5 * form + BALL_SPIN_VOICE * e + BALL_SPIN_HELD * grow) / fx.FPS
            tone = _tone(plate[t])
            tone_s = tone if tone_s is None else 0.85 * tone_s + 0.15 * tone
            pad = BALL_PAD * R0 * (1 + rel) + 8 * k
            box = _clip_box(c[0], c[1], pad, pad, w, h)
            done += 1
            if box is None:
                ball.spin += omega
                continue
            y0, y1, x0, x1 = box
            P = lin(out[t, y0:y1, x0:x1])  # (over the ball before it, still bursting)
            hand = None if mask is None else mask[min(t, len(mask) - 1), y0:y1, x0:x1].astype(np.float32) / 255.0
            noise = _grain(seed, _BALL, t, y1 - y0, x1 - x0)
            ball.draw(P, (x0, y0), c, R, normal[t], form, rel, BALL_IDLE + (1 - BALL_IDLE) * idle, omega, t, hand,
                      tone_s, noise, code, k)
            out[t, y0:y1, x0:x1] = gam(P)
            if done % 8 == 0:
                say("air", done / total, f"Spinning the air over your palm: frame {done} of {total}")


# ---- the blast ------------------------------------------------------------------------------------------------------

def push_direction(track: dict, t: int, normal: np.ndarray, aim: float | None, tap_frame: int, n: int, w: int,
                   h: int) -> tuple[np.ndarray, float]:
    """(unit direction in the picture, share of the push across the picture: 1 across it, 0 straight at the lens)
    of a push out of the palm on frame ``t``. The direction is the studio's arrow when one is drawn (``aim``, turned
    with the hand since the tap, as cuesync.bending.anchored carries it; a track already anchored with it is
    unchanged), else the track's own, turned round when the palm we film faces clearly the other way across the
    picture (the track's handedness flips an upturned palm's direction); the share is how far the palm's normal
    leans across the picture (at least AIM_ACROSS with an arrow drawn)."""
    _, u, _ = fx.track_arrays(track, n, w, h)
    d = u[t]
    q = float(np.hypot(normal[0], normal[1]))
    if aim is not None:
        a = math.radians(aim)
        want = np.array([math.sin(a), -math.cos(a)])
        ref = u[int(np.clip(tap_frame, 0, n - 1))]
        turn = math.atan2(ref[0] * want[1] - ref[1] * want[0], ref @ want)
        c, s = math.cos(turn), math.sin(turn)
        return np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c]), max(q, AIM_ACROSS)
    if q > 0.35 and d @ normal[:2] < 0:
        d = normal[:2] / q
    return d / max(float(np.hypot(*d)), 1e-9), q


class _Blast:
    """One push of air from ``origin`` (frame px on its ``start`` frame) along ``u`` (``q``: how much of it crosses
    the picture, the rest coming at the lens), off a palm facing ``normal`` (palm_normals), from a hand ``size`` px,
    ``gain`` as loud as the loudest hit. All of it lives in the room as the start frame saw it: each frame is drawn
    through the camera's motion since. Points in it are 3D: x, y in px at the palm's depth, z in palm depths (the lens
    at 0)."""

    def __init__(self, origin, u, q: float, normal, size: float, w: int, h: int, start: int, gain: float, seed: int,
                 salt: int):
        rng = fx._rng(seed, _BLAST, salt)
        self.o = np.asarray(origin, np.float64)
        self.u = np.asarray(u, np.float64)
        self.q, self.size, self.start, self.gain = float(q), float(size), int(start), float(gain)
        self.w, self.h = w, h
        side = min(w, h)
        self.side, self.k = side, side / 1080.0
        lobes = np.arange(3, 9)
        self.lob, self.lph = lobes, rng.uniform(0, 2 * np.pi, lobes.size)
        self.lamp = rng.uniform(0.5, 1.0, lobes.size) / lobes
        self.reach = FRONT_TO + 0.6 * (1 - self.q)  # straight at the lens it fills the frame
        self.wake = cv2.GaussianBlur(_tile_noise(rng, 64, 720, 1.2, 1.2, 2), (0, 0), sigmaX=0.8, sigmaY=3.0)
        self.arcs = _tile_noise(rng, 1, 720, 6.0, 0.5).reshape(1, -1)  # where round the front its band is strong
        self.curl = _Curl(rng, 0.1 * side)
        f = FOCAL * side
        self.f = f
        self.pp = np.array([w / 2, h / 2])
        n3 = np.array([self.u[0] * self.q, self.u[1] * self.q, math.sqrt(max(0.0, 1 - self.q * self.q))])
        self.nrm = np.asarray(normal, np.float64) / max(float(np.linalg.norm(normal)), 1e-9)
        e1 = np.cross(self.nrm, [0.0, 0.0, 1.0] if abs(self.nrm[2]) < 0.9 else [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.nrm, e1)

        def thrown(d: np.ndarray, sp: np.ndarray, spread: float, depth: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
            """Points leaving the palm's face (a hand ``spread`` across, just off its skin) along unit directions
            ``d`` (x, y and towards the lens) at ``sp`` px a frame, keeping ``depth`` of their speed towards or away
            from the lens: their positions and velocities."""
            a, b = rng.standard_normal((2, len(d))) * spread * size
            off = a[:, None] * e1 + b[:, None] * e2 + 0.08 * size * self.nrm
            p_ = np.column_stack([self.o[0] + off[:, 0], self.o[1] + off[:, 1], 1 - off[:, 2] / f])
            return p_, np.column_stack([d[:, 0] * sp, d[:, 1] * sp, -d[:, 2] * sp / f * depth])  # (the lens: z falls)

        def ahead(d: np.ndarray) -> np.ndarray:
            """Unit directions spread round the push's own from random vectors ``d``, none of them back into it."""
            d = d / np.linalg.norm(d, axis=-1, keepdims=True)
            d = n3 * 0.8 + d
            d[np.sum(d * n3, -1) < 0] *= -1
            return d / np.linalg.norm(d, axis=-1, keepdims=True)

        # dust and flakes out of the palm
        N = DUST_N + FLAKES
        sp = rng.uniform(0.15, 1.0, N) ** 1.8 * 0.33 * side * gain  # px a frame at the palm's depth
        p, v = thrown(ahead(rng.standard_normal((N, 3))), sp, 0.4)
        delay = rng.uniform(0, 2.5, N).astype(int)
        r0 = np.where(np.arange(N) < DUST_N, rng.uniform(1.1, 2.4, N), rng.uniform(2.4, 4.0, N))
        bright = rng.uniform(0.3, 1.0, N) ** 1.5
        dark = rng.random(N) < np.where(np.arange(N) < DUST_N, 0.12, 0.4)
        flake = np.arange(N) >= DUST_N
        # the dusty air: parcels in clumps round a few directions, a third of them on their own, slower than the
        # fastest dust (the front runs ahead of the air it pushes)
        C = CLOUD_N
        lobe = ahead(rng.standard_normal((CLOUD_CLUMPS, 3)))[rng.integers(0, CLOUD_CLUMPS, C)]
        dc = np.where((rng.random(C) < 0.33)[:, None], ahead(rng.standard_normal((C, 3))),
                      ahead(lobe + 0.35 * rng.standard_normal((C, 3))))
        pc, vc = thrown(dc, (0.03 + 0.1 * rng.uniform(0, 1, C) ** 0.8) * side * gain, 0.3, CLOUD_DEPTH)
        # the room's own dust: anywhere in front of the wall, still (and so unseen) until the front reaches it
        rr = fx._rng(seed, _ROOM, salt)
        M = ROOM_N
        zr = rr.uniform(0.45, 2.5, M)
        scr = np.column_stack([rr.uniform(-0.1, 1.1, M) * w, rr.uniform(-0.1, 1.1, M) * h])
        pr = np.column_stack([self.pp + (scr - self.pp) * zr[:, None], zr])
        self.p = np.concatenate([p, pc, pr])
        self.v = np.concatenate([v, vc, np.zeros((M, 3))])
        self.delay = np.concatenate([delay, rng.uniform(0, 3.0, C).astype(int), np.zeros(M, int)])
        self.r0 = np.concatenate([r0, np.ones(C), rr.uniform(1.0, 2.4, M)])
        self.bright = np.concatenate([bright, rng.uniform(0.6, 1.4, C) * gain / C, rr.uniform(0.3, 1.0, M) ** 1.5])
        # (a parcel's bright is its share of the dusty air)
        self.dark = np.concatenate([dark, np.zeros(C, bool), rr.random(M) < 0.1])
        self.flake = np.concatenate([flake, np.zeros(C + M, bool)])
        self.cloud = np.concatenate([np.zeros(N, bool), np.ones(C, bool), np.zeros(M, bool)])
        self.room = np.arange(N + C + M) >= N + C
        self.pushed = ~self.room
        self.scr0 = np.concatenate([np.zeros((N + C, 2)), scr])
        self.tum = rng.uniform(0, 2 * np.pi, N + C + M)
        self.tumf = rng.uniform(0.25, 0.8, N + C + M)
        self.alive = np.ones(N + C + M, bool)

    def cone(self, ca: np.ndarray) -> np.ndarray:
        """How strong the push is at directions whose cosine with its aim is ``ca``: all round when it comes at the
        lens, ahead of the palm when it crosses the picture."""
        return (1 - self.q) + self.q * np.clip(ca, 0, 1) ** 1.5

    def front(self, kf: float) -> tuple[float, float, float]:
        """The front on its ``kf``-th frame: (radius px, bulge px, share of its run)."""
        uu = min(1.0, kf / BLAST_FRAMES)
        radius = self.side * (FRONT_FROM + (self.reach - FRONT_FROM) * (1 - (1 - uu) ** 2))
        strength = (BUBBLE_PX * self.k * self.gain * min(1.0, (kf + 1) / 2)
                    * (BUBBLE_END + (1 - BUBBLE_END) * (1 - uu) ** 1.5) * min(1.0, 3 * (1 - uu)))
        return radius, strength, uu

    def reaches_lens(self) -> int:
        """Frames after the push the bubble has grown past the frame's corners: the gust hits the phone."""
        far = max(np.hypot(*(self.o - np.array(c))) for c in ((0, 0), (self.w, 0), (0, self.h), (self.w, self.h)))
        return next((kf for kf in range(BLAST_FRAMES + 1) if self.front(kf)[0] >= 0.75 * far), BLAST_FRAMES)

    def last_frame(self) -> int:
        return self.start + DUST_LIFE[0] + DUST_LIFE[1]

    def behind(self, p: np.ndarray) -> np.ndarray:
        """Which points are behind the plane of the palm (so behind the hand, where its matte is)."""
        return (p[:, 0] - self.o[0]) * self.nrm[0] + (p[:, 1] - self.o[1]) * self.nrm[1] - (
            p[:, 2] - 1.0) * self.f * self.nrm[2] < 0

    def project(self, p: np.ndarray) -> np.ndarray:
        return self.pp + (p[:, :2] - self.pp) / np.clip(p[:, 2:3], 0.07, None)

    def bend(self, P: np.ndarray, kf: int, o: np.ndarray, u: np.ndarray, zoom: float) -> None:
        """The pressure bubble on frame ``kf`` of the push (``o``, ``u``, ``zoom``: its origin, aim and scale as the
        camera sees them now): what is seen through it pushed outward (a bulge sweeping the room), its front a thin
        band of bend and a shadowgraph, and behind it a turbulent wake, in place."""
        h, w = P.shape[:2]
        radius, strength, uu = self.front(kf)
        strength *= float(_smooth((radius - BUBBLE_CLEAR[0] * self.size) / (BUBBLE_CLEAR[1] * self.size)))
        radius, strength = radius * zoom, strength * zoom
        ox, oy, ux, uy = (float(v) for v in (*o, *u))  # (plain floats keep the fields float32)
        box = _clip_box(ox, oy, 1.35 * radius + 0.06 * self.side, 1.35 * radius + 0.06 * self.side, w, h)
        if box is None:
            return
        y0, y1, x0, x1 = box
        gx, gy = _coarse_grid(box, FIELD_Q)
        xs, ys = gx - ox, gy - oy
        dist = np.sqrt(xs * xs + ys * ys) + 1e-3
        nx, ny = xs / dist, ys / dist
        ang = np.arctan2(ys, xs)
        cone = self.cone(nx * ux + ny * uy).astype(np.float32)
        a1 = np.linspace(-np.pi, np.pi, 721)  # its outline wobbles: a few lobes, turning as it goes
        wob1 = 1 + 0.06 * (self.lamp[:, None] * np.sin(self.lob[:, None] * a1 + self.lph[:, None] + 0.3 * kf)).sum(0)
        ia = ((ang + np.pi) * (720 / (2 * np.pi))).astype(np.int32)
        rr = radius * wob1[ia].astype(np.float32)
        x = dist / rr
        width = FRONT_W * self.side * zoom * (1 + uu) / 2
        zz = (dist - rr) / width
        bulge = strength * cone * np.clip(x, 0, 1.2) ** 1.5 * np.exp(-0.5 * zz * zz * (zz > 0)) * (x < 1.25)
        # (inside: a smooth outward push growing to the front; outside it falls off within one front's width)
        behind = np.clip((rr - dist) / (0.45 * radius + 1), 0, 1)
        wake_amp = WAKE_PX * self.k * zoom * self.gain * (1 - uu) ** 0.6 * np.exp(-3 * behind) * (dist < rr) * cone
        wake_amp *= _smooth((dist - 0.6 * self.size * zoom) / (1.2 * self.size * zoom))  # (its fine grain near the
        # palm sawed the fingers' edges)
        mu = ((ang + np.pi) * (720 / (2 * np.pi))).astype(np.float32)
        mv = (dist / (0.06 * self.side * zoom) - 0.9 * kf).astype(np.float32) % 64
        wn = cv2.remap(self.wake, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        radial = bulge + wake_amp * wn[..., 0]
        across = 0.35 * wake_amp * wn[..., 1]
        bh, bw = y1 - y0, x1 - x0
        _remap_rgb(P, _upsample((-nx * radial - ny * across).astype(np.float32), bh, bw),
                   _upsample((-ny * radial + nx * across).astype(np.float32), bh, bw), box, BLAST_CA)
        arc = cv2.remap(self.arcs, mu, np.zeros_like(mu), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        sg = (1 - zz * zz) * np.exp(0.5 * (1 - zz * zz)) * np.clip(0.8 + 0.4 * arc, 0.3, 1.5)  # bright on the front,
        # dark either side of it, stronger and weaker round it
        rel = strength / (BUBBLE_PX * self.k * zoom * self.gain + 1e-6)
        P[y0:y1, x0:x1] *= 1 + _upsample((SHADOWGRAPH * self.gain * rel * cone * sg).astype(np.float32), bh, bw)[
            ..., None]

    def dust(self, kf: int, H: np.ndarray, H_prev: np.ndarray, lit: list, drk: list, scale: float, haze: np.ndarray,
             hand: np.ndarray | None) -> None:
        """Moves the dust and the dusty air on a frame (``kf``: frames since the push; ``H``: the start frame's px ->
        this frame's, ``H_prev`` the frame before's) and splats them: the dust into ``lit`` and ``drk`` (one buffer per
        softness, each ``scale`` times smaller than the frame for the softer ones), the dusty air into ``haze``
        (CLOUD_Q times smaller). What is behind the palm is hidden by the hand's matte ``hand`` (0..1, or None)."""
        side, k = self.side, self.k
        before = self.project(self.p)
        go = kf >= self.delay
        # the room's dust the front has just reached is thrown outward from the palm, harder ahead of the push
        radius, strength, uu = self.front(min(kf, BLAST_FRAMES))
        if kf <= BLAST_FRAMES:
            rel_ = self.scr0 - self.o
            dd = np.hypot(rel_[:, 0], rel_[:, 1]) + 1e-6
            hit = self.room & ~self.pushed & (dd <= radius)
            if hit.any():
                ca = (rel_[hit] @ self.u) / dd[hit]
                D = np.column_stack([self.p[hit, 0] - self.o[0], self.p[hit, 1] - self.o[1],
                                     (self.p[hit, 2] - 1.0) * self.f])
                D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-6)
                sp = ROOM_PUSH * side * self.gain * self.cone(ca) * (0.35 + 0.65 * (1 - uu)) * \
                    np.exp(-dd[hit] / (0.9 * side))
                self.v[hit] = np.column_stack([D[:, 0] * sp, D[:, 1] * sp, D[:, 2] * sp / self.f])
                self.pushed |= hit
        self.v[go] *= math.exp(-1 / 5.0)  # drag: out fast, slowing
        cx, cy = self.curl(self.p[:, 0], self.p[:, 1], kf)
        sp = np.hypot(self.v[:, 0], self.v[:, 1])
        turb = 0.012 * side * self.gain * (1 - math.exp(-kf / 4.0))  # the eddies take it as it slows
        stir = go & self.pushed
        self.v[stir, 0] += cx[stir] * turb * np.where(self.room[stir], 0.5, 1.0)
        self.v[stir, 1] += cy[stir] * turb * np.where(self.room[stir], 0.5, 1.0)
        self.v[stir, 1] += 0.0008 * side * ((sp[stir] < 0.015 * side) & ~self.cloud[stir])  # the heavier grit settles
        self.p[go] += self.v[go]
        self.alive &= self.p[:, 2] > 0.07
        after = self.project(self.p)

        def cam(M: np.ndarray, xy: np.ndarray) -> np.ndarray:
            return xy @ M[:2, :2].T + M[:2, 2]

        before, after = cam(H_prev, before), cam(H, after)
        zz = np.clip(self.p[:, 2], 0.07, None)
        a0 = after + (before - after) * 0.5  # the shutter: half a frame of motion
        disp = np.hypot(*(after - a0).T)
        seen = np.ones(len(zz))
        if hand is not None:  # behind the palm, hidden by the hand
            hh, ww = hand.shape
            back = self.behind(self.p)
            xi = np.clip(after[back, 0].astype(np.int64), 0, ww - 1)
            yi = np.clip(after[back, 1].astype(np.int64), 0, hh - 1)
            seen[back] = 1 - hand[yi, xi]
        # the dusty air, smeared along its half frame of motion (the fast parcels draw the gust's streaks)
        cl = self.cloud & self.alive & go
        life_c = float(np.clip(1 - (kf - CLOUD_FADE[0]) / CLOUD_FADE[1], 0, 1) * _smooth((kf + 1) / 3))
        if cl.any() and life_c > 0:
            S = 6
            ss = np.linspace(0, 1, S)[None, :]
            X = (a0[cl, 0:1] + (after[cl, 0:1] - a0[cl, 0:1]) * ss + 0.5) / CLOUD_Q - 0.5
            Y = (a0[cl, 1:2] + (after[cl, 1:2] - a0[cl, 1:2]) * ss + 0.5) / CLOUD_Q - 0.5
            wv = self.bright[cl] * life_c * seen[cl] / np.clip(zz[cl], 0.5, None)  # (nearer the lens it spreads over
            # more of the picture, and thins less than its spread)
            _splat(haze, X.ravel(), Y.ravel(), np.repeat(wv / S, S))
        bright = self.bright * np.clip(disp / (3 * k), 0.4, 1.0)  # dust shows most as it moves
        bright = np.where(self.room, self.bright * np.where(self.pushed, np.clip(ROOM_SEEN + disp / (2.5 * k), 0, 1),
                                                            0.0), bright)  # the room's: unseen until the front
        # throws it, then catching the light as it hangs and drifts
        bright = np.where(self.flake, bright * (0.2 + 0.8 * np.abs(np.cos(self.tum + self.tumf * kf)) ** 3), bright)
        life = float(np.clip(1 - (kf - DUST_LIFE[0]) / DUST_LIFE[1], 0, 1) * _smooth(kf / 1.5))
        if life <= 0:
            return
        rad = self.r0 * k / zz  # bigger near the lens ...
        soft = np.maximum(rad, 0.8 * k) + 10 * k * np.clip((0.4 - zz) / 0.33, 0, 1) ** 1.5  # ... and out of focus
        S = 10
        ss = np.linspace(0, 1, S)[None, :]
        for j, (lo, hi) in enumerate(((0, 1.6), (1.6, 4), (4, 9), (9, 1e9))):
            sel = self.alive & go & ~self.cloud & (soft >= lo * k) & (soft < hi * k) & (bright > 0)
            if not sel.any():
                continue
            q = 1.0 if j < 2 else scale  # (a pixel's middle lands on the middle of the coarse pixel holding it)
            X = (a0[sel, 0:1] + (after[sel, 0:1] - a0[sel, 0:1]) * ss + 0.5) / q - 0.5
            Y = (a0[sel, 1:2] + (after[sel, 1:2] - a0[sel, 1:2]) * ss + 0.5) / q - 0.5
            L = np.maximum(disp[sel], 1.0)
            # each grain's light spread over its smear: a fast one is a fainter streak, never a brighter one
            wv = bright[sel] * life * seen[sel] * np.minimum(1.0, (rad[sel] / k) ** 2) * np.minimum(1.0, 2.0 / L) ** 0.5
            wv = wv * np.where(self.flake[sel], 2.5, 1.0) / (q * q)
            Wt = np.repeat((wv / S)[:, None], S, 1)
            dk = np.repeat(self.dark[sel][:, None], S, 1)
            _splat(lit[j], X[~dk], Y[~dk], Wt[~dk])
            _splat(drk[j], X[dk], Y[dk], Wt[dk])


DUST_SIGMA = (0.8, 2.8, 6.5, 13.5)  # px at 1080: how soft each class of dust is drawn (sharp ... out of focus)


def _blast_shot(plate: np.ndarray, out: np.ndarray, track: dict, mask: np.ndarray | None, tl: Timeline, seed: int,
                code: float, say) -> None:
    from .bending import camera_between

    n, h, w = plate.shape[:3]
    xy, _, size = fx.track_arrays(track, n, w, h)
    normal = palm_normals(track, n, w, h)
    hits = [f for f in tl.hits if 0 <= f < n] or ([tl.tap_frame] if 0 <= tl.tap_frame < n else [])
    sizes = [tl.sizes[i] for i, f in enumerate(tl.hits) if 0 <= f < n] if tl.hits else [1.0] * len(hits)
    blasts = []
    for j, (f, sz) in enumerate(zip(hits, sizes)):
        u, q = push_direction(track, f, normal[f], tl.aim, tl.tap_frame, n, w, h)
        gain = 0.55 + 0.45 * float(np.clip(sz, 0, 1))  # a "pf" pushes three quarters as hard as the loudest hit
        blasts.append(_Blast(xy[f], u, q, normal[f], size[f], w, h, f, gain, seed, j))
    if not blasts:
        return
    cam = tl.camera
    k = min(h, w) / 1080.0
    sc = 4  # the softer dust is drawn this many times smaller
    hq = (-(-h // CLOUD_Q), -(-w // CLOUD_Q))
    first, last = min(b.start for b in blasts), min(n - 1, max(b.last_frame() for b in blasts))
    buffets = []
    tone_s = None
    for t in range(first, last + 1):
        live = [b for b in blasts if b.start <= t <= b.last_frame()]
        if not live:
            continue
        P = lin(plate[t])
        tone = _tone(plate[t])
        tone_s = tone if tone_s is None else 0.85 * tone_s + 0.15 * tone
        hand = None if mask is None else mask[min(t, len(mask) - 1)].astype(np.float32) / 255.0
        lit = [np.zeros((h, w), np.float32) for _ in range(2)] + [
            np.zeros((-(-h // sc), -(-w // sc)), np.float32) for _ in range(2)]
        drk = [np.zeros_like(b) for b in lit]
        alpha, dark = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
        for b in live:
            kf = t - b.start
            H = camera_between(cam, b.start, t)
            Hp = camera_between(cam, b.start, max(b.start, t - 1))
            o = H[:2, :2] @ b.o + H[:2, 2]
            zoom = math.sqrt(abs(np.linalg.det(H[:2, :2])))
            u = H[:2, :2] @ b.u
            u = u / max(float(np.hypot(*u)), 1e-9)
            if kf <= BLAST_FRAMES:
                b.bend(P, kf, o, u, zoom)
            haze = np.zeros(hq, np.float32)
            b.dust(kf, H, Hp, lit, drk, sc, haze, hand)
            if haze.any():  # the dusty air: diffusing as it goes, as thick as its parcels are close together
                soft = (CLOUD_SOFT[0] + CLOUD_SOFT[1] * kf) * k * zoom / CLOUD_Q
                dens = cv2.GaussianBlur(haze, (0, 0), soft) * np.float32((min(h, w) / CLOUD_Q) ** 2)
                a_ = cv2.resize(CLOUD_ALPHA * (1 - np.exp(-CLOUD_THICK * dens)), (w, h), interpolation=cv2.INTER_LINEAR)
                alpha += a_
            if kf == b.reaches_lens():
                buffets.append((t, o, b.gain * (1 - b.q)))
        col = np.minimum(tone_s * 1.15, 0.95).astype(np.float32)
        for j, sig in enumerate(DUST_SIGMA):
            q = 1 if j < 2 else sc
            for buf, into in ((lit[j], alpha), (drk[j], dark)):
                if buf.any():
                    gain_ = 12.0 * (0.7 / sig) ** 0.6  # an out-of-focus grain is bigger and fainter
                    b_ = cv2.GaussianBlur(buf, (0, 0), max(0.5, sig * k / q)) * np.float32(gain_)
                    into += b_ if q == 1 else cv2.resize(b_, (w, h), interpolation=cv2.INTER_LINEAR)
        box = fx._box(alpha > 2e-3, 2)
        if box is not None:
            y0, y1, x0, x1 = box
            _over(P[y0:y1, x0:x1], np.clip(alpha[y0:y1, x0:x1], 0, 0.85), col,
                  _grain(seed, _BLAST, t, y1 - y0, x1 - x0), code)
        box = fx._box(dark > 2e-3, 2)
        if box is not None:
            y0, y1, x0, x1 = box
            _over(P[y0:y1, x0:x1], np.clip(dark[y0:y1, x0:x1], 0, 0.7), (tone_s * 0.22).astype(np.float32), None, code)
        out[t] = gam(P)
        if (t - first) % 8 == 0:
            say("air", (t - first) / max(1, last + 1 - first), f"Pushing the air out: frame {t - first + 1} of "
                                                                 f"{last + 1 - first}")
    for t, o, strength in buffets:
        buffet(out, t, o, strength)


def buffet(frames_u8: np.ndarray, at: int, centre, strength: float) -> None:
    """The gust reaching the phone, in place: the picture swells round the push and settles, smeared outward on the
    frames it hits (a zoom blur: the phone knocked back). ``strength`` 0..1: how much of a full push came at the
    lens."""
    if strength <= 0.05:
        return
    n, h, w = frames_u8.shape[:3]
    for d, a in {0: 0.4, 1: 1.0, 2: 0.65, 3: 0.3, 4: 0.1}.items():
        f = at + d
        if not 0 <= f < n:
            continue
        s = 1 + BUFFET * strength * a
        taps = 7 if d in (0, 1) else 1
        acc = np.zeros((h, w, 3), np.float32)
        for j in range(taps):
            sj = 1 + (s - 1) * (0.4 + 0.6 * j / max(taps - 1, 1)) if taps > 1 else s
            M = np.array([[sj, 0, centre[0] * (1 - sj)], [0, sj, centre[1] * (1 - sj)]], np.float32)
            acc += cv2.warpAffine(frames_u8[f], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frames_u8[f] = np.clip(acc / taps + 0.5, 0, 255).astype(np.uint8)


# ---- the contract -----------------------------------------------------------------------------------------------------

def render(plate: np.ndarray, track: dict, mask: np.ndarray | None, timeline: Timeline, form: str, seed: int = 0,
           progress=None) -> np.ndarray:
    """The shot's frames, uint8 RGB [T, H, W, 3]: ``plate`` (uint8 RGB, as displayed) with air made in it, in
    ``form`` "ball" (a spinning ball over the palm, held by the voice) or "blast" (a push out of the palm on each of
    the voice's hits). ``track``: the hand's track, its emitter carried to the tap (cuesync.bending.anchored);
    ``mask``: the hand's matte [T, H, W] uint8, or None; ``progress(stage, share, message)``. Frames the air never
    reaches are the plate's, byte for byte."""
    if form not in FORMS["air"]:
        raise ValueError(f"air has no form {form!r}; one of {FORMS['air']}")
    plate = np.asarray(plate)
    n, h, w = plate.shape[:3]
    if timeline.n != n:
        raise ValueError(f"the timeline has {timeline.n} frames, the plate {n}")
    say = progress or (lambda stage, share, msg: None)
    out = plate.copy()
    code = fx.shadow_noise(plate) / fx._unit_grain_sigma() / 255.0
    (_ball_shot if form == "ball" else _blast_shot)(plate, out, track, mask, timeline, seed, code, say)
    say("air", 1.0, "The air is made")
    return out
