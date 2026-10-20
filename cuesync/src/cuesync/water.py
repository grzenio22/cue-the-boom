"""Water bent from an open palm, made rather than rendered (cuesync.elements): what is behind it, seen through it.

Water throws no light, so nothing here is added the way fire is: it is a lens and a mirror over the plate. A frame's
water is a set of Gaussian metaballs (a lone one is an exact sphere of radius R, two near ones bridge the way water
does), read as a half-thickness map. Each pixel's camera ray is traced through it: bent in at the front surface, across,
out at the back (Snell both ways, with a little dispersion), then on to what is behind: the hand just behind it, at
the depth its landmarks give it, or the room far behind, where the plate is read, averaged over as much of it as the
water squeezes into the pixel. Over that: the room reflected (Fresnel), the key light's glints, aeration where the water
is thin or fast, the plate's own grain. A small drop is shaded as one: the plate right behind it, a thin edge and a
bright glint. Then the water is blurred along its motion over the exposure, pixel by pixel. Under it, on the hand: the
light the water focuses, inside the shadow it casts; and where it has been, the skin or the wall is left wet.

What makes it read as water and not glass is that it never holds still: its outline wobbles in its low modes (driven
by the hand's motion and the voice), small ripples travel over it and its glint breaks and stretches with them, the
water circulates inside it (seen as the room through it swirling, and bubbles carried round and rising), it drips from
its underside, and it froths where it is thin and fast.

Two forms (cuesync.elements.FORMS["water"]):

- ``orb``: water gathers out of the air into a ball floating over the open palm, sized to the hand, wobbling and
  circulating, carried on the hand with the lag a mass of water has. It is held while the voice holds, and each hit in
  it jolts it. When the voice ends with a hit, or the hand thrusts, it is thrown along the aim and into the room, and
  bursts flat on the room's back wall into a splash, leaving a wet patch and drips running down; a voice that just
  stops lets it fall onto the palm and splash off it.
- ``whip``: a stream of water rising out of a churning pool on the palm, that swirls round the hand and trails it like
  a ribbon when it moves (each part of it held to the swirl by a spring, stiff at the root and soft at the tail),
  swelling and thinning as bulges run down it, rippled, thinning to its tail, where it frays into beads. It pours out
  with the voice, each hit lashes it out along the aim, blurred by its speed, frothing and shedding drops, and when the
  voice ends it breaks into beads that fall.

The orb floats over the palm, not in it: it sits along the palm's normal, which is taken to face the camera (the
tracker's handedness, and with it the normal's sign, is unreliable for an upturned palm), and the hand hides water only
where the hand is in front of it, by depth (the palm's plane through its landmarks, the fingers off it), not wherever
its matte is.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np

from . import fx_light as fx
from .bending import emitter_at
from .elements import FORMS, Timeline
from .transfer import gam, lin

FPS = 24
_ORB, _WHIP, _NOISE, _GRAIN = 1, 2, 3, 4  # mixed into each seed, so one seed gives each its own randomness

# ---- the optics -----------------------------------------------------------------------------------------------------
ETA = (1.331, 1.336, 1.342)  # water's index for R, G, B: the room through it fringes faintly (a little exaggerated: at
# the true spread the fringe is under a pixel)
F0 = 0.02  # water's reflectance head on (Fresnel, Schlick)
TAU = 0.5  # the water's surface is where the blobs' summed field is this
LN_TAU = math.log(1.0 / TAU)
SIG_K = 1.0 / math.sqrt(2.0 * LN_TAU)  # a blob's sigma per unit radius: a lone blob is a sphere of radius R
REACH = 3.0  # sigmas: a blob's field is laid this far out, tapering to nothing over its last fifth (cut off without the
# taper, a neighbour's surface stepped where it crossed the edge; tapering from 1.9 sigmas, the blobs of a stream summed
# unevenly at its outline, rippled at their spacing: ribs across it)
ABSORB = np.array([0.30, 0.10, 0.07], np.float32)  # per hand size of path: water is faintly cyan where it is thick
LENS = 0.5  # how far a ray turns per unit of the surface's slope, through both faces (a ball lens: 2 (n - 1) / n);
# it says how much of the plate one pixel of water sees
ROOM_DEPTH = 0.55  # shares of the frame's long side: the room behind the water (a ray turned by an angle sees about
# what the camera sees in that direction)...
LENS_FULL, LENS_MIN = 0.25, 0.15  # ... seen that far through water this thick (radius, hand sizes) and over, and
# through the smallest at this share of it (a drop's little ripples, carried a room away, crumpled what a splash's
# pieces showed into glass and ribbed a stream like a hose)
NEAR = (0.08, 1.5)  # hand sizes: the hand behind the water is taken at least / at most this far behind its back...
NEAR_OPEN = 0.6  # ... and this far where no hand is right behind it (a ray turned onto the hand from beside it)
DEFOCUS = 0.5  # a magnified patch of the plate is blurred by this share of its magnification: a ball lens is never sharp
# on what is right behind it (read point by point it showed the video's blocks, magnified)
BASE_BLUR = 0.04  # of the water's local radius: the least the plate is blurred through it
LOD_MAX = 4  # pyramid levels the plate is read through (16 times smaller: the widest squeeze the water makes)
RIM_DARK = 0.3  # grazing rays see the room far round the side, through little water: the rim is this much darker
# (at the prototype's 0.55 the outline read as a glass marble's)
TIR_DIM = 0.5  # where a ray is turned back inside the water (total internal reflection), what it shows is this dim
SMALL_EDGE = 0.4  # how much of that rim darkening and dimming a stream or a drop keeps (all of it drew a stream's
# outline as a tube's wall)...
SMALL_RIM = 0.08  # ... and under this radius (hand sizes) less again, down to none for the smallest
TINY = (0.025, 0.05)  # hand sizes: a drop under the first radius is shaded as a small drop, over the second as water
# with a body; a small drop shows...
TINY_SEES = 0.6  # ... what is this many of its radii behind it, a little magnified...
TINY_EDGE, TINY_CORE = 0.25, 0.2  # ... inside a thin darker edge, its core this much darker round its glint...
TINY_MIRROR = 0.5  # ... and mirrors the room this much less
TIR_SOFT = 0.08  # Snell's discriminant over which a ray goes from leaving the water to being turned back in it
# (switched at once, the line between the two was a jagged stair inside the rim)
CALM = (0.5, 3.0)  # what is seen through water moving this many hand sizes a frame is bent half as far, and through a
# stream curved this much (its curvature times its radius) half as far again
FAR_FAST = 0.15  # water faster than WHIP_SHED shows the room through it from this share as far behind it
GLINT = (0.8, 0.07)  # the key light mirrored in the surface: its window (at its brightest, short of white: at 1.1 it
# was a white sticker), and a soft sheen round it...
GLINT_POW = 12.0  # ... how tight the sheen is...
WINDOW = (0.4, 0.26, 0.3, 0.25)  # ... and the window's half width and height (in the mirrored ray's direction, across
# and along the light's), how soft its edge is (of its size, either side), and the share of the finest waves (SPARKLE)
# it is mirrored in: mirrored in the surface's wobble and waves, it stretches and breaks up as the water moves (in a
# calm surface it was a sticker; in all of the finest waves, a crumpled white splat)
BACK = (0.3, 0.55)  # the window again, off the inside of the far wall: how bright, how big
GLINT_BREAK = (1.6, 0.05)  # along a stream its glints are broken by a noise this strong, its features this long (hand
# sizes)
CRESCENT = 0.45  # the key light gathered by the water onto its far inside edge, opposite the light
INNER_GLOW = 0.06  # light bent and gathered inside the body: a faint moving net of brighter lines where it is thick
KEY_ELEV = 0.9  # the key light's direction out of the picture, towards the camera, per unit across it
KEY_UP = 0.35  # the key light is taken this much from above (a room's lights are mostly over the camera) and the rest
# from where the plate is brightest (read alone, a window behind the hand lit the orb from below)
KEY_SHARE = 0.015  # the brightest share of the room read as its key light...
KEY_SMOOTH = 6.0  # ... held steady over this many frames (sigma): read on each frame alone, a glint jumped when a lamp
# came into view on a walk
ENV_TOP, ENV_Q, ENV_BLUR = 0.4, 4, 1.5  # the room the water reflects: the plate's top share, this many times
# smaller and blurred by this (at 16 times smaller and blurred again, its rim was a flat grey band)
ENV_FLOOR = 0.5  # what it mirrors from below level is this much darker
KEY_Q = 16  # the key light is found on the plate this many times smaller
F_SMALL = 0.25  # the most a small piece of water mirrors (under a tenth of a hand: it is all grazing surface, and at a
# glancing ray's full reflectance a splash's pieces were mercury)
DEPTH_SOFT = 0.08  # hand sizes: where the water passes behind the hand, it goes from seen to hidden over this depth
EDGE_SOFT = 0.5  # px: the camera's own softness on the water's outline (a cut-out edge was one pixel sharp)
FOAM_ALBEDO = 1.1  # froth is white matter lit by the room: this share of its key light comes back (at 0.75 it was
# no brighter than lit skin, and froth over the palm read as a grey haze)
FOAM_MAX = 0.8  # froth hides at most this much of what is seen through the water
FOAM_GLINT = (0.8, 0.5)  # the bubbles it is made of glint: how bright (of the key light), how small (of its bubbles)
FOAM_STREAK = (0.15, 0.3, 0.1)  # along a stream it lies in streaks running with it: this long (hand sizes), this wide
# (of its radius), and this much of it between them
RIPPLE = (0.2, 0.06, 1.0)  # a stream's surface is broken by ripples running out along it: their height (log-field
# units), their length (hand sizes) and width (of its radius)
DOT = (0.05, 1.6, 0.7, 0.22)  # a free drop under this radius (hand sizes) glints as one bright point: how bright (of
# the key light), at least how wide (px, sigma) and how wide for its radius
DOT_MIN = 0.5  # px: the smallest drop that glints
BLUR_STEP, BLUR_TAPS = 1.0, 64  # px between a moving pixel's taps along its path over the exposure, and at most this
# many taps

# ---- motion and imperfection ----------------------------------------------------------------------------------------
WAVE = (0.025, 0.04)  # log-field units: the surface's small waves at rest, and more at the voice's loudest (about 2 %
# and 5 % of the radius at the outline); without them the outline was a compass circle and the orb read as glass, at
# twice this what was seen through it folded like crumpled cellophane
WAVE_SIZE = 0.2  # hand sizes: the size of those waves
WAVE_EDGE = 0.3  # their share at the outline: they move the inside's surface, and the outline wobbles in its low modes
# (moved as much as the inside, the orb's outline was a lumpy, rounded square)
SPARKLE = 0.1  # slope: finer, faster waves that break up the edge of the key light's window, as water does
SPARKLE_SIZE = 0.035  # hand sizes
SPARKLE_BENDS = 0.15  # the share of that slope that bends what is seen through the water (all of it crumpled the room
# seen through the orb like cellophane: its glints break up, the lens stays whole)
INNER = 0.022  # slope: the water circulating inside turns what is seen through it this much, where it is thick
# (the room seen through it, a frame's height behind, moves about 20 px for it: at 0.035 it swirled into mush)
INNER_SIZE = 0.15  # hand sizes
FOAM_SIZE = 0.022  # hand sizes: the bubbles froth is made of
FLOW_CYCLE = 12  # frames: the water's patterns drift with its flow this long, then cross-fade into fresh ones (a pattern
# carried round for good winds itself into streaks)
SHUTTER = 0.5  # frames: the exposure (a phone indoors, 1/50 s), over which a moving drop smears into a streak
SMEAR_MAX = 6.0  # a piece's travel over the exposure is taken as at most this many times its size (a speck smeared over
# more was too faint to see)
G = 60.0  # hand sizes a second squared: gravity (a hand is about 9 cm: true gravity is 110; at that a splash left the
# frame before it read)
DRAG = 0.985  # per frame: what a flying drop keeps of its speed
ZCAM = 4.5  # hand sizes: how far the camera is from the hand (a first-person shot); a thrown orb shrinks as it goes
NOISE_SIZE, NOISE_CELL = 256, 4.0  # px: the tiling noise textures, and their features' sigma

# ---- the orb --------------------------------------------------------------------------------------------------------
ORB_R = 0.45  # hand sizes: its radius (a ball 8 cm across over a palm 9 cm long; at 6 cm it was a fifth of a phone
# screen's width, a marble)
ORB_GAP = 0.25  # hand sizes between its underside and the palm: it floats (at 0.1 its shadow and its focus sat under
# its own rim, and it read as set into the palm)
ORB_SPRING = (2.4, 0.45)  # Hz, damping: it follows the palm as a mass of water held on a spring would, trailing and
# overshooting
ORB_BOB = (0.025, 0.8)  # hand sizes, Hz: its slow float up and down
GATHER, GATHER_MIN, GATHER_LEAD = 12, 4, 2  # frames: gathering takes this long, at least this long squeezed in front of
# an early throw, and is done this long before a throw it was gathered for
GATHER_DROPS = 22  # drops that come together out of the air into it
GATHER_FROM = (1.3, 2.2)  # radii: how far out they start (from 3, they flew in so fast they were glass needles)...
GATHER_R = (0.03, 0.07)  # ... and how big they are (at up to a fifth of its radius, a cluster of clear balloons
# fusing)
MODE_HZ = {2: 1.6, 3: 3.1, 4: 4.8}  # its wobbles: two, three and four lobes round it (a drop's ratios, 1 : 1.9 : 3)
MODE_AMP = {3: 0.03, 4: 0.012}  # of the radius, at rest the third of it and at the voice's loudest all of it (the
# four lobes at 0.022 squared its outline)
MODE_REST = 0.35
MODE2_DRIVE, MODE2_DAMP, MODE2_MAX = 0.006, 0.25, 0.35  # the two-lobe stretch, driven by its acceleration (inertia
# squashes it), its damping, and the most it stretches
CIRCULATE = (1.9, 1.5)  # rad a second: it turns over inside this fast at rest (0.3 turns a second: slower, a held orb
# was a still lens), and more at the voice's loudest
BUBBLES = 26  # air caught inside, carried round by the circulation...
BUBBLE_R = (0.006, 0.022)  # ... this big (hand sizes)...
BUBBLE_RISE = 0.25  # ... and rising through it, this many turns of its height a second
DRIP_EVERY = (26, 60)  # frames between drips off its underside
DRIP_FORM, DRIP_R = 7, 0.035  # frames a drip takes to swell, neck and let go, and its radius (hand sizes: at a sixth
# of the orb's, a glass tulip bead hanging off it)
JOLT = (0.18, 4)  # a hit while it is held: how hard it stretches it, and how many drops it spits per unit of the hit
RING = (0.025, 0.5, 0.25, 6.0)  # the ripple a hit sends over it: log-field height, wavelength and speed (radii, a
# frame) and how long it lasts (frames)...
RING_FROM = (0.6, 0.9)  # ... struck this far out from its middle as seen (struck near the middle, its rings were a
# target drawn on the ball)
THROW_SPEED = 18.0  # hand sizes a second, at the loudest hit (0.6 of it at the quietest): at 11 gravity took back
# most of its rise and it burst a hand over the fingers
TAKEOFF = 2  # frames it takes to get up to speed (it jumped off the palm in one)
RECEDE = 1.0  # of the throw's speed: thrown along the aim it also goes away from the camera, into the room, until it
# hits the room's back wall (ROOM_DEPTH behind the palm) and bursts flat on it (at half this it arced up and fell back
# most of the way to the hand before it got there)
THROWN_STRETCH = 0.35  # the most it stretches along its flight (its motion blur besides; stretched 2 : 1, it was a
# sharp glass capsule)
NECK = (4, 3, 0.35)  # frames a neck pulled off its back takes to pinch off, the drops it leaves, their radius (of the
# neck's)
THRUST_SIZE = 0.8  # how hard a thrust with no hit throws it
DROP_SIZE = 0.35  # how hard it splashes when let fall onto the palm
LINGER = 6  # frames it is held on after the voice stops before it falls
THROW_TAIL = 24  # frames: the last hit throws it when the voice falls silent within this after it; a voice holding on
# longer was holding the orb, not ending on the hit (a walk's long breath, surging once near its start, threw the orb
# away a second in and left the rest of the walk empty)
SPLASH = (60, 8, 110, 3)  # drops, ligaments, specks of spray and big lumps a burst at full strength makes (45 broad
# sheet pieces flung out at once overlapped into a flat cellophane star; 130 drops of one size were a window pane)...
SPLASH_R = ((0.02, 0.13), (0.007, 0.018), (0.15, 0.25))  # ... their radii (of the ball's): the drops' range (a power
# law: about a dozen over 0.06, the rest smaller), the spray's and the lumps'
SPLASH_SPEED = (3.0, 12.0)  # hand sizes a second: how fast its drops leave (slower, the burst was a puff a hand across)
SPLASH_CONE = 1.0  # they fly on along its motion, spread over a cone this wide (1: about 45 degrees)...
SPLASH_CARRY = 0.3  # ... keeping this much of its speed
FAN = 0.6  # on the wall they fan out the way it was thrown, as if it had come in this much along the wall (of its speed
# into it)
SPRAY_LIFE = 7  # frames the fine spray lasts
SPLAT = (1.3, 0.3, 0.1, ((6, 0.06, 0.0), (9, 0.035, 1.3)))  # burst on a surface, its first frame is a splat
# flattened against it: its radius (of the ball's), its thickness (of a ball's), froth, and the lobes of the crown round
# its edge (wrinkled in three and five lobes and frothing more, it was a crumpled plastic bag)
SHEET = (4, 0.25, 0.6)  # on the wall it spreads on as a thinning sheet for this many frames, this much wider (of the
# ball's radius) a frame, keeping this much of its thickness a frame
WET = (0.1, 1.2, 0.05)  # the wet patch it leaves on the wall: how much darker, its radius (of the ball's there), and
# its faint gloss (of the key light)
WALL_DRIPS = (3, 0.09, 1.5, 0.25)  # drips running down the wall from it: how many, their radius (of the ball's), how
# fast at most (hand sizes a second, at the wall) and how soon they get to it (seconds)
BURST_SWELL = (1.12, 0.3, ((3, 0.05), (5, 0.025)))  # the burst's first frame: its body swollen to this share of its
# radius, frothing this much, wobbling in these lobes (of its radius) (spread into a ring of pieces with its torn skin
# inside, it was a flat lid stuck in the air; wrinkled in five to nine lobes, a cellophane flower; frothing more, a
# frosted disc)
LIGAMENT_R, LIGAMENT_STRETCH = (0.05, 0.08), (2.5, 3.5)  # its torn skin's ligaments: radius (of the ball's), and how
# many times longer than thick, besides their motion blur...
LIGAMENT_LIFE = 2  # ... and the frames each lasts before it has beaded up into drops

# ---- the whip -------------------------------------------------------------------------------------------------------
WHIP_NODES = 40  # the stream's nodes, root to tail
WHIP_ORBIT = 1.1  # hand sizes: the stream swirls round the hand this far out...
WHIP_SWEEP = (0.55, 0.8)  # ... wrapping this much of a turn round it, quiet and at the voice's loudest
WHIP_RISE = 0.3  # of its length: it rises out of the palm over this much of it before it swings out to that circle
WHIP_FACE = 0.7  # its circle faces the camera this much and the palm the rest (round the palm's own normal, an
# upturned palm's circle was seen edge on, a slug on the palm)
WHIP_TILT = 35.0  # degrees its circle is tipped about the hand's length, so one side passes behind the hand
WHIP_SPIN = (0.45, 0.5)  # turns a second it swirls at, quiet, and more at the voice's loudest
WHIP_WOBBLE = (0.14, 3.0, 1.3)  # its swirl's lobes: how far in and out (of its circle), how many, turns a second
WHIP_HZ = (3.0, 0.9)  # how stiffly it holds its swirl at the root and at the tail: the tail trails a moving hand,
# as a ribbon does (one stiffness throughout and the stream moved with the hand as one piece, a hoop)
WHIP_DAMP = 0.55  # their damping ratio
WHIP_R = (0.09, 0.03)  # hand sizes: the stream's radius at its root and at its tail (blobs weighted, as thick as
# designed: summed unweighted, 0.1 drew a hose)
WHIP_TIP = 0.35  # its last fifth thins by this much to its end
WHIP_SPACING = 0.6  # its blobs are at most this share of their radius apart
WHIP_POOL = (1.8, 0.3, 0.55, 0.16)  # the pool on the palm it rises out of: its radius (of the stream's at its root),
# its thickness (of a ball's), its froth, and how far its outline ripples (of its radius); a ball of water there was a
# test tube's closed end
WHIP_FLARE = (0.6, 0.1)  # the stream flares out this much wider where it leaves the pool, narrowing over this share of
# its length
WHIP_WET = 1.3  # the palm round the pool is wet this far out (of the pool's radius)
WHIP_LIFT = 0.06  # hand sizes it starts over the palm
WHIP_CRACK = (8, 3.6, 0.45, 2.5)  # a hit cracks it: frames, how far out along the aim (hand sizes), the kink that runs
# down it (hand sizes across), and how much stiffer it is while it lashes out
WHIP_SHED, WHIP_SHED_RATE = 9.0, 0.12  # hand sizes a second: where it is faster than this it sheds drops, each point
# of it past WHIP_SHED_FROM of its length at this chance a frame per time over it...
WHIP_SHED_FROM = 0.3  # (shed near the root, they wound round it like a screw's thread)
WHIP_SPRAY = (1, 4)  # fine drops its tail sprays a frame while it moves
WHIP_FOAM = (0.22, 0.3, 0.5, 0.45)  # how aerated it is all along, more at its tail, more where it is fast, and more at
# its root where it churns out of the pool (clear all along, a glass rod)
WHIP_AIR, WHIP_AIR_R, WHIP_AIR_SEEN = 10, (0.12, 0.28), 0.45  # bubbles carried along it, their size (of its radius),
# and how clearly they are seen (at 0.7, rings like glass beads)
WHIP_SAG = 0.12  # of gravity, while it is held up
WHIP_WAVES = ((0.22, 2.3, 0.9), (0.13, 5.1, 1.9), (0.06, 13.0, 3.5))  # bulges running down the stream: their depth,
# how many to its length, how many a second (about a third thicker and thinner along it: at a fifth, even tubing)
WHIP_FRAY = (0.5, 1.0)  # it necks into beads from this share of its length, as deep as this at its tail...
WHIP_BEAD = (8.0, 0.8)  # ... beads this many of its radii apart, flowing out along it, and how much thinner the stream
# between them is where it has necked through
WHIP_FLOW = 2.0  # hand sizes a second: the water flows out along it (its inside pattern drifts that way)
WHIP_GROW, WHIP_BREAK = 10, 5  # frames it takes to pour out, and to neck into beads when the voice stops
WHIP_LEAD = 6  # frames it is out before a hit that opens the voice

# ---- the voice ------------------------------------------------------------------------------------------------------
VOICE_ON = 0.1  # the voice sounds where its energy is over this (a breath, the room's hum is not the voice)
VOICE_GAP = 6  # frames: silences this short inside the voice are bridged ("ka-BOOM" is one sound)
VOICE_SMOOTH = 2.0  # frames: how the energy is smoothed before it drives the water (it shivered with every syllable)

PALM_LM = (0, 1, 5, 9, 13, 17)  # the landmarks the palm's plane goes through: the wrist, the thumb's root, the knuckles
TRACK_SMOOTH = 1.5  # frames: the hand's size and the palm's normal are smoothed over this (MediaPipe's jitter made a
# held orb breathe)
NO_LANDMARKS_TILT = 0.6  # with no landmarks the palm is taken to face this much along the emitter's direction, the
# rest towards the camera
FINGER_NEAR, FINGER_FAR = 0.15, 0.5  # hand sizes: a landmark's own depth holds this close to it and fades into the
# palm's plane by this far


def _rng(seed: int, *salt: int) -> np.random.Generator:
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, *salt])


def _ss(a: float, b: float, x):
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _tsmooth(x, sigma: float) -> np.ndarray:
    """``x`` (n, ...) smoothed along its first axis by a Gaussian of ``sigma`` frames, its ends held."""
    x = np.asarray(x, np.float64)
    if sigma <= 0 or len(x) < 2:
        return x.copy()
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return sum(k[i] * pad[i:i + len(x)] for i in range(2 * r + 1))


def _remap(img: np.ndarray, x: np.ndarray, y: np.ndarray, border: int = cv2.BORDER_REFLECT) -> np.ndarray:
    """``img`` read (bilinear) at float px ``x``, ``y`` of any shape: cv2.remap's maps, laid out in rows of 1024 (it
    takes at most 32767 a side)."""
    shape = np.shape(x)
    x = np.asarray(x, np.float32).reshape(-1)
    y = np.asarray(y, np.float32).reshape(-1)
    n = x.size
    if n == 0:
        return np.zeros(shape + img.shape[2:], np.float32)
    cols = 1024
    rows = -(-n // cols)
    pad = rows * cols - n
    if pad:
        x, y = np.concatenate([x, np.zeros(pad, np.float32)]), np.concatenate([y, np.zeros(pad, np.float32)])
    out = cv2.remap(img, x.reshape(rows, cols), y.reshape(rows, cols), cv2.INTER_LINEAR, borderMode=border)
    return out.reshape((rows * cols,) + img.shape[2:])[:n].reshape(shape + img.shape[2:])


# ---- the hand -------------------------------------------------------------------------------------------------------

@dataclass
class Hand:
    """The open palm frame by frame, in the picture's px, with depth z in px towards the camera."""

    p: np.ndarray  # (n, 2): where the water is held (the tracked emitter: the performer's tap, carried with the hand)
    nrm: np.ndarray  # (n, 3): unit normal out of the palm, on the side facing the camera
    along: np.ndarray  # (n, 3): unit, in the palm's plane, from the wrist towards the fingers
    s: np.ndarray  # (n,): hand size px
    plane: np.ndarray  # (n, 3): the palm's depth, z = a x + b y + c
    off: np.ndarray | None  # (n, 21, 3): each landmark's x, y and depth off that plane (None: no landmarks)

    def depth(self, k: int, x: np.ndarray, y: np.ndarray, fingers: bool = True) -> np.ndarray:
        """The hand's depth at px (x, y) on frame ``k``: its palm's plane, pulled to each landmark's own depth near it
        (a finger curled up out of the plane stands in front of what the plane is behind); with ``fingers`` False,
        the plane alone (where light lands on it: through the landmarks' bumps a focus broke into a string of them)."""
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        a, b, c = self.plane[k]
        z = np.float32(a) * x + np.float32(b) * y + np.float32(c)
        if self.off is None or not fingers:
            return z
        o = self.off[k].astype(np.float32)
        s = float(self.s[k])
        d2 = (x[..., None] - o[:, 0]) ** 2 + (y[..., None] - o[:, 1]) ** 2
        w = 1.0 / (d2 + np.float32((FINGER_NEAR * s) ** 2))
        return z + (w * o[:, 2]).sum(-1) / (w.sum(-1) + np.float32(1.0 / (FINGER_FAR * s) ** 2))


def hand_of(track: dict, n: int) -> Hand:
    """The palm from the track: its point (the emitter), size (smoothed), normal and depth (from its landmarks; without
    them, a palm facing the camera tipped towards the emitter's direction, at depth 0)."""
    em = [emitter_at(track, t) for t in range(n)]
    p = np.array([e[0] for e in em])
    d = np.array([e[1] for e in em])
    s = _tsmooth(np.array([e[2] for e in em]), TRACK_SMOOTH)
    lms = track.get("landmarks") or []
    seen = [t for t in range(min(n, len(lms))) if lms[t] is not None]
    if not seen:
        nrm = _unit(np.concatenate([NO_LANDMARKS_TILT * d, np.full((n, 1), math.sqrt(1 - NO_LANDMARKS_TILT ** 2))], 1))
        d3 = np.concatenate([d, np.zeros((n, 1))], 1)
        along = _unit(d3 - (d3 * nrm).sum(1, keepdims=True) * nrm)
        return Hand(p, nrm, along, s, np.zeros((n, 3)), None)
    L, last = [], lms[seen[0]]
    for t in range(n):
        last = lms[t] if t < len(lms) and lms[t] is not None else last
        L.append(np.asarray(last, np.float64)[:, :3])
    L = np.stack(L) * np.array([1.0, 1.0, -1.0])  # MediaPipe's z grows away from the camera
    raw = _unit(np.cross(L[:, 5] - L[:, 0], L[:, 17] - L[:, 0]))
    # one side of the palm all through, each frame agreeing with the one before, from the frame it most clearly faces
    # the camera on (that side): which hand MediaPipe says it is, and so which side its cross product is, flips for an
    # upturned palm
    ref = int(np.argmax(np.abs(raw[:, 2])))
    sign = np.ones(n)
    sign[ref] = 1.0 if raw[ref, 2] >= 0 else -1.0
    for t in list(range(ref + 1, n)) + list(range(ref - 1, -1, -1)):
        u = t - 1 if t > ref else t + 1
        sign[t] = sign[u] if raw[t] @ raw[u] >= 0 else -sign[u]
    nrm = _unit(_tsmooth(raw * sign[:, None], TRACK_SMOOTH))
    a3 = L[:, 9] - L[:, 0]
    along = _unit(a3 - (a3 * nrm).sum(1, keepdims=True) * nrm)
    plane = np.zeros((n, 3))
    off = L.copy()
    for t in range(n):
        A = np.c_[L[t, PALM_LM, :2], np.ones(len(PALM_LM))]
        plane[t] = np.linalg.lstsq(A, L[t, PALM_LM, 2], rcond=None)[0]
        off[t, :, 2] -= np.c_[L[t, :, :2], np.ones(21)] @ plane[t]
    return Hand(p, nrm, along, s, plane, off)


def _aim(tl: Timeline, hand: Hand, k: int) -> np.ndarray:
    """The unit direction in the picture a throw or a crack goes: the studio's arrow, else out of the palm as it shows
    in the picture, else (a palm facing the camera) the way the fingers point."""
    if tl.aim is not None:
        a = math.radians(tl.aim)
        return np.array([math.sin(a), -math.cos(a)])
    out = hand.nrm[k, :2]
    if np.hypot(*out) > 0.35:
        return out / np.hypot(*out)
    return _unit(hand.along[k, :2] + 1e-9)


# ---- the voice ------------------------------------------------------------------------------------------------------

def sounding(tl: Timeline) -> np.ndarray:
    """(n,) bool: the frames the voice sounds on (or holds a sound), short silences inside it bridged."""
    on = (tl.energy > VOICE_ON) | np.array([tl.held_at(t) for t in range(tl.n)], bool)
    idx = np.flatnonzero(on)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= VOICE_GAP + 1:
            on[a:b] = True
    return on


def _thrusts(track: dict) -> list[int]:
    return sorted(int(e["frame"]) for e in track.get("events") or [] if e.get("kind") == "extension")


@dataclass
class OrbPlan:
    """When the orb does what, in frames."""

    gather: int  # it starts to come together out of the air...
    formed: int  # ... and is whole
    release: int | None  # thrown or let fall (None: held to the end)
    throw: bool  # thrown along the aim (a hit, a thrust), else let fall onto the palm (the voice just stopped)
    size: float  # how hard: the throwing hit's size
    jolts: list[tuple[int, float]]  # the hits while it is held, with their sizes


def orb_plan(tl: Timeline, track: dict) -> OrbPlan:
    """The orb gathers when the voice first sounds and is thrown on its last hit when the voice ends on it (falls
    silent within THROW_TAIL frames), or on a thrust before that; a voice that opens with the very hit that throws it
    ("BOOM") has it gathered just before. Otherwise it is let fall LINGER frames after the voice stops (its hits jolt
    it); with no voice at all it gathers on the tap and is held."""
    heard = np.flatnonzero(sounding(tl))
    hits = sorted(zip(tl.hits, tl.sizes))
    start = int(heard[0]) if heard.size else int(tl.tap_frame)
    if hits:
        start = min(start, int(hits[0][0]))
    throw_at = next((f for f in _thrusts(track) if f >= start + GATHER_MIN), None)
    size = THRUST_SIZE
    ends_on = hits and (not heard.size or heard[-1] - hits[-1][0] <= THROW_TAIL)
    if ends_on and (throw_at is None or hits[-1][0] < throw_at):
        throw_at, size = int(hits[-1][0]), float(hits[-1][1])
    if throw_at is not None:
        g0 = max(0, min(start, throw_at - GATHER - GATHER_LEAD))
        g1 = max(g0 + 1, min(g0 + GATHER, throw_at - GATHER_LEAD))
        release, throw = throw_at, True
    else:
        g0, g1, throw, size = start, start + GATHER, False, DROP_SIZE
        release = max(int(heard[-1]) + 1 + LINGER, g1 + 1) if heard.size else None
    jolts = [(int(f), float(z)) for f, z in hits if g1 <= f and (release is None or f < release)]
    return OrbPlan(g0, g1, release, throw, size, jolts)


@dataclass
class WhipPlan:
    start: int  # it pours out of the palm...
    end: int | None  # ... and lets go here (None: held to the end)
    cracks: list[tuple[int, float]]  # the hits and thrusts that crack it, with how hard


def whip_plan(tl: Timeline, track: dict) -> WhipPlan:
    """The stream pours out when the voice first sounds (WHIP_LEAD frames before a hit that opens it, so the hit has
    a stream to crack), cracks on every hit and thrust, and lets go LINGER frames after the voice stops."""
    heard = np.flatnonzero(sounding(tl))
    hits = sorted(zip(tl.hits, tl.sizes))
    start = int(heard[0]) if heard.size else int(tl.tap_frame)
    if hits and hits[0][0] <= start + 1:
        start = max(0, int(hits[0][0]) - WHIP_LEAD)
    end = int(heard[-1]) + 1 + LINGER if heard.size else None
    cracks = [(int(f), float(z)) for f, z in hits] + [(f, THRUST_SIZE) for f in _thrusts(track)]
    cracks = sorted((f, z) for f, z in cracks if f >= start and (end is None or f < end))
    return WhipPlan(start, end, cracks)


# ---- what a frame's water is made of --------------------------------------------------------------------------------

class Blob(NamedTuple):
    x: float
    y: float
    z: float  # depth px, towards the camera
    r: float  # radius px
    shape: tuple = (1.0, 0.0, 0.0, 1.0)  # 2x2 row by row: takes a circle to the blob's outline (stretched, squashed)
    vx: float = 0.0  # px a frame: how the water moves here (what is seen inside it drifts with it)
    vy: float = 0.0
    spin: float = 0.0  # rad a frame: the water turning round the blob's centre
    foam: float = 0.0  # 0..1: how aerated it is
    modes: tuple = ()  # ((lobes, share of the radius, phase), ...): its outline's wobble
    w: float = 1.0  # its field's weight: blobs laid along a stream, each weighted by its share of it (_tube_weight),
    # sum to a tube of their radius (at 1 each, blobs 0.4 r apart summed to a tube 1.7 times as thick, its outline in
    # their tapering edges: a ribbed silicone hose)
    free: bool = False  # a piece flying on its own: it bridges to the body of water it overlaps, not to other free
    # pieces (summed, a splash's overlapping pieces fused into one flat, lumpy sheet stuck in the air)
    move: tuple = ()  # (mx, my): px it travels over the exposure; the shaded water is blurred along it (_motion_blur).
    # Drawn instead as a lens stretched along its motion, a lashing stream magnified what was behind it into zebra
    # stripes, and a splash's drops were glass needles
    stream: tuple = ()  # (arc px from the root, tx, ty): where along a stream it lies and the stream's direction there,
    # for the ripples, glints and froth that run along it
    flat: float = 1.0  # its thickness against a ball's (the pool a stream rises out of lies flat on the palm)


@dataclass
class Scene:
    """One frame's water."""

    blobs: list = field(default_factory=list)
    bubbles: list = field(default_factory=list)  # (x, y, r, seen): air in the water, and how clearly it is seen (0..1)
    rings: list = field(default_factory=list)  # ripples running over the orb: (its centre x, y and radius, the unit
    # direction from its centre to where it was struck, age frames, height, wavelength and speed px over its surface)
    wave: float = WAVE[0]  # how high its small waves are
    held: tuple | None = None  # (x, y, z, R, w): the orb over the palm, which focuses light onto the hand and shades
    # it, and how much it does (0..1: rising as it gathers, falling as it is thrown off; thrown, it stays where the orb
    # left the palm, so its shade fades there instead of sliding after it)
    lobes: tuple = ()  # the held orb's wobble ((lobes, share of the radius, phase), ...): its light on the palm wobbles
    # with it
    wet: list = field(default_factory=list)  # (x, y, rx, ry, angle, strength, on the hand, phase of its uneven edge):
    # wet patches it leaves, darker and faintly glossy, on the palm (under the stream's pool) or the wall (where the orb
    # hit it)


def _shape(u: np.ndarray, k: float, across: float = 1.0) -> tuple:
    """The shape matrix that stretches a circle k times along unit ``u`` and ``across`` times across it."""
    ux, uy = float(u[0]), float(u[1])
    return (k * ux * ux + across * uy * uy, (k - across) * ux * uy, (k - across) * ux * uy, k * uy * uy + across * ux * ux)


def _moving(r: float, v: np.ndarray, k: float = 1.0) -> tuple[tuple, tuple]:
    """A blob of radius ``r`` moving ``v`` px a frame, stretched ``k`` times along it (keeping its volume): (its shape,
    how far it travels over the shutter: Blob.move, at most SMEAR_MAX times its size)."""
    sp = float(np.hypot(*v))
    if sp < 1e-6:
        return _shape(np.array([1.0, 0.0]), k, 1.0 / math.sqrt(k)), ()
    travel = min(SMEAR_MAX * 2.0 * r, sp * SHUTTER)
    return _shape(v / sp, k, 1.0 / math.sqrt(k)), (float(v[0] / sp * travel), float(v[1] / sp * travel))


def _tube_weight(shape: tuple, sig: float, tang: np.ndarray, gap: float) -> float:
    """The weight that makes blobs ``gap`` px apart along unit ``tang`` sum to a tube as thick as one of them: a row
    of Gaussians sums, across it, to the Gaussian of its projected spread, scaled by sqrt(2 pi / a) / gap, a being the
    inverse covariance along the row."""
    S = np.asarray(shape, np.float64).reshape(2, 2)
    Ci = np.linalg.inv(S @ S.T) / (sig * sig)
    t = np.asarray(tang, np.float64)
    a = float(t @ Ci @ t)
    return gap * math.sqrt(a / (2.0 * math.pi))


class Field:
    """The blobs on a box of the frame: ``f`` the field the surface is drawn from (the body's blobs summed, plus the
    strongest free piece over them), per pixel the weighted means of the blobs' sigma^2 (``s2``), depth, flow, foam,
    motion over the exposure (``mx``, ``my``) and thickness (``th``, Blob.flat); over a stream, where along it a pixel
    is (``sa``, px from its root), how far across it (``sx``, in its radius, signed) and how much of the water there is
    the stream's (``sw``). ``dots``: the free pieces (x, y, z, r), for the glints of the small ones (_dots)."""

    def __init__(self, blobs: list[Blob], h: int, w: int):
        ext = []
        self.dots = [(b.x, b.y, b.z, b.r) for b in blobs if b.free and DOT_MIN <= b.r and 0 <= b.x < w and 0 <= b.y < h]
        for b in blobs:
            if b.r < 0.5 or b.w <= 0:
                continue
            sig = b.r * SIG_K * (1.0 + sum(abs(a) for _, a, _ in b.modes))
            S = np.asarray(b.shape, np.float64).reshape(2, 2)
            C = S @ S.T
            ex, ey = REACH * sig * math.sqrt(C[0, 0]), REACH * sig * math.sqrt(C[1, 1])
            ext.append((b, (b.y - ey, b.y + ey + 1, b.x - ex, b.x + ex + 1)))
        boxes = [(max(0, int(e[0])), min(h, int(e[1]) + 1), max(0, int(e[2])), min(w, int(e[3]) + 1)) for _, e in ext]
        boxes += [(max(0, int(y - 3)), min(h, int(y + 4)), max(0, int(x - 3)), min(w, int(x + 4)))
                  for x, y, *_ in self.dots]
        boxes = [bx for bx in boxes if bx[1] > bx[0] and bx[3] > bx[2]]
        if not boxes:
            self.box = None
            return
        self.box = (min(b[0] for b in boxes), max(b[1] for b in boxes), min(b[2] for b in boxes),
                    max(b[3] for b in boxes))
        y0, y1, x0, x1 = self.box
        (self.f, self.fm, self.wt, self.s2, self.zc, self.vx, self.vy, self.fo, self.mx, self.my, self.th, self.sa,
         self.sx, self.sw) = (np.zeros((y1 - y0, x1 - x0), np.float32) for _ in range(14))
        for b, e in ext:
            self._splat(b, e)
        self.f += self.fm
        wt = np.maximum(self.wt, 1e-12)
        for a in (self.s2, self.zc, self.vx, self.vy, self.fo, self.mx, self.my, self.th):
            a /= wt
        sw = np.maximum(self.sw, 1e-12)
        self.sa /= sw
        self.sx /= sw
        self.sw /= wt

    def _splat(self, b: Blob, e: tuple) -> None:
        y0, y1, x0, x1 = self.box
        ya, yb = max(y0, int(e[0])), min(y1, int(e[1]) + 1)
        xa, xb = max(x0, int(e[2])), min(x1, int(e[3]) + 1)
        if yb <= ya or xb <= xa:
            return
        sig = b.r * SIG_K
        dx = (np.arange(xa, xb, dtype=np.float32) - np.float32(b.x))[None, :]
        dy = (np.arange(ya, yb, dtype=np.float32) - np.float32(b.y))[:, None]
        Si = np.linalg.inv(np.asarray(b.shape, np.float64).reshape(2, 2)).astype(np.float32)
        u, v = Si[0, 0] * dx + Si[0, 1] * dy, Si[1, 0] * dx + Si[1, 1] * dy  # in the blob's own frame: a circle
        q = (u * u + v * v) * np.float32(1.0 / (sig * sig))
        if b.modes:
            th = np.arctan2(v, u)
            m = 1.0 + sum(np.float32(a) * np.cos(np.float32(l) * th - np.float32(ph)) for l, a, ph in b.modes)
            q = q / (m * m)
        r2 = REACH * REACH
        t = np.clip((r2 - q) * np.float32(1.0 / (r2 - 0.64 * r2)), 0.0, 1.0)
        g0 = np.exp(-0.5 * q) * (t * t * (3.0 - 2.0 * t))
        g = g0 * np.float32(b.w)
        sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
        if b.free:
            np.maximum(self.fm[sl], g, out=self.fm[sl])
        else:
            self.f[sl] += g
        self.wt[sl] += g
        self.s2[sl] += g * np.float32(sig * sig)
        self.zc[sl] += g * np.float32(b.z)
        self.vx[sl] += g * (np.float32(b.vx) - np.float32(b.spin) * dy)
        self.vy[sl] += g * (np.float32(b.vy) + np.float32(b.spin) * dx)
        self.th[sl] += g * np.float32(b.flat)
        if b.move:
            self.mx[sl] += g * np.float32(b.move[0])
            self.my[sl] += g * np.float32(b.move[1])
        if b.foam:
            self.fo[sl] += g * np.float32(b.foam)
        if b.stream:
            arc, tx, ty = (np.float32(a) for a in b.stream)
            self.sw[sl] += g
            self.sa[sl] += g * (arc + tx * dx + ty * dy)
            self.sx[sl] += g * (tx * dy - ty * dx) * np.float32(1.0 / b.r)


# ---- the room and the noise -----------------------------------------------------------------------------------------

class Room:
    """The room's light, frame by frame: what the water reflects (the upper room, as the plate's top shows it) and its
    key light."""

    def __init__(self, plate: np.ndarray):
        n, H, W = plate.shape[:3]
        top = max(1, int(ENV_TOP * H))
        self.env = [cv2.GaussianBlur(lin(cv2.resize(f[:top], (max(1, W // ENV_Q), max(1, top // ENV_Q)),
                                                    interpolation=cv2.INTER_AREA)), (0, 0), ENV_BLUR) for f in plate]
        q = KEY_Q
        at, colour, white = [], [], []
        for f in plate:
            e = cv2.GaussianBlur(lin(cv2.resize(f, (max(1, W // q), max(1, H // q)), interpolation=cv2.INTER_AREA)),
                                 (0, 0), 3.0)
            y = fx._lum(e)
            ys, xs = np.nonzero(y >= np.quantile(y, 1.0 - KEY_SHARE))
            at.append(((xs.mean() + 0.5) * q, (ys.mean() + 0.5) * q))
            c = e[ys, xs].mean(0)
            colour.append(c / max(float(c.max()), 1e-6))
            white.append(float(fx._lum(e[ys, xs]).mean()))
        self.at = _tsmooth(at, KEY_SMOOTH)
        self.colour = _tsmooth(colour, KEY_SMOOTH).astype(np.float32)
        self.white = _tsmooth(white, KEY_SMOOTH)

    def reflect(self, k: int, rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
        """(m, 3) linear: the room on frame ``k`` in the directions (rx, ry) a ray leaves the water in (unit vectors'
        x right and y down): straight up is the plate's top edge, level is down its upper strip, and what is below
        level is darker (the floor and the hand). Read by direction alone, every piece of water mirrors the same room
        (read off the plate beside it, drops over the hand mirrored the hand, pink)."""
        e = self.env[k]
        h, w = e.shape[:2]
        u = (0.5 + 0.5 * np.clip(rx, -1.0, 1.0)) * w - 0.5
        v = (0.5 + 0.5 * np.clip(ry, -1.0, 1.0)) * h - 0.5
        down = _ss(0.0, 1.0, ry).astype(np.float32)
        return _remap(e, u, v, cv2.BORDER_REPLICATE) * (1.0 - ENV_FLOOR * down)[:, None]

    def key(self, k: int, c) -> np.ndarray:
        """Unit vector from px ``c`` towards the key light on frame ``k`` (z towards the camera)."""
        d = self.at[k] - np.asarray(c, np.float64)
        d = (1.0 - KEY_UP) * d / max(float(np.hypot(*d)), 1e-6) + KEY_UP * np.array([0.0, -1.0])
        d = d / max(float(np.hypot(*d)), 1e-6)
        v = np.array([d[0], d[1], KEY_ELEV])
        return (v / np.linalg.norm(v)).astype(np.float32)

    def lit(self, k: int) -> np.ndarray:
        """The key light's colour and strength (linear)."""
        return self.colour[k] * np.float32(max(self.white[k], 0.3))


class Noise:
    """Tiling noise textures (zero mean, unit spread), read anywhere at fractional px, drifting with the water."""

    def __init__(self, seed: int, count: int = 10):
        rng = _rng(seed, _NOISE)
        N = NOISE_SIZE
        fy, fxq = np.fft.fftfreq(N)[:, None], np.fft.rfftfreq(N)[None, :]
        low = np.exp(-2.0 * math.pi ** 2 * NOISE_CELL ** 2 * (fy ** 2 + fxq ** 2))  # a Gaussian blur, periodic
        self.tex = []
        for _ in range(count):
            z = np.fft.irfft2(np.fft.rfft2(rng.standard_normal((N, N))) * low, (N, N))
            self.tex.append(((z - z.mean()) / max(float(z.std()), 1e-9)).astype(np.float32))

    def flow(self, i: int, x, y, vx, vy, k: int, size: float) -> np.ndarray:
        """Texture ``i`` at px (x, y) on frame ``k``, its features ``size`` px, carried by the flow (vx, vy) px a frame:
        two copies a half cycle apart, each drifting with the flow for FLOW_CYCLE frames and then taken up afresh
        elsewhere in the texture while the other is at its full weight (a flow map)."""
        N = NOISE_SIZE
        per = 2.0 * NOISE_CELL / max(size, 1e-3)  # texture px per frame px
        acc, w2 = 0.0, 0.0
        for ph in (0.0, 0.5):
            u = k / FLOW_CYCLE + ph
            cyc = math.floor(u)
            age = (u - cyc) * FLOW_CYCLE
            wgt = 1.0 - abs(2.0 * (u - cyc) - 1.0)
            ox, oy = (cyc * 97.31 + ph * 41.7) % N, (cyc * 61.13 + ph * 13.9) % N
            tx = np.mod((x - vx * age) * per + ox, N)
            ty = np.mod((y - vy * age) * per + oy, N)
            acc = acc + wgt * _remap(self.tex[i], tx, ty, cv2.BORDER_WRAP)
            w2 += wgt * wgt
        return acc / math.sqrt(w2)

    def along(self, i: int, a, c, k: int, size_a: float, size_c: float, speed: float) -> np.ndarray:
        """Texture ``i`` laid along a stream, at ``a`` px along it and ``c`` across it, its features ``size_a`` px long
        and ``size_c`` (in the units of ``c``) wide, carried out along it at ``speed`` px a frame (a flow map, as in
        flow): streaks that run with the water however the stream bends."""
        N = NOISE_SIZE
        pa, pc = 2.0 * NOISE_CELL / max(size_a, 1e-3), 2.0 * NOISE_CELL / max(size_c, 1e-3)
        acc, w2 = 0.0, 0.0
        for ph in (0.0, 0.5):
            u = k / FLOW_CYCLE + ph
            cyc = math.floor(u)
            age = (u - cyc) * FLOW_CYCLE
            wgt = 1.0 - abs(2.0 * (u - cyc) - 1.0)
            ox, oy = (cyc * 53.17 + ph * 29.3) % N, (cyc * 71.9 + ph * 7.7) % N
            tx = np.mod((a - speed * age) * pa + ox, N)
            ty = np.mod(c * pc + oy, N)
            acc = acc + wgt * _remap(self.tex[i], tx, ty, cv2.BORDER_WRAP)
            w2 += wgt * wgt
        return acc / math.sqrt(w2)


# ---- shading --------------------------------------------------------------------------------------------------------

def _pyramid(P: np.ndarray) -> list[list[np.ndarray]]:
    """The frame at LOD_MAX + 1 sizes, each split into its channels."""
    lv = [P]
    for _ in range(LOD_MAX):
        lv.append(cv2.pyrDown(lv[-1]))
    return [list(cv2.split(v)) for v in lv]


def _lod(pyr: list, c: int, x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Channel ``c`` of the frame at px (x, y), averaged over ``sigma`` px (between pyramid levels)."""
    lev = np.clip(np.log2(1.0 + np.maximum(sigma, 0.0)), 0.0, LOD_MAX)
    lo = np.minimum(lev.astype(np.int32), LOD_MAX - 1)
    fr = (lev - lo).astype(np.float32)
    out = np.empty(x.shape, np.float32)
    for L in range(LOD_MAX):
        sel = np.flatnonzero(lo == L)
        if not sel.size:
            continue
        xs, ys = x[sel], y[sel]
        a = _remap(pyr[L][c], (xs + 0.5) / 2 ** L - 0.5, (ys + 0.5) / 2 ** L - 0.5)
        b = _remap(pyr[L + 1][c], (xs + 0.5) / 2 ** (L + 1) - 0.5, (ys + 0.5) / 2 ** (L + 1) - 0.5)
        out[sel] = a + (b - a) * fr[sel]
    return out


def _spread(qx: np.ndarray, qy: np.ndarray, iy: np.ndarray, ix: np.ndarray, shape: tuple) -> np.ndarray:
    """How many px of the plate apart the points (qx, qy) read for neighbouring pixels (iy, ix) of a box of ``shape``
    are, as a blur (sigma px): the squeeze measured off the rays themselves. Estimated from the surface's curvature
    alone, it missed the squeeze of rays bent steeply near an outline, and the room seen there was read point by point:
    hard, aliased streaks inside a stream's edges and the orb's rim."""
    out = np.zeros(qx.shape, np.float32)
    for axis in (0, 1):
        A, B = np.full(shape, np.nan, np.float32), np.full(shape, np.nan, np.float32)
        A[iy, ix], B[iy, ix] = qx, qy
        step = np.hypot(np.diff(A, axis=axis), np.diff(B, axis=axis))  # NaN where either pixel is not water
        pad = [(0, 0), (0, 0)]
        pad[axis] = (1, 0)
        before = np.pad(step, pad, constant_values=np.nan)
        pad[axis] = (0, 1)
        after = np.pad(step, pad, constant_values=np.nan)
        out = np.fmax(out, np.nan_to_num(np.fmax(before, after))[iy, ix])
    return 0.5 * np.maximum(out - 1.0, 0.0)


def _footprint(d: np.ndarray, hxx, hxy, hyy) -> np.ndarray:
    """How many px of what is ``d`` px behind the water one pixel of it sees (sigma px): the map from the water to
    there is about 1 + d LENS times the surface's curvature. Where it squeezes the plate it is averaged; where it
    magnifies, defocused (DEFOCUS)."""
    j00, j01, j11 = 1.0 + d * LENS * hxx, d * LENS * hxy, 1.0 + d * LENS * hyy
    det = np.abs(j00 * j11 - j01 * j01)
    span = np.sqrt(np.maximum(j00 * j00 + j01 * j01, j01 * j01 + j11 * j11))
    mag = 1.0 / np.sqrt(np.maximum(det, 1e-6))
    return 0.5 * np.maximum(span - 1.0, 0.0) + DEFOCUS * np.maximum(mag - 1.0, 0.0)


@dataclass
class Layer:
    """The water over a box of the frame, to lay over the plate: premultiplied colour and coverage (linear)."""

    box: tuple  # (y0, y1, x0, x1)
    colour: np.ndarray  # (h, w, 3)
    alpha: np.ndarray  # (h, w)
    h: np.ndarray  # (h, w): its half thickness px...
    z: np.ndarray  # (h, w): ... its middle's depth (for the light it focuses onto the hand behind it)...
    r: np.ndarray  # (h, w): ... and the radius of the water there (a small drop's light on the hand is lost in the
    # room's soft light)


def _refract(ix, iy, iz, nx, ny, nz, eta):
    """Snell: unit incident ray i, unit normal n facing against it, ratio n1 / n2; (refracted ray, how far from
    being turned back it is: the discriminant, under 0 where it is)."""
    cosi = -(ix * nx + iy * ny + iz * nz)
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    a = eta * cosi - np.sqrt(np.maximum(k, 0.0))
    return (eta * ix + a * nx, eta * iy + a * ny, eta * iz + a * nz), k


def shade(fld: Field, sc: Scene, k: int, hand: Hand, room: Room, noise: Noise, pyr: list, matte: np.ndarray | None,
          code: float, seed: int) -> Layer | None:
    """The water of ``fld`` as seen on frame ``k``: traced through to the plate (``pyr``, the frame's pyramid) and the
    hand (``matte`` uint8, or None), reflecting the room, glinting, froth and grain; hidden where the hand is in front
    of it; blurred along its motion over the exposure. Pixels only where it covers some of the picture."""
    if fld.box is None:
        return None
    H, W = pyr[0][0].shape
    y0, y1, x0, x1 = fld.box
    s = float(hand.s[k])
    f = fld.f
    has = f > 1e-6
    lf = np.where(has, np.log(np.maximum(f, 1e-6) / TAU), -10.0).astype(np.float32)
    sig = np.sqrt(fld.s2)
    vx, vy = fld.vx, fld.vy
    ys, xs = np.nonzero(lf > -0.7)  # as far out as the waves can move the surface
    if not ys.size:
        return None
    X, Y = (xs + x0).astype(np.float32), (ys + y0).astype(np.float32)
    lf0 = lf[ys, xs]
    thick0 = np.clip(lf0 / np.float32(LN_TAU), 0.0, 1.0)  # (h / R)^2 for a lone blob: 0 at its outline
    at_edge = WAVE_EDGE + (1.0 - WAVE_EDGE) * thick0
    bump = sc.wave * at_edge * noise.flow(0, X, Y, vx[ys, xs], vy[ys, xs], k, WAVE_SIZE * s)
    sw0 = fld.sw[ys, xs]
    on_s = np.flatnonzero(sw0 > 0.02)
    if on_s.size:  # a stream's surface is broken by ripples running out along it
        sa0, sx0 = fld.sa[ys, xs][on_s], fld.sx[ys, xs][on_s]
        bump[on_s] += RIPPLE[0] * sw0[on_s] * at_edge[on_s] * noise.along(6, sa0, sx0, k, RIPPLE[1] * s, RIPPLE[2],
                                                                           WHIP_FLOW * s / FPS)
    if sc.rings:
        hz = np.sqrt(2.0 * sig[ys, xs] ** 2 * np.maximum(lf0, 0.0))  # the front surface's height over the middle
    for cx, cy, rr, ux, uy, uz, age, amp, lam, speed in sc.rings:  # a ripple from a hit, running out over the ball
        px, py = X - cx, Y - cy  # from where it struck, measured over its surface (measured flat on the picture, its
        pn = np.sqrt(px * px + py * py + hz * hz) + 1e-6  # rings were a bullseye drawn on the disc)
        d = rr * np.arccos(np.clip((px * ux + py * uy + hz * uz) / pn, -1.0, 1.0)) - speed * age
        bump += amp * np.exp(-age / RING[3]) * np.sin(2 * math.pi * d / lam) * np.exp(-(d / (1.5 * lam)) ** 2)
    lf[ys, xs] += bump.astype(np.float32)
    sd = np.where(has, sig * np.float32(SIG_K) * lf, -1e3)  # signed distance to the surface, px (+ inside), and far
    # outside where no blob reaches (taken as 0 there, every px of the box was half covered by water)
    cover = np.clip(0.5 + sd, 0.0, 1.0)
    act = cover > 0
    rows, cols = np.flatnonzero(act.any(1)), np.flatnonzero(act.any(0))
    if not rows.size:
        return None
    ra, rb = max(0, rows[0] - 3), min(act.shape[0], rows[-1] + 4)
    ca, cb = max(0, cols[0] - 3), min(act.shape[1], cols[-1] + 4)
    crop = (slice(ra, rb), slice(ca, cb))
    by0, bx0 = y0 + ra, x0 + ca
    th = fld.th[crop]  # a flat piece is thinner, and its edge tapers in gently, as a film's does (a ball's profile
    # scaled down still rose sheer at its outline, and a pool on the palm was a glass lump with a dark rim)
    h2 = (sig[crop] * np.float32(math.sqrt(2.0 * LN_TAU)) * th
          * np.clip(lf[crop] / np.float32(LN_TAU), 0.0, None) ** (1.0 - 0.5 * th)).astype(np.float32)
    hb = cv2.GaussianBlur(h2, (0, 0), 0.8)
    gy, gx = np.gradient(hb)
    gx, gy = np.clip(gx, -6, 6), np.clip(gy, -6, 6)
    hxx = cv2.GaussianBlur(np.gradient(gx, axis=1), (0, 0), 1.5)
    hyy = cv2.GaussianBlur(np.gradient(gy, axis=0), (0, 0), 1.5)
    hxy = cv2.GaussianBlur(np.gradient(gx, axis=0), (0, 0), 1.5)
    nd = np.sqrt(gx * gx + gy * gy + 1.0)
    dnx, dny, dnz = -gx / nd, -gy / nd, 1.0 / nd
    a_ = act[crop]
    iy, ix = np.nonzero(a_)
    X, Y = (ix + bx0).astype(np.float32), (iy + by0).astype(np.float32)
    h = h2[iy, ix]
    Rmap = sig[crop] / np.float32(SIG_K)
    R = Rmap[iy, ix]
    big = _ss(0.1 * s, 0.25 * s, R).astype(np.float32)  # 0 for a stream or a drop, 1 for the orb
    tiny = (1.0 - _ss(TINY[0] * s, TINY[1] * s, R)).astype(np.float32)  # a small drop: see-through, a glint, an edge
    zc = fld.zc[crop][iy, ix]
    fvx, fvy = vx[crop][iy, ix], vy[crop][iy, ix]
    foam = fld.fo[crop][iy, ix]
    sw = fld.sw[crop][iy, ix]
    st = np.flatnonzero(sw > 0.02)
    sa, sxs = fld.sa[crop][iy, ix][st], fld.sx[crop][iy, ix][st]
    mxc, myc = fld.mx[crop], fld.my[crop]
    fast = (np.hypot(mxc[iy, ix], myc[iy, ix]) / (SHUTTER * max(s, 1.0))).astype(np.float32)  # hand sizes a frame
    thick = np.clip(h / np.maximum(0.2 * R, 1.0), 0.0, 1.0)
    sx = SPARKLE * thick * noise.flow(3, X, Y, fvx, fvy, k, SPARKLE_SIZE * s)
    sy = SPARKLE * thick * noise.flow(4, X, Y, fvx, fvy, k, SPARKLE_SIZE * s)
    gxa, gya = gx[iy, ix] + sx, gy[iy, ix] + sy  # the surface the light glints off, its finest waves and all...
    na = np.sqrt(gxa * gxa + gya * gya + 1.0)
    nx, ny, nz = -gxa / na, -gya / na, 1.0 / na
    bends = SPARKLE_BENDS * big  # ... and the one seen through (through a stream, a room away, its fine waves were
    gxr, gyr = gx[iy, ix] + bends * sx, gy[iy, ix] + bends * sy  # ribs across it)
    nr = np.sqrt(gxr * gxr + gyr * gyr + 1.0)

    # in through the front, across, out through the back (whose normal is the front's mirrored), on to what is behind
    eta = ETA[1]
    (tx, ty, tz), _ = _refract(0.0, 0.0, -1.0, -gxr / nr, -gyr / nr, 1.0 / nr, 1.0 / eta)
    path = 2.0 * h / np.maximum(-tz, 0.25)
    ex, ey = X + tx * path, Y + ty * path
    bx = _remap(dnx, ex - bx0, ey - by0)
    byy = _remap(dny, ex - bx0, ey - by0)
    bz = _remap(dnz, ex - bx0, ey - by0)
    (ox, oy, oz), kk = _refract(tx, ty, tz, -bx, -byy, bz, eta)
    out = _ss(0.0, TIR_SOFT, kk).astype(np.float32)  # 1: out through the back; 0: turned back in, on as it was going
    ox, oy, oz = out * ox + (1 - out) * tx, out * oy + (1 - out) * ty, out * oz - (1 - out) * np.abs(tz)
    inner = noise.flow(1, X, Y, fvx, fvy, k, INNER_SIZE * s)
    deep = np.clip(h / np.maximum(R, 1.0), 0.0, 1.0)
    swirl = INNER * deep * big
    # fast water, and a stream bent tightly, sends a ray less far off its way (through a lashing stream the room
    # behind it was magnified into zebra stripes; in a stream's tight bend, into a fan of fine lines)
    curv = (np.abs(hxx) + np.abs(hyy) + 2.0 * np.abs(hxy))[iy, ix] * R
    calm = (1.0 / ((1.0 + fast / CALM[0]) * (1.0 + (1.0 - big) * (curv / CALM[1]) ** 2))).astype(np.float32)
    tanx = (np.clip(ox / np.maximum(-oz, 0.3), -3, 3) + swirl * inner) * calm
    tany = (np.clip(oy / np.maximum(-oz, 0.3), -3, 3) + swirl * noise.flow(2, X, Y, fvx, fvy, k, INNER_SIZE * s)) * calm

    vis = np.ones(X.shape, np.float32)
    dn = np.full(X.shape, NEAR_OPEN * s, np.float32)
    if matte is not None:
        mh = _remap(matte, X, Y).astype(np.float32) / 255.0
        on = np.flatnonzero(mh > 0.01)
        if on.size:
            zh = hand.depth(k, X[on], Y[on])
            zf, zb, m_ = zc[on] + h[on], zc[on] - h[on], mh[on]
            vis[on] = 1.0 - m_ * np.clip((zh - zf) / (DEPTH_SOFT * s) + 0.5, 0.0, 1.0)  # the hand in front hides it
            cut = np.clip((zf - zh) / np.maximum(2.0 * h[on], 1e-3), 0.0, 1.0)  # the hand through it: only the water
            path[on] *= 1.0 - m_ * (1.0 - cut)  # in front of it is crossed
            dn[on] = m_ * np.clip(zb - zh, NEAR[0] * s, NEAR[1] * s) + (1.0 - m_) * dn[on]  # (switched where the
            # matte crossed a half, the hand's outline behind the water showed through it as a jagged stair)
    dfar = ROOM_DEPTH * max(H, W) * np.clip(R / (LENS_FULL * s), LENS_MIN, 1.0)
    # the room's back wall is ROOM_DEPTH behind the palm: water near it sees it right behind (a splash on the wall,
    # seeing a room's depth behind it, was chrome)
    p = hand.p[k]
    zroom = float(hand.depth(k, np.float32([p[0]]), np.float32([p[1]]))[0]) - ROOM_DEPTH * max(H, W)
    dfar = np.minimum(dfar, np.maximum(zc - h - zroom, NEAR[0] * s))
    dfar = dfar * (1.0 - (1.0 - FAR_FAST) * _ss(0.5, 1.0, fast * FPS / WHIP_SHED)).astype(np.float32)
    seen = np.float32(TINY_SEES) * R  # a small drop shows what is right behind it, a little magnified (sent a room
    dfar = dfar + tiny * (seen - dfar)  # away, it showed the room shrunk into a grey bead: metal)
    dn = dn + tiny * (np.minimum(dn, seen) - dn)
    hxx_, hxy_, hyy_ = hxx[iy, ix] * calm, hxy[iy, ix] * calm, hyy[iy, ix] * calm
    qx, qy = ex + tanx * dn, ey + tany * dn
    sig_n = np.maximum(_footprint(dn, hxx_, hxy_, hyy_), _spread(qx, qy, iy, ix, a_.shape)) + BASE_BLUR * R
    near = np.stack([_lod(pyr, c, qx, qy, sig_n) for c in range(3)], -1)
    disp = dfar * np.float32((ETA[0] - 1) / (eta - 1)), dfar, dfar * np.float32((ETA[2] - 1) / (eta - 1))
    sig_f = np.maximum(_footprint(dfar, hxx_, hxy_, hyy_), _spread(ex + tanx * dfar, ey + tany * dfar, iy, ix,
                                                                  a_.shape))
    far = np.stack([_lod(pyr, c, ex + tanx * disp[c], ey + tany * disp[c], sig_f) for c in range(3)], -1)
    if matte is not None:  # which of the two a ray lands on, softened over as much of the hand as the ray sees (a thin
        # stream flicked between them pixel by pixel; read at one softness, the finger's outline seen squeezed through
        # a stream was a sawtooth)
        qm = 4
        msel = cv2.GaussianBlur(cv2.resize(matte, (W // qm, H // qm), interpolation=cv2.INTER_AREA).astype(np.float32)
                                / 255.0, (0, 0), max(1.0, 0.04 * s / qm))
        mpyr = [[msel]]
        for _ in range(LOD_MAX):
            mpyr.append([cv2.pyrDown(mpyr[-1][0])])
        m = _lod(mpyr, 0, (qx + 0.5) / qm - 0.5, (qy + 0.5) / qm - 0.5, 2.0 * sig_n / qm + 1.0)[:, None]
        col = m * near + (1.0 - m) * far
    else:
        col = far
    col *= np.exp(-ABSORB[None, :] * (path / max(s, 1.0))[:, None])
    # the rim's darkening and the dimming where rays are turned back belong to a body of water: a small drop is all
    # rim, and with them it was a grey metal sliver; it keeps a thin edge and a darker core round its glint
    small = np.clip(R / (SMALL_RIM * s), 0.0, 1.0)
    edge = (SMALL_EDGE + (1.0 - SMALL_EDGE) * big) * small
    rr = np.clip(1.0 - nz, 0.0, 1.0)
    col *= (1.0 - edge * RIM_DARK * _ss(0.35, 0.95, rr) ** 1.5).astype(np.float32)[:, None]
    col *= (1.0 - edge * (1.0 - TIR_DIM) * (1.0 - out))[:, None]
    col *= (1.0 - tiny * (TINY_EDGE * _ss(0.55, 0.95, rr) + TINY_CORE * (1.0 - _ss(0.0, 0.7, rr))))[:, None]
    # the room reflected, by the direction a ray leaves the surface in (a small piece of water is a weak mirror: at
    # full strength a splash's thin pieces were chrome)
    F = (F0 + (1.0 - F0) * (1.0 - nz) ** 5).astype(np.float32)
    F = np.minimum(F, F_SMALL + (1.0 - F_SMALL) * big) * (1.0 - TINY_MIRROR * tiny)
    flow_a = WHIP_FLOW * s / FPS
    if st.size:  # along a stream its glints and the room it mirrors break up with the ripples running down it (one
        # unbroken streak down its length was an acrylic rod's, a neon tube's)
        gm = np.clip(1.0 + GLINT_BREAK[0] * noise.along(7, sa, sxs, k, GLINT_BREAK[1] * s, 0.8, flow_a), 0.0, 2.0)
        F[st] *= 1.0 + 0.5 * sw[st] * (gm - 1.0)
    rx, ry = 2.0 * nz * nx, 2.0 * nz * ny
    ev = room.reflect(k, rx, ry)
    col = col * (1.0 - F)[:, None] + ev * F[:, None]
    # the key light: its window mirrored in the surface, waves and all, a soft sheen round it, the window again off the
    # inside of the far wall (opposite, smaller, dimmer), the light the water gathers onto its far inside edge, and a
    # faint moving net inside it
    Lk = room.key(k, ((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    lit = room.lit(k)
    gxw, gyw = gx[iy, ix] + WINDOW[3] * sx, gy[iy, ix] + WINDOW[3] * sy  # the window is mirrored in the surface's
    nw = np.sqrt(gxw * gxw + gyw * gyw + 1.0)  # wobble and waves, and only a little of its finest waves
    spec = GLINT[0] * _window(-gxw / nw, -gyw / nw, 1.0 / nw, Lk, 1.0)
    spec += (BACK[0] * big * deep) * _window(gxw / nw, gyw / nw, 1.0 / nw, Lk, BACK[1])
    hv = Lk + np.array([0.0, 0.0, 1.0], np.float32)
    hv /= np.linalg.norm(hv)
    ndh = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0.0, 1.0)
    spec += GLINT[1] * ndh ** GLINT_POW
    if st.size:
        spec[st] *= 1.0 + sw[st] * (gm - 1.0)
    lxy = math.hypot(float(Lk[0]), float(Lk[1])) + 1e-6
    away = -(nx * Lk[0] + ny * Lk[1]) / lxy / np.maximum(np.hypot(nx, ny), 1e-3)
    cres = CRESCENT * _ss(0.25, 0.85, rr) * (1.0 - _ss(0.9, 1.0, rr)) * _ss(0.2, 0.95, away)
    net = INNER_GLOW * np.exp(-(inner / 0.22) ** 2) * deep * (1.0 - F)
    col += (spec + cres + net).astype(np.float32)[:, None] * lit[None, :]
    # froth: white matter lit by the room, in bubbles, and along a stream in streaks running out with it
    fn = 0.55 + 0.45 * noise.flow(5, X, Y, fvx, fvy, k, FOAM_SIZE * s)
    if st.size:
        streak = FOAM_STREAK[2] + 1.6 * np.maximum(noise.along(8, sa, sxs, k, FOAM_STREAK[0] * s, FOAM_STREAK[1],
                                                               flow_a), 0.0)
        fn[st] += sw[st] * (streak * fn[st] - fn[st])
    fm = np.clip(foam * fn, 0.0, 1.0) * FOAM_MAX
    glint = FOAM_GLINT[0] * np.exp(-(noise.flow(9, X, Y, fvx, fvy, k, FOAM_GLINT[1] * FOAM_SIZE * s) / 0.3) ** 2)
    col = col * (1.0 - fm)[:, None] + (fm * ((0.75 + 0.25 * nz) * FOAM_ALBEDO + glint))[:, None] * lit[None, :]
    alpha = np.zeros(a_.shape, np.float32)
    alpha[iy, ix] = cover[crop][iy, ix] * vis
    C = np.zeros(alpha.shape + (3,), np.float32)
    C[iy, ix] = col * alpha[iy, ix, None]
    _dots(C, fld.dots, (by0, bx0), s, Lk, lit, hand, k, matte)
    box, C, alpha = _motion_blur(C, alpha, mxc, myc, (by0, by0 + C.shape[0], bx0, bx0 + C.shape[1]), H, W)
    C = cv2.GaussianBlur(C, (0, 0), EDGE_SOFT)
    alpha = cv2.GaussianBlur(alpha, (0, 0), EDGE_SOFT)
    if code > 0:  # the plate's grain, on what the water shows (what is seen through it was read smoothed, its glints
        # are clean); the sensor's, so laid on after the motion blur
        y = np.maximum(C @ fx._LUMA, 1e-6) / np.maximum(alpha, 1e-3)
        step = (2.2 * code) * y ** np.float32(1.2 / 2.2) / y
        g = _rng(seed, _GRAIN, k).standard_normal(y.shape).astype(np.float32)
        C = np.maximum(C * (1.0 + g * step)[..., None], 0.0)
    pad = ((by0 - box[0], box[1] - by0 - h2.shape[0]), (bx0 - box[2], box[3] - bx0 - h2.shape[1]))
    return Layer(box, C, alpha, np.pad(h2, pad), np.pad(fld.zc[crop], pad), np.pad(Rmap, pad))


def _window(nx, ny, nz, Lk: np.ndarray, size: float) -> np.ndarray:
    """The key light's window mirrored in a surface of normal n: a soft rounded shape (WINDOW), ``size`` times as wide,
    brighter at its top. Hard-edged and lit evenly, it was a white sticker on a crystal ball."""
    e1 = _unit(np.array([Lk[2], 0.0, -Lk[0]], np.float32))
    e2 = np.cross(Lk, e1)
    wx_, wy_, wz_ = 2.0 * nz * nx, 2.0 * nz * ny, 2.0 * nz * nz - 1.0
    wa = (wx_ * e1[0] + wy_ * e1[1] + wz_ * e1[2]) / np.float32(WINDOW[0] * size)
    wb = (wx_ * e2[0] + wy_ * e2[1] + wz_ * e2[2]) / np.float32(WINDOW[1] * size)
    win = _ss(1.0 + WINDOW[2], 1.0 - WINDOW[2], np.sqrt(wa * wa + wb * wb)).astype(np.float32)
    return win * (wx_ * Lk[0] + wy_ * Lk[1] + wz_ * Lk[2] > 0) * (0.75 + 0.25 * np.clip(-wb, -1.0, 1.0))


def _dots(C: np.ndarray, dots: list, at: tuple, s: float, Lk: np.ndarray, lit: np.ndarray, hand: Hand, k: int,
          matte: np.ndarray | None) -> None:
    """The key light's glint on each small free drop (Field.dots), in place on the premultiplied colour ``C`` whose top
    left is the frame's px ``at``: a bright point towards the light, which the motion blur draws out into a streak.
    Shaded pixel by pixel alone, a drop a few px across lost its glint and read as a grey speck. Hidden where the hand
    is in front of the drop."""
    if not dots:
        return
    hh, ww = C.shape[:2]
    d = np.asarray(dots, np.float64)
    keep = d[:, 3] < DOT[0] * s
    if matte is not None and keep.any():
        xi = np.clip(d[:, 0].astype(int), 0, matte.shape[1] - 1)
        yi = np.clip(d[:, 1].astype(int), 0, matte.shape[0] - 1)
        zh = hand.depth(k, d[:, 0].astype(np.float32), d[:, 1].astype(np.float32))
        keep &= ~((matte[yi, xi] > 127) & (zh > d[:, 2] + d[:, 3]))
    lxy = np.asarray(Lk[:2], np.float64) / (math.hypot(float(Lk[0]), float(Lk[1])) + 1e-6)
    for x, y, _, r in d[keep]:
        gw = float(1.0 - _ss(0.5 * DOT[0] * s, DOT[0] * s, r))  # a bigger drop has a glint of its own
        sg = max(DOT[2], DOT[3] * r)
        cx, cy = x + 0.35 * r * lxy[0] - at[1], y + 0.35 * r * lxy[1] - at[0]
        e = int(3 * sg + 2)
        ya, yb, xa, xb = max(0, int(cy) - e), min(hh, int(cy) + e + 1), max(0, int(cx) - e), min(ww, int(cx) + e + 1)
        if yb <= ya or xb <= xa:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        pk = DOT[1] * gw * min(1.0, r / (2.0 * DOT[2]))  # a speck under a pixel glints over its share of one
        C[ya:yb, xa:xb] += (pk * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sg * sg)))[..., None] * lit


def _motion_blur(C: np.ndarray, A: np.ndarray, mx: np.ndarray, my: np.ndarray, box: tuple, H: int,
                 W: int) -> tuple:
    """The premultiplied colour ``C`` (h, w, 3) and coverage ``A`` (h, w) over ``box`` of the frame, each pixel spread
    evenly along the path (mx, my) px it travels over the exposure, centred on it: scattered along its own motion, so a
    fast drop smears past its outline into a streak and a slow root stays sharp beside a fast tail. Returns (the box,
    grown to hold the smear, C, A)."""
    y0, y1, x0, x1 = box
    sp = np.hypot(mx, my)
    on = (A > 1e-4) | (C.max(-1) > 1e-4)
    if not on.any() or float(sp[on].max()) < 0.5:
        return box, C, A
    e = int(math.ceil(0.5 * float(sp[on].max()))) + 2
    ny0, ny1, nx0, nx1 = max(0, y0 - e), min(H, y1 + e), max(0, x0 - e), min(W, x1 + e)
    hh, ww = ny1 - ny0, nx1 - nx0
    iy, ix = np.nonzero(on)
    val = np.concatenate([C[iy, ix], A[iy, ix, None]], 1).astype(np.float32)
    n = np.clip(np.ceil(sp[iy, ix] / BLUR_STEP), 1, BLUR_TAPS).astype(np.int32)
    px, py = ix + (x0 - nx0), iy + (y0 - ny0)
    four = np.arange(4)
    acc = np.zeros(hh * ww * 4, np.float64)
    still = np.flatnonzero(n == 1)  # a pixel that hardly moves stays where it is
    idx = (py[still] * ww + px[still]) * 4
    acc += np.bincount((idx[:, None] + four).ravel(), val[still].ravel(), hh * ww * 4)
    moving = np.flatnonzero(n > 1)
    if moving.size:
        nm = n[moving]
        reps = np.repeat(moving, nm)  # each moving pixel once per tap...
        j = np.arange(reps.size) - np.repeat(np.cumsum(nm) - nm, nm)  # ... its taps numbered 0 .. n - 1
        t = ((j + 0.5) / n[reps] - 0.5).astype(np.float32)
        X = px[reps] + t * mx[iy[reps], ix[reps]]
        Y = py[reps] + t * my[iy[reps], ix[reps]]
        wt = (1.0 / n[reps]).astype(np.float32)
        xf, yf = np.floor(X), np.floor(Y)
        fx_, fy_ = (X - xf).astype(np.float32), (Y - yf).astype(np.float32)
        xi, yi = xf.astype(np.int64), yf.astype(np.int64)
        v = val[reps]
        for dx, dy, cw in ((0, 0, (1 - fx_) * (1 - fy_)), (1, 0, fx_ * (1 - fy_)), (0, 1, (1 - fx_) * fy_),
                           (1, 1, fx_ * fy_)):
            xx, yy = xi + dx, yi + dy
            ok = (xx >= 0) & (xx < ww) & (yy >= 0) & (yy < hh)
            idx = (yy[ok] * ww + xx[ok]) * 4
            acc += np.bincount((idx[:, None] + four).ravel(), (v[ok] * (wt[ok] * cw[ok])[:, None]).ravel(),
                               hh * ww * 4)
    acc = acc.reshape(hh, ww, 4).astype(np.float32)
    return (ny0, ny1, nx0, nx1), np.ascontiguousarray(acc[..., :3]), np.ascontiguousarray(acc[..., 3])


CAUSTIC_GAIN, CAUSTIC_KEEP = 1.0, 0.3  # the focus's strength, and the key light's share of what lights the hand (the
# rest is the room's soft light, which the water does not focus)
CAUSTIC_MAX = 2.0  # the most the focus multiplies the key light by (at 3, with 0.4 of it the key's, the skin under the
# orb clipped to white under a soft room light)
CAUSTIC_SOFT = (0.06, 0.3)  # of the distance it falls: how soft the focus and the shadow are (a room's key light is
# wide: a shadow blurred as little as its focus showed the orb's outline drawn on the palm, a hard-edged disc)
CAUSTIC_SMOOTH = 0.04  # hand sizes: the surface the light is bent through is smoothed over this (bent through its small
# waves, the focus was a jagged shape that jumped from frame to frame: the light on the palm under a held orb went from
# a third brighter to a quarter darker in one frame)
CAUSTIC_BINS = (0.2, 0.5)  # hand sizes: the water is blurred onto the hand in three lots by how far it falls (blurred
# all by the middle one's distance, drops near the palm left hard grey discs on it, like dirt)
CAUSTIC_ELEV = 1.8  # the light it focuses comes more from the camera's side than its glint does (from the glint's
# direction the focus landed a hand off, on the wall)
SHADE_AO, SHADE_R = 0.14, 0.6  # the palm right under the orb loses this much of the room's light, over this share of
# its radius and its height over the palm (sigma; at its radius and half its height, out to 3 sigmas, fingers a hand
# off it were left darker all through the hold, and the whole hand brightened when it was thrown)
SPOT = (0.85, 0.8, 0.1)  # the held orb's focus: the share of the light through it that reaches it, how wide it is
# against its rays' width there, and the share of the orb's light still traced cell by cell over it (the spot alone
# was a projector's disc; that share dances with its surface)
SPOT_LOBES = 4.0  # its outline's wobble shows this many times over in its focus...
SPOT_NET = (0.4, 0.2, 0.04)  # ... and the light in it lies in a net of brighter lines, as light through moving water
# does: how strong (of the spot), how big its cells (of the orb's radius), and how fast they shift (of its radius a
# frame; a smooth spot was a projector's disc)
PAD_MAX = 1.2  # hand sizes: the most the light it throws reaches off the water


def _splat(tgt: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
    """Adds weights ``w`` at fractional cells (x, y) of ``tgt``, each shared among its four nearest (rounded to one, a
    focus moving a fraction of a cell jumped from cell to cell, and shimmered); what lands off ``tgt`` is dropped (piled
    onto its edge, it drew the box's outline)."""
    gh, gw = tgt.shape
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx_, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    w = np.asarray(w, np.float32)
    for dx, dy, ww in ((0, 0, (1 - fx_) * (1 - fy)), (1, 0, fx_ * (1 - fy)), (0, 1, (1 - fx_) * fy), (1, 1, fx_ * fy)):
        xi, yi = x0 + dx, y0 + dy
        on = (xi >= 0) & (xi < gw) & (yi >= 0) & (yi < gh)
        np.add.at(tgt, (yi[on], xi[on]), (w * ww)[on])


def caustic(P: np.ndarray, layer: Layer, k: int, hand: Hand, room: Room, matte: np.ndarray,
            held: tuple | None = None, lobes: tuple = (), noise: Noise | None = None) -> tuple:
    """The water's light on the hand behind it, in place: each cell of water in front of the hand sends its share of
    the key light along the ray bent through it to the hand, at the hand's own depth there, and lights it there; the
    light the hand had there without the water is taken away (energy kept: a bright focus inside a soft shadow). A
    small drop's share fades: under a room's wide light its shadow and its focus spread wider than it is. ``held``,
    the orb over the palm (x, y, z, R, w): its light lands as one, wobbling with its ``lobes`` (Scene.lobes), in a net
    of brighter lines that shifts (``noise``), and its bulk also takes some of the room's soft light off the palm right
    under it, both at w of their strength, the shade shrinking as it goes. Returns the box it changed."""
    s = float(hand.s[k])
    ly0, ly1, lx0, lx1 = layer.box
    L = room.key(k, ((lx0 + lx1) / 2.0, (ly0 + ly1) / 2.0)) * np.float32([1.0, 1.0, CAUSTIC_ELEV / KEY_ELEV])
    L /= np.linalg.norm(L)
    q = 2
    hw = (max(1, (lx1 - lx0) // q), max(1, (ly1 - ly0) // q))
    hq = cv2.resize(layer.h, hw, interpolation=cv2.INTER_AREA)
    zq = cv2.resize(layer.z, hw, interpolation=cv2.INTER_AREA)
    rq = cv2.resize(layer.r, hw, interpolation=cv2.INTER_AREA)
    iy, ix = np.nonzero(hq > 0.6)
    if not iy.size:
        return (ly0, ly0, lx0, lx0)
    zh = hand.depth(k, (ix + 0.5) * q - 0.5 + lx0, (iy + 0.5) * q - 0.5 + ly0, fingers=False)
    front = np.clip((zq[iy, ix] - zh) / (DEPTH_SOFT * s) + 0.5, 0.0, 1.0)  # water behind the hand lights none of it
    d = np.clip(zq[iy, ix] - hq[iy, ix] - zh, NEAR[0] * s, NEAR[1] * s)  # px from the water's back to the hand
    ao_r, wh, reach = 0.0, 0.0, 0.0
    own = np.ones(iy.size, np.float32)
    if held is not None:  # the orb's light lands as one spot (_spot), handed back to its cells as it is thrown
        cx, cy, cz, R, wh = held  # (switched at once, the light on the palm blinked off, on and off)
        dh = float(np.clip(cz - R - hand.depth(k, np.float32([cx]), np.float32([cy]), fingers=False)[0],
                           NEAR[0] * s, NEAR[1] * s))
        orb = np.hypot((ix + 0.5) * q + lx0 - cx, (iy + 0.5) * q + ly0 - cy) < 1.2 * R
        d = np.where(orb, wh * dh + (1.0 - wh) * d, d)
        own = np.where(orb, 1.0 - wh * (1.0 - SPOT[2]), 1.0).astype(np.float32)
        ao_r, reach = SHADE_R * (R + dh) * (0.6 + 0.4 * wh), dh + R
    small = np.clip((rq[iy, ix] / (CAUSTIC_SOFT[1] * d)) ** 2, 0.0, 1.0) * own
    dmax = max(float(d.max()), reach)
    pad = int(min(dmax * (abs(L[0]) + abs(L[1])) / L[2] + 3.0 * CAUSTIC_SOFT[1] * dmax, PAD_MAX * s) + 4)
    H, W = P.shape[:2]
    shade_box = _shade(P, k, hand, matte, held, ao_r)
    y0, y1, x0, x1 = max(0, ly0 - pad), min(H, ly1 + pad), max(0, lx0 - pad), min(W, lx1 + pad)
    if y1 - y0 < 2 * q or x1 - x0 < 2 * q:
        return shade_box
    gh, gw = (y1 - y0) // q, (x1 - x0) // q
    gy, gx = np.gradient(cv2.GaussianBlur(hq / q, (0, 0), max(0.8, CAUSTIC_SMOOTH * s / q)))
    gx, gy = gx[iy, ix], gy[iy, ix]
    nd = np.sqrt(gx * gx + gy * gy + 1.0)
    nx, ny, nz = -gx / nd, -gy / nd, 1.0 / nd
    (tx, ty, tz), _ = _refract(-L[0], -L[1], -L[2], nx, ny, nz, 1.0 / ETA[1])
    hc = hq[iy, ix] / q
    sp = 2.0 * hc / np.maximum(-tz, 0.25)
    (ox, oy, oz), kk = _refract(tx, ty, tz, -nx, -ny, nz, ETA[1])
    out = _ss(0.0, TIR_SOFT, kk)
    ox, oy, oz = out * ox + (1 - out) * tx, out * oy + (1 - out) * ty, out * oz + (1 - out) * tz
    cx_, cy_ = ix + (lx0 - x0) / q, iy + (ly0 - y0) / q
    dq = d / q
    wx, wy = cx_ + tx * sp + ox / np.maximum(-oz, 0.3) * dq, cy_ + ty * sp + oy / np.maximum(-oz, 0.3) * dq
    sx, sy = cx_ - L[0] / L[2] * (dq + 2 * hc), cy_ - L[1] / L[2] * (dq + 2 * hc)
    T = (1.0 - (F0 + (1 - F0) * (1 - nz) ** 5)) * front * small
    S = front * small
    lit = np.zeros((gh, gw), np.float32)
    edges = (0.0, CAUSTIC_BINS[0] * s, CAUSTIC_BINS[1] * s, np.inf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = np.flatnonzero((d >= lo) & (d < hi))
        if not sel.size:
            continue
        dm = float(np.median(dq[sel]))
        acc, rem = np.zeros((gh, gw), np.float32), np.zeros((gh, gw), np.float32)
        _splat(acc, wx[sel], wy[sel], T[sel])
        _splat(rem, sx[sel], sy[sel], S[sel])
        lit += cv2.GaussianBlur(acc, (0, 0), max(1.2, CAUSTIC_SOFT[0] * dm))  # the room's light is no point: a soft
        lit -= cv2.GaussianBlur(rem, (0, 0), max(2.0, CAUSTIC_SOFT[1] * dm))  # focus, a softer shadow
    if wh > 0:
        lit += wh * (1.0 - SPOT[2]) * _spot(gh, gw, q, x0, y0, cx, cy, R, dh, L, lobes, noise, k)
    g = 1.0 + CAUSTIC_GAIN * lit  # rolled off into its most (clipped, a focus was a flat disc with a hard edge)
    top = CAUSTIC_MAX - 1.0
    g = np.where(g > 1.0, 1.0 + top * np.tanh((g - 1.0) / top), np.maximum(g, 0.55)).astype(np.float32)
    g = cv2.resize(g, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    m = matte[y0:y1, x0:x1].astype(np.float32) / 255.0
    lift = CAUSTIC_KEEP * (g - 1.0)
    edge = _box_fade(y1 - y0, x1 - x0, max(4, pad // 2))  # what reaches the box's edge fades by it (faded over an
    # eighth of the box, the light moved when a drip grew the box)
    P[y0:y1, x0:x1] *= (1.0 + lift * m * edge)[..., None]
    if shade_box[1] <= shade_box[0]:
        return (y0, y1, x0, x1)
    return (min(y0, shade_box[0]), max(y1, shade_box[1]), min(x0, shade_box[2]), max(x1, shade_box[3]))


def _spot(gh: int, gw: int, q: int, x0: int, y0: int, cx: float, cy: float, R: float, dh: float, L: np.ndarray,
          lobes: tuple = (), noise: Noise | None = None, k: int = 0) -> np.ndarray:
    """(gh, gw): the held orb's light on the palm, in key light units on the grid of q px cells from (x0, y0): a ball
    lens of radius R at (cx, cy), its middle dh + R over the palm, lit along L. Where the line through its middle
    along the light meets the palm: the disc of its shadow, and inside it the spot it focuses the light it lets through
    into (as wide as its rays still are there, a ball's focus being n R / 2 (n - 1) from its middle), both blurred by
    the light's own width (CAUSTIC_SOFT). Traced cell by cell instead, through a surface that never holds still and
    onto a palm near its focus, the light was a jagged shape that jumped from frame to frame; so the surface's life is
    laid over it: the spot wobbles with the orb's ``lobes`` (magnified, SPOT_LOBES) and its light lies in a shifting net
    of brighter lines (SPOT_NET, read off ``noise`` on frame ``k``)."""
    D = dh + R
    c = np.array([cx, cy]) - L[:2] / L[2] * D
    ux = ((np.arange(gw) + 0.5) * q - 0.5 + x0 - c[0]) / q
    uy = ((np.arange(gh) + 0.5) * q - 0.5 + y0 - c[1]) / q
    r2 = (uy[:, None] ** 2 + ux[None, :] ** 2).astype(np.float32)
    Rq = R / q
    shadow = cv2.GaussianBlur((r2 <= Rq * Rq).astype(np.float32), (0, 0), max(2.0, CAUSTIC_SOFT[1] * D / q))
    focal = ETA[1] * R / (2.0 * (ETA[1] - 1.0))
    sig = math.hypot(SPOT[1] * R * abs(1.0 - D / focal), CAUSTIC_SOFT[0] * D) / q
    if lobes:
        th = np.arctan2(uy[:, None], ux[None, :])
        m = 1.0 + SPOT_LOBES * sum(np.float32(a) * np.cos(l * th - ph) for l, a, ph in lobes)
        r2 = r2 / np.clip(m, 0.6, 1.4).astype(np.float32) ** 2
    spot = SPOT[0] * (Rq / sig) ** 2 / 2.0 * np.exp(-r2 / (2.0 * sig * sig))
    if noise is not None:  # two nets drifting across each other: where their ridges cross, the light gathers
        gx_ = np.broadcast_to(((np.arange(gw) + 0.5) * q + x0)[None, :], (gh, gw)).astype(np.float32)
        gy_ = np.broadcast_to(((np.arange(gh) + 0.5) * q + y0)[:, None], (gh, gw)).astype(np.float32)
        v = SPOT_NET[2] * R
        size = SPOT_NET[1] * R
        net = (np.exp(-(noise.flow(9, gx_, gy_, v, 0.4 * v, k, size) / 0.38) ** 2)
               + np.exp(-(noise.flow(7, gx_, gy_, -0.6 * v, v, k, 1.3 * size) / 0.38) ** 2))
        core = np.exp(-r2 / (2.0 * sig * sig))  # in its core only (over all of a wide spot, a scrawl on the wrist)
        spot = spot * (1.0 + SPOT_NET[0] * core * (net / 0.519 - 1.0))  # (0.519: the net's mean over unit noise)
    return spot - shadow


def _shade(P: np.ndarray, k: int, hand: Hand, matte: np.ndarray, held: tuple | None, ao_r: float) -> tuple:
    """The room's soft light the held orb's bulk takes off the palm right under it, in place (SHADE_AO, at the orb's
    w of it), over a box of its own (shaded inside the light's box, it moved when that box did). Returns the box."""
    if held is None or held[4] <= 0 or ao_r <= 0:
        return (0, 0, 0, 0)
    cx, cy, cz, R, wh = held
    H, W = P.shape[:2]
    foot = np.array([cx, cy]) - hand.nrm[k, :2] * (R + ORB_GAP * float(hand.s[k]))  # the palm under it
    e = 2.0 * ao_r  # nothing left of it by 2 sigmas
    y0, y1 = max(0, int(foot[1] - e)), min(H, int(foot[1] + e) + 1)
    x0, x1 = max(0, int(foot[0] - e)), min(W, int(foot[0] + e) + 1)
    if y1 <= y0 or x1 <= x0:
        return (0, 0, 0, 0)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    q2 = ((xx - foot[0]) ** 2 + (yy - foot[1]) ** 2) / ao_r ** 2
    ao = SHADE_AO * wh * np.exp(-0.5 * q2) * np.clip(1.0 - q2 / 4.0, 0.0, 1.0) ** 2
    P[y0:y1, x0:x1] *= (1.0 - ao * matte[y0:y1, x0:x1].astype(np.float32) / 255.0)[..., None]
    return (y0, y1, x0, x1)


def _box_fade(h: int, w: int, px: int) -> np.ndarray:
    """(h, w): 1 inside, falling smoothly to 0 over ``px`` at the edges."""
    fy = _ss(0.0, px, np.minimum(np.arange(h), np.arange(h)[::-1]) + 0.5)
    fx_ = _ss(0.0, px, np.minimum(np.arange(w), np.arange(w)[::-1]) + 0.5)
    return np.outer(fy, fx_).astype(np.float32)


def bubbles(P: np.ndarray, items: list, layer: Layer, lit: np.ndarray, L: np.ndarray) -> None:
    """Air caught in the water, over it, in place: each (x, y, r, seen) a small sphere of air, its rim silvered (light
    turned back at it) inside a faint dark edge, its middle a touch brighter, a pin of glint towards the light; at
    ``seen`` of its strength, and only where the water covers the picture."""
    H, W = P.shape[:2]
    by0, by1, bx0, bx1 = layer.box
    for x, y, r, seen in items:
        if r < 0.8 or not (by0 <= y < by1 and bx0 <= x < bx1):
            continue
        wgt = float(layer.alpha[int(y) - by0, int(x) - bx0]) * seen
        rr = int(2 * r + 3)
        ya, yb, xa, xb = max(0, int(y) - rr), min(H, int(y) + rr + 1), max(0, int(x) - rr), min(W, int(x) + rr + 1)
        if yb <= ya or xb <= xa or wgt <= 0.01:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        d = np.hypot(xx - x, yy - y)
        ring = np.exp(-((d - r) / (0.22 * r + 0.6)) ** 2)
        core = np.clip(1 - d / r, 0, 1)
        gx, gy = x + 0.45 * r * L[0], y + 0.45 * r * L[1]
        pin = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * (0.18 * r + 0.4) ** 2))
        patch = P[ya:yb, xa:xb]
        edge = np.exp(-((d - 1.2 * r) / (0.15 * r + 0.5)) ** 2)
        patch *= (1.0 + 0.10 * wgt * core - 0.25 * wgt * edge)[..., None]
        patch += (wgt * (0.3 * ring + 0.8 * pin))[..., None] * lit[None, None, :]


# ---- free water: drops, the burst, spray ----------------------------------------------------------------------------

class Drops:
    """Water flying free of the hand, each piece on its own: under gravity and air drag, in the room's coordinates (the
    first frame's px) when the camera moves, so what was already thrown moves with the room, not with the screen, and
    seen in perspective from where it left, so it shrinks going away. A piece with a ``life`` shrinks away over its last
    ``fade`` frames (a ligament necking into drops, spray, a drip soaking into the palm); one with a ``delay`` flies
    unseen for that many frames first (a drop flying inside the ligament it breaks off)."""

    def __init__(self, cam: np.ndarray | None, size: tuple[int, int], s: float):
        self.cam, self.size, self.zcam = cam, size, ZCAM * s
        self.x = np.zeros((0, 3))  # room px and depth towards the camera
        self.v = np.zeros((0, 3))  # px a frame
        self.z0 = np.zeros(0)  # the depth it left at (its perspective is from there)
        self.info = np.zeros((0, 10))  # r, foam, born, life, fade, stretch, gravity (px a frame squared), delay,
        # free (1: it bridges to no other free piece), field weight

    def add(self, k: int, x, v, r, foam=0.0, life=np.inf, fade=1.0, stretch=1.0, g=None, delay=0.0, free=True,
            w=1.0) -> None:
        """Pieces leaving frame ``k`` at px and depth ``x`` (m, 3), ``v`` px a frame (m, 3)."""
        x, v = np.atleast_2d(np.asarray(x, np.float64)).copy(), np.atleast_2d(np.asarray(v, np.float64)).copy()
        m = len(x)
        if self.cam is not None:  # into the room's coordinates
            M = np.linalg.inv(self.cam[k])
            x[:, :2] = x[:, :2] @ M[:2, :2].T + M[:2, 2]
            v[:, :2] = v[:, :2] @ M[:2, :2].T
        g = G * self.zcam / ZCAM / FPS ** 2 if g is None else g
        info = np.stack(np.broadcast_arrays(np.zeros(m), r, foam, k, life, fade, stretch, g, delay, float(free), w)[1:],
                        -1).astype(np.float64)
        self.x, self.v = np.concatenate([self.x, x]), np.concatenate([self.v, v])
        self.z0 = np.concatenate([self.z0, x[:, 2]])
        self.info = np.concatenate([self.info, info])

    def step(self) -> None:
        self.v[:, 1] += self.info[:, 6]
        self.v *= DRAG
        self.x += self.v

    def blobs(self, k: int) -> list[Blob]:
        """The pieces as frame ``k`` sees them, moving over the shutter; the ones gone (spent, out of the picture) are
        dropped."""
        if not len(self.x):
            return []
        H, W = self.size
        xy, vxy = self.x[:, :2], self.v[:, :2]
        if self.cam is not None:
            M = self.cam[k]
            xy, vxy = xy @ M[:2, :2].T + M[:2, 2], vxy @ M[:2, :2].T
        kz = self.zcam / np.maximum(self.zcam - (self.x[:, 2] - self.z0), 0.2 * self.zcam)
        ctr = np.array([W / 2.0, H / 2.0])
        xy = ctr + (xy - ctr) * kz[:, None]
        r0, foam, born, life, fade, stretch, _, delay, free, w = self.info.T
        age = k - born
        r = r0 * kz * np.clip((life - age) / fade, 0.0, 1.0) ** 0.8
        vxy = vxy * kz[:, None]
        z = self.x[:, 2]
        out = []
        for j in np.flatnonzero((age < life) & (age >= delay) & (r >= 0.5)):
            shape, move = _moving(r[j], vxy[j], stretch[j])
            out.append(Blob(xy[j, 0], xy[j, 1], z[j], r[j], shape, vxy[j, 0], vxy[j, 1], 0.0, foam[j], w=w[j],
                            free=bool(free[j]), move=move))
        keep = (age < life) & (xy[:, 1] < H + 4 * r0 * kz + 8) & (xy[:, 0] > -4 * r0 * kz - 8) & \
               (xy[:, 0] < W + 4 * r0 * kz + 8)
        self.x, self.v, self.z0, self.info = self.x[keep], self.v[keep], self.z0[keep], self.info[keep]
        return out


def _sizes(rng, m: int, lo: float, hi: float, alpha: float) -> np.ndarray:
    """``m`` sizes between ``lo`` and ``hi`` drawn from a power law (the share over r falls as r^-alpha): many small,
    few big, as a splash's drops are."""
    u = rng.uniform(0.0, 1.0, m)
    return lo * (1.0 - u * (1.0 - (lo / hi) ** alpha)) ** (-1.0 / alpha)


def splash(drops: Drops, k: int, c: np.ndarray, R: float, v: np.ndarray, size: float, s: float, rng,
           flat: np.ndarray | None = None, g: float | None = None) -> list[Blob]:
    """A ball of water of radius ``R`` at ``c`` (px, depth) moving ``v`` (px a frame) bursting on frame ``k``: on that
    frame its body swells, wrinkled and frothing; then it flies apart, on along its motion over a wide cone: its torn
    skin in ligaments that bead up into rows of drops, drops of every size (a few lumps, a dozen mid drops, many small
    ones, a fine spray of specks that glint), each piece on its own under gravity ``g`` (px a frame squared; None: at
    the hand's depth) from the first frame, blurred by its speed. ``s``: a hand's size px where it bursts. ``flat``:
    the normal of a surface it burst on (the palm, the wall), along which nothing goes in and little goes out: it is
    flattened against it and spreads over it. Returns the body, as frame ``k`` sees it."""
    sp = float(np.linalg.norm(v))
    ahead = v / sp if sp > 1e-6 else np.zeros(3)
    if flat is not None:
        ahead = ahead - (ahead @ flat) * flat
    carry = SPLASH_CARRY * (v if flat is None else v - (v @ flat) * flat)

    def dirs(m: int, cone: float = SPLASH_CONE) -> np.ndarray:
        d = _unit(rng.normal(size=(m, 3)) + cone * ahead)  # on along its motion, over a cone (thrown out evenly all
        if flat is not None:  # round, it was a starburst)
            d = _unit(d - (d @ flat)[:, None] * flat + 0.3 * np.abs(d @ flat)[:, None] * flat)
        return d

    pace = s / FPS * (0.6 + 0.4 * size)
    n_drop, n_lig, n_spray, n_lump = (max(2, int(a * (0.4 + 0.6 * size))) for a in SPLASH)
    swell, foam, wrinkles = BURST_SWELL
    shape, move = _moving(swell * R, v[:2])
    body = [Blob(c[0], c[1], c[2], swell * R, shape, v[0], v[1], foam=foam, move=move,
                 modes=tuple((l, a, rng.uniform(0, 2 * math.pi)) for l, a in wrinkles))]
    if flat is not None:  # flattened against what it hit: a wide, thin splat
        body = [body[0]._replace(r=SPLAT[0] * R, flat=SPLAT[1], foam=SPLAT[2], move=(),
                                 modes=tuple((l, a, ph + rng.uniform(0, 2 * math.pi)) for l, a, ph in SPLAT[3]))]
    d = dirs(n_lig)  # its torn skin: ligaments flung out of it, each beading up into a row of drops a frame on
    rl = R * rng.uniform(*LIGAMENT_R, n_lig)
    st = rng.uniform(*LIGAMENT_STRETCH, n_lig)
    x0 = c + d * (R * rng.uniform(0.9, 1.1, (n_lig, 1)) + (st * rl)[:, None])
    v0 = carry + d * rng.uniform(3.5, 6.0, (n_lig, 1)) * pace
    drops.add(k, x0, v0, rl, foam=0.35, life=1 + LIGAMENT_LIFE, fade=1.5, stretch=st, delay=1, g=g)
    per = 4
    off = np.linspace(-1.0, 1.0, per)[None, :, None] * (0.6 * st * rl)[:, None, None] * d[:, None, :]
    drops.add(k, (x0[:, None, :] + off).reshape(-1, 3), (v0[:, None, :] + 0.12 * off).reshape(-1, 3),
              np.repeat(0.75 * rl, per) * rng.uniform(0.6, 1.1, n_lig * per), foam=0.15, delay=LIGAMENT_LIFE, g=g)
    r = R * _sizes(rng, n_drop, *SPLASH_R[0], 1.2)  # its drops: the smaller, the faster they leave
    d = dirs(n_drop)
    speed = rng.uniform(*SPLASH_SPEED, (n_drop, 1)) * (SPLASH_R[0][0] * R / r[:, None]) ** 0.25
    drops.add(k, c + d * R * rng.uniform(0.5, 0.95, (n_drop, 1)), carry + d * speed * pace, r,
              foam=np.where(r < 0.05 * R, 0.3, 0.08), delay=1, g=g)
    d = dirs(n_spray, 0.6 * SPLASH_CONE)  # a fine spray of specks, glinting
    drops.add(k, c + d * swell * R, carry + d * rng.uniform(6.0, 14.0, (n_spray, 1)) * pace,
              R * rng.uniform(*SPLASH_R[1], n_spray), foam=0.0, life=rng.uniform(3, SPRAY_LIFE, n_spray),
              fade=SPRAY_LIFE / 2, delay=1, g=g)
    d = dirs(n_lump)  # and a few big, slow lumps of it
    drops.add(k, c + d * R * rng.uniform(0.3, 0.7, (n_lump, 1)), carry + d * rng.uniform(1.5, 3.5, (n_lump, 1)) * pace,
              R * rng.uniform(*SPLASH_R[2], n_lump), foam=0.12, delay=1, g=g)
    return body


# ---- the orb --------------------------------------------------------------------------------------------------------

class _Spring:
    """A second-order follower: the water's centre trails the hand and overshoots, as a mass of water would."""

    def __init__(self, x: np.ndarray, hz: float, zeta: float):
        self.x, self.v = np.asarray(x, np.float64).copy(), np.zeros(len(x))
        self.w, self.z = 2 * math.pi * hz, zeta

    def step(self, target: np.ndarray) -> np.ndarray:
        """One frame towards ``target``; returns the acceleration (px a second squared)."""
        h = 1.0 / FPS / 4
        for _ in range(4):
            a = self.w ** 2 * (target - self.x) - 2 * self.z * self.w * self.v
            self.v += a * h
            self.x += self.v * h
        return a


def _rotate(u: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    """Points ``u`` (m, 3) turned ``ang`` rad round unit ``axis`` (Rodrigues)."""
    c, s = math.cos(ang), math.sin(ang)
    return u * c + np.cross(axis, u) * s + np.outer(u @ axis, axis) * (1 - c)


def _air(bub: np.ndarray, bub_r: np.ndarray, rise: np.ndarray, axis: np.ndarray, circ: float, t: float,
         c: np.ndarray, r: float, s: float, seen: float) -> list[tuple]:
    """The air caught in the orb (unit offsets ``bub``) carried round by its circulation and rising through it at
    ``rise`` turns of its height a second (each wells up at its bottom and is gone at its top): (x, y, r, seen) each."""
    out = []
    for (bxu, byu, bzu), br, up in zip(_rotate(bub, axis, circ), bub_r, rise):
        lim = math.sqrt(max(0.72 ** 2 - bxu * bxu - bzu * bzu, 0.02))
        hgt = ((lim - byu) / (2.0 * lim) + up * t) % 1.0  # 0 at its bottom, 1 at its top
        byu = lim - 2.0 * lim * hgt
        grow = 1.0 + 0.12 * (1.0 - bzu)  # the far ones, seen through the water, a little magnified...
        rim = math.hypot(bxu, byu) * grow  # ... and those seen near its rim, through its grazing surface, gone (they sat
        vis = (0.75 + 0.25 * bzu) * (1.0 - float(_ss(0.55, 0.85, rim)))  # on its outline like pearls)
        vis *= min(1.0, 5.0 * hgt, 5.0 * (1.0 - hgt))
        out.append((c[0] + r * bxu * grow, c[1] + r * byu * grow, br * s, vis * seen))
    return out


def _wall(scenes: list, hit: tuple, n: int, cam: np.ndarray | None, rng) -> None:
    """What the orb leaves where it burst on the wall, onto the scenes after the frame ``hit`` = (frame, x, y, z, R, s)
    (px on that frame, the ball's radius and a hand's size there): a wet patch, darker, and drips running down from
    its lower edge, each leaving a wet trail; held to the room as the camera moves."""
    kh, xh, yh, zh, R, s = hit
    to_room = np.linalg.inv(cam[kh]) if cam is not None else np.eye(3)
    m = WALL_DRIPS[0]
    ang = rng.uniform(0.25 * math.pi, 0.75 * math.pi, m)  # from its lower edge
    start = np.stack([xh + WET[1] * R * 0.8 * np.cos(ang), yh + WET[1] * R * 0.8 * np.sin(ang)], 1)
    late = rng.uniform(0.0, WALL_DRIPS[3], m)
    vmax = WALL_DRIPS[2] * s / FPS * rng.uniform(0.5, 1.0, m)
    rd = WALL_DRIPS[1] * R * rng.uniform(0.7, 1.2, m)
    lobes = float(rng.uniform(0.0, 1000.0))
    for j, k in enumerate(range(kh + 1, min(n, kh + 1 + SHEET[0]))):  # the splat spreads on over the wall, thinning
        sc = scenes[k]
        q = (cam[k] @ to_room if cam is not None else np.eye(3)) @ np.array([xh, yh, 1.0])
        sc.blobs.append(Blob(q[0] / q[2], q[1] / q[2], zh, R * (SPLAT[0] + SHEET[1] * (j + 1)), foam=SPLAT[2] * 0.6,
                             flat=SPLAT[1] * SHEET[2] ** (j + 1),
                             modes=tuple((l, a, ph + lobes) for l, a, ph in SPLAT[3])))
    for k in range(kh + 1, n):
        M = cam[k] @ to_room if cam is not None else np.eye(3)

        def at(x, y):
            q = M @ np.array([x, y, 1.0])
            return q[0] / q[2], q[1] / q[2]

        sc = scenes[k]
        grow = 1.0 + 0.15 * float(_ss(0.0, 6.0, k - kh))  # it soaks out a little
        x, y = at(xh, yh)
        sc.wet.append((x, y, WET[1] * R * grow, WET[1] * R * 0.85 * grow, 0.0, WET[0], False, lobes))
        tau = (k - kh) / FPS - late
        for j in np.flatnonzero(tau > 0):
            T = 0.4  # s: it takes this long to get up to speed, held back by the wall
            fall = vmax[j] * FPS * (tau[j] ** 2 / (2 * T) if tau[j] < T else tau[j] - T / 2)
            x0, y0 = at(*start[j])
            x1, y1 = at(start[j][0], start[j][1] + fall)
            v = vmax[j] * min(1.0, tau[j] / T)
            sc.blobs.append(Blob(x1, y1, zh, rd[j], _shape(np.array([0.0, 1.0]), 1.5), 0.0, v, free=True,
                                 move=(0.0, v * SHUTTER), flat=0.5))
            sc.wet.append((0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.9 * rd[j], 0.5 * (y1 - y0) + rd[j], 0.0, 0.8 * WET[0],
                           False, lobes + j))


def orb_scenes(tl: Timeline, track: dict, hand: Hand, size: tuple[int, int], seed: int,
               cam: np.ndarray | None) -> list[Scene]:
    """The orb, frame by frame (see the module's docstring and orb_plan)."""
    n = tl.n
    H, W = size
    scenes = [Scene() for _ in range(n)]
    plan = orb_plan(tl, track)
    g0, g1, rel = plan.gather, plan.formed, plan.release
    if g0 >= n:
        return scenes
    rng = _rng(seed, _ORB)
    e = np.clip(_tsmooth(tl.energy, VOICE_SMOOTH), 0, 1)
    drops = Drops(cam, size, float(np.median(hand.s)))
    gather = [(rng.uniform(0, 2 * math.pi), rng.uniform(*GATHER_FROM), rng.uniform(*GATHER_R), rng.uniform(-0.8, 0.8),
               rng.uniform(0.0, 0.15), rng.uniform(g0 + 0.6 * (g1 - g0), g1), rng.choice([-1.0, 1.0]))
              for _ in range(GATHER_DROPS)]
    bub = _unit(rng.normal(size=(BUBBLES, 3))) * rng.uniform(0.15, 0.7, (BUBBLES, 1)) ** (1 / 3)
    bub_r = rng.uniform(*BUBBLE_R, BUBBLES)
    rise = BUBBLE_RISE * rng.uniform(0.5, 1.5, BUBBLES) * (bub_r / BUBBLE_R[1]) ** 0.5  # the bigger, the faster
    axis = _unit(np.array([rng.uniform(-1, 1), rng.uniform(-0.4, 0.4), rng.uniform(0.3, 1.0)]))
    phase = {l: (rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi), rng.uniform(-0.6, 0.6)) for l in (3, 4)}
    jolts = dict(plan.jolts)
    jolt_at: list[tuple[int, float]] = []  # (frame, size): for the lobes' ringing
    rings: list[tuple] = []  # (frame, unit offset from the centre, size)
    spring = None
    m2, m2v = np.zeros(2), np.zeros(2)
    circ = 0.0
    modes: list[tuple] = []
    next_drip = g1 + int(rng.integers(*DRIP_EVERY))
    fly = None  # the orb in the air: (frame, centre, velocity, radius, thrown, the palm's normal, the wall's depth);
    # centre None: burst
    for k in range(g0, n):
        sc = scenes[k]
        p, nrm, s = hand.p[k], hand.nrm[k], float(hand.s[k])
        R = ORB_R * s
        t = k / FPS
        sc.wave = WAVE[0] + WAVE[1] * float(e[k])
        circ += (CIRCULATE[0] + CIRCULATE[1] * float(e[k])) / FPS
        if rel is None or k < rel:
            zp = float(hand.depth(k, np.float32([p[0]]), np.float32([p[1]]))[0])
            bob = ORB_BOB[0] * s * math.sin(2 * math.pi * ORB_BOB[1] * t)
            target = np.array([p[0], p[1], zp]) + nrm * (R + ORB_GAP * s + bob)
            if spring is None:
                spring = _Spring(target, *ORB_SPRING)
            acc = spring.step(target)
            c = spring.x
            if k in jolts:  # a hit while it is held jolts it, sends a ripple over it and spits drops
                z = jolts[k]
                ang = rng.uniform(0, 2 * math.pi)
                m2v += JOLT[0] * z * 2 * math.pi * MODE_HZ[2] * np.array([math.cos(ang), math.sin(ang)])
                jolt_at.append((k, z))
                off = _unit(rng.normal(size=2)) * rng.uniform(*RING_FROM)
                rings.append((k, np.array([off[0], off[1], math.sqrt(1.0 - off @ off)]), z))
                m = max(1, int(round(JOLT[1] * z)))
                d = _unit(rng.normal(size=(m, 3)) + 0.6 * nrm)
                drops.add(k, c + d * R * 1.15, spring.v / FPS + d * rng.uniform(2.5, 4.5, (m, 1)) * s / FPS,
                          R * rng.uniform(0.03, 0.05, m))  # (spat from inside its rim, they sat on it like seeds)
            w2 = 2 * math.pi * MODE_HZ[2]
            drive = -acc[:2] / max(s, 1.0) * MODE2_DRIVE
            m2v += (-(w2 ** 2) * m2 - 2 * MODE2_DAMP * w2 * m2v + drive * w2 ** 2) / FPS
            m2 += m2v / FPS
            vol = float(_ss(g0 + 0.15 * (g1 - g0), g1, k)) if k < g1 else 1.0
            r = R * vol ** (1 / 3)
            st = float(np.hypot(*m2))
            u2 = m2 / st if st > 1e-6 else np.array([1.0, 0.0])
            k2 = 1.0 + min(MODE2_MAX, st)
            modes = []
            for l in (3, 4):
                ph0, psi, drift = phase[l]
                wl = 2 * math.pi * MODE_HZ[l]
                amp = MODE_AMP[l] * (MODE_REST + (1 - MODE_REST) * float(e[k])) * math.sin(wl * t + psi)
                amp += sum(1.5 * MODE_AMP[l] * z * math.exp(-(k - kj) / 8.0) * math.sin(wl * (k - kj) / FPS)
                           for kj, z in jolt_at)
                modes.append((l, amp, ph0 + drift * t))
            v_c = spring.v / FPS
            if r > 0.5:
                sc.blobs.append(Blob(c[0], c[1], c[2], r, _shape(u2, k2, 1.0 / math.sqrt(k2)), v_c[0], v_c[1],
                                     (CIRCULATE[0] + CIRCULATE[1] * float(e[k])) / FPS, 0.0, tuple(modes),
                                     move=tuple(v_c[:2] * SHUTTER)))
                sc.held = (c[0], c[1], c[2], r, vol)  # its shade comes on as it gathers (at full strength once a
                # third of it was there, it popped on with only a few drops in the air)
                sc.lobes = ((2, k2 - 1.0, math.atan2(u2[1], u2[0])), *modes)
            for ang, d0, rs, z0, ts, ta, spin in gather:  # drops coming together out of the air, spiralling in
                ts_ = g0 + ts * (g1 - g0)
                if not ts_ <= k < ta:
                    continue
                u = (k - ts_) / max(ta - ts_, 1.0)
                rad = d0 * (1 - u) ** 1.2 * R  # drawn in, slowing as it takes them (rushing in at the end, smeared
                # into glass seeds round it)
                a = ang + spin * 1.3 * u
                dd = np.array([math.cos(a), math.sin(a)])
                pos = c[:2] + rad * dd
                vel = -dd * 1.2 * d0 * R * (1 - u) ** 0.2 / max(ta - ts_, 1.0) + v_c[:2]
                rr = rs * R * float(_ss(0.0, 0.3, u))
                if rad > 0.9 * r and rr > 0.5:  # taken in as it touches it (drawn on over it, the drops arriving
                    # sat on the ball like glass seeds)
                    shape, move = _moving(rr, vel)
                    sc.blobs.append(Blob(pos[0], pos[1], c[2] + z0 * R * (1 - u), rr, shape, vel[0], vel[1],
                                         free=True, move=move))
            if k >= g1:  # air inside, carried round as it circulates; then a drip off its underside, now and then
                sc.bubbles += _air(bub, bub_r, rise, axis, circ, t, c, r, s, 1.0)
                if k >= next_drip:
                    u = (k - next_drip) / DRIP_FORM
                    down = -nrm
                    rd = DRIP_R * s
                    if u < 1.0:  # it swells under the orb, hanging off it by a neck that thins as it lets go
                        tip = c + down * (r + rd * (0.4 + 1.6 * u))
                        sc.blobs.append(Blob(*tip, rd * (0.6 + 0.4 * u), vx=v_c[0], vy=v_c[1]))
                        neck = rd * 0.55 * (1.0 - u) ** 0.7
                        if neck > 0.5:
                            sc.blobs.append(Blob(*(c + down * (r + rd * 0.6 * u)), neck, vx=v_c[0], vy=v_c[1]))
                    else:  # it lets go and falls onto the palm under it
                        fall = (ORB_GAP * s + 0.1 * r) * float(np.hypot(*down[:2]))
                        life = max(2.0, FPS * math.sqrt(2 * max(fall, 1.0) / (G * s)))
                        drops.add(k, c + down * (r + 2.0 * rd), v_c + down * 0.4 * s / FPS, rd, life=life, fade=2)
                        next_drip = k + DRIP_FORM + int(rng.integers(*DRIP_EVERY))
            for kj, u3, z in rings:
                if k - kj < 3 * RING[3]:
                    sc.rings.append((c[0], c[1], r, *u3, k - kj, RING[0] * z, RING[1] * r, RING[2] * r))
        elif fly is None or fly[1] is not None:  # let go and not yet burst
            if fly is None:  # let go: thrown along the aim, or dropped onto the palm
                c = spring.x if spring is not None else np.array([p[0], p[1], 0.0]) + nrm * (R + ORB_GAP * s)
                v0 = spring.v / FPS if spring is not None else np.zeros(3)
                if plan.throw:
                    aim = _aim(tl, hand, k)
                    sp = THROW_SPEED * s / FPS * (0.6 + 0.4 * plan.size)
                    v = 0.5 * v0 + np.array([aim[0] * sp, aim[1] * sp, -RECEDE * sp])
                else:
                    v = v0
                zp = float(hand.depth(k, np.float32([p[0]]), np.float32([p[1]]))[0])
                fly = (k, c.copy(), v, R, plan.throw, nrm.copy(), zp - ROOM_DEPTH * max(H, W))
                left = (c.copy(), R)  # where it left the palm: its shade fades there
            k0, c0, v, R0, thrown, nrm0, zw = fly
            tau = k - k0
            gone = hit = False
            if thrown:
                go = tau * tau / (2 * TAKEOFF) if tau < TAKEOFF else tau - TAKEOFF / 2
                c = c0 + v * go + np.array([0.0, 0.5 * G * s / FPS ** 2 * tau * tau, 0.0])
                vel = v * min(1.0, tau / TAKEOFF) + np.array([0.0, G * s / FPS ** 2 * tau, 0.0])  # (at full speed
                # from the frame it was let go on, it was drawn stretched into a long tear lying over the palm)
                hit = c[2] <= zw  # it reached the wall
                if hit:
                    c[2] = zw
                kz = ZCAM * s / max(ZCAM * s - (c[2] - c0[2]), 0.2 * ZCAM * s)
                xy = np.array([W / 2.0, H / 2.0]) + (c[:2] - np.array([W / 2.0, H / 2.0])) * kz
                gone = min(xy[0] + R0 * kz, W + R0 * kz - xy[0], xy[1] + R0 * kz, H + R0 * kz - xy[1]) < 0
                done = hit or gone
                hold = max(0.0, 1.0 - tau / (TAKEOFF + 2))  # it takes its light off the palm as it goes
            else:  # falling onto the palm under it
                c = c0 + v * tau - nrm0 * 0.5 * G * s / FPS ** 2 * tau * tau
                vel = v - nrm0 * G * s / FPS ** 2 * tau
                kz, xy = 1.0, c[:2]
                done = float((c0 - c) @ nrm0) >= ORB_GAP * s
                hold = max(0.0, 1.0 - tau / 3.0)
            sc.held = (left[0][0], left[0][1], left[0][2], left[1], hold)
            if done and not gone:  # burst: from here on only its drops are left
                at = np.array([xy[0], xy[1], c[2]])
                if hit:  # flat on the wall: its drops fly out along it, fanned the way it was thrown, and it leaves
                    # a wet patch and drips (spread evenly round where it hit, it was a starburst)
                    vh = vel * kz
                    vh[:2] += FAN * abs(vh[2]) * v[:2] / max(float(np.hypot(*v[:2])), 1e-6)
                    sc.blobs += splash(drops, k, at, R0 * kz, vh, plan.size, s * kz, rng, np.array([0., 0., 1.]),
                                       G * s * kz / FPS ** 2)
                    _wall(scenes, (k, xy[0], xy[1], c[2], R0 * kz, s * kz), n, cam, rng)
                else:
                    sc.blobs += splash(drops, k, at, R0 * kz, vel, plan.size, s, rng, nrm0)
            if done:
                fly = (k0, None, v, R0, thrown, nrm0, zw)
            else:  # in the air: stretched a little along its flight, and a neck pulled off its back pinching off
                fade = max(0.0, 1.0 - tau / TAKEOFF)  # its wobble and its air go as it is flung (dropped on the
                # frame it was let go, the ball went still and empty at once)
                vs = vel[:2] * kz
                sp = float(np.hypot(*vs))
                u = vs / max(sp, 1e-6)
                stretch = 1.0 + min(THROWN_STRETCH, sp / (2.5 * R0)) if thrown else 1.0
                shape, move = _moving(R0 * kz, vs, stretch)
                sc.blobs.append(Blob(xy[0], xy[1], c[2], R0 * kz, shape, vs[0], vs[1], CIRCULATE[0] / FPS,
                                     modes=tuple((l, a * fade, ph) for l, a, ph in modes), move=move))
                if fade > 0:
                    sc.bubbles += _air(bub, bub_r, rise, axis, circ, t, xy, R0 * kz, s * kz, fade)
                if thrown and 1 <= tau <= NECK[0]:
                    a = tau / NECK[0]  # how far it has pinched off
                    rn = R0 * kz * 0.3
                    for j in (1, 2, 3):
                        tp = xy - u * R0 * kz * (0.75 + 0.4 * j * (0.6 + 0.4 * a))
                        rt = rn * (1.0 - 0.18 * j) * (1.0 - 0.75 * a * (j == 1))  # thinnest next to the ball
                        _, mv = _moving(rt, vs * (1.0 - 0.1 * j))
                        if tau < NECK[0]:
                            sc.blobs.append(Blob(tp[0], tp[1], c[2], rt, (1.0, 0.0, 0.0, 1.0), vs[0], vs[1],
                                                 foam=0.1, move=mv))
                    if tau == NECK[0]:  # pinched off: its drops follow, slower
                        tp = xy[None, :] - u[None, :] * R0 * kz * (0.75 + 0.4 * np.arange(1, NECK[1] + 1)[:, None])
                        life = (c[2] - zw) / max(-0.6 * vel[2], 1e-3)  # (they wet the wall where it does)
                        drops.add(k, np.c_[tp, np.full(NECK[1], c[2])], np.tile(np.r_[0.6 * vs, 0.6 * vel[2]],
                                                                                 (NECK[1], 1)),
                                  NECK[2] * rn * rng.uniform(0.7, 1.2, NECK[1]), life=life, fade=2)
        sc.blobs += drops.blobs(k)
        drops.step()
    return scenes


# ---- the whip -------------------------------------------------------------------------------------------------------

def _chain(pts: np.ndarray, per: int) -> np.ndarray:
    """A Catmull-Rom curve through ``pts`` (m, d), ``per`` points a span."""
    P = np.concatenate([pts[:1], pts, pts[-1:]])
    t = np.arange(per)[:, None] / per
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1:])
    return np.concatenate(out)


def _swirl(hand: Hand, k: int, root: np.ndarray, u: np.ndarray, phase: float, sweep: float, t: float) -> np.ndarray:
    """(m, 3): the shape the stream holds on frame ``k`` at shares ``u`` of its length: out of the palm at ``root``,
    swung out over WHIP_RISE of it to a circle round the hand (facing the camera, tipped so one side goes behind the
    hand), wrapped ``sweep`` turns round it from ``phase`` rad, its lobes running round it (``t`` seconds in)."""
    s = float(hand.s[k])
    along = hand.along[k]
    tilt = math.radians(WHIP_TILT)
    axis = _unit(WHIP_FACE * np.array([0.0, 0.0, 1.0]) + (1.0 - WHIP_FACE) * hand.nrm[k])
    axis = _unit(axis * math.cos(tilt) + along * math.sin(tilt))
    e1 = _unit(along - (along @ axis) * axis)
    e2 = np.cross(axis, e1)
    th = phase + 2.0 * math.pi * sweep * u
    rho = WHIP_ORBIT * s * _ss(0.0, WHIP_RISE, u)
    rho = rho * (1.0 + WHIP_WOBBLE[0] * np.sin(WHIP_WOBBLE[1] * th - 2.0 * math.pi * WHIP_WOBBLE[2] * t))
    return root + rho[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)


def _lash(root: np.ndarray, aim: np.ndarray, u: np.ndarray, s: float, age: float) -> np.ndarray:
    """(m, 3): the stream lashed out straight along ``aim`` (unit, 3) by a hit ``age`` frames ago, a kink running down
    it from the root to the tail as a whip's loop does."""
    side = _unit(np.cross(aim, np.array([0.0, 0.0, 1.0])) + 1e-9)
    at = min(1.0, age / (0.6 * WHIP_CRACK[0]))
    kink = WHIP_CRACK[2] * s * np.exp(-((u - at) / 0.15) ** 2) * (1.0 - at)
    return root + np.outer(u * WHIP_CRACK[1] * s, aim) + np.outer(kink, side)


def _stream(pts: np.ndarray, vel: np.ndarray, rad: np.ndarray, foam: np.ndarray, flow: np.ndarray,
            arc: np.ndarray) -> list[Blob]:
    """Blobs along a stream through ``pts`` (m, 3) moving ``vel`` (px a frame), its radius ``rad``, ``arc`` px along
    it on the picture: at most WHIP_SPACING of their radius apart, each weighted by its share of the stream
    (_tube_weight), so the stream is a tube of that radius, blurred by its motion over the shutter."""
    keep = [0]
    for i in range(1, len(pts)):
        if arc[i] - arc[keep[-1]] >= WHIP_SPACING * rad[i] or i == len(pts) - 1:
            keep.append(i)
    keep = np.array(keep)
    gaps = np.diff(arc[keep])
    gap = 0.5 * (np.r_[gaps[:1], gaps] + np.r_[gaps, gaps[-1:]]) if gaps.size else np.ones(1)
    tang = np.gradient(pts[keep, :2], axis=0) if len(keep) > 1 else np.array([[1.0, 0.0]])
    out = []
    for j, i in enumerate(keep):
        if rad[i] < 0.5:
            continue
        t = tang[j] / max(float(np.hypot(*tang[j])), 1e-9)
        shape, move = _moving(rad[i], vel[i, :2])
        w = _tube_weight(shape, rad[i] * SIG_K, t, max(gap[j], 1e-3))
        out.append(Blob(pts[i, 0], pts[i, 1], pts[i, 2], rad[i], shape, flow[i, 0], flow[i, 1], foam=float(foam[i]),
                        w=w, move=move, stream=(float(arc[i]), float(t[0]), float(t[1]))))
    return out


def _pool(hand: Hand, k: int, r: float, t: float) -> tuple[Blob, tuple]:
    """The pool of radius ``r`` on the palm the stream rises out of on frame ``k`` (``t`` seconds in), flat on it (seen
    squashed as the palm turns from the camera), its outline rippling and churning: (its blob, the wet patch round it
    on the palm: Scene.wet)."""
    p, nrm = hand.p[k], hand.nrm[k]
    zp = float(hand.depth(k, np.float32([p[0]]), np.float32([p[1]]))[0])
    c = np.array([p[0], p[1], zp]) + nrm * 0.5 * WHIP_POOL[1] * r
    nxy = float(np.hypot(*nrm[:2]))
    un = nrm[:2] / nxy if nxy > 1e-6 else np.array([0.0, 1.0])
    squash = math.sqrt(nrm[2] ** 2 + WHIP_POOL[1] ** 2 * (1.0 - nrm[2] ** 2))  # a disc on the palm, seen from the camera
    a = WHIP_POOL[3]
    modes = ((3, a * math.sin(2 * math.pi * 2.3 * t), 1.7 * t),
             (5, 0.6 * a * math.sin(2 * math.pi * 3.7 * t + 1.0), -2.3 * t),
             (7, 0.4 * a * math.sin(2 * math.pi * 5.1 * t), 3.1 * t))
    blob = Blob(c[0], c[1], c[2], r, _shape(un, squash, 1.0), spin=CIRCULATE[0] / FPS, foam=WHIP_POOL[2], modes=modes,
                flat=WHIP_POOL[1])
    wet = (c[0], c[1], WHIP_WET * r, WHIP_WET * r * squash, math.atan2(un[1], un[0]) + 0.5 * math.pi, 0.08, True, 0.0)
    return blob, wet


def whip_scenes(tl: Timeline, track: dict, hand: Hand, size: tuple[int, int], seed: int,
                cam: np.ndarray | None) -> list[Scene]:
    """The whip, frame by frame (see the module's docstring and whip_plan): each node of the stream is held to the
    swirl it makes round the hand (_swirl) by a spring, stiff at the root and soft at the tail, so when the hand
    moves the stream trails it like a ribbon; a hit lashes it out along the aim (_lash). It rises, churning, out of a
    pool on the palm, swells and thins along its length as bulges run down it, thins to its tail and frays there into
    beads flowing out along it; fast, it froths, sheds drops and sprays off its tail. It is worked in the picture's px
    (the hand and a first-person camera move together); the drops it sheds fly in the room's."""
    n = tl.n
    scenes = [Scene() for _ in range(n)]
    plan = whip_plan(tl, track)
    if plan.start >= n:
        return scenes
    rng = _rng(seed, _WHIP)
    e = np.clip(_tsmooth(tl.energy, VOICE_SMOOTH), 0, 1)
    drops = Drops(cam, size, float(np.median(hand.s)))
    N = WHIP_NODES
    u_n = np.arange(N) / (N - 1)
    sub = 4
    dt = 1.0 / FPS / sub
    x = v = root0 = None
    phase = rng.uniform(0.0, 2.0 * math.pi)
    air = np.stack([rng.uniform(0.0, 1.0, WHIP_AIR), rng.uniform(-0.45, 0.45, WHIP_AIR),
                    rng.uniform(*WHIP_AIR_R, WHIP_AIR)], 1)  # its air: where along it, across it, how big (of it)
    for k in range(plan.start, n):
        sc = scenes[k]
        p, nrm, s = hand.p[k], hand.nrm[k], float(hand.s[k])
        zp = float(hand.depth(k, np.float32([p[0]]), np.float32([p[1]]))[0])
        root = np.array([p[0], p[1], zp]) + nrm * WHIP_LIFT * s
        alive = plan.end is None or k < plan.end
        broke = 0 if alive else k - plan.end  # frames since it let go
        grow = float(_ss(plan.start, plan.start + WHIP_GROW, k))
        t = k / FPS
        pool_r = WHIP_POOL[0] * WHIP_R[0] * s * max(grow, 0.3) ** 0.5 * (1.0 - float(_ss(0, 2 * WHIP_BREAK, broke)))
        if pool_r > 0.5:  # the pool on the palm, draining away once it lets go
            blob, wet = _pool(hand, k, pool_r, t)
            sc.blobs.append(blob)
            sc.wet.append(wet)
        sweep = WHIP_SWEEP[0] + (WHIP_SWEEP[1] - WHIP_SWEEP[0]) * float(e[k])
        phase += 2.0 * math.pi * (WHIP_SPIN[0] + WHIP_SPIN[1] * float(e[k])) / FPS
        u_t = u_n * max(grow, 0.02)  # poured out: the nodes lie along the part of the swirl out so far
        target = _swirl(hand, k, root, u_t, phase, sweep, t)
        crack = 0.0
        for kc, z in plan.cracks:
            age = k - kc
            if 0 <= age < WHIP_CRACK[0]:
                w = min(1.0, (age + 1) / 2.0) * float(_ss(WHIP_CRACK[0], 0.5 * WHIP_CRACK[0], age)) * min(1.0, z)
                if w > crack:
                    crack, crack_age = w, age
        if crack > 0:
            aim = _aim(tl, hand, k)
            aim3 = _unit(np.array([aim[0], aim[1], -0.2]))
            target = (1.0 - crack) * target + crack * _lash(root, aim3, u_t, s, crack_age)
        if x is None:
            x, v, root0 = target.copy(), np.zeros((N, 3)), root
        sc.wave = WAVE[0] + WAVE[1] * float(e[k])
        if broke <= WHIP_BREAK and len(x):
            om = 2.0 * math.pi * (WHIP_HZ[0] + (WHIP_HZ[1] - WHIP_HZ[0]) * u_n ** 0.7) * (1.0 + (WHIP_CRACK[3] - 1.0)
                                                                                         * crack)
            for j in range(sub):
                if alive:
                    a = om[:, None] ** 2 * (target - x) - 2.0 * WHIP_DAMP * om[:, None] * v
                    a[:, 1] += WHIP_SAG * G * s
                else:
                    a = np.zeros((N, 3))
                    a[:, 1] += G * s
                v = v + a * dt
                x = x + v * dt
                if alive:  # the root rides the palm (between this frame's place and the last's)
                    x[0], v[0] = root0 + (root - root0) * (j + 1) / sub, (root - root0) * FPS
            root0 = root
            pts = _chain(x, 6)
            vel = _chain(v / FPS, 6)
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
            u = arc / max(arc[-1], 1e-6)
            d2 = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
            arc2 = np.r_[0.0, np.cumsum(np.maximum(d2, 0.4 * np.linalg.norm(np.diff(pts, axis=0), axis=1)))]  # (along
            # the picture: end on, it piles up a little deeper)
            speed = np.linalg.norm(vel[:, :2], axis=1) * FPS / max(s, 1.0)  # hand sizes a second
            rad = s * (WHIP_R[0] * (1 - u) ** 0.9 + WHIP_R[1]) * grow ** 0.5
            rad *= 1.0 - WHIP_TIP * _ss(0.8, 1.0, u)  # its tail thins to a point (it ended in a rounded worm's head)
            rad *= 1.0 + WHIP_FLARE[0] * (1.0 - _ss(0.0, WHIP_FLARE[1], u))  # it rises out of the pool
            rad *= 1 + sum(a * np.sin(2 * math.pi * (m_ * u - hz * t)) for a, m_, hz in WHIP_WAVES)
            if alive:  # fast, it sheds drops off its leading side, and its tail sprays
                fast = np.flatnonzero((speed > WHIP_SHED) & (u > WHIP_SHED_FROM))
                go = fast[rng.random(fast.size) < WHIP_SHED_RATE * (speed[fast] / WHIP_SHED - 1.0)]
                if go.size:
                    lead = _unit(vel[go] + 1e-9)
                    drops.add(k, pts[go] + lead * rad[go, None], 0.85 * vel[go] + rng.normal(size=(go.size, 3)) * 0.05
                              * s, s * rng.uniform(0.006, 0.02, go.size), foam=0.1)
                m = int(rng.integers(*WHIP_SPRAY)) if speed[-1] > 0.5 * WHIP_SHED else 0
                if m:
                    drops.add(k, np.repeat(pts[-1:], m, 0), 0.8 * vel[-1] + rng.normal(size=(m, 3)) * 0.05 * s,
                              s * rng.uniform(0.004, 0.01, m), foam=0.0, life=rng.uniform(3, 6, m), fade=2)
            lam = np.maximum(WHIP_BEAD[0] * rad, 1.0)  # the beads it necks into, this far apart, flowing out along it
            ph = np.r_[0.0, np.cumsum(np.diff(arc) / lam[1:])] - WHIP_FLOW * s * t / float(np.mean(lam))
            neck = np.maximum(WHIP_FRAY[1] * _ss(WHIP_FRAY[0], 1.0, u), broke / WHIP_BREAK if not alive else 0.0)
            beads = np.flatnonzero((np.diff(np.floor(ph)) > 0) & (neck[1:] > 0.05)) + 1
            rb = (0.75 * rad[beads] ** 2 * lam[beads]) ** (1 / 3)  # a bead holds its length of the stream
            if not alive and broke == WHIP_BREAK:  # necked through: beads, falling
                drops.add(k, pts[beads], vel[beads] + rng.normal(size=(len(beads), 3)) * 0.02 * s, rb)
                x = x[:0]
            else:
                tang = _unit(np.gradient(pts, axis=0))
                foam = (WHIP_FOAM[0] + WHIP_FOAM[1] * _ss(0.6, 1.0, u) + WHIP_FOAM[3] * (1.0 - _ss(0.0, 0.15, u))
                        + WHIP_FOAM[2] * np.clip((speed - 4.0) / 8.0, 0, 1))
                fl = vel[:, :2] + tang[:, :2] * WHIP_FLOW * s / FPS
                rad = rad * (1.0 - WHIP_BEAD[1] * neck)  # the stream thins where its water is drawn into beads (its
                sc.blobs += _stream(pts, vel, rad, foam, fl, arc2)  # radius rippled instead, the beads summed to a
                # hair)
                t2 = tang[:, :2] / np.maximum(np.linalg.norm(tang[:, :2], axis=1, keepdims=True), 1e-9)
                for i, r_ in zip(beads, rb * np.sqrt(neck[beads])):
                    shape, move = _moving(r_, vel[i, :2])
                    sc.blobs.append(Blob(pts[i, 0], pts[i, 1], pts[i, 2], r_, shape, fl[i, 0], fl[i, 1],
                                         foam=float(foam[i]), move=move,
                                         stream=(float(arc2[i]), float(t2[i, 0]), float(t2[i, 1]))))
                for ua, across, ar in air:  # air carried out along it
                    i = int(np.searchsorted(u, (ua + WHIP_FLOW * s * t / max(arc[-1], 1.0)) % 1.0 * 0.85))
                    i = min(i, len(pts) - 1)
                    nx_, ny_ = -tang[i, 1], tang[i, 0]
                    q_ = math.hypot(nx_, ny_) + 1e-9
                    sc.bubbles.append((pts[i, 0] + across * rad[i] * nx_ / q_, pts[i, 1] + across * rad[i] * ny_ / q_,
                                       ar * rad[i], WHIP_AIR_SEEN * float(_ss(0.0, 0.1, u[i]))))
        sc.blobs += drops.blobs(k)
        drops.step()
        if not len(x) and not len(drops.x) and pool_r <= 0.5:
            break
    return scenes


# ---- the shot -------------------------------------------------------------------------------------------------------

GRAIN_UNIT = 0.943  # what fx_light's grain reading reads off unit white noise: the plate's grain is laid at its strength


def frame(k: int, plate_k: np.ndarray, matte: np.ndarray | None, sc: Scene, hand: Hand, room: Room, noise: Noise,
          code: float, seed: int) -> np.ndarray:
    """Frame ``k`` with its water: uint8 RGB, the plate's own bytes wherever there is none."""
    out = plate_k.copy()
    if not sc.blobs and not sc.wet:
        return out
    H, W = plate_k.shape[:2]
    P = lin(plate_k)
    boxes = _wet(P, sc.wet, matte, room.lit(k)) if sc.wet else []
    fld = Field(sc.blobs, H, W) if sc.blobs else None
    layer = None if fld is None or fld.box is None else shade(fld, sc, k, hand, room, noise, _pyramid(P), matte,
                                                              code, seed)
    if layer is not None:
        boxes.append(layer.box)
        if matte is not None:
            boxes.append(caustic(P, layer, k, hand, room, matte, sc.held, sc.lobes, noise))
        y0, y1, x0, x1 = layer.box
        P[y0:y1, x0:x1] = P[y0:y1, x0:x1] * (1.0 - layer.alpha[..., None]) + layer.colour
        if sc.bubbles:
            bubbles(P, sc.bubbles, layer, room.lit(k), room.key(k, (sc.held or (W / 2, H / 2))[:2]))
            bx = np.array([[b[0], b[1], b[2]] for b in sc.bubbles])
            rr = int(2 * bx[:, 2].max() + 4)
            boxes.append((max(0, int(bx[:, 1].min()) - rr), min(H, int(bx[:, 1].max()) + rr + 1),
                          max(0, int(bx[:, 0].min()) - rr), min(W, int(bx[:, 0].max()) + rr + 1)))
    for y0, y1, x0, x1 in boxes:
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] = gam(P[y0:y1, x0:x1])
    return out


def _wet(P: np.ndarray, patches: list, matte: np.ndarray | None, lit: np.ndarray) -> list[tuple]:
    """Wet patches (Scene.wet), in place: each a soft, uneven ellipse where what it lies on (the hand, by its matte, or
    the wall off it) is darker, as wet skin and paint are, with a faint gloss towards its middle. Returns the boxes it
    changed."""
    H, W = P.shape[:2]
    boxes = []
    for x, y, rx, ry, ang, strength, on_hand, lob in patches:
        if on_hand and matte is None:
            continue
        e = 1.25 * max(rx, ry) + 2
        y0, y1, x0, x1 = max(0, int(y - e)), min(H, int(y + e) + 1), max(0, int(x - e)), min(W, int(x + e) + 1)
        if y1 <= y0 or x1 <= x0 or min(rx, ry) < 0.5:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        ca, sa = math.cos(ang), math.sin(ang)
        u = ((xx - x) * ca + (yy - y) * sa) / np.float32(rx)
        v = (-(xx - x) * sa + (yy - y) * ca) / np.float32(ry)
        th = np.arctan2(v, u)
        rr = np.sqrt(u * u + v * v) / (1.0 + 0.12 * np.cos(3 * th + lob) + 0.07 * np.cos(5 * th + 1.7 * lob))
        m = _ss(1.0, 0.65, rr).astype(np.float32)
        if matte is not None:
            mh = matte[y0:y1, x0:x1].astype(np.float32) / 255.0
            m = m * (mh if on_hand else 1.0 - mh)
        patch = P[y0:y1, x0:x1]
        patch *= (1.0 - strength * m)[..., None]
        patch += (WET[2] * m * np.clip(1.0 - rr, 0.0, 1.0) ** 2)[..., None] * lit
        boxes.append((y0, y1, x0, x1))
    return boxes


def render(plate: np.ndarray, track: dict, mask: np.ndarray | None, timeline: Timeline, form: str, seed: int = 0,
           progress=None) -> np.ndarray:
    """The shot's frames with the water in them, uint8 RGB [T, H, W, 3] (cuesync.elements' contract): ``plate`` uint8
    RGB [T, H, W, 3], ``track`` the hand's (its emitter carried to the tap), ``mask`` the hand's matte [T, H, W] uint8
    (or None: then nothing hides the water), ``form`` "orb" or "whip". The same inputs and seed give the same bytes.
    ``progress(stage, share, message)``, if given, hears how far it has got."""
    if form not in FORMS["water"]:
        raise ValueError(f"water has no form {form!r}; one of {FORMS['water']}")
    plate = np.asarray(plate)
    n, H, W = plate.shape[:3]
    if timeline.n != n or (mask is not None and len(mask) != n):
        raise ValueError(f"the plate has {n} frames, the timeline {timeline.n}, the mask "
                         f"{None if mask is None else len(mask)}")
    say = progress or (lambda stage, share, msg: None)
    say("water", 0.0, "Working out where the water goes")
    hand = hand_of(track, n)
    cam = None if timeline.camera is None else np.asarray(timeline.camera, np.float64)
    scenes = (orb_scenes if form == "orb" else whip_scenes)(timeline, track, hand, (H, W), seed, cam)
    room = Room(plate)
    noise = Noise(seed)
    wet = [k for k in range(n) if scenes[k].blobs]
    code = fx.shadow_noise(plate) / GRAIN_UNIT / 255.0 if wet else 0.0
    out = np.empty_like(plate)
    done = [0]

    def one(k: int) -> None:
        out[k] = frame(k, plate[k], None if mask is None else mask[k], scenes[k], hand, room, noise, code, seed)
        done[0] += 1
        if done[0] % 8 == 0:
            say("water", done[0] / n, f"Shaping the water: frame {done[0]} of {n}")

    with ThreadPoolExecutor(fx.THREADS) as ex:
        list(ex.map(one, range(n)))
    say("water", 1.0, "The water is in")
    return out
