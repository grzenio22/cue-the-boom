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
by the hand's motion and the voice), small ripples travel over it and the sheen round its glint breaks and stretches
with them (the window the glint mirrors keeps its shape, as a window's reflection does), the water circulates inside it
(seen as the room through it swirling, and bubbles carried round and rising), it drips from its underside, and it
froths where it is thin and fast.

Two forms (cuesync.elements.FORMS["water"]):

- ``orb``: water gathers out of the air into a ball floating over the open palm, sized to the hand, wobbling and
  circulating, carried on the hand with the lag a mass of water has. It is held while the voice holds, and each hit in
  it jolts it. When the voice ends with a hit, or the hand thrusts, it is thrown along the aim and into the room, and
  bursts on the room's back wall, where the water stays: it lands as a squashed body churned white, bursting
  into spray, and spreads over the wall in a sheet torn into fingers, flinging drops that land round it, and over the
  next frames drains into a wet patch whose lower edge sags into runs while drips slide down from it, leaving wet
  trails; a voice that just stops lets it fall onto the palm and splash off it.
- ``whip``: a stream of water rising out of a churning puddle on the palm, that swirls round the hand and trails it
  like a ribbon when it moves (each part of it held to the swirl by a spring, stiff at the root and soft at the tail),
  swelling and thinning as bulges run down it, rippled, thinning to its tail, where it frays into beads; where it lies
  on the palm it flattens into a film and wets the skin round it, and where it hangs off the hand it thins, sags and
  drips. It pours out with the voice, each hit lashes it out along the aim, blurred by its speed into a clear sheet
  seen by how it shifts and darkens what is behind it, a thin glint along its rim, white only in its torn bits, each
  streak of it its own length, its front tearing into threads and drops, and when the voice ends it breaks into beads
  that fall.

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
RIM_DARK = 0.3  # grazing rays see the room far round the side, through little water: the rim is this much darker...
RIM_BAND = (0.55, 0.88)  # ... over this range of 1 - n_z (on a ball, its last tenth of the radius: from 0.35, a ring
# 15 % of the radius wide, even all round, said glass), broken up by the waves
RIM_BREAK = 0.8  # how much the waves running round the rim break up the room mirrored in it
ROUND = (0.5, 0.85)  # a ray leaving the water this far off its way (the sine of the turn) sees the room round the
# water, by its direction, not the plate behind: near a ball's rim, bright towards the lights over it and dark
# towards the floor
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
FAR_FAST = 0.35  # water faster than WHIP_SHED shows the room through it from this share as far behind it (at 0.15 a
# lashing stream was a clear window onto the room, gone at a phone's size)
GLINT = (0.8, 0.07)  # the key light mirrored in the surface: its window (of the key light), a soft sheen round it...
GLINT_CORE = (1.3, 0.55, 0.08, 0.8, 0.025)  # ... clipped to white in its core: how bright (of white, whatever the
# key's strength: dimmer than the brightest things in the frame, a puff of cloud inside the ball), how big (of the
# window), the share of the finest waves it is mirrored in, how much dimmer it is where the surface curves both ways,
# and the waves it is not mirrored in (the surface smoothed over this many hand sizes; on the orb, none of its waves)...
GLINT_POW = 12.0  # ... how tight the sheen is...
WINDOW = (0.4, 0.26, 0.07, 0.25, 4.0, 0.7)  # ... and the window's half width and height (in the mirrored ray's
# direction, across and along the light's), how soft its edge is (of its size, either side: still water mirrors
# sharply, a soft edge was a cloud floating in the ball), the share of the finest waves (SPARKLE) it is mirrored in
# (mirrored in the surface's wobble and waves, it stretches and breaks up as the water moves: in a calm surface it was a
# sticker; in all of the finest waves, a crumpled white splat), how square its corners are (the power of its rounded
# rectangle), and on the orb the share of it mirrored in its broad shape alone (round its core, a lumpy sheen made the
# whole highlight a cloud)
WINDOW_BARS = (0.1, -0.15, 0.16, 0.8)  # the orb's window core is crossed by its frame: a bar down it and one across
# it this far off its middle (of its half width and height), this thick, and this dark (a plain bright core read as a
# cloud)
BACK = (0.12, 0.45)  # the window again, off the inside of the far wall: how bright, how big (at 0.3, a white blob
# low in the ball: the far side's light is the thin crescent, CRESCENT)
GLINT_BREAK = (1.6, 0.05, 0.85)  # along a stream its glints are broken by a noise this strong, its features this long
# (hand sizes), and while it lashes this much less
GLINT_LINES = (1.2, 0.5, 0.1, 0.76)  # a stream's edges catch the key light as a cylinder's do: the line along its
# edge towards the light, how bright (of the key light), and fast, along the other, how wide (of its radius, at least
# about a px) and how far out (of its radius); they carry a lashing stream's read through its blur...
GLINT_FAST = 0.75  # ... its broader glints dimmed this much while it lashes...
GLINT_SEG = (0.25, -0.6, 0.2)  # ... and the line towards the light broken into lengths: where a noise along the stream
# (features this many hand sizes long) is over the first of these, fully over the second (about two thirds lit)
EDGE_SHADE = (0.35, 1.6, 1.0)  # along its edge away from the light, a darker band: how much darker, how far in from
# its outline and how wide (px)
CRESCENT = (0.45, 0.45, 0.72)  # the key light gathered by the water onto its far inside edge, opposite the light: how
# bright, and the band of 1 - n_z it lies in (a thin crescent hugging the rim; over a quarter of the radius it was not
# seen at all)
INNER_GLOW = 0.06  # light bent and gathered inside the body: a faint moving net of brighter lines where it is thick
KEY_ELEV = 0.25  # the key light's direction out of the picture, towards the camera, per unit across it: a room's lights
# are over the water more than in front of it, so its glint sits towards its upper rim (at 0.9, in the middle of the
# ball: a cloud inside it, not a light on its surface)
KEY_UP = 0.35  # the key light is taken this much from above (a room's lights are mostly over the camera) and the rest
# from where the plate is brightest (read alone, a window behind the hand lit the orb from below)
KEY_SHARE = 0.015  # the brightest share of the room read as its key light...
KEY_SMOOTH = 6.0  # ... held steady over this many frames (sigma): read on each frame alone, a glint jumped when a lamp
# came into view on a walk
ENV_TOP, ENV_Q, ENV_BLUR = 0.4, 4, 1.5  # the room the water reflects: the plate's top share, this many times
# smaller and blurred by this (at 16 times smaller and blurred again, its rim was a flat grey band)
ENV_FLOOR = 0.5  # what it mirrors from below level is this much darker...
ENV_SAT = (0.6, 0.12)  # ... and its colour this much as strong, and at most this far off grey (of its brightness): a
# plant at the top of the plate, mirrored in full, was a green fleck stuck on the orb's rim
KEY_Q = 16  # the key light is found on the plate this many times smaller
F_SMALL = 0.25  # the most a small piece of water mirrors (under a tenth of a hand: it is all grazing surface, and at a
# glancing ray's full reflectance a splash's pieces were mercury)
DEPTH_SOFT = 0.08  # hand sizes: where the water passes behind the hand, it goes from seen to hidden over this depth
EDGE_SOFT = 0.5  # px: the camera's own softness on the water's outline (a cut-out edge was one pixel sharp)
FOAM_ALBEDO = 1.1  # froth is white matter lit by the room: this share of its key light comes back (at 0.75 it was
# no brighter than lit skin, and froth over the palm read as a grey haze)
FOAM_FAST = 0.4  # froth on water faster than WHIP_SHED is this much whiter (more air, finer bubbles)...
FOAM_EDGE = (0.2, 0.3, 0.75)  # ... its core keeping this share of its froth, clear: froth fully only from this far
# across it to this (of its radius), at its broken edges...
FOAM_TORN = (0.75, 1.25, 2.5)  # ... and there only in torn bits: where a noise (features this many times its bubbles)
# is over the first of these, fully over the second (about a sixth of it: torn into streaks all over, smeared, it was
# a frosted brush, a whisk broom)...
FOAM_LEAD = (0.5, 0.25)  # ... more along its leading edge, where it tears into the air, but hiding at most this much
# there (its froth all at the front, its streaks' white ends lined up like a comb's teeth)
FOAM_MAX = 0.8  # froth hides at most this much of what is seen through the water
FOAM_GLINT = (0.8, 0.5)  # the bubbles it is made of glint: how bright (of the key light), how small (of its bubbles)
FOAM_STREAK = (0.15, 0.3, 0.1)  # along a stream it lies in streaks running with it: this long (hand sizes), this wide
# (of its radius), and this much of it between them
RIPPLE = (0.14, 0.25, 0.3)  # a stream's surface is broken by ripples running out along it: their height (log-field
# units), their length (hand sizes) and width (of its radius): flow lines along it (0.06 long and the stream's whole
# width across, they were rings: a ribbed hose, a larva)
MENISCUS = (0.55, 1.2, 0.9, 0.12, 3.2, 0.25)  # water lying flat on something (the pool, a stream along the palm) shows
# its edge towards the key light as a thin bright line where its surface curls down to meet it, mirroring the room: how
# bright, how far in from its outline and how wide (px), then a darker line inside it, how dark and how far in, and on
# the other side a dark line where it meets the skin, how dark (bright on both sides, a silicone tube)
REST = (3.0, 10.0, 0.3)  # a stream whose middle is within the first to the second of its radii in front of the skin
# lies on it, flattened to this share of its thickness (as round as a tube there, it was a hose on the hand; the swirl
# holds it up to nine radii over the palm, and from 5 its round middle over the palm was a glass snake)
FLAT_CALM = 0.75  # water lying flat is moved this much less by the small waves (on the pool on the palm they were
# facets: crumpled cellophane, crushed ice; at half, a gummy sweet)
DOT = (0.05, 1.6, 0.7, 0.22)  # a free drop under this radius (hand sizes) glints as one bright point: how bright (of
# the key light), at least how wide (px, sigma) and how wide for its radius
DOT_MIN = 0.5  # px: the smallest drop that glints
DOT_FACING = (0.15, 0.6, 0.85)  # only the drops whose facet faces the key light glint in full, about a quarter of
# them, the rest at this share (all glinting alike, a splash's spray was a shower of white dashes: rain, sparks)
BLUR_STEP, BLUR_TAPS = 1.0, 256  # px between a moving pixel's taps along its path over the exposure, and at most this
# many taps, the same for every pixel moving alike (a lash smears up to about 200 px: at 64 taps spaced 3 px apart,
# shifted pixel by pixel by a patterned share of a step to hide it, the pattern showed as a screen-door hatch on slow
# water and as evenly spaced bars across a lash)
BLUR_TAPER = 0.6  # the share of the exposure the shutter spends opening and closing (_shutter)
BLUR_KEEP = (0.32, 0.25, 0.85)  # along their smear, as the shutter lets them through, the water's own light (glints,
# froth) keeps at least this much of itself, its froth's cover this much, and the sheet a lash sweeps (SHEET) this much
# (spread evenly, a lashing stream's froth and glints thinned out with its refraction and it was gone at a phone's size;
# kept most at its front, its white ends lined up into bristles)...
BLUR_RIM = (0.9, 2.5)  # ... and torn water's own light this much more within this many px of its smear's front end,
# where it is when the shutter closes: its leading glint drawn as one thin bright line along the sheet's rim...
DOT_KEEP = 0.4  # ... a small drop's glint this share of that (kept in full, its spray was a shower of white dashes)...
BLUR_SOFT = 1.5  # ... softened over this many px (sigma)
BLUR_RAGGED = (0.3, 3.0)  # torn water's smear is up to this much longer or shorter, streak by streak (features this
# many px across its motion), so the streaks' ends never line up (cut by one line, the front of a lash was a comb, a
# plastic cup's straight rim)
SHEET = (0.16, 5.0, 1.2)  # a lashing stream is seen by what it does to what is behind it: over the sheet it sweeps,
# the plate is this much darker and shifted up to this many px, in streaks running with its motion this many times as
# long as its smear (froth would carry it only by turning it milky)

# ---- motion and imperfection ----------------------------------------------------------------------------------------
WAVE = (0.028, 0.04)  # log-field units: the surface's small waves at rest, and more at the voice's loudest (about 3 %
# and 5 % of the radius at the outline); without them the outline was a compass circle and the orb read as glass, at
# twice this what was seen through it folded like crumpled cellophane
WAVE_SIZE = 0.2  # hand sizes: the size of those waves
WAVE_EDGE = 0.55  # their share at the outline: they move the inside's surface, and the outline wobbles in its low modes
# (moved as much as the inside, the orb's outline was a lumpy, rounded square; at 0.3, a still showed a perfect circle)
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
GATHER_LIGHT = 5.0  # its shade and the light it focuses come on as this power of how much of it has gathered
GATHER_FROM = (1.3, 2.2)  # radii: how far out they start (from 3, they flew in so fast they were glass needles)...
GATHER_R = (0.03, 0.07)  # ... and how big they are (at up to a fifth of its radius, a cluster of clear balloons
# fusing)
MODE_HZ = {2: 1.6, 3: 3.1, 4: 4.8}  # its wobbles: two, three and four lobes round it (a drop's ratios, 1 : 1.9 : 3)
MODE_AMP = {3: 0.03, 4: 0.012}  # of the radius, at rest the third of it and at the voice's loudest all of it (the
# four lobes at 0.022 squared its outline)
MODE_REST = 0.55
MODE2_DRIVE, MODE2_DAMP, MODE2_MAX = 0.006, 0.25, 0.35  # the two-lobe stretch, driven by its acceleration (inertia
# squashes it), its damping, and the most it stretches
CIRCULATE = (1.9, 1.5)  # rad a second: it turns over inside this fast at rest (0.3 turns a second: slower, a held orb
# was a still lens), and more at the voice's loudest
BUBBLES = 10  # air caught inside, carried round by the circulation (26 of them sat in it like a constellation of
# stickers)...
BUBBLE_R = (0.005, 0.035)  # ... this big (hand sizes, a power law: a few big, more small)...
BUBBLE_RISE = 0.25  # ... and rising through it, this many turns of its height a second...
BUBBLE_DRIFT = (0.5, 1.5, 0.06)  # ... each carried round at its own share of the circulation, and wandering this far
# (of the orb's radius) on its own (turned all together, they were a fixed constellation)...
BUBBLE_LOOK = (0.4, 0.9)  # ... each dark towards its edge, this much, with a glint this bright (of the key light)
DRIP_EVERY = (26, 60)  # frames between drips off its underside
DRIP_FORM, DRIP_R = 7, 0.035  # frames a drip takes to swell, neck and let go, and its radius (hand sizes: at a sixth
# of the orb's, a glass tulip bead hanging off it)
DRIP_SIDE = (0.3, 0.55)  # it hangs this far to one side of the orb's lowest point (of its radius, across): straight
# under it, it landed in the focus and its wet dot was lost in the light
DRIP_WET = (1.1, 48, 0.16, 3, (1.5, 3.0), 0.6)  # where it lands on the palm it leaves a wet dot this many of its radii
# across, drying over this many frames (half gone in half of them: a wet spot on skin stays for seconds; gone in 8, it
# was barely seen), at first this much darker with a point of glint, and this many tiny drops jump off it as it lands,
# this fast (hand sizes a second); round, feathered over this share of its radius (small, hard-edged and uneven, a mole)
SKIN_WET = (0.12, 0.3, 0.05, 1.0, 0.05)  # wet skin: how much more saturated, how much darker again round a patch's
# outline, its gloss towards its middle, the point of glint off a fresh drop's (of the key light), and how dark a patch
# is at which its colour and gloss are full (drying, they fade with it)
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
FAN = 0.6  # on the wall it spreads further the way it was thrown, as if it had come in this much along the wall (of its
# speed into it)
SPRAY_LIFE = 7  # frames the fine spray lasts
SPLAT = (1.3, 0.3, 0.1, 4, (7, 17), (0.02, 0.05))  # burst on the palm, its first frame is a splat flattened against it:
# its radius (of the ball's), its thickness (of a ball's), froth, and the uneven fingers round its edge: so many sets of
# lobes, how many lobes each, how far out (of its radius) (in six and nine even lobes, a frosted flower decal; in three
# and five frothing more, a crumpled plastic bag)
BURST_SWELL = (1.12, 0.3, ((3, 0.05), (5, 0.025)))  # the burst's first frame: its body swollen to this share of its
# radius, frothing this much, wobbling in these lobes (of its radius) (spread into a ring of pieces with its torn skin
# inside, it was a flat lid stuck in the air; wrinkled in five to nine lobes, a cellophane flower; frothing more, a
# frosted disc)
LIGAMENT_R, LIGAMENT_STRETCH = (0.025, 0.035), (5.0, 9.0)  # its torn skin's ligaments, as round pieces (radius, of the
# ball's) drawn out this many times longer, keeping their volume: threads a hundredth of the ball thick, tapering to
# both ends (three times as thick, they were shards of glass)...
LIGAMENT_LIFE, LIGAMENT_BEADS = 2, (3, 1.4)  # ... lasting this many frames before each has beaded up into this many
# drops this many times its radius, spread along it apart (packed close, a stack of discs: a caterpillar)

# ---- water on the wall ----------------------------------------------------------------------------------------------
WALL_SPREAD = (1.45, 1.8, 1.4)  # burst on the wall, it lands as a body squashed to this radius (of the ball's) and
# spreads as a sheet to this, most of the way in this many frames (e-folding) (landing at the ball's own radius, the
# hit was the smallest change of the splash: a thin ring on the strongest beat)
WALL_FINGERS = (6, 10, 1.2)  # its edge torn into this many fingers, grown out over this many frames (e-folding)...
FINGER_LEN = (0.3, 0.6)  # ... this long (of the sheet's radius: drawn from a power law, most were too short to see and
# the splash was a disc, a puddle)...
FINGER_DIR = (0.55, 0.3, 1.1, 0.5)  # ... this share of them running downwards and this share out sideways the way it
# was thrown, the rest anywhere, spread about those ways by this much and that (rad)...
FINGER_W = (0.04, 0.07)  # ... each a capsule this wide (half its width, of the sheet's radius), as wide all along...
FINGER_BEAD = (1.3, 1.6)  # ... ending in a bead this many times its half width (as triangular spikes it was a cartoon
# splat; two up, a cat's ears)...
FINGER_UP = 0.15  # ... reaching at most this far upwards (of the sheet's radius)...
FINGER_BLEND = 0.8  # ... and meeting the sheet in a fillet this many of its half widths wide
WALL_WOBBLE = ((2, 0.06), (3, 0.05), (5, 0.03))  # the outline's broad unevenness: lobes, how far (of the radius)
WALL_EDGE = (1.3, 0.05, 1.2)  # its edge broken up: by this many px, in features this big (of its radius), and soft over
# this many px either side (an even, one-pixel outline was a line drawing)
WALL_THICK = (0.3, 1.5, 0.02)  # the sheet's thickness at the hit (of the ball's radius; its middle 1.6 times this),
# how fast it thins (frames, e-folding), and the film left at the end
WALL_FROTH = (0.8, 2.0, 0.03, 0.3, 9)  # at the hit its middle is churned white: how much, gone this many frames after
# (at under a third of it on the next: lingering, a milky fog), its bubbles (of its radius), and the fine streaks it
# tears out in along its radius, this long (of its radius) and this many round it (in soft clumps, camouflage, a
# snowball)
WALL_DOME = (0.18, 0.12)  # while it bulges, the light it gathers onto the wall at its foot and the shade at its top
WALL_LOOK = (3, 4)  # frames the sheet shows as running water, then frames it drains into the wet patch (all of a
# splash gone between two frames, on the strongest beat, was the element's hardest cut)
WALL_RIM = (5.0, 1.0, 0.45, 0.3, (0.35, 0.9), 12.0)  # the sheet's rounded rim: its width px, its height (of that), the
# bright line on its side towards the key light (of it) and the dark line where it meets the wall on the other side
# (how much darker), both broken up (bright all round, it was a vector outline); the bright line only along its body's
# edge where that faces the light this much, and in lengths: a noise this many px across (along its fingers, round
# their beads, its wobbles, it was a doodle of little closed loops)
WALL_BEND = 1.6  # px the wall seen through the sheet shifts per px of its thickness's slope
WALL_RAYS = (0.25, 0.45, 0.35, 3)  # the streaks it runs out in, thicker and thinner (spread, of its thickness), the
# ridge along each finger, how long they are (of its radius), and how many tiles of noise round it (narrow across:
# about a twentieth of a turn) (flat and even it was a decal; in regular rays, a cut lemon)
WALL_GLINT = 0.6  # the key light mirrored in its slopes (of the key light), its core clipped to white as the orb's is
WALL_WET = (0.2, 0.12, 0.05, 6, 0.12)  # the wet patch it leaves: how much darker the wall, how much more saturated, its
# gloss towards the key light, the frames it takes to come on fully (at a tenth darker with a hard top edge, a soft
# shadow), and how much darker its outer part is, softly (a dark line round it was a stain's outline)
WALL_ROUND = (0.05, 2.5)  # the patch's outline is the sheet's rounded over this share of its radius and feathered over
# this many px (it kept the sheet's spikes as a hard-edged stain)
WALL_SAG = (3, (3, 6), (0.1, 0.2), (0.05, 0.1), 0.6, 6)  # from this frame after the hit the patch's lower edge sags
# into this many runs, this long and wide (of its radius), the film collecting there this much darker and glossier, all
# of it over this many frames (in ten frames it never sagged)
WALL_SOAK = (0.12, 10)  # it soaks out this much wider over this many frames
WALL_BEADS = (16, (0.025, 0.09), (0.3, 1.5), (1.2, 2.5), (6, 10))  # drops flung out of the sheet onto the wall: how
# many, their radii (of the ball's), how far beyond the sheet's edge they come to rest (of the ball's radius), in how
# many frames (e-folding, held back by the wall), and after how many frames the small ones are gone into the wall
WALL_TIPS = ((0.05, 0.09), (0.03, 0.15), 4)  # a drop left at each finger's tip: its radius (of the ball's), how far it
# runs on past the tip (of the ball's radius), and the frame it gathers on, as the sheet drains (drawn over the finger's
# own bead while the water still ran, a loop on a hook)
WALL_SLIDE = (0.08, 0.8, 0.25)  # a drop on the wall bigger than this (of the ball's radius) slides down it, at most
# this fast (hand sizes a second, for the biggest), taking this long (seconds) to get going
WALL_DRIPS = (3, (0.13, 0.2), 2.6, 0.15, 2)  # drips running down from the runs at its lower edge: how many, their
# radius (of the ball's), how fast at most (hand sizes a second, at the wall: at 1.5 they ran 40 px in the ten frames
# left), how soon they get to it (seconds), and the frame they set off on
WALL_TRAIL = (0.5, 0.5, 1.0, 0.12)  # the wet trail a running drop leaves: how dark against the patch (a thinner film:
# as dark, a needle-thin dark stem), its half width at its top (px) and at the drop (of the drop's radius: as wide as
# the drop at its foot), and the gloss along its side towards the key light
WALL_FALL = (4, (0.035, 0.07), (2, 8))  # drops that fall off the splash's lower edge into the air: how many, their
# radii (of the ball's), the frames after the hit they let go on (spread out: falling all at once, a sudden shower)
WALL_SPRAY = (70, (10.0, 20.0), (0.012, 0.04))  # spray off the hit, flying out along the wall and off it: how many
# specks at full strength, how fast (hand sizes a second) and how big (of the ball's radius, a power law)
WALL_CROWN = (8, (0.07, 0.13), (3.0, 6.0), (2.0, 4.0), (5, 8))  # bigger drops thrown off its rim at the hit towards
# the camera: how many, how big (of the ball's radius), how fast out along the wall and towards the camera (hand sizes
# a second), and the frames before they are past
BEAD_LOOK = (1.12, 0.14, 0.1, 0.15, 1.2, 0.7, 0.3, 0.45)  # a drop on the wall: the wall through it magnified this
# much and this much darker (it bends the darker room in), the light it gathers onto the wall inside its foot and the
# shade inside its top (lit from above, a clear drop is dark at its top and bright at its foot; gathering a quarter
# more, it glowed like a pearl), its glint's brightness (of white) and size (px: a point towards the light), and the
# thin bright crescent inside its rim on its side towards the light, only where it faces it this much (with a dark
# line round the rest, outlined all round, it was an icon)
BEAD_SHAPE = (1.2, 0.07, 0.8, 0.7)  # it lies flattened on the wall, sagging this many times longer than wide, its
# outline uneven by this much (of its radius); sliding, it draws out behind into its trail over this many radii,
# narrowing to this share of its width (round and even, a ball bearing)
BEAD_SHADOW = (0.35, 0.14)  # its soft shadow on the wall: this far away from the light (of its radius) and how dark

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
WHIP_R = (0.1, 0.033)  # hand sizes: the stream's radius at its root and at its tail (blobs weighted, as thick as
# designed: summed unweighted, they drew a hose 1.7 times as thick)
WHIP_TIP = 0.35  # its last fifth thins by this much to its end
WHIP_SPACING = 0.6  # its blobs are at most this share of their radius apart
WHIP_POOL = (1.35, 0.14, 0.12, 0.07)  # the pool on the palm it rises out of: its radius (of the stream's at its root),
# its thickness (of a ball's), its froth, and how far its outline ripples (of its radius); a ball of water there was a
# test tube's closed end; frothing all over and rippling at 0.16 it was crumpled cellophane with a frilled edge (the
# stream churns it where it rises: WHIP_FOAM); at 1.8 a bulb on the stream, a tadpole's head
WHIP_POOL_ALONG = (1.15, 0.2)  # it is drawn out this many times longer along the way the stream leaves it (keeping its
# area) and lies this far that way (of its radius): the stream widens into it (round, centred on the root, a spoon;
# drawn out 1.5 times, with the stream narrowing out of it, a pointed snake's head)
WHIP_POOL_PARTS = (3, (0.45, 0.8), (0.45, 0.7), 0.12)  # an irregular puddle: this many flat pools besides it, each
# this far from its middle (of its radius) and this big (of it), drifting round it at up to this many turns a second
WHIP_FLARE = (0.35, 0.1)  # the stream flares out this much wider where it leaves the pool, narrowing over this share
# of its length
WHIP_WET = 1.3  # the palm round the pool is wet this far out (of the pool's radius)
WHIP_TRAIL = (0.15, 4, 2, 0.03, 1.5, 0.05, 0.7)  # where the stream lies on the palm it wets the skin round it: how
# much darker at first, drying over this many frames (left for a second where the stream had swirled past, a string of
# dark spots), marked every this many frames at points this far apart along it (hand sizes: further apart, beads),
# wet this many of its radii out and this many hand sizes besides, feathered over this share of that (a clear stream
# with nothing round it on the skin was a glass snake)
WHIP_LIFT = 0.06  # hand sizes it starts over the palm
WHIP_CRACK = (8, 3.6, 0.45, 2.5)  # a hit cracks it: frames, how far out along the aim (hand sizes), the kink that runs
# down it (hand sizes across), and how much stiffer it is while it lashes out
WHIP_SHED, WHIP_SHED_RATE = 9.0, 0.12  # hand sizes a second: where it is faster than this it sheds drops, each point
# of it past WHIP_SHED_FROM of its length at this chance a frame per time over it...
WHIP_SHED_FROM = 0.3  # (shed near the root, they wound round it like a screw's thread)
WHIP_SPRAY = (1, 4, 18)  # fine drops its tail sprays a frame while it moves, and at most this many threads and drops
# off its front while it lashes out (a smooth front was a hard sheet)
WHIP_FOAM = (0.14, 0.3, 0.9, 0.45)  # how aerated it is all along, more at its tail, more where it is fast (a fast jet
# goes white: at 0.5 the lash was a clear smear, gone at a phone's size), and more at its root where it churns out of
# the pool (clear all along, a glass rod)
WHIP_AIR, WHIP_AIR_R, WHIP_AIR_SEEN = 10, (0.12, 0.28), 0.45  # bubbles carried along it, their size (of its radius),
# and how clearly they are seen (at 0.7, rings like glass beads)
WHIP_SAG = (0.12, 0.55)  # of gravity, while it is held up: over the hand, and hanging off it (at 0.3, a loop held up
# off the side of the hand was a carry handle, a hose)...
WHIP_HANG = 0.6  # ... where it also thins to this share of its width...
OFF_HAND = ((0.04, 0.2), (0.6, 0.9))  # ... which it is from this far outside the outline round the hand's landmarks to
# this (hand sizes), or with none, from its palm's point
WHIP_DRIP = ((8, 13), 1.6)  # hanging off the hand, it drops a drip from its lowest point every so many frames, swelling
# there to this many times its width as it necks above it (held up stiffly, it was a hose)
WHIP_WIDTH = (0.3, 0.7, 0.25, 0.42)  # its width varies along it besides its bulges, this much (spread, of its radius),
# in features this long (hand sizes), drifting out at this share of its flow, and at most this much either way (even
# all along, a silicone tube; at a spread of 0.18 it was not seen at a phone's size)
WHIP_THREADS = 0.5  # the share of what its front tears off that are threads, drawn out 2.5 to 5 times their width...
WHIP_TORN = ((0.004, 0.02), (0.8, 1.3))  # ... their radii (hand sizes, a power law), and their speed against the
# stream's there: each flung on along its own path, smeared to its own length (all at the stream's speed, their
# streaks ended in a line)
WHIP_CULL = (1.3, 2.0)  # hand sizes: what it sheds while it pours is gone between these distances from it
WHIP_SWELL = (0.26, 0.13, 0.38)  # bulges carried down the stream with its flow: how much thicker and thinner (the
# spread, of its radius), their size (hand sizes: about eight of its radii apart, at no set spacing), and the most
# they swell or thin it (in three sets of regular waves, the shortest 13 to its length, it was a ribbed larva)
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
    glint: float = 1.0  # a small free drop: how much of the key light's glint it catches (DOT_FACING)


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
    wet: list = field(default_factory=list)  # (x, y, rx, ry, angle, strength, phase of its uneven edge, glint, soft):
    # wet patches it leaves on the palm, darker and faintly glossy (under the stream's pool, where a drip lands)
    band: list = field(default_factory=list)  # ((m, 2) px, (m,) radii px, strength): the skin wet round where the
    # stream lies, a band through those points
    film: list = field(default_factory=list)  # the water on the room's wall where the orb burst on it: a Sheet, Trails
    # and Beads (_wall), drawn on the plate behind the water in the air (_film)


class Sheet(NamedTuple):
    """The water spread over the wall where the orb burst on it, on one frame (Scene.film)."""

    x: float  # its middle, px
    y: float
    r: float  # its radius px, its fingers besides
    fingers: tuple  # ((angle rad, length past r (of r), half width (of r), tip bead (of the half width)), ...)
    wobble: tuple  # ((lobes, share of r, phase), ...): its outline's broad unevenness
    thick: float  # px
    dome: float  # 0..1: how much it still bulges in the middle (at the hit) rather than lying flat
    look: float  # 0..1: how much it still shows as running water (0: only the wet patch it leaves)
    wet: float  # 0..1: how far the wet patch has come on
    froth: float = 0.0  # 0..1: how churned white its middle is (at the hit)
    sag: float = 0.0  # 0..1: how far the wet patch's lower edge has run down into its runs
    runs: tuple = ()  # ((angle rad, width (of r), length (of r)), ...): where its lower edge runs down


class Bead(NamedTuple):
    """A drop lying on the wall (Scene.film)."""

    x: float
    y: float
    r: float  # px
    vx: float  # px a frame: it is drawn out along its motion
    vy: float
    a: float  # 0..1: how much of it is left (the smallest go into the wall)
    ph: float = 0.0  # its uneven outline's phase


class Trail(NamedTuple):
    """The wet trail a drop running down the wall leaves (Scene.film), from (x0, y0) to the drop of radius r at
    (x1, y1)."""

    x0: float
    y0: float
    x1: float
    y1: float
    r: float


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
        self.dots = [(b.x, b.y, b.z, b.r, b.glint) for b in blobs
                     if b.free and DOT_MIN <= b.r and 0 <= b.x < w and 0 <= b.y < h]
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
        out = _remap(e, u, v, cv2.BORDER_REPLICATE) * (1.0 - ENV_FLOOR * down)[:, None]
        lum = (out @ fx._LUMA)[:, None]
        tint = out - lum
        most = np.abs(tint).max(1, keepdims=True)
        return lum + tint * np.minimum(ENV_SAT[0], ENV_SAT[1] * lum / np.maximum(most, 1e-6))

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

    def __init__(self, seed: int, count: int = 14):
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
    sheet: np.ndarray | None = None  # (h, w, 3): the sheet a lashing stream sweeps over the exposure: its share, and
    # how far px (x, y) it shifts the plate under it (SHEET; None: no lash)


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
    calm0 = 1.0 - FLAT_CALM * np.clip(1.0 - fld.th[ys, xs], 0.0, 1.0)
    bump = sc.wave * at_edge * calm0 * noise.flow(0, X, Y, vx[ys, xs], vy[ys, xs], k, WAVE_SIZE * s)
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
    broad = lf.copy()  # its broad shape, its waves and ripples aside (the held orb's window glint is mirrored in it)
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
    Rmap = sig[crop] / np.float32(SIG_K)
    th = fld.th[crop]  # a flat piece is thinner, and its edge tapers in gently, as a film's does (a ball's profile
    # scaled down still rose sheer at its outline, and a pool on the palm was a glass lump with a dark rim)
    if matte is not None:  # a stream lying on the skin is flattened against it, a rivulet
        th = th * (1.0 - (1.0 - REST[2]) * _resting(hand, k, matte, (by0, bx0), fld.zc[crop], Rmap, s)
                   * _ss(0.6, 0.9, th)).astype(np.float32)
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
    flat = np.clip(1.0 - th[iy, ix], 0.0, 1.0)  # water lying flat on something
    thick = np.clip(h / np.maximum(0.2 * R, 1.0), 0.0, 1.0) * (1.0 - FLAT_CALM * flat)
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
    # the rim's darkening and the dimming where rays are turned back belong to a body of water: a small drop is all
    # rim, and with them it was a grey metal sliver; it keeps a thin edge and a darker core round its glint
    small = np.clip(R / (SMALL_RIM * s), 0.0, 1.0)
    edge = (SMALL_EDGE + (1.0 - SMALL_EDGE) * big) * small
    oxy = np.maximum(np.hypot(ox, oy), 1e-3)  # a ray turned far off its way sees the room round the water (near a
    rnd = (edge * _ss(ROUND[0], ROUND[1], oxy)).astype(np.float32)  # ball's rim, turned down: the floor under its top)
    col += rnd[:, None] * (room.reflect(k, ox / oxy, oy / oxy) - col)
    col *= np.exp(-ABSORB[None, :] * (path / max(s, 1.0))[:, None])
    rr = np.clip(1.0 - nz, 0.0, 1.0)
    band = _ss(RIM_BAND[0], RIM_BAND[1], rr).astype(np.float32)
    col *= (1.0 - edge * RIM_DARK * band * np.clip(0.75 + 0.5 * inner, 0.25, 1.25))[:, None]
    col *= (1.0 - edge * (1.0 - TIR_DIM) * (1.0 - out) * band)[:, None]
    col *= (1.0 - tiny * (TINY_EDGE * _ss(0.55, 0.95, rr) + TINY_CORE * (1.0 - _ss(0.0, 0.7, rr))))[:, None]
    # the room reflected (a small piece of water is a weak mirror: at full strength a splash's thin pieces were
    # chrome), read by the way the surface faces: up at the top of the rim, down at its foot (read by the mirrored
    # ray, which goes straight on past a grazing rim, the whole rim mirrored one patch of the room: an even ring)
    F = (F0 + (1.0 - F0) * (1.0 - nz) ** 5).astype(np.float32)
    F = np.minimum(F, F_SMALL + (1.0 - F_SMALL) * big) * (1.0 - TINY_MIRROR * tiny)
    F *= 1.0 + RIM_BREAK * _ss(0.3, 0.6, rr) * np.clip(inner, -1.0, 1.0)  # the rim's mirror broken into arcs by the
    # waves running round it (its finest waves die out at the outline, and it was an even ring)
    flow_a = WHIP_FLOW * s / FPS
    whip = (_ss(0.4, 1.0, fast * FPS / WHIP_SHED) * np.clip(sw / 0.5, 0.0, 1.0)).astype(np.float32)  # a lashing stream
    if st.size:  # along a stream its glints and the room it mirrors break up with the ripples running down it (one
        # unbroken streak down its length was an acrylic rod's, a neon tube's), but hardly while it lashes (each bright
        # bit, smeared across the stream by its speed, was a bar: a ladder's rungs, a zebra ribbon)
        gm = np.clip(1.0 + GLINT_BREAK[0] * noise.along(7, sa, sxs, k, GLINT_BREAK[1] * s, 0.8, flow_a), 0.0, 2.0)
        gm = 1.0 + (gm - 1.0) * (1.0 - GLINT_BREAK[2] * whip[st])
        F[st] *= 1.0 + 0.5 * sw[st] * (gm - 1.0)
        seg = _ss(GLINT_SEG[1], GLINT_SEG[2], noise.along(13, sa, sxs, k, GLINT_SEG[0] * s, 3.0, flow_a))
    ev = room.reflect(k, nx, ny)
    col = col * (1.0 - F)[:, None] + ev * F[:, None]
    Lk = room.key(k, ((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    lit = room.lit(k)
    lxy = math.hypot(float(Lk[0]), float(Lk[1])) + 1e-6
    face = (nx * Lk[0] + ny * Lk[1]) / lxy / np.maximum(np.hypot(nx, ny), 1e-3)  # its slope towards the light, -1..1
    sdp = sd[crop][iy, ix]  # water lying flat: a bright line round its edge towards the light, where its surface curls
    # down to meet what it lies on, and a dark line where it meets it on the other side (bright all round, a tube)
    men = MENISCUS[0] * flat * np.exp(-((sdp - MENISCUS[1]) / MENISCUS[2]) ** 2) * _ss(0.1, 0.7, face)
    if st.size:  # (along a stream, in lengths: lit all along, a tube)
        men[st] *= 1.0 - sw[st] * (1.0 - seg)
    col *= (1.0 - MENISCUS[3] * flat * np.exp(-((sdp - MENISCUS[4]) / 1.5) ** 2)
            - MENISCUS[5] * flat * np.exp(-((sdp - 0.8) / 0.9) ** 2) * _ss(-0.1, 0.7, -face))[:, None]
    col += men[:, None] * (0.6 * ev + 0.4 * lit[None, :])
    if st.size:  # a stream's edge away from the light: a thin darker band, the room round it bent in there (bright
        # lines down both its edges, it was a glass tube)
        col[st] *= (1.0 - EDGE_SHADE[0] * sw[st] * np.exp(-((sdp[st] - EDGE_SHADE[1]) / EDGE_SHADE[2]) ** 2)
                    * _ss(-0.2, 0.6, -face[st]))[:, None]
    # the key light: its window mirrored in the surface, waves and all, a soft sheen round it, the window again off the
    # inside of the far wall (opposite, smaller, dimmer), the light the water gathers onto its far inside edge, and a
    # faint moving net inside it
    gxw, gyw = gx[iy, ix] + WINDOW[3] * sx, gy[iy, ix] + WINDOW[3] * sy  # the window is mirrored in the surface's
    # wobble and waves, and only a little of its finest waves
    gys, gxs = np.gradient(cv2.GaussianBlur(h2, (0, 0), max(1.0, GLINT_CORE[4] * s)))  # its clipped core keeps the
    gxc = np.clip(gxs, -6, 6)[iy, ix] + GLINT_CORE[2] * (1.0 - big) * sx  # window's shape, bent by the surface's broad
    gyc = np.clip(gys, -6, 6)[iy, ix] + GLINT_CORE[2] * (1.0 - big) * sy  # shape (by its waves too, it was a cloud or a
    if float(big.max()) > 0.01:  # heart); on the orb by its broad shape alone, its waves aside, the same from frame to
        # frame (moved by its waves, on some frames it was a lumpy cloud), the sheen round it mostly so too
        hbr = (sig[crop] * np.float32(math.sqrt(2.0 * LN_TAU)) * th
               * np.clip(broad[crop] / np.float32(LN_TAU), 0.0, None) ** (1.0 - 0.5 * th)).astype(np.float32)
        gyb, gxb = np.gradient(cv2.GaussianBlur(hbr, (0, 0), max(1.0, GLINT_CORE[4] * s)))
        gxb, gyb = np.clip(gxb, -6, 6)[iy, ix], np.clip(gyb, -6, 6)[iy, ix]
        gxc += big * (gxb - gxc)
        gyc += big * (gyb - gyc)
        gxw += WINDOW[5] * big * (gxb - gxw)
        gyw += WINDOW[5] * big * (gyb - gyw)
    nw = np.sqrt(gxw * gxw + gyw * gyw + 1.0)
    spec = GLINT[0] * _window(-gxw / nw, -gyw / nw, 1.0 / nw, Lk, 1.0)
    nc = np.sqrt(gxc * gxc + gyc * gyc + 1.0)
    saddle = _ss(-0.02, -0.3, (hxx * hyy - hxy * hxy)[iy, ix] * R * R)  # ... and where the surface curves both ways it
    spec += (GLINT_CORE[0] / max(float(room.white[k]), 0.3) * (1.0 - GLINT_CORE[3] * saddle)  # mirrors it twice over:
             * _window(-gxc / nc, -gyc / nc, 1.0 / nc, Lk, GLINT_CORE[1], big))  # softened there (a bow tie); on the
    # orb crossed by its frame's bars
    spec += (BACK[0] * big * deep) * _window(gxw / nw, gyw / nw, 1.0 / nw, Lk, BACK[1])
    hv = Lk + np.array([0.0, 0.0, 1.0], np.float32)
    hv /= np.linalg.norm(hv)
    ndh = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0.0, 1.0)
    spec += GLINT[1] * ndh ** GLINT_POW
    if st.size:
        spec[st] *= 1.0 + sw[st] * (gm - 1.0)
    # a fast stream only (the thrown orb at its speed went milky): froth in torn bits, its edges catching the key light
    # as a cylinder's do, and the sheet it sweeps
    spec *= 1.0 - GLINT_FAST * whip  # (its broad glints, smeared by its speed, were a slab of white streaks)
    if st.size:  # a thin bright line along its edge towards the light, broken into lengths, and fast, a dimmer one
        # along the other (where the light grazed both edges, lit all along, they were a glass tube's)
        wl = np.maximum(GLINT_LINES[2], 0.9 / np.maximum(R[st], 1.0))
        lines =((GLINT_LINES[0] * _ss(0.15, 0.7, face[st]) * seg
                  + GLINT_LINES[1] * whip[st] * _ss(-0.1, 0.6, -face[st])) * np.exp(-((np.abs(sxs) - GLINT_LINES[3])
                                                                                    / wl) ** 2))
        spec[st] += sw[st] * lines * np.clip(gm, 0.3, 2.0)
    cres = (CRESCENT[0] * _ss(CRESCENT[1], CRESCENT[1] + 0.12, rr) * (1.0 - _ss(CRESCENT[2] - 0.1, CRESCENT[2], rr))
            * _ss(0.4, 0.95, -face))
    net = INNER_GLOW * np.exp(-(inner / 0.22) ** 2) * deep * (1.0 - F)
    col += (spec + cres + net).astype(np.float32)[:, None] * lit[None, :]
    # froth: white matter lit by the room, in bubbles, and along a stream in streaks running out with it
    fn = 0.55 + 0.45 * noise.flow(5, X, Y, fvx, fvy, k, FOAM_SIZE * s)
    if st.size:
        streak = FOAM_STREAK[2] + 1.6 * np.maximum(noise.along(8, sa, sxs, k, FOAM_STREAK[0] * s, FOAM_STREAK[1],
                                                               flow_a), 0.0)
        fn[st] += sw[st] * (streak * fn[st] - fn[st])
    mxp, myp = mxc[iy, ix], myc[iy, ix]
    lead = np.clip(-(gx[iy, ix] * mxp + gy[iy, ix] * myp) / (np.hypot(gx[iy, ix], gy[iy, ix]) * np.hypot(mxp, myp)
                                                             + 1e-6), 0.0, 1.0)
    front = whip * lead * np.exp(-np.maximum(sdp, 0.0) / np.maximum(0.5 * R, 1.0))
    foam = foam + FOAM_LEAD[0] * front
    if st.size:  # fast, its core runs clear, aerated at its broken edges and its front (evenly white, frosted plastic)
        foam[st] *= 1.0 - whip[st] * (1.0 - FOAM_EDGE[0]) * (1.0 - _ss(FOAM_EDGE[1], FOAM_EDGE[2], np.abs(sxs)))
    torn = _ss(FOAM_TORN[0], FOAM_TORN[1], noise.flow(10, X, Y, fvx, fvy, k, FOAM_TORN[2] * FOAM_SIZE * s))
    fn *= 1.0 - whip * (1.0 - torn)  # only its torn bits white, clear water between (smeared, short white threads)
    fm = np.clip(foam * fn, 0.0, 1.0) * FOAM_MAX
    fm = np.minimum(fm, FOAM_MAX - (FOAM_MAX - FOAM_LEAD[1]) * _ss(0.0, 0.5, front))
    glint = FOAM_GLINT[0] * np.exp(-(noise.flow(9, X, Y, fvx, fvy, k, FOAM_GLINT[1] * FOAM_SIZE * s) / 0.3) ** 2)
    froth = (0.75 + 0.25 * nz) * FOAM_ALBEDO * (1.0 + FOAM_FAST * whip) + glint
    col = col * (1.0 - fm)[:, None] + (fm * froth)[:, None] * lit[None, :]
    alpha = np.zeros(a_.shape, np.float32)
    alpha[iy, ix] = cover[crop][iy, ix] * vis
    C = np.zeros(alpha.shape + (3,), np.float32)
    C[iy, ix] = col * alpha[iy, ix, None]
    # what keeps some of itself through the blur: its own light (key light units), its froth's cover, and a lash's
    # sheet: its share, and the shift (px) it gives the plate under it, in streaks along its motion
    own = np.zeros(alpha.shape + (5,), np.float32)
    own[iy, ix, 0] = ((spec + cres) * (1.0 - fm) + fm * froth) * alpha[iy, ix]
    own[iy, ix, 1] = fm * alpha[iy, ix]
    rag = np.zeros(alpha.shape, np.float32)
    fastp = np.flatnonzero(whip > 0.01)
    if fastp.size:
        rag[iy[fastp], ix[fastp]] = whip[fastp]
        spd = np.maximum(np.hypot(mxp[fastp], myp[fastp]), 1e-3)
        ux, uy = mxp[fastp] / spd, myp[fastp] / spd
        a_l, a_c = X[fastp] * ux + Y[fastp] * uy, Y[fastp] * ux - X[fastp] * uy  # px along its motion and across
        length = SHEET[2] * max(float(np.median(spd)), 0.1 * s)
        d_l = noise.along(11, a_l, a_c, k, length, 0.025 * s, 0.0)
        d_c = noise.along(12, a_l, a_c, k, length, 0.04 * s, 0.0)
        pres = whip[fastp] * alpha[iy[fastp], ix[fastp]]
        own[iy[fastp], ix[fastp], 2] = pres
        own[iy[fastp], ix[fastp], 3] = pres * SHEET[1] * np.clip(0.6 * (d_l * ux - d_c * uy), -1.0, 1.0)
        own[iy[fastp], ix[fastp], 4] = pres * SHEET[1] * np.clip(0.6 * (d_l * uy + d_c * ux), -1.0, 1.0)
    _dots(C, fld.dots, (by0, bx0), s, Lk, lit, hand, k, matte, own)
    box, C, alpha, sheet = _motion_blur(C, alpha, mxc, myc, (by0, by0 + C.shape[0], bx0, bx0 + C.shape[1]), H, W,
                                        own, lit, rag, k)
    C = cv2.GaussianBlur(C, (0, 0), EDGE_SOFT)
    alpha = cv2.GaussianBlur(alpha, (0, 0), EDGE_SOFT)
    if code > 0:  # the plate's grain, on what the water shows (what is seen through it was read smoothed, its glints
        # are clean); the sensor's, so laid on after the motion blur
        y = np.maximum(C @ fx._LUMA, 1e-6) / np.maximum(alpha, 1e-3)
        step = (2.2 * code) * y ** np.float32(1.2 / 2.2) / y
        g = _rng(seed, _GRAIN, k).standard_normal(y.shape).astype(np.float32)
        C = np.maximum(C * (1.0 + g * step)[..., None], 0.0)
    pad = ((by0 - box[0], box[1] - by0 - h2.shape[0]), (bx0 - box[2], box[3] - bx0 - h2.shape[1]))
    if sheet is not None:
        sheet = cv2.GaussianBlur(sheet, (0, 0), EDGE_SOFT)
    return Layer(box, C, alpha, np.pad(h2, pad), np.pad(fld.zc[crop], pad), np.pad(Rmap, pad), sheet)


def _resting(hand: Hand, k: int, matte: np.ndarray, at: tuple, zc: np.ndarray, R: np.ndarray, s: float) -> np.ndarray:
    """(h, w) 0..1 over a box of the frame whose top left is px ``at``: how much the water there (its middle's depth
    ``zc``, its radius ``R``) lies on the skin (by the matte) rather than running clear of it (REST). The orb, which
    floats, never does. The hand's depth is read on a grid 4 px apart."""
    hh, ww = zc.shape
    mh = matte[at[0]:at[0] + hh, at[1]:at[1] + ww].astype(np.float32) / 255.0
    if mh.shape != zc.shape or float(mh.max()) < 0.01:
        return np.zeros(zc.shape, np.float32)
    q = 4
    gh, gw = -(-hh // q), -(-ww // q)
    gx = ((np.arange(gw) + 0.5) * (ww / gw) - 0.5 + at[1]).astype(np.float32)
    gy = ((np.arange(gh) + 0.5) * (hh / gh) - 0.5 + at[0]).astype(np.float32)
    zh = hand.depth(k, *np.meshgrid(gx, gy)).astype(np.float32)
    zh = cv2.resize(zh, (ww, hh), interpolation=cv2.INTER_LINEAR)
    lie = _ss(REST[1], REST[0], (zc - zh) / np.maximum(R, 1.0)) * (1.0 - _ss(0.1 * s, 0.25 * s, R))
    return (lie * mh).astype(np.float32)


def _window(nx, ny, nz, Lk: np.ndarray, size: float, bars=0.0) -> np.ndarray:
    """The key light's window mirrored in a surface of normal n: a soft rounded rectangle (WINDOW), ``size`` times as
    wide, brighter at its top, crossed at ``bars`` (0..1) of their strength by its frame's thin dark bars (WINDOW_BARS).
    Hard-edged and lit evenly, it was a white sticker on a crystal ball."""
    e1 = _unit(np.array([Lk[2], 0.0, -Lk[0]], np.float32))
    e2 = np.cross(Lk, e1)
    wx_, wy_, wz_ = 2.0 * nz * nx, 2.0 * nz * ny, 2.0 * nz * nz - 1.0
    wa = (wx_ * e1[0] + wy_ * e1[1] + wz_ * e1[2]) / np.float32(WINDOW[0] * size)
    wb = (wx_ * e2[0] + wy_ * e2[1] + wz_ * e2[2]) / np.float32(WINDOW[1] * size)
    p = WINDOW[4]  # a rounded rectangle: a window, a lamp's shade
    win = _ss(1.0 + WINDOW[2], 1.0 - WINDOW[2], (np.abs(wa) ** p + np.abs(wb) ** p) ** (1.0 / p)).astype(np.float32)
    if np.any(np.asarray(bars) > 0):
        a_, b_, w_, dark = WINDOW_BARS
        frame_ = np.maximum(np.exp(-((wa - a_) / w_) ** 2), np.exp(-((wb - b_) / w_) ** 2))
        win = win * (1.0 - dark * np.asarray(bars, np.float32) * frame_)
    return win * (wx_ * Lk[0] + wy_ * Lk[1] + wz_ * Lk[2] > 0) * (0.75 + 0.25 * np.clip(-wb, -1.0, 1.0))


def _dots(C: np.ndarray, dots: list, at: tuple, s: float, Lk: np.ndarray, lit: np.ndarray, hand: Hand, k: int,
          matte: np.ndarray | None, own: np.ndarray | None = None) -> None:
    """The key light's glint on each small free drop (Field.dots), in place on the premultiplied colour ``C`` whose top
    left is the frame's px ``at`` (and, in key light units, on ``own[..., 0]``, the light the motion blur keeps): a
    bright point towards the light, which the motion blur draws out into a streak, as bright as the drop's facet faces
    the light (Blob.glint). Shaded pixel by pixel alone, a drop a few px across lost its glint and read as a grey speck.
    Hidden where the hand is in front of the drop."""
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
    for x, y, _, r, facing in d[keep]:
        gw = float(1.0 - _ss(0.5 * DOT[0] * s, DOT[0] * s, r)) * facing  # a bigger drop has a glint of its own
        sg = max(DOT[2], DOT[3] * r)
        cx, cy = x + 0.35 * r * lxy[0] - at[1], y + 0.35 * r * lxy[1] - at[0]
        e = int(3 * sg + 2)
        ya, yb, xa, xb = max(0, int(cy) - e), min(hh, int(cy) + e + 1), max(0, int(cx) - e), min(ww, int(cx) + e + 1)
        if yb <= ya or xb <= xa:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        pk = DOT[1] * gw * min(1.0, r / (2.0 * DOT[2]))  # a speck under a pixel glints over its share of one
        g = pk * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sg * sg))
        C[ya:yb, xa:xb] += g[..., None] * lit
        if own is not None:
            own[ya:yb, xa:xb, 0] += DOT_KEEP * g


def _shutter(t) -> np.ndarray:
    """How open the shutter is at ``t`` (-0.5 .. 0.5) of the exposure: fully over its middle, opening and closing in a
    raised cosine over BLUR_TAPER of it at either end."""
    a = 0.5 * BLUR_TAPER
    edge = np.clip((np.abs(t) - (0.5 - a)) / a, 0.0, 1.0)
    return (0.5 + 0.5 * np.cos(math.pi * edge)).astype(np.float32)


_RAGS = np.random.default_rng(20261020).uniform(-1.0, 1.0, 4096).astype(np.float32)  # the values _ragged runs through
BLUR_CHUNK = 1 << 21  # taps worked at once in _motion_blur (all of a lash's at once held hundreds of MB a frame)


def _ragged(c) -> np.ndarray:
    """A smooth random value in [-1, 1] at each ``c`` px along a line, its features BLUR_RAGGED[1] px: value noise off
    a fixed random table, so no two neighbouring streaks are alike and nothing repeats across a frame."""
    u = np.asarray(c, np.float64) / BLUR_RAGGED[1]
    i = np.floor(u)
    f = (u - i).astype(np.float32)
    i = i.astype(np.int64) % _RAGS.size
    f = f * f * (3.0 - 2.0 * f)
    return _RAGS[i] + (_RAGS[(i + 1) % _RAGS.size] - _RAGS[i]) * f


def _motion_blur(C: np.ndarray, A: np.ndarray, mx: np.ndarray, my: np.ndarray, box: tuple, H: int, W: int,
                 own: np.ndarray | None = None, lit: np.ndarray | None = None, rag: np.ndarray | None = None,
                 k: int = 0) -> tuple:
    """The premultiplied colour ``C`` (h, w, 3) and coverage ``A`` (h, w) over ``box`` of the frame, each pixel spread
    along the path (mx, my) px it travels over the exposure, centred on it, as the shutter lets it through (_shutter),
    in taps BLUR_STEP apart, the same for every pixel moving alike: scattered along its own motion, so a fast drop
    smears past its outline into a streak and a slow root stays sharp beside a fast tail.

    ``own`` (h, w, 5) is what keeps some of itself wherever it passes over the exposure (BLUR_KEEP: a fast jet's
    glints and froth smear into streaks, not into nothing): the water's own light in units of the key light ``lit``,
    its froth's cover, and the sheet a lash sweeps (its share, and that times the shift px, x and y, it gives the plate
    under it). ``rag`` (h, w) 0..1: how torn the water is: its smear is longer or shorter streak by streak
    (BLUR_RAGGED, drawn afresh on frame ``k``) and its own light kept more just behind its smear's front end
    (BLUR_RIM).
    Returns (the box, grown to hold the smear, C, A, the sheet (h, w, 3): its share and the shift x, y px it gives the
    plate; None where there is none)."""
    y0, y1, x0, x1 = box
    sp = np.hypot(mx, my)
    on = (A > 1e-4) | (C.max(-1) > 1e-4)
    if not on.any() or float(sp[on].max()) < 0.5:
        return box, C, A, None
    iy, ix = np.nonzero(on)
    mxs, mys = mx[iy, ix].astype(np.float32), my[iy, ix].astype(np.float32)
    rg = np.zeros(iy.size, np.float32) if rag is None else rag[iy, ix]
    tr = np.flatnonzero(rg > 0)
    if tr.size:  # each streak of torn water its own length
        spn = np.maximum(sp[iy[tr], ix[tr]], 1e-3)
        across = ((iy[tr] + y0) * mxs[tr] - (ix[tr] + x0) * mys[tr]) / spn
        stretch = 1.0 + BLUR_RAGGED[0] * rg[tr] * _ragged(across + 1013.7 * k)
        mxs[tr] *= stretch
        mys[tr] *= stretch
    lens = np.hypot(mxs, mys)
    e = int(math.ceil(0.5 * float(lens.max()))) + 2
    ny0, ny1, nx0, nx1 = max(0, y0 - e), min(H, y1 + e), max(0, x0 - e), min(W, x1 + e)
    hh, ww = ny1 - ny0, nx1 - nx0
    keep = own is not None and lit is not None
    val = np.concatenate([C[iy, ix], A[iy, ix, None]] + ([own[iy, ix]] if keep else []), 1).astype(np.float32)
    nch = val.shape[1]
    n = np.clip(np.ceil(lens / BLUR_STEP), 1, BLUR_TAPS).astype(np.int64)
    px, py = ix + (x0 - nx0), iy + (y0 - ny0)
    acc = np.zeros((nch, hh * ww), np.float64)
    peak = np.zeros((3, hh * ww), np.float32)
    still = np.flatnonzero(n == 1)  # a pixel that hardly moves stays where it is
    idx = py[still] * ww + px[still]
    for c in range(nch):
        acc[c] += np.bincount(idx, val[still, c], hh * ww)
    moving = np.flatnonzero(n > 1)
    ends = np.cumsum(n[moving])
    lo = 0
    while lo < moving.size:  # in chunks of about BLUR_CHUNK taps
        hi = max(lo + 1, int(np.searchsorted(ends, (ends[lo - 1] if lo else 0) + BLUR_CHUNK)))
        mv = moving[lo:hi]
        lo = hi
        nm = n[mv]
        first = np.cumsum(nm) - nm
        reps = np.repeat(mv, nm)  # each moving pixel once per tap...
        t = ((np.arange(reps.size) - np.repeat(first, nm) + 0.5) / n[reps] - 0.5).astype(np.float32)  # ... at its
        # share of the exposure
        X = px[reps] + t * mxs[reps]
        Y = py[reps] + t * mys[reps]
        win = _shutter(t)  # the shutter opens and closes over the exposure: the smear's ends fade (squared off, a slab)
        wt = win / np.repeat(np.add.reduceat(win, first), nm)
        xf, yf = np.floor(X), np.floor(Y)
        fx_, fy_ = (X - xf).astype(np.float32), (Y - yf).astype(np.float32)
        xi, yi = xf.astype(np.int64), yf.astype(np.int64)
        v = val[reps]
        for dx, dy, cw in ((0, 0, (1 - fx_) * (1 - fy_)), (1, 0, fx_ * (1 - fy_)), (0, 1, (1 - fx_) * fy_),
                           (1, 1, fx_ * fy_)):
            xx, yy = xi + dx, yi + dy
            ok = np.flatnonzero((xx >= 0) & (xx < ww) & (yy >= 0) & (yy < hh))
            idx = yy[ok] * ww + xx[ok]
            w_ = wt[ok] * cw[ok]
            for c in range(nch):
                acc[c] += np.bincount(idx, v[ok, c] * w_, hh * ww)
        if keep:  # the most of what keeps that passed over each pixel
            rim = BLUR_RIM[0] * rg[reps] * np.exp(-0.5 * (((0.5 - t) * lens[reps] - 1.0) / (0.5 * BLUR_RIM[1])) ** 2)
            xn, yn = np.rint(X).astype(np.int64), np.rint(Y).astype(np.int64)
            ok = np.flatnonzero((xn >= 0) & (xn < ww) & (yn >= 0) & (yn < hh))
            at = yn[ok] * ww + xn[ok]
            np.maximum.at(peak[0], at, (BLUR_KEEP[0] * win[ok] + rim[ok]) * v[ok, 4])
            np.maximum.at(peak[1], at, BLUR_KEEP[1] * win[ok] * v[ok, 5])
            if float(val[:, 6].max()) > 0:
                np.maximum.at(peak[2], at, BLUR_KEEP[2] * win[ok] * v[ok, 6])
    acc = acc.reshape(nch, hh, ww).astype(np.float32)
    C, A = np.ascontiguousarray(acc[:3].transpose(1, 2, 0)), acc[3].copy()
    sheet = None
    if keep:  # (softened a little: the most taken pixel by pixel combed the froth's smear into hard parallel lines)
        pk = cv2.GaussianBlur(peak.reshape(3, hh, ww).transpose(1, 2, 0), (0, 0), BLUR_SOFT)
        gain = np.maximum(pk[..., 0] - acc[4], 0.0)
        cover = np.maximum(pk[..., 1] - acc[5], 0.0)
        C += gain[..., None] * lit
        A = np.minimum(A + cover, 1.0)
        share = np.clip(np.maximum(pk[..., 2], acc[6]), 0.0, 1.0)
        if float(share.max()) > 0.01:
            mean = share / np.maximum(acc[6], 1e-4)  # the shift of the sheet that passed here, at its share
            sheet = np.stack([share, acc[7] * mean, acc[8] * mean], -1).astype(np.float32)
    return (ny0, ny1, nx0, nx1), C, A, sheet


CAUSTIC_GAIN, CAUSTIC_KEEP = 1.0, 0.3  # the focus's strength, and the key light's share of what lights the hand (the
# rest is the room's soft light, which the water does not focus)
CAUSTIC_NEAR = 0.02  # hand sizes: the least the water's light falls to the hand (water lying on the palm, its pool,
# throws its shadow right under it: taken as falling NEAR's least, the pool cast a shadow beside it like a raised lump)
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
SPOT_MAX = 3.5  # ... which multiplies the key light by at most this, rolled off (held to CAUSTIC_MAX with the rest of
# the water's light, the spot was a pale pink disc barely brighter than the lit skin: a lens gathers the light through
# all of it into a hot spot)...
SPOT_NET = (96, 4, 1.2, (0.6, 1.0), 3.5, 0.03, 0.5)  # ... and its light lies in the net of bright lines that rays
# through the orb's moving surface (its waves and their finest ripples) draw where they land: a grid of this many rays
# across it, laid between each two as if refined this many times (the light between neighbouring rays lands between
# where they do: splatted as points, a few to a cell, they were a cluster of soft blobs, bubble wrap), splatted at the
# picture's resolution and blurred this much (px), the light there this base plus this gain times how many rays land
# there against the spot's mean, at most this many times it, how fast its ripples drift (of its radius a frame), and
# the share of the finest ripples the rays are bent by (all of them folded the light into glitter, not long lines)
PAD_MAX = 1.2  # hand sizes: the most the light it throws reaches off the water


def _splat(tgt: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
    """Adds weights ``w`` at fractional cells (x, y) of ``tgt``, each shared among its four nearest (rounded to one, a
    focus moving a fraction of a cell jumped from cell to cell, and shimmered); what lands off ``tgt`` is dropped (piled
    onto its edge, it drew the box's outline)."""
    gh, gw = tgt.shape
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx_, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    w = np.asarray(w, np.float32)
    acc = np.zeros(gh * gw, np.float64)
    for dx, dy, ww in ((0, 0, (1 - fx_) * (1 - fy)), (1, 0, fx_ * (1 - fy)), (0, 1, (1 - fx_) * fy), (1, 1, fx_ * fy)):
        xi, yi = x0 + dx, y0 + dy
        on = (xi >= 0) & (xi < gw) & (yi >= 0) & (yi < gh)
        acc += np.bincount(yi[on] * gw + xi[on], (w * ww)[on], gh * gw)
    tgt += acc.reshape(gh, gw).astype(tgt.dtype)


def caustic(P: np.ndarray, layer: Layer, k: int, hand: Hand, room: Room, matte: np.ndarray,
            held: tuple | None = None, lobes: tuple = (), noise: Noise | None = None) -> tuple:
    """The water's light on the hand behind it, in place: each cell of water in front of the hand sends its share of
    the key light along the ray bent through it to the hand, at the hand's own depth there, and lights it there; the
    light the hand had there without the water is taken away (energy kept: a bright focus inside a soft shadow). A
    small drop's share fades: under a room's wide light its shadow and its focus spread wider than it is. ``held``,
    the orb over the palm (x, y, z, R, w): its light lands as one, wobbling with its ``lobes`` (Scene.lobes), in the
    net of brighter lines rays through its moving surface draw (_rays; ``noise``: its finest ripples), and its bulk
    also takes some of the room's soft light off the palm right under it, both at w of their strength, the shade
    shrinking as it goes. Returns the box it changed."""
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
    d = np.clip(zq[iy, ix] - hq[iy, ix] - zh, CAUSTIC_NEAR * s, NEAR[1] * s)  # px from the water's back to the hand
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
    g = 1.0 + CAUSTIC_GAIN * lit  # rolled off into its most (clipped, a focus was a flat disc with a hard edge)
    top = CAUSTIC_MAX - 1.0
    g = np.where(g > 1.0, 1.0 + top * np.tanh((g - 1.0) / top), np.maximum(g, 0.55)).astype(np.float32)
    net = None
    if wh > 0:  # the held orb's spot, rolled off into its own most
        rays = None if noise is None else _rays(layer, cx, cy, R, dh, L, noise, k, s)
        light, net = _spot(gh, gw, q, x0, y0, cx, cy, R, dh, L, lobes, rays)
        sp = wh * (1.0 - SPOT[2]) * light
        hot = SPOT_MAX - 1.0
        g = np.maximum(g + np.where(sp > 0.0, hot * np.tanh(sp / hot), sp), 0.55).astype(np.float32)
    g = cv2.resize(g, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    if net is not None:  # its net, at the picture's resolution: thin lines
        net = np.pad(net, ((0, g.shape[0] - net.shape[0]), (0, g.shape[1] - net.shape[1])), constant_values=1.0)
        g = np.where(g > 1.0, 1.0 + (g - 1.0) * (1.0 + wh * (net - 1.0)), g)
    m = matte[y0:y1, x0:x1].astype(np.float32) / 255.0
    lift = CAUSTIC_KEEP * (g - 1.0)
    edge = _box_fade(y1 - y0, x1 - x0, max(4, pad // 2))  # what reaches the box's edge fades by it (faded over an
    # eighth of the box, the light moved when a drip grew the box)
    P[y0:y1, x0:x1] *= (1.0 + lift * m * edge)[..., None]
    if shade_box[1] <= shade_box[0]:
        return (y0, y1, x0, x1)
    return (min(y0, shade_box[0]), max(y1, shade_box[1]), min(x0, shade_box[2]), max(x1, shade_box[3]))


def _rays(layer: Layer, cx: float, cy: float, R: float, dh: float, L: np.ndarray, noise: Noise, k: int,
          s: float) -> tuple[np.ndarray, np.ndarray]:
    """Where rays of the key light (along -``L``) through the held orb (radius ``R`` at (cx, cy), its back ``dh`` over
    the palm) land on the palm: ((n, n, 2) px about their mean, (n, n) which went through it), a grid of SPOT_NET[0]
    across it, bent in at its front and out at its back (the front's normal mirrored, as shade traces it) by its surface
    as ``layer`` holds it (its waves) and the finest ripples its glints break up in (SPARKLE, off ``noise`` on frame
    ``k``; ``s`` a hand's size px). Where neighbouring rays cross, the light piles up into the bright lines light
    through moving water throws."""
    n = SPOT_NET[0]
    u = np.linspace(-0.92, 0.92, n)
    ux, uy = np.meshgrid(u, u)
    on = ux * ux + uy * uy < 0.92 ** 2
    px, py = (cx + R * ux).astype(np.float32).ravel(), (cy + R * uy).astype(np.float32).ravel()
    ly0, lx0 = layer.box[0], layer.box[2]
    gy, gx = np.gradient(cv2.GaussianBlur(layer.h, (0, 0), 0.7))
    gy, gx = np.clip(gy, -6, 6), np.clip(gx, -6, 6)

    v = SPOT_NET[5] * R

    def normal(x, y):
        fine = SPOT_NET[6] * SPARKLE
        ax = _remap(gx, x - lx0, y - ly0) + fine * noise.flow(3, x, y, v, 0.4 * v, k, SPARKLE_SIZE * s)
        ay = _remap(gy, x - lx0, y - ly0) + fine * noise.flow(4, x, y, v, 0.4 * v, k, SPARKLE_SIZE * s)
        nd = np.sqrt(ax * ax + ay * ay + 1.0)
        return -ax / nd, -ay / nd, 1.0 / nd

    nx, ny, nz = normal(px, py)
    (tx, ty, tz), _ = _refract(-L[0], -L[1], -L[2], nx, ny, nz, 1.0 / ETA[1])
    path = 2.0 * _remap(layer.h, px - lx0, py - ly0) / np.maximum(-tz, 0.25)
    ex, ey = px + tx * path, py + ty * path
    bx, by, bz = normal(ex, ey)
    (ox, oy, oz), kk = _refract(tx, ty, tz, -bx, -by, bz, ETA[1])
    out = _ss(0.0, TIR_SOFT, kk)
    ox, oy, oz = out * ox + (1 - out) * tx, out * oy + (1 - out) * ty, out * oz + (1 - out) * tz
    land = np.stack([ex + ox / np.maximum(-oz, 0.3) * dh, ey + oy / np.maximum(-oz, 0.3) * dh], 1).reshape(n, n, 2)
    return land - land[on].mean(0), on


def _spot(gh: int, gw: int, q: int, x0: int, y0: int, cx: float, cy: float, R: float, dh: float, L: np.ndarray,
          lobes: tuple = (), rays: tuple | None = None) -> tuple:
    """The held orb's light on the palm, in key light units on the grid (gh, gw) of q px cells from (x0, y0), and the
    net its light lies in, on the (gh q, gw q) px under it (a factor on the light it adds, once that is rolled off;
    None without ``rays``): a ball lens of radius R at (cx, cy), its middle dh + R over the palm, lit along L. Where the
    line through its middle along the light meets the palm: the disc of its shadow, and inside it the spot it focuses
    the light it lets through into (as wide as its rays still are there, a ball's focus being n R / 2 (n - 1) from its
    middle), both blurred by the light's own width (CAUSTIC_SOFT). Traced cell by cell instead, through a surface that
    never holds still and onto a palm near its focus, the light was a jagged shape that jumped from frame to frame; so
    the surface's life is laid over it: the spot wobbles with the orb's ``lobes`` (magnified, SPOT_LOBES), and in its
    core its light lies in the net of bright lines its ``rays`` draw (_rays: where they land, scaled to the spot, the
    light between neighbouring rays laid between them, splatted px by px: SPOT_NET). (Laid on the light before it is
    rolled off into its most, the net was flattened out of a spot many times over it.)"""
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
    net = None
    if rays is not None and int(rays[1].sum()) > 8:  # the rays, spread as wide as the spot, splatted where they land
        land, on = rays
        spread = max(float(np.sqrt((land[on] ** 2).sum(1).mean() / 2.0)), 1e-6)
        at = (land / spread * sig * q).astype(np.float32)  # px about the spot's middle
        up = SPOT_NET[1]
        fine = cv2.resize(at, None, fx=up, fy=up, interpolation=cv2.INTER_LINEAR)
        inside = cv2.resize(on.astype(np.float32), None, fx=up, fy=up, interpolation=cv2.INTER_LINEAR) > 0.999
        acc = np.zeros((gh * q, gw * q), np.float32)
        _splat(acc, fine[..., 0][inside] + float(c[0] - x0), fine[..., 1][inside] + float(c[1] - y0),
               np.ones(int(inside.sum()), np.float32))
        acc = cv2.GaussianBlur(acc, (0, 0), SPOT_NET[2])
        core = _ss(0.35, 0.6, cv2.resize((spot / max(float(spot.max()), 1e-9)).astype(np.float32), acc.shape[::-1],
                                         interpolation=cv2.INTER_LINEAR))  # in its brightest half only (over all of
        csum = max(float(core.sum()), 1e-9)  # a wide spot, a scrawl on the wrist)
        base, gain = SPOT_NET[3]
        lines = base + gain * np.minimum(acc / max(float((acc * core).sum()) / csum, 1e-9), SPOT_NET[4])
        lines /= max(float((lines * core).sum()) / csum, 1e-6)  # as much light as the spot has, gathered into lines
        net = (1.0 + core * (lines - 1.0)).astype(np.float32)
    return spot - shadow, net


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
    """Air caught in the water, over it, in place: each (x, y, r, seen) a small sphere of air, a lens that spreads the
    light through it: dark towards its edge (light turned away from the eye there), clear in its middle, one tiny glint
    towards the light; at ``seen`` of its strength, and only where the water covers the picture. (With a silvered rim
    they were rings: glass beads, stickers.)"""
    H, W = P.shape[:2]
    by0, by1, bx0, bx1 = layer.box
    for x, y, r, seen in items:
        if r < 0.8 or not (by0 <= y < by1 and bx0 <= x < bx1):
            continue
        wgt = float(layer.alpha[int(y) - by0, int(x) - bx0]) * seen
        rr = int(r + 3)
        ya, yb, xa, xb = max(0, int(y) - rr), min(H, int(y) + rr + 1), max(0, int(x) - rr), min(W, int(x) + rr + 1)
        if yb <= ya or xb <= xa or wgt <= 0.01:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        d = np.hypot(xx - x, yy - y) / np.float32(r)
        inside = _ss(1.0 + 0.8 / r, 1.0 - 0.8 / r, d)
        dark = BUBBLE_LOOK[0] * inside * _ss(0.35, 0.95, d)
        gx, gy = x + 0.4 * r * L[0], y + 0.4 * r * L[1]
        pin = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * (0.12 * r + 0.45) ** 2))
        patch = P[ya:yb, xa:xb]
        patch *= (1.0 - wgt * dark)[..., None]
        patch += (wgt * BUBBLE_LOOK[1] * pin)[..., None] * lit[None, None, :]


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
        self.info = np.zeros((0, 11))  # r, foam, born, life, fade, stretch, gravity (px a frame squared), delay,
        # free (1: it bridges to no other free piece), field weight, how much of the key light's glint it catches
        self.count = 0  # pieces added so far

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
        u = ((self.count + np.arange(m)) * 0.6180339887) % 1.0  # spread evenly over the pieces, one after another
        self.count += m
        facing = DOT_FACING[0] + (1.0 - DOT_FACING[0]) * _ss(DOT_FACING[1], DOT_FACING[2], u)
        info = np.stack(np.broadcast_arrays(np.zeros(m), r, foam, k, life, fade, stretch, g, delay, float(free), w,
                                            facing)[1:], -1).astype(np.float64)
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
        r0, foam, born, life, fade, stretch, _, delay, free, w, facing = self.info.T
        age = k - born
        r = r0 * kz * np.clip((life - age) / fade, 0.0, 1.0) ** 0.8
        vxy = vxy * kz[:, None]
        z = self.x[:, 2]
        out = []
        for j in np.flatnonzero((age < life) & (age >= delay) & (r >= 0.5)):
            shape, move = _moving(r[j], vxy[j], stretch[j])
            out.append(Blob(xy[j, 0], xy[j, 1], z[j], r[j], shape, vxy[j, 0], vxy[j, 1], 0.0, foam[j], w=w[j],
                            free=bool(free[j]), move=move, glint=float(facing[j])))
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
    the normal of a surface it burst on (the palm), along which nothing goes in and little goes out: it is flattened
    against it and spreads over it. Returns the body, as frame ``k`` sees it. (On the room's wall it bursts as water
    that stays there: _wall.)"""
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
    if flat is not None:  # flattened against what it hit: a wide, thin splat, its edge torn into uneven fingers
        sets, lobes, amp = SPLAT[3:]
        fingers = tuple((int(rng.integers(*lobes)), float(rng.uniform(*amp)), float(rng.uniform(0, 2 * math.pi)))
                        for _ in range(sets))
        body = [body[0]._replace(r=SPLAT[0] * R, flat=SPLAT[1], foam=SPLAT[2], move=(), modes=fingers)]
    d = dirs(n_lig)  # its torn skin: thin threads flung out of it, each beading up into a few drops a frame on
    rl = R * rng.uniform(*LIGAMENT_R, n_lig)
    st = rng.uniform(*LIGAMENT_STRETCH, n_lig)
    x0 = c + d * (R * rng.uniform(0.9, 1.1, (n_lig, 1)) + (st * rl)[:, None])
    v0 = carry + d * rng.uniform(3.5, 6.0, (n_lig, 1)) * pace
    drops.add(k, x0, v0, rl, foam=0.2, life=1 + LIGAMENT_LIFE, fade=1.5, stretch=st, delay=1, g=g)
    per, big = LIGAMENT_BEADS
    off = np.linspace(-0.7, 0.7, per)[None, :, None] * (st * rl)[:, None, None] * d[:, None, :]
    drops.add(k, (x0[:, None, :] + off).reshape(-1, 3), (v0[:, None, :] + 0.12 * off).reshape(-1, 3),
              np.repeat(big * rl, per) * rng.uniform(0.7, 1.2, n_lig * per), foam=0.1, delay=LIGAMENT_LIFE, g=g)
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


def _air(bub: np.ndarray, bub_r: np.ndarray, rise: np.ndarray, spin: np.ndarray, axis: np.ndarray, circ: float,
         t: float, c: np.ndarray, r: float, s: float, seen: float) -> list[tuple]:
    """The air caught in the orb (unit offsets ``bub``) carried round by its circulation, each at its own share of it
    and wandering on its own (``spin``: share, phase, rate; BUBBLE_DRIFT), and rising through it at ``rise`` turns of
    its height a second (each wells up at its bottom and is gone at its top): (x, y, r, seen) each."""
    out = []
    for b0, br, up, (share, ph, hz) in zip(bub, bub_r, rise, spin):
        bxu, byu, bzu = _rotate(b0[None], axis, circ * share)[0]
        bxu += BUBBLE_DRIFT[2] * math.sin(2.0 * math.pi * hz * t + ph)
        byu += BUBBLE_DRIFT[2] * math.sin(2.0 * math.pi * 0.8 * hz * t + 1.7 * ph)
        lim = math.sqrt(max(0.72 ** 2 - bxu * bxu - bzu * bzu, 0.02))
        hgt = ((lim - byu) / (2.0 * lim) + up * t) % 1.0  # 0 at its bottom, 1 at its top
        byu = lim - 2.0 * lim * hgt
        grow = 1.0 + 0.12 * (1.0 - bzu)  # the far ones, seen through the water, a little magnified...
        rim = math.hypot(bxu, byu) * grow  # ... and those seen near its rim, through its grazing surface, gone (they sat
        vis = (0.75 + 0.25 * bzu) * (1.0 - float(_ss(0.55, 0.85, rim)))  # on its outline like pearls)
        vis *= min(1.0, 5.0 * hgt, 5.0 * (1.0 - hgt))
        out.append((c[0] + r * bxu * grow, c[1] + r * byu * grow, br * s, vis * seen))
    return out


def _body(theta, r: float, wobble: tuple) -> np.ndarray:
    """The wall sheet's body: its radius px at angles ``theta`` (Sheet's r and wobble), its fingers aside."""
    th = np.asarray(theta, np.float64)
    return r * (1.0 + sum(a * np.cos(l * th - ph) for l, a, ph in wobble))


def _smax(a, b, k: float):
    """The smooth union of two insides (signed px, + inside): the larger, blended over ``k`` px into a fillet."""
    h = np.clip(0.5 + 0.5 * (a - b) / k, 0.0, 1.0)
    return b + (a - b) * h + np.float32(k) * h * (1.0 - h)


def _inside(dx, dy, r: float, fingers: tuple, wobble: tuple) -> tuple:
    """How far px inside a wall sheet's outline points (dx, dy) px from its middle are (+ inside), and its body's radius
    at their angles: its body, and each finger a capsule from inside it out to a bead at its tip, of near even width,
    meeting the body in a fillet (Sheet's r, fingers and wobble)."""
    rho = np.hypot(dx, dy)
    re = _body(np.arctan2(dy, dx), r, wobble).astype(np.float32)
    d = re - rho
    xs, ys = dx[0], dy[:, 0]  # (a grid: each finger is worked over its own box of it)
    for ang, ln, hw, bead in fingers:
        w = hw * r
        if ln <= 0.0 or w < 0.3:
            continue
        ux, uy = math.cos(ang), math.sin(ang)
        a0, a1 = 0.6 * r, (1.0 + ln) * r
        e = max(bead, 1.1) * w + FINGER_BLEND * w + 2.0
        bx = np.searchsorted(xs, [min(a0 * ux, a1 * ux) - e, max(a0 * ux, a1 * ux) + e])
        by = np.searchsorted(ys, [min(a0 * uy, a1 * uy) - e, max(a0 * uy, a1 * uy) + e])
        sl = (slice(by[0], by[1]), slice(bx[0], bx[1]))
        if by[1] <= by[0] or bx[1] <= bx[0]:
            continue
        fx_, fy_ = dx[sl], dy[sl]
        a = np.clip(fx_ * ux + fy_ * uy, a0, a1)
        f = np.float32(w) * (1.1 - 0.2 * (a - a0) / (a1 - a0)) - np.hypot(fx_ - a * ux, fy_ - a * uy)
        tip = np.float32(bead * w) - np.hypot(fx_ - a1 * ux, fy_ - a1 * uy)
        d[sl] = _smax(d[sl], np.maximum(f, tip), FINGER_BLEND * w)
    return d, re


def _slide(t, vmax, T):
    """How far px a drop setting off down the wall ``t`` frames ago has run, and how fast (px a frame): up to ``vmax``
    over ``T`` frames, held back by the wall at first."""
    t = np.maximum(np.asarray(t, np.float64), 0.0)
    T = np.maximum(T, 1e-6)
    run = vmax * np.where(t < T, t * t / (2.0 * T), t - T / 2.0)
    return run, vmax * np.minimum(1.0, t / T)


def _wall(scenes: list, drops: Drops, hit: tuple, n: int, cam: np.ndarray | None, rng) -> list:
    """The orb bursting on the room's wall on frame ``hit[0]``: ``hit`` = (frame, x, y, z, R, s, fan, size), the px it
    hits at on that frame and its depth, the ball's radius and a hand's size there, the unit direction along the wall
    it was thrown in and how hard. The water stays on the wall (Scene.film, from that frame on, held to the room as
    the camera moves): it lands as a squashed body churned white in its middle, throwing a burst of spray out along
    the wall and bigger drops off its rim towards the camera, and spreads as a sheet, its edge torn into fingers each
    ending in a bead, thinning; drops flung out of it land on the wall round it and slow to a stop, the smallest going
    into the wall, the big ones sliding down; drips run down from its lower edge, each leaving a wet trail; and the
    sheet drains, over a few frames, into the wet patch it leaves, whose lower edge sags into runs. Over the next
    frames a few drops let go of its lower edge and fall. Returns those, (frame, px and depth, velocity px a frame,
    radius, gravity px a frame squared) each, for the caller to let go on their frame. Every piece eases in: nothing
    of it comes or goes between two frames."""
    kh, xh, yh, zh, R, s, fan, size = hit
    to_room = np.linalg.inv(cam[kh]) if cam is not None else np.eye(3)
    q = to_room @ np.array([xh, yh, 1.0])
    c0 = q[:2] / q[2]
    fa = math.atan2(fan[1], fan[0])
    nf = int(rng.integers(WALL_FINGERS[0], WALL_FINGERS[1] + 1))  # its fingers: mostly down and out sideways the way
    side, turn = (0.0, 1.0) if fan[0] >= 0 else (math.pi, -1.0)  # it was thrown (tipped down from there), short
    n_down = int(round(FINGER_DIR[0] * nf))  # upwards
    n_side = int(round(FINGER_DIR[1] * nf))
    f_ang = np.r_[0.5 * math.pi + rng.uniform(-0.5, 0.5, n_down) * FINGER_DIR[2],
                  side + turn * np.abs(rng.normal(0.0, FINGER_DIR[3], n_side)),
                  rng.uniform(0.0, 2.0 * math.pi, nf - n_down - n_side)]
    f_len = rng.uniform(*FINGER_LEN, nf) * (0.8 + 0.2 * np.maximum(np.cos(f_ang - fa), 0.0))
    f_len = np.minimum(f_len, FINGER_UP + (FINGER_LEN[1] - FINGER_UP) * _ss(0.7, 0.0, -np.sin(f_ang)))
    f_wid = rng.uniform(*FINGER_W, nf)
    f_bead = rng.uniform(*FINGER_BEAD, nf)
    wob = tuple((l, a * float(rng.uniform(0.5, 1.5)), float(rng.uniform(0.0, 2.0 * math.pi))) for l, a in WALL_WOBBLE)
    s0, s1, tau_s = WALL_SPREAD
    fan2 = np.asarray(fan, np.float64)

    def radius(t):  # px, frame(s) ``t`` after the hit
        return R * (s1 - (s1 - s0) * np.exp(-np.maximum(t, 0.0) / tau_s))

    def centre(t):  # it spreads further the way it was thrown
        return c0 + fan2 * (FAN * 0.4 * R * (1.0 - np.exp(-np.maximum(t, 0.0) / tau_s)))[..., None]

    def grown(t):  # how far its fingers are out
        return 1.0 - math.exp(-(max(t, 0.0) + 0.5) / WALL_FINGERS[2])

    def edge(ang, t):  # px on the room's wall where its body's edge is at angles ``ang`` on frame ``t`` after the hit
        return centre(t) + np.stack([np.cos(ang), np.sin(ang)], -1) * _body(ang, radius(t), wob)[..., None]

    def tip(j, t):  # px on the room's wall of finger ``j``'s tip
        return centre(t) + np.array([math.cos(f_ang[j]), math.sin(f_ang[j])]) * (1.0 + f_len[j] * grown(t)) * radius(t)

    # the drops on the wall: where on its edge they leave (angle, frame), how far they run on and how fast that stops,
    # their size, when the smallest are gone, how fast the big ones slide down and how soon they get going
    m = max(4, int(WALL_BEADS[0] * (0.4 + 0.6 * size)))
    ang = np.where(rng.random(m) < 0.6, fa + rng.normal(0.0, 0.9, m), rng.uniform(0.0, 2.0 * math.pi, m))
    rb = R * _sizes(rng, m, *WALL_BEADS[1], 1.5)
    far = R * rng.uniform(*WALL_BEADS[2], m) * (WALL_BEADS[1][0] * R / rb) ** 0.2  # the smaller, the further
    tau = rng.uniform(*WALL_BEADS[3], m)
    born = rng.integers(0, 2, m).astype(float)
    start = edge(ang, born)
    # a drop left at each finger's tip, carried out with it as it grows and running on a little past it
    kt = float(WALL_TIPS[2])
    t_start = np.stack([tip(j, kt) for j in range(nf)])
    t_far = np.hypot(*(np.stack([tip(j, 60.0) for j in range(nf)]) - t_start).T) + R * rng.uniform(*WALL_TIPS[1], nf)
    t_r = np.clip(0.8 * f_bead * f_wid * s1, *WALL_TIPS[0]) * R
    # the drips, off its lower edge, well apart (in an even row along it, three pearls on a string; two together, a
    # clump); each runs down from one of the runs its wet patch sags into
    d_ang: list[float] = []
    for a in rng.uniform(0.25 * math.pi, 0.75 * math.pi, 24):
        if len(d_ang) < WALL_DRIPS[0] and all(abs(float(np.angle(np.exp(1j * (a - b))))) > 0.35 for b in d_ang):
            d_ang.append(float(a))
    md = len(d_ang)
    d_born = WALL_DRIPS[4] + rng.integers(0, 2, md).astype(float)
    d_start = c0 + (edge(np.array(d_ang), d_born) - c0) * 0.95  # setting off from just inside the edge
    n_runs = max(md, int(rng.integers(WALL_SAG[1][0], WALL_SAG[1][1] + 1)))
    r_ang = np.r_[d_ang, rng.uniform(0.2 * math.pi, 0.8 * math.pi, n_runs - md)]
    runs = tuple(zip(r_ang.tolist(), rng.uniform(*WALL_SAG[3], n_runs).tolist(),
                     (rng.uniform(*WALL_SAG[2], n_runs) * np.r_[np.full(md, 1.3), np.ones(n_runs - md)]).tolist()))
    ang = np.r_[ang, f_ang, d_ang]
    rb = np.r_[rb, t_r, R * rng.uniform(*WALL_DRIPS[1], md)]
    far = np.r_[far, t_far, np.zeros(md)]
    tau = np.r_[tau, np.full(nf, 1.7), np.ones(md)]
    born = np.r_[born, np.full(nf, kt), d_born]
    start = np.r_[start, t_start, d_start]
    ph = rng.uniform(0.0, 2.0 * math.pi, len(rb))
    gone = np.where(rb < WALL_SLIDE[0] * R, rng.uniform(*WALL_BEADS[4], len(rb)), 1e6)  # (the big ones stay)
    drip = np.r_[np.zeros(m + nf, bool), np.ones(md, bool)]
    vmax = np.where(drip, WALL_DRIPS[2] * rng.uniform(0.8, 1.0, len(rb)),
                    WALL_SLIDE[1] * np.clip(rb / (2.0 * WALL_SLIDE[0] * R), 0.0, 1.0)) * s / FPS
    vmax = np.where(drip | (rb >= WALL_SLIDE[0] * R), vmax, 0.0)
    T = np.where(drip, WALL_DRIPS[3], WALL_SLIDE[2]) * FPS
    wait = np.where(drip, 0.0, 2.0 * tau)  # a flung drop slides once it has come to rest
    out = np.stack([np.cos(ang), np.sin(ang)], 1)

    pace = s / FPS * (0.6 + 0.4 * size)
    g = G * s / FPS ** 2
    nsp = max(4, int(WALL_SPRAY[0] * (0.4 + 0.6 * size)))  # the spray, bursting out along the wall and off it
    a_sp = np.where(rng.random(nsp) < 0.5, fa + rng.normal(0.0, 0.9, nsp), rng.uniform(0.0, 2.0 * math.pi, nsp))
    d_sp = np.c_[np.cos(a_sp), np.sin(a_sp), rng.uniform(0.1, 0.6, nsp)]
    p_sp = np.c_[np.array([xh, yh]) + d_sp[:, :2] * R * rng.uniform(0.8, 1.15, (nsp, 1)), np.full(nsp, zh)]
    drops.add(kh, p_sp, d_sp * rng.uniform(*WALL_SPRAY[1], (nsp, 1)) * pace, R * _sizes(rng, nsp, *WALL_SPRAY[2], 1.5),
              foam=0.0, life=rng.uniform(3, SPRAY_LIFE, nsp), fade=SPRAY_LIFE / 2, g=g)
    nc = max(3, int(WALL_CROWN[0] * (0.4 + 0.6 * size)))  # and bigger drops off its rim, towards the camera
    a_c = rng.uniform(0.0, 2.0 * math.pi, nc)
    d_c = np.c_[np.cos(a_c), np.sin(a_c)]
    p_c = np.c_[np.array([xh, yh]) + d_c * R * s0 * rng.uniform(0.85, 1.05, (nc, 1)), np.full(nc, zh)]
    v_c = np.c_[d_c * rng.uniform(*WALL_CROWN[2], (nc, 1)), rng.uniform(*WALL_CROWN[3], nc)] * pace
    drops.add(kh, p_c, v_c, R * rng.uniform(*WALL_CROWN[1], nc), foam=0.05, life=rng.uniform(*WALL_CROWN[4], nc),
              fade=3.0, g=g)  # (falling on through the frames after, a shower of rain)
    falling = []  # drops letting go of its lower edge, one by one
    for _ in range(WALL_FALL[0]):
        kf = kh + int(rng.integers(*WALL_FALL[2]))
        if kf >= n:
            continue
        pf = edge(np.array([rng.uniform(0.3 * math.pi, 0.7 * math.pi)]), kf - kh)[0]
        qf = (cam[kf] @ to_room if cam is not None else np.eye(3)) @ np.array([pf[0], pf[1], 1.0])
        falling.append((kf, np.array([qf[0] / qf[2], qf[1] / qf[2], zh]), np.array([0.0, 0.3 * s / FPS, 0.1 * s / FPS]),
                        R * float(rng.uniform(*WALL_FALL[1])), g))

    look0, look1 = WALL_LOOK
    for k in range(kh, n):
        t = k - kh
        M = cam[k] @ to_room if cam is not None else np.eye(3)

        def at(p):  # room px (m, 2) -> this frame's
            h = np.c_[p, np.ones(len(p))] @ M.T
            return h[:, :2] / h[:, 2:3]

        c = at(centre(t)[None])[0]
        look = 1.0 - float(_ss(look0, look0 + look1, t))
        wet = 0.5 + 0.5 * float(_ss(0.0, WALL_WET[3], t))
        film = [Sheet(c[0], c[1], radius(t) * (1.0 + WALL_SOAK[0] * float(_ss(0.0, WALL_SOAK[1], t))),
                      tuple(zip(f_ang.tolist(), (f_len * grown(t)).tolist(), f_wid.tolist(), f_bead.tolist())), wob,
                      R * (WALL_THICK[0] * math.exp(-t / WALL_THICK[1]) + WALL_THICK[2]), math.exp(-t / WALL_THICK[1]),
                      look, wet, WALL_FROTH[0] * max(0.0, 1.0 - t / WALL_FROTH[1]) ** 2,
                      float(_ss(WALL_SAG[0], WALL_SAG[0] + WALL_SAG[5], t)), runs)]
        age = t - born
        on = age >= 0
        if on.any():
            e = np.exp(-np.maximum(age, 0.0) / tau)
            run = (far * (1.0 - e))[:, None] * out
            vrun = (far / tau * e)[:, None] * out
            fall, vfall = _slide(age - wait, vmax, T)
            stop = start + run
            p = stop + np.c_[np.zeros(len(fall)), fall]
            v = vrun + np.c_[np.zeros(len(fall)), vfall]
            a = 1.0 - _ss(gone, gone + 3.0, age)
            a = a * _ss(-1.0, 0.5, age)  # it gathers as it leaves the edge
            P1, P0 = at(p), at(stop)
            V = v @ M[:2, :2].T
            for j in np.flatnonzero(on & (a > 0.01)):
                if fall[j] > rb[j]:
                    film.append(Trail(P0[j, 0], P0[j, 1], P1[j, 0], P1[j, 1], rb[j]))
            for j in np.flatnonzero(on & (a > 0.01)):
                film.append(Bead(P1[j, 0], P1[j, 1], rb[j], V[j, 0], V[j, 1], float(a[j]), float(ph[j])))
        scenes[k].film = film
    return falling


def _film(P: np.ndarray, items: list, matte: np.ndarray | None, room: Room, k: int, noise: Noise) -> list[tuple]:
    """The water on the wall (Scene.film) on frame ``k``'s plate ``P`` (linear) in place, behind the water in the air;
    nothing on the hand, which is in front of the wall.

    The Sheet: the wet patch it leaves (the wall darker and a little more saturated, unevenly in the streaks the water
    ran out in, its outline the sheet's rounded and feathered, its lower edge sagging into runs where the film
    collects darker and glossier, a faint gloss on its side towards the key light) and, at its ``look``, the running
    water itself, shaded as water: its thickness (a dome at the hit flattening into a film, a rounded rim, its fingers
    rounded capsules, in uneven streaks) bends the wall seen through it and mirrors the room and the key light by its
    slopes; its edge is soft and broken, a bright line in lengths inside its body's edge where that faces the light and
    a dark line where it meets the wall on the other side; at the hit its middle is churned white, in streaks along its
    radius. Each Trail: a thin wet streak, a pixel wide at its top and as wide as its drop at its foot, glossy along its
    side towards the light. Each Bead: a drop lying flattened on the wall, uneven, sagging, drawn out behind into its
    trail as it slides: the wall seen through it a little magnified and darker, dark at its top and bright at its foot
    as a clear drop lit from above is, a thin bright crescent inside its rim towards the light, a point of glint, a soft
    shadow. Returns the boxes it changed."""
    H, W = P.shape[:2]
    boxes = []
    lit, white = room.lit(k), max(float(room.white[k]), 0.3)
    c = next(((it.x, it.y) for it in items if isinstance(it, Sheet)), (W / 2.0, H / 2.0))
    L3 = room.key(k, c)
    kx, ky = (float(v) for v in _unit(L3[:2]))

    def box(x, y, e):
        y0, y1 = max(0, int(y - e)), min(H, int(y + e) + 2)
        x0, x1 = max(0, int(x - e)), min(W, int(x + e) + 2)
        if y1 <= y0 or x1 <= x0:
            return None
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        free = 1.0 if matte is None else 1.0 - matte[y0:y1, x0:x1].astype(np.float32) / 255.0
        boxes.append((y0, y1, x0, x1))
        return (y0, y1, x0, x1), yy, xx, free

    def wetten(patch, m):
        lum = (patch @ fx._LUMA)[..., None]
        patch[:] = (lum + (patch - lum) * (1.0 + WALL_WET[1] * m[..., None])) * (1.0 - WALL_WET[0] * m[..., None])

    for it in items:
        if isinstance(it, Sheet):
            ext = max((ln + hw * bd for _, ln, hw, bd in it.fingers), default=0.0)
            sag = it.sag * max((ln for _, _, ln in it.runs), default=0.0)
            reach = (it.r * (1.0 + sum(abs(a) for _, a, _ in it.wobble) + ext + sag + 3.0 * WALL_ROUND[0])
                     + 3.0 * WALL_ROUND[1] + WALL_EDGE[0] + 6)
            b = box(it.x, it.y, reach)
            if b is None:
                continue
            (y0, y1, x0, x1), yy, xx, free = b
            dx, dy = xx - np.float32(it.x), yy - np.float32(it.y)
            din, re = _inside(dx, dy, it.r, it.fingers, it.wobble)
            rho = np.hypot(dx, dy)
            th_ = np.arctan2(dy, dx)
            u = np.clip(rho / np.maximum(re, 1.0), 0.0, 1.0)
            ragged = WALL_EDGE[0] * noise.flow(10, dx, dy, 0.0, 0.0, 0, WALL_EDGE[1] * it.r)  # (in the sheet's own
            # frame: its unevenness moves with it)
            edge = din + ragged
            m = (_ss(-WALL_EDGE[2], WALL_EDGE[2], edge) * free).astype(np.float32)
            # the water runs out in uneven streaks (a noise laid round it, long outwards and narrow across, stretched
            # as it spreads, a whole number of its tiles round so it has no seam), thicker along each finger, over its
            # outer part only (streaks meeting in its middle, it was a cut lemon)
            rays = WALL_RAYS[0] * noise.along(8, rho / it.r, th_ * it.r, 0, WALL_RAYS[2],
                                              4.0 * math.pi * NOISE_CELL * it.r / (NOISE_SIZE * WALL_RAYS[3]), 0.0)
            for ang, ln, hw, _ in it.fingers:
                rays = rays + WALL_RAYS[1] * min(1.0, ln / 0.2) * np.exp(-(np.angle(np.exp(1j * (th_ - ang)))
                                                                           / (0.6 * hw)) ** 2)
            rays = (rays * _ss(0.45, 0.9, u)).astype(np.float32)
            towards = (dx * kx + dy * ky) / np.maximum(rho, 1.0)  # its side towards the key light, -1..1
            patch = P[y0:y1, x0:x1]
            # the wet patch: the sheet's outline rounded (blurred and cut again), sagging into its runs, feathered
            wm = cv2.GaussianBlur(_ss(-1.0, 1.0, din).astype(np.float32), (0, 0), max(0.8, WALL_ROUND[0] * it.r))
            wm = _ss(0.3, 0.7, wm).astype(np.float32)
            collect = 0.0
            if it.sag > 0.0 and it.runs:
                fall = np.zeros(xx.shape[1], np.float32)
                for ang, wd, ln in it.runs:
                    xr = math.cos(ang) * float(_body(ang, it.r, it.wobble))
                    fall += np.float32(ln) * np.exp(-((dx[0] - xr) / (wd * it.r)) ** 2)
                down = _ss(-0.1, 0.6, dy / it.r).astype(np.float32)
                wm = cv2.remap(wm, xx - x0, (yy - y0 - np.float32(it.sag * it.r) * fall[None, :] * down).astype(
                    np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                collect = WALL_SAG[4] * it.sag * _ss(0.3, 1.1, dy / it.r)
            wm = cv2.GaussianBlur(wm, (0, 0), WALL_ROUND[1]) * free
            soft = 1.0 + WALL_WET[4] * _ss(0.6, 1.0, u)
            wetten(patch, it.wet * wm * np.clip(1.0 + 0.2 * rays, 0.7, 1.3) * soft * (1.0 + collect))
            patch += (WALL_WET[2] * it.wet * wm * (_ss(-0.6, 1.0, towards) + collect))[..., None] * lit
            if it.look > 0.01:  # the running water
                rw = WALL_RIM[0] * (1.0 + 0.6 * it.dome)  # (thick at the hit, where the ball's skin lands)
                e = np.clip(edge / rw, 0.0, 1.0)
                prof = np.sqrt(e * (2.0 - e))  # its edge rounded as it meets the wall
                inner = 1.6 * it.dome * (1.0 - u * u) ** 0.6 + 0.6 * (1.0 - it.dome)
                ridge = WALL_RIM[1] * rw * np.exp(-((edge - 0.8 * rw) / (0.7 * rw)) ** 2)
                th = prof * (it.thick * inner + ridge) * np.clip(1.0 + rays, 0.3, 2.2)
                # the light a body of water lying on the wall bends onto it: gathered at its foot, opposite the light,
                # its top in its shade, as much as it bulges (as a drop on the wall is)
                patch *= (1.0 + it.dome * it.look * m * (WALL_DOME[0] * np.exp(-((u - 0.62) / 0.22) ** 2)
                                                          * _ss(0.0, 0.9, -towards)
                                                          - WALL_DOME[1] * _ss(0.0, 0.9, towards) * u))[..., None]
                gy, gx = np.gradient(cv2.GaussianBlur(th.astype(np.float32), (0, 0), 1.0))
                nd = np.sqrt(gx * gx + gy * gy + 1.0)
                nx, ny, nz = -gx / nd, -gy / nd, 1.0 / nd
                bend = WALL_BEND * it.look
                seen = cv2.remap(np.ascontiguousarray(patch), (xx - x0 - bend * gx).astype(np.float32),
                                 (yy - y0 - bend * gy).astype(np.float32), cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
                F = np.minimum(F0 + (1.0 - F0) * (1.0 - nz) ** 5, 0.7).astype(np.float32)[..., None]
                env = room.reflect(k, nx.ravel(), ny.ravel()).reshape(seen.shape)
                spec = WALL_GLINT * _window(nx, ny, nz, L3, 1.0)
                spec += GLINT_CORE[0] / white * _window(nx, ny, nz, L3, GLINT_CORE[1])
                brk = np.clip(0.8 + 0.7 * noise.flow(11, dx, dy, 0.0, 0.0, 0, 0.1 * it.r), 0.0, 1.5)
                body = re - rho + ragged  # px inside its body's edge, its fingers aside
                lengths = _ss(-0.1, 0.4, noise.flow(12, dx, dy, 0.0, 0.0, 0, WALL_RIM[5]))
                line = (WALL_RIM[2] * np.exp(-((body - 0.45 * rw) / 0.9) ** 2) * _ss(*WALL_RIM[4], towards) * lengths
                        * _ss(2.0, 0.0, din - body + ragged))
                contact = WALL_RIM[3] * np.exp(-((edge - 0.8) / 0.9) ** 2) * _ss(-0.1, 0.7, -towards) * brk
                col = (seen * (1.0 - F) + env * F) * (1.0 - contact)[..., None] + (spec + line)[..., None] * lit
                if it.froth > 0.01:  # churned white at the hit, in fine bubbles in its middle, torn out in fine
                    # streaks along its radius further out (in soft clumps, camouflage; in streaks meeting in its
                    # middle, a starburst)
                    fine = noise.flow(5, dx, dy, 0.0, 0.0, 0, WALL_FROTH[2] * it.r)
                    streak = noise.along(9, rho / it.r, th_ * it.r, 0, WALL_FROTH[3],
                                         4.0 * math.pi * NOISE_CELL * it.r / (NOISE_SIZE * WALL_FROTH[4]), 0.0)
                    out_ = _ss(0.3, 0.55, u)
                    shape_ = ((1.0 - out_) * 0.35 * (0.6 + 0.4 * fine)
                              + out_ * _ss(-0.2, 0.8, streak) * (0.6 + 0.4 * fine))
                    fm = np.clip(it.froth * _ss(1.0, 0.4, u) * shape_, 0.0, FOAM_MAX)
                    col = col * (1.0 - fm)[..., None] + (fm * FOAM_ALBEDO * (0.8 + 0.2 * nz))[..., None] * lit
                ml = (m * it.look)[..., None]
                patch[:] = patch * (1.0 - ml) + col * ml
        elif isinstance(it, Trail):
            L = math.hypot(it.x1 - it.x0, it.y1 - it.y0)
            if L < 1.0:
                continue
            b = box(0.5 * (it.x0 + it.x1), 0.5 * (it.y0 + it.y1), 0.5 * L + it.r + 3)
            if b is None:
                continue
            (y0, y1, x0, x1), yy, xx, free = b
            ux, uy = (it.x1 - it.x0) / L, (it.y1 - it.y0) / L
            a = np.clip(((xx - it.x0) * ux + (yy - it.y0) * uy) / L, 0.0, 1.0)  # 0 at its top, 1 at the drop
            across = (xx - (it.x0 + a * L * ux)) * -uy + (yy - (it.y0 + a * L * uy)) * ux  # signed px across it
            hw = WALL_TRAIL[1] + (WALL_TRAIL[2] * it.r - WALL_TRAIL[1]) * a ** 0.8
            m = (_ss(0.8, -0.8, np.abs(across) - hw) * free).astype(np.float32)
            wetten(P[y0:y1, x0:x1], WALL_TRAIL[0] * m * (0.5 + 0.5 * a))
            side = across / np.maximum(hw, 0.5) * math.copysign(1.0, -uy * kx + ux * ky + 1e-9)
            gloss = WALL_TRAIL[3] * np.exp(-((side - 0.5) / 0.3) ** 2) * (0.3 + 0.7 * a) * m
            P[y0:y1, x0:x1] += gloss[..., None] * lit
        elif isinstance(it, Bead):
            if it.a <= 0.01 or it.r < 0.6:
                continue
            sp = math.hypot(it.vx, it.vy)
            ux, uy = (it.vx / sp, it.vy / sp) if sp > 1e-3 else (0.0, 1.0)  # lying still, it sags downwards
            slide = float(_ss(0.1, 1.0, sp))
            ka = BEAD_SHAPE[0] + min(sp * SHUTTER / it.r, 2.0)  # sagging, and drawn out over the exposure
            kb = ka + slide * BEAD_SHAPE[2]  # behind its middle, drawn out into its trail as it slides
            b = box(it.x, it.y, kb * it.r + BEAD_SHADOW[0] * it.r + 4)
            if b is None:
                continue
            (y0, y1, x0, x1), yy, xx, free = b
            dx, dy = xx - np.float32(it.x), yy - np.float32(it.y)

            def shape(dx, dy):  # 0 at its middle, 1 on its outline
                a = (dx * ux + dy * uy) / it.r
                cr = (-dx * uy + dy * ux) / it.r
                back = a < 0.0
                al = a / np.where(back, kb, ka)
                w = 1.0 - slide * (1.0 - BEAD_SHAPE[3]) * np.where(back, _ss(0.0, 1.0, -al), 0.0)
                th = np.arctan2(cr, a)
                lobes = 1.0 + BEAD_SHAPE[1] * (np.cos(2 * th + it.ph) + 0.7 * np.cos(3 * th + 2.1 * it.ph)
                                               + 0.4 * np.cos(5 * th + 3.7 * it.ph))
                return np.sqrt(al * al + (cr / w) ** 2) / lobes

            rho = shape(dx, dy)
            hyp = np.maximum(np.hypot(dx, dy), 1e-3)
            towards = (dx * kx + dy * ky) / hyp
            m = (_ss(1.0 + 1.0 / it.r, 1.0 - 1.0 / it.r, rho) * it.a * free).astype(np.float32)
            patch = P[y0:y1, x0:x1]
            off = BEAD_SHADOW[0] * it.r  # its soft shadow on the wall, away from the light, under it
            shadow = BEAD_SHADOW[1] * _ss(1.4, 0.3, shape(dx + off * kx, dy + off * ky)) * it.a * free
            patch *= (1.0 - shadow * (1.0 - m))[..., None]
            mag = BEAD_LOOK[0]
            seen = cv2.remap(np.ascontiguousarray(patch), (it.x - x0 + dx / mag).astype(np.float32),
                             (it.y - y0 + dy / mag).astype(np.float32), cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
            shade_ = ((1.0 - BEAD_LOOK[1]) * (1.0 + BEAD_LOOK[2] * np.exp(-((rho - 0.55) / 0.25) ** 2)
                                               * _ss(0.0, 0.9, -towards))
                      * (1.0 - BEAD_LOOK[3] * _ss(0.3, 1.0, rho) * _ss(0.0, 0.9, towards)))
            patch[:] = patch * (1.0 - m[..., None]) + (seen * shade_[..., None]) * m[..., None]
            px = np.float32(1.0 / it.r)
            lines = float(_ss(2.0, 4.0, it.r))  # (a drop a few px across drawn with it, a ring: a bubble)
            line = (lines * BEAD_LOOK[6] * np.exp(-((rho - (1.0 - 1.4 * px)) / px) ** 2)
                    * _ss(BEAD_LOOK[7], 1.0, towards))
            gx_, gy_ = it.x + 0.45 * it.r * kx, it.y + 0.45 * it.r * ky
            sg = BEAD_LOOK[5]
            pin = BEAD_LOOK[4] / white * np.exp(-((xx - gx_) ** 2 + (yy - gy_) ** 2) / (2.0 * sg * sg))
            patch += ((line * m + pin * it.a * free) / (1.0 + min(sp * SHUTTER / it.r, 2.0)))[..., None] * lit
    return boxes


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
    bub_r = _sizes(rng, BUBBLES, *BUBBLE_R, 1.5)
    rise = BUBBLE_RISE * rng.uniform(0.5, 1.5, BUBBLES) * (bub_r / BUBBLE_R[1]) ** 0.5  # the bigger, the faster
    bub_spin = np.stack([rng.uniform(*BUBBLE_DRIFT[:2], BUBBLES), rng.uniform(0.0, 2.0 * math.pi, BUBBLES),
                         rng.uniform(0.4, 1.1, BUBBLES)], 1)  # each one's share of the circulation, and its wander
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
    drip_side = float(rng.uniform(*DRIP_SIDE) * rng.choice([-1.0, 1.0]))
    fly = None  # the orb in the air: (frame, centre, velocity, radius, thrown, the palm's normal, the wall's depth);
    # centre None: burst
    falling: list[tuple] = []  # drops to let go off the wall later (_wall)
    landed: list[tuple] = []  # drips landed on the palm: (frame, px off the palm's point, radius)
    for k in range(g0, n):
        sc = scenes[k]
        p, nrm, s = hand.p[k], hand.nrm[k], float(hand.s[k])
        R = ORB_R * s
        for kf, xf, vf, rf, gf in falling:
            if kf == k:
                drops.add(k, xf, vf, rf, foam=0.05, g=gf)
        for kl, off, rd in landed:  # a wet dot where a drip landed, carried on the palm, drying
            if kl <= k < kl + DRIP_WET[1]:  # (round on the palm, seen squashed as it turns from the camera)
                dry = 1.0 - (k - kl) / DRIP_WET[1]
                rw = DRIP_WET[0] * rd
                across = math.atan2(float(nrm[1]), float(nrm[0])) + 0.5 * math.pi
                sc.wet.append((p[0] + off[0], p[1] + off[1], rw, rw * max(abs(float(nrm[2])), 0.5), across,
                               DRIP_WET[2] * dry, float(kl), dry, DRIP_WET[5]))
            if k == kl:  # it splashes as it lands: a few tiny drops jump off the palm
                m = int(DRIP_WET[3])
                at = np.array([p[0] + off[0], p[1] + off[1]])
                zl = float(hand.depth(k, np.float32([at[0]]), np.float32([at[1]]))[0])
                d = _unit(rng.normal(size=(m, 3)) + 1.5 * nrm)
                drops.add(k, np.tile(np.r_[at, zl], (m, 1)), d * rng.uniform(*DRIP_WET[4], (m, 1)) * s / FPS,
                          rd * rng.uniform(0.25, 0.4, m), life=4.0, fade=2.0)
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
                sc.held = (c[0], c[1], c[2], r, vol ** GATHER_LIGHT)  # its shade and its focus come on as it
                # gathers (at full strength once a third of it was there, it popped on with only a few drops in the
                # air; at its share of the water, the focus was white hot while the orb was still forming)
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
                sc.bubbles += _air(bub, bub_r, rise, bub_spin, axis, circ, t, c, r, s, 1.0)
                if k >= next_drip:
                    u = (k - next_drip) / DRIP_FORM
                    across = np.cross(nrm, [0.0, 0.0, 1.0])
                    across = across / np.linalg.norm(across) if np.linalg.norm(across) > 1e-3 else np.array([1.0, 0, 0])
                    down = _unit(-nrm + drip_side * across)
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
                        x0 = c + down * (r + 2.0 * rd)
                        v0 = v_c + down * 0.4 * s / FPS
                        drops.add(k, x0, v0, rd, life=life, fade=2)
                        land = x0[:2] + v0[:2] * life + np.array([0.0, 0.5 * G * s / FPS ** 2 * life * life])
                        landed.append((k + int(round(life)), land - hand.p[min(n - 1, k + int(round(life)))], rd))
                        next_drip = k + DRIP_FORM + int(rng.integers(*DRIP_EVERY))
                        drip_side = float(rng.uniform(*DRIP_SIDE) * rng.choice([-1.0, 1.0]))
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
                if hit:  # on the wall: it spreads over it the way it was thrown and stays there, draining, dripping
                    # (flung off it as pieces in the air, it was soap bubbles and shards that all went at once)
                    fan = v[:2] / max(float(np.hypot(*v[:2])), 1e-6)
                    falling += _wall(scenes, drops, (k, xy[0], xy[1], c[2], R0 * kz, s * kz, fan, plan.size), n, cam,
                                     rng)
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
                    sc.bubbles += _air(bub, bub_r, rise, bub_spin, axis, circ, t, xy, R0 * kz, s * kz, fade)
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
            arc: np.ndarray, flat: np.ndarray) -> list[Blob]:
    """Blobs along a stream through ``pts`` (m, 3) moving ``vel`` (px a frame), its radius ``rad``, ``arc`` px along
    it on the picture, ``flat`` its thickness against a tube's (Blob.flat): at most WHIP_SPACING of their radius
    apart, each weighted by its share of the stream (_tube_weight), so the stream is a tube of that radius, blurred
    by its motion over the shutter."""
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
                        w=w, move=move, stream=(float(arc[i]), float(t[0]), float(t[1])), flat=float(flat[i])))
    return out


def _off_hand(hand: Hand, k: int, xy: np.ndarray) -> np.ndarray:
    """(m,) 0..1: how far off the hand points ``xy`` (m, 2) px are on frame ``k``, as the picture shows it (OFF_HAND):
    by how far outside the outline round its landmarks they are (by the nearest landmark alone, the middle of the palm
    counted as half off it), or with none by their distance from its palm's point."""
    s = float(hand.s[k])
    xy = np.asarray(xy, np.float64)
    if hand.off is None:
        return _ss(OFF_HAND[1][0] * s, OFF_HAND[1][1] * s, np.hypot(*(xy - hand.p[k]).T))
    hull = cv2.convexHull(hand.off[k][:, :2].astype(np.float32))
    out = np.array([-cv2.pointPolygonTest(hull, (float(a), float(b)), True) for a, b in xy])
    return _ss(OFF_HAND[0][0] * s, OFF_HAND[0][1] * s, out)


def _pool(hand: Hand, k: int, r: float, t: float, out: np.ndarray, parts: np.ndarray) -> tuple[list, tuple]:
    """The pool of radius ``r`` on the palm the stream rises out of on frame ``k`` (``t`` seconds in), flat on it (seen
    squashed as the palm turns from the camera), drawn out a little along ``out`` (the unit direction the stream leaves
    it in, in the picture) and lying more that way, so the stream widens into it rather than ending in it; an
    irregular puddle, flat pools run into it (``parts``: (angle, distance and size of its radius, turns a second) each,
    WHIP_POOL_PARTS), its outline rippling and churning: (its blobs, the wet patch round it on the palm: Scene.wet)."""
    p, nrm = hand.p[k], hand.nrm[k]
    zp = float(hand.depth(k, np.float32([p[0]]), np.float32([p[1]]))[0])
    c = np.array([p[0], p[1], zp]) + nrm * 0.5 * WHIP_POOL[1] * r
    c[:2] += out * WHIP_POOL_ALONG[1] * r
    nxy = float(np.hypot(*nrm[:2]))
    un = nrm[:2] / nxy if nxy > 1e-6 else np.array([0.0, 1.0])
    squash = math.sqrt(nrm[2] ** 2 + WHIP_POOL[1] ** 2 * (1.0 - nrm[2] ** 2))  # a disc on the palm, seen from the camera
    disc = np.asarray(_shape(un, squash, 1.0)).reshape(2, 2)
    ka = WHIP_POOL_ALONG[0]
    shape = disc @ np.asarray(_shape(out, ka, 1.0 / ka)).reshape(2, 2)
    a = WHIP_POOL[3]
    modes = ((3, a * math.sin(2 * math.pi * 2.3 * t), 1.7 * t),
             (5, 0.6 * a * math.sin(2 * math.pi * 3.7 * t + 1.0), -2.3 * t),
             (7, 0.4 * a * math.sin(2 * math.pi * 5.1 * t), 3.1 * t))
    blobs = [Blob(c[0], c[1], c[2], r, tuple(shape.ravel()), spin=CIRCULATE[0] / FPS, foam=WHIP_POOL[2], modes=modes,
                  flat=WHIP_POOL[1])]
    for ang, dist, size, turns in parts:  # flat pools run into it (on the palm: squashed with it)
        th = ang + 2.0 * math.pi * turns * t
        off = disc @ np.array([math.cos(th), math.sin(th)]) * dist * r
        blobs.append(Blob(c[0] + off[0], c[1] + off[1], c[2], size * r, tuple(disc.ravel()), spin=CIRCULATE[0] / FPS,
                          foam=0.5 * WHIP_POOL[2], flat=WHIP_POOL[1],
                          modes=((2, 0.5 * a * math.sin(2 * math.pi * 1.3 * t + ang), ang - t),)))
    wet = (c[0], c[1], WHIP_WET * r, WHIP_WET * r * squash, math.atan2(un[1], un[0]) + 0.5 * math.pi, 0.08, 0.0, 0.0,
           0.3)
    return blobs, wet


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
    dripping = Drops(cam, size, float(np.median(hand.s)))  # its drips, falling free of it
    trail: list[tuple] = []  # where it has lain on the palm: (frame, (m, 2) px off the palm's point, (m,) radii px)
    N = WHIP_NODES
    u_n = np.arange(N) / (N - 1)
    sub = 4
    dt = 1.0 / FPS / sub
    x = v = root0 = None
    phase = rng.uniform(0.0, 2.0 * math.pi)
    air = np.stack([rng.uniform(0.0, 1.0, WHIP_AIR), rng.uniform(-0.45, 0.45, WHIP_AIR),
                    rng.uniform(*WHIP_AIR_R, WHIP_AIR)], 1)  # its air: where along it, across it, how big (of it)
    cell = WHIP_SWELL[1] / 4.0  # its bulges: a smooth random profile along the water flowing out of the palm, in
    m_sw = 4096  # samples this many hand sizes apart, its features 4 samples (sigma), wrapping round
    swell = np.fft.irfft(np.fft.rfft(rng.standard_normal(m_sw)) * np.exp(-2.0 * (math.pi * 4.0 * np.fft.rfftfreq(m_sw))
                                                                           ** 2), m_sw)
    swell = np.r_[swell, swell[:1]] / max(float(swell.std()), 1e-9)
    width = np.fft.irfft(np.fft.rfft(rng.standard_normal(m_sw)) * np.exp(-2.0 * (math.pi * 4.0 * np.fft.rfftfreq(m_sw))
                                                                           ** 2), m_sw)  # its width, likewise
    width = np.r_[width, width[:1]] / max(float(width.std()), 1e-9)
    drip_at, drip_u = None, 0.0  # a drip swelling at the lowest point of the stream where it hangs off the hand: the
    # frame it started on and where along the stream it is
    drip_every = int(rng.integers(*WHIP_DRIP[0]))
    prng = _rng(seed, _WHIP, 1)
    m_p, (d0, d1), (r0, r1), spin = WHIP_POOL_PARTS
    parts = np.stack([prng.uniform(0.0, 2.0 * math.pi, m_p), prng.uniform(d0, d1, m_p), prng.uniform(r0, r1, m_p),
                      prng.uniform(-spin, spin, m_p)], 1)  # the flat pools its puddle is made of
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
            lead = x[min(3, len(x) - 1), :2] - x[0, :2] if x is not None and len(x) > 1 else hand.along[k, :2]
            blobs, wet = _pool(hand, k, pool_r, t, _unit(np.asarray(lead, np.float64) + 1e-9), parts)
            sc.blobs += blobs
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
        body = None  # where the stream is on this frame (px), for the drops it sheds
        if broke <= WHIP_BREAK and len(x):
            om = 2.0 * math.pi * (WHIP_HZ[0] + (WHIP_HZ[1] - WHIP_HZ[0]) * u_n ** 0.7) * (1.0 + (WHIP_CRACK[3] - 1.0)
                                                                                         * crack)
            off = _off_hand(hand, k, x[:, :2])  # it sags more where it hangs off the hand (held up as stiffly there,
            sag = (WHIP_SAG[0] + (WHIP_SAG[1] - WHIP_SAG[0]) * off) * G * s  # it was a hose)
            for j in range(sub):
                if alive:
                    a = om[:, None] ** 2 * (target - x) - 2.0 * WHIP_DAMP * om[:, None] * v
                    a[:, 1] += sag
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
            bulge = np.interp(((arc / s - WHIP_FLOW * t) / cell) % m_sw, np.arange(m_sw + 1), swell)
            rad *= 1.0 + np.clip(WHIP_SWELL[0] * bulge, -WHIP_SWELL[2], WHIP_SWELL[2]) * _ss(0.0, 0.15, u)  # (formed
            # as it flows out: swelling where it rises out of the pool, it was a bulb on it)
            wide = np.interp(((arc / s - WHIP_WIDTH[2] * WHIP_FLOW * t) / (WHIP_WIDTH[1] / 4.0)) % m_sw,
                             np.arange(m_sw + 1), width)
            rad *= 1.0 + np.clip(WHIP_WIDTH[0] * wide, -WHIP_WIDTH[3], WHIP_WIDTH[3]) * _ss(0.0, 0.1, u)  # (even all
            # along, a tube)
            offp = _off_hand(hand, k, pts[:, :2])
            if alive:  # hanging slack off the hand it thins, its water pouring down out of it (flung out, it does not)
                rad *= 1.0 - (1.0 - WHIP_HANG) * offp * (1.0 - _ss(0.15, 0.4, speed / WHIP_SHED)) * (1.0 - crack)
            if alive and crack == 0.0:  # hanging off the hand, it swells into a drip at its lowest point, necks and
                # lets go of it, now and then
                hang = np.flatnonzero((offp > 0.5) & (u > 0.3) & (u < 0.95) & (speed < 0.5 * WHIP_SHED))
                if drip_at is None and hang.size:
                    drip_at, drip_u = k, float(u[hang[np.argmax(pts[hang, 1])]])
                if drip_at is not None:
                    ph_ = (k - drip_at) / drip_every
                    i = int(np.argmin(np.abs(u - drip_u)))
                    if ph_ >= 1.0:  # (its drips fall on out of the picture: culled with what it sheds, they hung)
                        down = np.array([0.0, 1.0, 0.0])
                        dripping.add(k, pts[i] + down * rad[i], vel[i] + down * 0.3 * s / FPS, WHIP_DRIP[1] * rad[i])
                        drip_at, drip_every = None, int(rng.integers(*WHIP_DRIP[0]))
                    else:
                        da = (arc - arc[i]) / np.maximum(1.5 * rad[i], 1.0)
                        rad *= 1.0 + (WHIP_DRIP[1] - 1.0) * ph_ * np.exp(-da * da)  # it swells there...
                        rad *= 1.0 - 0.5 * ph_ ** 2 * np.exp(-((da + 1.6) / 0.6) ** 2)  # ... necking just above it
            else:
                drip_at = None
            if alive:  # fast, it sheds drops off its leading side, and its tail sprays
                fast = np.flatnonzero((speed > WHIP_SHED) & (u > WHIP_SHED_FROM))
                go = fast[rng.random(fast.size) < WHIP_SHED_RATE * (speed[fast] / WHIP_SHED - 1.0)]
                if go.size:
                    lead = _unit(vel[go] + 1e-9)
                    drops.add(k, pts[go] + lead * rad[go, None], 0.85 * vel[go] + rng.normal(size=(go.size, 3)) * 0.05
                              * s, s * _sizes(rng, go.size, 0.005, 0.02, 1.5), foam=0.1)
                m = int(rng.integers(WHIP_SPRAY[0], WHIP_SPRAY[1])) if speed[-1] > 0.5 * WHIP_SHED else 0
                if m:
                    drops.add(k, np.repeat(pts[-1:], m, 0), 0.8 * vel[-1] + rng.normal(size=(m, 3)) * 0.05 * s,
                              s * _sizes(rng, m, 0.004, 0.012, 1.5), foam=0.0, life=rng.uniform(3, 6, m), fade=2)
                fastp = np.flatnonzero((speed > WHIP_SHED) & (u > 0.4))
                m = int(round(WHIP_SPRAY[2] * crack)) if fastp.size else 0
                if m:  # lashing out, its front tears into threads and drops, flung on ahead of it and away from it
                    i = rng.choice(fastp, m)
                    lead = _unit(vel[i] + 1e-9)
                    drops.add(k, pts[i] + lead * rad[i, None], vel[i] * rng.uniform(*WHIP_TORN[1], (m, 1))
                              + lead * rng.uniform(1.0, 3.0, (m, 1)) * s / FPS + rng.normal(size=(m, 3)) * 0.05 * s,
                              s * _sizes(rng, m, *WHIP_TORN[0], 1.5), foam=0.0, life=rng.uniform(3, 7, m), fade=2,
                              stretch=np.where(rng.random(m) < WHIP_THREADS, rng.uniform(2.5, 5.0, m), 1.0))
            lam = np.maximum(WHIP_BEAD[0] * rad, 1.0)  # the beads it necks into, this far apart, flowing out along it
            ph = np.r_[0.0, np.cumsum(np.diff(arc) / lam[1:])] - WHIP_FLOW * s * t / float(np.mean(lam))
            neck = np.maximum(WHIP_FRAY[1] * _ss(WHIP_FRAY[0], 1.0, u) * (1.0 - _ss(0.4, 1.0, speed / WHIP_SHED)),
                              broke / WHIP_BREAK if not alive else 0.0)  # (lashing, its tail is drawn out whole: its
            # beads, each smeared across the frame, were stacked like a staircase)
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
                # hair); it rises out of the pool as flat as the pool is, widening into it (as round as a tube there,
                # it ended in a bulb: a tadpole's head)
                root_flat = 1.0 - (1.0 - WHIP_POOL[1]) * (1.0 - _ss(0.0, WHIP_FLARE[1] * 1.5, u))
                sc.blobs += _stream(pts, vel, rad, foam, fl, arc2, root_flat)
                if alive:
                    body = pts[::3, :2]
                if alive and (k - plan.start) % WHIP_TRAIL[2] == 0:  # where it lies on the palm, the skin gets wet
                    at = np.flatnonzero(np.diff(np.floor(arc / (WHIP_TRAIL[3] * s))) > 0) + 1
                    zh = hand.depth(k, pts[at, 0].astype(np.float32), pts[at, 1].astype(np.float32))
                    lie = at[(offp[at] < 0.3) & (pts[at, 2] - zh < REST[1] * rad[at]) & (u[at] > 0.05)]
                    if lie.size:
                        trail.append((k, pts[lie, :2] - p, WHIP_TRAIL[4] * rad[lie] + WHIP_TRAIL[5] * s))
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
        flying = drops.blobs(k)
        if body is not None and flying:  # while it pours, what it sheds is gone by the time it is far from it (flying
            # on across the frame, a shower of rain)
            xy = np.array([[b_.x, b_.y] for b_ in flying])
            far = np.sqrt(((xy[:, None, :] - body[None]) ** 2).sum(-1)).min(1)
            keep = 1.0 - _ss(WHIP_CULL[0] * s, WHIP_CULL[1] * s, far)
            flying = [b_._replace(r=b_.r * float(q_)) for b_, q_ in zip(flying, keep) if b_.r * q_ >= 0.5]
        sc.blobs += flying + dripping.blobs(k)
        drops.step()
        dripping.step()
        trail = [tr for tr in trail if k - tr[0] < WHIP_TRAIL[1]]
        for kt, off, rw in trail:  # the skin it has lain on, wet round it, drying
            sc.band.append((p + off, rw, WHIP_TRAIL[0] * (1.0 - (k - kt) / WHIP_TRAIL[1])))
        if not len(x) and not len(drops.x) and not len(dripping.x) and pool_r <= 0.5 and not trail:
            break
    return scenes


# ---- the shot -------------------------------------------------------------------------------------------------------

GRAIN_UNIT = 0.943  # what fx_light's grain reading reads off unit white noise: the plate's grain is laid at its strength


def frame(k: int, plate_k: np.ndarray, matte: np.ndarray | None, sc: Scene, hand: Hand, room: Room, noise: Noise,
          code: float, seed: int) -> np.ndarray:
    """Frame ``k`` with its water: uint8 RGB, the plate's own bytes wherever there is none."""
    out = plate_k.copy()
    if not sc.blobs and not sc.wet and not sc.band and not sc.film:
        return out
    H, W = plate_k.shape[:2]
    P = lin(plate_k)
    boxes = (_wet(P, sc.wet, matte, room.lit(k), _unit(room.key(k, (sc.wet or [(W / 2.0, H / 2.0)])[0][:2])[:2]),
                  sc.band) if sc.wet or sc.band else [])
    if sc.film:
        boxes += _film(P, sc.film, matte, room, k, noise)
    fld = Field(sc.blobs, H, W) if sc.blobs else None
    layer = None if fld is None or fld.box is None else shade(fld, sc, k, hand, room, noise, _pyramid(P), matte,
                                                              code, seed)
    if layer is not None:
        boxes.append(layer.box)
        if matte is not None:
            boxes.append(caustic(P, layer, k, hand, room, matte, sc.held, sc.lobes, noise))
        y0, y1, x0, x1 = layer.box
        if layer.sheet is not None:  # the plate seen through the sheet a lash sweeps, shifted and darker
            share, sx, sy = (np.ascontiguousarray(layer.sheet[..., i]) for i in range(3))
            yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
            seen = cv2.remap(np.ascontiguousarray(P[y0:y1, x0:x1]), xx + sx, yy + sy, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
            P[y0:y1, x0:x1] = seen * (1.0 - SHEET[0] * share)[..., None]
        P[y0:y1, x0:x1] = P[y0:y1, x0:x1] * (1.0 - layer.alpha[..., None]) + layer.colour
        if sc.bubbles:
            bubbles(P, sc.bubbles, layer, room.lit(k), room.key(k, (sc.held or (W / 2, H / 2))[:2]))
            bx = np.array([[b[0], b[1], b[2]] for b in sc.bubbles])
            rr = int(2 * bx[:, 2].max() + 4)
            boxes.append((max(0, int(bx[:, 1].min()) - rr), min(H, int(bx[:, 1].max()) + rr + 1),
                          max(0, int(bx[:, 0].min()) - rr), min(W, int(bx[:, 0].max()) + rr + 1)))
    changed = np.zeros((H, W), bool)  # each changed px brought back once, however many boxes overlap on it
    for y0, y1, x0, x1 in boxes:
        if y1 > y0 and x1 > x0:
            changed[y0:y1, x0:x1] = True
    rows = np.flatnonzero(changed.any(1))
    if rows.size:
        span = slice(rows[0], rows[-1] + 1)
        sel = changed[span]
        out[span][sel] = gam(P[span][sel])
    return out


def _wet(P: np.ndarray, patches: list, matte: np.ndarray | None, lit: np.ndarray, key: np.ndarray,
         bands: list = ()) -> list[tuple]:
    """Wet skin on the hand, in place: patches (Scene.wet), each a soft ellipse, uneven unless ``soft`` (the share of
    its radius its edge feathers over, and how round it is), and ``bands`` (Scene.band), each a feathered band through
    its points; where the skin (by its matte) is darker and a little more saturated, as wet skin is, with a faint gloss
    towards a patch's middle and, at its ``pin``, a point of glint on its side towards the key light (``key``, a unit
    direction in the picture). What overlaps is one wet patch, darker again round its outline only (each darkened in
    turn and ringed at its own rim, a band of patches was a string of beads). Returns the boxes it changed."""
    H, W = P.shape[:2]
    if matte is None or not (patches or bands):
        return []
    lx, ly = (float(v) for v in key)
    parts = []
    for x, y, rx, ry, ang, strength, lob, pin, soft in patches:
        e = 1.3 * max(rx, ry) + 2
        b = max(0, int(y - e)), min(H, int(y + e) + 1), max(0, int(x - e)), min(W, int(x + e) + 1)
        if b[1] > b[0] and b[3] > b[2] and min(rx, ry) >= 0.5 and strength > 0:
            parts.append((b, (x, y, rx, ry, ang, strength, lob, pin, soft)))
    lines = []
    for pts, rw, strength in bands:
        e = 1.5 * float(np.max(rw)) + 2
        b = (max(0, int(pts[:, 1].min() - e)), min(H, int(pts[:, 1].max() + e) + 1), max(0, int(pts[:, 0].min() - e)),
             min(W, int(pts[:, 0].max() + e) + 1))
        if b[1] > b[0] and b[3] > b[2] and strength > 0:
            lines.append((b, (pts, rw, strength)))
    if not parts and not lines:
        return []
    every = parts + lines
    Y0, Y1 = min(b[0] for b, _ in every), max(b[1] for b, _ in every)
    X0, X1 = min(b[2] for b, _ in every), max(b[3] for b, _ in every)
    dark, sat, seen, gloss = (np.zeros((Y1 - Y0, X1 - X0), np.float32) for _ in range(4))
    if lines:  # drawn as discs on a grid of q px, faintest first, then feathered
        q = 2
        mask = np.zeros((-(-(Y1 - Y0) // q), -(-(X1 - X0) // q)), np.float32)
        for _, (pts, rw, strength) in sorted(lines, key=lambda it: it[1][2]):
            for (x, y), r in zip(pts, rw):
                cv2.circle(mask, (int(round((x - X0) / q)), int(round((y - Y0) / q))), max(1, int(round(r / q))),
                           float(strength), -1)
        rm = float(np.median(np.concatenate([np.atleast_1d(rw) for _, (_, rw, _) in lines])))
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, WHIP_TRAIL[6] * rm / (2.0 * q)))
        band = cv2.resize(mask, (mask.shape[1] * q, mask.shape[0] * q), interpolation=cv2.INTER_LINEAR)[:Y1 - Y0,
                                                                                                         :X1 - X0]
        dark = np.maximum(dark, band)
        sat = np.maximum(sat, np.minimum(band / SKIN_WET[4], 1.0))
        seen += np.minimum(band / max(WHIP_TRAIL[0], 1e-6) * 2.0, 1.0)
    for (y0, y1, x0, x1), (x, y, rx, ry, ang, strength, lob, pin, soft) in parts:
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        ca, sa = math.cos(ang), math.sin(ang)
        u = ((xx - x) * ca + (yy - y) * sa) / np.float32(rx)
        v = (-(xx - x) * sa + (yy - y) * ca) / np.float32(ry)
        th = np.arctan2(v, u)
        uneven = 1.0 - soft
        rr = np.sqrt(u * u + v * v) / (1.0 + uneven * (0.14 * np.cos(3 * th + lob) + 0.09 * np.cos(5 * th + 1.7 * lob)
                                                       + 0.06 * np.cos(8 * th + 2.9 * lob)))
        m = _ss(1.0, 1.0 - soft, rr).astype(np.float32)
        on = min(1.0, strength / SKIN_WET[4])  # (drying, its gloss and colour go with its darkening)
        sl = (slice(y0 - Y0, y1 - Y0), slice(x0 - X0, x1 - X0))
        np.maximum(dark[sl], strength * m, out=dark[sl])
        np.maximum(sat[sl], on * m, out=sat[sl])
        seen[sl] += m  # (their union by the most of each, feathered edges overlapping inside it were rings)
        g = on * SKIN_WET[2] * m * np.clip(1.0 - rr, 0.0, 1.0) ** 2
        if pin > 0:  # a point of the key light off its wet surface, on its side towards it
            gx, gy = x + 0.4 * rx * lx, y + 0.4 * ry * ly
            g = g + pin * SKIN_WET[3] * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * 0.6 ** 2))
        np.maximum(gloss[sl], g, out=gloss[sl])
    skin = matte[Y0:Y1, X0:X1].astype(np.float32) / 255.0
    seen = np.minimum(seen, 1.0)
    d = dark * (1.0 + SKIN_WET[1] * 4.0 * seen * (1.0 - seen)) * skin
    patch = P[Y0:Y1, X0:X1]
    lum = (patch @ fx._LUMA)[..., None]
    patch[:] = (lum + (patch - lum) * (1.0 + SKIN_WET[0] * (sat * skin)[..., None])) * (1.0 - d[..., None])
    patch += (gloss * skin)[..., None] * lit
    return [b for b, _ in every]

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
