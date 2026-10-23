"""Water bent from an open palm, made rather than rendered (cuesync.elements): what is behind it, seen through it.

Water throws no light, so nothing here is added the way fire is: it is a lens and a mirror over the plate. A frame's
water is a set of Gaussian metaballs (a lone one is an exact sphere of radius R, two near ones bridge the way water
does), read as a half-thickness map. Each pixel's camera ray is traced through it: bent in at the front surface, across,
out at the back (Snell both ways, with a little dispersion), then on to what is behind: the hand just behind it, at
the depth its landmarks give it, or the room far behind, where the plate is read, averaged over as much of it as the
water squeezes into the pixel. Over that: the room reflected (Fresnel), the key light's glints, aeration where the water
is thin or fast, the plate's own grain. A small drop is shaded as one: what is behind it turned over and shrunk, as a
ball lens shows it, a faint core, a darker edge and a glint. Then the water is blurred along its motion over the
exposure, pixel by pixel (a drop flung free over all of its travel a frame, its glint a bright dash through a clear
smear; a thrown body sharp at its front and trailing off behind). Under it, on the hand: the light the water focuses,
inside the shadow it casts; and where it has been, the skin or the wall is left wet.

What makes it read as water and not glass is that it never holds still: its outline wobbles in lobes that breathe and
run round it (driven by the hand's motion and the voice), long, slow waves run over it in trains, dying down at its
outline, bending the room seen through it into moving bands and breaking the window its glint mirrors into glints along
their crests, points of light flash on their crests on its side towards the light, the water circulates inside it (seen
as the room through it swirling, its air and a long highlight carried round the loop it turns in), it drips from its
underside, and it froths where it is thin and fast; and that it is lit as water is: its rim mirrors the key light on
its side towards it and darkens away from it, broken by the waves, and inside its far edge it gathers the light into a
crescent that swells and fades as the waves run past, as a ball lens does.

Two forms (cuesync.elements.FORMS["water"]):

- ``orb``: water gathers out of the air into a ball floating over the open palm, drops drawn in from far round it into
  a seed that grows as they come, sized to the hand, wobbling and circulating, carried on the hand with the lag a mass
  of water has. It is held while the voice holds, and each hit in it squashes it along the aim, flings drops out round
  it, sends a ripple over it and flares the light it focuses. A measured palm thrust throws it along the aim and into
  the room, a ball stretched a little along its flight and smeared over half its travel,
  clear through its body with a little entrained air, the key light drawn out along it in a glint, sharp at its front,
  the air tearing drops off its back in a cone, leaving the palm wet where it sat; its shadow closes in on it over the
  wall as it nears, and it bursts on the room's back wall, where the water stays: it lands as a squashed body flashing
  white, throwing clear drops out along the wall that fall, and spreads over the wall in a lobed sheet drawn out the way
  it was thrown, its edge torn into tongues, its rim breaking up in arcs and gone in a few frames, flinging drops that
  land round it, and over the next frames drains into a wet patch the shape of the crown it landed in: its body, its
  tongues narrowing to points, the drops thrown on past their tips; its edge broken, mottled and streaked down, darker
  where the water lies thicker and in a band along its edge, where it dries last, soaking only into the wall (not a
  switch on it), whose lower edge creeps down where it faces down and runs down in narrow runs of their own lengths, a
  bead at the head of each, the fresh water there glossy, while drips break off the longest and slide down, leaving wet
  trails, and which dries while its thicker tongue roots stay wet. Where what it strikes is no plain wall (coats, a
  shelf) it bursts and its drops fall, leaving nothing. On a stationary hand the final hit releases it onto the palm;
  a voice that just stops does so after a short pause.
  It spreads at the skin contact plane and spills off the hand, leaving draining beads.
- ``whip``: a stream of water rising out of a thin film of a pool on the palm, wetting the skin round it, that stands
  up out of the palm as a round rod of water, swaying side to side as it rises (it rises from the palm: it does not wrap
  round the hand), one connected stream one and a half to two palms long, and trails the hand when it moves (each part
  of it held to its sway by a spring, stiff at the root and soft at the tail), hardly swelling or thinning as bulges run
  down it, lit as a rod of water is, a bright line down its side towards the light and its edges darker, most away from
  it, its ripples and highlights carried up it with its flow, white threads of air along its outer edge, churning white
  at its head, spraying off its outer part and spilling drops that fall back into the pool, a ring running out where
  each lands, thinning to its tail, where it frays into beads; where it flows out of the pool it spreads into an uneven
  film that thins into the skin and wets it round it, and where it hangs off the hand it sags and drips. It pours out
  with the voice; each hit flings its water out along the aim, its tail fastest, curving round as it goes, out over the
  picture, and away into the room: out in two frames, one stream tapering to a fine tail, it necks into beads and
  threads and breaks from its tail back into drops of uneven sizes, drawn out, some still threads, falls, and its drops
  fly on and strike the room's wall, where (if it is a plain wall) they leave a wet streak that runs as the orb's
  splash does, one wet patch with any left there before; its pool drains to a film in a few frames, leaving the skin
  wet and a few drops on it, and while the voice sounds it pours out again. When the voice ends it breaks into beads
  that fall.

The orb floats over the palm, not in it: it sits along the palm's normal, which is taken to face the camera (the
tracker's handedness, and with it the normal's sign, is unreliable for an upturned palm). The hand hides airborne water
where it is in front of it, by depth (the palm's plane through its landmarks, the fingers off it). A thin film on the
skin instead follows the palm's depth and silhouette.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np

from . import fx_light as fx
from .bending import LOST_FADE, emitter_at, still_there
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
ABSORB = np.array([0.06, 0.025, 0.015], np.float32)  # per hand size of path: a palm-sized volume is almost colourless
LENS = 0.5  # how far a ray turns per unit of the surface's slope, through both faces (a ball lens: 2 (n - 1) / n);
# it says how much of the plate one pixel of water sees
ROOM_DEPTH = 0.55  # shares of the frame's long side: the room behind the water (a ray turned by an angle sees about
# what the camera sees in that direction)...
LENS_FULL, LENS_MIN = 0.25, 0.15  # ... seen that far through water this thick (radius, hand sizes) and over, and
# through the smallest at this share of it (a drop's little ripples, carried a room away, crumpled what a splash's
# pieces showed into glass and ribbed a stream like a hose)
NEAR = (0.08, 1.5)  # hand sizes: the hand behind the water is taken at least / at most this far behind its back...
NEAR_OPEN = 0.6  # ... and this far where no hand is right behind it (a ray turned onto the hand from beside it)
DEFOCUS = 0.25  # a magnified patch of the plate is blurred by this share of its magnification: a ball lens is never
# sharp on what is right behind it (read point by point it showed the video's blocks, magnified; at 0.5 the fingers seen
# through the orb were a pink smear)
BASE_BLUR = 0.015  # of the water's local radius: the least the plate is blurred through it
LOD_MAX = 4  # pyramid levels the plate is read through (16 times smaller: the widest squeeze the water makes)
RIM_DARK = 0.22  # grazing rays see the room far round the side, through little water: the rim is this much darker...
RIM_BAND = (0.55, 0.88)  # ... over this range of 1 - n_z (on a ball, its last tenth of the radius: from 0.35, a ring
# 15 % of the radius wide, even all round, said glass), broken up by the waves...
RIM_FAR = 0.15  # ... in full on its side away from the key light and this share of it on the side towards it (dark
# evenly all round, a glass marble's or a soap bubble's outline)
RIM_LIT = (0.5, (0.6, 0.9))  # on its side towards the key light the rim mirrors the light: how bright (of the key
# light), coming on over this range of 1 - n_z (Fresnel: a grazing surface mirrors nearly all of the light it faces),
# so the rim is at least twice as bright towards the light as away from it
RIM_BREAK = 0.8  # how much the waves running round the rim break up the room mirrored in it
HELD_RIM = (0.12, (0.82, 0.98), 0.08)  # the held body's rim light: strength, grazing band, and least facet share;
# a narrow reflection broken by moving slopes leaves clear gaps instead of a solid glass collar
ROUND = (0.5, 0.85)  # a ray leaving the water this far off its way (the sine of the turn) sees the room round the
# water, by its direction, not the plate behind: near a ball's rim, bright towards the lights over it and dark
# towards the floor
TIR_DIM = 0.5  # where a ray is turned back inside the water (total internal reflection), what it shows is this dim
SMALL_EDGE = 0.4  # how much of that rim darkening and dimming a stream or a drop keeps (all of it drew a stream's
# outline as a tube's wall)...
SMALL_RIM = 0.08  # ... and under this radius (hand sizes) less again, down to none for the smallest
TINY = (0.025, 0.05)  # hand sizes: a drop under the first radius is shaded as a small drop, over the second as water
# with a body; a small drop shows...
TINY_SEES = 3.0  # ... what is this many of its radii behind it, turned over and shrunk, as a ball lens shows what is
# past its focus (right behind it, a little magnified, it was a clear disc in a dark ring: a soap bubble; a room away,
# the room shrunk into a grey bead: metal)...
TINY_EDGE, TINY_CORE = 0.14, 0.2  # ... inside a thin, faintly darker edge, its core lit by the light it gathers (of
# the key light) (an edge 0.25 darker round a core 0.2 darker, it was a hollow ring, an air bubble; a core 0.35 bright,
# smeared, a white blob: sleet)...
TINY_MIRROR = 0.5  # ... and mirrors the room this much less
TIR_SOFT = 0.5  # Snell's discriminant over which a ray goes from leaving the water to being turned back in it
# (switched at once, the line between the two was a jagged stair inside the rim; over 0.08, a second ring inside the
# orb's outline, a glass dome, a contact lens)
CALM = (0.5, 3.0)  # what is seen through water moving this many hand sizes a frame is bent half as far, and through a
# stream curved this much (its curvature times its radius) half as far again
FAR_FAST = 0.35  # water faster than WHIP_SHED shows the room through it from this share as far behind it (at 0.15 a
# lashing stream was a clear window onto the room, gone at a phone's size)
GLINT = (0.8, 0.07)  # the key light mirrored in the surface: its window (of the key light), a soft sheen round it...
GLINT_CORE = (1.3, 0.55, 0.08, 0.8, 0.025)  # ... clipped to white in its core: how bright (of white, whatever the
# key's strength: dimmer than the brightest things in the frame, a puff of cloud inside the ball), how big (of the
# window), the share of the finest waves it is mirrored in, how much dimmer it is where the surface curves both ways,
# and the width of normal smoothing in hand sizes; a held reservoir keeps finer moving reflection fragments...
GLINT_HOLD = 0.05  # ... on the orb this share of its core mirrored in its broad shape alone, the rest in its waves and
# ripples, so the highlight shivers and breaks up into glints along their crests as the water moves (all of it in the
# broad shape, one window sat fixed on the ball for seconds: a glass marble's; at 0.65, it still was)...
GLINT_POW = 12.0  # ... how tight the sheen is...
WINDOW = (0.4, 0.26, 0.07, 0.25, 4.0, 0.05)  # ... and the window's half width and height (in the mirrored ray's
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
GLINT_BREAK = (0.6, 0.14, 0.85, 0.3)  # along a stream its glints swell and fade with a noise this strong in streaks
# this long (hand sizes), carried up along it with its flow, while it lashes this much less, and this wide (of its
# radius) (in blobs 0.05 long and most of its width across, its highlights were a twisted glass or ice sculpture's;
# broken by a noise 1.8 strong, crumpled cellophane)
GLINT_LINES = (1.3, 0.5, 0.1, 0.72)  # a stream's edges catch the key light as a cylinder's do: the line along its
# edge towards the light, how bright (of the key light), all along it, and fast, along the other, how wide (of its
# radius, at least about a px) and how far out (of its radius); they carry a lashing stream's read through its blur...
GLINT_FAST = 0.3  # ... its broader glints dimmed this much while it is flung (at 0.75, a flung stream was a ghost)...
GLINT_SEG = (0.25, -0.6, 0.2)  # ... and lying on the skin, the line towards the light broken into lengths: where a
# noise along it (features this many hand sizes long) is over the first of these, fully over the second (in the air
# broken so, its highlights were cellophane's folds)
EDGE_SHADE = (0.34, (0.45, 0.85), 0.7)  # a stream bends the room round it in at its edges, darker: how much darker
# over this band across it (of its radius), and this share of that on its side towards the light (in a band a pixel
# wide on one side, a faint ribbon, cellophane)
CREST_LIGHT = (0.35, 0.4)  # how much the crests of the orb's ripples running past brighten and dim the crescent and its
# lit rim (either way, of them): one arc along its whole length, swelling and fading (broken into arcs by them, the
# crescent was a row of bright dashes; even, a crystal paperweight's)
CRESCENT = (0.32, (0.07, 0.10), (0.16, 0.22), 0.3)  # the key light gathered by the water onto its far inside edge,
# opposite the light, as a ball lens images it: how bright (of the key light), the band it lies in, in from its outline
# (of its radius: on a ball about 0.7 to 0.9 of the way out, inside the dark rim, not on it: on it, the rim was as
# bright away from the light as towards it; at 0.45 brightness it was not seen), and on a stream or a drop this share
# of it
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
FOAM_EDGE = (0.5, 0.3, 0.75)  # ... its core keeping this share of its froth, clear: froth fully only from this far
# across it to this (of its radius), at its broken edges...
FOAM_TORN = (0.2, 0.9, 2.5)  # ... and there only in torn bits: where a noise (features this many times its bubbles)
# is over the first of these, fully over the second (about two fifths of it: torn into streaks all over, smeared, it
# was a frosted brush, a whisk broom; in a sixth, a flung stream was clear and lost against the wall)...
FOAM_LEAD = (0.5, 0.25)  # ... more along its leading edge, where it tears into the air, but hiding at most this much
# there (its froth all at the front, its streaks' white ends lined up like a comb's teeth)
FOAM_MAX = 0.8  # froth hides at most this much of what is seen through the water
FOAM_GLINT = (0.8, 0.5)  # the bubbles it is made of glint: how bright (of the key light), how small (of its bubbles)
FOAM_STREAK = (0.15, 0.3, 0.1)  # along a stream it lies in streaks running with it: this long (hand sizes), this wide
# (of its radius), and this much of it between them
ROOT_CALM = (0.1, 0.45)  # hand sizes along a stream from its root: its ripples, broken glints and froth streaks come
# on from the first to the second (laid out by the distance along it, they ringed its root in the pool: a whorl)
RIPPLE = (0.1, 0.25, 0.3, (0.07, 0.07, 1.0))  # a stream's surface is broken by ripples running out along it: their
# height (log-field units), their length (hand sizes) and width (of its radius): flow lines along it (0.06 long and the
# stream's whole width across, they were rings: a ribbed hose, a larva; 0.22 high, folds: cellophane); and finer ones
# along its edges that ripple its outline in and out by a few hundredths of its radius as they run up it: their height,
# length and width (smooth-sided, a glass rod; all across it, crumpled cellophane)
MENISCUS = (0.55, 1.2, 0.9, 0.12, 3.2, 0.25)  # water lying flat on something (the pool, a stream along the palm) shows
# its edge towards the key light as a thin bright line where its surface curls down to meet it, mirroring the room: how
# bright, how far in from its outline and how wide (px), then a darker line inside it, how dark and how far in, and on
# the other side a dark line where it meets the skin, how dark (bright on both sides, a silicone tube)
REST = (3.0, 10.0, 0.3, 0.6)  # a stream whose middle is within the first to the second of its radii in front of the
# skin lies on it, flattened to this share of its thickness (as round as a tube there, it was a hose on the hand; the
# swirl holds it up to nine radii over the palm, and from 5 its round middle over the palm was a glass snake), bending
# the skin seen through it this much less, as a puddle does (a film of water on the skin hardly bends it: bent as much
# as a stream bends the room, a curl of it on the palm was a glass hook, and its pool a glass foot)
RIVULET = (0.4, 0.6, 3.5, 0.55, (1.5, 3.0), 0.16, 0.4, 0.5)  # lying on the skin, a stream spreads this much wider, its
# width varying this much more, thins into the skin over this many px inside its outline, and catches the light along
# its edge only where a noise along it is over this (a few short glints: lit in long lengths down both edges, as round
# as a tube, it was a glass rod, a hose); it spreads with its middle within the first to the second of its radii of the
# skin (spread where the swirl held it up off the palm, from 6 radii, it was a wide faint fold: cellophane; spread 0.8
# wider, a wide glass ribbon), to at most this radius (hand
# sizes: spread further, where it flared out of the pool it was a fat glass horn); the wet skin shows through it
# this much (covering the skin, a curl of it on the palm was a clear glass hook), and its glint line is this much
# dimmer
POOL_EDGE = (0.45, 0.9, 0.12)  # a puddle on the skin catches the light along its edge only where that faces the light
# from the first of these to the second, in lengths (a noise this many hand sizes across) (lit round half its outline
# and lined dark round the rest, it was a hard glass foot)
POOL_BEND = 3.0  # px: the most a puddle on the skin bends what is seen through it...
POOL_FEATHER = 0.35  # ... and it thins into the skin over this share of its radius (over 3.5 px, its edge was the hard
# rim of a clear disc on the palm: a contact lens)
FLAT_CALM = 0.75  # water lying flat is moved this much less by the small waves (on the pool on the palm they were
# facets: crumpled cellophane, crushed ice; at half, a gummy sweet)
DOT = (0.05, 1.15, 0.55, 0.18)  # a free drop under this radius (hand sizes) glints as one bright point: how bright (of
# the key light), at least how wide (px, sigma) and how wide for its radius
DOT_MIN = 0.5  # px: the smallest drop that glints
DOT_FACING = (0.5, 0.6, 0.85)  # only the drops whose facet faces the key light glint in full, about a quarter of
# them, the rest at this share (all glinting alike, a splash's spray was a shower of white dashes: rain, sparks; at
# 0.15, most drops were hollow rings)
BLUR_STEP, BLUR_TAPS = 1.0, 256  # px between a moving pixel's taps along its path over the exposure, and at most this
# many taps, the same for every pixel moving alike (a lash smears up to about 200 px: at 64 taps spaced 3 px apart,
# shifted pixel by pixel by a patterned share of a step to hide it, the pattern showed as a screen-door hatch on slow
# water and as evenly spaced bars across a lash)
BLUR_TAPER = 0.6  # the share of the exposure the shutter spends opening and closing (_shutter)
BLUR_KEEP = (0.32, 0.25)  # along their smear, as the shutter lets them through, the water's own light (glints,
# froth) keeps at least this much of itself, and its froth's cover this much (spread evenly, a fast stream's froth and
# glints thinned out with its refraction and it was gone at a phone's size; kept most at its front, its white ends
# lined up into bristles)...
DOT_KEEP = (0.4, 1.1, (2.0, 5.0))  # ... a small drop's glint this many times that, a speck's the first, a drop's from
# the first of these radii (px) to the second the second (kept in full, a spray of specks was a shower of white dashes;
# at 0.4 a drop flung off the orb was a see-through smudge, not a bright dash; at 1.8, with its body's light, sleet)...
BLUR_SOFT = 1.5  # ... softened over this many px (sigma)
BLUR_LEAD = (0.04, 0.1)  # a thrown body's smear is sharp at its front and trails off behind: its exposure lets all of
# it through at its front end, this share of it at its back, falling off between over this share of the exposure
# (e-folding) (smeared evenly over its travel, a thrown ball was a pale, see-through capsule: a ghost, a jellyfish)
BLUR_RAGGED = (0.3, 30.0)  # torn water's smear is up to this much longer or shorter (a noise this many px across its
# motion, so neighbouring streams end alike and its front is a ragged curve: each streak its own length 3 px apart, the
# front of a lash was a row of teeth, pinking shears; cut by one line, a comb)

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
INNER = 0.015  # slope: the water circulating inside turns what is seen through it this much, where it is thick
# (the room seen through it, a frame's height behind, moves about 15 px for it: at 0.035 it swirled into mush, at
# 0.022 the hand seen through the orb was smeared)
INNER_SIZE = 0.15  # hand sizes
FOAM_SIZE = 0.022  # hand sizes: the bubbles froth is made of
FLOW_CYCLE = 12  # frames: the water's patterns drift with its flow this long, then cross-fade into fresh ones (a pattern
# carried round for good winds itself into streaks)
SHUTTER = 0.5  # frames: the exposure (a phone indoors, 1/50 s), over which a moving drop smears into a streak
FREE_MOVE = 8.0  # a free piece's motion counts this many times its share of the field against a body of water it
# overlaps (counted evenly, a drop flung off the orb's rim took on the orb's stillness: a round ring, not a streak)
SMEAR_MAX = 6.0  # a piece's travel over the exposure is taken as at most this many times its size (a speck smeared over
# more was too faint to see)
G = 60.0  # hand sizes a second squared: gravity (a hand is about 9 cm: true gravity is 110; at that a splash left the
# frame before it read)
DRAG = 0.985  # per frame: what a flying drop keeps of its speed
ZCAM = 4.5  # hand sizes: how far the camera is from the hand (a first-person shot); a thrown orb shrinks as it goes
NOISE_SIZE, NOISE_CELL = 256, 4.0  # px: the tiling noise textures, and their features' sigma

# ---- the orb --------------------------------------------------------------------------------------------------------
ORB_R = 0.50  # hand sizes: its radius (a ball 8 cm across over a palm 9 cm long; at 6 cm it was a fifth of a phone
# screen's width, a marble)
ORB_GAP = 0.25  # hand sizes between its underside and the palm: it floats (at 0.1 its shadow and its focus sat under
# its own rim, and it read as set into the palm)
ORB_SPRING = (2.4, 0.45)  # Hz, damping: it follows the palm as a mass of water held on a spring would, trailing and
# overshooting
ORB_BOB = (0.025, 0.8)  # hand sizes, Hz: its slow float up and down
GATHER, GATHER_MIN, GATHER_LEAD = 16, 4, 2  # frames: gathering takes this long (in 12, whole by its eighth frame, it
# was a poof), at least this long squeezed in front of an early throw, and is done this long before a throw it was
# gathered for
GATHER_DROPS = 64  # drops that come together out of the air into it (22 were a puff, not water drawn in)
GATHER_LIGHT = 5.0  # its shade and the light it focuses come on as this power of how much of it has gathered
GATHER_FROM = (1.5, 3.2)  # radii: how far out they start (about 400 px on a phone's picture)...
GATHER_R = (0.012, 0.04)  # ... how big they are (hand sizes: at up to 0.07, a cluster of clear balloons fusing)...
GATHER_TRIP = (6, 11)  # ... and the frames each takes to come in, slowing as it reaches the ball (from 3 radii in 3
# frames, they were glass needles)
GATHER_SEED = 0.15  # the ball is this big (of its radius) when the first drops reach it, and grows evenly to its full
# size as the rest come in, more of them late (grown as the cube root of its water, it was nearly whole after a third
# of them: a poof)
MODE_HZ = {2: 1.6, 3: 3.1, 4: 4.8, 5: 6.7}  # its wobbles: two to five lobes round it (a drop's ratios, 1 : 1.9 : 3 :
# 4.2), each breathing in and out at its own rate...
MODE_AMP = {3: 0.13, 4: 0.018, 5: 0.018}  # ... asymmetric lobes, with little fourth harmonic to avoid square symmetry;
# at rest MODE_REST of
# it and at the voice's loudest all of it, breathing between MODE_BREATHE of that and all of it...
MODE_RUN = {3: 1.3, 4: 1.7, 5: 2.1}  # ... and running round it, this many lobes passing a point a second (standing
# still, three lobes at 3 % left a still showing a near circle: a marble; four alone at 0.022, a rounded square)
MODE_REST, MODE_BREATHE = 0.7, 0.6
MODE2_DRIVE, MODE2_DAMP, MODE2_MAX = 0.006, 0.25, 0.35  # the two-lobe stretch, driven by its acceleration (inertia
# squashes it), its damping, and the most it stretches
CIRCULATE = (1.9, 1.5)  # rad a second: it turns over inside this fast at rest (0.3 turns a second: slower, a held orb
# was a still lens), and more at the voice's loudest
BUBBLES = 4  # air caught inside, carried round the loop the water circulates in (26 of them scattered through it sat
# in it like a constellation of stickers; nine 2 to 6 px across were lost among its glints)...
BUBBLE_R = (0.024, 0.032)  # ... this big (hand sizes: about 14 to 20 px across, legible as entrained air)...
BUBBLE_LOOP = (0.58, 0.32, 0.45, 0.6, (1.6, 2.0), 0.05)  # ... round a loop this wide and this tall (of its radius),
# tipped this far towards and away from the camera (of its radius) and turned in the picture by up to this (rad), each
# carried round at its own share of the circulation (about 8 to 15 px a frame: at the water's own pace, 5 px, they
# hardly moved) and wandering this far off it (of its radius) (scattered and each turned on its own, they were specks,
# and the water was not seen to go round)...
BUBBLE_LOOK = (0.18, 0.4)  # ... a faint inner edge and a silvery directional rim, rather than black pinholes
RIPPLES = (3, (0.5, 0.75), (0.09, 0.14), (1.1, 1.8), 2.5, 0.2)  # small waves run over the held orb in trains, out of
# this many points round its outline, carried round with its circulation: their wavelengths (of its radius: at most
# three crests across it; at a quarter of it, short ripples all over it were hammered glass), their slope at rest and at
# the voice's loudest, how many crests pass a point a second, how far over its surface they run (of its radius,
# e-folding), and how far towards or away from the camera their points are (of its radius): they bend the room seen
# through it into moving bands and break the light it mirrors into glints along their crests (its small waves alone
# were a noise too broad to be seen: a still showed a glass marble; run out of points on its face, their rings were a
# target drawn on it)
ORB_SKIN = (0.3, 0.28, 0.5)  # on a body of water its small waves and ripples die down to this share at its outline,
# coming on in full this far in ((h / R)^2: about a seventh of its radius), so its outline wobbles in its lobes alone,
# and its finest waves are this much smaller (in full over all of it, its outline crinkled into a score of small bumps
# and its face was beaten glass)
SPARKS = (10, (2, 3), (0.9, 1.5), 2.4, (0.35, 0.88), 0.75, 0.2)  # points of the key light off the crests of its
# ripples, on its side towards the light, each flashing and gone, then flashing again elsewhere: how many, the frames
# each flash takes, how big (px, sigma), how bright (of the key light), how far out on it (of its radius), how far
# round either side of the light's way (rad: about a quadrant), and the least crest one flashes on (24 all over its
# face, two to five frames each, were glitter in a snow globe)
SWEEP = (1.25, (0.8, 0.5), 1.2, 2.8, 1.8, 0.35)  # one long curved highlight carried round the held orb with its water,
# along the loop its air goes round: seconds a turn, the loop's half width and height (of its radius), how long the
# streak is (rad round it), how wide (px, sigma), how bright (of the key light), and how much of that it keeps round
# the far side, seen through the water (with only its air going round, the water was not seen to circulate; 1.1 px
# wide and 0.9 bright, it was a faint scratch)
LINE_HALO = (0.15, 3.0)  # a highlight drawn as a line has a faint halo, this bright, this many times as wide
DRIP_EVERY = (12, 22)  # frames between drips: liquid sheds during a short hold, rather than behaving as a solid lens
DRIP_FORM, DRIP_R = 7, 0.035  # frames a drip takes to swell, neck and let go, and its radius (hand sizes: at a sixth
# of the orb's, a glass tulip bead hanging off it)
DRIP_SIDE = (0.3, 0.55)  # it hangs this far to one side of the orb's lowest point (of its radius, across): straight
# under it, it landed in the focus and its wet dot was lost in the light
DRIP_WET = (1.1, 48, 0.16, 3, (1.5, 3.0), 0.6)  # where it lands on the palm it leaves a wet dot this many of its radii
# across, drying over this many frames (half gone in half of them: a wet spot on skin stays for seconds; gone in 8, it
# was barely seen), at first this much darker with a point of glint, and this many tiny drops jump off it as it lands,
# this fast (hand sizes a second); round, feathered over this share of its radius (small, hard-edged and uneven, a mole)
SKIN_WET = (0.15, 0.08, 0.05, 1.0, 0.05)  # wet skin: how much more saturated, how much darker again round a patch's
# outline (at 0.3, a puddle's wet skin was a closed ring round it), its gloss towards its middle, the point of glint
# off a fresh drop's (of the key light), and how dark a patch is at which its colour and gloss are full (drying, they
# fade with it)
SKIN_CREASE = (8.0, 2.0)  # wet skin darkens more in its creases: its darkening raised by this times how much darker
# than round it the skin is there (a share, read over this many px) (evenly darker, a wet patch was a shadow)
PALM_WET = (0.2, 0.9, 2, 150, 0.55)  # where the orb sat the palm is left wet as it leaves: this much darker, over
# this share of its radius, coming on over this many frames and drying over this many (half gone in half of them),
# feathered over this share of its radius (thrown off a palm left as dry as before, the ball had never been on it)...
PALM_DROPS = ((3, 6), (0.009, 0.015), 0.7)  # ... with this many drops left on it, this big (hand sizes), spread over
# this share of the wet patch
JOLT = (0.08, (20, 40), (1.0, 1.7), (0.008, 0.022))  # a hit while it is held: how hard it kicks its slow wobble, how
# many drops it flings at the quietest and the loudest hit, how fast (hand sizes a second: about 300 to 500 px a
# second) and how big (hand sizes, a power law: most small), out round its middle across the aim and along it...
HIT_SQUASH = ((0.25, 0.35), 2, 6.0, 4.0, 0.06)  # ... squashing it along the aim by this much at the quietest and the
# loudest hit, for this many frames, then springing back through a stretch, ringing every this many frames and dying
# over this many (e-folding), and hopping it this far along the aim (of its radius) (a kick of its wobble alone
# changed 3 % of the picture on a hit: a marble that sat still on the beat)...
HIT_FLARE = (0.5, 0.2)  # ... the light it focuses this much brighter on the hit's frame and on the next...
RING = (0.07, 0.5, 0.3, 6.0)  # ... and the ripple it sends over it: log-field height, wavelength and speed (radii, a
# frame) and how long it lasts (frames; at 0.025 high it was not seen)...
RING_FROM = (0.6, 0.9)  # ... struck this far out from its middle as seen (struck near the middle, its rings were a
# target drawn on the ball)
THROW_SPEED = 18.0  # hand sizes a second, at the loudest hit (0.6 of it at the quietest): at 11 gravity took back
# most of its rise and it burst a hand over the fingers
TAKEOFF = 2  # frames it takes to get up to speed (it jumped off the palm in one)
RECEDE = 1.0  # of the throw's speed: thrown along the aim it also goes away from the camera, into the room, until it
# hits the room's back wall (ROOM_DEPTH behind the palm) and bursts flat on it (at half this it arced up and fell back
# most of the way to the hand before it got there)
THROWN_STRETCH = 0.15  # the most it stretches along its flight (its motion blur besides; stretched 1.35 : 1 and
# smeared over all its travel, it was a tall see-through capsule, a jellyfish)
THROWN_SHUTTER = 0.5  # frames: thrown, it is smeared over this much of its travel a frame, a phone's shutter...
THROWN_LONG = 1.3  # ... its whole length along its flight, smear and stretch, at most this many times its width: a
# ball of water in flight, its speed carried by the drops it trails (smeared over a whole frame's travel, twice as tall
# as wide)
THROWN_FOAM = (0.12, 0.8)  # a little entrained air during flight; the room must still transmit through the body
THROWN_STREAKS = ((0.45, 1.0, 1.8), (-0.68, 0.35, 1.2))  # the key light mirrored in it, drawn out into streaks along
# its flight: how far off its middle (of its radius, towards the light), how bright (of the key light) and how wide (px,
# sigma) (its window glint smeared with it was lost in its white core; a line along all of it, a seam)...
THROWN_GLINT = 0.14  # ... each the glint's half length (of its radius) and half its smear either side of it
TRAIL = ((6, 10), (0.02, 0.06), (0.35, 0.85), 0.55)  # drops it sheds off its back in flight: how many a frame, how big
# (of its radius), how fast against it, and over how wide a cone behind it (rad either side) (two or three a frame in a
# line behind it, it was not torn by the air but a comet)
DROP_SHUTTER = 1.0  # frames: a drop flung off the orb (by a hit, gathering into it, torn off it in flight) is smeared
# over all of its travel a frame, a short bright dash (over half of it, drops 12 to 20 px a frame apart were round
# rings: soap bubbles)
THRUST_SIZE = 0.8  # how hard a thrust with no hit throws it
THRUST_LOOKBACK = 6  # the tracker labels the stopped hand up to six frames after its speed peak
DROP_SIZE = 0.35  # how hard it splashes when let fall onto the palm
LINGER = 6  # frames it is held on after the voice stops before it falls
THROW_TAIL = 24  # frames: the last hit releases it when the voice falls silent within this after it; a voice holding on
# longer was holding the orb, not ending on the hit (a walk's long breath, surging once near its start, threw the orb
# away a second in and left the rest of the walk empty)
SPLASH = (60, 8, 110, 3)  # drops, ligaments, specks of spray and big lumps a burst at full strength makes (45 broad
# sheet pieces flung out at once overlapped into a flat cellophane star; 130 drops of one size were a window pane)...
SPLASH_R = ((0.02, 0.13), (0.007, 0.018), (0.15, 0.25))  # ... their radii (of the ball's): the drops' range (a power
# law: about a dozen over 0.06, the rest smaller), the spray's and the lumps'
SPLASH_SPEED = (3.0, 12.0)  # hand sizes a second: how fast its drops leave (slower, the burst was a puff a hand across)
SPLASH_CONE = 1.0  # they fly on along its motion, spread over a cone this wide (1: about 45 degrees)...
SPLASH_CARRY = 0.3  # ... keeping this much of its speed
SPRAY_LIFE = 7  # frames the fine spray lasts
SPLAT = (1.0, 0.12, 0.03)  # palm contact: radius, thickness and air content of a thin sheet spreading on the skin
COLLAPSE = (3, 18, 32, 84)  # frames from release to first contact, last pouring volume, drained sheet and dry skin;
# the sheet and the falling body overlap, so a spill has continuous visible volume instead of a one-frame burst
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
WALL_SPREAD = (1.3, 2.15, 0.9)  # burst on the wall, it lands as a body squashed to this radius (of the ball's) and
# spreads as a sheet to this, most of the way in this many frames (e-folding): about 0.6, 0.9 and all of it on its
# first three frames (at the ball's own radius, the hit was the smallest change of the splash, a thin ring on the
# strongest beat; landing at full size in one frame, it had no energy)
WALL_THROWN = (0.18, 0.15)  # it lies drawn out the way it was thrown, this much longer and narrower (of its radius:
# about 1.45 times as long as wide), its middle this far on that way (of its radius) (round and centred, with its
# drips hanging under it, it was a jellyfish)
WALL_FINGERS = (11, 15, 0.6)  # its edge torn into this many fingers, grown out over this many frames (e-folding), the
# tongues of the sheet it bursts out in (four or five, a spiked crown: a stamp; eight, most of them stubs, the stain
# they left had two ears)...
FINGER_LEN = (0.3, 1.0, 1.6)  # ... this long (of the sheet's radius: at 0.3 to 0.6 they were not seen, and the splash
# was a disc, a puddle, a starburst), between the two as this power of a share drawn evenly (most short, a few long)...
FINGER_DIR = (0.75, 0.6, 0.4)  # ... this share of them flung out round the way it was thrown, spread about it by this
# much (rad), the rest anywhere and shorter, down to this share across it and against it (more of them down, they were a
# jellyfish's legs; as long out to the sides, a figure's arms; at a quarter, stubs)...
FINGER_W = (0.03, 0.09)  # ... each this wide (half its width, of the sheet's radius), narrowing to half that at its
# tip...
FINGER_BEAD = (1.3, 1.6)  # ... ending in a bead this many times its half width (as triangular spikes it was a cartoon
# splat; two up, a cat's ears)...
FINGER_UP = 0.15  # ... reaching at most this far upwards (of the sheet's radius)...
FINGER_BLEND = 0.8  # ... and meeting the sheet in a fillet this many of its half widths wide...
FINGER_DRAIN = (2, 0.12, 1.2)  # ... and from this frame after the hit draining into the drop at its tip, down to
# this long at most (of the sheet's radius), over this many frames (e-folding) (left at its length, the longest made the
# stain an elephant's trunk, a kiwi bird)
WALL_JETS = 2.5  # the jets it bursts out in are this many of their finger's half widths wide (round it)
WALL_WOBBLE = ((3, 0.13), (5, 0.09), (7, 0.04))  # broad unequal bays interrupt both sides; four equal lobes left a
# rounded square once the thin tongues dried (amplitudes are half to one and a half times these shares of the radius)
WALL_EDGE = (1.3, 0.1, 1.2)  # its edge broken up: by this many px, in features this big (of its radius), and soft over
# this many px either side (an even, one-pixel outline was a line drawing; broken finer, a crinkly border)
WALL_THICK = (0.3, 1.5, 0.02)  # the sheet's thickness at the hit (of the ball's radius; its middle 1.6 times this),
# how fast it thins (frames, e-folding), and the film left at the end
WALL_FROTH = (0.35, 2.0, 0.1, 0.3, 9)  # at the hit it flashes white: at most this much, gone this many frames after
# (whiter, or lingering, a milky core, a cotton ball), its unevenness's features (of its radius: in bubbles 0.03 across,
# a stipple), and the streaks it tears out in along its jets, this long (of its radius) and this many round it (in soft
# clumps, camouflage, a snowball)
WALL_DOME = (0.18, 0.12)  # while it bulges, the light it gathers onto the wall at its foot and the shade at its top
WALL_LOOK = (3, 4)  # frames the sheet shows as running water, then frames it drains into the wet patch (all of a
# splash gone between two frames, on the strongest beat, was the element's hardest cut)
WALL_RIM = (5.0, 1.0, 0.45, 0.3, (0.35, 0.9), 0.13, (0.3, 0.8), (0.03, 0.3))  # the sheet's rounded rim: its width
# px, its height (of that), the bright line on its side towards the key light (of it) and the dark line where it meets
# the wall on the other side (how much darker), both broken up (bright all round, it was a vector outline); the bright
# line only along its body's edge where that faces the light this much, and both in arcs: a noise this many of its
# radii across, about ten arcs round it (along its fingers, round their beads, its wobbles, it was a doodle of little
# closed loops; in 12 px lengths, a closed glass ring); it bends, mirrors and glints only where it faces the light this
# much (about a third of it: all round, a crinkly bright border, a disc's outline); and the lines are there while it
# still bulges this much, gone this much (Sheet.dome: two frames after the hit, gone by the fourth; staying while it
# drained, a glass ring on the wall)
WALL_PLAIN = (2.0, 40, 0.3, 0.06, (0.04, 0.04))  # the water stays on the room's wall only where the plate round where
# it strikes (this many of its radii round it, at least this many px) is a plain, light surface: brighter than this
# (0..1), its brightness spread under this, and under this share of it edges (steps over the first of these between
# px 4 apart); elsewhere (a coat rack, a shoe shelf) it bursts and falls (laid on them, the wet patch was a flat grey
# disc with a white outline floating over coats and shoes)
WALL_CAST = (4, 0.9, 0.3, 0.14)  # over the last this many frames of its flight the thrown ball's shadow lies on the
# wall, this far off where it will strike per frame still to go (of its radius there), away from the key light, this
# much bigger and softer per frame to go, and at most this much darker, coming on as it nears and meeting it as it
# strikes (with none, a ball flew at a wall it never came near)
WALL_BEND = 1.6  # px the wall seen through the sheet shifts per px of its thickness's slope
WALL_RAYS = (0.08, 0.7, 0.35, 3)  # the streaks it runs out in, thicker and thinner (spread, of its thickness), the
# ridge along each of its jets, how long they are (of its radius), and how many tiles of noise round it (narrow across:
# about a twentieth of a turn) (flat and even it was a decal; in regular rays, a cut lemon)
WALL_GLINT = 0.6  # the key light mirrored in its slopes (of the key light), its core clipped to white as the orb's is
WALL_WET = (0.15, 0.05, 30, (0.3, 0.22, 20, 90))  # the wet patch it leaves: how much darker the wall where its film
# is even (linear), how much more saturated, the frames it takes to soak in fully (half as dark at first), and how it
# dries: this much paler over its body and this much over its tongues, which retain a narrow wet core longer, from this
# many frames after the hit to this
# (a fifth darker and a tenth more saturated all over from its sixth frame, it was a flat brown decal stamped on the
# wall; glossy over all its side towards the key light, that half of it was as light as the dry wall)
WALL_Q = 2  # px: the wet patch is worked on a grid this far apart (its finest features are ten times that)
WALL_MERGE = (0.3, 0.16)  # where patches overlap (a lash striking where the last one did) they are one wet patch: the
# lighter adds this share of itself to the darker, and the wall is at most about this much darker for its cover, its
# film's unevenness laid on that (rolled off: each darkened in turn, three strikes left three grey smudges stacked on
# each other; its film rolled off with it, the stain was one flat tint)
WALL_FILM = (0.25, ((0.28, 0.14), (0.15, 0.5)), (0.06, 0.45, 0.05), 0.15)  # its film is uneven, the wall darker by as
# much again where it is thicker: thicker by this in the middle, where the ball landed; mottled by noises this strong
# in features this big (of its radius: about 25 px and half of it across); streaked down the wall as it drained, this
# strong, in streaks this long and wide (of its radius); and thicker by this over its lower half, where it collects
# (even all over, or rolled off into one darkness, a flat tint: a shadow, ink; blotched by 0.45 in features a third of
# it across, camouflage)...
WALL_RING = (0.6, (4.0, 8.0))  # ... and darker by this along its edge (about 3 % more), in a band just inside where it
# feathers out, this many px wide, fading out over the next few, where its film dries last (a drying stain's ring)
WALL_SHEEN = (0.07, (10, 34))  # where it has run, down its lower edge and its runs, the fresh water mirrors the room's
# light, this bright (of the key light), fading from this many frames after the hit to this (with no sheen it read as a
# shadow; mirrored in blotches over all of it, pale stencilled shapes on a stain)
WALL_PATCH = ((0.025, 0.12), ((0.08, 0.045), (0.22, 0.08)), (2.5, 0.06), 2.0, (0.18, 0.35), (0.35, 1.0))  # the patch
# keeps the sheet's own outline, the crown it landed in: its tongues as far as they ran (drained back into their tips,
# they left none on the wall), as wide as they were but within these (half their width at their root, of its radius:
# widened, with round tips, they were a cartoon's ears; as narrow as their fingers' tips, they were not seen); its edge
# broken at two scales (features this big, of its radius, moved in and out this much, spread), feathered over at least
# this many px and, unevenly, up to this share of its radius more (fading out over the wall in places), and smoothed
# over this many px (rounded over a twentieth of its radius and cut again, its tongues were lost and it was a grey
# tombstone, a tile); it soaks only into the wall it landed on, where the plate is as bright as round it to within the
# first share of that, none of it from the second (a light switch in its way was stained grey, a dirty smear); and it is
# drawn out along the throw this many times as far as the sheet, its broad lobes this many times as deep (a lobed oval
# soaked in, it filled most of its bounding box: a tile; lobes 1.6 times as deep, a waist and a round belly, a figure's
# silhouette; drawn out as far, with its foot creeping down, a tall rounded slab)
WALL_TONGUE = ((1.4, 0.2), (1, 3), (0.05, 0.3), (0.3, 0.75), 0.05)  # its tongues on the wall narrow from this to this
# share of their width at their tip, a point, with this many drops of it past their tip, from this far past it to this
# (of its radius), each this big (of the tongue's half width at its root), and their edges are broken this much less
# than its body's (broken as much, a thin tongue fell apart)
WALL_SAG = (3, (5, 8), (0.2, 1.0), (0.02, 0.045), 0.6)  # from this frame after the hit its lower edge runs down in this
# many narrow runs, each its own length (of its radius, a power law: most short, one or two long) and half width (of
# its radius), the film collecting in them this much darker and glossier...
WALL_RUN = ((0, 10), (10, 36), 1.5, (0.3, 0.7), (0.25, 0.75))  # ... each setting off this many frames after that and
# creeping on, slowing, over this many frames (e-folding), with a bead at its head this many times its half width (three
# to six runs set at once, all over in six frames, and the patch kept its outline for seconds: a stamp), leaving its
# outline between these angles (of a half turn, from its right round its bottom) and drawn down from this far below its
# middle to this (of its radius) (run off its sides from its middle down, they drew its sides into straight edges)
WALL_CREEP = (0.25, 30, 0.05, 4.0)  # its lower edge creeps down this far (of its radius: an eighth of the patch's
# height) over this many frames (e-folding), as the film drains: as much as its outline faces down (to this power), and
# at its sides this share of that (pulled down column by column across its width, its sides were drawn into straight
# lines: a rectangle, a stamp; as the square of how much it faced down, its lower sides with it: a tall rounded slab)
WALL_SOAK = (0.12, 10)  # it soaks out this much wider over this many frames
WALL_BEADS = (40, (0.04, 0.14), (0.4, 2.4), (1.2, 2.5), (6, 10))  # drops flung out of the sheet onto the wall: how
# many, their radii (of the ball's), how far beyond the sheet's edge they come to rest (of the ball's radius), in how
# many frames (e-folding, held back by the wall), and after how many frames the small ones are gone into the wall (16
# under three px, round a stain, were not seen)
WALL_TIPS = ((0.05, 0.09), (0.03, 0.15), 3)  # a drop left at each finger's tip: its radius (of the ball's), how far it
# runs on past the tip (of the ball's radius), and the frame it gathers on, as the sheet drains (drawn over the finger's
# own bead while the water still ran, a loop on a hook)
WALL_SLIDE = (0.11, 0.8, 0.25)  # a drop on the wall bigger than this (of the ball's radius) slides down it, at most
# this fast (hand sizes a second, for the biggest), taking this long (seconds) to get going
WALL_DRIPS = (2, (0.15, 0.22), 2.6, 0.15, 6, (4, 9), 8, (0.45, 1.0))  # drips breaking off its runs and sliding on down
# the wall: how many, their radius (of the ball's), how fast at most (hand sizes a second, at the wall: at 1.5 they ran
# 40 px in the ten frames left), how soon they get to it (seconds), the frame the first sets off on and the frames
# between one and the next, the frames each takes to swell out of the edge, and its share of that speed (all setting
# off on one frame, as long as each other, a row of legs: gravity cannot draw drips in the 40 ms after a hit)
WALL_TRAIL = (0.5, 0.5, 1.0, 0.12)  # the wet trail a running drop leaves: how dark against the patch (a thinner film:
# as dark, a needle-thin dark stem), its half width at its top (px) and at the drop (of the drop's radius: as wide as
# the drop at its foot), and the gloss along its side towards the key light
WALL_FALL = (4, (0.035, 0.07), (2, 8))  # drops that fall off the splash's lower edge into the air: how many, their
# radii (of the ball's), the frames after the hit they let go on (spread out: falling all at once, a sudden shower)
WALL_SPRAY = (64, (3.5, 7.5), (0.015, 0.07), (8, 16), 2.0)  # drops off the hit, flying out along the wall and off
# it and falling: how many at full strength, how fast (hand sizes a second at the wall: about 400 to 800 px a second),
# how big (of the ball's radius: most 2 to 4 px across, a few up to 10) by a power law this steep, and the frames they
# last (70 specks under a px at twice the speed, gone in 3 to 7 frames, were not seen: the burst was a stamp; 6 to 18
# px across, soft white dashes: sleet)
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
WHIP_HEIGHT = 2.6  # hand sizes: the stream rises this far out of the palm, along its normal...
WHIP_UP = 1.4  # ... bent this much towards the picture's up (a palm facing the camera sent it straight at it, a palm
# facing up sent it along the fingers, foreshortened away from the camera)...
WHIP_COIL = 0.38  # ... and coils round that way as it rises, this far out (hand sizes): with its rise, one and a half
# to two palms long (swirled round the whole hand 1.1 hand sizes out, most of it behind and beside the hand, it read as
# a puddle with a faint hoop; round the palm 0.42 out, as a ring of water lying on it)...
WHIP_SWEEP = (0.6, 0.85)  # ... wrapping this many turns round it, quiet and at the voice's loudest...
WHIP_FLAT = 0.48  # ... its coil this much as deep towards and away from the camera as it is wide across the picture (as
# deep, where it came towards the camera end on it piled up into a wide, twisting fold: a cellophane ribbon)...
WHIP_RISE = 0.3  # ... its coil opening over this much of its length from the root
WHIP_SPIN = (0.45, 0.5)  # turns a second its coil turns at, quiet, and more at the voice's loudest
WHIP_WOBBLE = (0.14, 3.0, 1.3)  # its coil's lobes: how far in and out (of its coil), how many, turns a second
WHIP_HZ = (3.0, 0.9)  # how stiffly it holds its swirl at the root and at the tail: the tail trails a moving hand,
# as a ribbon does (one stiffness throughout and the stream moved with the hand as one piece, a hoop)
WHIP_DAMP = 0.55  # their damping ratio
WHIP_R = (0.13, 0.038)  # hand sizes: the stream's radius at its root and at its tail (blobs weighted, as thick as
# designed: summed unweighted, they drew a hose 1.7 times as thick; at 0.1 at its root, a thick glass sculpture; at
# 0.07, lost on a phone's screen)
WHIP_TIP = 0.35  # its last fifth thins by this much to its end
WHIP_SPACING = 0.6  # its blobs are at most this share of their radius apart
WHIP_POOL = (1.9, 0.035, 0.06, 0.06)  # the pool on the palm it rises out of: its radius (of the stream's at its root),
# its thickness (of a ball's: at 0.14 it was a domed gel lens with a raised rim, and what was seen through it was bent
# 10 px), its froth, and how far its outline ripples (of its radius; at 0.16 a frilled edge)
WHIP_POOL_ALONG = (1.15, 0.2)  # it is drawn out this many times longer along the way the stream leaves it (keeping its
# area) and lies this far that way (of its radius): the stream widens into it (round, centred on the root, a spoon;
# drawn out 1.5 times, with the stream narrowing out of it, a pointed snake's head)
WHIP_POOL_PARTS = (4, (0.6, 1.1), (0.4, 0.65), 0.12)  # an irregular puddle: this many flat pools besides it, each
# this far from its middle (of its radius) and this big (of it), drifting round it at up to this many turns a second
# (kept close in, the puddle was round, and with the stream out of it a comma, a tadpole)
WHIP_FLARE = (0.08, 0.15, 1.0, 0.4, 0.5)  # the stream flares out this much wider where it leaves the pool, narrowing
# over this share of its length (at 0.35, the pool and the stream were a tadpole's head and tail, a comma), more on one
# side than the other: its flare leaning up to this share of itself to one side, swaying at this many Hz, and swelling
# and thinning along it by this share of itself (at 0.7, even all round and smooth, the foot of a crystal vase, a glass
# boot)
WHIP_WET = (2.3, 0.17, 0.65)  # the palm round the pool is wet this far out (of the pool's radius), this much darker,
# fading out over this share of that (at 0.08 out to 1.3, the skin round the pool was as dry as ever; as dark out to the
# pool's own edge, a grey sticker on the palm)
WHIP_TRAIL = (0.15, 4, 1, 0.03, 1.5, 0.05, 0.7)  # where the stream lies on the palm it wets the skin round it: how
# much darker at first, drying over this many frames (left for a second where the stream had swirled past, a string of
# dark spots), marked every this many frames (every other one, the palm flickered frame to frame) at points this far
# apart along it (hand sizes: further apart, beads), wet this many of its radii out and this many hand sizes besides,
# feathered over this share of that (a clear stream with nothing round it on the skin was a glass snake)
WHIP_DRAIN = 3.5  # frames its pool takes to drain to a film on the palm, when a hit flings its water or it lets go (as
# a clear disc, then a ring, it stayed on the palm: a contact lens)...
WHIP_LEFT = ((3, 6), (0.009, 0.015), 0.8)  # ... leaving the skin wet (drying as where the orb sat, PALM_WET) and this
# many drops this big (hand sizes) over this share of where the pool was
WHIP_LIFT = 0.06  # hand sizes it starts over the palm
WHIP_SHED, WHIP_SHED_RATE = 9.0, 0.12  # hand sizes a second: where it is faster than this it sheds drops, each point
# of it past WHIP_SHED_FROM of its length at this chance a frame per time over it...
WHIP_SHED_FROM = 0.3  # (shed near the root, they wound round it like a screw's thread)
WHIP_SPRAY = (1, 4)  # fine drops its tail sprays a frame while it moves
WHIP_MIST = (0.3, 0.1, (0.004, 0.014), (0.8, 2.0))  # turning, the stream sprays off its outer part: from this share
# of its length out, at this chance a frame at each node, drops this big (hand sizes, a power law) flung out of its
# coil this fast (hand sizes a second) (smooth all along, it was a crisp glass hoop; at 0.25, with what it spilt, a
# row of beads all along its edge)
WHIP_LIES = 0.15  # only its first this share, flowing out of the pool, lies on the skin (lying wherever it came near
# the skin, where its swirl carried it past a finger it spread into a wide, faint fold: cellophane)
WHIP_FOAM = (0.30, 0.45, 0.65, 0.35)  # how aerated it is all along, more at its tail, its head churning into the air (a
# bright leading edge), more where it is fast (a fast jet goes white), and more at its root where it churns out of the
# pool (clear all along, a glass rod; at 0.14 standing up out of the palm, a glass sculpture)...
WHIP_AERATE = (0.3, (0.5, 0.85), (0.12, 0.12), (0.45, 1.0), 0.5)  # ... held, its froth lying in white threads along
# its outer edge, its core keeping this share of it: the threads from this far across it to this (of its radius), this
# long (hand sizes) and this wide (of its radius), where a noise is over the first of these, fully over the second
# (about a quarter of its edge), this white (froth all through it, it was frosted glass; none, clear glass)
WHIP_SPILL = ((2, 5), (0.004, 0.011), (0.3, 0.95), (0.2, 0.6), (5, 0.06, 0.25, 0.25))  # held and turning, drops
# spill off its coils and fall back into the pool: how many a frame, how big (hand sizes), from how far along it, flung
# off it this fast (hand sizes a second), and where this share of them lands in the pool, a ring running out on it:
# over this many frames, as far as this (hand sizes), this bright (of the key light) (a stream that only turned, never
# spilling, was a sculpture; a ring for every drop, the pool was full of bubbles)
WHIP_AIR, WHIP_AIR_R, WHIP_AIR_SEEN = 4, (0.12, 0.28), 0.3  # bubbles carried along it, their size (of its radius),
# and how clearly they are seen (at 0.7, rings like glass beads; at 0.45 down a thinner stream, still beads; ten of
# them, specks all through it)...
WHIP_AIR_FAST = (8.0, 15.0)  # ... gone where it moves from this many px a frame to this (drawn sharp inside the
# stream's motion blur, they were a glass paperweight's)
WHIP_SAG = (0.12, 0.55)  # of gravity, while it is held up: over the hand, and hanging off it (at 0.3, a loop held up
# off the side of the hand was a carry handle, a hose)...
OFF_HAND = ((0.04, 0.2), (0.6, 0.9), 0.09)  # ... which it is from this far outside the hand to this (hand sizes), or
# with none, from its palm's point; its fingers taken as this thick (half their width, hand sizes)...
SKIN = ((0.15, 0.6), 8, 0.05)  # ... and wherever the hand's matte, softened, is under the first of these, on it over
# the second, read this many times smaller and softened over this many hand sizes (by the landmarks alone, a stream
# beside the hand's edge or between its fingers lay on skin that was not there, and hung there as a glass bowl)
BONES = ((1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19), (2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20))
# the fingers' bones, landmark to landmark
WHIP_DRIP = ((8, 13), 1.35, 0.35)  # hanging off the hand, it drops a drip from its lowest point every so many frames,
# swelling there to this many times its width as it necks above it by up to this share of it (held up stiffly, it was
# a hose; swelling 1.6 times under a neck half as wide, a bulb on a ribbon)
WHIP_WIDTH = (0.06, 0.7, 0.25, 0.08)  # its width varies along it besides its bulges, this much (spread, of its
# radius), in features this long (hand sizes), drifting out at this share of its flow, and at most this much either way
# (even all along, a silicone tube; by 0.3, up to 0.42, wide faint folds and thin edge-on lines: a twisting ribbon)
WHIP_CULL = (1.3, 2.0)  # hand sizes: what it sheds while it pours is gone between these distances from it
WHIP_SWELL = (0.09, 0.13, 0.12)  # bulges carried down the stream with its flow: how much thicker and thinner (the
# spread, of its radius), their size (hand sizes: about eight of its radii apart, at no set spacing), and the most
# they swell or thin it (in three sets of regular waves, the shortest 13 to its length, it was a ribbed larva; by up to
# 0.38, with its width's, a twisting ribbon: a round rod of water stays within about a seventh of its width)
WHIP_PULSE = (0.23, 0.30, 0.48)  # a broad varicose pulse: extra radius, length in hand sizes, and seconds between
# pulses. It travels with the water so even a held stream shows flow rather than the fixed contour of a glass rod.
WHIP_FRAY = (0.5, 1.0)  # it necks into beads from this share of its length, as deep as this at its tail...
WHIP_BEAD = (8.0, 0.8, 0.4)  # ... beads this many of its radii apart, flowing out along it, how much thinner the
# stream between them is where it has necked through, and how much more or less apart they are (a smooth noise
# carried with the water)
WHIP_FLOW = 3.2  # hand sizes a second: the water flows out along it (its inside pattern drifts that way: 30 to 40 px
# a frame on a phone's picture; at 20, its ripples and glints hardly moved, and the flow was not seen)
WHIP_GROW, WHIP_BREAK = 10, 5  # frames it takes to pour out, and to neck into beads when the voice stops
WHIP_LEAD = 6  # frames it is out before a hit that opens the voice

# ---- the lash: a hit flings the stream's water out along the aim ----------------------------------------------------
LASH_SPEED = (1.5, 0.6, 0.8)  # hand sizes a frame: its tail is flung this fast at the loudest hit and this share of
# it at the quietest, its root not at all, in between as this power of how far along it (driven to a straight line
# and held there for six frames, it was a glass rod standing to the frame's top)...
LASH_CURL = 1.2  # ... its tail this far round from its root's way (rad, swinging out over the picture: the hand's
# flick), so it curves along the aim...
LASH_RECEDE = 0.6  # ... and this share of that away from the camera, into the room, where it strikes the wall (as
# fast as along it, it went off into the room foreshortened, a squirt, not a lash)
LASH_DRAG = 0.62  # a frame: what the flung stream keeps of its speed across the picture while it is whole (a thin jet
# tearing into drops in the air): out in two frames, it falls under gravity within three of its apex, its drops flying
# on into the room
LASH_R = (0.2, 0.025)  # hand sizes: the flung stream's radius at its root and at its tail (about 80 px wide to a 10 px
# tip on a phone's picture), thinning as it is drawn out...
LASH_FOOT = (0.35, 0.2)  # ... but at its root, on the palm, this share of that, lying flat there as the pool does, both
# up to this share of its length (as thick as further out and round there, it left a clear glass disc on the palm)
LASH_BREAK = (1.5, 3.0, 0.3)  # frames after the hit its tail breaks into drops, the frames more its root end takes,
# and the share of it from the root that never breaks but falls back into the pool (its top third in drops by its
# second frame, all of its outer part by its fourth)...
LASH_NECK = 1.3  # ... necking into beads joined by threads over this many frames before it breaks (ligaments)...
LASH_DROPS = (1.9, 2, (0.3, 0.55))  # ... each length of it beading up into a drop at most this many times its radius,
# with this many satellite drops this big (of it) beside it...
LASH_BEADS = ((0.5, 1.5), (1.0, 1.25), 0.18, (1.4, 1.8), 0.4)  # ... the drops of uneven sizes (these times that), drawn
# out along their flight this many times, this share of them still joined to the next as threads drawn out this many
# times, and shifted along it by up to this share of their length (each length one round drop where it was, a string
# of pearls, a dotted line)
LASH_BACK = (3, 6)  # frames after the hit the part left at its root has drained into the pool, and it pours out again
LASH_GONE = 30  # frames after the hit what it flung is let go of, if it has not struck the wall (thrown away from it)
LASH_SPRAY = 18  # threads and drops its tail tears into as it is flung (a smooth front was a hard sheet)
WHIP_THREADS = 0.5  # the share of them that retain short ligaments, drawn out 1.2 to 1.8 times their width...
WHIP_TORN = ((0.004, 0.02), (0.8, 1.3))  # ... their radii (hand sizes, a power law), and their speed against the
# stream's there: each flung on along its own path, smeared to its own length (all at the stream's speed, their
# streaks ended in a line)
LASH_WALL = (0.16, 0.5)  # what of it strikes the wall lands as a ball of this radius (hand sizes) would, drawn out this
# much along its way: a wet streak, draining and running as the orb's splash does

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
    presence: np.ndarray | None = None  # (n,): held tracker positions are visible only while the hand is still there

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


def hand_presence(track: dict, n: int, plate: np.ndarray | None = None) -> np.ndarray:
    """Visibility of the tracked hand: reject a copied leading position and fade a genuinely lost hand. A tracker
    gap over an unchanged picture retains the hand, as in lightning; without a plate only the tracker is available."""
    src = list(track.get("src") or [])[:n] + ["mp"] * n
    presence = np.ones(n)
    first = next((t for t in range(n) if src[t] not in ("hold", "lost")), None)
    if first is None:
        return np.zeros(n)  # no observation exists to justify a copied position
    here = track
    if plate is not None:
        h, w = plate.shape[1:3]
        xy, u, s = fx.track_arrays(track, len(track["xy"]), w, h)
        here = track | {"xy": xy.tolist(), "dir": u.tolist(), "scale": s.tolist()}
    for t in range(first):
        if plate is None or not still_there(plate, here, first, t):
            presence[t] = 0.0
    run, seen = 0, first
    for t in range(first, n):
        gone = src[t] == "lost" and (plate is None or not still_there(plate, here, seen, t))
        run = run + 1 if gone else 0
        seen = t if src[t] not in ("lost", "hold") else seen
        presence[t] = max(0.0, 1.0 - run / LOST_FADE)
    return presence


def hand_of(track: dict, n: int, plate: np.ndarray | None = None) -> Hand:
    """The palm from the track: its point (the emitter), size (smoothed), normal and depth (from its landmarks; without
    them, a palm facing the camera tipped towards the emitter's direction, at depth 0)."""
    em = [emitter_at(track, t) for t in range(n)]
    p = np.array([e[0] for e in em])
    d = np.array([e[1] for e in em])
    s = _tsmooth(np.array([e[2] for e in em]), TRACK_SMOOTH)
    presence = hand_presence(track, n, plate)
    lms = track.get("landmarks") or []
    seen = [t for t in range(min(n, len(lms))) if lms[t] is not None]
    if not seen:
        nrm = _unit(np.concatenate([NO_LANDMARKS_TILT * d, np.full((n, 1), math.sqrt(1 - NO_LANDMARKS_TILT ** 2))], 1))
        d3 = np.concatenate([d, np.zeros((n, 1))], 1)
        along = _unit(d3 - (d3 * nrm).sum(1, keepdims=True) * nrm)
        return Hand(p, nrm, along, s, np.zeros((n, 3)), None, presence)
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
    return Hand(p, nrm, along, s, plane, off, presence)


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
    throw: bool  # a measured thrust throws it; sound alone releases it over the palm
    size: float  # how hard: the throwing hit's size
    jolts: list[tuple[int, float]]  # the hits while it is held, with their sizes


def orb_plan(tl: Timeline, track: dict) -> OrbPlan:
    """Gather with the voice; release on its final hit or a measured palm thrust. Only a thrust imparts flight:
    sound on a quiet hand collapses the water locally. A voice that simply stops lets it fall after LINGER frames.
    Extension labels need a palm speed above 1.4 hand lengths/s, so a tracking label alone cannot throw water."""
    heard = np.flatnonzero(sounding(tl))
    hits = sorted(zip(tl.hits, tl.sizes))
    start = int(heard[0]) if heard.size else int(tl.tap_frame)
    if hits:
        start = min(start, int(hits[0][0]))
    speed = np.asarray(track.get("speed") or np.zeros(tl.n), float)
    scale = np.asarray(track.get("scale") or np.ones(tl.n), float)
    thrusts = [f for f in _thrusts(track) if start + GATHER_MIN <= f < min(tl.n, len(speed), len(scale))
               and np.max(speed[max(0, f - THRUST_LOOKBACK):f + 1]) > 1.4 * max(float(scale[f]), 1.0)]
    throw_at = next(iter(thrusts), None)
    size = THRUST_SIZE
    ends_on = hits and (not heard.size or heard[-1] - hits[-1][0] <= THROW_TAIL)
    release_at = throw_at
    if ends_on and (release_at is None or hits[-1][0] <= release_at):
        release_at, size = int(hits[-1][0]), float(hits[-1][1])
    if release_at is not None:
        g0 = max(0, min(start, release_at - GATHER - GATHER_LEAD))
        g1 = max(g0 + 1, min(g0 + GATHER, release_at - GATHER_LEAD))
        release, throw = release_at, release_at == throw_at
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
    lie: float = 0.0  # 0..1: how much of a stream here lies on the skin, a rivulet (RIVULET, REST)
    lead: float = 0.0  # 0..1: how much its smear over the exposure is sharp at its front and trails behind (BLUR_LEAD;
    # 0: even along it)
    folds: tuple = ()  # ((angle, angular width, reach in radii), ...): localized flowing folds in this body's own
    # surface, so a stretching lobe shares its optical depth instead of becoming a second little lens at the join


@dataclass
class Scene:
    """One frame's water."""

    blobs: list = field(default_factory=list)
    ribbon: list = field(default_factory=list)  # falling water, shaded behind its reservoir to share a clean neck
    bubbles: list = field(default_factory=list)  # (x, y, r, seen): air in the water, and how clearly it is seen (0..1)
    rings: list = field(default_factory=list)  # ripples running over the orb: (its centre x, y and radius, the unit
    # direction from its centre to where it was struck, age frames, height, wavelength and speed px over its surface)
    trains: list = field(default_factory=list)  # trains of small waves running over the held orb (RIPPLES): (its
    # centre x, y and radius, the unit direction from its centre to where they run out of, height, wavelength px, how
    # far their crests have run px)
    sparks: list = field(default_factory=list)  # points of the key light off the crests of its ripples (SPARKS): (its
    # centre x, y and radius, how far round from the light's way (-1..1 of SPARKS' spread), how far out (of its radius),
    # size px, strength)
    sweep: tuple | None = None  # the highlight carried round the held orb with its water (SWEEP): (its centre x, y and
    # radius, how far its loop is turned in the picture, rad, where round the loop the streak is, rad, strength)
    streak: tuple | None = None  # the key light mirrored along the thrown orb's flight (THROWN_STREAKS): (its centre x,
    # y, the unit way it flies, x and y, how far it smears px, its radius px, strength)
    plops: list = field(default_factory=list)  # rings running out on the palm where drops land (WHIP_SPILL): (x, y,
    # radius px, strength)
    cast: list = field(default_factory=list)  # the thrown orb's shadow on the wall it is about to strike (WALL_CAST):
    # (where it will strike, x, y, frames to go, its radius there px, how dark)
    wave: float = WAVE[0]  # how high its small waves are
    held: tuple | None = None  # (x, y, z, R, w): the orb over the palm, which focuses light onto the hand and shades
    # it, and how much it does (0..1: rising as it gathers, falling as it is thrown off; thrown, it stays where the orb
    # left the palm, so its shade fades there instead of sliding after it)
    lobes: tuple = ()  # the held orb's wobble ((lobes, share of the radius, phase), ...): its light on the palm wobbles
    # with it...
    net: float = 1.0  # ... and lies this much in the net of lines its surface draws (SPOT_IN)
    flare: float = 1.0  # the light it focuses, this many times as bright (on a hit: HIT_FLARE)
    wet: list = field(default_factory=list)  # (x, y, rx, ry, angle, strength, phase of its uneven edge, glint, soft):
    # wet patches it leaves on the palm, darker and faintly glossy (under the stream's pool, where a drip lands)
    band: list = field(default_factory=list)  # ((m, 2) px, (m,) radii px, strength): the skin wet round where the
    # stream lies, a band through those points
    film: list = field(default_factory=list)  # the water on the room's wall where the orb burst on it: a Sheet, Trails
    # and Beads (_wall), drawn on the plate behind the water in the air (_film)
    coat: tuple | None = None  # (x, y, radius, wetness, flowing[, exits]): the palm's rippled receiving sheet;
    # optional exits are silhouette-measured offsets in film radii, shared with its airborne runoff roots


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
    sag: float = 0.0  # how far the wet patch's whole lower edge has crept down (of r)
    runs: tuple = ()  # ((angle rad, half width (of r), length (of r)), ...): the runs its lower edge runs down in
    reach: tuple = ()  # (length past r (of r), ...): how far each finger has run out, which the wet patch keeps after
    # it drains back (empty: as long as its fingers are now)
    dry: float = 0.0  # 0..1: how far the wet patch has dried (WALL_WET)
    dots: tuple = ()  # ((finger, how far past its tip, how far across its way, radius: of r), ...): the drops it threw
    # past its tongues' tips, which the wet patch keeps (WALL_TONGUE)
    fresh: float = 0.0  # 0..1: how freshly the water has run down it (WALL_SHEEN)


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


def _moving(r: float, v: np.ndarray, k: float = 1.0, shutter: float = SHUTTER,
            most: float = math.inf) -> tuple[tuple, tuple]:
    """A blob of radius ``r`` moving ``v`` px a frame, stretched ``k`` times along it (keeping its volume): (its shape,
    how far it travels over ``shutter`` frames: Blob.move, at most SMEAR_MAX times its size, and at most ``most``
    px)."""
    sp = float(np.hypot(*v))
    if sp < 1e-6:
        return _shape(np.array([1.0, 0.0]), k, 1.0 / math.sqrt(k)), ()
    travel = min(SMEAR_MAX * 2.0 * r, sp * shutter, most)
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
    strongest free piece over them), per pixel the weighted means of the blobs' sigma^2 (``s2``, each weighted by the
    fourth power of its own field over its size's: where a drop flies by a big body of water, it is the drop's), depth,
    flow, foam, motion over the exposure (``mx``, ``my``, a free piece's counted FREE_MOVE times over), thickness
    (``th``, Blob.flat) and how much of it lies on the skin
    (``li``, Blob.lie); over a stream, where along it a pixel is (``sa``, px from its root), how far across it (``sx``,
    in its radius, signed) and how much of the water there is the stream's (``sw``). ``dots``: the free pieces (x, y,
    z, r), for the glints of the small ones (_dots)."""

    def __init__(self, blobs: list[Blob], h: int, w: int):
        ext = []
        self.dots = [(b.x, b.y, b.z, b.r, b.glint) for b in blobs
                     if b.free and DOT_MIN <= b.r and 0 <= b.x < w and 0 <= b.y < h]
        for b in blobs:
            if b.r < 0.5 or b.w <= 0:
                continue
            sig = b.r * SIG_K * (1.0 + sum(abs(a) for _, a, _ in b.modes) + sum(a for _, _, a in b.folds))
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
        (self.f, self.fm, self.wt, self.ws, self.wv, self.s2, self.zc, self.vx, self.vy, self.fo, self.mx, self.my,
         self.th, self.sa, self.sx, self.sw, self.li, self.ld) = (np.zeros((y1 - y0, x1 - x0), np.float32)
                                                                   for _ in range(18))
        for b, e in ext:
            self._splat(b, e)
        self.f += self.fm
        wt = np.maximum(self.wt, 1e-12)
        self.s2 /= np.maximum(self.ws, 1e-30)
        self.mx /= np.maximum(self.wv, 1e-12)
        self.my /= np.maximum(self.wv, 1e-12)
        for a in (self.zc, self.vx, self.vy, self.fo, self.th, self.li, self.ld):
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
        if b.modes or b.folds:
            th = np.arctan2(v, u)
            m = 1.0 + sum(np.float32(a) * np.cos(np.float32(l) * th - np.float32(ph)) for l, a, ph in b.modes)
            for angle, width, reach in b.folds:
                turn = np.arctan2(np.sin(th - angle), np.cos(th - angle))
                m += np.float32(reach) * np.exp(-0.5 * (turn / np.float32(width)) ** 2)
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
        gm = g * np.float32(FREE_MOVE if b.free else 1.0)
        self.wv[sl] += gm
        g4 = (g0 * g0) ** 2 / np.float32(sig ** 4) + np.float32(1e-20) * g  # (far out, where each is a trace, by its
        self.ws[sl] += g4  # plain weight)
        self.s2[sl] += g4 * np.float32(sig * sig)
        self.zc[sl] += g * np.float32(b.z)
        self.vx[sl] += g * (np.float32(b.vx) - np.float32(b.spin) * dy)
        self.vy[sl] += g * (np.float32(b.vy) + np.float32(b.spin) * dx)
        self.th[sl] += g * np.float32(b.flat)
        if b.move:
            self.mx[sl] += gm * np.float32(b.move[0])
            self.my[sl] += gm * np.float32(b.move[1])
        if b.foam:
            self.fo[sl] += g * np.float32(b.foam)
        if b.lie > 0.0:
            self.li[sl] += g * np.float32(b.lie)
        if b.lead > 0.0:
            self.ld[sl] += g * np.float32(b.lead)
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

    def __init__(self, seed: int, count: int = 22):
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
    crest: np.ndarray | None = None  # (h, w) -1..1: where the held orb's small waves crest (its points of light flash
    # there: glitter)


def _refract(ix, iy, iz, nx, ny, nz, eta):
    """Snell: unit incident ray i, unit normal n facing against it, ratio n1 / n2; (refracted ray, how far from
    being turned back it is: the discriminant, under 0 where it is)."""
    cosi = -(ix * nx + iy * ny + iz * nz)
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    a = eta * cosi - np.sqrt(np.maximum(k, 0.0))
    return (eta * ix + a * nx, eta * iy + a * ny, eta * iz + a * nz), k


def _exit_surface(tx, ty, tz, X, Y, front_height, back_height, x0, y0):
    """Intersect an internal ray with the far height surface. A front-thickness step overshoots a curved back,
    stretching the fingers into pointed lobes; a bounded search retains their silhouette through the liquid."""
    lo = np.zeros_like(front_height)
    hi = (front_height + float(back_height.max()) + 1.0) / np.maximum(-tz, 0.1)
    for _ in range(9):
        path = 0.5 * (lo + hi)
        back = _remap(back_height, X + tx * path - x0, Y + ty * path - y0)
        inside = front_height + tz * path + back > 0.0
        lo = np.where(inside, path, lo)
        hi = np.where(inside, hi, path)
    path = 0.5 * (lo + hi)
    return path, X + tx * path, Y + ty * path


def shade(fld: Field, sc: Scene, k: int, hand: Hand, room: Room, noise: Noise, pyr: list, matte: np.ndarray | None,
          code: float, seed: int, falling: bool = False) -> Layer | None:
    """The water of ``fld`` as seen on frame ``k``: traced through to the plate (``pyr``, the frame's pyramid) and the
    hand (``matte`` uint8, or None), reflecting the room, glinting, froth and grain; hidden where the hand is in front
    of it; blurred along its motion over the exposure. ``falling`` follows the ribbon's descending centerline in both
    surface normals. Pixels only where it covers some of the picture."""
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
    big0 = _ss(0.1 * s, 0.25 * s, sig[ys, xs] / np.float32(SIG_K)).astype(np.float32)  # a body of water (the orb)...
    held_liquid = sc.held is not None and sc.held[3] > 0.18 * s and sc.streak is None
    # A held surface fragments the reflections while its low modes retain the broad flowing outline.
    skin_floor = ORB_SKIN[0] + (0.10 if held_liquid else 0.0)
    skin0 = 1.0 - big0 * (1.0 - skin_floor) * (1.0 - _ss(0.0, ORB_SKIN[1], thick0)).astype(np.float32)  # ... its
    # waves dying down at its outline, which wobbles in its lobes alone
    wave_height = sc.wave * (0.85 if held_liquid else 1.0)
    wave_size = (0.24 if held_liquid else WAVE_SIZE) * s
    bump = wave_height * at_edge * skin0 * calm0 * noise.flow(0, X, Y, vx[ys, xs], vy[ys, xs], k, wave_size)
    sw0 = fld.sw[ys, xs]
    on_s = np.flatnonzero(sw0 > 0.02)
    if on_s.size:  # a stream's surface is broken by ripples running out along it
        sa0, sx0 = fld.sa[ys, xs][on_s], fld.sx[ys, xs][on_s]
        root_calm = 1.0 if falling else _ss(ROOT_CALM[0] * s, ROOT_CALM[1] * s, sa0)
        bump[on_s] += (sw0[on_s] * at_edge[on_s] * root_calm
                       * (RIPPLE[0] * noise.along(6, sa0, sx0, k, RIPPLE[1] * s, RIPPLE[2], WHIP_FLOW * s / FPS)
                          + RIPPLE[3][0] * _ss(0.3, 0.8, np.abs(sx0))
                          * noise.along(20, sa0, sx0, k, RIPPLE[3][1] * s, RIPPLE[3][2], WHIP_FLOW * s / FPS)))
    if sc.rings or sc.trains:
        hz = np.sqrt(2.0 * sig[ys, xs] ** 2 * np.maximum(lf0, 0.0))  # the front surface's height over the middle
    for cx, cy, rr, ux, uy, uz, age, amp, lam, speed in sc.rings:  # a ripple from a hit, running out over the ball
        px, py = X - cx, Y - cy  # from where it struck, measured over its surface (measured flat on the picture, its
        pn = np.sqrt(px * px + py * py + hz * hz) + 1e-6  # rings were a bullseye drawn on the disc)
        d = rr * np.arccos(np.clip((px * ux + py * uy + hz * uz) / pn, -1.0, 1.0)) - speed * age
        bump += (skin0 * amp * np.exp(-age / RING[3]) * np.sin(2 * math.pi * d / lam)
                 * np.exp(-(d / (1.5 * lam)) ** 2))
    crest = np.zeros(X.shape, np.float32)  # where the trains of small waves crest (-1..1)
    if sc.trains:
        total = sum(tr[6] for tr in sc.trains)
        for cx, cy, rr, ux, uy, uz, amp, lam, ran in sc.trains:  # out of a point on it, over its surface
            px, py = X - cx, Y - cy
            pn = np.sqrt(px * px + py * py + hz * hz) + 1e-6
            d = rr * np.arccos(np.clip((px * ux + py * uy + hz * uz) / pn, -1.0, 1.0))
            w_ = (amp * np.sin(2.0 * math.pi * (d - ran) / lam) * np.exp(-d / (RIPPLES[4] * rr))).astype(np.float32)
            bump += skin0 * w_
            crest += w_ / max(total, 1e-9)
        crest = np.clip(crest * 2.5, -1.0, 1.0)
    broad = lf.copy()  # its broad shape, its waves and ripples aside (the held orb's window glint is mirrored in it)
    lf[ys, xs] += bump.astype(np.float32)
    crest2 = np.zeros(lf.shape, np.float32)
    crest2[ys, xs] = crest
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
    rest = np.zeros(th.shape, np.float32)  # how much of a stream lies on the skin
    if matte is not None:  # a stream lying on the skin is flattened against it, a rivulet
        rest = (_resting(hand, k, matte, (by0, bx0), fld.zc[crop], Rmap, s, fld.li[crop])
                * _ss(0.6, 0.9, th)).astype(np.float32)
        th = th * (1.0 - (1.0 - REST[2]) * rest)
    h2 = (sig[crop] * np.float32(math.sqrt(2.0 * LN_TAU)) * th
          * np.clip(lf[crop] / np.float32(LN_TAU), 0.0, None) ** (1.0 - 0.5 * th)).astype(np.float32)
    # The front and back of a falling ribbon share its descending centerline, rather than a screen-parallel slab.
    descent = fld.zc[crop] - sc.ribbon[0].z if falling else 0.0
    hb = cv2.GaussianBlur(h2 + descent, (0, 0), 0.8)
    gy, gx = np.gradient(hb)
    if matte is not None and float(th.min()) < 0.5:
        # A shallow receiving sheet lies on the tilted palm. Its surface normal must include that plane;
        # a screen-parallel film misses the room light and vanishes as soon as the raised body drains.
        yy, xx = np.mgrid[by0:by0 + h2.shape[0], bx0:bx0 + h2.shape[1]].astype(np.float32)
        on_palm = (_ss(0.5, 0.8, 1.0 - th) * matte[by0:by0 + h2.shape[0], bx0:bx0 + h2.shape[1]] / 255.0
                   * _ss(0.5 * s, 0.1 * s, np.abs(fld.zc[crop] - hand.depth(k, xx, yy, fingers=False))))
        gx += on_palm * hand.plane[k, 0]
        gy += on_palm * hand.plane[k, 1]
    gx, gy = np.clip(gx, -6, 6), np.clip(gy, -6, 6)
    hxx = cv2.GaussianBlur(np.gradient(gx, axis=1), (0, 0), 1.5)
    hyy = cv2.GaussianBlur(np.gradient(gy, axis=0), (0, 0), 1.5)
    hxy = cv2.GaussianBlur(np.gradient(gx, axis=0), (0, 0), 1.5)
    if np.any(descent):
        back_y, back_x = np.gradient(cv2.GaussianBlur(h2 - descent, (0, 0), 0.8))
        back_x, back_y = np.clip(back_x, -6, 6), np.clip(back_y, -6, 6)
    else:
        back_x, back_y = gx, gy
    if held_liquid:
        # The thick reservoir transmits one coherent lens. Capillary slopes remain in its reflected light, while
        # the fingers seen through its body follow the broad flowing surface rather than many small glass facets.
        lens_height = (sig[crop] * np.float32(math.sqrt(2.0 * LN_TAU)) * th
                       * np.clip(broad[crop] / np.float32(LN_TAU), 0.0, None) ** (1.0 - 0.5 * th))
        lens_height *= 0.65  # the held and pouring lenses share depth, without folding finger roots into a pink cap
        back_y, back_x = np.gradient(cv2.GaussianBlur(lens_height, (0, 0), max(1.0, 0.025 * s)))
        back_x, back_y = np.clip(back_x, -6, 6), np.clip(back_y, -6, 6)
    nd = np.sqrt(back_x * back_x + back_y * back_y + 1.0)
    dnx, dny, dnz = -back_x / nd, -back_y / nd, 1.0 / nd
    a_ = act[crop]
    iy, ix = np.nonzero(a_)
    X, Y = (ix + bx0).astype(np.float32), (iy + by0).astype(np.float32)
    h = h2[iy, ix]
    rest_p = rest[iy, ix]
    R = Rmap[iy, ix]
    big = _ss(0.1 * s, 0.25 * s, R).astype(np.float32)  # 0 for a stream or a drop, 1 for the orb
    tiny = (1.0 - _ss(TINY[0] * s, TINY[1] * s, R)).astype(np.float32)  # a small drop: see-through, a glint, an edge
    zc = fld.zc[crop][iy, ix]
    crest = crest2[crop][iy, ix]
    fvx, fvy = vx[crop][iy, ix], vy[crop][iy, ix]
    foam = fld.fo[crop][iy, ix]
    sw = fld.sw[crop][iy, ix]
    rivulet = rest_p * np.clip(sw / 0.5, 0.0, 1.0)  # a stream lying on the skin (the orb as it gathers over the palm
    # is none)
    st = np.flatnonzero(sw > 0.02)
    sa, sxs = fld.sa[crop][iy, ix][st], fld.sx[crop][iy, ix][st]
    mxc, myc = fld.mx[crop], fld.my[crop]
    fast = (np.hypot(mxc[iy, ix], myc[iy, ix]) / (SHUTTER * max(s, 1.0))).astype(np.float32)  # hand sizes a frame
    flat = np.clip(1.0 - th[iy, ix], 0.0, 1.0)  # water lying flat on something...
    pool = _ss(0.5, 0.8, flat)  # ... a puddle on the skin
    fine_calm = ORB_SKIN[2] - (0.30 if held_liquid else 0.0)
    thick = np.clip(h / np.maximum(0.2 * R, 1.0), 0.0, 1.0) * (1.0 - FLAT_CALM * flat) * (1.0 - fine_calm * big)
    sx = SPARKLE * thick * noise.flow(3, X, Y, fvx, fvy, k, SPARKLE_SIZE * s)
    sy = SPARKLE * thick * noise.flow(4, X, Y, fvx, fvy, k, SPARKLE_SIZE * s)
    gxa, gya = gx[iy, ix] + sx, gy[iy, ix] + sy  # the surface the light glints off, its finest waves and all...
    na = np.sqrt(gxa * gxa + gya * gya + 1.0)
    nx, ny, nz = -gxa / na, -gya / na, 1.0 / na
    # Fine capillary slopes break reflected highlights but bend the transmitted room only weakly in a held body.
    bends = (SPARKLE_BENDS if held_liquid else SPARKLE_BENDS + 0.65) * big
    lens_x, lens_y = (back_x, back_y) if held_liquid else (gx, gy)
    gxr, gyr = lens_x[iy, ix] + bends * sx, lens_y[iy, ix] + bends * sy
    nr = np.sqrt(gxr * gxr + gyr * gyr + 1.0)

    # in through the front, across, out through the back (whose normal is the front's mirrored), on to what is behind
    eta = ETA[1]
    (tx, ty, tz), _ = _refract(0.0, 0.0, -1.0, -gxr / nr, -gyr / nr, 1.0 / nr, 1.0 / eta)
    path = 2.0 * h / np.maximum(-tz, 0.25)
    ex, ey = X + tx * path, Y + ty * path
    if held_liquid:
        path, ex, ey = _exit_surface(tx, ty, tz, X, Y, lens_height[iy, ix], lens_height, bx0, by0)
    exit_path = path.copy() if held_liquid else None
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
    calm = (1.0 / ((1.0 + (1.0 - big) * fast / CALM[0]) * (1.0 + (1.0 - big) * (curv / CALM[1]) ** 2))
            * (1.0 - REST[3] * np.maximum(rivulet, pool))).astype(np.float32)  # (the thrown orb bends as much as the
    # held one: calmed by its speed, it was a hollow outline filled with the wall)
    tanx = (np.clip(ox / np.maximum(-oz, 0.3), -3, 3) + swirl * inner) * calm
    tany = (np.clip(oy / np.maximum(-oz, 0.3), -3, 3) + swirl * noise.flow(2, X, Y, fvx, fvy, k, INNER_SIZE * s)) * calm

    vis = np.ones(X.shape, np.float32)
    dn = np.full(X.shape, NEAR_OPEN * s, np.float32)
    if matte is not None:
        mh = _remap(matte, X, Y).astype(np.float32) / 255.0
        skin_depth = hand.depth(k, X, Y)
        contact = pool * _ss(0.5 * s, 0.1 * s, np.abs(zc - skin_depth))  # only nearby films belong to the hand;
        # a flat splash on a distant nonplain wall must remain at that wall's depth
        vis *= 1.0 - contact * (1.0 - mh)
        on = np.flatnonzero(mh > 0.01)
        if on.size:
            zh = skin_depth[on]
            zc[on] += contact[on] * (zh + 0.5 * h[on] - zc[on])  # conform a thin puddle to the tilted palm; a constant
            # centre depth intersected the hand as a straight-edged glass pedestal
            zf, zb, m_ = zc[on] + h[on], zc[on] - h[on], mh[on]
            vis[on] *= 1.0 - m_ * np.clip((zh - zf) / (DEPTH_SOFT * s) + 0.5, 0.0, 1.0)  # the hand in front hides it
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
    dfar = dfar * (1.0 - (1.0 - FAR_FAST) * (1.0 - big) * _ss(0.5, 1.0, fast * FPS / WHIP_SHED)).astype(np.float32)
    seen = np.float32(TINY_SEES) * R  # a small drop shows what is a few of its radii behind it, turned over (sent a
    dfar = dfar + tiny * (seen - dfar)  # room away, it showed the room shrunk into a grey bead: metal)
    dn = dn + tiny * (np.minimum(dn, seen) - dn)
    if float(pool.max()) > 0.0:  # a puddle bends what is under it POOL_BEND px at most (bent 10 px, a gel lens)
        lim = np.minimum(1.0, POOL_BEND / np.maximum(np.hypot(tanx, tany) * np.maximum(dfar, dn), 1e-3))
        tanx, tany = tanx * (1.0 - pool * (1.0 - lim)), tany * (1.0 - pool * (1.0 - lim))
    near_valid = 1.0
    if held_liquid and matte is not None:
        # Intersect the outgoing ray with the tilted hand, rather than borrowing the depth at its entry pixel.
        # The latter projects fingers that lie in front of the exit as a broad inverted pink cap.
        a, b, _ = hand.plane[k]
        exit_z = zc + lens_height[iy, ix] + tz * exit_path
        separation = exit_z - hand.depth(k, ex, ey)
        tilt = 1.0 + a * tanx + b * tany
        near_valid = np.maximum(pool, ((separation > 0.0) & (tilt > 0.1)).astype(np.float32))
        # The extended palm plane is only a local depth estimate: its steep tilt otherwise sends a ray onto the
        # wrist and magnifies it into a pink cap. Keep the near hand close to the back of the held liquid lens.
        hand_distance = np.clip(separation / np.maximum(tilt, 0.1), NEAR[0] * s, 0.18 * s).astype(np.float32)
        dn = dn * pool + hand_distance * (1.0 - pool)
    hxx_, hxy_, hyy_ = hxx[iy, ix] * calm, hxy[iy, ix] * calm, hyy[iy, ix] * calm
    qx, qy = ex + tanx * dn, ey + tany * dn
    if held_liquid and matte is not None:
        # During contact the hand can meet the ray inside the water, before its back surface. Trace that internal
        # segment to the skin instead of showing a wall-coloured hole through the receiving palm.
        a, b, _ = hand.plane[k]
        entry_z = zc + lens_height[iy, ix]
        denominator = -tz + a * tx + b * ty
        internal_path = (entry_z - hand.depth(k, X, Y)) / np.maximum(denominator, 0.1)
        hx, hy = X + tx * internal_path, Y + ty * internal_path
        inside = ((internal_path > 0.0) & (internal_path < exit_path) & (denominator > 0.1)).astype(np.float32)
        inside *= _remap(matte, hx, hy).astype(np.float32) / 255.0
        qx += inside * (hx - qx)
        qy += inside * (hy - qy)
        near_valid = np.maximum(near_valid, inside)
    sig_n = np.maximum(_footprint(dn, hxx_, hxy_, hyy_), _spread(qx, qy, iy, ix, a_.shape)) + BASE_BLUR * R
    sig_n *= 1.0 - 0.30 * big
    if held_liquid:
        # The ray map supplies its own antialiasing footprint. A curvature-only defocus estimate diverges near a
        # focus and turns recognisable fingers into a broad pink veil.
        sig_n = _spread(qx, qy, iy, ix, a_.shape) + np.maximum(0.65, 0.004 * R)
    near = np.stack([_lod(pyr, c, qx, qy, sig_n) for c in range(3)], -1)
    dispersion = 0.25 if held_liquid else 1.0
    disp = tuple(dfar * np.float32(1.0 + dispersion * (v - eta) / (eta - 1.0)) for v in ETA)
    sig_f = np.maximum(_footprint(dfar, hxx_, hxy_, hyy_), _spread(ex + tanx * dfar, ey + tany * dfar, iy, ix,
                                                                  a_.shape))
    sig_f *= 1.0 - 0.30 * big
    if held_liquid:
        sig_f = _spread(ex + tanx * dfar, ey + tany * dfar, iy, ix, a_.shape) + 0.010 * R
    far_pyr = pyr
    far_scale = 1.0
    if held_liquid and matte is not None:
        # Foreground fingers cannot also be the distant wall. Fill their missing background at room-detail scale;
        # the near ray still reads the original fingers, with their own depth and full colour.
        distant = np.stack([pyr[2][c] for c in range(3)], -1)
        hidden = cv2.resize(matte, (distant.shape[1], distant.shape[0]), interpolation=cv2.INTER_AREA)
        hidden = cv2.dilate(np.uint8(hidden > 16), np.ones((3, 3), np.uint8)) * 255
        distant = lin(cv2.inpaint(gam(distant), hidden, 3.0, cv2.INPAINT_TELEA))
        far_pyr = _pyramid(distant)
        far_scale = 4.0
    far = np.stack([_lod(far_pyr, c, (ex + tanx * disp[c] + 0.5) / far_scale - 0.5,
                        (ey + tany * disp[c] + 0.5) / far_scale - 0.5, sig_f / far_scale) for c in range(3)], -1)
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
        m *= np.asarray(near_valid)[..., None]
        col = m * near + (1.0 - m) * far
    else:
        col = far
    # Near-normal rays keep a clear view; moving thin facets interrupt the otherwise continuous glass-like lip.
    direct = np.stack([_remap(pyr[0][c], X, Y) for c in range(3)], -1)
    rim_facets = _ss(*((-0.05, 0.45) if held_liquid else (-0.30, 0.40)), crest + 0.35 * inner)
    liquid_arcs = HELD_RIM[2] + (1.0 - HELD_RIM[2]) * rim_facets if held_liquid else 1.0
    clear_face = (0.04 if held_liquid else 0.35) * nz ** 4
    clear = big * (1.0 - pool) * (clear_face + (0.12 if held_liquid else 0.65)
                                * _ss(0.45, 0.95, 1.0 - nz) * (1.0 - rim_facets))
    clear = np.minimum(clear, 0.90)  # overlapping clear facets cannot amplify the transmitted contrast
    col += clear[:, None] * (direct - col)
    # the rim's darkening and the dimming where rays are turned back belong to a body of water: a small drop is all
    # rim, and with them it was a grey metal sliver; it keeps a thin edge and a darker core round its glint; a rivulet
    # has no rim
    small = np.clip(R / (SMALL_RIM * s), 0.0, 1.0)
    edge = (SMALL_EDGE + (1.0 - SMALL_EDGE) * big) * small * (1.0 - np.maximum(rivulet, pool))  # (a puddle has none:
    # with a ball's dark rim, the pool on the palm was a contact lens)
    oxy = np.maximum(np.hypot(ox, oy), 1e-3)  # a ray turned far off its way sees the room round the water (near a
    rnd = (edge * _ss(ROUND[0], ROUND[1], oxy)).astype(np.float32)  # ball's rim, turned down: the floor under its top)
    rnd *= 1.0 - 0.75 * big  # grazing rays retain the transmitted room instead of forming a continuous metal lip
    rnd *= liquid_arcs  # the grazing room reflection opens between the held surface's moving facets
    col += rnd[:, None] * (room.reflect(k, ox / oxy, oy / oxy) - col)
    col *= np.exp(-ABSORB[None, :] * (path / max(s, 1.0))[:, None])
    rr = np.clip(1.0 - nz, 0.0, 1.0)
    band = _ss(RIM_BAND[0], RIM_BAND[1], rr).astype(np.float32)
    Lk = room.key(k, ((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    lit = room.lit(k)
    lxy = math.hypot(float(Lk[0]), float(Lk[1])) + 1e-6
    face = (nx * Lk[0] + ny * Lk[1]) / lxy / np.maximum(np.hypot(nx, ny), 1e-3)  # its slope towards the light, -1..1
    away = (RIM_FAR + (1.0 - RIM_FAR) * _ss(-0.3, 0.7, -face)).astype(np.float32)
    rim_density = 0.25 if held_liquid else 1.0
    col *= (1.0 - rim_density * edge * RIM_DARK * band * away * np.clip(0.75 + 0.5 * inner, 0.25, 1.25))[:, None]
    col *= (1.0 - rim_density * edge * (1.0 - TIR_DIM) * (1.0 - 0.65 * big) * (1.0 - out) * band)[:, None]
    col *= (1.0 - tiny * TINY_EDGE * _ss(0.55, 0.95, rr))[:, None]
    # the room reflected (a small piece of water is a weak mirror: at full strength a splash's thin pieces were
    # chrome), read by the way the surface faces: up at the top of the rim, down at its foot (read by the mirrored
    # ray, which goes straight on past a grazing rim, the whole rim mirrored one patch of the room: an even ring)
    F = (F0 + (1.0 - F0) * (1.0 - nz) ** 5).astype(np.float32)
    F = np.minimum(F, F_SMALL + (1.0 - F_SMALL) * big) * (1.0 - TINY_MIRROR * tiny)
    F *= 1.0 + RIM_BREAK * _ss(0.3, 0.6, rr) * np.clip(inner + crest, -1.0, 1.0)  # the rim's mirror broken into arcs
    F *= 1.0 - 0.4 * big
    if held_liquid:
        F *= 1.0 - _ss(0.3, 0.6, rr) * (1.0 - liquid_arcs)
        F *= 0.65
    # by the waves running round it (its finest waves die out at the outline, and it was an even ring)
    flow_a = WHIP_FLOW * s / FPS
    whip = (_ss(0.4, 1.0, fast * FPS / WHIP_SHED) * np.clip(sw / 0.5, 0.0, 1.0)).astype(np.float32)  # a lashing stream
    if st.size:  # along a stream its glints and the room it mirrors break up with the ripples running down it (one
        # unbroken streak down its length was an acrylic rod's, a neon tube's), but hardly while it lashes (each bright
        # bit, smeared across the stream by its speed, was a bar: a ladder's rungs, a zebra ribbon)
        gm = np.clip(1.0 + GLINT_BREAK[0] * noise.along(7, sa, sxs, k, GLINT_BREAK[1] * s, GLINT_BREAK[3], flow_a),
                     0.0, 2.0)
        root_calm = 1.0 if falling else _ss(ROOT_CALM[0] * s, ROOT_CALM[1] * s, sa)
        gm = 1.0 + (gm - 1.0) * (1.0 - GLINT_BREAK[2] * whip[st]) * root_calm
        F[st] *= 1.0 + 0.5 * sw[st] * (gm - 1.0)
        n_seg = noise.along(13, sa, sxs, k, GLINT_SEG[0] * s, 3.0, flow_a)
        seg = _ss(GLINT_SEG[1], GLINT_SEG[2], n_seg)
        seg += rivulet[st] * (_ss(RIVULET[3], RIVULET[3] + 0.4, n_seg) - seg)  # lying on the skin, a few short glints
        # (lit along most of its edge, a stream on the palm was a glass rim)
    if st.size and float(rivulet.max()) > 0.0:  # a rivulet mirrors the room at a grazing angle along its edge towards
        # the light alone, in lengths (mirrored along both its edges, the stream was a glass tube's closed outline)
        lit_edge = np.zeros(X.shape, np.float32)
        lit_edge[st] = seg * _ss(0.15, 0.7, face[st])
        F = F0 + (F - F0) * (1.0 - rivulet * (1.0 - lit_edge))
    ev = room.reflect(k, nx, ny)
    col = col * (1.0 - F)[:, None] + ev * F[:, None]
    sdp = sd[crop][iy, ix]  # water lying flat: a bright line round its edge towards the light, where its surface curls
    # down to meet what it lies on, and a dark line where it meets it on the other side (bright all round, a tube)
    facing = _ss(0.1, 0.7, face)  # (on a puddle, only along a short, broken arc facing the light)
    facing += pool * (_ss(POOL_EDGE[0], POOL_EDGE[1], face)
                      * _ss(-0.2, 0.6, noise.flow(13, X, Y, fvx, fvy, k, POOL_EDGE[2] * s)) - facing)
    men = MENISCUS[0] * flat * np.exp(-((sdp - MENISCUS[1]) / MENISCUS[2]) ** 2) * facing
    if sc.coat is not None:
        men *= 1.0 - 0.75 * pool
    if st.size:  # (along a stream, in lengths: lit all along, a tube)
        men[st] *= 1.0 - sw[st] * (1.0 - seg)
    film = np.maximum(rest_p, pool)  # (lying on the skin, no dark lines round it but with its bright one: all round,
    # they drew the outline of a clear glass hook on the palm, the foot of a glass vase)
    col *= (1.0 - MENISCUS[3] * flat * np.exp(-((sdp - MENISCUS[4]) / 1.5) ** 2)
            * (1.0 - np.maximum(rivulet, pool) * _ss(0.4, -0.2, face)) * (1.0 - pool)
            - MENISCUS[5] * flat * np.exp(-((sdp - 0.8) / 0.9) ** 2) * _ss(-0.1, 0.7, -face) * (1.0 - film))[:, None]
    col += men[:, None] * (0.6 * ev + 0.4 * lit[None, :])
    face_s = np.zeros(X.shape, np.float32)  # its broad shape's slope towards the light, -1..1 (its waves aside)
    if st.size:  # a stream's edges: darker bands, the room round it bent in there, most away from the light, as a rod
        # of water's are (bright lines down both its edges, it was a glass tube); none where it lies on the skin (a
        # rivulet's edge there only thins into the skin: lined on its shadow side, it was a glass rod)
        g_ = np.maximum(np.hypot(gx[iy, ix], gy[iy, ix]), 1e-4)
        face_s = (-(gx[iy, ix] * Lk[0] + gy[iy, ix] * Lk[1]) / (lxy * g_)).astype(np.float32)
        if falling:
            tangent = _unit(np.array([sc.ribbon[-1].x - sc.ribbon[0].x,
                                      sc.ribbon[-1].y - sc.ribbon[0].y]))
            # The room's side light follows the stream's round cross section, independent of its descent in depth.
            face_s[st] = (-tangent[1] * Lk[0] + tangent[0] * Lk[1]) / lxy * np.sign(sxs)
        away_ = EDGE_SHADE[2] + (1.0 - EDGE_SHADE[2]) * _ss(-0.2, 0.4, -face_s[st])
        col[st] *= (1.0 - EDGE_SHADE[0] * sw[st] * _ss(*EDGE_SHADE[1], np.abs(sxs)) * away_ * (1.0 - rest_p[st])
                    * (1.0 - rivulet[st]))[:, None]
    # the key light: its window mirrored in the surface, waves and all, a soft sheen round it, the window again off the
    # inside of the far wall (opposite, smaller, dimmer), the light the water gathers onto its far inside edge, and a
    # faint moving net inside it
    gxw, gyw = gx[iy, ix] + WINDOW[3] * sx, gy[iy, ix] + WINDOW[3] * sy  # the window is mirrored in the surface's
    # wobble and waves, and only a little of its finest waves
    # Fine slopes break the white reflection into moving facets; the transmitted room keeps its broad waves.
    core_sigma = GLINT_CORE[4] * s * (0.30 if sc.held is not None else 1.0)
    gys, gxs = np.gradient(cv2.GaussianBlur(h2 + descent, (0, 0), max(1.0, core_sigma)))
    slope_share = GLINT_CORE[2] * (1.0 - big) + 0.65 * big
    gxc = np.clip(gxs, -6, 6)[iy, ix] + slope_share * sx
    gyc = np.clip(gys, -6, 6)[iy, ix] + slope_share * sy
    if float(big.max()) > 0.01:
        hbr = (sig[crop] * np.float32(math.sqrt(2.0 * LN_TAU)) * th
               * np.clip(broad[crop] / np.float32(LN_TAU), 0.0, None) ** (1.0 - 0.5 * th)).astype(np.float32)
        gyb, gxb = np.gradient(cv2.GaussianBlur(hbr + descent, (0, 0), max(1.0, GLINT_CORE[4] * s)))
        gxb, gyb = np.clip(gxb, -6, 6)[iy, ix], np.clip(gyb, -6, 6)[iy, ix]
        gxc += GLINT_HOLD * big * (gxb - gxc)
        gyc += GLINT_HOLD * big * (gyb - gyc)
        gxw += WINDOW[5] * big * (gxb - gxw)
        gyw += WINDOW[5] * big * (gyb - gyw)
    nw = np.sqrt(gxw * gxw + gyw * gyw + 1.0)
    spec = GLINT[0] * (1.0 - 0.25 * big) * _window(-gxw / nw, -gyw / nw, 1.0 / nw, Lk, 0.80 if held_liquid else 1.0)
    if sc.coat is not None:
        # The shallow junction between the spreading sheet and a run must not mirror one long window stripe.
        spec *= 1.0 - 0.75 * pool
    nc = np.sqrt(gxc * gxc + gyc * gyc + 1.0)
    # A saddle spreads the light into two weaker reflections; the orb retains the window's frame bars.
    saddle = _ss(-0.02, -0.3, (hxx * hyy - hxy * hxy)[iy, ix] * R * R)
    spec += (GLINT_CORE[0] * (1.0 + 0.30 * big) / max(float(room.white[k]), 0.3)
             * (1.0 - GLINT_CORE[3] * saddle)
             * _window(-gxc / nc, -gyc / nc, 1.0 / nc, Lk, GLINT_CORE[1] * (0.80 if held_liquid else 1.0), big))
    spec += (BACK[0] * big * deep) * _window(gxw / nw, gyw / nw, 1.0 / nw, Lk, BACK[1])
    hv = Lk + np.array([0.0, 0.0, 1.0], np.float32)
    hv /= np.linalg.norm(hv)
    ndh = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0.0, 1.0)
    spec += GLINT[1] * ndh ** GLINT_POW
    if held_liquid:
        spec *= 0.50  # shallow liquid facets mirror smaller broken pieces of the broad room light
    spec += (TINY_CORE * tiny * (1.0 - _ss(0.1, 0.6, rr))  # a small drop's core, lit by the light it gathers, in the
             * _ss(0.7, 0.95, th[iy, ix]))  # air (a drop lying on the palm lit so, a white pearl)
    if st.size:
        spec[st] *= 1.0 + sw[st] * (gm - 1.0)
    # a fast stream only (the thrown orb at its speed went milky): froth in torn bits, and its edges catching the key
    # light as a cylinder's do
    spec *= 1.0 - GLINT_FAST * whip  # (its broad glints, smeared by its speed, were a slab of white streaks)
    if sc.coat is not None:
        # The skin film supplies its own crease reflections; a pool's broad window would paint the palm white.
        spec *= 1.0 - 0.65 * pool
    if st.size:  # a thin bright line along its edge towards the light, all along it in the air and in lengths lying on
        # the skin, and fast, a dimmer one along the other (where the light grazed both edges, lit all along, they were
        # a glass tube's; broken into lengths in the air, cellophane's folds)
        wl = np.maximum(GLINT_LINES[2], 0.9 / np.maximum(R[st], 1.0))
        lines = ((GLINT_LINES[0] * (1.0 - RIVULET[7] * rivulet[st]) * _ss(0.1, 0.5, face_s[st])
                  * (1.0 + rivulet[st] * (seg - 1.0))
                  + GLINT_LINES[1] * whip[st] * (1.0 - rivulet[st]) * _ss(-0.1, 0.5, -face_s[st]))
                 * np.exp(-((np.abs(sxs) - GLINT_LINES[3]) / wl) ** 2))
        if falling:
            # A room reflection catches the falling sheet's lit side in moving lengths, with clear skin between.
            lines = (0.35 * GLINT_LINES[0] * _ss(0.02, 0.22, face_s[st]) * (0.25 + 0.75 * seg)
                     * np.exp(-((np.abs(sxs) - GLINT_LINES[3]) / wl) ** 2))
        spec[st] += sw[st] * lines * np.clip(gm, 0.3, 2.0)
    gsy, gsx = np.gradient(cv2.GaussianBlur(np.clip(sd[crop], -20.0, None).astype(np.float32), (0, 0), 2.0))
    gsx, gsy = gsx[iy, ix], gsy[iy, ix]  # its outline's way out (laid off its waves' slopes, the crescent was frilled)
    out_face = -(gsx * Lk[0] + gsy * Lk[1]) / lxy / np.maximum(np.hypot(gsx, gsy), 1e-4)
    ins = sdp / np.maximum(R, 1.0)  # how far in from its outline, of its radius
    cres = (CRESCENT[0] * (CRESCENT[3] + (1.0 - CRESCENT[3]) * big) * _ss(*CRESCENT[1], ins)
            * (1.0 - _ss(*CRESCENT[2], ins)) * _ss(0.2, 0.9, -out_face) * np.clip(0.9 + 0.2 * inner, 0.6, 1.1)
            * np.clip(1.0 + CREST_LIGHT[0] * crest, 0.6, 1.4) * (1.0 - rivulet))  # (a rivulet gathers none onto its
    # far edge; it flickers with the water inside, and the small waves running over it brighten and dim it along its
    # length)
    if held_liquid:
        cres *= 0.20 * (0.35 + 0.65 * rim_facets)  # a faint far-side arc must not form a second glass collar
        cres += (0.18 * np.exp(-((ins - 0.035) / 0.016) ** 2) * _ss(0.25, 0.85, -out_face)
                 * (0.30 + 0.70 * rim_facets) * (1.0 - pool))  # a narrow broken lower meniscus gives clear water weight
    rim_gain, rim_band = (HELD_RIM[:2] if held_liquid else RIM_LIT)
    rim_light = (rim_gain * edge * _ss(*rim_band, rr) * _ss(0.0, 0.85, face)
                 * np.clip(1.0 + CREST_LIGHT[1] * crest, 0.6, 1.4))
    if held_liquid:
        rim_light *= liquid_arcs
    cres += rim_light  # the rim towards the light stays thin and opens between the moving facets
    if sc.coat is not None:
        cres *= 1.0 - 0.85 * _ss(0.10, 0.45, flat)
    net = INNER_GLOW * np.exp(-(inner / 0.22) ** 2) * deep * (1.0 - F)
    rim_open = 0.75 * big * (1.0 - pool) * _ss(0.35, 0.90, rr) * (1.0 - rim_facets)
    if held_liquid:
        rim_open = 0.18 * big * (1.0 - pool) * _ss(0.40, 0.92, rr) * (1.0 - 0.65 * rim_facets)
    col += rim_open[:, None] * (direct - col)
    col += (spec + cres + net).astype(np.float32)[:, None] * lit[None, :]
    if held_liquid:
        # Sparse entrained microbubbles reveal moving water inside the reservoir.
        # Scatter existing room light through its thick interior, leaving the
        # thin edge and most of the transmitted hand clear.
        churn = noise.flow(12, X, Y, fvx, fvy, k, .22*s)
        suspended = (.12 + .20*np.maximum(churn, 0)) * deep * (1-pool)
        scatter = lit[None,:] * np.array([.35,.72,.90],np.float32)
        col += suspended[:,None] * (scatter-col)
    # froth: sparse entrained air lit by the room, carried in streaks along a stream. The airborne body remains clear
    fn = 0.55 + 0.45 * noise.flow(5, X, Y, fvx, fvy, k, FOAM_SIZE * s)
    fn += big * (1.0 - fn)
    if st.size:
        streak = FOAM_STREAK[2] + 1.6 * np.maximum(noise.along(8, sa, sxs, k, FOAM_STREAK[0] * s, FOAM_STREAK[1],
                                                               flow_a), 0.0)
        fn[st] += sw[st] * _ss(ROOT_CALM[0] * s, ROOT_CALM[1] * s, sa) * (streak * fn[st] - fn[st])
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
    fm *= 1.0 - THROWN_FOAM[1] * big * (1.0 - _ss(0.2, 0.65, deep))  # a body's froth is in its core, its rim clear
    fm = np.minimum(fm, FOAM_MAX - (FOAM_MAX - FOAM_LEAD[1]) * _ss(0.0, 0.5, front))
    if st.size:  # held, its froth lies in white threads along its outer edge, its core clearer
        held = (sw[st] * (1.0 - whip[st]) * (1.0 - rivulet[st]) * _ss(ROOT_CALM[0] * s, ROOT_CALM[1] * s, sa))
        band = _ss(*WHIP_AERATE[1], np.abs(sxs))
        threads = _ss(*WHIP_AERATE[3], noise.along(21, sa, sxs, k, WHIP_AERATE[2][0] * s, WHIP_AERATE[2][1], flow_a))
        fm[st] = np.maximum(fm[st] * (1.0 - held * (1.0 - WHIP_AERATE[0]) * (1.0 - band)),
                            held * WHIP_AERATE[4] * FOAM_MAX * band * threads)
        gap = WHIP_FLOW * WHIP_PULSE[2]
        phase = ((sa / s - WHIP_FLOW * k / FPS + 0.5 * gap) % gap) - 0.5 * gap
        air = np.exp(-(phase / 0.16) ** 2)  # a broad moving pocket of entrained air occupies a short part of the stream
        fm[st] = np.maximum(fm[st], 0.20 * held * air * (0.4 + 0.6 * band))
    glint = (FOAM_GLINT[0] * (1.0 - big)
             * np.exp(-(noise.flow(9, X, Y, fvx, fvy, k, FOAM_GLINT[1] * FOAM_SIZE * s) / 0.3) ** 2))
    froth = (0.75 + 0.25 * nz) * FOAM_ALBEDO * (1.0 + FOAM_FAST * whip) + glint
    col = col * (1.0 - fm)[:, None] + (fm * froth)[:, None] * lit[None, :]
    alpha = np.zeros(a_.shape, np.float32)
    feather = 1.0 + film * (np.clip((sdp + 0.5) / np.maximum(RIVULET[2], POOL_FEATHER * pool * R), 0.0, 1.0)
                            - 1.0)  # a rivulet, a puddle, thins into the skin (a puddle's outline drawn hard, it was a
    # glass foot set on the palm)
    alpha[iy, ix] = (cover[crop][iy, ix] * vis * feather
                     * (1.0 - RIVULET[6] * np.maximum(rivulet, pool)))  # (a rivulet, a puddle, is a film: the wet
    # skin shows through it)
    C = np.zeros(alpha.shape + (3,), np.float32)
    C[iy, ix] = col * alpha[iy, ix, None]
    # what keeps some of itself through the blur: its own light (key light units) and its froth's cover
    own = np.zeros(alpha.shape + (2,), np.float32)
    loose = (np.clip(fld.fm[crop][iy, ix] / np.maximum(f[crop][iy, ix], 1e-6), 0.0, 1.0)
             * (1.0 - _ss(DOT[0] * s, 2.5 * DOT[0] * s, R)))  # a drop flying free...
    own[iy, ix, 0] = ((spec + cres) * (1.0 - whip) * (1.0 - np.maximum(tiny, loose)) * (1.0 - fm)
                      + fm * froth) * alpha[iy, ix]  # (a fast stream keeps only its froth: every glint along it kept,
    # each drawn into a line by the motion, it was a fan of hair-thin streaks; ... only its glint, _dots: its body's
    # light kept too, beside its glint's, it was a soft white blob, doubled: sleet)
    own[iy, ix, 1] = fm * alpha[iy, ix]
    rag = np.zeros(alpha.shape, np.float32)
    rag[iy, ix] = whip
    lead = fld.ld[crop]
    _dots(C, fld.dots, (by0, bx0), s, Lk, lit, hand, k, matte, own)
    box, C, alpha = _motion_blur(C, alpha, mxc, myc, (by0, by0 + C.shape[0], bx0, bx0 + C.shape[1]), H, W, own, lit,
                                 rag, k, lead)
    C = cv2.GaussianBlur(C, (0, 0), EDGE_SOFT)
    alpha = cv2.GaussianBlur(alpha, (0, 0), EDGE_SOFT)
    if code > 0:  # the plate's grain, on what the water shows (what is seen through it was read smoothed, its glints
        # are clean); the sensor's, so laid on after the motion blur
        y = np.maximum(C @ fx._LUMA, 1e-6) / np.maximum(alpha, 1e-3)
        step = (2.2 * code) * y ** np.float32(1.2 / 2.2) / y
        g = _rng(seed, _GRAIN, k).standard_normal(y.shape).astype(np.float32)
        C = np.maximum(C * (1.0 + g * step)[..., None], 0.0)
    pad = ((by0 - box[0], box[1] - by0 - h2.shape[0]), (bx0 - box[2], box[3] - bx0 - h2.shape[1]))
    depth = fld.zc[crop].copy()
    depth[iy, ix] = zc  # caustics must use the same conforming contact surface as the visible film
    return Layer(box, C, alpha, np.pad(h2, pad), np.pad(depth, pad), np.pad(Rmap, pad),
                 np.pad(crest2[crop], pad) if sc.trains else None)


def _resting(hand: Hand, k: int, matte: np.ndarray, at: tuple, zc: np.ndarray, R: np.ndarray, s: float,
             told: np.ndarray | None = None) -> np.ndarray:
    """(h, w) 0..1 over a box of the frame whose top left is px ``at``: how much the water there (its middle's depth
    ``zc``, its radius ``R``) lies on the skin (by the matte) rather than running clear of it (REST), or as much as a
    stream says it does (``told``, Field.li: its rivulet, spread on the skin, is read here thinner than it was laid).
    The orb, which floats, never does. The hand's depth is read on a grid 4 px apart."""
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
    over = (zc - zh) / np.maximum(R, 1.0)
    lie = _ss(REST[1], REST[0], over) * _ss(-1.0, 0.0, over) * (1.0 - _ss(0.25 * s, 0.35 * s, R))  # (the stream
    # flared into the pool lies on the skin too: taken off it there, its root was a clear glass hook; swung round
    # behind the hand, it lies on nothing)
    if told is not None:
        lie = np.maximum(lie, told)
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
        sg = min(0.85, max(DOT[2], DOT[3] * r))  # at 1080 a spray glint is at most two pixels wide
        cx, cy = x + 0.35 * r * lxy[0] - at[1], y + 0.35 * r * lxy[1] - at[0]
        e = int(3 * sg + 2)
        ya, yb, xa, xb = max(0, int(cy) - e), min(hh, int(cy) + e + 1), max(0, int(cx) - e), min(ww, int(cx) + e + 1)
        if yb <= ya or xb <= xa:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        pk = DOT[1] * gw * float(_ss(0.5, 1.5, r))  # sub-three-pixel spray fades instead of sparkling white
        g = pk * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sg * sg))
        C[ya:yb, xa:xb] += g[..., None] * lit
        if own is not None:
            own[ya:yb, xa:xb, 0] += (DOT_KEEP[0] + (DOT_KEEP[1] - DOT_KEEP[0]) * float(_ss(*DOT_KEEP[2], r))) * g


def _shutter(t) -> np.ndarray:
    """How open the shutter is at ``t`` (-0.5 .. 0.5) of the exposure: fully over its middle, opening and closing in a
    raised cosine over BLUR_TAPER of it at either end."""
    a = 0.5 * BLUR_TAPER
    edge = np.clip((np.abs(t) - (0.5 - a)) / a, 0.0, 1.0)
    return (0.5 + 0.5 * np.cos(math.pi * edge)).astype(np.float32)


_RAGS = np.random.default_rng(20261020).uniform(-1.0, 1.0, 4096).astype(np.float32)  # the values _ragged runs through
BLUR_CHUNK = 1 << 21  # taps worked at once in _motion_blur (all of a lash's at once held hundreds of MB a frame)


def _ragged(c, size: float) -> np.ndarray:
    """A smooth random value in [-1, 1] at each ``c`` px along a line, its features ``size`` px: value noise off a
    fixed random table, so nothing repeats across a frame."""
    u = np.asarray(c, np.float64) / size
    i = np.floor(u)
    f = (u - i).astype(np.float32)
    i = i.astype(np.int64) % _RAGS.size
    f = f * f * (3.0 - 2.0 * f)
    return _RAGS[i] + (_RAGS[(i + 1) % _RAGS.size] - _RAGS[i]) * f


def _motion_blur(C: np.ndarray, A: np.ndarray, mx: np.ndarray, my: np.ndarray, box: tuple, H: int, W: int,
                 own: np.ndarray | None = None, lit: np.ndarray | None = None, rag: np.ndarray | None = None,
                 k: int = 0, lead: np.ndarray | None = None) -> tuple:
    """The premultiplied colour ``C`` (h, w, 3) and coverage ``A`` (h, w) over ``box`` of the frame, each pixel spread
    along the path (mx, my) px it travels over the exposure, centred on it, as the shutter lets it through (_shutter),
    in taps BLUR_STEP apart, the same for every pixel moving alike: scattered along its own motion, so a fast drop
    smears past its outline into a streak and a slow root stays sharp beside a fast tail.

    ``own`` (h, w, 2) is what keeps some of itself wherever it passes over the exposure (BLUR_KEEP: a fast jet's
    glints and froth smear into streaks, not into nothing): the water's own light in units of the key light ``lit``,
    and its froth's cover. ``rag`` (h, w) 0..1: how torn the water is: its smear is longer or shorter from stream to
    stream (BLUR_RAGGED, drawn afresh on frame ``k``). ``lead`` (h, w) 0..1: how much a pixel's smear is sharp at its
    front and trails behind it (BLUR_LEAD). Returns (the box, grown to hold the smear, C, A)."""
    y0, y1, x0, x1 = box
    sp = np.hypot(mx, my)
    on = (A > 1e-4) | (C.max(-1) > 1e-4)
    if not on.any() or float(sp[on].max()) < 0.5:
        return box, C, A
    iy, ix = np.nonzero(on)
    mxs, mys = mx[iy, ix].astype(np.float32), my[iy, ix].astype(np.float32)
    rg = np.zeros(iy.size, np.float32) if rag is None else rag[iy, ix]
    ld = None if lead is None or float(lead.max()) <= 0.0 else lead[iy, ix].astype(np.float32)
    tr = np.flatnonzero(rg > 0)
    if tr.size:  # each stream of torn water its own length
        spn = np.maximum(sp[iy[tr], ix[tr]], 1e-3)
        across = ((iy[tr] + y0) * mxs[tr] - (ix[tr] + x0) * mys[tr]) / spn
        stretch = 1.0 + BLUR_RAGGED[0] * rg[tr] * _ragged(across + 1013.7 * k, BLUR_RAGGED[1])
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
    peak = np.zeros((2, hh * ww), np.float32)
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
        if ld is not None:  # ... or most of it at its front, trailing behind
            lw = ld[reps]
            win = win + lw * (BLUR_LEAD[0] + (1.0 - BLUR_LEAD[0]) * np.exp((t - 0.5) / BLUR_LEAD[1]) - win)
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
            xn, yn = np.rint(X).astype(np.int64), np.rint(Y).astype(np.int64)
            ok = np.flatnonzero((xn >= 0) & (xn < ww) & (yn >= 0) & (yn < hh))
            at = yn[ok] * ww + xn[ok]
            np.maximum.at(peak[0], at, BLUR_KEEP[0] * win[ok] * v[ok, 4])
            np.maximum.at(peak[1], at, BLUR_KEEP[1] * win[ok] * v[ok, 5])
    acc = acc.reshape(nch, hh, ww).astype(np.float32)
    C, A = np.ascontiguousarray(acc[:3].transpose(1, 2, 0)), acc[3].copy()
    if keep:  # (softened a little: the most taken pixel by pixel combed the froth's smear into hard parallel lines)
        pk = cv2.GaussianBlur(peak.reshape(2, hh, ww).transpose(1, 2, 0), (0, 0), BLUR_SOFT)
        gain = np.maximum(pk[..., 0] - acc[4], 0.0)
        cover = np.maximum(pk[..., 1] - acc[5], 0.0)
        C += gain[..., None] * lit
        A = np.minimum(A + cover, 1.0)
    return (ny0, ny1, nx0, nx1), C, A


CAUSTIC_GAIN, CAUSTIC_KEEP = 1.0, 0.3  # the focus's strength, and the key light's share of what lights the hand (the
# rest is the room's soft light, which the water does not focus)
CAUSTIC_NEAR = 0.02  # hand sizes: the least the water's light falls to the hand (water lying on the palm, its pool,
# throws its shadow right under it: taken as falling NEAR's least, the pool cast a shadow beside it like a raised lump)
CAUSTIC_MAX = 1.6  # the most the focus multiplies the key light by (at 3, with 0.4 of it the key's, the skin under the
# orb clipped to white under a soft room light)
CAUSTIC_KNEE = (0.6, 0.86)  # linear: what the water's light raises is rolled off channel by channel from the first of
# these towards the second, and never below the plate (lifted straight, the focus clipped to white: a lamp glued to
# the palm)
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
SPOT = (0.45, 0.65, 0.1)  # the held orb's focus: the share of the light through it that reaches it, how wide it is
# against its rays' width there, and the share of the orb's light still traced cell by cell over it (the spot alone
# was a projector's disc; that share dances with its surface)
SPOT_LOBES = 4.0  # its outline's wobble shows this many times over in its focus...
SPOT_MAX = 2.2  # ... which multiplies the key light by at most this, rolled off (held to CAUSTIC_MAX with the rest of
# the water's light, the spot was a pale pink disc barely brighter than the lit skin: a lens gathers the light through
# all of it into a hot spot; at 3.5, a lamp glued to the palm)...
SPOT_NET = (96, 4, 4.5, (0.6, 1.0), 1.8, 0.03, 0.25, 0.12)  # ... and its light lies in the uneven patches that rays
# through the orb's moving surface gather it into where they land: a grid of this many rays across it, laid between
# each two as if refined this many times (the light between neighbouring rays lands between where they do: splatted as
# points, a few to a cell, they were a cluster of soft blobs, bubble wrap), splatted at the picture's resolution and
# blurred this much (px: at 2, thin lines meeting in a white star, a lamp glued to the palm), the light there this base
# plus this gain times how many rays land there against the spot's mean, at most this many times it, how fast its
# ripples drift (of its radius a frame), the share of the finest ripples the rays are bent by (all of them folded the
# light into glitter), and its surface smoothed over this share of its radius: bent by its broad wobble and a little by
# its ripples, the light breaks into soft patches round the hot spot that move with the water (bent by its small waves
# in full, and lines up to 3.5 times as bright, it branched into a white filament, an electric spark)
SPOT_IN = 4  # frames after the orb has formed its light comes into that net (drawn through a surface still settling
# from its gathering, the net was a four-armed cross, a lens flare's star)
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
            held: tuple | None = None, lobes: tuple = (), noise: Noise | None = None, net_in: float = 1.0,
            flare: float = 1.0) -> tuple:
    """The water's light on the hand behind it, in place: each cell of water in front of the hand sends its share of
    the key light along the ray bent through it to the hand, at the hand's own depth there, and lights it there; the
    light the hand had there without the water is taken away (energy kept: a bright focus inside a soft shadow). A
    small drop's share fades: under a room's wide light its shadow and its focus spread wider than it is. ``held``,
    the orb over the palm (x, y, z, R, w): its light lands as one, wobbling with its ``lobes`` (Scene.lobes), at
    ``net_in`` of it in the net of brighter lines rays through its moving surface draw (_rays; ``noise``: its finest
    ripples; Scene.net), and its bulk also takes some of the room's soft light off the palm right under it, both at w
    of their strength, the shade shrinking as it goes. What it raises is ``flare`` times as bright (a hit: Scene.flare)
    and rolled off below white (CAUSTIC_KNEE). Returns the box it changed."""
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
    small *= _ss(0.04, 0.16, hq[iy, ix] / np.maximum(rq[iy, ix], 1.0))  # a meniscus has too little optical thickness
    # to cast a strong rectangular focus or shadow under a stream's root
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
        g = np.where(g > 1.0, 1.0 + (g - 1.0) * (1.0 + wh * net_in * (net - 1.0)), g)
    m = matte[y0:y1, x0:x1].astype(np.float32) / 255.0
    lift = CAUSTIC_KEEP * (g - 1.0) * np.where(g > 1.0, np.float32(flare), np.float32(1.0))
    edge = _box_fade(y1 - y0, x1 - x0, max(4, pad // 2))  # what reaches the box's edge fades by it (faded over an
    # eighth of the box, the light moved when a drip grew the box)
    patch = P[y0:y1, x0:x1]
    raised = patch * (1.0 + lift * m * edge)[..., None]
    k0, k1 = CAUSTIC_KNEE
    soft = np.where(raised > k0, k0 + (k1 - k0) * np.tanh((raised - k0) / (k1 - k0)), raised)
    patch[:] = np.where(raised > patch, np.maximum(soft, patch), raised)
    if shade_box[1] <= shade_box[0]:
        return (y0, y1, x0, x1)
    return (min(y0, shade_box[0]), max(y1, shade_box[1]), min(x0, shade_box[2]), max(x1, shade_box[3]))


def _rays(layer: Layer, cx: float, cy: float, R: float, dh: float, L: np.ndarray, noise: Noise, k: int,
          s: float) -> tuple[np.ndarray, np.ndarray]:
    """Where rays of the key light (along -``L``) through the held orb (radius ``R`` at (cx, cy), its back ``dh`` over
    the palm) land on the palm: ((n, n, 2) px about their mean, (n, n) which went through it), a grid of SPOT_NET[0]
    across it, bent in at its front and out at its back (the front's normal mirrored, as shade traces it) by its surface
    as ``layer`` holds it, smoothed to its broad wobble, and a little by the finest ripples (SPARKLE, off ``noise`` on
    frame ``k``; ``s`` a hand's size px). Where neighbouring rays cross, the light piles up into the bright lines light
    through moving water throws."""
    n = SPOT_NET[0]
    u = np.linspace(-0.92, 0.92, n)
    ux, uy = np.meshgrid(u, u)
    on = ux * ux + uy * uy < 0.92 ** 2
    px, py = (cx + R * ux).astype(np.float32).ravel(), (cy + R * uy).astype(np.float32).ravel()
    ly0, lx0 = layer.box[0], layer.box[2]
    gy, gx = np.gradient(cv2.GaussianBlur(layer.h, (0, 0), max(0.7, SPOT_NET[7] * R)))
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
    """Entrained air carried by the water: flattened clear centres, a restrained inner edge and a directional silver
    rim. Bubbles are large enough to read as air instead of dark sediment, and inherit the water's visibility."""
    H, W = P.shape[:2]
    by0, by1, bx0, bx1 = layer.box
    for x, y, r, seen in items:
        if r < 0.8 or not (by0 <= y < by1 and bx0 <= x < bx1):
            continue
        wgt = float(layer.alpha[int(y) - by0, int(x) - bx0]) * seen
        rr = int(1.3 * r + 3)
        ya, yb, xa, xb = max(0, int(y) - rr), min(H, int(y) + rr + 1), max(0, int(x) - rr), min(W, int(x) + rr + 1)
        if yb <= ya or xb <= xa or wgt <= 0.01:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        dx, dy = (xx - x) / (1.2 * r), (yy - y) / (0.9 * r)
        d = np.hypot(dx, dy)
        inside = _ss(1.0 + 0.8 / r, 1.0 - 0.8 / r, d)
        dark = BUBBLE_LOOK[0] * inside * np.exp(-((d - 0.73) / 0.16) ** 2)
        gx, gy = x + 0.4 * r * L[0], y + 0.4 * r * L[1]
        pin = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * (0.12 * r + 0.45) ** 2))
        patch = P[ya:yb, xa:xb]
        patch *= (1.0 - wgt * dark)[..., None]
        facing = np.clip((dx * L[0] + dy * L[1]) / np.maximum(d, 0.1), 0.0, 1.0)
        rim = np.exp(-((d - 0.91) / 0.10) ** 2) * (0.2 + 0.8 * facing)
        patch += (wgt * BUBBLE_LOOK[1] * (0.45 * pin + rim))[..., None] * lit[None, None, :]


def glitter(P: np.ndarray, items: list, layer: Layer, lit: np.ndarray, L: np.ndarray) -> list[tuple]:
    """Points of the key light off the crests of the orb's ripples (Scene.sparks), over it, in place: each a bright
    point with a faint halo, on its side towards the light ``L`` (a unit vector, z towards the camera), within SPARKS'
    spread of the light's way from its middle, on the crest nearest how far out it is drawn (layer.crest: none where
    no crest passes), as much as the water covers the picture there (where the hand is in front of it, none). Returns
    the boxes it changed."""
    H, W = P.shape[:2]
    by0, by1, bx0, bx1 = layer.box
    base = math.atan2(float(L[1]), float(L[0]))
    boxes = []
    for cx, cy, r, turn, out, sg, strength in items:
        if strength <= 0.01:
            continue
        a = base + turn * SPARKS[5]
        reach = np.clip(out + np.linspace(-0.15, 0.15, 7), 0.3, 0.92) * r
        xs, ys = cx + reach * math.cos(a), cy + reach * math.sin(a)
        on = (ys >= by0) & (ys < by1) & (xs >= bx0) & (xs < bx1)
        if not on.any():
            continue
        xs, ys = xs[on], ys[on]
        crest = np.ones(xs.size) if layer.crest is None else layer.crest[ys.astype(int) - by0, xs.astype(int) - bx0]
        j = int(np.argmax(crest))
        x, y = float(xs[j]), float(ys[j])
        wgt = (float(layer.alpha[int(y) - by0, int(x) - bx0]) * strength
               * float(_ss(SPARKS[6], SPARKS[6] + 0.3, crest[j])))
        e = int(4 * sg + 3)
        ya, yb, xa, xb = max(0, int(y) - e), min(H, int(y) + e + 1), max(0, int(x) - e), min(W, int(x) + e + 1)
        if yb <= ya or xb <= xa or wgt <= 0.01:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        g = np.exp(-d2 / (2.0 * sg * sg)) + 0.12 * np.exp(-d2 / (2.0 * (3.0 * sg) ** 2))
        P[ya:yb, xa:xb] += (wgt * g)[..., None] * lit[None, None, :]
        boxes.append((ya, yb, xa, xb))
    return boxes


def _sweep_line(sweep: tuple, L: np.ndarray) -> tuple:
    """The held orb's highlight carried round it with its water (Scene.sweep, SWEEP), lit by the key light ``L`` (a
    unit vector): (points (m, 2) px, strengths (m,), width px) along the loop its air goes round, tapering to both ends,
    brighter on its side towards the light and dimmer round the far side of the loop, seen through the water."""
    cx, cy, r, turn, at, strength = sweep
    _, (a, b), length, sg, bright, far = SWEEP
    ph = at + np.linspace(-0.5, 0.5, 25) * length
    lx, ly = a * r * np.cos(ph), b * r * np.sin(ph)
    cs, sn = math.cos(turn), math.sin(turn)
    x, y = cx + cs * lx - sn * ly, cy + sn * lx + cs * ly
    lxy = np.asarray(L[:2], np.float64) / (math.hypot(float(L[0]), float(L[1])) + 1e-6)
    facing = ((x - cx) * lxy[0] + (y - cy) * lxy[1]) / np.maximum(np.hypot(x - cx, y - cy), 1e-6)
    st = (bright * strength * np.sin(np.linspace(0.0, math.pi, ph.size)) ** 1.5
          * (far + (1.0 - far) * _ss(-0.3, 0.3, np.sin(ph))) * (0.4 + 0.6 * _ss(-0.5, 0.8, facing)))
    st *= 0.15 + 0.85 * _ss(-0.2, 0.5, np.cos(7.0 * ph - 0.7 * at))
    return np.c_[x, y], st, sg


def _streak_lines(streak: tuple, L: np.ndarray) -> list[tuple]:
    """The key light mirrored in the thrown orb (Scene.streak, THROWN_STREAKS), lit by the key light ``L`` (a unit
    vector): (points (m, 2) px, strengths (m,), width px) each, the glint off its side towards the light (or the far
    one) drawn out along its flight over its smear, brightest at its front."""
    x, y, ux, uy, smear, r, strength = streak
    lxy = np.asarray(L[:2], np.float64) / (math.hypot(float(L[0]), float(L[1])) + 1e-6)
    out = []
    for off, bright, sg in THROWN_STREAKS:
        cx, cy = x + lxy[0] * off * r, y + lxy[1] * off * r
        a = np.array([1.0, 0.35, -0.35, -1.0]) * (THROWN_GLINT * r + 0.5 * smear)
        out.append((np.c_[cx + ux * a, cy + uy * a], bright * strength * np.array([0.3, 1.0, 0.75, 0.1]), sg))
    return out


def _lines(P: np.ndarray, lines: list, layer: Layer, lit: np.ndarray) -> list[tuple]:
    """Thin highlights over the water, in place: each (points (m, 2) px, strengths (m,) of the key light ``lit``,
    width px, sigma) a line through its points as bright as its strengths there, in a faint halo (LINE_HALO), only as
    much as the water covers the picture there (where the hand is in front of it, none). Returns the boxes it
    changed."""
    by0, by1, bx0, bx1 = layer.box
    halo, wide = LINE_HALO
    boxes = []
    for pts, st, sg in lines:
        e = 3.0 * wide * sg + 2.0
        y0, y1 = max(by0, int(pts[:, 1].min() - e)), min(by1, int(pts[:, 1].max() + e) + 1)
        x0, x1 = max(bx0, int(pts[:, 0].min() - e)), min(bx1, int(pts[:, 0].max() + e) + 1)
        if y1 <= y0 or x1 <= x0 or float(np.max(st)) <= 0.01:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        best = np.zeros(yy.shape, np.float32)
        for (ax, ay), (bx, by), sa, sb in zip(pts[:-1], pts[1:], st[:-1], st[1:]):
            dx, dy = bx - ax, by - ay
            t = np.clip(((xx - ax) * dx + (yy - ay) * dy) / max(dx * dx + dy * dy, 1e-6), 0.0, 1.0)
            d2 = (xx - ax - t * dx) ** 2 + (yy - ay - t * dy) ** 2
            np.maximum(best, (sa + t * (sb - sa)) * (np.exp(-d2 / (2.0 * sg * sg))
                                                     + halo * np.exp(-d2 / (2.0 * (wide * sg) ** 2))), out=best)
        P[y0:y1, x0:x1] += (best * layer.alpha[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0])[..., None] * lit
        boxes.append((y0, y1, x0, x1))
    return boxes


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
        self.info = np.zeros((0, 12))  # r, foam, born, life, fade, stretch, gravity (px a frame squared), delay,
        # free (1: it bridges to no other free piece), field weight, how much of the key light's glint it catches, the
        # frames it is smeared over
        self.count = 0  # pieces added so far

    def add(self, k: int, x, v, r, foam=0.0, life=np.inf, fade=1.0, stretch=1.0, g=None, delay=0.0, free=True,
            w=1.0, shutter=SHUTTER) -> None:
        """Pieces leaving frame ``k`` at px and depth ``x`` (m, 3), ``v`` px a frame (m, 3), each smeared over
        ``shutter`` frames of its travel."""
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
                                            facing, shutter)[1:], -1).astype(np.float64)
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
        r0, foam, born, life, fade, stretch, _, delay, free, w, facing, shut = self.info.T
        age = k - born
        r = r0 * kz * np.clip((life - age) / fade, 0.0, 1.0) ** 0.8
        vxy = vxy * kz[:, None]
        z = self.x[:, 2]
        out = []
        for j in np.flatnonzero((age < life) & (age >= delay) & (r >= 0.5)):
            shape, move = _moving(r[j], vxy[j], stretch[j], shut[j])
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
        down = np.array([0.25, 1.0, 0.0])
        down -= (down @ flat) * flat
        ahead = _unit(ahead - (ahead @ flat) * flat + 1.5 * down)
    carry = SPLASH_CARRY * (v if flat is None else v - (v @ flat) * flat)

    def dirs(m: int, cone: float = SPLASH_CONE) -> np.ndarray:
        d = _unit(rng.normal(size=(m, 3)) + cone * ahead)  # on along its motion, over a cone (thrown out evenly all
        if flat is not None:  # round, it was a starburst)
            d = _unit(d - (d @ flat)[:, None] * flat + 0.3 * np.abs(d @ flat)[:, None] * flat)
        return d

    pace = s / FPS * (0.6 + 0.4 * size)
    n_drop, n_lig, n_spray, n_lump = (max(2, int(a * (0.4 + 0.6 * size))) for a in SPLASH)
    if flat is not None:
        n_lig, n_spray = max(3, n_lig // 2), n_spray // 2
    swell, foam, wrinkles = BURST_SWELL
    shape, move = _moving(swell * R, v[:2])
    body = [Blob(c[0], c[1], c[2], swell * R, shape, v[0], v[1], foam=foam, move=move,
                 modes=tuple((l, a, rng.uniform(0, 2 * math.pi)) for l, a in wrinkles))]
    if flat is not None:  # an uneven thin sheet spreading on the skin, with a few broad tongues rather than spokes
        fingers = ((3, 0.14, float(rng.uniform(0, 2 * math.pi))),
                   (5, 0.055, float(rng.uniform(0, 2 * math.pi))))
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


def _air(bub: np.ndarray, loop: tuple, circ: float, t: float, c: np.ndarray, r: float, s: float,
         seen: float) -> list[tuple]:
    """The air caught in the orb of radius ``r`` at ``c``, carried round the loop its water circulates in
    (BUBBLE_LOOP; ``loop``: how far it is tipped towards the camera, of the radius, and turned in the picture, rad),
    ``circ`` rad round it ``t`` seconds in, each bubble (``bub``: where round it it started, its share of the
    circulation, its wander's phase and rate, its radius hand sizes; ``s`` a hand's size px) at its own pace and
    wandering a little off it: (x, y, r, seen) each. Those round the far side of the loop, seen through the water, are
    dimmer and a little magnified."""
    a, b, _, _, _, wander = BUBBLE_LOOP
    tip, turn = loop
    cs, sn = math.cos(turn), math.sin(turn)
    out = []
    for ph0, share, wph, whz, br in bub:
        ph = ph0 + circ * share
        lx = a * math.cos(ph) + wander * math.sin(2.0 * math.pi * whz * t + wph)
        ly = b * math.sin(ph) + wander * math.sin(2.0 * math.pi * 0.8 * whz * t + 1.7 * wph)
        near = math.sin(ph)  # 1 nearest the camera, -1 round the far side
        grow = 1.0 + 0.12 * (1.0 - near) * abs(tip)
        out.append((c[0] + r * (cs * lx - sn * ly) * grow, c[1] + r * (sn * lx + cs * ly) * grow, br * s,
                    (0.65 + 0.35 * near) * seen))
    return out


def _sparks(seed: int, k: int, c: np.ndarray, r: float, w: float) -> list[tuple]:
    """The points of the key light off the crests of the held orb's ripples on frame ``k`` (SPARKS; the orb of radius
    ``r`` at ``c``): each flashes over its own two or three frames, then flashes again somewhere else on it, at ``w`` of
    their strength, on its side towards the light, on the crest nearest where it is drawn (glitter): (its centre x, y
    and radius, how far round from the light's way, -1..1, how far out, size px, strength) each."""
    m, (p0, p1), (s0, s1), bright, (o0, o1) = SPARKS[:5]
    out = []
    for i in range(m):
        g = _rng(seed, _ORB, 7, i)
        u = k / g.uniform(p0, p1) + g.uniform(0.0, 1.0)
        cyc = math.floor(u)
        h = _rng(seed, _ORB, 8, i, cyc)  # (where this flash of it is)
        flash = math.sin(math.pi * (u - cyc)) ** 2
        out.append((c[0], c[1], r, h.uniform(-1.0, 1.0), h.uniform(o0, o1), h.uniform(s0, s1), bright * flash * w))
    return out


def _body(theta, r: float, wobble: tuple) -> np.ndarray:
    """The wall sheet's body: its radius px at angles ``theta`` (Sheet's r and wobble), its fingers aside."""
    th = np.asarray(theta, np.float64)
    return r * (1.0 + sum((a * np.cos(l * th - ph) for l, a, ph in wobble), np.zeros(th.shape)))


def _smax(a, b, k: float):
    """The smooth union of two insides (signed px, + inside): the larger, blended over ``k`` px into a fillet."""
    h = np.clip(0.5 + 0.5 * (a - b) / k, 0.0, 1.0)
    return b + (a - b) * h + np.float32(k) * h * (1.0 - h)


def _inside(dx, dy, r: float, fingers: tuple, wobble: tuple, taper: tuple = (1.4, 0.7)) -> tuple:
    """How far px inside a wall sheet's outline points (dx, dy) px from its middle are (+ inside), and its body's radius
    at their angles: its body, and each finger from inside it out to a bead at its tip, narrowing from ``taper[0]`` to
    ``taper[1]`` of its half width, meeting the body in a fillet (Sheet's r, fingers and wobble)."""
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
        e = max(bead, 1.1, taper[0]) * w + FINGER_BLEND * w + 2.0
        bx = np.searchsorted(xs, [min(a0 * ux, a1 * ux) - e, max(a0 * ux, a1 * ux) + e])
        by = np.searchsorted(ys, [min(a0 * uy, a1 * uy) - e, max(a0 * uy, a1 * uy) + e])
        sl = (slice(by[0], by[1]), slice(bx[0], bx[1]))
        if by[1] <= by[0] or bx[1] <= bx[0]:
            continue
        fx_, fy_ = dx[sl], dy[sl]
        a = np.clip(fx_ * ux + fy_ * uy, a0, a1)
        f = (np.float32(w) * (taper[0] - (taper[0] - taper[1]) * (a - a0) / (a1 - a0))
             - np.hypot(fx_ - a * ux, fy_ - a * uy))
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


def _wall(scenes: list, drops: Drops, hit: tuple, n: int, cam: np.ndarray | None, rng,
          drawn: float = WALL_THROWN[0], bead_count: int | None = None) -> list:
    """A ball of water bursting on the room's wall on frame ``hit[0]`` (the orb, or what a lash flings): ``hit`` =
    (frame, x, y, z, R, s, fan, size), the px it hits at on that frame and its depth, the ball's radius and a hand's
    size there, the unit direction along the wall it was thrown in and how hard. The water stays on the wall
    (Scene.film, from that frame on, added to what is there, held to the room as the camera moves): it lands as a
    squashed body flashing white for a moment, throwing a burst of drops out along the wall and off it, and bigger
    drops off its rim towards the camera, and spreads over its first frames as a lobed sheet drawn out the way it was
    thrown (``drawn`` longer than wide: WALL_THROWN), its edge torn into tongues flung out round it, mostly that way,
    each ending in a bead, thinning; drops flung out of it land on the wall round it and slow to a stop, the smallest
    going into the wall, the big ones sliding down; the sheet drains, over a few frames, into the wet patch it leaves,
    soaking in, whose lower edge creeps down and runs down in narrow runs of their own lengths, each with a bead at its
    head; drips break off the longest and slide on down, leaving a wet trail. Over the next frames a few drops let go
    of its lower edge and fall. Returns those, (frame, px and depth, velocity px a frame, radius, gravity px a frame
    squared) each, for the caller to let go on their frame. Every piece eases in: nothing of it comes or goes between
    two frames. ``bead_count`` limits the smaller lash's satellite beads, retaining only beads at least four pixels
    wide so they read as refracting water rather than pinpoints."""
    kh, xh, yh, zh, R, s, fan, size = hit
    to_room = np.linalg.inv(cam[kh]) if cam is not None else np.eye(3)
    q = to_room @ np.array([xh, yh, 1.0])
    c0 = q[:2] / q[2]
    fa = math.atan2(fan[1], fan[0])
    nf = int(rng.integers(WALL_FINGERS[0], WALL_FINGERS[1] + 1))  # its fingers: tongues flung out all round it, most
    n_on = int(round(FINGER_DIR[0] * nf))  # the way it was thrown, short upwards unless it was thrown so
    f_ang = np.r_[fa + rng.normal(0.0, FINGER_DIR[1], n_on), rng.uniform(0.0, 2.0 * math.pi, nf - n_on)]
    along = np.maximum(np.cos(f_ang - fa), 0.0)  # (1: the way it was thrown)
    reach = rng.uniform(0.0, 1.0, nf) ** FINGER_LEN[2]  # (most short, one or two long, the way it was thrown: all
    reach[int(np.argmax(along))] = max(float(reach.max()), 0.8)  # long, they were legs; long across, arms; all short, a
    # disc)
    f_len = (FINGER_LEN[0] + (FINGER_LEN[1] - FINGER_LEN[0]) * reach) * (FINGER_DIR[2] + (1.0 - FINGER_DIR[2]) * along)
    f_len = np.minimum(f_len, FINGER_UP + (FINGER_LEN[1] - FINGER_UP) * np.maximum(_ss(0.7, 0.0, -np.sin(f_ang)),
                                                                                  along ** 2))  # (up only if thrown so)
    f_wid = rng.uniform(*FINGER_W, nf)
    broad = []  # several substantial tongues survive at phone size; choose separated directions before widening them
    for j in np.argsort(-f_len):
        if all(abs(float(np.angle(np.exp(1j * (f_ang[j] - f_ang[b]))))) > 0.4 for b in broad):
            broad.append(int(j))
        if len(broad) == 5:
            break
    if len(broad) < 5:
        broad += [int(j) for j in np.argsort(-f_len) if j not in broad][:5 - len(broad)]
    f_wid[broad] = rng.uniform(0.078, 0.108, len(broad))
    f_len[broad] = rng.uniform(0.5, 0.9, len(broad))
    across = along < 0.3
    f_len[across] = np.minimum(f_len[across], 0.45 * f_len.max())  # substantial roots, but no pair of long sideways
    # arms: the longest tongues keep the direction in which the water struck the wall
    f_bead = rng.uniform(*FINGER_BEAD, nf)
    dots = []  # drops thrown on past each finger's tip, left on the wall in a line with it
    for j in range(nf):
        past = np.sort(rng.uniform(*WALL_TONGUE[2], int(rng.integers(WALL_TONGUE[1][0], WALL_TONGUE[1][1] + 1))))
        for i, pj in enumerate(past):
            dots.append((j, float(pj), float(rng.normal(0.0, 0.6) * f_wid[j]),
                         float(max(np.clip(f_wid[j], *WALL_PATCH[0]) * rng.uniform(*WALL_TONGUE[3]) * 0.75 ** i,
                                   0.012))))
    wob = ((2, drawn, 2.0 * fa),) + tuple(
        (l, a * float(rng.uniform(0.5, 1.5)), float(rng.uniform(0.0, 2.0 * math.pi))) for l, a in WALL_WOBBLE)
    s0, s1, tau_s = WALL_SPREAD
    fan2 = np.asarray(fan, np.float64)

    def radius(t):  # px, frame(s) ``t`` after the hit
        return R * (s1 - (s1 - s0) * np.exp(-np.maximum(t, 0.0) / tau_s))

    def centre(t):  # it lies further on the way it was thrown
        return c0 + fan2 * (WALL_THROWN[1] * (radius(t) - s0 * R))[..., None] / max(1.0 - s0 / s1, 1e-6)

    def grown(t):  # how far its fingers are out
        return 1.0 - math.exp(-(max(t, 0.0) + 0.5) / WALL_FINGERS[2])

    def edge(ang, t):  # px on the room's wall where its body's edge is at angles ``ang`` on frame ``t`` after the hit
        return centre(t) + np.stack([np.cos(ang), np.sin(ang)], -1) * _body(ang, radius(t), wob)[..., None]

    def tip(j, t):  # px on the room's wall of finger ``j``'s tip
        return centre(t) + np.array([math.cos(f_ang[j]), math.sin(f_ang[j])]) * (1.0 + f_len[j] * grown(t)) * radius(t)

    # the drops on the wall: where on its edge they leave (angle, frame), how far they run on and how fast that stops,
    # their size, when the smallest are gone, how fast the big ones slide down and how soon they get going
    m = max(4, int((WALL_BEADS[0] if bead_count is None else bead_count) * (0.4 + 0.6 * size)))
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
    # its lower edge creeping down as the film drains, in narrow runs well apart (in an even row along it, a comb; two
    # together, a clump), each its own length, setting off on its own frame and slowing, a bead at its head; the
    # drips break off the first few and slide on down the wall
    def soaked(t):  # the patch's radius px ``t`` frames after the hit
        return radius(t) * (1.0 + WALL_SOAK[0] * float(_ss(0.0, WALL_SOAK[1], t)))

    def creep(t):  # how far its whole lower edge has crept down (of its radius)
        return WALL_CREEP[0] * (1.0 - math.exp(-max(t - WALL_SAG[0], 0.0) / WALL_CREEP[1]))

    r_ang: list[float] = []
    for a in rng.uniform(WALL_RUN[3][0] * math.pi, WALL_RUN[3][1] * math.pi, 64):
        if len(r_ang) < int(rng.integers(WALL_SAG[1][0], WALL_SAG[1][1] + 1)) and all(
                abs(float(np.angle(np.exp(1j * (a - b))))) > 0.15 for b in r_ang):
            r_ang.append(float(a))
    nr = len(r_ang)
    r_wid = rng.uniform(*WALL_SAG[3], nr)
    r_len = WALL_SAG[2][0] + (WALL_SAG[2][1] - WALL_SAG[2][0]) * rng.uniform(0.0, 1.0, nr) ** 2.5  # (most short)
    r_t0 = WALL_SAG[0] + rng.uniform(*WALL_RUN[0], nr)
    r_tau = rng.uniform(*WALL_RUN[1], nr)
    md = min(WALL_DRIPS[0], nr)
    r_len[:md] = np.maximum(r_len[:md], 0.6 * WALL_SAG[2][1])  # (the drips break off the long ones)

    def run_len(t):  # each run's length (of the radius) ``t`` frames after the hit, and how fast it grows a frame
        e_ = np.exp(-np.maximum(t - r_t0, 0.0) / r_tau)
        return r_len * (1.0 - e_), np.where(t > r_t0, r_len / r_tau * e_, 0.0)

    def heads(t):  # px on the room's wall of each run's head, and how fast it moves down (px a frame)
        rs = soaked(t)
        ln_, v_ = run_len(t)
        top = centre(t) + np.stack([np.cos(r_ang), np.sin(r_ang)], -1) * _body(np.array(r_ang), rs, wob)[:, None]
        return top + np.c_[np.zeros(nr), (creep(t) * _sags(np.array(r_ang)) + ln_) * rs], v_ * rs

    d_born = WALL_DRIPS[4] + np.r_[0.0, np.cumsum(rng.uniform(*WALL_DRIPS[5], max(md - 1, 0)))].round()[:md]
    d_start = np.array([heads(float(b))[0][j] for j, b in enumerate(d_born)]).reshape(-1, 2)
    d_ang = r_ang[:md]
    ang = np.r_[ang, f_ang, d_ang]
    rb = np.r_[rb, t_r, R * rng.uniform(*WALL_DRIPS[1], md)]
    far = np.r_[far, t_far, np.zeros(md)]
    tau = np.r_[tau, np.full(nf, 1.7), np.ones(md)]
    born = np.r_[born, np.full(nf, kt), d_born]
    start = np.r_[start, t_start, d_start]
    ph = rng.uniform(0.0, 2.0 * math.pi, len(rb))
    gone = np.where(rb < WALL_SLIDE[0] * R, rng.uniform(*WALL_BEADS[4], len(rb)), 1e6)  # (the big ones stay)
    drip = np.r_[np.zeros(m + nf, bool), np.ones(md, bool)]
    vmax = np.where(drip, WALL_DRIPS[2] * rng.uniform(*WALL_DRIPS[7], len(rb)),
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
    drops.add(kh, p_sp, d_sp * rng.uniform(*WALL_SPRAY[1], (nsp, 1)) * pace,
              R * _sizes(rng, nsp, *WALL_SPRAY[2], WALL_SPRAY[4]), foam=0.0, life=rng.uniform(*WALL_SPRAY[3], nsp),
              fade=4.0, g=g)
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
        wet = 0.5 + 0.5 * float(_ss(0.0, WALL_WET[2], t))
        ran = f_len * grown(t)  # how far its fingers have run out (the wet patch keeps it)
        short = np.minimum(ran, FINGER_DRAIN[1])  # draining into its tip's drop
        ln = short + (ran - short) * math.exp(-max(t - FINGER_DRAIN[0], 0.0) / FINGER_DRAIN[2])
        rs_t, ln_runs = soaked(t), run_len(float(t))[0]
        film = [Sheet(c[0], c[1], rs_t, tuple(zip(f_ang.tolist(), ln.tolist(), f_wid.tolist(), f_bead.tolist())),
                      wob, R * (WALL_THICK[0] * math.exp(-t / WALL_THICK[1]) + WALL_THICK[2]),
                      math.exp(-t / WALL_THICK[1]), look, wet, WALL_FROTH[0] * max(0.0, 1.0 - t / WALL_FROTH[1]) ** 2,
                      creep(t), tuple(zip(r_ang, r_wid.tolist(), ln_runs.tolist())), tuple(ran.tolist()),
                      float(_ss(WALL_WET[3][2], WALL_WET[3][3], t)), tuple(dots),
                      1.0 - float(_ss(*WALL_SHEEN[1], t)))]
        hp, hv = heads(float(t))  # each run's head: a bead of the water collecting there, swelling as it runs
        hp = at(hp)
        for j in np.flatnonzero(ln_runs * rs_t > 1.0):
            rh = WALL_RUN[2] * r_wid[j] * rs_t * float(_ss(1.0, 6.0, ln_runs[j] * rs_t))
            film.append(Bead(hp[j, 0], hp[j, 1] - 0.5 * rh, rh, 0.0, float(hv[j]), 1.0, float(r_ang[j] * 7.0)))
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
            rn = rb * np.where(drip, _ss(-1.0, WALL_DRIPS[6], age), 1.0)  # a drip swells out of the edge
            visible = on & (a > 0.01) & ((rn >= 2.0) if bead_count is not None else True)
            for j in np.flatnonzero(visible):
                if fall[j] > rn[j]:
                    film.append(Trail(P0[j, 0], P0[j, 1], P1[j, 0], P1[j, 1], rn[j]))
            for j in np.flatnonzero(visible):
                film.append(Bead(P1[j, 0], P1[j, 1], rn[j], V[j, 0], V[j, 1], float(a[j]), float(ph[j])))
        scenes[k].film = scenes[k].film + film
    return falling


def _sags(ang, floor: float = WALL_CREEP[2]):
    """How much of its creep a wet patch's edge facing ``ang`` (rad, y down) sags by: as much as it faces down (to
    WALL_CREEP's power), at least ``floor`` of it."""
    return floor + (1.0 - floor) * np.maximum(np.sin(ang), 0.0) ** WALL_CREEP[3]


def _same_wall(patch: np.ndarray, on: np.ndarray, step: int = 1) -> np.ndarray:
    """(h, w) 0..1 over ``patch`` (linear RGB, its px ``step`` px of the picture apart): how much each pixel is the wall
    the water landed on (``on``, where it landed), by its brightness against the wall's own, a plane fitted to its log
    brightness there with what stands out of it left out (the room's light falls off across a wall; a switch on it is a
    different surface): WALL_PATCH."""
    lg = np.log(np.maximum(patch @ fx._LUMA, 1e-4)).astype(np.float32)
    ys, xs = np.nonzero(on)
    if ys.size < 8:
        return np.ones(lg.shape, np.float32)
    every = max(1, ys.size // 4000)
    ys, xs = ys[::every], xs[::every]
    A = np.c_[xs, ys, np.ones(xs.size)].astype(np.float64)
    v = lg[ys, xs].astype(np.float64)
    keep = np.ones(v.size, bool)
    for _ in range(3):
        coef = np.linalg.lstsq(A[keep], v[keep], rcond=None)[0]
        keep = np.abs(v - A @ coef) < WALL_PATCH[4][0]
        if keep.sum() < 8:
            return np.ones(lg.shape, np.float32)
    yy, xx = np.mgrid[0:lg.shape[0], 0:lg.shape[1]].astype(np.float32)
    plane = coef[0] * xx + coef[1] * yy + coef[2]
    same = _ss(WALL_PATCH[4][1], WALL_PATCH[4][0], np.abs(lg - plane)).astype(np.float32)
    e = max(3, (7 // step) | 1)
    return cv2.erode(same, np.ones((e, e), np.uint8))  # (and not the half-lit px along what stands out: the switch's
    # edge was drawn as a dark line)


def _soaked(it: Sheet, dx: np.ndarray, dy: np.ndarray, body: np.ndarray, noise: Noise, step: float = 1.0) -> tuple:
    """The wet patch a wall sheet ``it`` leaves, over px (``dx``, ``dy``, a grid ``step`` px apart) off its middle
    (``body``: px inside its body's edge, its fingers aside): (how wet the wall is there, 0..1; the band along its edge,
    just inside where it feathers out, 0..1 (WALL_RING); how much of it is wall the water has run down onto since, its
    lower edge creeping and its runs, 0..1; how much more its film collects there). Its outline is the crown the water
    landed in: its body, and the tongues it ran
    out in as far as they ran, narrowing to points, with the drops it threw on past their tips (WALL_PATCH,
    WALL_TONGUE), its edge broken at two scales and feathered unevenly, its tongues' less; its lower edge creeps down
    as much as it faces down (WALL_CREEP, _sags) and runs down in its runs (WALL_SAG); its tongues retain a narrow wet
    core as the wider body dries (WALL_WET)."""
    r = it.r
    ran = it.reach or tuple(f[1] for f in it.fingers)
    tongues = tuple((a, ln, float(np.clip(hw, *WALL_PATCH[0])), 0.0) for (a, _, hw, _), ln in zip(it.fingers, ran))
    lobes = tuple((l, a * WALL_PATCH[5][0 if l == 2 else 1], ph) for l, a, ph in it.wobble)
    d, _ = _inside(dx, dy, r, tongues, lobes, WALL_TONGUE[0])
    xs, ys = dx[0], dy[:, 0]
    for j, past, across, rad in it.dots:  # the drops thrown on past its tongues' tips
        if j >= len(it.fingers):
            continue
        ux, uy = math.cos(it.fingers[j][0]), math.sin(it.fingers[j][0])
        out = (1.0 + ran[j] + past) * r
        cx, cy, rd = out * ux - across * r * uy, out * uy + across * r * ux, rad * r
        bx = np.searchsorted(xs, [cx - rd - 4.0 * step, cx + rd + 4.0 * step])
        by = np.searchsorted(ys, [cy - rd - 4.0 * step, cy + rd + 4.0 * step])
        if by[1] > by[0] and bx[1] > bx[0]:
            sl = (slice(by[0], by[1]), slice(bx[0], bx[1]))
            d[sl] = np.maximum(d[sl], np.float32(rd) - np.hypot(dx[sl] - cx, dy[sl] - cy))
    (f1, a1), (f2, a2) = WALL_PATCH[1]
    on_body = _ss(-0.1 * r, 0.05 * r, body)  # (1 at its body's edge, 0 out along its tongues and drops)
    broken = (r * (a1 * noise.flow(15, dx, dy, 0.0, 0.0, 0, f1 * r) + a2 * noise.flow(16, dx, dy, 0.0, 0.0, 0, f2 * r))
              * (WALL_TONGUE[4] + (1.0 - WALL_TONGUE[4]) * on_body))
    feather = WALL_PATCH[2][0] + WALL_PATCH[2][1] * r * on_body * _ss(-0.5, 1.2,
                                                                      noise.flow(17, dx, dy, 0.0, 0.0, 0, 0.25 * r))
    edge = d + broken
    wm = _ss(-feather, feather, edge)
    ring = (_ss(0.3 * feather, feather + 1.0, edge)
            * (1.0 - _ss(feather + WALL_RING[1][0], feather + WALL_RING[1][1], edge))).astype(np.float32)
    tongue = _ss(0.05 * r, -0.1 * r, body + broken)  # (1 off its body, in its tongues)
    dry0, dry1 = WALL_WET[3][:2]
    wm = (wm * (1.0 - it.dry * (dry0 + (dry1 - dry0) * tongue))).astype(np.float32)
    collect, ran_ = 0.0, np.zeros(wm.shape, np.float32)
    if it.sag + max((ln for _, _, ln in it.runs), default=0.0) > 0.0:
        hh, ww = dx.shape
        pull = np.float32(it.sag) * _sags(np.arctan2(dy, dx)) * _ss(0.0, 0.3 * r, np.hypot(dx, dy))
        fall = np.zeros(ww, np.float32)
        for ang, wd, ln in it.runs:
            xr = math.cos(ang) * float(_body(ang, r, it.wobble))
            fall = np.maximum(fall, np.float32(ln) * np.exp(-((dx[0] - xr) / (wd * r)) ** 2))
        pull = pull + fall[None, :] * _ss(*WALL_RUN[4], dy / r)
        gx, gy = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))
        gy = (gy - np.float32(r / step) * pull).astype(np.float32)
        moved = cv2.remap(wm, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        ran_ = np.clip(moved - wm, 0.0, 1.0)
        wm = moved
        ring = cv2.remap(ring, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        collect = WALL_SAG[4] * _ss(0.3, 1.1, dy / r) * min(1.0, it.sag / (0.25 * WALL_CREEP[0]))
    return wm, ring, ran_, collect


def _film(P: np.ndarray, items: list, matte: np.ndarray | None, room: Room, k: int, noise: Noise) -> list[tuple]:
    """The water on the wall (Scene.film) on frame ``k``'s plate ``P`` (linear) in place, behind the water in the air;
    nothing on the hand, which is in front of the wall.

    The Sheet: the wet patch it leaves (_soaked: the wall darker and a little more saturated, mottled and unevenly in
    the streaks the water ran out in, its outline the crown it landed in, broken, its lower edge sagging into runs where
    the film collects darker and glossier; only on the wall it landed on; where patches overlap, one patch, worked
    WALL_Q px apart) and, at its ``look``, the running
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

    def wetten(patch, m, tone=1.0):  # (its cover rolled off into WALL_MERGE's most, its film's unevenness laid on that)
        d = np.clip(WALL_MERGE[1] * np.tanh(WALL_WET[0] * m / WALL_MERGE[1]) * tone, 0.0, 0.6)[..., None]
        lum = (patch @ fx._LUMA)[..., None]
        patch[:] = (lum + (patch - lum) * (1.0 + WALL_WET[1] / WALL_WET[0] * d)) * (1.0 - d)

    def laid(it, y0, y1, x0, x1, q):  # a sheet's shape on a grid q px apart over its box: px off its middle, how far
        # inside its outline and its body's radius, how far out (of that), its streaks and jets and how far they reach,
        # its side towards the key light
        ys = np.arange(y0, y1, q, dtype=np.float32) + np.float32(0.5 * (q - 1))
        xs = np.arange(x0, x1, q, dtype=np.float32) + np.float32(0.5 * (q - 1))
        dx, dy = np.meshgrid(xs - np.float32(it.x), ys - np.float32(it.y))
        din, re = _inside(dx, dy, it.r, it.fingers, it.wobble)
        rho = np.hypot(dx, dy)
        th_ = np.arctan2(dy, dx)
        u = np.clip(rho / np.maximum(re, 1.0), 0.0, 1.0)
        # the water runs out in uneven streaks (a noise laid round it, long outwards and narrow across, stretched as it
        # spreads, a whole number of its tiles round so it has no seam), thicker along each finger, over its outer part
        # only (streaks meeting in its middle, it was a cut lemon)
        rays = WALL_RAYS[0] * noise.along(8, rho / it.r, th_ * it.r, 0, WALL_RAYS[2],
                                          4.0 * math.pi * NOISE_CELL * it.r / (NOISE_SIZE * WALL_RAYS[3]), 0.0)
        jets = np.zeros(dx.shape, np.float32)  # the jets it bursts out in, along its fingers, each as strong as it is
        far = np.zeros(dx.shape, np.float32)  # long, and how far past its body the strongest there reaches
        for ang, ln, hw, _ in it.fingers:
            j_ = min(1.0, ln / FINGER_LEN[1]) * np.exp(-(np.angle(np.exp(1j * (th_ - ang))) / (WALL_JETS * hw)) ** 2)
            far = np.where(j_ > jets, np.float32(ln), far)
            jets = np.maximum(jets, j_)
        rays = (rays + WALL_RAYS[1] * jets) * _ss(0.45, 0.9, u).astype(np.float32)
        towards = (dx * kx + dy * ky) / np.maximum(rho, 1.0)  # its side towards the key light, -1..1
        return dx, dy, din, re, rho, th_, u, rays, jets, far, towards

    def full(a, h, w):  # a grid WALL_Q px apart, at the picture's px
        q = WALL_Q
        return cv2.resize(a, (a.shape[1] * q, a.shape[0] * q), interpolation=cv2.INTER_LINEAR)[:h, :w]

    sheets = []  # each sheet and its box, its running water drawn once the wall under them is wet
    for it in items:
        if not isinstance(it, Sheet):
            continue
        ran = it.reach or tuple(f[1] for f in it.fingers)
        ext = max((max(ln, rl) + max(hw * bd, WALL_TONGUE[2][1] + WALL_TONGUE[3][1] * WALL_PATCH[0][1])
                   for (_, ln, hw, bd), rl in zip(it.fingers, ran)), default=0.0)  # (its beads, the drops past them)
        sag = it.sag + max((ln for _, _, ln in it.runs), default=0.0)
        lobes = max(WALL_PATCH[5]) * sum(abs(a) for _, a, _ in it.wobble)
        broken = 3.0 * sum(a for _, a in WALL_PATCH[1]) + WALL_PATCH[2][1]
        reach = (it.r * (1.0 + lobes + ext + sag + broken) + WALL_PATCH[2][0] + 3.0 * WALL_PATCH[3] + WALL_EDGE[0]
                 + 6)
        b = box(it.x, it.y, reach)
        if b is None:
            continue
        (y0, y1, x0, x1), yy, xx, free = b
        q = WALL_Q  # the wet patch, worked q px apart (it has no detail finer)
        dx, dy, din, re, rho, th_, u, rays, jets, far, towards = laid(it, y0, y1, x0, x1, q)
        wm, ring, ran_, collect = _soaked(it, dx, dy, re - rho, noise, q)
        same = _same_wall(cv2.resize(P[y0:y1, x0:x1], dx.shape[::-1], interpolation=cv2.INTER_AREA), rho < re, q)
        wm = cv2.GaussianBlur(wm, (0, 0), WALL_PATCH[3] / q) * same
        tone = (1.0 + WALL_FILM[0] * (1.0 - u * u) + 0.2 * rays + collect
                + sum(a * noise.flow(i, dx, dy, 0.0, 0.0, 0, f * it.r) for i, (a, f) in zip((14, 18), WALL_FILM[1]))
                + WALL_FILM[2][0] * noise.along(19, dy, dx / it.r, 0, WALL_FILM[2][1] * it.r, WALL_FILM[2][2], 0.0)
                + WALL_FILM[3] * _ss(-0.2, 0.8, dy / it.r)
                + WALL_RING[0] * ring)
        sheen = WALL_SHEEN[0] * it.fresh * cv2.GaussianBlur(ran_, (0, 0), 1.0) * (0.5 + 0.5 * _ss(-0.6, 1.0, towards))
        h, w = y1 - y0, x1 - x0
        sheets.append((it, (y0, y1, x0, x1), full((it.wet * wm).astype(np.float32), h, w) * free,
                       full(np.clip(tone, 0.3, 2.5).astype(np.float32), h, w),
                       full((sheen * wm).astype(np.float32), h, w) * free, (xx, yy, free)))
    if sheets:  # the wall under them wet, as one patch where they overlap, its film the most covering one's
        Y0, Y1 = min(s_[1][0] for s_ in sheets), max(s_[1][1] for s_ in sheets)
        X0, X1 = min(s_[1][2] for s_ in sheets), max(s_[1][3] for s_ in sheets)
        acc, shine = np.zeros((Y1 - Y0, X1 - X0), np.float32), np.zeros((Y1 - Y0, X1 - X0), np.float32)
        film = np.ones((Y1 - Y0, X1 - X0), np.float32)
        for _, (y0, y1, x0, x1), amount, tone, gloss, _ in sheets:
            sl = (slice(y0 - Y0, y1 - Y0), slice(x0 - X0, x1 - X0))
            a = acc[sl]
            film[sl] = np.where(amount > a, tone, film[sl])
            acc[sl] = np.maximum(a, amount) + WALL_MERGE[0] * np.minimum(a, amount)
            np.maximum(shine[sl], gloss, out=shine[sl])
        wetten(P[Y0:Y1, X0:X1], acc, film)
        P[Y0:Y1, X0:X1] += shine[..., None] * lit
    for it, (y0, y1, x0, x1), _, _, _, (xx, yy, free) in sheets:
        if it.look > 0.01:  # the running water
            patch = P[y0:y1, x0:x1]
            dx, dy, din, re, rho, th_, u, rays, jets, reach, towards = laid(it, y0, y1, x0, x1, 1)
            ragged = WALL_EDGE[0] * noise.flow(10, dx, dy, 0.0, 0.0, 0, WALL_EDGE[1] * it.r)  # (in the sheet's own
            # frame: its unevenness moves with it)
            edge = din + ragged
            m = (_ss(-WALL_EDGE[2], WALL_EDGE[2], edge) * free).astype(np.float32)
            rw = WALL_RIM[0] * (1.0 + 0.6 * it.dome)  # (thick at the hit, where the ball's skin lands)
            e = np.clip(edge / rw, 0.0, 1.0)
            prof = np.sqrt(e * (2.0 - e))  # its edge rounded as it meets the wall
            inner = 1.6 * it.dome * (1.0 - u * u) ** 0.6 + 0.6 * (1.0 - it.dome)
            rim_on = float(_ss(*WALL_RIM[7], it.dome))  # (its rounded rim and the lines along it go as it flattens)
            ridge = rim_on * WALL_RIM[1] * rw * np.exp(-((edge - 0.8 * rw) / (0.7 * rw)) ** 2)
            th = prof * (it.thick * inner + ridge) * np.clip(1.0 + rays, 0.3, 2.2)
            # the light a body of water lying on the wall bends onto it: gathered at its foot, opposite the light,
            # its top in its shade, as much as it bulges (as a drop on the wall is)
            patch *= (1.0 + it.dome * it.look * m * (WALL_DOME[0] * np.exp(-((u - 0.62) / 0.22) ** 2)
                                                      * _ss(0.0, 0.9, -towards)
                                                      - WALL_DOME[1] * _ss(0.0, 0.9, towards) * u))[..., None]
            gy, gx = np.gradient(cv2.GaussianBlur(th.astype(np.float32), (0, 0), 1.0))
            nd = np.sqrt(gx * gx + gy * gy + 1.0)
            nx, ny, nz = -gx / nd, -gy / nd, 1.0 / nd
            # its rim bends, mirrors and glints only on its arc towards the light (all round, a crinkly bright
            # border, the outline of a disc)
            lim = (1.0 - _ss(2.0 * rw, 0.0, edge) * (1.0 - _ss(*WALL_RIM[6], towards))).astype(np.float32)
            bend = WALL_BEND * it.look * lim
            seen = cv2.remap(np.ascontiguousarray(patch), (xx - x0 - bend * gx).astype(np.float32),
                             (yy - y0 - bend * gy).astype(np.float32), cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
            F = (np.minimum(F0 + (1.0 - F0) * (1.0 - nz) ** 5, 0.7) * lim).astype(np.float32)[..., None]
            env = room.reflect(k, nx.ravel(), ny.ravel()).reshape(seen.shape)
            spec = WALL_GLINT * _window(nx, ny, nz, L3, 1.0)
            spec += GLINT_CORE[0] / white * _window(nx, ny, nz, L3, GLINT_CORE[1])
            spec *= lim
            body = re - rho + ragged  # px inside its body's edge, its fingers aside
            lengths = rim_on * _ss(0.0, 0.5, noise.flow(12, dx, dy, 0.0, 0.0, 0, WALL_RIM[5] * it.r))
            line = (WALL_RIM[2] * np.exp(-((body - 0.45 * rw) / 0.9) ** 2) * _ss(*WALL_RIM[4], towards) * lengths
                    * _ss(2.0, 0.0, din - body + ragged))
            contact = (WALL_RIM[3] * np.exp(-((edge - 0.8) / 0.9) ** 2) * _ss(-0.1, 0.7, -towards) * rim_on
                       * _ss(-0.2, 0.3, noise.flow(11, dx, dy, 0.0, 0.0, 0, WALL_RIM[5] * it.r)))
            col = (seen * (1.0 - F) + env * F) * (1.0 - contact)[..., None] + (spec + line)[..., None] * lit
            if it.froth > 0.01:  # churned white at the hit, a smooth sheet in its middle, torn out further out
                # along its jets, in streaks (in soft clumps, camouflage; in even streaks all round its middle, a
                # starburst, a sea urchin; in fine bubbles, its middle a stipple)
                fine = noise.flow(5, dx, dy, 0.0, 0.0, 0, WALL_FROTH[2] * it.r)
                streak = noise.along(9, rho / it.r, th_ * it.r, 0, WALL_FROTH[3],
                                     4.0 * math.pi * NOISE_CELL * it.r / (NOISE_SIZE * WALL_FROTH[4]), 0.0)
                out_ = _ss(0.3, 0.55, u)
                far = rho / np.maximum(re, 1.0)  # out past its body along its jets, white to their ends
                shape_ = ((1.0 - out_) * 0.3 * (0.85 + 0.15 * fine) * _ss(1.0, 0.4, u)
                          + out_ * jets * _ss(-0.6, 0.6, streak) * (0.6 + 0.4 * fine)
                          * _ss(1.0 + reach, 0.6 + 0.5 * reach, far))
                fm = np.clip(it.froth * shape_, 0.0, FOAM_MAX)
                col = col * (1.0 - fm)[..., None] + (fm * FOAM_ALBEDO * (0.8 + 0.2 * nz))[..., None] * lit
            ml = (m * it.look)[..., None]
            patch[:] = patch * (1.0 - ml) + col * ml
    for it in items:
        if isinstance(it, Trail):
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


def _orb_folds(sc: Scene, k: int, formed: int, c: np.ndarray, r: float, shape: np.ndarray,
               modes: list, seed: int, drops: Drops) -> None:
    """Pull one continuous surface into a travelling fold, then narrow it and release a drop. Giving the root a
    separate small blob makes its thickness switch to a small ball lens and draws a glass ring inside the body."""
    cycle, age = divmod(k - formed + seed * 3, 20)
    grow = float(_ss(0.0, 13.0, age))
    relax = 1.0 - float(_ss(14.0, 20.0, age))
    angle = (-2.5 if cycle % 2 == 0 else -0.25) + 0.22 * math.sin(0.10 * k + seed)
    d = np.array([math.cos(angle), math.sin(angle)])
    radius = 1.0 + sum(a * math.cos(l * angle - ph) for l, a, ph in modes)
    reach = (0.12 + 0.45 * grow) * relax
    width = 0.38 - 0.08 * grow
    sc.blobs[0] = sc.blobs[0]._replace(folds=((angle, width, reach),))
    tip = c.copy()
    tip[:2] += shape @ d * r * (radius + reach + 0.07)
    tip[2] -= 0.06 * r
    rd = 0.10 * r
    if age == 14:
        velocity = np.r_[shape @ d * (0.05 * r) + np.array([0.0, -0.018 * r]), 0.0]
        drops.add(k, tip, velocity, rd, life=9, fade=3, shutter=SHUTTER, g=0.012 * r)
    if 10 <= age < 14:
        sc.blobs.append(Blob(*tip, rd * float(_ss(9, 12, age))))


PALM_CHANNELS = ((2.35, 1.02, 0.090, 0.14), (0.95, 1.08, 0.065, -0.12), (-2.95, 0.80, 0.040, 0.10))
# Angle, length, half-width and bend in palm-film radii: unequal curved routes drain toward the wrist and little-finger
# side. Their terminal beads use the same paths so the connected water separates before the final isolated glints.


def _palm_spill(scenes: list[Scene], release: int, c0: np.ndarray, R: float, shape: np.ndarray,
                modes: list, hand: Hand, seed: int, mask: np.ndarray | None, strength: float,
                cam: np.ndarray | None) -> None:
    """Transfer a released orb into a palm sheet over several exposures, then drain it. The raised body overlaps
    the sheet; returning splash drops merge into it, and runoff leaves the measured silhouette as airborne water.
    The film follows the tracked skin; airborne water inherits its birth velocity and then moves independently.
    Each frame is evaluated from those fixed paths, with no frame-order dependence."""
    rng = _rng(seed, _ORB, 19)
    p0, s0 = hand.p[release], float(hand.s[release])
    n = len(scenes)
    contact, slump, drain, dry = COLLAPSE
    pace = 0.65 + 0.35 * float(np.clip(strength, 0.0, 1.0))
    # A finger web and two palm sides receive the sheet; each exit is measured against the hand silhouette.
    if hand.off is not None:
        lm = hand.off[release, :, :2]
        routes = [0.5 * (lm[a] + lm[b]) + 0.35 * (0.5 * (lm[a + 1] + lm[b + 1])
                                                               - 0.5 * (lm[a] + lm[b])) - p0
                  for a, b in ((5, 9),)]
    else:
        routes = [np.array([-0.27, -0.48]) * s0]
    routes += [np.array([-0.53, 0.05]) * s0, np.array([0.48, 0.20]) * s0]
    edges = []
    for route in routes:
        d = route / max(np.linalg.norm(route), 1e-6)
        reach = max(float(np.linalg.norm(route)), 0.4 * s0)
        if mask is not None:
            distances = np.linspace(0.12 * s0, max(reach, 1.2 * s0), 96)
            pts = p0 + distances[:, None] * d
            xx = np.clip(np.rint(pts[:, 0]).astype(int), 0, mask.shape[2] - 1)
            yy = np.clip(np.rint(pts[:, 1]).astype(int), 0, mask.shape[1] - 1)
            off = np.flatnonzero(mask[release, yy, xx] < 128)
            if off.size:
                reach = float(distances[max(0, off[0] - 1)])
        edges.append(d * reach / s0)
    exits = edges
    bead_offsets = rng.uniform((-0.30, -0.20), (0.30, 0.20), (5, 2))
    bead_offsets[:3] = np.asarray(exits) * 0.78
    bead_sizes = rng.uniform(0.018, 0.030, 5)
    inherited = scenes[release - 1].blobs[0].folds if release and scenes[release - 1].blobs else ()
    # Unequal launch times and arcs keep the spray away from a circular crown or an even dotted ring.
    spray = [(int(rng.integers(contact, contact + 5)), float(rng.uniform(12.0, 18.0)),
              rng.uniform((-0.22, -0.18), (0.22, 0.08)), rng.uniform(-0.035, 0.035),
              float(rng.uniform(0.014, 0.029)), float(rng.uniform(0.38, 0.78))) for _ in range(8)]
    runoff = [(14 + j * 2, edge, float(rng.uniform(0.045, 0.058))) for j, edge in enumerate(exits)]
    births = {}
    for born in {item[0] for item in spray + runoff}:
        kb = min(n - 1, release + born)
        previous = hand.p[max(0, kb - 1)]
        inverse = None if cam is None else np.linalg.inv(cam[kb])
        if cam is not None and kb:
            motion = cam[kb] @ np.linalg.inv(cam[kb - 1])
            previous = previous @ motion[:2, :2].T + motion[:2, 2]
        births[born] = (kb, hand.p[kb], float(hand.s[kb]), hand.p[kb] - previous, inverse)

    def airborne(born, tau, relative, velocity, k):
        """An airborne path in its birth frame's units, projected through subsequent camera movement."""
        kb, pb, sb, impulse, inverse = births[born]
        xy = pb + sb * relative + impulse * tau
        vel = sb * velocity + impulse
        if cam is not None:
            motion = cam[k] @ inverse
            xy = xy @ motion[:2, :2].T + motion[:2, 2]
            vel = vel @ motion[:2, :2].T
        return xy, vel, sb, kb

    landings = []
    for born, life, off, vx, rb, height in spray:
        kl = min(n - 1, release + born + int(math.ceil(life)))
        landing = off + np.array([vx * pace * life, 0.08 * pace])
        xy, _, sb, kb = airborne(born, life, landing, np.zeros(2), kl)
        on_skin = np.linalg.norm(xy - hand.p[kl]) < 0.7 * float(hand.s[kl])
        if mask is not None:
            ix, iy = int(round(xy[0])), int(round(xy[1]))
            on_skin = 0 <= ix < mask.shape[2] and 0 <= iy < mask.shape[1] and mask[kl, iy, ix] > 128
        landings.append((kl, xy, bool(on_skin)))
    for k in range(release, min(n, release + 100)):
        age = k - release
        sc = scenes[k]
        p, nrm, s = hand.p[k], hand.nrm[k], float(hand.s[k])
        ratio = s / s0
        zp = float(hand.depth(k, p[0:1], p[1:2], fingers=False)[0])
        un = _unit(nrm[:2]) if np.linalg.norm(nrm[:2]) > 1e-6 else np.array([0.0, 1.0])
        disc = np.asarray(_shape(un, max(abs(float(nrm[2])), 0.45), 1.0)).reshape(2, 2)
        fall = float(_ss(0, slump, age))
        if age < slump:
            # The upper reservoir empties through a neck before its last water reaches the receiving skin.
            rr = R * ratio * (1.0 - fall) ** 0.85
            flatten = float(_ss(2, 10, age))
            thickness = (1.0 - 0.65 * fall) * (1.0 - 0.85 * flatten)
            target = np.r_[p, zp] + nrm * (0.24 * s * (1.0 - flatten))
            carried = np.r_[p + (c0[:2] - p0) * ratio, zp + (c0[2]
                                - float(hand.depth(release, p0[0:1], p0[1:2], fingers=False)[0])) * ratio]
            touching = float(_ss(0, 0.55 * slump, age))
            at = carried * (1.0 - touching) + target * touching
            deform = np.asarray(_shape(un, (1.0 - 0.25 * fall) * (1.0 - 0.75 * flatten),
                                      1.0 + 0.40 * flatten)).reshape(2, 2) @ shape
            sc.blobs.append(Blob(*at, rr, tuple(deform.ravel()), vy=0.025 * s,
                                 modes=tuple((l, a * (1.0 - 0.3 * fall), ph + 0.1 * age) for l, a, ph in modes),
                                 flat=thickness, foam=0.012 * fall, move=(0.0, 0.012 * s),
                                 folds=tuple((ang, width, reach * (1.0 - fall)) for ang, width, reach in inherited)))
            if age >= 1:
                u = np.linspace(0.0, 1.0, 40)
                foot = np.r_[p, zp + 0.012 * s]
                root = at + (foot - at) * 0.18
                arrive = float(_ss(0, contact, age))
                end = root + (foot - root) * arrive
                pts = root[None] + (end - root)[None] * u[:, None]
                pts[:, 0] += 0.025 * s * np.sin(math.pi * u) * math.sin(age * 0.4)
                radius = s * (0.080 - 0.038 * u + 0.010 * u ** 6)
                radius *= (1.0 - 0.55 * float(_ss(8, slump, age))) * min(1.0, rr / (0.12 * s))
                radius *= 1.0 + 0.10 * np.sin(2.7 * math.pi * u - 0.7 * age)
                arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))]
                speed = _unit(foot - root) * 0.065 * s
                vel = np.tile(speed, (len(u), 1))
                sc.ribbon = _stream(pts, vel, radius, np.full(len(u), 0.008), vel[:, :2], arc,
                                    0.85 - 0.73 * u ** 5)
                # At the finite stream's foot the incoming flow turns sideways into the sheet; retain that thin cap
                # when the last sampling interval is much shorter than the preceding ones.
                if sc.ribbon:
                    sc.ribbon[-1] = sc.ribbon[-1]._replace(w=max(sc.ribbon[-1].w, 0.16))
                sc.blobs += sc.ribbon
            sc.held = (*at, rr, (1.0 - float(_ss(0, slump, age))) * (1.0 - flatten))
            sc.wave = WAVE[0] * (1.0 - 0.3 * fall)
            for j in range(2):
                ang = 1.4 + j * 2.0
                sc.trains.append((at[0], at[1], rr, math.cos(ang), math.sin(ang), 0.1,
                                  0.035 * (1.0 - fall), 0.8 * rr, age * 0.22 * rr))
        # Contact grows into a broad sheet while the upper reservoir is still flattening; hold it before draining.
        spread = 0.20 * float(_ss(contact - 1, 5, age)) + 0.80 * float(_ss(3, 9, age))
        remaining = 1.0 - float(_ss(15, drain, age))
        if contact - 1 <= age < drain:
            rr = R * ratio * (0.38 + 0.75 * spread)
            sc.blobs.append(Blob(p[0], p[1], zp + 0.015 * s, rr, tuple(disc.ravel()),
                                 modes=((3, 0.08, 0.3 + 0.12 * age), (5, 0.035, 1.3 - 0.08 * age)),
                                 flat=(0.045 - 0.025 * spread) * remaining,
                                 w=TAU + (1.0 - TAU) * remaining, foam=0.018 * remaining))
            for j, edge in enumerate(exits):
                front = float(_ss(contact + j * 0.35, 11 + j * 0.5, age))
                tip = p + s * edge * front
                u = np.linspace(0.12, 1.0, 8)
                xy = p + (tip - p)[None, :] * u[:, None]
                xy[:, 0] += 0.018 * s * np.sin(u * math.pi) * math.sin(j + age * 0.2)
                z = hand.depth(k, xy[:, 0], xy[:, 1]) + 0.012 * s
                pts = np.c_[xy, z]
                arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
                width = (0.085 - 0.055 * u) * s * remaining * front
                if width.max() >= 0.5 and arc[-1] > 1.0:
                    # The receiving tongues stay shallow; the shared skin film supplies their broad reflections.
                    stream = _stream(pts, np.zeros_like(pts), width, np.zeros(8), np.zeros((8, 2)), arc,
                                     np.full(8, 0.035))
                    sc.blobs += [b._replace(stream=()) for b in stream]
        wet = spread * (1.0 - float(_ss(25, dry, age)))
        if wet > 0.001:
            across = math.atan2(un[1], un[0]) + 0.5 * math.pi
            wet_radius = R * ratio * (0.336 + 1.10 * spread * spread)
            sc.wet.append((p[0], p[1], wet_radius, 0.8 * wet_radius,
                           across, 0.14 * wet, 2.7, 0.0, 0.25))
            sc.coat = (p[0], p[1], wet_radius, wet, 1.0 - float(_ss(17, 36, age)),
                       tuple(tuple(edge * s / wet_radius) for edge in exits))
            for edge in exits:
                points = p + np.linspace(0.2, 1.0, 8)[:, None] * edge * s
                sc.band.append((points, np.linspace(0.055, 0.025, 8) * s, 0.16 * wet))
        if age >= 28:
            for j, (off, rb) in enumerate(zip(bead_offsets, bead_sizes)):
                left = max(0.0, 1.0 - (age - 28) / (57 + 3 * j))
                if left > 0:
                    xy = p + s * (off + np.array([0.015, 0.11]) * float(_ss(28, 62, age)))
                    z = float(hand.depth(k, xy[0:1], xy[1:2])[0])
                    sc.blobs.append(Blob(xy[0], xy[1], z + rb * s * 0.45, rb * s * math.sqrt(left),
                                         free=True, flat=0.59,
                                         glint=(0.10 + 0.20 * remaining) * left * float(_ss(24, 36, age))))
        for (born, life, off, vx, rb, height), (kl, land_xy, on_skin) in zip(spray, landings):
            tau = age - born
            if not 0 <= tau <= life + (3 if on_skin else 18):
                continue
            u = tau / life
            if 2.0 <= tau and (tau <= life or not on_skin):
                relative = off + pace * np.array([vx * tau, 0.08 * u - 4.0 * height * u * (1.0 - u)])
                velocity = pace * np.array([vx, (0.08 - 4.0 * height * (1.0 - 2.0 * u)) / life])
                xy, vel, sb, kb = airborne(born, tau, relative, velocity, k)
                pb = hand.p[kb] + sb * off
                z = float(hand.depth(kb, pb[0:1], pb[1:2])[0]) + sb * (0.04 + 0.30 * max(0.0, u * (1.0 - u)))
                radius = rb * sb * min(1.0, (life + 18 - tau) / 3.0)
                shp, move = _moving(radius, vel, 1.15, SHUTTER)
                sc.blobs.append(Blob(xy[0], xy[1], z, radius, shp, *vel, free=True, move=move, glint=0.55))
            elif tau > life and remaining > 0.05:
                xy = land_xy + p - hand.p[kl]
                sc.plops.append((xy[0], xy[1], (tau - life + 1) * 0.018 * s,
                                 0.12 * (1.0 - (tau - life) / 4.0)))
        for born, edge, rb in runoff:
            tau = age - born
            if not 0 <= tau < 23:
                continue
            outward = _unit(edge) * rb * 1.3
            # The fed stream lengthens more slowly than a free drop, keeping its broad root visibly attached.
            fed_age = min(tau, 14)
            free_age = max(0, tau - 14)
            fed_fall = 0.018 * fed_age + 0.002 * fed_age ** 2 + 0.074 * free_age + 0.007 * free_age ** 2
            relative = edge + outward + np.array([0.004 * tau * np.sign(edge[0]), fed_fall])
            xy, velocity, sb, kb = airborne(born, tau, relative,
                                           np.array([0.004 * np.sign(edge[0]),
                                                     0.018 + 0.004 * fed_age + 0.014 * max(0, tau - 14)]), k)
            lip = hand.p[kb] + sb * edge
            z = float(hand.depth(kb, lip[0:1], lip[1:2])[0]) + 0.04 * sb
            if tau >= 15:
                radius = rb * sb * min(1.0, (23 - tau) / 7.0)
                shp, move = _moving(radius, velocity, 1.0 + min(1.5, tau * 0.12), SHUTTER)
                sc.blobs.append(Blob(xy[0], xy[1], z, radius, shp, *velocity, free=True, move=move, glint=0.5))
            if tau < 15:
                # Water still feeds the falling bead from the lip before the thinning connection pinches apart.
                root = p + s * edge
                u = np.linspace(0.0, 1.0, 18)
                root_z = float(hand.depth(k, root[0:1], root[1:2])[0]) + 0.012 * s
                pts = np.c_[root + (xy - root)[None] * u[:, None], root_z + (z - root_z) * u]
                pts[:, 0] += 0.008 * s * np.sin(u * math.pi) * math.sin(age * 0.4)
                arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))]
                widths = rb * s * (0.85 - 0.32 * np.sin(math.pi * u) + 0.15 * u)
                widths *= 1.0 - 0.80 * float(_ss(9, 15, tau))
                vel = np.tile(np.r_[velocity, 0.0], (len(u), 1))
                sc.blobs += _stream(pts, vel, widths, np.full(len(u), 0.035), vel[:, :2], arc,
                                    0.60 + 0.25 * u)


def orb_scenes(tl: Timeline, track: dict, hand: Hand, size: tuple[int, int], seed: int,
               cam: np.ndarray | None, plate: np.ndarray | None = None,
               mask: np.ndarray | None = None) -> list[Scene]:
    """The orb, frame by frame (see the module's docstring and orb_plan). ``plate`` (the shot's frames, uint8 RGB, or
    None: every wall taken as plain) says whether what it is thrown at is a plain wall it stays on (_plain)."""
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
    span = max(g1 - g0, 1)
    gather = []  # the drops it gathers from: (angle, how far out (radii), size, depth off it, set off, arrive, spin)
    for _ in range(GATHER_DROPS):
        ta = g0 + span * (0.2 + 0.8 * math.sqrt(rng.uniform()))  # (more of them late, as the ball grows)
        trip = rng.uniform(*GATHER_TRIP) * span / GATHER
        ts = max(float(g0) - 0.5, ta - trip)
        d0 = max(1.2, rng.uniform(*GATHER_FROM) * (ta - ts) / trip)  # (one setting off late comes from nearer)
        gather.append((rng.uniform(0, 2 * math.pi), d0, rng.uniform(*GATHER_R), rng.uniform(-0.8, 0.8), ts, ta,
                       rng.choice([-1.0, 1.0])))
    g_seed = min(ta for *_, ta, _ in gather)  # the first drops reach it: its seed
    bub = np.stack([2.0 * math.pi * (np.arange(BUBBLES) + rng.uniform(-0.35, 0.35, BUBBLES)) / BUBBLES,
                    rng.uniform(*BUBBLE_LOOP[4], BUBBLES), rng.uniform(0.0, 2.0 * math.pi, BUBBLES),
                    rng.uniform(0.4, 1.1, BUBBLES), _sizes(rng, BUBBLES, *BUBBLE_R, 1.5)], 1)  # round the loop
    loop = (BUBBLE_LOOP[2] * float(rng.choice([-1.0, 1.0])), float(rng.uniform(-BUBBLE_LOOP[3], BUBBLE_LOOP[3])))
    sweep0 = float(rng.uniform(0.0, 2.0 * math.pi))  # where round that loop its highlight starts
    axis = _unit(np.array([rng.uniform(-1, 1), rng.uniform(-0.4, 0.4), rng.uniform(0.3, 1.0)]))
    phase = {l: (rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi), float(rng.choice([-1.0, 1.0])))
             for l in (3, 4, 5)}  # (each set of lobes: its place round it, its breathing's, and the way it runs)
    a_src = rng.uniform(0.0, 2.0 * math.pi) + 2.0 * math.pi * (np.arange(RIPPLES[0])
                                                         + rng.uniform(-0.2, 0.2, RIPPLES[0])) / RIPPLES[0]  # where
    # its trains of small waves run out of, round its outline...
    sources = _unit(np.c_[np.cos(a_src), np.sin(a_src), rng.uniform(-RIPPLES[5], RIPPLES[5], RIPPLES[0])])
    lams = rng.uniform(*RIPPLES[1], RIPPLES[0])  # ... their wavelengths (of its radius) and crests a second
    hzs = rng.uniform(*RIPPLES[3], RIPPLES[0])
    jolts = dict(plan.jolts)
    jolt_at: list[tuple[int, float]] = []  # (frame, size): for the lobes' ringing
    squash: list[tuple] = []  # (frame, size, the unit aim): the hits it squashes on
    rings: list[tuple] = []  # (frame, unit offset from the centre, size)
    spring = None
    m2, m2v = np.zeros(2), np.zeros(2)
    circ = 0.0
    modes: list[tuple] = []
    shape = np.eye(2)
    next_drip = g1 + int(rng.integers(*DRIP_EVERY))
    drip_side = float(rng.uniform(*DRIP_SIDE) * rng.choice([-1.0, 1.0]))
    fly = None  # the thrown orb in the air: (frame, centre, velocity, radius, the wall's depth);
    # centre None: burst
    falling: list[tuple] = []  # drops to let go off the wall later (_wall)
    cast = None  # thrown at a plain wall: (the frame it strikes it on, px there, its radius px there)
    landed: list[tuple] = []  # drips landed on the palm: (frame, px off the palm's point, radius)
    sat = None  # where it sat on the palm, left wet as it leaves: (frame, px off the palm's point, its radius, the
    # drops left there: (m, 2) px off that, (m,) radii)
    for k in range(g0, n):
        sc = scenes[k]
        p, nrm, s = hand.p[k], hand.nrm[k], float(hand.s[k])
        R = ORB_R * s
        for kf, xf, vf, rf, gf in falling:
            if kf == k:
                drops.add(k, xf, vf, rf, foam=0.05, g=gf)
        if sat is not None:  # the palm where it sat, wet, with a few drops left on it, carried on the palm, drying
            ks, off, rs_, d_off, d_r = sat
            age = k - ks
            on = float(_ss(-0.5, PALM_WET[2], age)) * 0.5 ** (age / (0.5 * PALM_WET[3]))
            rw = PALM_WET[1] * rs_
            across = math.atan2(float(nrm[1]), float(nrm[0])) + 0.5 * math.pi
            squash_ = max(abs(float(nrm[2])), 0.5)
            sc.wet.append((p[0] + off[0], p[1] + off[1], rw, rw * squash_, across, PALM_WET[0] * on, float(ks), 0.0,
                           PALM_WET[4]))
            if age >= 1:
                for j, ((dx_, dy_), radius) in enumerate(zip(d_off, d_r)):
                    dry = max(0.0, 1.0 - age / (70.0 + 5.0 * j))
                    r_ = radius * math.sqrt(dry)
                    if r_ < 0.1:
                        continue
                    slide = (0.05 + 0.025 * (j % 3)) * s * float(_ss(2.0, 50.0, age))
                    at = np.array([p[0] + off[0] + dx_ + 0.2 * slide,
                                   p[1] + off[1] + dy_ * squash_ + slide])
                    zl = float(hand.depth(k, np.float32([at[0]]), np.float32([at[1]]), fingers=False)[0])
                    sc.blobs.append(Blob(at[0], at[1], zl + 0.3 * r_, r_, free=True, flat=0.45,
                                         glint=0.3 * dry))
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
        if rel is None or k <= rel or plan.throw:
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
            if k in jolts:  # a hit while it is held squashes it along the aim and hops it, sends a ripple over it,
                # flares the light it focuses and flings drops out round it
                z = min(float(jolts[k]), 1.0)
                aim = _aim(tl, hand, k)
                ang = rng.uniform(0, 2 * math.pi)
                m2v += JOLT[0] * z * 2 * math.pi * MODE_HZ[2] * np.array([math.cos(ang), math.sin(ang)])
                jolt_at.append((k, z))
                squash.append((k, z, aim))
                spring.v[:2] += aim * HIT_SQUASH[4] * R * 2.0 * math.pi * ORB_SPRING[0] * 1.6
                off = _unit(rng.normal(size=2)) * rng.uniform(*RING_FROM)
                rings.append((k, np.array([off[0], off[1], math.sqrt(1.0 - off @ off)]), z))
                m = int(round(JOLT[1][0] + (JOLT[1][1] - JOLT[1][0]) * z))
                across = np.array([-aim[1], aim[0]])
                th_ = rng.uniform(0.0, 2.0 * math.pi, m)  # out round its middle, squeezed out across the aim...
                d = _unit(np.c_[np.outer(np.cos(th_), across) + np.outer(0.6 * np.sin(th_), aim),
                                rng.uniform(-0.5, 0.5, m)])
                v_ = (d + 0.35 * np.r_[aim, 0.0]) * rng.uniform(*JOLT[2], (m, 1)) * s / FPS  # ... thrown up with it
                wide = 1.0 / math.sqrt(1.0 - HIT_SQUASH[0][0] - (HIT_SQUASH[0][1] - HIT_SQUASH[0][0]) * z)  # (its
                # reach across, squashed)
                drops.add(k, c + d * R * (wide + 0.1), spring.v / FPS + v_, s * _sizes(rng, m, *JOLT[3], 1.5),
                          shutter=DROP_SHUTTER)  # (spat
                # from inside its rim, they sat on it like seeds)
            w2 = 2 * math.pi * MODE_HZ[2]
            drive = -acc[:2] / max(s, 1.0) * MODE2_DRIVE
            # Circulating water deforms even over a stationary hand. A slow forcing
            # of the existing damped bulk mode preserves volume and liquid inertia.
            circulate = .10 + .08 * float(e[k])
            drive += circulate * np.array([math.cos(t * 4.1 + seed), .7 * math.sin(t * 3.3 + seed)])
            m2v += (-(w2 ** 2) * m2 - 2 * MODE2_DAMP * w2 * m2v + drive * w2 ** 2) / FPS
            m2 += m2v / FPS
            if k < g1:  # a seed when the first drops reach it, growing evenly as the rest come in
                r = R * (GATHER_SEED + (1.0 - GATHER_SEED) * float(_ss(g_seed, g1, k))) if k >= g_seed - 0.5 else 0.0
            else:
                r = R
            vol = (r / R) ** 3
            st = float(np.hypot(*m2))
            u2 = m2 / st if st > 1e-6 else np.array([1.0, 0.0])
            k2 = 1.0 + min(MODE2_MAX, st)
            shape = np.asarray(_shape(u2, k2, 1.0 / math.sqrt(k2))).reshape(2, 2)
            for kj, zj, aj in squash:  # squashed along the aim on a hit, then springing back through a stretch
                age = k - kj
                if 0 <= age < 6 * HIT_SQUASH[3]:
                    ring_ = 1.0 if age < HIT_SQUASH[1] else (
                        math.cos(2.0 * math.pi * (age - HIT_SQUASH[1] + 1) / HIT_SQUASH[2])
                        * math.exp(-(age - HIT_SQUASH[1] + 1) / HIT_SQUASH[3]))
                    q_ = 1.0 - (HIT_SQUASH[0][0] + (HIT_SQUASH[0][1] - HIT_SQUASH[0][0]) * zj) * ring_
                    shape = np.asarray(_shape(aj, q_, 1.0 / math.sqrt(q_))).reshape(2, 2) @ shape
                    if age < 2:
                        sc.flare = max(sc.flare, 1.0 + HIT_FLARE[age] * zj)
            modes = []
            for l in (3, 4, 5):  # its lobes, breathing and running round it
                ph0, psi, way = phase[l]
                wl = 2 * math.pi * MODE_HZ[l]
                amp = (MODE_AMP[l] * (MODE_REST + (1 - MODE_REST) * float(e[k]))
                       * (MODE_BREATHE + (1 - MODE_BREATHE) * (0.5 + 0.5 * math.sin(wl * t + psi))))
                amp += sum(1.5 * MODE_AMP[l] * z * math.exp(-(k - kj) / 8.0) * math.sin(wl * (k - kj) / FPS)
                           for kj, z in jolt_at)
                modes.append((l, amp, ph0 + way * 2.0 * math.pi * MODE_RUN[l] / l * t))
            v_c = spring.v / FPS
            if r > 0.5:
                sc.blobs.append(Blob(c[0], c[1], c[2], r, tuple(shape.ravel()), v_c[0], v_c[1],
                                     (CIRCULATE[0] + CIRCULATE[1] * float(e[k])) / FPS, 0.0, tuple(modes),
                                     move=tuple(v_c[:2] * SHUTTER)))
                sc.held = (c[0], c[1], c[2], r, vol ** GATHER_LIGHT)  # its shade and its focus come on as it
                # gathers (at full strength once a third of it was there, it popped on with only a few drops in the
                # air; at its share of the water, the focus was white hot while the orb was still forming)
                sc.lobes = ((2, k2 - 1.0, math.atan2(u2[1], u2[0])), *modes)
                if k >= g1:
                    _orb_folds(sc, k, g1, c, r, shape, modes, seed, drops)
                sc.net = float(_ss(g1, g1 + SPOT_IN, k))
                slope = RIPPLES[2][0] + (RIPPLES[2][1] - RIPPLES[2][0]) * float(e[k])
                for u3, lam, hz in zip(sources, lams, hzs):  # its small waves, running out over it in trains from
                    # points carried round with its water (slower than it: they are on its skin)
                    u3 = _rotate(u3[None], axis, 0.35 * circ)[0]
                    sc.trains.append((c[0], c[1], r, *u3, slope * lam / (2.0 * math.pi * SIG_K ** 2) * vol, lam * r,
                                      lam * r * hz * t))
                if k >= g1 - 2:  # points of the key light off its ripples' crests, and a highlight carried round it
                    on = float(_ss(g1 - 2, g1 + 2, k))
                    sc.sparks = _sparks(seed, k, c, r, on)
                    sc.sweep = (c[0], c[1], r, loop[1], sweep0 + 2.0 * math.pi * t / SWEEP[0], on)
            for ang, d0, rs, z0, ts, ta, spin in gather:  # drops coming together out of the air, spiralling in
                if not ts <= k < ta:
                    continue
                u = (k - ts) / max(ta - ts, 1.0)
                rad = d0 * (1 - u) ** 1.2 * R  # drawn in, slowing as it takes them (rushing in at the end, smeared
                # into glass seeds round it)
                a = ang + spin * 1.3 * u
                dd = np.array([math.cos(a), math.sin(a)])
                pos = c[:2] + rad * dd
                vel = -dd * 1.2 * d0 * R * (1 - u) ** 0.2 / max(ta - ts, 1.0) + v_c[:2]
                rr = rs * s * float(_ss(0.0, 0.3, u))  # (condensing out of the air as it sets off)
                if rad > 0.9 * r and rr > 0.5:  # taken in as it touches it (drawn on over it, the drops arriving
                    # sat on the ball like glass seeds)
                    drop_shape, drop_move = _moving(rr, vel, 1.0, DROP_SHUTTER)
                    sc.blobs.append(Blob(pos[0], pos[1], c[2] + z0 * R * (1 - u), rr, drop_shape, vel[0], vel[1],
                                         free=True, move=drop_move))
            if k >= g1:  # air inside, carried round as it circulates; then a drip off its underside, now and then
                sc.bubbles += _air(bub, loop, circ, t, c, r, s, 1.0)
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
        elif not plan.throw:
            if k == rel:
                c = spring.x.copy() if spring is not None else np.r_[p, 0.0] + nrm * (R + ORB_GAP * s)
                _palm_spill(scenes, k, c, R, shape, modes, hand, seed, mask, plan.size, cam)
        elif fly is None or fly[1] is not None:  # thrown and not yet burst
            if fly is None:  # the measured thrust throws it along the aim
                c = spring.x if spring is not None else np.array([p[0], p[1], 0.0]) + nrm * (R + ORB_GAP * s)
                v0 = spring.v / FPS if spring is not None else np.zeros(3)
                aim = _aim(tl, hand, k)
                sp = THROW_SPEED * s / FPS * (0.6 + 0.4 * plan.size)
                v = 0.5 * v0 + np.array([aim[0] * sp, aim[1] * sp, -RECEDE * sp])
                zp = float(hand.depth(k, np.float32([p[0]]), np.float32([p[1]]))[0])
                fly = (k, c.copy(), v, R, zp - ROOM_DEPTH * max(H, W))
                for tq in range(1, 4 * FPS):  # where and when it will strike the wall
                    sq = float(hand.s[min(n - 1, k + tq)])
                    cq = c + v * (tq * tq / (2 * TAKEOFF) if tq < TAKEOFF else tq - TAKEOFF / 2) + np.array(
                        [0.0, 0.5 * G * sq / FPS ** 2 * tq * tq, 0.0])
                    kq = ZCAM * sq / max(ZCAM * sq - (min(cq[2], fly[4]) - c[2]), 0.2 * ZCAM * sq)
                    xq = np.array([W / 2.0, H / 2.0]) + (cq[:2] - np.array([W / 2.0, H / 2.0])) * kq
                    if min(xq[0] + R * kq, W + R * kq - xq[0], xq[1] + R * kq, H + R * kq - xq[1]) < 0:
                        break
                    if cq[2] <= fly[4]:
                        if _plain(None if plate is None or k + tq >= n else plate[k + tq], xq[0], xq[1], R * kq):
                            cast = (k + tq, xq, R * kq)
                        break
                left = (c.copy(), R)  # where it left the palm: its shade fades there
                foot = c[:2] - nrm[:2] * (R + ORB_GAP * s)
                m = int(rng.integers(PALM_DROPS[0][0], PALM_DROPS[0][1] + 1))
                d_off = rng.normal(size=(m, 2)) * np.array([0.35, 0.16]) * PALM_DROPS[2] * R
                sat = (k, foot - p, R, d_off, s * rng.uniform(*PALM_DROPS[1], m))
            k0, c0, v, R0, zw = fly
            tau = k - k0
            gone = hit = False
            if cast is not None and 0 < cast[0] - k <= WALL_CAST[0]:  # its shadow on the wall it will strike
                sc.cast.append((cast[1][0], cast[1][1], cast[0] - k, cast[2],
                                WALL_CAST[3] * float(_ss(WALL_CAST[0] + 1.0, 1.0, cast[0] - k))))
            go = tau * tau / (2 * TAKEOFF) if tau < TAKEOFF else tau - TAKEOFF / 2
            c = c0 + v * go + np.array([0.0, 0.5 * G * s / FPS ** 2 * tau * tau, 0.0])
            vel = v * min(1.0, tau / TAKEOFF) + np.array([0.0, G * s / FPS ** 2 * tau, 0.0])
            hit = c[2] <= zw
            if hit:
                c[2] = zw
            kz = ZCAM * s / max(ZCAM * s - (c[2] - c0[2]), 0.2 * ZCAM * s)
            xy = np.array([W / 2.0, H / 2.0]) + (c[:2] - np.array([W / 2.0, H / 2.0])) * kz
            gone = min(xy[0] + R0 * kz, W + R0 * kz - xy[0], xy[1] + R0 * kz, H + R0 * kz - xy[1]) < 0
            done = hit or gone
            hold = max(0.0, 1.0 - tau / (TAKEOFF + 2))
            sc.held = (left[0][0], left[0][1], left[0][2], left[1], hold)
            if done and not gone:  # burst: from here on only its drops are left
                at = np.array([xy[0], xy[1], c[2]])
                if _plain(None if plate is None else plate[k], xy[0], xy[1], R0 * kz):  # on the wall: it
                    # spreads over it the way it was thrown and stays there, draining, dripping (flung off it as
                    # pieces in the air, it was soap bubbles and shards that all went at once)
                    fan = v[:2] / max(float(np.hypot(*v[:2])), 1e-6)
                    falling += _wall(scenes, drops, (k, xy[0], xy[1], c[2], R0 * kz, s * kz, fan, plan.size), n, cam,
                                     rng)
                else:  # on what is no plain wall (a coat rack, shelves): it bursts flat against it and its drops
                    # fall, leaving nothing on it
                    sc.blobs += splash(drops, k, at, R0 * kz, vel, plan.size, s * kz, rng, np.array([0.0, 0.0, 1.0]))
            if done:
                fly = (k0, None, v, R0, zw)
            else:  # airborne: slight stretch and a detached, dispersed trail, without a neck bridging back to the palm
                fade = max(0.0, 1.0 - tau / TAKEOFF)  # its wobble and its air go as it is flung (dropped on the
                # frame it was let go, the ball went still and empty at once)
                vs = vel[:2] * kz
                sp = float(np.hypot(*vs))
                u = vs / max(sp, 1e-6)
                stretch = 1.0 + min(THROWN_STRETCH, sp / (2.5 * R0))
                drop_shape, drop_move = _moving(R0 * kz, vs, stretch, THROWN_SHUTTER,
                                               (THROWN_LONG - stretch) * 2.0 * R0 * kz)
                sc.blobs.append(Blob(xy[0], xy[1], c[2], R0 * kz, drop_shape, vs[0], vs[1], CIRCULATE[0] / FPS,
                                     THROWN_FOAM[0] * min(1.0, 2.0 * (1.0 - fade)),
                                     modes=tuple((l, a * fade, ph) for l, a, ph in modes), move=drop_move,
                                     lead=1.0))  # (sharp at its front, smeared behind)
                if tau >= 1:  # the key light mirrored along its flight
                    sc.streak = (xy[0], xy[1], u[0], u[1], float(np.hypot(*drop_move)) if drop_move else 0.0, R0 * kz,
                                 min(1.0, tau / TAKEOFF))
                if fade > 0:  # (its air, small waves and points of light go with its wobble)
                    sc.bubbles += _air(bub, loop, circ, t, xy, R0 * kz, s * kz, fade)
                    slope = RIPPLES[2][0] + (RIPPLES[2][1] - RIPPLES[2][0]) * float(e[k])
                    for u3, lam, hz in zip(sources, lams, hzs):
                        u3 = _rotate(u3[None], axis, 0.35 * circ)[0]
                        sc.trains.append((xy[0], xy[1], R0 * kz, *u3, slope * lam / (2.0 * math.pi * SIG_K ** 2) * fade,
                                          lam * R0 * kz, lam * R0 * kz * hz * t))
                    sc.sparks = _sparks(seed, k, xy, R0 * kz, fade)
                    sc.sweep = (xy[0], xy[1], R0 * kz, loop[1], sweep0 + 2.0 * math.pi * t / SWEEP[0], fade)
                if tau >= 1:  # drops torn off its back by the air, flung out in a cone behind it
                    m = int(rng.integers(TRAIL[0][0], TRAIL[0][1] + 1))
                    f_ = rng.uniform(*TRAIL[2], (m, 1))
                    ca = rng.uniform(-TRAIL[3], TRAIL[3], m)
                    back = np.c_[np.cos(ca) * -u[0] - np.sin(ca) * -u[1], np.sin(ca) * -u[0] + np.cos(ca) * -u[1]]
                    distance = rng.uniform(1.0, 2.8, (m, 1))
                    tp = xy[None, :] + back * R0 * kz * distance
                    vt = f_ * vs[None, :] + back * (1.0 - f_) * sp * 0.5  # (slower than it, thrown out off its back)
                    life = (c[2] - zw) / np.maximum(-f_[:, 0] * vel[2], 1e-3)  # (they wet the wall where it does)
                    drops.add(k, np.c_[tp, np.full(m, c[2])], np.c_[vt, f_[:, 0] * vel[2]],
                              R0 * kz * _sizes(rng, m, *TRAIL[1], 1.5) * (1.2 - 0.25 * distance[:, 0]),
                              foam=0.05, life=life, fade=2,
                              shutter=DROP_SHUTTER)
        sc.blobs += drops.blobs(k)
        drops.step()
    return scenes


def _plain(frame: np.ndarray | None, x: float, y: float, R: float) -> bool:
    """Whether the plate ``frame`` (uint8 RGB, or None: taken as plain) round px (``x``, ``y``) is a plain, light
    surface water striking it there would lie on, as a wall is (WALL_PLAIN; ``R``: the water's radius px)."""
    if frame is None:
        return True
    n_r, least, light, spread, (step, share) = WALL_PLAIN
    H, W = frame.shape[:2]
    e = max(n_r * R, least)
    y0, y1, x0, x1 = max(0, int(y - e)), min(H, int(y + e) + 1), max(0, int(x - e)), min(W, int(x + e) + 1)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return False
    g = cv2.cvtColor(np.ascontiguousarray(frame[y0:y1, x0:x1]), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    q = cv2.resize(g, (max(2, g.shape[1] // 4), max(2, g.shape[0] // 4)), interpolation=cv2.INTER_AREA)
    mag = np.hypot(cv2.Sobel(q, cv2.CV_32F, 1, 0), cv2.Sobel(q, cv2.CV_32F, 0, 1)) / 8.0
    return float(g.mean()) > light and float(g.std()) < spread and float((mag > step).mean()) < share


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
    rising WHIP_HEIGHT out of the palm, bent towards the picture's up (WHIP_UP), and coiling round that way as it rises
    (WHIP_COIL, its coil opening over WHIP_RISE of it, flattened towards the picture: WHIP_FLAT), ``sweep`` turns round
    it from ``phase`` rad, its lobes running round it (``t`` seconds in): a tendril of water standing out of the palm,
    swaying side to side."""
    s = float(hand.s[k])
    axis = _unit(hand.nrm[k] + WHIP_UP * np.array([0.0, -1.0, 0.0]))
    e1 = _unit(np.cross(axis, np.array([0.0, 0.0, 1.0])) + 1e-9)
    e2 = np.cross(axis, e1)
    th = phase + 2.0 * math.pi * sweep * u
    rho = WHIP_COIL * s * _ss(0.0, WHIP_RISE, u)
    rho = rho * (1.0 + WHIP_WOBBLE[0] * np.sin(WHIP_WOBBLE[1] * th - 2.0 * math.pi * WHIP_WOBBLE[2] * t))
    coil = rho[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    coil[:, 2] *= WHIP_FLAT
    return root + np.outer(WHIP_HEIGHT * s * u, axis) + coil


def _stream(pts: np.ndarray, vel: np.ndarray, rad: np.ndarray, foam: np.ndarray, flow: np.ndarray,
            arc: np.ndarray, flat: np.ndarray, lie: np.ndarray | None = None, root_feather: float = 0.0) -> list[Blob]:
    """Blobs along a stream through ``pts`` (m, 3) moving ``vel`` (px a frame), its radius ``rad``, ``arc`` px along
    it on the picture, ``flat`` its thickness against a tube's (Blob.flat), ``lie`` how much of it lies on the skin
    (Blob.lie): at most WHIP_SPACING of their radius apart, each weighted by its share of the stream (_tube_weight), so
    the stream is a tube of that radius, blurred by its motion over the shutter. ``root_feather`` picture pixels of
    its root blend into the pool as a thin meniscus, leaving the pool's refraction to carry contact with the palm."""
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
        if root_feather > 0.0:
            w *= 0.04 + 0.96 * float(_ss(0.0, root_feather, arc[i] - arc[0]))
        out.append(Blob(pts[i, 0], pts[i, 1], pts[i, 2], rad[i], shape, flow[i, 0], flow[i, 1], foam=float(foam[i]),
                        w=w, move=move, stream=(float(arc[i]), float(t[0]), float(t[1])), flat=float(flat[i]),
                        lie=0.0 if lie is None else float(lie[i])))
    return out


def _necked(arc: np.ndarray, rad: np.ndarray, neck: np.ndarray, jit: np.ndarray, flowed: float) -> tuple:
    """Where a stream necks into beads (``neck`` 0..1 along it, ``arc`` px along it, its radius ``rad``): beads
    WHIP_BEAD[0] of its radii apart, at no set spacing (``jit``, a smooth noise along it), carried out along it with
    its water, ``flowed`` px so far: (its radius, thinned between them, the indices of the points they sit at, their
    radii: each holds its length of the stream, as far as it has necked)."""
    lam = np.maximum(WHIP_BEAD[0] * rad * (1.0 + WHIP_BEAD[2] * np.clip(jit, -1.5, 1.5)), 1.0)  # (evenly spaced, a
    ph = np.r_[0.0, np.cumsum(np.diff(arc) / lam[1:])] - flowed / float(np.mean(lam))  # barber's pole)
    beads = np.flatnonzero((np.diff(np.floor(ph)) > 0) & (neck[1:] > 0.05)) + 1
    rb = (0.75 * rad[beads] ** 2 * lam[beads]) ** (1 / 3) * np.sqrt(neck[beads])
    seen = rb >= 0.5  # (under half a px, no bead)
    return rad * (1.0 - WHIP_BEAD[1] * neck), beads[seen], rb[seen]


def _skin(matte: np.ndarray, s: float) -> tuple[np.ndarray, int]:
    """The hand's matte (uint8, one frame) as SKIN reads it: SKIN[1] times smaller, softened over SKIN[2] hand sizes;
    (that, how many times smaller)."""
    q = SKIN[1]
    h, w = matte.shape[:2]
    m = cv2.resize(matte, (max(1, w // q), max(1, h // q)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return cv2.GaussianBlur(m, (0, 0), max(0.5, SKIN[2] * s / q)), q


def _off_hand(hand: Hand, k: int, xy: np.ndarray, skin: tuple | None = None) -> np.ndarray:
    """(m,) 0..1: how far off the hand points ``xy`` (m, 2) px are on frame ``k``, as the picture shows it (OFF_HAND):
    by how far outside the hand they are, the hand being the outline round its palm's landmarks and its fingers' bones
    as capsules (by the nearest landmark alone, the middle of the palm counted as half off it; by the outline round all
    its landmarks, the room between a finger and the wrist counted as on it), or with none by their distance from its
    palm's point; and, given its matte (``skin``, _skin), wherever the picture shows no skin there (SKIN)."""
    s = float(hand.s[k])
    xy = np.asarray(xy, np.float64)
    if hand.off is None:
        off = _ss(OFF_HAND[1][0] * s, OFF_HAND[1][1] * s, np.hypot(*(xy - hand.p[k]).T))
    else:
        L = hand.off[k][:, :2].astype(np.float64)
        hull = cv2.convexHull(L[list(PALM_LM) + [2]].astype(np.float32))
        out = np.array([-cv2.pointPolygonTest(hull, (float(a), float(b)), True) for a, b in xy])
        a_, b_ = L[list(BONES[0])], L[list(BONES[1])]  # its fingers' bones
        ab = b_ - a_
        f = np.clip(((xy[:, None, :] - a_[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0.0,
                    1.0)
        bone = np.hypot(*(xy[:, None, :] - a_[None] - f[..., None] * ab[None]).transpose(2, 0, 1)).min(1)
        off = _ss(OFF_HAND[0][0] * s, OFF_HAND[0][1] * s, np.minimum(out, bone - OFF_HAND[2] * s))
    if skin is not None:
        m, q = skin
        on = _remap(m, (xy[:, 0] + 0.5) / q - 0.5, (xy[:, 1] + 0.5) / q - 0.5, cv2.BORDER_CONSTANT)
        off = np.maximum(off, _ss(SKIN[0][1], SKIN[0][0], on))
    return off


def _pool(hand: Hand, k: int, r: float, t: float, out: np.ndarray, parts: np.ndarray,
          v: np.ndarray) -> tuple[list, tuple]:
    """The pool of radius ``r`` on the palm the stream rises out of on frame ``k`` (``t`` seconds in), a thin film flat
    on it (seen squashed as the palm turns from the camera), drawn out a little along ``out`` (the unit direction the
    stream leaves it in, in the picture) and lying more that way, so the stream widens into it rather than ending in it;
    an irregular puddle, flat pools run into it (``parts``: (angle, distance and size of its radius, turns a second)
    each, WHIP_POOL_PARTS), its outline rippling, carried with the palm at `v` px a frame: (its blobs, the wet skin
    round it on the palm: Scene.wet)."""
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
    blobs = [Blob(c[0], c[1], c[2], r, tuple(shape.ravel()), v[0], v[1], foam=WHIP_POOL[2], modes=modes,
                  flat=WHIP_POOL[1])]  # (still, not turning: turned, the patterns its flow carries wound into rings)
    for ang, dist, size, turns in parts:  # flat pools run into it (on the palm: squashed with it)
        th = ang + 2.0 * math.pi * turns * t
        off = disc @ np.array([math.cos(th), math.sin(th)]) * dist * r
        blobs.append(Blob(c[0] + off[0], c[1] + off[1], c[2], size * r, tuple(disc.ravel()), v[0], v[1],
                          foam=0.5 * WHIP_POOL[2], flat=WHIP_POOL[1],
                          modes=((2, 0.5 * a * math.sin(2 * math.pi * 1.3 * t + ang), ang - t),)))
    wet = (c[0], c[1], WHIP_WET[0] * r, WHIP_WET[0] * r * squash, math.atan2(un[1], un[0]) + 0.5 * math.pi,
           WHIP_WET[1], 0.0, 0.0, WHIP_WET[2])
    return blobs, wet


@dataclass
class Flung:
    """The stream's water a hit flings out along the aim (the lash), node by node from its root to its tail, in the
    picture's px at the depth it was flung from (seen in perspective from there)."""

    k: int  # the hit's frame
    x: np.ndarray  # (N, 3) px and depth
    v: np.ndarray  # (N, 3) px a frame
    u: np.ndarray  # (N,) how far along it, 0 at its root to 1 at its tail
    r: np.ndarray  # (N,) radius px, as it was flung
    seg: np.ndarray  # (N,) px of the stream each node stands for, as it was flung
    breaks: np.ndarray  # (N,) frames after the hit each node's water breaks into drops (inf: it falls back)
    broken: np.ndarray  # (N,) bool
    size: float  # how hard
    wall: float  # the room's wall's depth
    struck: bool = False  # it has struck the wall


def _fling(k: int, x: np.ndarray, v: np.ndarray, aim: np.ndarray, size: float, s: float, wall: float,
           size_hw: tuple) -> Flung:
    """The stream (nodes ``x`` (N, 3) px, moving ``v`` px a second) flung on frame ``k`` by a hit ``size`` strong
    along ``aim`` (unit, in the picture): its tail fastest, its root still, curling round from its root's way, its
    tail swinging out the way the picture (``size_hw``) has the more room, and away from the camera into the room
    (LASH_*); tapering from LASH_R's root to its tail."""
    u = np.linspace(0.0, 1.0, len(x))
    sp = LASH_SPEED[0] * (LASH_SPEED[1] + (1.0 - LASH_SPEED[1]) * min(size, 1.0)) * s * u ** LASH_SPEED[2]

    def room(a: float) -> float:  # px from its root to the picture's edge along the aim turned ``a`` rad
        d = np.array([aim[0] * math.cos(a) - aim[1] * math.sin(a), aim[0] * math.sin(a) + aim[1] * math.cos(a)])
        lim = [(size_hw[1 - i] - x[0, i] if d[i] > 0 else x[0, i]) / max(abs(d[i]), 1e-6) for i in (0, 1)]
        return min(lim)

    way = 1.0 if room(0.5 * LASH_CURL) >= room(-0.5 * LASH_CURL) else -1.0
    ang = LASH_CURL * way * (u - 0.5)  # (its tail swung out towards the picture's nearer edge, it stood up as a rod)
    d = np.c_[aim[0] * np.cos(ang) - aim[1] * np.sin(ang), aim[0] * np.sin(ang) + aim[1] * np.cos(ang)]
    vel = np.c_[d * sp[:, None], -LASH_RECEDE * sp] + 0.3 * v / FPS
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))]
    seg = np.maximum(np.gradient(arc), 1e-3)
    r = (s * (LASH_R[0] * (1.0 - u) ** 1.2 + LASH_R[1])
         * (LASH_FOOT[0] + (1.0 - LASH_FOOT[0]) * _ss(0.0, LASH_FOOT[1], u)))
    breaks = np.where(u >= LASH_BREAK[2], LASH_BREAK[0] + (1.0 - u) / (1.0 - LASH_BREAK[2]) * LASH_BREAK[1], np.inf)
    return Flung(k, x.copy(), vel, u, r, seg, breaks, np.zeros(len(x), bool), size, wall)


def _lash(F: Flung, k: int, root: np.ndarray, s: float, size_hw: tuple, thrown: Drops, scenes: list, n: int,
          cam: np.ndarray | None, rng, width: np.ndarray, cell: float, frame: np.ndarray | None = None) -> tuple:
    """The flung stream on frame ``k``: moved on a frame (gravity, and the air holding back a jet tearing into drops,
    LASH_DRAG), its nodes breaking into drops from its tail back (LASH_BREAK), necked into beads joined by threads just
    before (LASH_NECK; ``width``, a smooth noise its beads are spaced by, its features ``cell`` hand sizes), the drops
    flying on (``thrown``) and, if they reach the room's wall, striking it as a wet streak (_wall, LASH_WALL); what is
    left at its root falls back and drains into the pool. Its root rides the palm (``root``; ``s`` a hand's size px,
    ``size_hw`` the picture's). Returns (its blobs, drops to let go later off the wall streak)."""
    H, W = size_hw
    age = k - F.k
    g = G * s / FPS ** 2
    zcam = ZCAM * s
    F.v[:, 1] += g
    F.v *= DRAG
    F.v[~F.broken, :2] *= LASH_DRAG / DRAG
    F.x += F.v
    F.x[0] = root
    ctr = np.array([W / 2.0, H / 2.0])
    kz = zcam / np.maximum(zcam - (F.x[:, 2] - root[2]), 0.2 * zcam)
    xy = ctr + (F.x[:, :2] - ctr) * kz[:, None]
    stretch = np.maximum(np.linalg.norm(np.gradient(F.x, axis=0), axis=1) / F.seg, 1.0)
    rad = F.r / stretch ** 0.35 * kz
    falling = []
    new = np.flatnonzero(~F.broken & (age >= F.breaks))
    if new.size:  # each length of it beads up into a drop with a few satellites, flying on
        F.broken[new] = True
        rb = np.minimum(LASH_DROPS[0] * rad[new], (0.75 * rad[new] ** 2 * F.seg[new] * kz[new]) ** (1 / 3))
        vz = F.v[new, 2]
        life = np.where(vz < -1e-3, (F.x[new, 2] - F.wall) / np.maximum(-vz, 1e-3), np.inf)  # (gone into the wall)
        tang = _unit(np.gradient(xy, axis=0)[new] + 1e-9)
        sizes, drawn, joined, threads, shift = LASH_BEADS  # uneven, drawn out, some still threads, not evenly spaced
        side = np.c_[-tang[:, 1], tang[:, 0]]
        off = (tang * (F.seg[new] * kz[new] * rng.uniform(-shift, shift, new.size))[:, None]
               + side * (rad[new] * rng.uniform(-1.0, 1.0, new.size))[:, None])
        p3 = np.c_[xy[new] + off, F.x[new, 2]]
        v3 = np.c_[F.v[new, :2] * kz[new, None]
                   + side * (rad[new] * rng.uniform(-0.6, 0.6, new.size))[:, None], vz]
        thrown.add(k, p3, v3, rb * rng.uniform(*sizes, new.size), foam=0.05, life=life + 1.0, fade=1.0,
                   stretch=np.where(rng.random(new.size) < joined, rng.uniform(*threads, new.size),
                                    rng.uniform(*drawn, new.size)), shutter=0.3)
        m, (lo, hi) = LASH_DROPS[1], LASH_DROPS[2]
        for j in range(m):
            off = (tang * (F.seg[new] * kz[new] * rng.uniform(-0.6, 0.6, new.size))[:, None]
                   + side * (rad[new] * rng.uniform(-1.5, 1.5, new.size))[:, None])
            vs = v3.copy()
            vs[:, :2] += side * (rad[new] * rng.uniform(-0.8, 0.8, new.size))[:, None]
            thrown.add(k, np.c_[xy[new] + off, F.x[new, 2]], vs * rng.uniform(0.9, 1.1, (new.size, 1)),
                       rb * rng.uniform(lo, hi, new.size), life=life + 1.0, fade=1.0, shutter=0.3)
        if age == 2:  # a little sideways mist softens the break without adding a row of bright points
            mi = rng.choice(new, 15)
            mt = _unit(np.gradient(xy, axis=0)[mi] + 1e-9)
            ms = np.c_[-mt[:, 1], mt[:, 0]]
            mv = F.v[mi, :2] * kz[mi, None] + ms * rng.uniform(-1.1, 1.1, (15, 1)) * s / FPS
            thrown.add(k, np.c_[xy[mi] + ms * rad[mi, None] * rng.uniform(-0.8, 0.8, (15, 1)), F.x[mi, 2]],
                       np.c_[mv, F.v[mi, 2]], np.clip(s * rng.uniform(0.003, 0.008, 15), 0.5, 1.4),
                       life=rng.uniform(3, 6, 15), fade=2.0, shutter=0.25)
    m = int(round(LASH_SPRAY * min(F.size, 1.0) / 2)) if age <= 1 else 0
    if m:  # flung out, its tail tears into threads and drops flung on ahead of it and away from it
        i = rng.choice(np.flatnonzero(F.u > 0.6), m)
        lead = _unit(F.v[i, :2] + 1e-9)
        vz = F.v[i, 2] * rng.uniform(*WHIP_TORN[1], m)
        life = np.where(vz < -1e-3, (F.x[i, 2] - F.wall) / np.maximum(-vz, 1e-3), np.inf)
        thrown.add(k, np.c_[xy[i] + lead * rad[i, None], F.x[i, 2]],
                   np.c_[F.v[i, :2] * kz[i, None] * rng.uniform(*WHIP_TORN[1], (m, 1))
                         + lead * rng.uniform(1.0, 3.0, (m, 1)) * s / FPS, vz],
                   s * _sizes(rng, m, *WHIP_TORN[0], 1.5), life=np.minimum(life, rng.uniform(3, 7, m)), fade=2,
                   stretch=np.where(rng.random(m) < WHIP_THREADS, rng.uniform(1.2, 1.8, m), 1.0), shutter=0.3)
    out = F.u > 0.6  # its outer part striking the wall: a wet streak there, drawn out the way it went, if the plate
    # (``frame``) shows a plain wall there (_plain; else its drops are gone into what is there)
    if not F.struck and F.broken[out].all() and float(F.x[out, 2].mean()) <= F.wall:
        F.struck = True
        kw = float(kz[out].mean())
        at = xy[out].mean(0)
        way = F.v[out, :2].mean(0)
        rw = LASH_WALL[0] * s * kw * (0.6 + 0.4 * F.size) ** (1 / 3)
        if _plain(frame, at[0], at[1], rw):
            falling = _wall(scenes, thrown, (k, at[0], at[1], F.wall, rw, s * kw,
                                             way / max(float(np.hypot(*way)), 1e-6), F.size), n, cam, rng,
                            LASH_WALL[1], bead_count=8)
    whole = np.flatnonzero(~F.broken)
    if whole.size < 2:
        return [], falling
    last = int(whole.max()) + 1  # from its root out to where it has broken
    drain = 1.0 - float(_ss(LASH_BACK[0] - 1.0, LASH_BACK[0] + 1.0, age))  # (what is left falls back into the pool)
    pts = _chain(np.c_[xy[:last], F.x[:last, 2]], 4)
    vel = _chain(np.c_[F.v[:last, :2] * kz[:last, None], F.v[:last, 2]], 4)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))]
    q = np.linspace(0.0, last - 1.0, len(pts))
    r_ = np.interp(q, np.arange(last), rad[:last]) * drain
    neck = np.interp(q, np.arange(last), np.clip((age - (F.breaks[:last] - LASH_NECK)) / LASH_NECK, 0.0, 1.0))
    jit = np.interp(((arc / s) / cell + 0.5 * len(width)) % (len(width) - 1), np.arange(len(width)), width)
    r_, beads, rb = _necked(arc, r_, neck, jit, 0.0)
    foam = WHIP_FOAM[0] + WHIP_FOAM[2] * _ss(0.3, 1.0, q / max(last - 1.0, 1.0))
    blobs = _stream(pts, vel, r_, foam, vel[:, :2], arc,  # (lying flat on the palm at its root, as the pool does)
                    1.0 - (1.0 - WHIP_POOL[1]) * (1.0 - _ss(0.0, LASH_FOOT[1], q / max(len(F.u) - 1.0, 1.0))))
    for i, rr in zip(beads, rb):
        shape, move = _moving(rr, vel[i, :2])
        blobs.append(Blob(pts[i, 0], pts[i, 1], pts[i, 2], rr, shape, vel[i, 0], vel[i, 1], foam=float(foam[i]),
                          move=move))
    return blobs, falling


def whip_scenes(tl: Timeline, track: dict, hand: Hand, size: tuple[int, int], seed: int, cam: np.ndarray | None,
                matte: np.ndarray | None = None, plate: np.ndarray | None = None) -> list[Scene]:
    """The whip, frame by frame (see the module's docstring and whip_plan): each node of the stream is held to the
    swirl it makes over the palm (_swirl) by a spring, stiff at the root and soft at the tail, so when the hand moves
    the stream trails it; a hit flings its water out along the aim (_fling, _lash), and it pours out of the pool again
    after. It rises out of a thin pool on the palm, swells and thins along its length as bulges run down it, thins to
    its tail and frays there into beads flowing out along it, and sprays off its outer part as it swirls; fast, it
    froths, sheds drops and sprays off its tail. It is worked in the picture's px (the hand and a first-person camera
    move together); the drops it sheds fly in the room's. ``matte`` (n, H, W) uint8, if given, says where the hand's
    skin is under it (_off_hand); ``plate`` (the shot's frames, uint8 RGB), if given, whether what its lash strikes is a
    plain wall it stays on (_plain)."""
    n = tl.n
    H, W = size
    scenes = [Scene() for _ in range(n)]
    plan = whip_plan(tl, track)
    if plan.start >= n:
        return scenes
    rng = _rng(seed, _WHIP)
    e = np.clip(_tsmooth(tl.energy, VOICE_SMOOTH), 0, 1)
    drops = Drops(cam, size, float(np.median(hand.s)))
    dripping = Drops(cam, size, float(np.median(hand.s)))  # its drips, falling free of it
    thrown = Drops(cam, size, float(np.median(hand.s)))  # the water a hit flings, flying free
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
    sway = float(prng.uniform(0.0, 2.0 * math.pi))  # the side its flare leans to, swaying
    srng = _rng(seed, _WHIP, 2)  # the drops it spills and leaves on the palm
    cracks = dict(plan.cracks)
    heard = sounding(tl)
    poured = plan.start  # the frame it last began to pour out of the pool (again after each hit flings its water)
    drained = None  # the frame its pool last began to drain (a hit flung its water, or it let go)
    full = None  # the last frame its pool was there (from then on the skin it lay on dries)
    left = None  # drops left on the palm when its pool drains: ((m, 2) px off the palm's point, (m,) radii)
    spilt: list[tuple] = []  # where drops spilling off it land: (frame, px off the palm's point there)
    flung: list[Flung] = []
    falling: list[tuple] = []  # drops to let go off a wall streak later (_wall)
    for k in range(plan.start, n):
        sc = scenes[k]
        p, nrm, s = hand.p[k], hand.nrm[k], float(hand.s[k])
        zp = float(hand.depth(k, np.float32([p[0]]), np.float32([p[1]]))[0])
        root = np.array([p[0], p[1], zp]) + nrm * WHIP_LIFT * s
        alive = plan.end is None or k < plan.end
        broke = 0 if alive else k - plan.end  # frames since it let go
        t = k / FPS
        for kf, xf, vf, rf, gf in falling:
            if kf == k:
                thrown.add(k, xf, vf, rf, foam=0.05, g=gf)
        if alive and k in cracks and x is not None and len(x) > 1:  # a hit flings its water out along the aim
            flung.append(_fling(k, x, v, _aim(tl, hand, k), cracks[k], s, zp - ROOM_DEPTH * max(H, W), size))
            x, poured, drained = None, k + LASH_BACK[1], k
        if plan.end is not None and k == plan.end:  # let go: its pool drains
            drained = k
        if drained == k:  # (a few drops left on the palm as it drains)
            m = int(srng.integers(WHIP_LEFT[0][0], WHIP_LEFT[0][1] + 1))
            rl = WHIP_POOL[0] * WHIP_R[0] * s
            left = (srng.normal(size=(m, 2)) * np.array([0.35, 0.16]) * WHIP_LEFT[2] * rl,
                    s * srng.uniform(*WHIP_LEFT[1], m))
        grow = float(_ss(poured, poured + WHIP_GROW, k))
        pooled = float(_ss(plan.start, plan.start + WHIP_GROW, k))
        fill = 1.0 if drained is None else 1.0 - float(_ss(drained, drained + WHIP_DRAIN, k))  # (drained to a film in
        fill = max(fill, grow if alive else 0.0)  # a few frames, it fills again as the stream pours out again)
        lead = x[min(3, len(x) - 1), :2] - x[0, :2] if x is not None and len(x) > 1 else hand.along[k, :2]
        lead = _unit(np.asarray(lead, np.float64) + 1e-9)
        pool_r = WHIP_POOL[0] * WHIP_R[0] * s * max(pooled, 0.3) ** 0.5 * fill
        if pool_r > 0.5:  # the pool on the palm
            blobs, _ = _pool(hand, k, pool_r, t, lead, parts, hand.p[k] - hand.p[max(k - 1, 0)])
            sc.blobs += blobs
        if fill > 0.5:
            full = k
        if full is not None:  # the skin under and round it wet, drying once it has drained
            _, wet = _pool(hand, k, WHIP_POOL[0] * WHIP_R[0] * s * max(pooled, 0.3) ** 0.5, t, lead, parts,
                           hand.p[k] - hand.p[max(k - 1, 0)])
            sc.wet.append(wet[:5] + (wet[5] * 0.5 ** ((k - full) / (0.5 * PALM_WET[3])),) + wet[6:])
        if left is not None and fill < 0.5:  # the drops left on it once its pool has drained, carried on the palm
            age = k - drained
            for j, ((dx_, dy_), radius) in enumerate(zip(*left)):
                dry = max(0.0, 1.0 - age / (70.0 + 5.0 * j))
                r_ = radius * math.sqrt(dry)
                if r_ < 0.1:
                    continue
                slide = (0.05 + 0.025 * (j % 3)) * s * float(_ss(2.0, 50.0, age))
                at = np.array([p[0] + dx_ + 0.2 * slide, p[1] + dy_ + slide])
                zl = float(hand.depth(k, np.float32([at[0]]), np.float32([at[1]]), fingers=False)[0])
                sc.blobs.append(Blob(at[0], at[1], zl + 0.5 * r_, r_ * float(_ss(drained, drained + WHIP_DRAIN, k)),
                                     free=True, flat=0.45, glint=0.3 * dry))
        for kl, off in spilt:  # a ring running out where a spilt drop lands in the pool
            age = k - kl
            if 0 <= age < WHIP_SPILL[4][0] and pool_r > 0.5:
                sc.plops.append((p[0] + off[0], p[1] + off[1], WHIP_SPILL[4][1] * s * (age + 1) / WHIP_SPILL[4][0],
                                 WHIP_SPILL[4][2] * (1.0 - age / WHIP_SPILL[4][0])))
        sweep = WHIP_SWEEP[0] + (WHIP_SWEEP[1] - WHIP_SWEEP[0]) * float(e[k])
        phase += 2.0 * math.pi * (WHIP_SPIN[0] + WHIP_SPIN[1] * float(e[k])) / FPS
        sc.wave = WAVE[0] + WAVE[1] * float(e[k])
        body = None  # where the stream is on this frame (px), for the drops it sheds
        if x is None and k >= poured and alive and (k == plan.start or heard[min(n - 1, k + WHIP_GROW)]):  # (poured
            # out again only if the voice sounds on as long as it takes: after its last hit, out for a moment and let
            # go, it was a stutter)
            x, v, root0 = _swirl(hand, k, root, u_n * 0.02, phase, sweep, t), np.zeros((N, 3)), root
            poured = k
        if x is not None and broke <= WHIP_BREAK and len(x):
            u_t = u_n * max(grow, 0.02)  # poured out: the nodes lie along the part of the swirl out so far
            target = _swirl(hand, k, root, u_t, phase, sweep, t)
            om = 2.0 * math.pi * (WHIP_HZ[0] + (WHIP_HZ[1] - WHIP_HZ[0]) * u_n ** 0.7)
            skin = None if matte is None else _skin(matte[k], s)
            off = _off_hand(hand, k, x[:, :2], skin)  # it sags more where it hangs off the hand (held up as stiffly
            sag = (WHIP_SAG[0] + (WHIP_SAG[1] - WHIP_SAG[0]) * off) * G * s  # there, it was a hose)
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
            rad = s * (WHIP_R[0] * (1 - u) ** 0.68 + WHIP_R[1]) * grow ** 0.5
            rad *= 1.0 - WHIP_TIP * _ss(0.8, 1.0, u)  # its tail thins to a point (it ended in a rounded worm's head)
            flare = (WHIP_FLARE[0] * (1.0 - _ss(0.0, WHIP_FLARE[1], u)) * rad  # it rises out of the pool, flaring
                     * (1.0 + WHIP_FLARE[4] * np.sin(2.0 * math.pi * (u / WHIP_FLARE[1] + WHIP_FLARE[3] * t) + sway)))
            rad = rad + flare  # unevenly, and more to one side, the side swaying
            lead = pts[min(6, len(pts) - 1), :2] - pts[0, :2]
            side = np.array([-lead[1], lead[0]]) / max(float(np.hypot(*lead)), 1e-6)
            pts = pts.copy()
            pts[:, :2] += np.outer(WHIP_FLARE[2] * math.sin(2.0 * math.pi * WHIP_FLARE[3] * t + sway) * flare, side)
            bulge = np.interp(((arc / s - WHIP_FLOW * t) / cell) % m_sw, np.arange(m_sw + 1), swell)
            rad *= 1.0 + np.clip(WHIP_SWELL[0] * bulge, -WHIP_SWELL[2], WHIP_SWELL[2]) * _ss(0.0, 0.15, u)  # (formed
            # as it flows out: swelling where it rises out of the pool, it was a bulb on it)
            pulse_amp, pulse_len, pulse_period = WHIP_PULSE
            pulse_gap = WHIP_FLOW * pulse_period
            pulse = ((arc / s - WHIP_FLOW * t + 0.5 * pulse_gap) % pulse_gap) - 0.5 * pulse_gap
            rad *= (1.0 + pulse_amp * np.exp(-(pulse / (0.5 * pulse_len)) ** 2) * _ss(0.0, 0.12, u)
                    * (1.0 - _ss(0.62, 0.85, u)))
            wide = np.interp(((arc / s - WHIP_WIDTH[2] * WHIP_FLOW * t) / (WHIP_WIDTH[1] / 4.0)) % m_sw,
                             np.arange(m_sw + 1), width)
            rad *= 1.0 + np.clip(WHIP_WIDTH[0] * wide, -WHIP_WIDTH[3], WHIP_WIDTH[3]) * _ss(0.0, 0.1, u)  # (even all
            # along, a tube)
            offp = _off_hand(hand, k, pts[:, :2], skin)
            zs = hand.depth(k, pts[:, 0].astype(np.float32), pts[:, 1].astype(np.float32))
            over = (pts[:, 2] - zs) / np.maximum(rad, 1.0)  # its radii in front of the hand's plane
            lie = np.zeros(len(pts))  # how much of it lies on the skin (Blob.lie)
            if alive:  # lying on the skin, it spreads into a rivulet, unevenly (from its root: its flared foot lying
                # along the palm, taken as held up, was a clear glass hook; swung round behind the hand it lies on
                # nothing)
                lie = ((1.0 - offp) * _ss(RIVULET[4][1], RIVULET[4][0], over) * _ss(-1.0, 0.0, over)
                       * (1.0 - _ss(0.5 * WHIP_LIES, WHIP_LIES, u)))
                spread = rad * (1.0 + lie * np.maximum(RIVULET[0] + RIVULET[1] * np.clip(wide, -1.5, 1.5), 0.0))
                rad = np.minimum(spread, np.maximum(rad, RIVULET[5] * s))
                # hanging off the hand, it swells into a drip at its lowest point, necks and lets go of it, now and then
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
                        rad *= 1.0 - WHIP_DRIP[2] * ph_ ** 2 * np.exp(-((da + 1.6) / 0.6) ** 2)  # ... necking above it
                # fast, it sheds drops off its leading side, and its tail sprays
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
                elif grow > 0.8 and (k - plan.start) % 6 == 0:  # a held stream still sheds its moving tip, four drops
                    # a second, with a little lateral velocity; a fixed round tip looked like the end of a glass rod
                    tangent = _unit(pts[-1] - pts[-3] + 1e-9)
                    dv = tangent * (0.5 * WHIP_FLOW * s / FPS) + rng.normal(size=3) * (0.15 * s / FPS)
                    drops.add(k, pts[-1] + tangent * rad[-1], vel[-1] + dv,
                              s * rng.uniform(0.008, 0.016), life=7.0, fade=2.0, shutter=0.3)
                # swirling, its outer part sprays out of the swirl
                mist = np.flatnonzero((u > WHIP_MIST[0]) & (rng.random(len(u)) < WHIP_MIST[1] * N / len(u)))
                if mist.size and grow > 0.5:
                    away = _unit(np.c_[pts[mist, :2] - root[:2], np.zeros(mist.size)] + 1e-9)
                    drops.add(k, pts[mist] + away * rad[mist, None], vel[mist] + away * rng.uniform(
                        *WHIP_MIST[3], (mist.size, 1)) * s / FPS, s * _sizes(rng, mist.size, *WHIP_MIST[2], 1.5),
                              foam=0.0, life=rng.uniform(4, 9, mist.size), fade=2)
                coil = np.flatnonzero((u > WHIP_SPILL[2][0]) & (u < WHIP_SPILL[2][1]))
                if coil.size and grow > 0.5 and float(speed.max()) < WHIP_SHED:  # held, drops spill off its coils
                    # and fall back into the pool, each running out in a ring on the palm where it lands
                    m = int(srng.integers(*WHIP_SPILL[0]))
                    i = srng.choice(coil, m)
                    away = _unit(np.c_[pts[i, :2] - root[:2], np.zeros(m)] + 0.4 * srng.normal(size=(m, 3)) + 1e-9)
                    v0 = 0.5 * vel[i] + away * srng.uniform(*WHIP_SPILL[3], (m, 1)) * s / FPS
                    g_ = G * float(np.median(hand.s)) / FPS ** 2
                    y_ = p[1] + srng.uniform(-0.5, 0.5, m) * pool_r  # where in the pool's depth each lands
                    drop_ = np.maximum(y_ - pts[i, 1], 0.0)  # px down to there in the picture
                    T = np.clip((-v0[:, 1] + np.sqrt(v0[:, 1] ** 2 + 2.0 * g_ * drop_)) / g_, 1.0, 24.0)
                    dripping.add(k, pts[i] + away * rad[i, None], v0, s * _sizes(srng, m, *WHIP_SPILL[1], 1.5),
                                 life=T, fade=1.0, shutter=DROP_SHUTTER)
                    for T_, x_, yl in zip(T, pts[i, 0] + v0[:, 0] * T, y_):
                        if abs(x_ - p[0]) < 1.2 * pool_r and srng.random() < WHIP_SPILL[4][3]:  # (in the pool)
                            kl = k + int(round(T_))
                            spilt.append((kl, np.array([x_, yl]) - hand.p[min(n - 1, kl)]))
            jit = np.interp(((arc / s - WHIP_FLOW * t) / cell + 0.5 * m_sw) % m_sw, np.arange(m_sw + 1), width)
            neck = np.maximum(WHIP_FRAY[1] * _ss(WHIP_FRAY[0], 1.0, u) * (1.0 - _ss(0.4, 1.0, speed / WHIP_SHED)),
                              broke / WHIP_BREAK if not alive else 0.0)  # (fast, its tail is drawn out whole: its
            # beads, each smeared across the frame, were stacked like a staircase)
            rad, beads, rb = _necked(arc, rad, neck, jit, WHIP_FLOW * s * t)
            if not alive and broke == WHIP_BREAK:  # necked through: beads, falling
                drops.add(k, pts[beads], vel[beads] + rng.normal(size=(len(beads), 3)) * 0.02 * s, rb)
                x = x[:0]
            else:
                tang = _unit(np.gradient(pts, axis=0))
                foam = (WHIP_FOAM[0] + WHIP_FOAM[1] * _ss(0.75, 1.0, u) + WHIP_FOAM[3] * (1.0 - _ss(0.0, 0.15, u))
                        + WHIP_FOAM[2] * np.clip((speed - 4.0) / 8.0, 0, 1))
                out_ = _ss(0.0, 0.2, u)[:, None]  # (in the pool it moves with the palm: flowing out of it at full
                # speed, and swirled round with the stream, the patterns it carries wound into rings there, a whorl)
                fl = vel[:1, :2] + out_ * (vel[:, :2] - vel[:1, :2] + tang[:, :2] * WHIP_FLOW * s / FPS)
                root_flat = 1.0 - (1.0 - WHIP_POOL[1]) * (1.0 - _ss(0.0, WHIP_FLARE[1] * 1.5, u))  # (it rises out of
                # the pool as flat as the pool is, widening into it: as round as a tube there, it ended in a bulb)
                sc.blobs += _stream(pts, vel, rad, foam, fl, arc2, root_flat, lie,
                                    root_feather=max(20.0, 0.08 * s))
                if alive:
                    body = pts[::3, :2]
                if alive and (k - plan.start) % WHIP_TRAIL[2] == 0:  # where it lies on the palm, the skin gets wet
                    at = np.flatnonzero(np.diff(np.floor(arc / (WHIP_TRAIL[3] * s))) > 0) + 1
                    zh = hand.depth(k, pts[at, 0].astype(np.float32), pts[at, 1].astype(np.float32))
                    d_ = (pts[at, 2] - zh) / np.maximum(rad[at], 1.0)  # (in front of the palm, not swung round
                    wets = at[(offp[at] < 0.3) & (d_ > -0.5) & (d_ < REST[1]) & (u[at] > 0.05)]  # behind it)
                    if wets.size:
                        trail.append((k, pts[wets, :2] - p, WHIP_TRAIL[4] * rad[wets] + WHIP_TRAIL[5] * s))
                t2 = tang[:, :2] / np.maximum(np.linalg.norm(tang[:, :2], axis=1, keepdims=True), 1e-9)
                for i, r_ in zip(beads, rb):
                    shape, move = _moving(r_, vel[i, :2])
                    sc.blobs.append(Blob(pts[i, 0], pts[i, 1], pts[i, 2], r_, shape, fl[i, 0], fl[i, 1],
                                         foam=float(foam[i]), move=move,
                                         stream=(float(arc2[i]), float(t2[i, 0]), float(t2[i, 1])),
                                         lie=float(lie[i])))
                for ua, across, ar in air:  # air carried out along it
                    i = int(np.searchsorted(u, (ua + WHIP_FLOW * s * t / max(arc[-1], 1.0)) % 1.0 * 0.85))
                    i = min(i, len(pts) - 1)
                    nx_, ny_ = -tang[i, 1], tang[i, 0]
                    q_ = math.hypot(nx_, ny_) + 1e-9
                    sc.bubbles.append((pts[i, 0] + across * rad[i] * nx_ / q_, pts[i, 1] + across * rad[i] * ny_ / q_,
                                       ar * rad[i], WHIP_AIR_SEEN * float(_ss(0.0, 0.1, u[i]))
                                       * (1.0 - float(_ss(*WHIP_AIR_FAST, np.hypot(*vel[i, :2]))))))
        for F in flung:  # the water the hits have flung
            blobs, fall = _lash(F, k, root, s, size, thrown, scenes, n, cam, rng, width, cell,
                                None if plate is None else plate[k])
            sc.blobs += blobs
            falling += fall
        flung = [F for F in flung if k - F.k < LASH_GONE and not (F.struck and F.broken.all())]
        flying = drops.blobs(k)
        if body is not None and flying:  # while it pours, what it sheds is gone by the time it is far from it (flying
            # on across the frame, a shower of rain)
            xy = np.array([[b_.x, b_.y] for b_ in flying])
            far = np.sqrt(((xy[:, None, :] - body[None]) ** 2).sum(-1)).min(1)
            keep = 1.0 - _ss(WHIP_CULL[0] * s, WHIP_CULL[1] * s, far)
            flying = [b_._replace(r=b_.r * float(q_)) for b_, q_ in zip(flying, keep) if b_.r * q_ >= 0.5]
        sc.blobs += flying + dripping.blobs(k) + thrown.blobs(k)
        drops.step()
        dripping.step()
        thrown.step()
        trail = [tr for tr in trail if k - tr[0] < WHIP_TRAIL[1]]
        for kt, off, rw in trail:  # the skin it has lain on, wet round it, drying
            sc.band.append((p + off, rw, WHIP_TRAIL[0] * (1.0 - (k - kt) / WHIP_TRAIL[1])))
        spilt = [sp_ for sp_ in spilt if k - sp_[0] < WHIP_SPILL[4][0]]
        if (not alive and (x is None or not len(x)) and not len(drops.x) and not len(dripping.x)
                and not len(thrown.x) and pool_r <= 0.5 and not trail and not flung and not spilt
                and (full is None or k - full > 3 * PALM_WET[3])):
            break
    return scenes


# ---- the shot -------------------------------------------------------------------------------------------------------

GRAIN_UNIT = 0.943  # what fx_light's grain reading reads off unit white noise: the plate's grain is laid at its strength


def frame(k: int, plate_k: np.ndarray, matte: np.ndarray | None, sc: Scene, hand: Hand, room: Room, noise: Noise,
          code: float, seed: int) -> np.ndarray:
    """Frame ``k`` with its water: uint8 RGB, the plate's own bytes wherever there is none."""
    out = plate_k.copy()
    if not sc.blobs and not sc.wet and not sc.band and not sc.film and not sc.plops and not sc.cast and sc.coat is None:
        return out
    H, W = plate_k.shape[:2]
    P = lin(plate_k)
    boxes = (_wet(P, sc.wet, matte, room.lit(k), _unit(room.key(k, (sc.wet or [(W / 2.0, H / 2.0)])[0][:2])[:2]),
                  sc.band, sheen=sc.coat is None) if sc.wet or sc.band else [])
    if sc.coat is not None and matte is not None:
        boxes += _palm_gloss(P, sc.coat, matte, hand, room, noise, k, plate_k)
    if sc.plops:
        boxes += _plops(P, sc.plops, matte, room.lit(k), _unit(room.key(k, sc.plops[0][:2])[:2]))
    if sc.cast:
        boxes += _cast(P, sc.cast, matte, room, k)
    if sc.film:
        boxes += _film(P, sc.film, matte, room, k, noise)
    # A separate optical layer lets the falling stream join inside the reservoir without a second lens collar.
    ribbon = set(sc.ribbon)
    visible = [b for b in sc.blobs if b not in ribbon and not (b.free and sc.coat is not None and sc.held is not None
               and b.z < sc.held[2] + 0.5 * sc.held[3]
               and math.hypot(b.x - sc.held[0], b.y - sc.held[1]) < sc.held[3] + b.r)]
    fld = Field(visible, H, W) if visible else None
    pyr = _pyramid(P) if fld is not None or sc.ribbon else None
    layer = None if fld is None or fld.box is None else shade(fld, sc, k, hand, room, noise, pyr, matte, code, seed)
    if sc.ribbon:
        stream_field = Field(sc.ribbon, H, W)
        stream_scene = Scene(ribbon=sc.ribbon, wave=sc.wave, coat=sc.coat)
        stream_layer = shade(stream_field, stream_scene, k, hand, room, noise, pyr, matte, code, seed, falling=True)
        if stream_layer is not None:
            boxes.append(stream_layer.box)
            ya, yb, xa, xb = stream_layer.box
            P[ya:yb, xa:xb] = P[ya:yb, xa:xb] * (1.0 - stream_layer.alpha[..., None]) + stream_layer.colour
            if layer is not None and sc.held is not None:
                y0, y1, x0, x1 = layer.box
                ylo, yhi, xlo, xhi = max(y0, ya), min(y1, yb), max(x0, xa), min(x1, xb)
                if yhi > ylo and xhi > xlo:
                    yy, xx = np.mgrid[ylo:yhi, xlo:xhi]
                    cx, cy, _, rr, _ = sc.held
                    down = _unit(np.array([sc.ribbon[-1].x - cx, sc.ribbon[-1].y - cy]))
                    along = ((xx - cx) * down[0] + (yy - cy) * down[1]) / max(rr, 1.0)
                    join = stream_layer.alpha[ylo - ya:yhi - ya, xlo - xa:xhi - xa] * _ss(0.35, 0.80, along)
                    sl = np.s_[ylo - y0:yhi - y0, xlo - x0:xhi - x0]
                    # Open the reservoir's bottom rim into the stream instead of leaving a closed glass lip over it.
                    layer.alpha[sl] *= 1.0 - join
                    layer.colour[sl] *= (1.0 - join)[..., None]
    if layer is not None:
        boxes.append(layer.box)
        if matte is not None:
            boxes.append(caustic(P, layer, k, hand, room, matte, sc.held, sc.lobes, noise, sc.net, sc.flare))
        y0, y1, x0, x1 = layer.box
        P[y0:y1, x0:x1] = P[y0:y1, x0:x1] * (1.0 - layer.alpha[..., None]) + layer.colour
        if sc.sparks:
            boxes += glitter(P, sc.sparks, layer, room.lit(k), room.key(k, (sc.held or (W / 2, H / 2))[:2]))
        lines = [] if sc.sweep is None else [_sweep_line(sc.sweep, room.key(k, sc.sweep[:2]))]
        if sc.streak is not None:
            lines += _streak_lines(sc.streak, room.key(k, sc.streak[:2]))
        if lines:
            boxes += _lines(P, lines, layer, room.lit(k))
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


def _palm_radius(u: np.ndarray, v: np.ndarray, phase: float = 2.7) -> np.ndarray:
    """One irregular contact front for both the film reflection and the wet skin underneath it."""
    theta = np.arctan2(v, u)
    front = (1.0 + 0.08 * np.cos(theta + 0.22 * phase) + 0.055 * np.cos(2.0 * theta + 0.78 * phase)
             + 0.12 * np.cos(3.0 * theta + phase) + 0.07 * np.cos(5.0 * theta + 1.7 * phase)
             + 0.04 * np.cos(8.0 * theta + 2.9 * phase))
    return np.hypot(u, v) / np.clip(front, 0.72, 1.28)


def _palm_gloss(P: np.ndarray, coat: tuple, matte: np.ndarray, hand: Hand, room: Room, noise: Noise,
                k: int, skin: np.ndarray) -> list[tuple]:
    """Keep one shallow, rimmed volume visible on the palm before it drains into three connected exit channels.
    The sheet and its exits share surface relief and refraction; late glints follow the original skin creases."""
    x, y, r, wet, flowing = coat[:5]
    H, W = P.shape[:2]
    s = float(hand.s[k])
    exits = coat[5] if len(coat) > 5 else ()
    reach = max(1.32, max((math.hypot(*exit_) + 0.20 for exit_ in exits), default=0.0))
    y0, y1 = max(0, int(y - reach * r)), min(H, int(y + reach * r) + 2)
    x0, x1 = max(0, int(x - reach * r)), min(W, int(x + reach * r) + 2)
    if min(y1 - y0, x1 - x0) < 3:
        return []
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = (xx - x) / r, (yy - y) / r
    drift = np.full_like(yy, 0.008 * s * flowing)
    # Coordinates relative to the palm keep the drained film attached when the camera or the hand moves.
    ripple = noise.flow(13, xx - x, yy - y, np.zeros_like(xx), drift, k, 0.13 * s)
    fine = noise.flow(17, xx - x, yy - y, np.zeros_like(xx), drift, 0, 0.045 * s)
    angle = math.atan2(float(hand.nrm[k, 1]), float(hand.nrm[k, 0])) + 0.5 * math.pi
    u, v = dx * math.cos(angle) + dy * math.sin(angle), (-dx * math.sin(angle) + dy * math.cos(angle)) / 0.8
    radius = _palm_radius(u, v)
    sheet = float(_ss(0.28, 0.82, flowing))
    draining = float(_ss(0.02, 0.20, flowing) * (0.35 + 0.65 * (1.0 - _ss(0.45, 0.98, flowing))))
    # One shallow volume first covers the skin, then separates into unequal draining channels. Only their curved
    # surfaces reflect: no painted white paths, and the palm's original texture remains under the water.
    channels = np.zeros_like(xx)
    if exits:
        routes = []
        for j, (ex, ey) in enumerate(exits):
            eu, ev = ex * math.cos(angle) + ey * math.sin(angle), (-ex * math.sin(angle) + ey * math.cos(angle)) / 0.8
            routes.append((math.atan2(ev, eu), math.hypot(eu, ev) + 0.07, 0.16 - 0.020 * j, 0.035 * (-1) ** j))
    else:
        routes = PALM_CHANNELS
    for theta, length, width, bend in routes:
        along = u * math.cos(theta) + v * math.sin(theta)
        across = -u * math.sin(theta) + v * math.cos(theta) - bend * np.sin(3.0 * along)
        taper = _ss(-0.12, 0.12, along) * _ss(length, length - 0.08, along)
        taper *= 0.75 + 0.25 * np.sin(8.0 * along + 2.0 * theta)
        channels += np.exp(-(across / (width * (1.0 - 0.45 * np.clip(along, 0.0, 1.0)))) ** 2) * taper
    height = s * ((0.0005 + 0.007 * sheet) * ripple + 0.0002 * fine
                  + 0.024 * sheet * _ss(1.04, 0.55, radius) + 0.018 * draining * channels)
    gy, gx = np.gradient(cv2.GaussianBlur(height.astype(np.float32), (0, 0), max(0.6, 0.0015 * s)))
    patch = P[y0:y1, x0:x1]
    footprint = _ss(1.025, 0.96, radius)
    coverage = (wet ** 1.3 * np.maximum(footprint, _ss(0.04, 0.45, channels) * float(_ss(0.01, 0.15, draining)))
                * matte[y0:y1, x0:x1].astype(np.float32) / 255.0)
    # A shallow sheet displaces the real palm folds by a few capture pixels. This survives phone reduction as
    # liquid thickness while preserving the skin's detail and the matte's silhouette.
    bend_x = np.clip(-24.0 * gx, -6.0, 6.0)
    bend_y = np.clip(-24.0 * gy, -6.0, 6.0)
    warped = _remap(P, xx + bend_x, yy + bend_y)
    patch += (sheet + 0.8 * draining * np.clip(channels, 0.0, 1.0))[..., None] * coverage[..., None] * (warped - patch)
    lum = lin(skin[y0:y1, x0:x1]) @ fx._LUMA
    # Resolve the palm folds above the phone codec's fine texture, which would otherwise turn into glitter.
    skin_log = cv2.GaussianBlur(np.log(np.maximum(lum, 0.01)), (0, 0), max(0.7, 0.016 * s))
    relief = skin_log - cv2.GaussianBlur(skin_log, (0, 0), max(2.0, 0.055 * s))
    sy, sx = np.gradient(relief)
    hyy, _ = np.gradient(sy)
    hxy, hxx = np.gradient(sx)
    curvature = np.sqrt((hxx - hyy) ** 2 + 4.0 * hxy * hxy)
    high, low = 0.5 * (hxx + hyy + curvature), 0.5 * (hxx + hyy - curvature)
    angle_bank = 0.5 * np.arctan2(2.0 * hxy, hxx - hyy)
    bx, by = np.cos(angle_bank), np.sin(angle_bank)
    slope = sx * bx + sy * by
    offset = slope / np.maximum(high, 1e-6)
    ridge = _ss(6.0, 24.0, (high - np.abs(low)) * s * s)
    support = (radius < 1.03) & (matte[y0:y1, x0:x1] > 128)
    folds = np.uint8((ridge > 0.15) & (np.abs(offset) < 0.014 * s) & support)
    # Only a fold that stays aligned over a short span can mirror a connected sliver of the room light.
    length = max(3, int(round(0.085 * s))) | 1
    aligned = np.zeros_like(folds)
    mid = length // 2
    for theta in np.linspace(0.0, math.pi, 8, endpoint=False):
        kernel = np.zeros((length, length), np.uint8)
        vx, vy = int(round(mid * math.cos(theta))), int(round(mid * math.sin(theta)))
        cv2.line(kernel, (mid - vx, mid - vy), (mid + vx, mid + vy), 1, 1)
        aligned |= cv2.morphologyEx(folds, cv2.MORPH_OPEN, kernel)
    reach = max(1, int(round(0.012 * s))) | 1
    folds = cv2.morphologyEx(aligned, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(folds, connectivity=8)
    # Keep a few extended folds, not isolated pores. Their geometry is measured anew from the original skin.
    candidates = [j for j in range(1, count)
                  if max(stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]) > 0.12 * s]
    keep = sorted(candidates, key=lambda j: int(stats[j, cv2.CC_STAT_AREA]), reverse=True)[:3]
    joined = cv2.GaussianBlur(np.isin(labels, keep).astype(np.float32) * ridge,
                              (0, 0), max(0.7, 0.007 * s))
    normal = hand.nrm[k]
    # The cupped center faces the camera more than the plane through the wrist and knuckles.
    nx_base, ny_base = normal[0] + 0.30 * dx - gx, normal[1] + 0.30 * dy - gy
    nx = nx_base - np.clip(0.32 * s * sx, -0.40, 0.40)
    ny = ny_base - np.clip(0.32 * s * sy, -0.40, 0.40)
    nz = np.full_like(nx, max(float(normal[2]) + 0.15, 0.55))
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    L = room.key(k, (x, y))
    # A light-facing bank lies beside the measured valley, preserving a dark crease under the reflection.
    shift = np.float32([[1.0, 0.0, -0.006 * s * L[0]], [0.0, 1.0, -0.006 * s * L[1]]])
    crease = cv2.warpAffine(joined, shift, (joined.shape[1], joined.shape[0]), flags=cv2.INTER_LINEAR)
    half = _unit(L + np.array([0.0, 0.0, 1.0]))
    ndh = np.clip((nx * half[0] + ny * half[1] + nz * half[2]) / norm, 0.0, 1.0)
    envelope = coverage
    # The whole wet surface mirrors the same light. The sharper lobe is confined to long, light-facing fold banks.
    reflection = _ss(-0.65, 0.50, ripple)
    broad_norm = np.sqrt(nx_base * nx_base + ny_base * ny_base + nz * nz)
    broad = np.clip((nx_base * half[0] + ny_base * half[1] + nz * half[2]) / broad_norm, 0.0, 1.0)
    bank_facing = _ss(0.0001, 0.0010, -(sx * L[0] + sy * L[1]))
    sharp = (ndh ** 110 * crease * bank_facing).astype(np.float32)
    count, labels, _, _ = cv2.connectedComponentsWithStats(np.uint8(sharp > 0.025), connectivity=8)
    energy = np.bincount(labels.ravel(), weights=sharp.ravel(), minlength=count)
    bright = sorted(range(1, count), key=lambda j: energy[j], reverse=True)[:3]
    # The light's finite extent broadens three tapered banks into glossy patches, not a network of white strokes.
    sharp *= np.isin(labels, bright)
    sharp = 3.0 * cv2.GaussianBlur(sharp, (0, 0), max(0.7, 0.010 * s))
    shoulder = 0.10 * ndh ** 45 * crease * bank_facing
    glints = 1.0 - 0.90 * float(_ss(0.05, 0.50, flowing))
    specular = (0.28 * broad ** 70 * _ss(1.0, 0.50, radius) + glints * (shoulder + 6.0 * sharp)) * (0.75 + 0.25 * reflection)
    liquid = (sheet * (0.10 + 0.12 * reflection) * broad ** 5 * _ss(1.0, 0.75, radius)
              + draining * channels * (0.45 + 0.45 * reflection) * (0.15 + 0.85 * ndh ** 4))
    specular += liquid
    # A connected light-facing meniscus survives phone reduction, then drains into the measured crease banks.
    Lxy = max(math.hypot(float(L[0]), float(L[1])), 1e-6)
    facing = (dx * L[0] + dy * L[1] / 0.8) / np.maximum(np.hypot(dx, dy / 0.8) * Lxy, 1e-5)
    edge = np.exp(-((radius - 0.965) * r / max(1.2, 0.022 * s)) ** 2)
    theta = np.arctan2(v, u)
    arcs = 0.35 + 0.65 * _ss(0.05, 0.75, np.cos(3.0 * theta + 0.6) + 0.3 * np.cos(5.0 * theta + 2.0))
    specular += (1.20 * sheet * edge * (0.16 + 0.84 * _ss(-0.25, 0.65, facing)) * arcs
                 * (0.65 + 0.35 * reflection) * (1.0 - 0.85 * draining * np.clip(channels, 0.0, 1.0)))
    specular = 0.32 * (1.0 - np.exp(-specular / 0.32)) * envelope
    contact_edge = np.exp(-((radius - 0.990) * r / max(0.9, 0.007 * s)) ** 2)
    thickness = (sheet * 0.06 * np.exp(-((radius - 0.91) / 0.07) ** 2)
                 * (1.0 - _ss(-0.25, 0.65, facing)) * arcs * (0.25 + 0.75 * reflection))
    thickness += 0.025 * draining * np.clip(channels, 0.0, 1.0)
    patch *= (1.0 - (0.10 * sheet * contact_edge + thickness) * envelope)[..., None]
    light = room.lit(k)
    film_light = 0.35 * light + 0.65 * float(light.max())
    patch += specular[..., None] * film_light
    return [(y0, y1, x0, x1)]


def _cast(P: np.ndarray, items: list, matte: np.ndarray | None, room: Room, k: int) -> list[tuple]:
    """The thrown orb's shadow on the wall it is about to strike (Scene.cast), in place: for each (x, y, frames to go,
    radius, strength) a soft disc of shade lying off where it will strike, away from the key light, the further and the
    bigger and softer the more frames there are to go (WALL_CAST), so it closes in on the ball as the ball closes in on
    the wall; nothing on the hand, which is in front of the wall. Returns the boxes it changed."""
    H, W = P.shape[:2]
    boxes = []
    for x, y, ahead, r, strength in items:
        away = -_unit(room.key(k, (x, y))[:2].astype(np.float64))
        cx, cy = x + away[0] * WALL_CAST[1] * ahead * r, y + away[1] * WALL_CAST[1] * ahead * r
        rr = r * (1.0 + WALL_CAST[2] * ahead)
        soft = rr * (0.25 + 0.1 * ahead)
        e = rr + 3.0 * soft
        y0, y1, x0, x1 = max(0, int(cy - e)), min(H, int(cy + e) + 1), max(0, int(cx - e)), min(W, int(cx + e) + 1)
        if y1 <= y0 or x1 <= x0 or strength <= 0.005:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        shade = strength * _ss(rr + 1.5 * soft, rr - 1.5 * soft, np.hypot(xx - cx, yy - cy))
        if matte is not None:
            shade = shade * (1.0 - matte[y0:y1, x0:x1].astype(np.float32) / 255.0)
        P[y0:y1, x0:x1] *= (1.0 - shade)[..., None]
        boxes.append((y0, y1, x0, x1))
    return boxes


def _plops(P: np.ndarray, items: list, matte: np.ndarray | None, lit: np.ndarray, key: np.ndarray) -> list[tuple]:
    """Rings running out in the pool where drops land (Scene.plops), in place: each (x, y, radius, strength) the key
    light mirrored in the ripple, on its side towards the light (``key``, a unit direction in the picture) alone (lit
    all round, they were bubbles), only on the hand (its matte). Returns the boxes it changed."""
    if matte is None:
        return []
    H, W = P.shape[:2]
    boxes = []
    for x, y, rad, strength in items:
        e = int(rad + 4)
        y0, y1, x0, x1 = max(0, int(y - e)), min(H, int(y + e) + 1), max(0, int(x - e)), min(W, int(x + e) + 1)
        if y1 <= y0 or x1 <= x0 or strength <= 0.01:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.hypot(xx - x, yy - y)
        side = ((xx - x) * key[0] + (yy - y) * key[1]) / np.maximum(d, 1.0)
        ring = strength * np.exp(-((d - rad) / 1.3) ** 2) * _ss(0.1, 0.8, side)
        P[y0:y1, x0:x1] += (ring * matte[y0:y1, x0:x1].astype(np.float32) / 255.0)[..., None] * lit
        boxes.append((y0, y1, x0, x1))
    return boxes


def _wet(P: np.ndarray, patches: list, matte: np.ndarray | None, lit: np.ndarray, key: np.ndarray,
         bands: list = (), sheen: bool = True) -> list[tuple]:
    """Wet skin on the hand, in place: patches (Scene.wet), each a soft ellipse, uneven unless ``soft`` (the share of
    its radius its edge feathers over, and how round it is), and ``bands`` (Scene.band), each a feathered band through
    its points; where the skin (by its matte) is darker and a little more saturated, as wet skin is, with a faint gloss
    towards a patch's middle and, at its ``pin``, a point of glint on its side towards the key light (``key``, a unit
    direction in the picture). What overlaps is one wet patch, darker again round its outline only (each darkened in
    turn and ringed at its own rim, a band of patches was a string of beads). ``sheen`` is false under a resolved
    skin film, whose narrow reflections replace the soft gloss. Returns the boxes it changed."""
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
        rr = (_palm_radius(u, v, lob) if not sheen else
              np.sqrt(u * u + v * v) / (1.0 + (1.0 - soft)
                   * (0.14 * np.cos(3 * th + lob) + 0.09 * np.cos(5 * th + 1.7 * lob)
                      + 0.06 * np.cos(8 * th + 2.9 * lob))))
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
    patch = P[Y0:Y1, X0:X1]
    lum = (patch @ fx._LUMA)[..., None]
    around = cv2.GaussianBlur(lum[..., 0], (0, 0), SKIN_CREASE[1])
    crease = np.clip((around - lum[..., 0]) / np.maximum(around, 1e-4), 0.0, 0.3)  # (its lines, darker than round them)
    d = dark * (1.0 + SKIN_WET[1] * 4.0 * seen * (1.0 - seen)) * (1.0 + SKIN_CREASE[0] * crease) * skin
    saturation = SKIN_WET[0] * (1.0 if sheen else 0.3)
    patch[:] = (lum + (patch - lum) * (1.0 + saturation * (sat * skin)[..., None])) * (1.0 - d[..., None])
    if sheen:
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
    hand = hand_of(track, n, plate)
    cam = None if timeline.camera is None else np.asarray(timeline.camera, np.float64)
    scenes = (orb_scenes(timeline, track, hand, (H, W), seed, cam, plate, mask) if form == "orb"
              else whip_scenes(timeline, track, hand, (H, W), seed, cam, mask, plate))
    room = Room(plate)
    noise = Noise(seed)
    wet = [k for k in range(n) if scenes[k].blobs]
    code = fx.shadow_noise(plate) / GRAIN_UNIT / 255.0 if wet else 0.0
    out = np.empty_like(plate)
    done = [0]

    def one(k: int) -> None:
        presence = float(hand.presence[k])
        if presence <= 0.0:
            out[k] = plate[k]
        else:
            painted = frame(k, plate[k], None if mask is None else mask[k], scenes[k], hand, room, noise, code, seed)
            out[k] = painted if presence >= 1.0 else gam(lin(plate[k]) * (1.0 - presence) + lin(painted) * presence)
        done[0] += 1
        if done[0] % 8 == 0:
            say("water", done[0] / n, f"Shaping the water: frame {done[0]} of {n}")

    with ThreadPoolExecutor(fx.THREADS) as ex:
        list(ex.map(one, range(n)))
    say("water", 1.0, "The water is in")
    return out
