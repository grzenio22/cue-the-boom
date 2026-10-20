"""Air, made rather than rendered (cuesync.elements): a spinning ball of air over the open palm (``ball``), or a push
of air out of it (``blast``).

Air is invisible. What shows is what it bends (the plate seen through it, shifted where the air is denser or swirls,
so a straight edge behind it wobbles), the light it gathers and spreads (a shadowgraph: denser air focuses the room's
light into brighter ripples with darker ones beside them, as the shadow of a candle's hot air does on a wall) and what
it carries (sparse specks of dust, and a faint sheen of moving air). All of it is drawn from the plate itself, the
hand's track and its matte, in linear light, with numpy and OpenCV, every random part from ``seed``: the same inputs
give the same bytes. Air throws no light of its own, so nothing here lights or darkens the hand or the room: what it
carries takes the colour of what is behind it, a little towards the room's lit surfaces (fx_light._lit_tone), and the
plate's own grain where it covers the plate. Nothing it paints is milky: dust and the dusty air of a push cover what is
behind them a third at the most, and the sheen of a streak only lifts it a little.

- ``ball``: a sphere of air standing on the palm, held by the voice, nearer the lens than the palm and so drawn bigger,
  never wider than about one and a half spans of the hand. Its signature is the bend: what is behind it is magnified a
  little (a weak lens; less over the hand, so the fingers seen through it keep their shape), twisted round its spin
  axis by lobes that turn with it, and bent 6-12 px at 1080 wide by the swirling air inside it (its shell's streaky
  density, a puff of swirling air through its middle and the streaks it carries, each a filament of denser air), all
  turning with the spin, so the straight edges behind it wobble; the same density's shadowgraph ripples over the plate,
  and a broken shadowgraph band runs round its wobbling outline. Streaks of air run round one spin axis on bands of
  latitude out towards its poles, narrowing and thinning there as a sphere's do, on its outer shells, all one way, each
  band leaning a little its own way and turning slowly, with a few wisps on orbits leaning off it: each shows as a
  faint sheen in the colour of what is behind it (a head or two flaring), brightest on the near side; the far half of
  them goes behind the hand's matte, and over the skin they are barely there. Sparse dust turns inside with the spin,
  brighter with the voice, a little of the room's dust is drawn in round its equator, and a few bigger motes glint as
  they turn: specks of 2-4 px in the scene's colours, lit a little.
  It gathers in from wide as the voice starts, grows and spins faster the longer and louder the voice holds, rides out
  a breath, pops when the voice comes back after a dip (its streaks widen, its bend flashes and it swells on the hit,
  the whole picture lifting little more than before, throwing a push of air whose dust is there at once, and it
  gathers tight again within about eight frames), and when the voice lets go bursts into a push of air, as the
  blast's. The hand's edge is redrawn as the plate's mix of skin and wall, each taking the air's change on its side.
- ``blast``: a push of air out of the palm on each of the voice's hits. A pressure front runs out across the room at
  about 1500 px a second at 1080 wide (what is seen through it is bent outward in a band 40-60 px wide at its front,
  which is a shadowgraph band), a turbulent wake behind it smears what it passes, a thin vortex ring of dusty air rolls
  out of the palm after it, bending what is seen through it and showing only as a faint haze in the colour of what it
  covers (brighter on its side towards the room's light, less so on the other, under a third opaque, far less over the
  hand; each push's ring a little different) trailing a wake, sparse dust and flakes fly out of the palm in
  3D (towards the lens they grow and go soft), the room's own dust near the push is shoved outward as the front passes
  it (one pool for the shot, which the next push throws again), and all of it then slows, swirls and settles over
  fifteen to twenty frames; what goes behind the palm's plane is hidden by the hand, what is in front of it is drawn
  over the hand, in soft faint puffs near the palm's plane and with its body further in front or when the whole ring
  crosses the hand. A ring swelling past the frame shows only its lumps and body, never a haze over the whole frame. A
  true push at the lens sends a doughnut of dusty air at the lens, starting off the palm with its middle above it so
  it rings the hand, its lower arc in front of the hand, drifting to the frame's middle and turning to face the lens
  as it swells and goes soft, its lumps sweeping off the frame's edges together and fading out as it fills the lens.
  What was thrown stays in the room: on a handheld shot it moves with the room, not with the screen. Every push knocks
  the phone 10-15 px at 1080 wide, back within three frames; one at the lens also swells and smears the picture
  outward as the gust reaches it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import fx_light as fx
from .elements import FORMS, Timeline
from .transfer import gam, lin

_BALL, _BLAST, _ROOM, _GRAIN, _CLOUD, _SWIRL = 11, 12, 13, 14, 15, 16  # mixed into each seed, so one seed gives each
# part its own randomness (_SWIRL: a ball's or a push's second stream, for its bands' tilts and its eddies, so what the
# first stream draws is the same)

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
OPEN_LOOK, OPEN = 4, 0.85  # a push is placed on the first of the hit's frame and the next this many on which the hand
# is at least this open (the fingers' reach past their knuckles against the take's most open), else on the most open
# of them; its timing stays the hit's (on a fist opening on the hit, the push came from between the knuckles; at 0.8
# the fingers were still flinging open)

# ---- the ball -------------------------------------------------------------------------------------------------------
BALL_R = 0.62  # hand sizes: the ball's radius once formed (about as wide as the spread hand) ...
BALL_SPAN = 0.85  # ... and at least this many knuckle spans (index to little finger: the track's hand size runs
# small on a palm turned to the lens, and there the ball sat sunk in the palm at half the hand's width)
BALL_GROW, BALL_GROW_S = 0.3, 3.0  # ... and it grows this much more over this many seconds of held voice
BALL_BREATH = 0.06  # ... and swells this much more at the voice's loudest
BALL_MAX = 0.62  # ... but its radius, swelled, popped or bursting, is never more than this many of the hand's span
# (thumb tip to little fingertip), so it and what it carries stay within about one and a half spans across (grown and
# popped it was wider than the frame by the take's last hit)
SPAN_SIZES = 1.9  # hand sizes: the span of a hand whose track has no landmarks
BALL_GAP = 0.15  # hand sizes between the palm and the ball's underside (along the palm's normal)
BALL_LIFT = 0.35  # hand sizes: a palm turned to the lens (its normal's z past 0.7) lifts the ball up the picture by up
# to this, over the fingertips and onto the room (on the palm it was white over pink, and vanished at phone size)
BALL_FORM, BALL_RELEASE = 12, 6  # frames it gathers over (half a second), and bursts and thins out over while the
# push of air it lets go carries the burst on (on its own, a 9-frame fade, letting go had no payoff)
RELEASE_GAIN = (0.5, 0.4)  # that push is this hard, and this much harder at the voice's loudest (of a loudest hit's)
BALL_LINGER = 6  # frames it stays after the voice last feeds it, before it bursts
BALL_BRIDGE = 30  # frames: a silence no longer than this (a breath between two "ffff"s) keeps the ball, thinned out
# and slowing (BALL_IDLE); a longer one lets it go, and the next sound gathers a new one
BALL_IDLE = 0.6  # between two sounds the ball thins to this much of itself
BALL_MIN = 22  # frames it lives at least: a short "pf" still shows a ball forming, spinning and bursting
BALL_SPIN = 0.9  # revolutions a second once formed on a quiet voice ...
BALL_SPIN_VOICE, BALL_SPIN_HELD = 1.0, 0.6  # ... this much faster at the voice's loudest, and this much after
# BALL_GROW_S seconds held (about 2.5 a second at the most: faster smeared every streak into a ring)
BALL_LENS = 0.08  # the plate behind its middle is magnified this much (dense spinning air, a weak lens; at 0.16 it
# moved what was behind by 20 px in one smooth piece, which reads as the camera, not as air)
BALL_RIM = 0.02  # radii: at its rim what is behind is pulled in from this much further out (the rim draws the ball's
# outline across the fingers seen through it; 0.05 drew it as hard as a glass globe's)
BALL_TWIST = 0.09  # rad: how far the shell twists what is behind it round the axis, at its densest ...
BALL_DRAFT = 0.4  # ... and radii: the air it drags round it twists what is just outside it too, out over this much
# (the straight edges near a ball, a switch plate's, a door frame's, are what show its bend: a plain wall shows none)
BALL_EASE = (0.08, 0.14, 0.5)  # radii: the twist and the squeeze are eased over the first, and over the second round
# its rim (the third either side of it), so what is behind shifts by under 0.4 px per px anywhere (the shell's chord
# rises steeply at its rim and round its core: unsmoothed, a straight edge behind the rim kinked by 9 px over 2, and
# eased over 0.05 a switch plate's edge still had an elbow)
BALL_TURB = 3.5  # px at 1080: the co-rotating shimmer
BALL_REFRACT = (11.0, 28.0)  # px at 1080: the swirling air inside it bends what is behind it this far where its
# density changes fastest (the gradient of its density, as a denser filament of air bends light towards itself), that
# density smoothed over this many px first, so the bend's slope stays under about half a px per px (folded nowhere;
# smoothed over 13 px, the streaks put elbows in straight lines) while a straight edge behind the ball wobbles by 6-12
# px (with milky streaks and a smooth lens, the switch plate behind the ball never moved) ...
BALL_DENS = (1.0, 0.45, 0.6, 1.0)  # ... that density the shell's near side's this much, its far side's this, its
# middle's puff this and the streaks it carries this (each streak a filament of denser air, so the bend carries the
# streaks' shape) ...
BALL_RIPPLE = 0.07  # ... and its shadowgraph: the plate under the ball this much brighter and darker in ripples where
# that density curves (what reads of air over a plain wall, which has nothing to bend)
BALL_HAND = (0.65, 0.8)  # over the hand the bend is cut by this share and the shimmer by this (bent in full, a palm
# under the ball melted into a disc and the fingers' outlines broke)
BALL_CA = 0.03  # colour fringe of the bend, none over the hand and just past its edge (0.22 drew red and cyan
# outlines round the fingers; over the matte, a thin cyan line ran along the thumb)
BALL_WOB = (0.09, 0.06)  # radii: its outline wobbles this much in three lobes and in five, turning with the spin and
# slowly changing (round to 3.5 % and ringed by a crisp band, it was a crystal ball)
BALL_SHADOW = 0.05  # the shadowgraph: the rim focuses the room's light into a band this much brighter, with a darker
# band inside it (at 0.12 its band drew a hard round outline; at 0.07, with the streaks no longer painted, a grey ring)
BALL_BAND = 0.05  # radii: the band's softness (at 0.035 it was a drawn line)
MIST_CORE = 0.55  # the shell's density: this share of it through the ball's depth, densest through its middle as a
# solid ball of air is (all of it densest at the limb, the ball bent as a hollow jar) ...
BAND_MIST = (0.8, 0.1)  # ... plus this much (of the limb's) along each band of latitude the streaks run on, this many
# rad wide, on the near side only: the bands bend as sheets of moving air with a body, not as separate wires
BALL_BACK = 0.45  # the far side shows this much (seen through the near side), and not through the fingers
BALL_CAUSTIC = 0.03  # the palm under the ball takes its light focused into moving patches, this much brighter / darker
BALL_STREAKS = 90  # streaks orbiting the ball (long arms drawing the room's air in from 3.4 radii swept past the
# frame's edges)
STREAK_WISPS = 0.22  # share of the streaks that are wisps on orbits leaning off the spin axis; the rest (about 70) run
# round it on a few shared bands of latitude, all one way, overlapping into sheets (all leaning, the ball was a ball of
# yarn with no axis; none, a jar of rings; 39 on the bands read as separate white wires at phone size) ...
BAND_ALPHA = 0.75  # ... each of those drawn this much as bright (so the sheet, not each streak, is what reads) ...
STREAK_FINE = 0.25  # ... and this share of them fine streaks, the rest broad wisps (half fine, the bands read as wire)
STREAK_BANDS = (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2)  # rad: those bands' latitudes (each ball shifts them a
# little), out to about 70 degrees, so they narrow towards its poles as a sphere's do (out to 40, the top and bottom
# bands were about as wide as the middle one: a jar of stacked rings) ...
BAND_POLE = 0.5  # ... each band's streaks drawn cos(latitude) to this power as bright, the poles thinning out ...
BAND_DEPTH = (0.8, 2.0)  # ... on shells from the ball's rim in to this share of its radius, most near the rim (this
# power; in to 0.62, the inner shells' streaks were more stripes stacked inside it) ...
BAND_TILT = (0.2, 12.0)  # ... each band leaning off the spin axis by up to this many rad about a level axis of its
# own, which turns round the spin axis this many degrees a second (all square to the axis, the bands stayed stacked and
# parallel frame after frame: a coil; leaning up to 8 degrees, they still read as stacked)
CORE_PUFF = (0.6, 2.5)  # inside this many radii the ball's middle is a puff of swirling air (in place of the inner
# shells' streaks), its swirl wound this many rad from its edge to its middle
STREAK_TILT = 0.5  # rad: the wisps' orbits lean off the spin axis by up to this
BALL_AXIS_Z = 0.7  # the spin axis leans at least this far off the lens's axis (the largest z it keeps), so its bands
# read as ellipses round an axis (a palm turned to the lens stood the axis at the lens: rings round a whirlpool)
BALL_NEAR = 0.35  # radii: the streaks brighten from the far side's BALL_BACK to full over this much either side of
# the limb (cut at the limb, every streak broke there)
STREAK_ALPHA = 0.75  # opacity of a streak of brightness 1, in the colour of what is behind it lifted a little
# (STREAK_TONE): a sheen on the bend that carries the streaks' shape (at 2.0 in the room's lit tone, milky white
# ribbons were all the ball was) ...
STREAK_TONE = (0.4, 1.45)  # ... that colour moved this share of the way towards the room's lit tone, and this much
# brighter
STREAK_PEAK = 1.3  # ... and no streak is brighter than this many times the mean of the brighter half (one crisp
# bright arc read as wire) ...
FLARE = (0.7, 40)  # ... but a few heads flare as the accent: this much brighter, in a broad wisp, on a pulse this
# sharp (about two at a time)
STREAK_SKIN = (0.06, 0.3)  # over the hand the streaks are drawn only as broad wisps, cover it this much at the most
# and show this much of themselves, their far half not at all (a fine bright arc across the palm read as a white glare
# smear on the skin; at 0.25, every streak sat in front of the hand)
HAND_SOFT = 6.0  # px at 1080: the matte is softened this much wherever the ball eases off over the hand (cut at its
# edge, a fingertip got a jagged bite)
STREAK_PX = (1.3, 9.0)  # px at 1080: a fine streak's softness, and a broad wisp's, at its head (a wisp under 3 px
# was under a pixel at phone size, and the ball a ghost; at 6.5 the bands were thin lines) ...
STREAK_TAPER = (0.4, 0.55)  # ... and past this share of its length it is drawn this much as wide (as wide all along,
# 46 streaks read as wire wound round a globe)
STREAK_STEP = 0.7  # px between the points a streak is drawn from (further apart a soft streak turned to beads)
STREAK_END = (3.0, 0.3)  # where a streak turns towards the lens it is brighter by its foreshortening (up to the first)
# to this power
BALL_DUST, ROOM_DUST, MOTES = 45, 24, 8  # dust inside; the room's dust drawn in; bigger motes that glint (drawn
# as the blast's dust is, _specks: each a few px, smeared along its half frame of motion; 250 read as a snow globe)
DUST_SPIN = (0.75, 0.35)  # the dust inside turns this many times as fast as the ball's spin, and this much more at
# its middle
DUST_LOUD = (0.55, 0.9)  # ... and shows this much on a quiet voice, and this much more at its loudest
BALL_PAD = 1.4  # radii: the ball's air (its bend, the dust it draws in) reaches this far from its middle
PULSE_DIP, PULSE_GAP = 0.35, 12  # a hit the voice comes back with after dipping under this (of its loudest) in the six
# frames before, at least this many frames into the ball's life and after its last pulse, ...
PULSE_BURST, PULSE_TAU = 0.5, 3.0  # ... bursts the ball this far, swelling it this much wider by the frame after the
# hit, and it gathers again over about this many frames, tight again about eight frames after it (thinned out as it
# swelled, the ball blinked out on the frame after the hit, then hung on as a huge faint swirl) ...
PULSE_FLASH = (0.5, 0.2, 0.3, 0.5, 1.0)  # ... its shell flashing on the hit and the frame after it: the broad wisps
# this much brighter, the fine streaks this much, all of them this much wider, and its bend this much stronger, then
# easing off over this many frames (all of it brighter, the whole picture lifted by 4 grey levels, an exposure flash,
# and the fine streaks were crisp wire) ...
PULSE_RISE = 1.5  # ... and it lifts the whole picture by at most this many grey levels more on the hit and the two
# frames after it than on the frame before (the push it throws lifts it a little more): the punch stays round the ball
PULSE_GAIN = (0.3, 0.3)  # ... throwing a push of air this hard, and this much harder at the voice's loudest, its
# dusty air there on the hit (held through a take, the ball never let go and its shot had no payoff)

# ---- the blast ------------------------------------------------------------------------------------------------------
BLAST_FRAMES = 18  # the front's run: three quarters of a second
FRONT_FROM = 0.06  # share of the short side: the front's radius as it leaves the palm ...
FRONT_SPEED = (62.5, 0.25)  # ... running out at this many px a frame at 1080 (1500 a second), this share faster on
# screen for a push at the lens, slowing a fifth by the end of its run (out to the frame's edge within eight frames, a
# blink: it was gone before the eye found it)
BUBBLE_PX = 18.0  # px at 1080: how far the pressure front pushes what is seen through it outward, at its strongest
# (30, held under a slope of 0.2, spread the front over a band 200 px wide: a bulge, not a front) ...
BUBBLE_END = 0.3  # ... and the share of that it still has at the end of its run
BUBBLE_INNER = 0.2  # ... most of it a thin band at the front, and this share of it a push growing out to the front
# from inside (all of it growing to the front, the whole room bulged like a funhouse mirror)
FRONT_W = (40.0, 60.0)  # px at 1080: the front's band as it leaves the palm, and at the end of its run ...
FRONT_SLOPE = 0.45  # ... and at least as wide as keeps its bend under this many px per px of picture
SHADOWGRAPH = (0.14, 0.1)  # the front gathers and spreads the light behind it: a bright band, dark either side, this
# deep (at 0.06 the front was invisible on a plain wall), the band trailing it only this share as dark (as dark, it
# sank the inner edge of the ring of dusty air following the front under the wall)
WAKE_PX = 3.5  # px at 1080: the turbulent wake's radial smear behind the front ...
WAKE_ROUND = 8.0  # ... its eddies this many half-degrees across round the front (at 0.8 they were a few px across
# where the front crossed the switch plate, and waved its straight edge like a glitch; at 2.5 they still bent what was
# behind them by over a px per px, folding it)
BLAST_CA = 0.02  # (0.12 split an orange fringe off the fingertips, 0.05 a cyan one off the switch plate)
BUBBLE_CLEAR = (0.6, 0.9)  # hand sizes: the bubble bends nothing until its front is this far out, and is whole this
# much further on (bent while it was still over the hand, the fingers wobbled on the hit like a glitch)
CLOUD_N = 4000  # parcels of dusty air the push sets rolling: a vortex ring, what a palm pushed through air makes (a
# cloud of clumps thrown off the palm slowed and faded where it was: a glow stuck on the hand, not air going somewhere)
RING_R = (0.5, 1.8)  # hand sizes: the ring's radius as it leaves the palm, and how many times wider it ends
RING_CORE = 0.075  # ring radii (as it leaves): the radius of its core, which the dusty air rolls round (it fattens as
# it slows; at 0.16 the ring was a thick scalloped torus, a smoke ring stamped on the room; at a quarter, a ring seen at
# a slant closed its hole and read as a cauliflower)
RING_BEND = (10.0, 22.0)  # px at 1080: the ring's dusty air bends what is seen through it this far where it thickens
# fastest, its thickness smoothed over this many px first (its body is a refractive band with a faint haze, not paint;
# smoothed over its lumps alone, a few px, the bend would be steeper than a px per px and fold what is behind it)
RING_SPEED = 0.06  # share of the short side a frame: how fast the ring leaves the palm on the loudest hit ...
RING_SLOW = 9.0  # ... slowing over this many frames (a ring holds together and carries on past the dust's 5)
RING_ROLL = 0.3  # rad a frame: how fast its air rolls round its core at first (slowing with the ring)
RING_WAKE = 0.2  # share of the dusty air trailing behind the ring, back to the palm (none on a push at the lens: there
# the wake fills the ring's hole, and its parcels smeared down over the floor)
RING_BILLOW = 0.6  # ring radii (as it leaves): the room's eddies push the ring's air this far about as it slows
# (lumps that turn; a ring kept round read as a drawn halo)
RING_LUMPS = 0.8  # how lumpy its air is along the ring and round its core (the parcels' share of it, from a smooth
# noise fixed in the rolling air: spread evenly, the ring was a smooth tube)
RING_NEAR = 0.4  # a ring seen at a slant is this much thicker on its near side, and thinner on its far side (even,
# a tilted ring's near half read no nearer than its far one)
RING_VARY = (0.17, 0.15, 0.2, 0.1, 0.1)  # each push's ring is tilted up to this many rad off the push, this much
# bigger or smaller, rolls this much faster or slower, leaves this much faster or slower and starts up to this many
# hand sizes off the palm's middle (three pushes off one palm drew one ring three times: a replay)
LENS_Q = (0.3, 0.7)  # a push whose share across the picture (push_direction) is under the first comes at the lens,
# over the second crosses it, and between them it is a blend (at 0.45, a push mostly at the lens drew grey puffs
# between the fingers) ...
LENS_TRUE = (0.4, 0.65)  # ... and a push out of the palm is a true push at the lens as that blend grows over this
# much: the second of each pair below is a true push's, the first a push partly at the lens or a ball's burst (it
# starts where the ball is, as wide as it), each per unit of the blend (one tuning for both, retuned for the true
# push, halved a ball's release ring and broke it up) ...
LENS_RUSH = (2.0, 1.4)  # ... a push at the lens sends its ring this much faster, straight at the lens, so a true one
# swells past the frame's edges in about ten frames, as a smoke ring blown at a camera does (at the sideways speed it
# hung over the hand; starting off the palm at 2.0, it was gone in eight, the last of it a flash of puffs) ...
LENS_WIDE = 0.8  # ... this much wider as it leaves, so it rings the hand (dust in front of a lit palm hardly shows:
# a ring on the palm read as puffs between the fingers and a smudge by the wrist) ...
LENS_THICK = (4.5, 2.0)  # ... and draws it this much thicker: seen face on, a ring's air is spread round a wide
# annulus rather than piled up along a slanted band, and at the sideways thickness it was a faint veil (a true push at
# 3.5 round a fat core sat at full opacity, flat translucent petals; a ball's release ring at 3.5, its shadow side and
# its inner edge now kept off the wall's tone, was a little fainter than before) ...
LENS_CORE = (-0.4, 0.5)  # ... round a core this much fatter: a true push's a doughnut with a body (blurred with its
# spread, a fat ring closed its hole and read as a blob; kept sharp (CLOUD_SOFT_MAX), a thin one read as a rope), a
# push partly at the lens thinner (it goes soft as it nears the lens, and fat it closed its hole) ...
LENS_LIFT = (0.5, 0.6)  # ... a true push's starts this many hand sizes off the palm along its normal, its middle this
# many up the picture from the palm, so its lower arc crosses the hand (centred on a palm low in the frame, only its
# top arch was on screen, and read as smoke rising behind the hand) ...
LENS_MID = 0.33  # ... and drifts to the frame's middle by the time it is this near the lens (palm depths), turning to
# face the lens as it does, so the whole ring stays on screen until it swells past every edge at once (straight at the
# lens, its lower half left the frame within four frames; aimed at a point near the lens, it overshot and flew off the
# top; tilted as the palm sent it, its lower arc, nearer the lens, swelled off the bottom four frames before its top
# left the top, and it read as leaving through the top)
LENS_GONE = (0.7, 3.2)  # the ring fades out as its radius on screen grows from this share of the frame's half
# diagonal to this many times that, over about six frames (by its depth alone, the last of it was a haze over the
# whole frame for three)
CLOUD_ALPHA = 0.3  # opacity the dusty air tends to where it is thickest (at 0.48 in the room's lit tone, an opaque
# cream ring) ...
CLOUD_HALF = (5.0, 1.6)  # ... reaching half of it where it is the first this thick, on a curve this steep (a cloud
# spread evenly over the frame's short side squared is 1, a ring's air 2-40: saturating at 5, every lump sat at the
# most and the ring was one flat tone; on a gentle curve its lumps lost their edges) ...
CLOUD_ON_HAND = (0.08, 0.12)  # ... and over the hand it covers what is behind it this much at the most near the
# palm's plane, and this much where it is well in front of it (CLOUD_FORE; at 0.6 a ring coming at the lens laid a
# milky veil over the hand) ...
CLOUD_FORE = (0.95, 0.8)  # palm depths: dusty air nearer the lens than the first is in front of the hand, and well in
# front of it by the second
CLOUD_LUMPS = (240.0, 120.0, 3.0)  # a ring swelling past the frame shows only its lumps and body, what is left of it
# less its blur this many px at 1080 wide or this many of its lumps (the third), and over the hand, near the palm's
# plane, any ring only its lumps, less its blur this many px wide (about three core radii of a ring leaving the palm;
# well in front of the hand (CLOUD_FORE), its body too, so a ring coming at the lens crosses the fingers): its broad
# mean is dropped (a ring swelling past the frame laid a fog over all of it that lifted the whole picture by 8 grey
# levels, the exposure breathing; over the palm, the film between the lumps read as milk on the skin; at 120 px in the
# room too, a ring coming at the lens lost its body and read as a string of popcorn, and as it neared the lens, at
# 240, as a scatter of petals) ...
LUMPS_WIDE = (0.35, 0.35)  # ... its broad mean dropped as its radius on screen grows from the first share of the
# frame's half diagonal to that plus the second (dropped from every ring, a ball's release ring lost a third of its
# band and broke into a string of lumps) ...
CLOUD_KEEP = 0.25  # ... but inside its band never thinner than this much of its mean over two lumps (the band's middle
# was punched through: dirty grey holes; at 0.35 the band was one flat cream torus)
HAND_PUFF = (0.3, 0.8, 5.2)  # over the hand, a ring that does not cross it whole (one leaving the palm sideways) shows
# only in puffs: where the ring's own lump noise, blurred this many cells more (about twice its lumps, so each puff
# carries on from a lump in the room), read as 0.5 + 0.3 of it, is over the first, fully by the second; its wake not at
# all (thresholding the air's own lumps kept a ribbon along the band's crest: a hard-edged cream worm on the palm; its
# film between them read as a pale band across the base of the fingers; the wake's stem lingered mid-palm) ...
HAND_NEAR = (0.05, 0.08, 8.0)  # ... covering the skin this much at the most near the palm's plane, this much well in
# front of it (CLOUD_FORE), and its puffs blurred this many px at 1080 more (at 0.35, a cream shape on the palm with a
# hard edge: toothpaste) ...
BURST_HAND = 12.0  # ... and over the hand a ball's burst ring's alpha is blurred this many px at 1080 more (its small
# lumps, laid with the skin scattered under them, were pale polygonal blotches: a rash, or soap foam)
RING_SWIRL = (0.25, 2.0)  # a ring's air is this much thicker and thinner in eddies this many cells of its lump noise
# across, rolling with it (with lumps alone, a fat ring was one flat cream torus inside its outline)
LENS_PASS = (0.05, 0.2)  # palm depths: dusty air nearer the lens than the second fades out by the first (passing
# the lens, a ring drawn as thick as it came laid a veil over the whole picture; the ring as a whole goes as it
# swells past the frame, LENS_GONE)
CLOUD_Q = 4  # the dusty air is drawn this many times coarser than the frame, then blurred and drawn at full size
CLOUD_SOFT = (6.0, 1.2)  # px at 1080 at the palm's depth: how soft the dusty air is as it leaves the palm, and how much
# softer a frame (it diffuses as it slows; at 16 + 4 a frame every lump was blurred away into a smooth veil) ...
CLOUD_DEFOCUS = 6.0  # ... and px at 1080 more per unit of (1 / depth - 1) nearer the lens than the palm: the lens is
# focused on the hand, so a ring coming at it goes soft as it grows ...
CLOUD_SOFT_MAX = 25.0  # ... but never softer on screen than this many px at 1080 (blurred with its spread, a ring
# filling the lens lost every lump) ...
BURST_SOFT = 10.0  # ... and a ball's burst ring starts at least this soft: its air was the ball's, spread through it
# (as sharp as a ring off the palm, its parcels spread round a ring as wide as the ball were a chain of beads: popcorn)
CLOUD_FADE = (8, 24)  # frames: the dusty air thins out from this many frames after the push, over this many more
CLOUD_IN = 3  # ... and it comes in over the push's first this many frames (the bend and the shock ring own them;
# there at once, it sat at the base of the fingers as a grey halo: dirt on the hand), but at once out of a ball's
# burst (there it is the burst)
CLOUD_OVER = (2, 3)  # what is in front of the palm's plane shows over the hand from this many frames after the push,
# fully this many later (behind it, the hand hides it; thinned over the whole matte, the ring rose from behind the
# fingers instead of off the palm), at once where the ring starts off the palm (a ball's, one at the lens)
CLOUD_SHADE = (0.5, 0.3)  # its colour: what it covers, lit a little by the room (CLOUD_LIFT) on the side its lumps
# face towards the room's light, the light looked for this many lump sizes towards it (its core as blurred: by the core
# alone, a blurred lump's slope was too weak to read at phone size), the shadowed side only this share as dark as the
# lit side is bright (as dark, a thin band's shadow half sank to the wall's tone and the ring broke up) ...
CLOUD_LIFT = (1.2, 0.5)  # ... as bright as what it covers times the first, plus or minus up to the second towards or
# away from the light, never darker than what it covers: never cream, never a stain, in any room (in the room's lit
# tone going towards white, a ring was a cream torus over a beige wall) ...
CLOUD_TONE = 0.25  # ... its hue moved this share towards the room's lit tone's
SCATTER = (3.0, 0.3, 0.15, 3.0)  # ... and it blurs what is behind it towards this many px at 1080, this share of the way
# off the hand where it is thickest and this share over the hand, its share taken from its opacity blurred this many
# times as far (dust scatters what is seen through it: a white switch plate's crisp edge cut every lump that crossed
# it; over the hand in full, a ring in front of the palm blurred it under a veil)
DUST_N, FLAKES = 450, 24  # dust and grit flung out of the palm, sparse; bigger flakes that tumble (900 in the room's
# lit tone were a spray of white dots; 1400 read as sensor snow)
DUST_R = (2.0, 4.0)  # px at 1080 at the palm's depth: a speck's size (1.1-2.4: too fine to read at phone size)
DUST_ALPHA = 0.35  # a speck covers what is behind it this much at the most (at 0.85 in the room's lit tone, every speck
# was a white dot) ...
DUST_TONE = (0.35, 1.45, 0.55)  # ... in the colour of what is behind it moved this share towards the room's lit tone and
# this much brighter (lit a little: in a dark room it still shows), grit that much darker
DUST_STIR = 0.012  # share of the short side a frame: how hard the eddies push the dust about once it has slowed ...
DUST_CALM = 14.0  # ... calming over this many frames (stirred as hard for its whole life, the hanging dust zipped
# about in streaks: film scratches)
ROOM_N = 220  # the room's own dust, one pool for the shot: still and unseen until a front shoves it ...
ROOM_NEAR = 0.6  # short sides: ... most of it near the first push and ahead of it, thinning out this far (1500 specks
# over the whole frame read as sensor snow)
ROOM_PUSH = 0.06  # share of the short side a frame: how fast the front throws the room's dust it passes
ROOM_SEEN = 0.45  # thrown, it shows at least this much as it slows, fading out with its life (seen only while fast, it
# flashed past and was gone before the eye found it; held, it hung through the next push)
DUST_LIFE = (8, 12)  # frames: thrown dust hangs this long after the push, then settles out over the next this many
# (at 22 + 30 it hung for two seconds, too faint and fine to see once it had slowed)
SMEAR_MIN = 2.0  # px at 1080: a speck moving less than this over the shutter's half frame is drawn round (smeared,
# the slow ones were ticks: hairs on the skin)
FOCAL = 1.0  # focal length in short sides (a phone's main camera: about 53 degrees across the short side)
BUFFET = 0.035  # a push straight at the lens swells the picture this much when the gust reaches the phone (0.018 was no
# kick at phone size; it never dims it: on one frame the dip was an exposure flicker)
KNOCK_PX = 13.0  # px at 1080: every push knocks the phone this far along it at its loudest (3 px, and only for a push
# at the lens, was no knock at all) ...
KNOCK_SHAPE = (1.0, -0.35, 0.1)  # ... and it is this far off on the knock's frame and the next two, then back
KNOCK_AT = (1, 3)  # frames after the push: a push across the picture knocks the phone on the first (the hand's
# shove), one straight at the lens on the second, as its gust arrives (waiting for the front to pass the frame's
# corners, the knock came 0.4 s after the voice, with nothing arriving at the lens)
FIELD_Q = 2  # the bend is worked out on a grid this many times coarser and drawn at full size (it is smooth)
EDGE = (9.0, 3.0, 0.4)  # px at 1080: the band round the hand's matte, outside it and inside it, whose pixels are
# redrawn as the plate's mix of skin and what is behind it, keeping this share of what the mix does not explain (the
# plate's edge pixels carry the wall the fingers stood against, and the air eases off over the hand: lifted less than
# the wall round them, the fingertips got a dark outline; with all of the lens's orange fringe kept, a thin orange one)


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


def _broad_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """``x`` (h, w) blurred to ``sigma`` px, worked out at a quarter of its size when that is wide (it is smooth)."""
    h, w = x.shape
    if sigma < 8 or min(h, w) < 16:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    small = cv2.resize(x, (-(-w // 4), -(-h // 4)), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / 4), (w, h), interpolation=cv2.INTER_LINEAR)


def _tile_noise(rng: np.random.Generator, h: int, w: int, sx: float, sy: float, ch: int = 1) -> np.ndarray:
    """Smooth unit noise (h, w, ch) that wraps round both ways, so it can be sampled round and round."""
    n = rng.standard_normal((h, w, ch)).astype(np.float32)
    py, px = int(3 * sy) + 1, int(3 * sx) + 1
    n = cv2.GaussianBlur(np.pad(n, ((py, py), (px, px), (0, 0)), mode="wrap"), (0, 0), sigmaX=sx, sigmaY=sy)
    n = n.reshape(h + 2 * py, w + 2 * px, ch)[py:-py, px:-px]
    return np.ascontiguousarray(n / max(float(n.std()), 1e-6)).squeeze()


def _wrap_blur(tile: np.ndarray, sigma: float) -> np.ndarray:
    """A 2D noise tile that wraps round both ways (_tile_noise) blurred ``sigma`` cells more, still wrapping, back to
    unit spread."""
    p = int(3 * sigma) + 1
    b = cv2.GaussianBlur(np.pad(tile, p, mode="wrap"), (0, 0), sigma)[p:-p, p:-p]
    return np.ascontiguousarray(b / max(float(b.std()), 1e-6))


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


def _remap_rgb(P: np.ndarray, dx: np.ndarray, dy: np.ndarray, box: tuple[int, int, int, int],
               ca: float | np.ndarray) -> None:
    """In place: ``P`` inside ``box`` (y0, y1, x0, x1) resampled from (x + dx, y + dy), red bent (1 + ca) as far and
    blue (1 - ca): the slight colour fringe of a bend (``ca`` one share, or one per pixel of the box). Past the frame's
    edge it reads the edge."""
    y0, y1, x0, x1 = box
    gx = np.arange(x0, x1, dtype=np.float32)[None, :]
    gy = np.arange(y0, y1, dtype=np.float32)[:, None]
    src = P.copy()
    ca = np.float32(ca) if np.ndim(ca) == 0 else ca.astype(np.float32)
    for c, sgn in enumerate((1, 0, -1)):
        kk = 1 + sgn * ca if sgn else np.float32(1)
        P[y0:y1, x0:x1, c] = cv2.remap(src[..., c], gx + kk * dx, gy + kk * dy, cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)


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


SCENE_Q = 8  # what is behind the air is read this many times coarser than the frame (the colour it lends is smooth)


def _behind(P: np.ndarray) -> np.ndarray:
    """``P`` (linear (h, w, 3)) SCENE_Q times smaller and blurred a cell: the colour of what is behind each part of
    it."""
    h, w = P.shape[:2]
    small = cv2.resize(P, (max(1, -(-w // SCENE_Q)), max(1, -(-h // SCENE_Q))), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (0, 0), 1.0)


def _up(small: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """The part of a map SCENE_Q times smaller than the frame (_behind) under ``box`` (y0, y1, x0, x1), at full size."""
    y0, y1, x0, x1 = box
    q = SCENE_Q
    cy0, cx0 = y0 // q, x0 // q
    cy1, cx1 = min(small.shape[0], -(-y1 // q)), min(small.shape[1], -(-x1 // q))
    big = cv2.resize(small[cy0:cy1, cx0:cx1], ((cx1 - cx0) * q, (cy1 - cy0) * q), interpolation=cv2.INTER_LINEAR)
    oy, ox = y0 - cy0 * q, x0 - cx0 * q
    big = big[oy: oy + y1 - y0, ox: ox + x1 - x0]
    if big.shape[:2] != (y1 - y0, x1 - x0):  # (past the map's last cell: its edge held)
        big = np.pad(big, ((0, y1 - y0 - big.shape[0]), (0, x1 - x0 - big.shape[1]), (0, 0)), mode="edge")
    return big


def _lit_by(behind: np.ndarray, tone: np.ndarray, toward: float, lift: float) -> np.ndarray:
    """The colour of what the air carries over ``behind`` (linear): that colour moved ``toward`` of the way up to the
    room's lit ``tone`` (never down to it) and ``lift`` times as bright, under 0.95: the scene's own colour, lit a
    little."""
    return np.minimum((behind + np.float32(toward) * np.maximum(tone.astype(np.float32) - behind, 0))
                      * np.float32(lift), 0.95)


_STEP_LUT = (2.2 * np.maximum(np.linspace(0, 1, 1024) ** 2, 1e-4) ** (-1 / 2.2)).astype(np.float32)  # one 8-bit
# gamma step at linear luminance y, as a share of y, read at sqrt(y) (a power per pixel was a third of a frame's time)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(y0, y1, x0, x1) of the True pixels of ``mask`` (h, w), or None."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return None if rows.size == 0 else (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)


def _over(P: np.ndarray, alpha: np.ndarray, colour: np.ndarray, noise: np.ndarray | None, code: float) -> None:
    """In place: ``colour`` ((3,) or (h, w, 3)) laid over ``P`` (h, w, 3) at ``alpha`` (h, w), with the plate's grain
    in the share of it it covers (``noise``: unit grain the size of ``P``; ``code``: one unit of it in 8-bit gamma
    steps): the made layer is no cleaner than the footage under it. Only where it shows is touched: its box, or just
    its pixels when it is sparse (dust)."""
    shows = alpha > 1e-3
    box = _bbox(shows)
    if box is None:
        return
    y0, y1, x0, x1 = box
    shows = shows[y0:y1, x0:x1]
    if shows.mean() < 0.25:  # sparse: its pixels alone
        ys, xs = np.nonzero(shows)
        ys, xs = ys + y0, xs + x0
        Q = P[ys, xs]
        _mix(Q, alpha[ys, xs], colour[ys, xs] if colour.ndim == 3 else colour,
             None if noise is None else noise[ys, xs], code)
        P[ys, xs] = Q
        return
    _mix(P[y0:y1, x0:x1], alpha[y0:y1, x0:x1], colour[y0:y1, x0:x1] if colour.ndim == 3 else colour,
         None if noise is None else noise[y0:y1, x0:x1], code)


def _reedge(P: np.ndarray, P0: np.ndarray, hand: np.ndarray, k: float) -> None:
    """In place: the hand's edge in ``P`` (linear (h, w, 3), the air laid on ``P0``, the same pixels before it) redrawn
    over a band round the matte ``hand`` (0..1; EDGE), eased in over its edges: each pixel there is unmixed into skin
    and what is behind it (the plate's colours just inside and just outside the band) and takes, on each share, the
    change the air made just on that side of the band (where one side has none near, its own change). The colours
    either side are worked out four times coarser (they are smooth) and only the band's pixels are touched."""
    m = (hand > 0.5).astype(np.uint8)
    ro, ri = max(1, round(EDGE[0] * k)), max(1, round(EDGE[1] * k))
    grown = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ro + 1, 2 * ro + 1)))
    skin = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ri + 1, 2 * ri + 1)))
    box = _bbox(grown > skin)
    if box is None:
        return
    h, w = m.shape
    pad = 3 * ro
    y0, y1, x0, x1 = max(0, box[0] - pad), min(h, box[1] + pad), max(0, box[2] - pad), min(w, box[3] + pad)
    mb = m[y0:y1, x0:x1]
    reach = np.where(mb > 0, (ri - cv2.distanceTransform(mb, cv2.DIST_L2, 3)) / max(0.5 * ri, 1.0),
                     (ro - cv2.distanceTransform(1 - mb, cv2.DIST_L2, 3)) / (0.5 * ro))  # (cut at the band's edge,
    # the room's fine streaks ended in a jagged seam)
    ys, xs = np.nonzero(reach > 0)
    Pb, P0b = P[y0:y1, x0:x1], P0[y0:y1, x0:x1]
    p, p0 = Pb[ys, xs], P0b[ys, xs]
    if ys.size == 0 or not np.abs(p - p0).max() > 1e-4:
        return
    q = 4
    sh, sw = max(1, -(-(y1 - y0) // q)), max(1, -(-(x1 - x0) // q))
    sig = 0.75 * ro / q

    def small(x):
        return cv2.resize(x.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)

    P0s = small(P0b)
    Ds = small(Pb) - P0s
    fields = []
    for wt in (small(1 - grown[y0:y1, x0:x1]), small(skin[y0:y1, x0:x1])):  # spread from the room, from the skin
        n = cv2.GaussianBlur(wt, (0, 0), sig)
        div = (1 / np.maximum(n, 1e-4))[..., None]
        fields += [cv2.GaussianBlur(Ds * wt[..., None], (0, 0), sig) * div,
                   cv2.GaussianBlur(P0s * wt[..., None], (0, 0), sig) * div, _smooth((n - 0.01) / 0.05)[..., None]]
    F = np.concatenate(fields, -1)
    fy = np.clip((ys + 0.5) / q - 0.5, 0, sh - 1)
    fx = np.clip((xs + 0.5) / q - 0.5, 0, sw - 1)
    iy, ix = fy.astype(np.int64), fx.astype(np.int64)
    ty, tx = (fy - iy)[:, None], (fx - ix)[:, None]
    iy1, ix1 = np.minimum(iy + 1, sh - 1), np.minimum(ix + 1, sw - 1)
    f = (F[iy, ix] * (1 - ty) + F[iy1, ix] * ty) * (1 - tx) + (F[iy, ix1] * (1 - ty) + F[iy1, ix1] * ty) * tx
    Dr, Wr, hr, Dk, Sk, hk = f[:, 0:3], f[:, 3:6], f[:, 6:7], f[:, 7:10], f[:, 10:13], f[:, 13:14]
    d = p - p0
    Dr, Dk = d + (Dr - d) * hr, d + (Dk - d) * hk
    sk = Sk - Wr
    s2 = (sk * sk).sum(-1)
    mixed = ((hr[:, 0] * hk[:, 0] > 0.5) & (s2 > 3e-3))[:, None]  # (else skin and backdrop alike, or one side far: the
    # matte's share, and the plate's own pixel)
    s = np.where(mixed[:, 0], np.clip(((p0 - Wr) * sk).sum(-1) / np.maximum(s2, 1e-5), 0, 1),
                 np.clip(hand[y0:y1, x0:x1][ys, xs], 0, 1))[:, None]
    fringe = np.where(mixed, p0 - (s * Sk + (1 - s) * Wr), 0) * _smooth(np.abs(Dr).max(-1) / 0.03)[:, None]  # what
    # the mix does not explain, the lens's colour fringe, as far as the air has changed what is behind
    Pb[ys, xs] = p + (p0 - (1 - EDGE[2]) * fringe + (1 - s) * Dr + s * Dk - p) * _smooth(reach[ys, xs])[:, None]


def _mix(P: np.ndarray, alpha: np.ndarray, colour: np.ndarray, noise: np.ndarray | None, code: float) -> None:
    """_over's work on pixels ``P`` (..., 3) in place, ``alpha``, ``colour`` and ``noise`` matching them."""
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


def _landmarks_px(track: dict, n: int, w: int, h: int) -> list[np.ndarray | None]:
    """Each frame's 21 landmarks (x, y) in this plate's px, or None on a frame without them."""
    kx, ky = w / float(track.get("W", w)), h / float(track.get("H", h))
    lms = list(track.get("landmarks") or [])[:n]
    return [None if L is None else np.asarray(L, np.float64)[:, :2] * (kx, ky) for L in lms + [None] * (n - len(lms))]


def _nearest_filled(x: np.ndarray, fill: float) -> np.ndarray:
    """``x`` (n,) with each nan taken from the nearest frame that has a value (all ``fill`` when none has)."""
    have = np.flatnonzero(~np.isnan(x))
    if have.size == 0:
        return np.full(len(x), fill)
    return x[have[np.abs(np.arange(len(x))[:, None] - have[None, :]).argmin(1)]]


def knuckle_span(track: dict, n: int, w: int, h: int) -> np.ndarray:
    """(n,) px between the index and little fingers' knuckles (landmarks 5 and 17), from the nearest frame with
    landmarks; zeros for a track without them."""
    L = _landmarks_px(track, n, w, h)
    return _nearest_filled(np.array([np.nan if p is None else np.linalg.norm(p[5] - p[17]) for p in L]), 0.0)


def hand_span(track: dict, n: int, w: int, h: int) -> np.ndarray:
    """(n,) px between the thumb's tip and the little finger's (landmarks 4 and 20), from the nearest frame with
    landmarks; zeros for a track without them."""
    L = _landmarks_px(track, n, w, h)
    return _nearest_filled(np.array([np.nan if p is None else np.linalg.norm(p[4] - p[20]) for p in L]), 0.0)


def openness(track: dict, n: int, w: int, h: int) -> np.ndarray:
    """(n,) how open the hand is, 1 for the take's most open: the four fingers' mean reach past their knuckles over the
    palm's length (wrist to middle knuckle), which a hand turned to the lens foreshortens alike (the wrist-to-fingertip
    reach peaked as the fingers flung open, a frame before they were); ones for a track without landmarks."""
    L = _landmarks_px(track, n, w, h)
    reach = np.array([np.nan if p is None else np.linalg.norm(p[[8, 12, 16, 20]] - p[[5, 9, 13, 17]], axis=1).mean()
                      / max(float(np.linalg.norm(p[9] - p[0])), 1e-6) for p in L])
    reach = _nearest_filled(reach, 1.0)
    return reach / max(float(reach.max()), 1e-9)


def _open_from(opened: np.ndarray, f: int) -> int:
    """The frame a push on frame ``f`` is placed on (OPEN_LOOK, OPEN)."""
    look = opened[f: min(len(opened), f + OPEN_LOOK + 1)]
    ok = np.flatnonzero(look >= OPEN)
    return f + int(ok[0] if ok.size else np.argmax(look))


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
    for a, b in zip(np.r_[0, cut + 1], np.r_[cut, frames.size - 1], strict=True):
        t0 = int(frames[a])
        t_rel = max(int(frames[b]) + 1 + BALL_LINGER, t0 + BALL_MIN)
        if lives and t0 < lives[-1][1] + BALL_RELEASE:  # the last one has not burst yet: it carries on
            lives[-1] = (lives[-1][0], max(lives[-1][1], t_rel))
        else:
            lives.append((t0, t_rel))
    return lives


def ball_pulses(timeline: Timeline, level: np.ndarray, life: tuple[int, int]) -> list[int]:
    """The frames a ball living over ``life`` (first, release) bursts and gathers again on: the hits the voice comes
    back with after dipping under PULSE_DIP in the six frames before, at least PULSE_GAP frames after the ball
    started and after its last pulse, and BALL_RELEASE frames before its release at the latest."""
    t0, t_rel = life
    out, last = [], t0
    for f in timeline.hits:
        if f - last >= PULSE_GAP and f < t_rel - BALL_RELEASE and float(level[max(0, f - 6): f].min()) < PULSE_DIP:
            out.append(int(f))
            last = f
    return out


def ball_pulse(t: int, pulses: list[int]) -> tuple[float, float]:
    """(how far the ball is burst on frame ``t`` by its comeback hits ``pulses``, swelling it as much wider: PULSE_BURST
    at the most, on the frame after the hit, gathered again over PULSE_TAU; how much its shell flashes, 0..1: fully on
    the hit and the frame after it, easing off over PULSE_FLASH's last)."""
    burst = max([PULSE_BURST * float(_smooth((t - f + 1) / 2)) * math.exp(-max(0, t - f - 1) / PULSE_TAU)
                 for f in pulses if f <= t] or [0.0])
    flash = max([math.exp(-max(0, t - f - 1) / PULSE_FLASH[4]) for f in pulses if f <= t] or [0.0])
    return burst, flash


def spin_axis(normal: np.ndarray) -> np.ndarray:
    """The ball's spin axis off a palm facing ``normal``: the normal, leant further the way it leans across the
    picture (up it, for a palm flat to the lens) as far as it takes to keep its z (towards the lens) under
    BALL_AXIS_Z."""
    if normal[2] <= BALL_AXIS_Z:
        return normal
    d = np.asarray(normal[:2], np.float64)
    d = d + (0.0, -0.1 * max(0.0, 1 - float(np.hypot(*d)) / 0.2))
    d = d / max(float(np.hypot(*d)), 1e-9)
    return np.array([*(d * math.sqrt(1 - BALL_AXIS_Z ** 2)), BALL_AXIS_Z])


class _Ball:
    """One ball of air: its seeded streaks, dust and textures, and the state it carries from frame to frame (how far
    it has spun, its axes)."""

    def __init__(self, seed: int, salt: int):
        rng = fx._rng(seed, _BALL, salt)
        N = BALL_STREAKS
        self.wisp = rng.random(N) < STREAK_WISPS
        tilt = np.where(self.wisp, rng.uniform(0.3, 1, N) ** 0.7 * STREAK_TILT, np.abs(rng.normal(0, 0.04, N)))
        az = rng.uniform(0, 2 * np.pi, N)
        ct, st, ca, sa = np.cos(tilt), np.sin(tilt), np.cos(az), np.sin(az)
        # each streak's own orbit: (u, v) span its plane, w is its axis, leaning off the ball's by its tilt
        self.u = np.column_stack([ct * ca, ct * sa, -st])
        self.v = np.column_stack([-sa, ca, np.zeros(N)])
        self.w = np.column_stack([st * ca, st * sa, ct])
        self.bands = bands = np.asarray(STREAK_BANDS) + rng.uniform(-0.06, 0.06, len(STREAK_BANDS))
        lat = np.where(self.wisp, rng.normal(0, 0.45, N),
                       bands[rng.integers(0, len(bands), N)] + rng.normal(0, 0.05, N))
        self.lat = np.clip(lat, -1.3, 1.3)
        self.band_of = np.where(self.wisp, -1, np.abs(self.lat[:, None] - bands[None, :]).argmin(1))
        more = fx._rng(seed, _BALL, salt, _SWIRL)  # each band's own lean (BAND_TILT): how far, about which way, and
        # which way round that turns
        self.b_tilt = BAND_TILT[0] * more.uniform(0.5, 1.0, bands.size)
        self.b_az = more.uniform(0, 2 * np.pi, bands.size)
        self.b_turn = more.choice((-1.0, 1.0), bands.size)
        self.rho = np.where(self.wisp, rng.uniform(0.86, 1.03, N), 1.03 - (
            1.03 - BAND_DEPTH[0]) * rng.random(N) ** BAND_DEPTH[1])
        self.phase = rng.uniform(0, 2 * np.pi, N)
        self.span = np.where(self.wisp, rng.uniform(1.1, 3.0, N), rng.uniform(0.9, 2.1, N))  # (arcs, not rings: whole
        # turns on shared bands stacked into a jar)
        self.speed = np.where(self.wisp, rng.uniform(0.85, 1.2, N), rng.uniform(0.93, 1.07, N))  # (the bands turn as
        # one)
        self.helix = np.where(self.wisp, rng.uniform(-0.18, 0.18, N), rng.uniform(-0.05, 0.05, N))
        self.bright = rng.uniform(0.12, 1.0, N) ** 2 * np.where(self.wisp, 0.6, BAND_ALPHA) * np.cos(
            self.lat) ** BAND_POLE  # most faint, a few bright (as bright as each other, the streaks read as wire)
        self.glow = rng.uniform(0, 2 * np.pi, N)  # ... and each brightens and fades as it turns, so a different few
        self.glow_f = rng.uniform(0.08, 0.2, N)  # read at a time, at this many times the spin
        self.fl_ph = rng.uniform(0, 2 * np.pi, N)  # ... and now and then a band's head flares (FLARE), at this many
        self.fl_f = np.where(self.wisp, 0.0, rng.uniform(0.2, 0.5, N))  # times the spin
        self.soft = (rng.random(N) >= STREAK_FINE) | self.wisp  # broad soft wisps; the rest fine streaks
        self.wob = rng.uniform(0, 2 * np.pi, N)
        self.gaps = rng.uniform(1.0, 2.6, N)  # each streak comes and goes along its length this many times (whole,
        # each was a wire drawn round the ball)
        self.gph = rng.uniform(0, 2 * np.pi, N)
        m = BALL_DUST  # dust inside, carried round the axis with the spin, a little faster further in
        self.d_lat = np.arcsin(rng.uniform(-0.95, 0.95, m))
        self.d_rho = rng.uniform(0.2, 1.0, m) ** 0.6
        self.d_phase = rng.uniform(0, 2 * np.pi, m)
        self.d_glint = rng.uniform(0, 2 * np.pi, m)
        self.d_gf = rng.uniform(0.2, 0.9, m)
        self.d_bright = rng.uniform(0.3, 1.0, m) ** 1.5
        self.d_r0 = rng.uniform(*DUST_R, m)
        self.d_dark = rng.random(m) < 0.12  # specks of grit, darker than the wall
        q = ROOM_DUST  # the room's dust, drawn in and round as the ball feeds on the air, from just outside it (from
        # 2.7 radii it swept past the frame's edges)
        self.r_rho = rng.uniform(1.05, 1.2, q)
        self.r_phase = rng.uniform(0, 2 * np.pi, q)
        self.r_lat = rng.normal(0, 0.2, q)
        self.r_bright = rng.uniform(0.3, 1.0, q) ** 1.5
        self.r_r0 = rng.uniform(*DUST_R, q)
        k = MOTES  # bigger flecks riding the shell, flashing as they turn to the light
        self.m_lat = np.arcsin(rng.uniform(-0.8, 0.8, k))
        self.m_rho = rng.uniform(0.7, 1.0, k)
        self.m_phase = rng.uniform(0, 2 * np.pi, k)
        self.m_tum = rng.uniform(0, 2 * np.pi, k)
        self.m_tf = rng.uniform(0.15, 0.45, k)
        self.m_r0 = rng.uniform(DUST_R[1], 4.5, k)
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

    def band_turns(self, t: int) -> np.ndarray:
        """(bands, 3, 3): how each band of latitude leans off the spin axis on frame ``t`` (BAND_TILT), as rotations
        in the ball's own axes (the spin axis the third)."""
        psi = self.b_az + self.b_turn * math.radians(BAND_TILT[1]) * t / fx.FPS
        h = np.stack([np.cos(psi), np.sin(psi), np.zeros_like(psi)], -1)  # the level axis each leans about
        K = np.zeros((psi.size, 3, 3))
        K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -h[:, 2], h[:, 1], -h[:, 0]
        K -= K.transpose(0, 2, 1)
        c, s = np.cos(self.b_tilt)[:, None, None], np.sin(self.b_tilt)[:, None, None]
        return c * np.eye(3) + s * K + (1 - c) * h[:, :, None] * h[:, None, :]

    def _mist(self, sx, sy, sign, B, normals):
        """The shell's density on the near (sign +1) or far (-1) side, fixed in the spinning air, long along the flow
        and thin across it; ``sx``, ``sy`` in radii from the middle. On the near side a band of it runs along each
        band of latitude the streaks run on (BAND_MIST; ``normals``, each band's axis in the camera's axes), broken
        along the flow as the shell is."""
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
        limb = MIST_CORE * np.abs(z) + (1 - MIST_CORE) * (0.22 + 0.78 * (1 - np.abs(z)) ** 1.6)  # the eye looks
        # through the most air at its middle, and grazes the most of its shell at the edge
        out = dens * polar * limb
        if sign > 0:
            gap = np.min([np.abs(np.arcsin(np.clip(sx * nb[0] + sy * nb[1] + z * nb[2], -1, 1)) - b)
                          for nb, b in zip(normals, self.bands, strict=True)], axis=0)  # to the nearest band
            band = np.exp(-(gap / BAND_MIST[1]) ** 2)
            out = out + BAND_MIST[0] * band * np.clip(0.6 + 0.4 * m, 0, None) * np.sqrt(z)  # (on the face, not
            # piled on the limb)
        return out

    def _core(self, sx, sy, B):
        """The puff of swirling air through the ball's middle (CORE_PUFF) at ``sx``, ``sy`` (radii from the middle):
        smooth noise fixed in the spinning air, wound into a swirl round the spin axis, as thick as the eye's path
        through it. (Read in angles round the axis, every feature converged on it: a star.)"""
        c = CORE_PUFF[0]
        l1, l2, l3 = (sx * B[i, 0] + sy * B[i, 1] for i in range(3))  # (its middle plane, in the ball's axes)
        th = CORE_PUFF[1] * (1 - np.clip(np.sqrt(l1 * l1 + l2 * l2) / c, 0, 1)) - self.spin
        q1, q2 = l1 * np.cos(th) - l2 * np.sin(th), l1 * np.sin(th) + l2 * np.cos(th)
        m = cv2.remap(np.ascontiguousarray(self.noise[..., 2]), (q1 / c * 40 + 128).astype(np.float32),
                      ((q2 + 0.6 * l3) / c * 40 + 128).astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_WRAP)
        return np.sqrt(np.clip(c * c - sx * sx - sy * sy, 0, None)) / c * np.clip(0.45 + 0.55 * m, 0, None) ** 2

    def points(self, c, R, B, form, rel, vis, omega, k, turns):
        """The streaks the ball carries, as points: (x, y, depth px towards the lens, weight, kind) with kind 0 a fine
        streak, 1 a broad wisp, 2 and 3 the tails of a fine streak and of a broad wisp (drawn thinner:
        STREAK_TAPER). No streak is brighter than STREAK_PEAK times the brighter half's mean, but a few heads flare
        (FLARE), drawn again as broad wisps; each streak is brightest on the near side (BALL_NEAR); a band's streaks
        lean with it (``turns``: band_turns)."""
        spread = 1 + 1.4 * (1 - form) ** 2  # forming: it gathers in from wide
        on = self.band_of >= 0
        Rs = turns[np.maximum(self.band_of, 0)]
        u, v, w_ = ((np.where(on[:, None], np.einsum("nij,nj->ni", Rs, e), e)) for e in (self.u, self.v, self.w))
        # streaks: each a comet of moving air along its orbit, its head a half frame on (the shutter)
        L = self.span * R * self.rho * spread * (1 + rel)
        S = int(np.clip(L.max() / (STREAK_STEP * k), 16, 2400))
        s = np.linspace(0, 1, S)[None, :]
        head = self.phase + self.speed * (self.spin + 0.5 * omega)
        span = (self.span * (0.4 + 0.6 * form) * (1 - 0.6 * rel))[:, None]
        phi = head[:, None] - s * span
        lat = self.lat[:, None] + self.helix[:, None] * s + 0.1 * np.sin(0.2 * self.spin + self.wob)[:, None]
        rho = R * spread * self.rho[:, None] * (1 + 0.5 * rel * (1 + s))
        cl, sl = np.cos(lat), np.sin(lat)
        loc = rho[..., None] * (cl[..., None] * (np.cos(phi)[..., None] * u[:, None] + np.sin(phi)[..., None]
                                                 * v[:, None]) + sl[..., None] * w_[:, None])
        pts = loc @ B  # (N, S, 3) in the camera's axes
        tail = (1 - s) ** 1.2 * _smooth(s / 0.08 + 0.15) * np.clip(0.55 + 0.6 * np.sin(
            2 * np.pi * self.gaps[:, None] * s + self.gph[:, None] + 0.15 * self.spin), 0.06, 1.0)  # in wisps, coming
        # and going
        ds = (span[:, 0] * rho.mean(1) / (S - 1))[:, None]  # px of streak each point stands for ...
        d2 = np.hypot(*np.moveaxis(np.diff(pts[..., :2], axis=1), -1, 0))
        d2 = np.concatenate([d2, d2[:, -1:]], 1)  # ... and covers in the picture: where a band turns towards the
        # lens its points bunch up, and weighed by their length in the room they piled into a white crescent; a
        # little of that is kept (STREAK_END), as a filament seen end on does gather more light
        g = self.bright * (0.5 + 0.5 * np.sin(self.glow + self.glow_f * self.spin) ** 2)
        g = np.minimum(g, STREAK_PEAK * float(g[g >= np.median(g)].mean()))
        wt = g[:, None] * tail * vis * d2 * np.clip(ds / np.maximum(d2, 1e-3), 1.0, STREAK_END[0]) ** STREAK_END[1]
        kind = (self.soft[:, None].astype(np.int8) + 2 * (s > STREAK_TAPER[0])).astype(np.int8)
        xs, ys, zs, ws, ks = [pts[..., 0].ravel()], [pts[..., 1].ravel()], [pts[..., 2].ravel()], [wt.ravel()], [
            kind.ravel()]
        fl = np.where(self.fl_f > 0, np.maximum(np.cos(self.fl_ph + self.fl_f * self.spin), 0.0) ** FLARE[1], 0.0)
        hot = np.flatnonzero(fl > 0.02)
        if hot.size:  # the flaring heads, again as broad wisps
            nh = max(2, S // 4)
            p = pts[hot, :nh]
            xs.append(p[..., 0].ravel()), ys.append(p[..., 1].ravel()), zs.append(p[..., 2].ravel())
            ws.append((FLARE[0] * fl[hot, None] * wt[hot, :nh]).ravel()), ks.append(np.ones(p.shape[0] * nh, np.int8))
        x, y, z, wgt, kd_ = (np.concatenate(v) for v in (xs, ys, zs, ws, ks))
        wgt = wgt * (BALL_BACK + (1 - BALL_BACK) * _smooth((z / R + BALL_NEAR) / (2 * BALL_NEAR)))
        return c[0] + x, c[1] + y, z, wgt, kd_

    def specks(self, R, B, form, rel, vis, omega, t, loud):
        """The dust the ball carries, as specks for _specks: (where each is half a frame back and now, (n, 2) px off
        the ball's middle; depth px towards the lens; weight; size px at 1080; dark): dust inside turning with the
        spin, brighter with the voice (``loud``: DUST_LOUD); the room's dust drawn in, spiralling slowly in round the
        equator; and bigger motes on the shell, flashing as each turns its face to the room's light."""
        spread = 1 + 1.4 * (1 - form) ** 2
        turn = (DUST_SPIN[0] + DUST_SPIN[1] * (1 - self.d_rho))
        prog = (self.r_phase * 0.37 + 0.02 * self.spin) % 1.0
        rr2 = R * (1.0 + (self.r_rho - 1.0) * (1 - prog))
        rr = self.d_rho * R * spread * (1 + 0.8 * rel)
        rr3 = self.m_rho * R * spread * (1 + 0.9 * rel)

        def at(spin):
            la = self.d_lat + 0.1 * np.sin(0.3 * spin + self.d_glint)
            ph = self.d_phase + turn * spin
            ph2 = self.r_phase + 0.55 * spin * R / rr2
            ph3 = self.m_phase + 1.05 * spin
            return np.concatenate([
                (rr * np.cos(la))[:, None] * (np.cos(ph)[:, None] * B[0] + np.sin(ph)[:, None] * B[1])
                + (rr * np.sin(la))[:, None] * B[2],
                rr2[:, None] * (np.cos(ph2)[:, None] * B[0] + np.sin(ph2)[:, None] * B[1])
                + (self.r_lat * R)[:, None] * B[2],
                (rr3 * np.cos(self.m_lat))[:, None] * (np.cos(ph3)[:, None] * B[0] + np.sin(ph3)[:, None] * B[1])
                + (rr3 * np.sin(self.m_lat))[:, None] * B[2]])

        was, now = at(self.spin), at(self.spin + 0.5 * omega)
        gl = 0.3 + 0.7 * np.abs(np.cos(self.d_glint + self.d_gf * t)) ** 6
        glint = 0.25 + 0.75 * np.abs(np.cos(self.m_tum + self.m_tf * t)) ** 8
        wgt = np.concatenate([self.d_bright * gl * (DUST_LOUD[0] + DUST_LOUD[1] * loud),
                              self.r_bright * np.sin(np.pi * prog) * (1 - rel), 1.5 * glint]) * vis
        r0 = np.concatenate([self.d_r0, self.r_r0, self.m_r0])
        dark = np.concatenate([self.d_dark, np.zeros(ROOM_DUST + MOTES, bool)])
        return was[:, :2], now[:, :2], now[:, 2], wgt, r0, dark

    def field(self, dx: np.ndarray, dy: np.ndarray, R: float, vis: float, t: int, turn: float,
              keep: np.ndarray | None, k: float) -> tuple[np.ndarray, ...]:
        """The ball's bend at points ``dx`` (1, m), ``dy`` (n, 1) px off its middle, on a grid FIELD_Q px apart:
        radius ``R`` px, ``vis`` how much of itself it is, ``turn`` the sense its near side crosses the picture in,
        ``keep`` how much of each point is over the hand (None: no hand). Returns (the px each point reads the plate
        from, x and y, off itself; r, its distance from the middle in radii of the wobbling outline; phi, its angle;
        wob; the co-rotating noise, 3 channels). The twist and the squeeze are continuous, the rim included, and
        eased over BALL_EASE (the shell's chord rises steeply at its rim and round its core)."""
        phi = np.arctan2(dy, dx)
        wob = 1 + BALL_WOB[0] * np.sin(3 * phi - 0.4 * self.spin + 1.3 + 0.9 * math.sin(0.07 * t)) \
            + BALL_WOB[1] * np.sin(5 * phi + 0.7 * self.spin - 0.11 * t)
        r = np.hypot(dx, dy) / (R * wob)
        z = np.sqrt(np.clip(1 - r * r, 0, 1))
        shell = (z - np.sqrt(np.clip(0.36 - r * r, 0, 1))) / 0.8  # the chord through the shell round a 0.6 core
        draft = np.exp(-((r - 1) / np.where(r > 1, BALL_DRAFT, 0.25)) ** 2)  # the swirl drags the air round it a
        # little (continuous across the rim: switched on outside it only, the twist jumped there and tore a straight
        # edge behind the ball)
        sw = (np.clip(shell, 0, 1) + 0.45 * draft) * vis
        lobes = 0.6 + 0.4 * np.cos(3 * (phi - turn * self.spin) + 1.2 * np.sin(2 * phi - 0.7 * turn * self.spin))
        rim = np.exp(-((r - 0.97) / 0.06) ** 2)
        ease, wide = (max(0.5, e * R / FIELD_Q) for e in BALL_EASE[:2])
        outer = np.exp(-((r - 1) / BALL_EASE[2]) ** 2)

        def eased(x):  # over BALL_EASE's first, and its second in the outer shell
            x = x.astype(np.float32)
            near = cv2.GaussianBlur(x, (0, 0), ease)
            return near + (_broad_blur(x, wide) - near) * outer

        theta = eased(turn * BALL_TWIST * sw * lobes)
        m = eased(vis * (BALL_LENS * z * z - BALL_RIM * rim))  # the middle magnified, easing off to the rim (as the
        # chord, it fell off steeply there and kinked what was behind), the rim pulled in from outside it
        sx, sy = dx * (1 - m), dy * (1 - m)
        ct, st = np.cos(-theta), np.sin(-theta)  # read from behind the flow: the plate is dragged with the spin
        rx, ry = sx * ct - sy * st, sx * st + sy * ct
        cs, sn = math.cos(-turn * self.spin), math.sin(-turn * self.spin)
        u = (dx * cs - dy * sn) / (0.5 * R) * 20 + 128
        v = (dx * sn + dy * cs) / (0.5 * R) * 20 + 128
        nz = cv2.remap(self.noise, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_WRAP)
        tb = BALL_TURB * k * cv2.GaussianBlur(sw.astype(np.float32), (0, 0), ease)
        bend = 1.0
        if keep is not None:
            bend, tb = 1 - BALL_HAND[0] * keep, tb * (1 - BALL_HAND[1] * keep)
        return (rx - dx) * bend + nz[..., 0] * tb, (ry - dy) * bend + nz[..., 1] * tb, r, phi, wob, nz

    def draw(self, P: np.ndarray, off: tuple[int, int], c: np.ndarray, R: float, a: np.ndarray, form: float,
             rel: float, pulse: float, flash: float, idle: float, omega: float, loud: float, t: int,
             hand: np.ndarray | None, tone: np.ndarray, noise: np.ndarray | None, code: float,
             k: float) -> np.ndarray | None:
        """Draws the ball on ``P`` (a linear crop of the frame whose top-left is ``off`` = (x, y) in frame px) in
        place: centre ``c`` (frame px), radius ``R`` (px), spin axis ``a``; ``form`` 0..1 how gathered, ``rel`` 0..1
        how far it has let go, ``pulse`` how far a comeback hit has burst it (PULSE_BURST at the most) and ``flash``
        0..1 how much its shell flashes with it (PULSE_FLASH), ``idle`` how much of itself it is between two sounds,
        ``omega`` its spin this frame (rad), ``loud`` the voice's level, ``hand`` the matte over the crop (0..1;
        softened by HAND_SOFT, and bent as the hand seen through the ball is); the hand's edge redrawn last
        (_reedge). First its bend: the lens and twist (field) and the swirling air's density (BALL_DENS: its shell,
        its middle's puff and its streaks, smoothed) bending what is behind towards where it is denser, with that
        density's shadowgraph and the band round its rim; then the faint sheen of its streaks and its dust, in the
        colours of what is behind them. Returns the crop as bent, before what the ball carries was laid on it (None
        when it drew nothing)."""
        self.spin += omega
        B = self.turn_axes(a)
        turns = self.band_turns(t)
        normals = turns[:, :, 2] @ B  # each band's axis in the camera's axes
        h, w = P.shape[:2]
        matte = hand
        if hand is not None:
            hand = cv2.GaussianBlur(hand, (0, 0), HAND_SOFT * k)
        ox, oy = off
        cx, cy, R = float(c[0] - ox), float(c[1] - oy), float(R)  # (plain floats keep the fields float32)
        vis = float(form * (1 - rel) ** 1.5 * idle)
        if vis <= 1e-3:
            return None
        spread = max(rel, pulse)  # how far its streaks and dust are flung out
        # which way the near side's air crosses the picture: the plate behind is dragged the same way
        ph = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        pt = np.cos(ph)[:, None] * B[0] + np.sin(ph)[:, None] * B[1]
        vel = -np.sin(ph)[:, None] * B[0] + np.cos(ph)[:, None] * B[1]
        near = pt[:, 2] > 0
        turn = float(np.sign(np.mean(pt[near, 0] * vel[near, 1] - pt[near, 1] * vel[near, 0])) or 1.0) if near.any() \
            else 1.0

        # ---- the streaks it carries, splatted: filaments of denser air in its bend, and a faint sheen; the far
        # side's behind the fingers, and over the skin only broad and barely there
        x, y, z, wgt, kind = self.points(c, R, B, form, spread, vis, omega, k, turns)
        wgt = wgt * (1 + PULSE_FLASH[0] * flash * (kind % 2) + PULSE_FLASH[1] * flash * (1 - kind % 2))
        x, y = x - ox, y - oy
        if hand is not None:
            xi, yi = np.clip(x.astype(np.int64), 0, w - 1), np.clip(y.astype(np.int64), 0, h - 1)
            over = hand[yi, xi]
            wgt = wgt * (1 - over * (1 - _smooth((z / R + BALL_NEAR) / (2 * BALL_NEAR)))) * (
                1 - (1 - STREAK_SKIN[1]) * over)
            kind = np.where((over > 0.5) & (kind % 2 == 0), kind + 1, kind).astype(np.int8)
        bufs = [np.zeros((h, w), np.float32) for _ in range(4)]
        for j, buf in enumerate(bufs):
            sel = kind == j
            _splat(buf, x[sel], y[sel], wgt[sel])
        lit = np.zeros((h, w), np.float32)
        for j, px, a_ in ((0, STREAK_PX[0], 1.0), (1, STREAK_PX[1], 0.8), (2, STREAK_PX[0] * STREAK_TAPER[1], 1.0),
                          (3, STREAK_PX[1] * STREAK_TAPER[1], 0.8)):
            lit += _line_blur(bufs[j], px * k * (1 + PULSE_FLASH[2] * flash)) * np.float32(a_ * STREAK_ALPHA)

        # ---- the bend and the shadowgraph: on a coarse grid round the ball
        box = _clip_box(cx, cy, 1.7 * R, 1.7 * R, w, h)
        if box is not None:
            y0, y1, x0, x1 = box
            bh, bw = y1 - y0, x1 - x0
            gx, gy = _coarse_grid(box, FIELD_Q)
            dx, dy = gx - cx, gy - cy
            hc = None if hand is None else cv2.resize(hand[y0:y1, x0:x1], (gx.shape[1], gy.shape[0]),
                                                      interpolation=cv2.INTER_AREA)
            keep = None if hc is None else cv2.GaussianBlur(np.minimum(2 * hc, 1), (0, 0), 0.06 * R / FIELD_Q)  # over
            # the hand and a little past the matte's edge (a fingertip it missed was bent apart from the rest of the
            # finger), eased in over a few px round it (cut there, the fingers showed a ghost of themselves beside them)
            fx_, fy_, r, phi, wob, nz = self.field(dx, dy, R, vis * (1 - pulse), t, turn, keep, k)  # (burst, its air
            # is flung out thin and bends less: swelled at full strength, it twisted the switch plate behind it)
            bend = 1.0 if keep is None else 1 - BALL_HAND[0] * keep
            # the swirling air's density, smoothed: what is behind is bent towards where it is denser
            sx, sy = dx / (R * wob), dy / (R * wob)
            edge = _smooth((1 - r) / 0.25 + 0.3)  # a soft edge, not a rim (over 0.12 radii, a straight edge behind
            # the rim was bent with a jog)
            far = self._mist(sx, sy, -1, B, normals)
            if hc is not None:  # the far side of the shell passes behind the fingers
                far = far * (1 - 0.9 * hc)
            dens = (BALL_DENS[0] * self._mist(sx, sy, 1, B, normals) + BALL_DENS[1] * far) * edge \
                + BALL_DENS[2] * self._core(sx, sy, B) + BALL_DENS[3] * cv2.resize(
                    lit[y0:y1, x0:x1], (gx.shape[1], gy.shape[0]), interpolation=cv2.INTER_AREA)
            sig = BALL_REFRACT[1] * k / FIELD_Q
            dens = cv2.GaussianBlur(dens.astype(np.float32), (0, 0), sig) * np.float32(
                vis * (1 + PULSE_FLASH[3] * flash))
            ddy, ddx = np.gradient(dens)
            pull = np.float32(BALL_REFRACT[0] * k * sig * math.sqrt(math.e)) * bend  # (a bump of density 1 as wide
            # as the smoothing bends what is behind it BALL_REFRACT's first at its steepest)
            fx_, fy_ = fx_ + pull * ddx, fy_ + pull * ddy  # (read from up the density: a bump of it magnifies what is
            # behind it, as the lens does)
            ca = BALL_CA if keep is None else _upsample((BALL_CA * (1 - keep)).astype(np.float32), bh, bw)
            DX, DY = _upsample(fx_.astype(np.float32), bh, bw), _upsample(fy_.astype(np.float32), bh, bw)
            _remap_rgb(P, DX, DY, box, ca)
            if matte is not None:  # the matte bent with the hand seen through the ball
                matte = matte.copy()
                matte[y0:y1, x0:x1] = cv2.remap(matte[y0:y1, x0:x1], np.arange(bw, dtype=np.float32)[None, :] + DX,
                                                np.arange(bh, dtype=np.float32)[:, None] + DY, cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_REPLICATE)
                hand = cv2.GaussianBlur(matte, (0, 0), HAND_SOFT * k)
            # the shadowgraph: ripples where the density curves (light gathered where it is denser), and a soft band
            # on the rim, broken into arcs that turn with the spin and wander in and out with them, darker inside it;
            # over the hand eased off as the bend is (its bright arc across a lit palm read as glare on the skin)
            ripple = np.clip(-cv2.Laplacian(dens, cv2.CV_32F) * np.float32(sig * sig / 2), -1.5, 1.5)
            arc = cv2.remap(self.arcs, (((phi - turn * self.spin * 0.6) % (2 * np.pi)) * (512 / (2 * np.pi)))
                            .astype(np.float32), np.zeros_like(phi, np.float32), cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_WRAP)
            band = np.exp(-((r - 1.0 - 0.06 * arc) / BALL_BAND) ** 2) * np.clip(0.75 + 0.45 * arc, 0.2, 1.6) \
                - 0.55 * np.exp(-((r - 0.88) / 0.07) ** 2)
            gain_ = (BALL_SHADOW * vis * band + BALL_RIPPLE * ripple) * bend
            if hc is not None:  # the palm under it takes the light the spinning air focuses, in moving patches
                gain_ = gain_ + BALL_CAUSTIC * vis * hc * (np.clip(nz[..., 2], 0, None) ** 2 - 0.4) * \
                    _smooth((1.15 - r) / 0.3)
            P[y0:y1, x0:x1] *= 1 + _upsample(gain_.astype(np.float32), bh, bw)[..., None]
        bent = P.copy()  # (bent, before what the ball carries is laid on)

        # ---- the streaks' sheen, in the colour of what is behind them, lifted a little
        sb = _bbox(lit > 1e-3)
        behind = None
        if sb is not None:
            if hand is not None:
                lit = np.minimum(lit, 1 - (1 - STREAK_SKIN[0]) * hand)
            behind = _behind(P)
            y0, y1, x0, x1 = sb
            _over(P[y0:y1, x0:x1], lit[y0:y1, x0:x1], _lit_by(_up(behind, sb), tone, *STREAK_TONE),
                  None if noise is None else noise[y0:y1, x0:x1], code)

        # ---- the dust and motes it carries, drawn as the blast's dust is
        was, now, zp, wgt, r0, dark = self.specks(R, B, form, spread, vis, omega, t, loud)
        wgt = wgt * (BALL_BACK + (1 - BALL_BACK) * _smooth((zp / R + BALL_NEAR) / (2 * BALL_NEAR)))
        was, now = was + (cx, cy), now + (cx, cy)
        if hand is not None:
            xi, yi = np.clip(now[:, 0].astype(np.int64), 0, w - 1), np.clip(now[:, 1].astype(np.int64), 0, h - 1)
            wgt = wgt * (1 - hand[yi, xi] * (zp < 0))
        sc = 4
        lit_s, drk_s = _speck_bufs(h, w, sc)
        _specks(was, now, np.hypot(*(now - was).T), np.clip(1 - zp / (FOCAL * 1080 * k), 0.5, 1.5), wgt, r0, dark,
                lit_s, drk_s, sc, k)
        _lay_specks(P, lit_s, drk_s, sc, tone, None if noise is None else (lambda y0, y1, x0, x1: noise[y0:y1, x0:x1]),
                    code, k, behind)
        if matte is not None:
            _reedge(P, bent, matte, k)
        return bent


def _ball_shot(plate: np.ndarray, out: np.ndarray, track: dict, mask: np.ndarray | None, tl: Timeline, seed: int,
               code: float, say) -> None:
    n, h, w = plate.shape[:3]
    level = voice_level(tl)
    lives = ball_lives(tl, level)
    xy, _, size = fx.track_arrays(track, n, w, h)
    xy, size = _gauss_smooth(xy, SMOOTH_XY), _gauss_smooth(size, SMOOTH_SIZE)
    span = _gauss_smooth(knuckle_span(track, n, w, h), SMOOTH_SIZE)
    tips = _gauss_smooth(hand_span(track, n, w, h), SMOOTH_SIZE)  # thumb tip to little fingertip (BALL_MAX)
    tips = np.where(tips > 0, tips, SPAN_SIZES * size)
    normal = palm_normals(track, n, w, h)
    feed = fed(tl, level)
    k = min(h, w) / 1080.0
    total = sum(min(n, b + BALL_RELEASE) - a for a, b in lives)
    done = 0
    blasts = []
    for j, (t0, t_rel) in enumerate(lives):
        ball = _Ball(seed, j)
        held, idle, tone_s, shift_before = 0, 1.0, None, 0.0
        pulses = ball_pulses(tl, level, (t0, t_rel))
        for t in range(t0, min(n, t_rel + BALL_RELEASE)):
            held += int(feed[t])
            idle += (0.35 if feed[t] else 0.06) * (float(feed[t]) - idle)  # thins out in a silence, back at once
            grow = min(1.0, held / (BALL_GROW_S * fx.FPS))
            form = float(_smooth((t - t0 + 1) / BALL_FORM))
            burst, flash = ball_pulse(t, pulses)
            rel = float(np.clip((t - t_rel + 1) / BALL_RELEASE, 0, 1))
            e = float(level[t])
            R0 = max(BALL_R * size[t] * (1 + BALL_GROW * grow), BALL_SPAN * span[t]) * (1 + BALL_BREATH * e)
            off = R0 + BALL_GAP * size[t]  # px off the palm, along its normal: nearer the lens, so drawn bigger
            near = 1.0 / max(0.5, 1 - normal[t, 2] * off / (FOCAL * min(h, w)))
            top = BALL_MAX * tips[t]
            if R0 * near > top:
                R0 = top / near
                off = R0 + BALL_GAP * size[t]
            R0 *= near
            R = min(R0 * (0.3 + 0.7 * form) * (1 + 1.0 * rel ** 0.7) * (1 + burst), top)
            lift = BALL_LIFT * size[t] * float(np.clip((normal[t, 2] - 0.7) / 0.3, 0, 1))
            c = xy[t] + normal[t, :2] * off * near - np.array([0.0, lift])
            omega = 2 * np.pi * BALL_SPIN * (0.5 + 0.5 * form + BALL_SPIN_VOICE * e + BALL_SPIN_HELD * grow) / fx.FPS
            if t == t_rel or t in pulses:  # letting go, or a pulse: the ball bursts into a push of air, its ring as
                # wide as the ball
                u, q = push_direction(track, t, normal[t], tl.aim, tl.tap_frame, n, w, h)
                gain = RELEASE_GAIN if t == t_rel else PULSE_GAIN
                blasts.append(_Blast(c, u, q, normal[t], size[t], w, h, t, gain[0] + gain[1] * e, seed,
                                     j if t == t_rel else 1000 + 64 * j + pulses.index(t), ring=R0, palm=xy[t],
                                     cloud_in=CLOUD_IN if t == t_rel else 0))
            tone = _tone(plate[t])
            tone_s = tone if tone_s is None else 0.85 * tone_s + 0.15 * tone
            pad = BALL_PAD * R * (1 + rel) + 8 * k
            box = _clip_box(c[0], c[1], pad, pad, w, h)
            done += 1
            if box is None:
                ball.spin += omega
                continue
            y0, y1, x0, x1 = box
            P = lin(out[t, y0:y1, x0:x1])  # (over the ball before it, still bursting)
            hand = None if mask is None else mask[min(t, len(mask) - 1), y0:y1, x0:x1].astype(np.float32) / 255.0
            noise = _grain(seed, _BALL, t, y1 - y0, x1 - x0)
            bent = ball.draw(P, (x0, y0), c, R, spin_axis(normal[t]), form, rel, burst, flash,
                             BALL_IDLE + (1 - BALL_IDLE) * idle, omega, e, t, hand, tone_s, noise, code, k)
            drawn = gam(P)
            if bent is not None and any(f - 1 <= t <= f + 2 for f in pulses):  # the whole picture's lift round a
                # pulse (PULSE_RISE): only the light it lays on is cut (cut with the bend, the fingers seen through it
                # were cross-faded with themselves unbent, a ghost)
                shift = float(drawn.astype(np.float32).sum() - out[t, y0:y1, x0:x1].astype(np.float32).sum()) / (
                    h * w * 3)
                if t + 1 in pulses:
                    shift_before = shift
                elif shift > shift_before + PULSE_RISE:
                    under = gam(bent).astype(np.float32)
                    light = drawn - under
                    cut = min(1.0, (shift - shift_before - PULSE_RISE) / max(float(light.sum()) / (h * w * 3), 1e-6))
                    drawn = np.clip(under + light * np.float32(1 - cut) + 0.5, 0, 255).astype(np.uint8)
            out[t, y0:y1, x0:x1] = drawn
            if done % 8 == 0:
                say("air", 0.5 * done / total, f"Spinning the air over your palm: frame {done} of {total}")
    if blasts:
        _push_frames(out, blasts, mask, tl, seed, code, lambda stage, share, msg: say(stage, 0.5 + 0.5 * share, msg))


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
    ``gain`` as loud as the loudest hit. Out of the palm by default; out of a ball when its vortex ring is ``ring`` px
    across as it leaves (RING_R's first by default), the ball's middle the origin and ``palm`` (frame px) the palm
    whose plane hides what is behind it, its dusty air coming in over its first ``cloud_in`` frames. All of it lives in
    the room as the start frame saw it: each frame is drawn through the camera's motion since. Points in it are 3D:
    x, y in px at the palm's depth, z in palm depths (the lens at 0)."""

    def __init__(self, origin, u, q: float, normal, size: float, w: int, h: int, start: int, gain: float, seed: int,
                 salt: int, ring: float | None = None, palm=None, cloud_in: int = CLOUD_IN):
        rng = fx._rng(seed, _BLAST, salt)
        self.o = np.asarray(origin, np.float64)
        self.palm = self.o if palm is None else np.asarray(palm, np.float64)
        self.cloud_in = int(cloud_in)
        self.u = np.asarray(u, np.float64)
        self.q, self.size, self.start, self.gain = float(q), float(size), int(start), float(gain)
        self.w, self.h = w, h
        side = min(w, h)
        self.side, self.k = side, side / 1080.0
        lobes = np.arange(3, 9)
        self.lob, self.lph = lobes, rng.uniform(0, 2 * np.pi, lobes.size)
        self.lamp = rng.uniform(0.5, 1.0, lobes.size) / lobes
        self.run = FRONT_SPEED[0] * self.k * (1 + FRONT_SPEED[1] * (1 - self.q))  # px a frame the front runs out at
        self.wake = cv2.GaussianBlur(_tile_noise(rng, 64, 720, 1.2, 1.2, 2), (0, 0), sigmaX=WAKE_ROUND, sigmaY=3.0)
        self.arcs = _tile_noise(rng, 1, 720, 6.0, 0.5).reshape(1, -1)  # where round the front its band is strong
        self.curl = _Curl(rng, 0.1 * side)
        f = FOCAL * side
        self.f = f
        self.pp = np.array([w / 2, h / 2])
        n3 = np.array([self.u[0] * self.q, self.u[1] * self.q, math.sqrt(max(0.0, 1 - self.q * self.q))])
        self.n3 = n3
        self.nrm = np.asarray(normal, np.float64) / max(float(np.linalg.norm(normal)), 1e-9)
        e1 = np.cross(self.nrm, [0.0, 0.0, 1.0] if abs(self.nrm[2]) < 0.9 else [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.nrm, e1)

        def thrown(d: np.ndarray, sp: np.ndarray, spread: float) -> tuple[np.ndarray, np.ndarray]:
            """Points leaving the palm's face (a hand ``spread`` across, just off its skin) along unit directions
            ``d`` (x, y and towards the lens) at ``sp`` px a frame: their positions and velocities."""
            a, b = rng.standard_normal((2, len(d))) * spread * size
            off = a[:, None] * e1 + b[:, None] * e2 + 0.08 * size * self.nrm
            p_ = np.column_stack([self.o[0] + off[:, 0], self.o[1] + off[:, 1], 1 - off[:, 2] / f])
            return p_, np.column_stack([d[:, 0] * sp, d[:, 1] * sp, -d[:, 2] * sp / f])  # (the lens: z falls)

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
        r0 = np.where(np.arange(N) < DUST_N, rng.uniform(*DUST_R, N), rng.uniform(DUST_R[1], 4.5, N))
        bright = rng.uniform(0.3, 1.0, N) ** 1.5
        dark = rng.random(N) < np.where(np.arange(N) < DUST_N, 0.12, 0.4)
        flake = np.arange(N) >= DUST_N
        # the dusty air: a vortex ring (ring() places its parcels), a share of it a wake. At the lens it comes at the
        # lens, and a true push's starts off the palm with its middle above it (LENS_LIFT; a ball's starts where the
        # ball is) and drifts to the frame's middle (LENS_MID), growing round the hand as it comes (along the push's
        # own axis, it slid off sideways on a hand off the picture's middle); each push's ring is tilted (less at the
        # lens, where a tilt sends it off sideways), sized, placed and paced a little off the last one's (RING_VARY)
        C = CLOUD_N
        self.at = float(_smooth((LENS_Q[1] - self.q) / (LENS_Q[1] - LENS_Q[0])))
        self.lens = 0.0 if ring is not None else float(  # how much of a true push at the lens it is
            _smooth((self.at - LENS_TRUE[0]) / (LENS_TRUE[1] - LENS_TRUE[0])))
        self.burst = ring is not None  # a ball's burst (BURST_SOFT, BURST_HAND)
        self.whole = 1.0 if ring is not None else self.lens  # how whole its ring crosses the hand (HAND_PUFF)

        def at_lens(pair: tuple[float, float]) -> float:
            """1 plus a LENS_* pair's share for this push: the partial one's, turning into the true one's."""
            return 1 + (pair[0] + (pair[1] - pair[0]) * self.lens) * self.at

        lift = np.zeros(3)  # px (x, y, and towards the lens) the ring's middle starts off the origin
        if ring is None:
            z0 = 1 - LENS_LIFT[0] * size * max(self.nrm[2], 0.0) / f
            seen0 = self.o + np.array([0.0, -LENS_LIFT[1] * size])
            lift = self.at * self.lens * np.array([*(self.pp + (seen0 - self.pp) * z0 - self.o), (1 - z0) * f])
        start = np.array([self.o[0] + lift[0], self.o[1] + lift[1], 1 - lift[2] / f])
        self.z0 = float(start[2])
        lens = np.array([self.pp[0] - start[0], self.pp[1] - start[1], start[2] * f])  # (straight at the lens: it
        # stays where it is in the picture, and ring() draws it to the frame's middle, LENS_MID)
        self.off_palm = 1.0 if ring is not None else self.at * self.lens  # how far off the palm the ring starts
        # (CLOUD_OVER)
        axis = (1 - self.at) * n3 + self.at * lens / np.linalg.norm(lens)
        axis /= np.linalg.norm(axis)
        a1 = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0])
        a1 /= np.linalg.norm(a1)
        lean = rng.uniform(0, 2 * np.pi)
        tilt = RING_VARY[0] * rng.uniform(0.3, 1.0) * (1 - 0.7 * self.at)
        axis = math.cos(tilt) * axis + math.sin(tilt) * (math.cos(lean) * a1 + math.sin(lean) * np.cross(axis, a1))
        a1 = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0])
        a1 /= np.linalg.norm(a1)
        self.axis = axis
        self.spokes = np.stack([a1, np.cross(axis, a1)])  # the ring's plane
        vary = rng.uniform(-1, 1, 3)
        self.ring0 = (RING_R[0] * (1 + LENS_WIDE * self.at) * size if ring is None else float(ring)) * (
            1 + RING_VARY[1] * vary[0])
        self.core0 = RING_CORE * self.ring0 * at_lens(LENS_CORE)
        self.roll = RING_ROLL * (1 + RING_VARY[2] * vary[1])
        self.speed = RING_SPEED * at_lens(LENS_RUSH) * (1 + RING_VARY[3] * vary[2])
        place = rng.uniform(0, 2 * np.pi)
        self.ring_o = RING_VARY[4] * size * (math.cos(place) * e1 + math.sin(place) * e2) + lift  # px, off the
        # origin
        self.c_th = rng.uniform(0, 2 * np.pi, C)  # where round the ring
        self.c_ph = rng.uniform(0, 2 * np.pi, C)  # where round its core
        self.c_r = np.sqrt(rng.uniform(0, 1, C))  # how far out in its core, in core radii (a disc's even spread)
        self.c_wake = rng.random(C) < RING_WAKE * (1 - self.at)
        self.c_lag = np.where(self.c_wake, rng.uniform(0.05, 0.75, C), 1.0)  # share of the ring's travel made
        lumps = _tile_noise(rng, 48, 160, 3.0, 3.0)  # round the core x round the ring, fixed in the rolling air
        eddies = _tile_noise(fx._rng(seed, _BLAST, salt, _SWIRL), 48, 160, RING_SWIRL[1], RING_SWIRL[1])
        where = ((self.c_th * (160 / (2 * np.pi))).astype(np.float32)[None],
                 (self.c_ph * (48 / (2 * np.pi))).astype(np.float32)[None])

        def held(tile: np.ndarray, rolls: bool = True) -> np.ndarray:  # a tile's value at each parcel (or at its
            # place round the ring alone)
            return cv2.remap(tile, where[0], where[1] if rolls else np.zeros_like(where[1]), cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_WRAP)[0]

        share = np.exp(RING_LUMPS * held(lumps) - 0.5 * RING_LUMPS ** 2) * np.maximum(
            1 + RING_SWIRL[0] * held(eddies), 0.2)  # (mean 1)
        share *= 1 + RING_NEAR * (np.cos(self.c_th) * self.spokes[0, 2] + np.sin(self.c_th) * self.spokes[1, 2])
        self.puff = np.where(self.c_wake, 0.0, _smooth(
            (0.5 + 0.3 * held(_wrap_blur(lumps, HAND_PUFF[2]), False) - HAND_PUFF[0]) / (HAND_PUFF[1] - HAND_PUFF[0])))
        # (how much of each parcel shows over the hand when its ring does not cross it whole, HAND_PUFF: by where it is
        # round the ring alone, as the band seen there holds the air from all round its core)
        self.p = np.concatenate([p, self.ring(0.0)])
        self.v = np.concatenate([v, np.zeros((C, 3))])
        self.delay = np.concatenate([delay, np.zeros(C, int)])
        self.r0 = np.concatenate([r0, np.ones(C)])
        self.bright = np.concatenate([bright, share * rng.uniform(0.6, 1.4, C) * gain * at_lens(LENS_THICK) / C])  # (a
        # parcel's bright is its share of the dusty air)
        self.dark = np.concatenate([dark, np.zeros(C, bool)])
        self.flake = np.concatenate([flake, np.zeros(C, bool)])
        self.cloud = np.arange(N + C) >= N
        self.tum = rng.uniform(0, 2 * np.pi, N + C)
        self.tumf = rng.uniform(0.25, 0.8, N + C)
        self.alive = np.ones(N + C, bool)
        self.room_hit: np.ndarray | None = None  # which of the room's dust this push has thrown (_Room.throw)

    def ring(self, kf: float) -> np.ndarray:
        """The dusty air's parcels on the push's ``kf``-th frame, as self.p holds them: on a torus round the ring's
        axis that travels out along it, widens and slows, its air rolling round its core; the wake's parcels lag back
        towards the palm, narrower and looser; the room's eddies push it about as it slows. Coming at the lens it
        swells as it nears it and passes it (kept to a share of its travel there, it sat on a palm turned to the lens
        and fogged it), drifting to the frame's middle as it comes (LENS_MID)."""
        go = 1 - math.exp(-kf / RING_SLOW)
        travel = self.speed * self.side * self.gain * RING_SLOW * go
        radius = self.ring0 * (1 + (RING_R[1] - 1) * go)
        core = self.core0 * (1 + 0.6 * go)
        ph = self.c_ph + self.roll * RING_SLOW * go * np.where(self.c_wake, 0.3, 1.0)
        rc = self.c_r * core * np.where(self.c_wake, 1.5, 1.0)
        rad = np.where(self.c_wake, radius * (0.25 + 0.5 * self.c_lag), radius)  # (a stem narrowing back to the palm;
        # as wide as the ring, it filled the ring's hole and the ring read as a cauliflower)
        zc = 1 - (self.ring_o[2] + travel * self.axis[2]) / self.f  # its middle's depth
        drift = 0.0 if self.lens <= 0 else min(1.0, 2 * self.at) * self.lens * float(
            _smooth((self.z0 - zc) / max(self.z0 - LENS_MID, 1e-3)))  # a true push at the lens: how far it is drawn
        # to the frame's middle and turned to face the lens as it nears it (LENS_MID)
        s0, s1, ax = _turned(np.stack([self.spokes[0], self.spokes[1], self.axis]), self.axis, drift)
        spoke = np.cos(self.c_th)[:, None] * s0 + np.sin(self.c_th)[:, None] * s1
        off = (rad + rc * np.cos(ph))[:, None] * spoke + (rc * np.sin(ph))[:, None] * ax \
            + (travel * self.c_lag)[:, None] * self.axis + self.ring_o
        x, y = self.o[0] + off[:, 0], self.o[1] + off[:, 1]
        if drift > 0:
            seen = self.pp + (self.o + self.ring_o[:2] + travel * self.axis[:2] - self.pp) / max(zc, 0.07)
            pull = (self.pp - seen) * zc * drift
            x, y = x + pull[0], y + pull[1]
        cx, cy = self.curl(x, y, kf)
        sway = RING_BILLOW * self.ring0 * go * go
        return np.column_stack([x + cx * sway, y + cy * sway, 1 - off[:, 2] / self.f])

    def cone(self, ca: np.ndarray) -> np.ndarray:
        """How strong the push is at directions whose cosine with its aim is ``ca``: all round when it comes at the
        lens, ahead of the palm when it crosses the picture."""
        return (1 - self.q) + self.q * np.clip(ca, 0, 1) ** 1.5

    def front(self, kf: float) -> tuple[float, float, float]:
        """The front on its ``kf``-th frame: (radius px, bulge px, share of its run). It runs out at FRONT_SPEED,
        slowing a fifth by the end of its run, and fades as it goes."""
        uu = min(1.0, kf / BLAST_FRAMES)
        radius = self.side * FRONT_FROM + self.run * kf * (1 - 0.1 * uu)
        strength = (BUBBLE_PX * self.k * self.gain * min(1.0, (kf + 1) / 2)
                    * (BUBBLE_END + (1 - BUBBLE_END) * (1 - uu) ** 1.5) * min(1.0, 3 * (1 - uu)))
        return radius, strength, uu

    def knock(self) -> int:
        """Frames after the push it knocks the phone: KNOCK_AT's first for a push across the picture, its second for
        one straight at the lens, between them as it leans."""
        return round(KNOCK_AT[0] + (KNOCK_AT[1] - KNOCK_AT[0]) * (1 - self.q))

    def last_frame(self) -> int:
        """The last frame anything it throws shows on (the room's dust its front reaches as it ends included)."""
        return self.start + BLAST_FRAMES + DUST_LIFE[0] + DUST_LIFE[1]

    def behind(self, p: np.ndarray) -> np.ndarray:
        """Which points are behind the plane of the palm (so behind the hand, where its matte is)."""
        return (p[:, 0] - self.palm[0]) * self.nrm[0] + (p[:, 1] - self.palm[1]) * self.nrm[1] - (
            p[:, 2] - 1.0) * self.f * self.nrm[2] < 0

    def bend(self, P: np.ndarray, kf: int, o: np.ndarray, u: np.ndarray, zoom: float) -> None:
        """The pressure bubble on frame ``kf`` of the push (``o``, ``u``, ``zoom``: its origin, aim and scale as the
        camera sees them now): what is seen through it pushed outward, most in a thin band of bend at its front
        sweeping the room, which is a shadowgraph, and behind it a turbulent wake, in place."""
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
        width = max((FRONT_W[0] + (FRONT_W[1] - FRONT_W[0]) * uu) * self.k * zoom / 2.355,
                    0.61 * strength / FRONT_SLOPE)  # (a Gaussian's sigma: its band at half height is 2.355 of it, and
        # its steepest slope 0.61 of its strength over it)
        zz = (dist - rr) / width
        bulge = strength * cone * (BUBBLE_INNER * np.clip(x, 0, 1) ** 3 * np.exp(-0.5 * zz * zz * (zz > 0))
                                   + np.exp(-0.5 * zz * zz))  # (inside it barely moves; a band one front wide sweeps
        # across; outside it all falls off within one front's width)
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
        sg = (1 - zz * zz) * np.exp(0.5 * (1 - zz * zz))  # bright on the front, dark either side of it ...
        sg = np.where((zz < 0) & (sg < 0), SHADOWGRAPH[1] * sg, sg) * np.clip(0.8 + 0.4 * arc, 0.3, 1.5)  # ... less
        # behind it, stronger and weaker round it
        rel = strength / (BUBBLE_PX * self.k * zoom * self.gain + 1e-6)
        P[y0:y1, x0:x1] *= 1 + _upsample((SHADOWGRAPH[0] * self.gain * rel * cone * sg).astype(np.float32), bh, bw)[
            ..., None]

    def seen(self, p: np.ndarray, xy: np.ndarray, hand: np.ndarray | None, front=1.0) -> np.ndarray:
        """How much of each point shows (``p`` as self.p holds them, drawn at ``xy`` frame px): behind the palm's
        plane, only off the hand's matte ``hand`` (0..1, or None); in front of it, ``front`` of it over the hand (a
        share, or one a point)."""
        if hand is None:
            return np.ones(len(p))
        hh, ww = hand.shape
        xi = np.clip(xy[:, 0].astype(np.int64), 0, ww - 1)
        yi = np.clip(xy[:, 1].astype(np.int64), 0, hh - 1)
        return 1 - hand[yi, xi] * np.where(self.behind(p), 1.0, 1.0 - front)

    def dust(self, kf: int, H: np.ndarray, H_prev: np.ndarray, lit: list, drk: list, scale: float, haze: np.ndarray,
             fore: np.ndarray, puffs: np.ndarray, hand: np.ndarray | None) -> tuple[float, float, float]:
        """Moves the dust and the dusty air on a frame (``kf``: frames since the push; ``H``: the start frame's px ->
        this frame's, ``H_prev`` the frame before's) and splats them: the dust into ``lit`` and ``drk`` (_specks), the
        dusty air into ``haze`` (CLOUD_Q times smaller), and again into ``fore`` as much as it is well in front of the
        palm (CLOUD_FORE) and into ``puffs`` as much as it shows over the hand in puffs (HAND_PUFF). What is behind the
        palm is hidden by the hand's matte ``hand`` (0..1, or None); the dusty air in front of it shows over the hand
        once it has left the palm (CLOUD_OVER). Returns (the ring's depth, palm depths from the lens, its parcels'
        median; how much of the dusty air still shows, as it comes in and thins out (CLOUD_IN, CLOUD_FADE) and as it
        swells past the frame (LENS_GONE), which the caller lays on as its opacity: thinned in its density, every lump
        shrank as it went; how far its broad mean is dropped as it swells, LUMPS_WIDE)."""
        side, k = self.side, self.k
        before = _project(self.p, self.pp)
        go = (kf >= self.delay) & ~self.cloud
        self.v[go] *= math.exp(-1 / 5.0)  # drag: out fast, slowing
        cx, cy = self.curl(self.p[go, 0], self.p[go, 1], kf)
        turb = DUST_STIR * side * self.gain * float(_eddies(kf))
        sp = np.hypot(self.v[go, 0], self.v[go, 1])
        self.v[go, 0] += cx * turb
        self.v[go, 1] += cy * turb + 0.0008 * side * (sp < 0.015 * side)  # the heavier grit settles
        self.p[go] += self.v[go]
        self.p[self.cloud] = self.ring(kf)
        self.alive &= self.p[:, 2] > 0.07
        after = _project(self.p, self.pp)
        before, after = _cam(H_prev, before), _cam(H, after)
        zz = np.clip(self.p[:, 2], 0.07, None)
        a0 = after + (before - after) * 0.5  # the shutter: half a frame of motion
        disp = np.hypot(*(after - a0).T)
        over = max(float(_smooth((kf - CLOUD_OVER[0]) / CLOUD_OVER[1])), self.off_palm)
        seen = self.seen(self.p, after, hand, np.where(self.cloud, over, 1.0))
        # the dusty air, smeared along its half frame of motion
        cl = self.cloud & self.alive
        ring = cl & ~np.r_[np.zeros(len(self.p) - CLOUD_N, bool), self.c_wake]
        depth = float(np.median(zz[ring])) if ring.any() else 1.0
        life_c = float(np.clip(1 - (kf - CLOUD_FADE[0]) / CLOUD_FADE[1], 0, 1))
        if self.cloud_in > 0:
            life_c *= float(_smooth(kf / self.cloud_in))
        broad = 0.0
        if ring.any():  # swelling past the frame, it is gone
            c_ = np.median(after[ring], 0)
            wide = float(np.median(np.hypot(*(after[ring] - c_).T))) / (0.5 * math.hypot(self.w, self.h))
            life_c *= 1 - float(_smooth(math.log(max(wide / LENS_GONE[0], 1.0)) / math.log(LENS_GONE[1])))
            broad = float(_smooth((wide - LUMPS_WIDE[0]) / LUMPS_WIDE[1]))
        if cl.any() and life_c > 0:
            S = int(np.clip(np.ceil(disp[cl].max() / CLOUD_Q), 2, 24))  # a sample a coarse px (six whatever the
            # speed drew a ring coming fast at the lens as parallel streaks)
            ss = np.linspace(0, 1, S)[None, :]
            X = (a0[cl, 0:1] + (after[cl, 0:1] - a0[cl, 0:1]) * ss + 0.5) / CLOUD_Q - 0.5
            Y = (a0[cl, 1:2] + (after[cl, 1:2] - a0[cl, 1:2]) * ss + 0.5) / CLOUD_Q - 0.5
            wv = self.bright[cl] * seen[cl] / np.clip(zz[cl], 0.15, None) ** 2 * _smooth(
                (zz[cl] - LENS_PASS[0]) / (LENS_PASS[1] - LENS_PASS[0]))  # (nearer the lens it spreads over more of
            # the picture and is drawn softer with it, as thick as it was: thinned with its spread, a ring coming at
            # the lens faded as it came)
            wake = float(_smooth((1 - math.exp(-kf / RING_SLOW)) / 0.3)) * (1 - self.at)  # the wake trails a ring
            wv = wv * np.where(self.c_wake[cl[self.cloud]], wake, 1.0)  # that has gone somewhere (there from the
            # start, it lay in the palm as a grey disc), and hardly any at the lens (RING_WAKE)
            _splat(haze, X.ravel(), Y.ravel(), np.repeat(wv / S, S))
            ahead = _smooth((CLOUD_FORE[0] - zz[cl]) / (CLOUD_FORE[0] - CLOUD_FORE[1]))
            _splat(fore, X.ravel(), Y.ravel(), np.repeat(wv * ahead / S, S))
            if self.whole < 1:
                _splat(puffs, X.ravel(), Y.ravel(), np.repeat(wv * self.puff[cl[self.cloud]] / S, S))
        life = float(np.clip(1 - (kf - DUST_LIFE[0]) / DUST_LIFE[1], 0, 1) * _smooth(kf / 1.5))
        sel = self.alive & go
        if life <= 0 or not sel.any():
            return depth, life_c, broad
        bright = self.bright * np.clip(disp / (3 * k), 0.4, 1.0)  # dust shows most as it moves
        bright = np.where(self.flake, 2.5 * bright * (0.2 + 0.8 * np.abs(np.cos(self.tum + self.tumf * kf)) ** 3),
                          bright)
        _specks(a0[sel], after[sel], disp[sel], zz[sel], (bright * life * seen)[sel], self.r0[sel], self.dark[sel], lit,
                drk, scale, k)
        return depth, life_c, broad


def _turned(rows: np.ndarray, a: np.ndarray, share: float) -> np.ndarray:
    """Vectors ``rows`` (n, 3) turned with the unit vector ``a`` the ``share`` of the way from it to the lens's axis
    (0, 0, 1)."""
    k = np.cross(a, (0.0, 0.0, 1.0))
    s = float(np.linalg.norm(k))
    if share <= 0 or s < 1e-9:
        return rows
    k /= s
    th = share * math.atan2(s, float(a[2]))
    c, sn = math.cos(th), math.sin(th)
    return rows * c + np.cross(k, rows) * sn + np.outer(rows @ k, k) * (1 - c)


def _project(p: np.ndarray, pp: np.ndarray) -> np.ndarray:
    """Points (n, 3: x, y in px at the palm's depth, z in palm depths) as the camera sees them (principal point
    ``pp``)."""
    return pp + (p[:, :2] - pp) / np.clip(p[:, 2:3], 0.07, None)


def _cam(M: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Frame px ``xy`` (n, 2) carried by the camera's 3x3 ``M`` (cuesync.bending.camera_between)."""
    return xy @ M[:2, :2].T + M[:2, 2]


def _eddies(age):
    """How hard the room's eddies stir dust ``age`` frames after it was thrown: they take it as it slows, then calm."""
    return (1 - np.exp(-np.asarray(age, np.float64) / 4.0)) * np.exp(-np.asarray(age, np.float64) / DUST_CALM)


DUST_SIGMA = (0.8, 1.5, 4.0, 9.0)  # px at 1080: how soft each class of dust is drawn (sharp ... out of focus) ...
DUST_CLASS = (2.2, 5.0, 11.0)  # ... a speck this many px across or more drawn in the next class (a 2-4 px speck drawn
# 6.6 px soft was a faint smudge, and only the 1-2 px ones read, as white dots)


def _specks(a0: np.ndarray, after: np.ndarray, disp: np.ndarray, zz: np.ndarray, weight: np.ndarray, r0: np.ndarray,
            dark: np.ndarray, lit: list, drk: list, scale: float, k: float) -> None:
    """Splats specks of dust into ``lit`` and ``drk`` (grit darker than the wall: ``dark``), one buffer per softness
    (DUST_SIGMA; the softer two ``scale`` times smaller than the frame): each smeared from ``a0`` to ``after`` (frame
    px: the shutter's half frame), or round when it moves less than SMEAR_MIN, ``zz`` palm depths from the lens,
    ``r0`` px across at the palm's depth (at 1080), showing ``weight`` of itself."""
    rad = r0 * k / zz  # bigger near the lens ...
    soft = np.maximum(rad, 0.8 * k) + 10 * k * np.clip((0.4 - zz) / 0.33, 0, 1) ** 1.5  # ... and out of focus
    tip = np.where((disp < SMEAR_MIN * k)[:, None], a0, after)
    S = 10
    ss = np.linspace(0, 1, S)[None, :]
    for j, (lo, hi) in enumerate(zip((0.0, *DUST_CLASS), (*DUST_CLASS, 1e9), strict=True)):
        sel = (soft >= lo * k) & (soft < hi * k) & (weight > 0)
        if not sel.any():
            continue
        q = 1.0 if j < 2 else scale  # (a pixel's middle lands on the middle of the coarse pixel holding it)
        X = (a0[sel, 0:1] + (tip[sel, 0:1] - a0[sel, 0:1]) * ss + 0.5) / q - 0.5
        Y = (a0[sel, 1:2] + (tip[sel, 1:2] - a0[sel, 1:2]) * ss + 0.5) / q - 0.5
        L = np.maximum(disp[sel], 1.0)
        # each grain's light spread over its smear: a fast one is a fainter streak, never a brighter one
        wv = weight[sel] * np.minimum(1.0, (rad[sel] / k) ** 2) * np.minimum(1.0, 2.0 / L) ** 0.5 / (q * q)
        Wt = np.repeat((wv / S)[:, None], S, 1)
        dk = np.repeat(dark[sel][:, None], S, 1)
        _splat(lit[j], X[~dk], Y[~dk], Wt[~dk])
        _splat(drk[j], X[dk], Y[dk], Wt[dk])


def _speck_bufs(h: int, w: int, scale: int) -> tuple[list, list]:
    """Empty buffers for _specks over an (h, w) picture: (lit, dark), each one per class of DUST_SIGMA, the softer two
    ``scale`` times smaller."""
    lit = [np.zeros((h, w), np.float32) for _ in range(2)] + [
        np.zeros((-(-h // scale), -(-w // scale)), np.float32) for _ in range(2)]
    return lit, [np.zeros_like(b) for b in lit]


def _lay_specks(P: np.ndarray, lit: list, drk: list, scale: int, tone: np.ndarray, grain, code: float, k: float,
                behind: np.ndarray | None = None) -> None:
    """In place: the specks splatted into ``lit`` and ``drk`` (_specks) laid over ``P`` (linear, (h, w, 3)), each
    class blurred to its softness, covering it DUST_ALPHA at the most, in the colour of what is behind them (``behind``:
    _behind of ``P``, worked out here when None) lit a little towards the room's lit ``tone``, the grit darker
    (DUST_TONE), the lit ones with the plate's grain (``grain(y0, y1, x0, x1)``: unit grain for that box, or None)."""
    h, w = P.shape[:2]
    alpha, dark = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    for j, sig_ in enumerate(DUST_SIGMA):
        q = 1 if j < 2 else scale
        for buf, into in ((lit[j], alpha), (drk[j], dark)):
            if buf.any():
                gain_ = 12.0 * (0.7 / sig_) ** 0.6  # an out-of-focus grain is bigger and fainter
                b_ = cv2.GaussianBlur(buf, (0, 0), max(0.5, sig_ * k / q)) * np.float32(gain_)
                into += b_ if q == 1 else cv2.resize(b_, (w, h), interpolation=cv2.INTER_LINEAR)
    for a, toward, lift, gr in ((alpha, DUST_TONE[0], DUST_TONE[1], grain), (dark, 0.0, DUST_TONE[2], None)):
        box = _bbox(a > 2e-3)
        if box is None:
            continue
        if behind is None:
            behind = _behind(P)
        y0, y1, x0, x1 = box
        _over(P[y0:y1, x0:x1], np.clip(a[y0:y1, x0:x1], 0, DUST_ALPHA), _lit_by(_up(behind, box), tone, toward, lift),
              None if gr is None else gr(y0, y1, x0, x1), code)


class _Room:
    """The room's own dust, one pool for all of a shot's pushes: still and unseen until a front reaches it, then
    thrown outward from the palm, slowing, swirling and fading out as it settles, and thrown again by the next push
    (each push brought its own and it built up: the dust of one still hung through the next). Most of it is near the
    first push and ahead of it, thinning out with distance (spread evenly over the frame, it read as sensor snow). It
    lives in the room as the first push's start frame saw it."""

    def __init__(self, first: _Blast, seed: int):
        rr = fx._rng(seed, _ROOM)
        w, h, side = first.w, first.h, first.side
        m = 4 * ROOM_N  # where it may be: anywhere in front of the wall
        zr = rr.uniform(0.45, 2.5, m)
        scr = np.column_stack([rr.uniform(-0.1, 1.1, m) * w, rr.uniform(-0.1, 1.1, m) * h])
        r0, bright, dark = rr.uniform(*DUST_R, m), rr.uniform(0.3, 1.0, m) ** 1.5, rr.random(m) < 0.1
        rel = scr - first.o
        dd = np.hypot(rel[:, 0], rel[:, 1]) + 1e-6
        near = first.cone((rel @ first.u) / dd) * np.exp(-dd / (ROOM_NEAR * side))
        keep = np.sort(rr.choice(m, ROOM_N, replace=False, p=near / near.sum()))
        self.start, self.side, self.k, self.f, self.pp = first.start, side, first.k, first.f, first.pp
        self.p = np.column_stack([first.pp + (scr[keep] - first.pp) * zr[keep, None], zr[keep]])
        self.v = np.zeros((ROOM_N, 3))
        self.r0, self.bright, self.dark = r0[keep], bright[keep], dark[keep]
        self.kick = np.full(ROOM_N, -1)  # the frame each speck was last thrown on (-1: not yet)
        self.gain = np.zeros(ROOM_N)  # ... and how hard that push was
        self.curl = _Curl(rr, 0.1 * side)

    def throw(self, b: _Blast, kf: int, t: int, Hb: np.ndarray) -> None:
        """Throws the dust that push ``b``'s front reaches on its ``kf``-th frame (frame ``t``; ``Hb``: the room's px
        -> ``b``'s start frame's) outward from its palm, harder ahead of the push and nearer it."""
        if b.room_hit is None:
            b.room_hit = np.zeros(ROOM_N, bool)
        radius, _, uu = b.front(kf)
        rel = _cam(Hb, _project(self.p, self.pp)) - b.o
        dd = np.hypot(rel[:, 0], rel[:, 1]) + 1e-6
        hit = ~b.room_hit & (dd <= radius)
        if not hit.any():
            return
        o = np.linalg.solve(Hb[:2, :2], b.o - Hb[:2, 2])  # its palm in the room's px
        D = np.column_stack([self.p[hit, 0] - o[0], self.p[hit, 1] - o[1], (self.p[hit, 2] - 1.0) * self.f])
        D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-6)
        sp = ROOM_PUSH * self.side * b.gain * b.cone((rel[hit] @ b.u) / dd[hit]) * (0.35 + 0.65 * (1 - uu)) * \
            np.exp(-dd[hit] / (0.9 * self.side))
        self.v[hit] = np.column_stack([D[:, 0] * sp, D[:, 1] * sp, D[:, 2] * sp / self.f])
        self.kick[hit], self.gain[hit] = t, b.gain
        b.room_hit |= hit

    def dust(self, t: int, H: np.ndarray, H_prev: np.ndarray, lit: list, drk: list, scale: float,
             hand: np.ndarray | None, b: _Blast, Hb: np.ndarray) -> None:
        """Moves the thrown dust on frame ``t`` and splats it into ``lit`` and ``drk`` (_specks; ``H``: the room's px
        -> this frame's, ``H_prev`` the frame before's). What is behind the palm of the latest push ``b`` (``Hb``: the
        room's px -> its start frame's) is hidden by the hand's matte ``hand``."""
        thrown = self.kick >= 0
        if not thrown.any():
            return
        age = t - self.kick
        before = _project(self.p, self.pp)
        self.v[thrown] *= math.exp(-1 / 5.0)
        cx, cy = self.curl(self.p[thrown, 0], self.p[thrown, 1], t)
        turb = 0.5 * DUST_STIR * self.side * self.gain[thrown] * _eddies(age[thrown])  # (the room's air is calmer)
        self.v[thrown, 0] += cx * turb
        self.v[thrown, 1] += cy * turb
        self.p[thrown] += self.v[thrown]
        after = _project(self.p, self.pp)
        before, after = _cam(H_prev, before), _cam(H, after)
        zz = np.clip(self.p[:, 2], 0.07, None)
        a0 = after + (before - after) * 0.5
        disp = np.hypot(*(after - a0).T)
        seen = b.seen(np.column_stack([_cam(Hb, self.p[:, :2]), self.p[:, 2]]), after, hand)
        life = np.clip(1 - (age - DUST_LIFE[0]) / DUST_LIFE[1], 0, 1) * _smooth(age / 1.5)
        fade = np.clip(1 - age / (DUST_LIFE[0] + DUST_LIFE[1]), 0, 1)
        weight = self.bright * np.clip(ROOM_SEEN * fade + disp / (2.5 * self.k), 0, 1) * life * seen * thrown
        sel = weight > 0
        if sel.any():
            _specks(a0[sel], after[sel], disp[sel], zz[sel], weight[sel], self.r0[sel], self.dark[sel], lit, drk,
                    scale, self.k)


def _light_dir(frame_u8: np.ndarray, hand: np.ndarray | None, origin) -> np.ndarray:
    """Unit direction in the picture from ``origin`` towards the room's light: the middle of the brightest tenth of
    the room off the hand's matte ``hand`` (up when that is too near to tell)."""
    h, w = frame_u8.shape[:2]
    q = 8
    small = cv2.resize(frame_u8, (max(1, w // q), max(1, h // q)), interpolation=cv2.INTER_AREA).astype(np.float32)
    y = small @ fx._LUMA
    room = np.ones(y.shape, bool) if hand is None else \
        cv2.resize(hand, y.shape[::-1], interpolation=cv2.INTER_AREA) < 0.1
    up = np.array([0.0, -1.0])
    if room.sum() < 16:
        return up
    ys, xs = np.nonzero(room & (y >= np.percentile(y[room], 90)))
    d = np.array([(xs.mean() + 0.5) * w / y.shape[1], (ys.mean() + 0.5) * h / y.shape[0]]) - np.asarray(origin)
    nd = float(np.hypot(*d))
    return d / nd if nd > 0.1 * min(h, w) else up


@dataclass
class _Haze:
    """A frame's dusty air on a grid CLOUD_Q times coarser than the frame, each push's added in: its density, and the
    same weighted by how much of each push's still shows (its opacity), by how much of it is well in front of the palm
    (CLOUD_FORE), by how far its broad mean is dropped (LUMPS_WIDE), by how whole its ring crosses the hand, by whether
    it is a ball's burst (BURST_HAND) and by how much of it shows over the hand in puffs (HAND_PUFF)."""

    dens: np.ndarray
    shown: np.ndarray
    front: np.ndarray
    broad: np.ndarray
    whole: np.ndarray
    burst: np.ndarray
    puff: np.ndarray

    @classmethod
    def empty(cls, shape: tuple[int, int]) -> _Haze:
        return cls(*(np.zeros(shape, np.float32) for _ in range(7)))


def _cloud_bend(P: np.ndarray, haze: _Haze, lump: float, hand: np.ndarray | None) -> None:
    """In place: what is seen through the dusty air ``haze`` bent towards where it is thicker, RING_BEND's first at the
    steepest, its opacity's curve (CLOUD_HALF, as much of it as still shows) smoothed over RING_BEND's second or its
    lumps (``lump``: coarse px), whichever is wider, so the bend never folds; over the hand's matte ``hand`` cut as the
    ball's is (BALL_HAND)."""
    h, w = P.shape[:2]
    k = min(h, w) / 1080.0
    body = haze.dens ** CLOUD_HALF[1]
    a = body / (body + CLOUD_HALF[0] ** CLOUD_HALF[1]) * np.clip(haze.shown / np.maximum(haze.dens, 1e-6), 0, 1)
    sig = max(RING_BEND[1] * k / CLOUD_Q, lump)
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), sig)
    ddy, ddx = np.gradient(a)
    pull = np.float32(RING_BEND[0] * k * sig * math.sqrt(math.e))  # (a bump of opacity 1 as wide as the smoothing
    # bends what is behind it RING_BEND's first at its steepest)
    if hand is not None:
        pull = pull * (1 - BALL_HAND[0] * cv2.resize(hand, a.shape[::-1], interpolation=cv2.INTER_AREA))
    dx, dy = pull * ddx, pull * ddy
    box = _bbox(cv2.resize((np.hypot(dx, dy) > 0.25).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
    if box is None:
        return
    y0, y1, x0, x1 = box
    full = cv2.resize(np.dstack([dx, dy]).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]
    _remap_rgb(P, np.ascontiguousarray(full[..., 0]), np.ascontiguousarray(full[..., 1]), box, BLAST_CA)


def _lay_cloud(P: np.ndarray, haze: _Haze, tone: np.ndarray, light: np.ndarray, lump: float, hand: np.ndarray | None,
               grain: tuple, code: float) -> None:
    """In place: the dusty air ``haze`` (``lump``: its lumps' size in coarse px, its core as blurred) laid over ``P``:
    a ring swelling past the frame only its lumps and body (CLOUD_LUMPS, CLOUD_KEEP), in the colour of what it covers,
    lit a little by the room (CLOUD_LIFT) with a cheap self-shadow: brighter on the side its lumps face towards the
    room's light (``light``: its direction in the picture), less so on the far side, as the air CLOUD_SHADE lump sizes
    towards the light is thinner or thicker than its own, never darker than what it covers, its hue a little towards
    the room's lit ``tone`` (CLOUD_TONE); over the hand's matte ``hand`` a ring that does not cross it whole only in
    soft puffs (HAND_PUFF, HAND_NEAR), one that does its lumps near the palm's plane and its body too well in front of
    it, softened for a ball's burst (BURST_HAND), covering it CLOUD_ON_HAND at the most; and it blurs what is behind it
    (SCATTER). ``grain``: (seed, frame) of the plate's grain laid in it."""
    h, w = P.shape[:2]
    k = min(h, w) / 1080.0
    step = CLOUD_SHADE[0] * lump
    dens = haze.dens
    inv = 1 / np.maximum(dens, 1e-6)
    shows = np.clip(haze.shown * inv, 0, 1)

    def opacity(body):  # of what is left of the dusty air once its broad mean is dropped
        body = body ** CLOUD_HALF[1]
        return CLOUD_ALPHA * shows * body / (body + CLOUD_HALF[0] ** CLOUD_HALF[1])

    body = dens
    if haze.broad.any():
        sig = max(CLOUD_LUMPS[0] * k / CLOUD_Q, CLOUD_LUMPS[2] * lump)
        near, mean = cv2.GaussianBlur(dens, (0, 0), 2 * lump), _broad_blur(dens, sig)
        keep = CLOUD_KEEP * np.minimum(dens, near) * _smooth((near - mean) / np.maximum(mean, 1e-6))  # (in a band:
        # where the air over two lumps is thicker than its broad mean, not in an even fog)
        body = np.maximum(dens - _broad_blur(haze.broad, sig), keep)
    a_room = opacity(body)
    a_hand = hc = None
    if hand is not None:  # over the hand: a ring crossing it whole its lumps near the palm's plane, well in front of it
        # its body too (a ball's burst softened); any other ring only in soft puffs
        hc = cv2.resize(hand, dens.shape[::-1], interpolation=cv2.INTER_AREA)
        ahead = np.clip(haze.front * inv, 0, 1)
        whole = np.clip(haze.whole * inv, 0, 1)
        lumps = np.maximum(dens - _broad_blur(dens, CLOUD_LUMPS[1] * k / CLOUD_Q), 0)
        a_hand = opacity(lumps)
        a_hand = np.minimum(a_hand + (a_room - a_hand) * ahead,
                            CLOUD_ON_HAND[0] + (CLOUD_ON_HAND[1] - CLOUD_ON_HAND[0]) * ahead)
        if haze.burst.any():
            a_hand = a_hand + (cv2.GaussianBlur(a_hand, (0, 0), BURST_HAND * k / CLOUD_Q) - a_hand) * np.clip(
                haze.burst * inv, 0, 1)
        a_hand = a_hand * whole
        if haze.puff.any():
            a_hand = a_hand + (HAND_NEAR[0] + (HAND_NEAR[1] - HAND_NEAR[0]) * ahead) * np.float32(1 / CLOUD_ALPHA) * (
                1 - whole) * opacity(cv2.GaussianBlur(haze.puff, (0, 0), HAND_NEAR[2] * k / CLOUD_Q))
    a_c = a_room if hc is None else a_room + (a_hand - a_room) * hc  # (coarse, the hand's share in)
    box = _bbox(cv2.resize(a_c.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 2e-3)
    if box is None:
        return
    top = max(float(np.percentile(dens, 99.5)), 0.25 * float(dens.max()), 1e-6)  # its thickest (a hot spot or two
    # set as the scale flattened the rest)
    toward = cv2.warpAffine(dens, np.float32([[1, 0, light[0] * step], [0, 1, light[1] * step]]), dens.shape[::-1],
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)  # (past
    # the frame's edge, the air at the edge: taken as clear, a ring swelling past the frame lit a block along the edge
    # white, its inner side a straight line a lump in from it)
    lit = (dens - toward) / (0.5 * (dens + toward) + 0.1 * top)  # (the air towards the light against its own: each
    # lump shaded alike, however thick)
    shade = CLOUD_LIFT[0] + CLOUD_LIFT[1] * np.clip(lit, -CLOUD_SHADE[1], 1)
    behind = cv2.GaussianBlur(cv2.resize(P, dens.shape[::-1], interpolation=cv2.INTER_AREA), (0, 0), 2.0)  # what it
    # covers
    bl = np.maximum(behind @ fx._LUMA, 1e-4)
    tone_ = tone.astype(np.float32) / max(float(tone @ fx._LUMA), 1e-4)  # (the lit tone's hue, at luminance 1)
    col = (behind + CLOUD_TONE * (tone_ * bl[..., None] - behind)) * np.maximum(shade, 1.0)[..., None]
    spread = cv2.GaussianBlur(a_c, (0, 0), SCATTER[3] * SCATTER[0] * k / CLOUD_Q) * np.float32(1 / CLOUD_ALPHA) * (
        SCATTER[1] if hc is None else SCATTER[1] + (SCATTER[2] - SCATTER[1]) * hc)
    maps = [col, spread[..., None], a_room[..., None]] + ([] if hc is None else [a_hand[..., None]])
    full = cv2.resize(np.concatenate(maps, -1).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    y0, y1, x0, x1 = box
    full = full[y0:y1, x0:x1]
    col, blur, a = full[..., :3], full[..., 3], full[..., 4]
    if hc is not None:
        a = a + (full[..., 5] - a) * hand[y0:y1, x0:x1]
    Q = P[y0:y1, x0:x1]
    sb = _bbox(blur > 0.02)
    if sb is not None:  # what is seen through it, scattered
        m = int(3 * SCATTER[0] * k) + 1
        sy0, sy1, sx0, sx1 = max(0, sb[0] - m), min(y1 - y0, sb[1] + m), max(0, sb[2] - m), min(x1 - x0, sb[3] + m)
        R = Q[sy0:sy1, sx0:sx1]
        R += (cv2.GaussianBlur(R, (0, 0), SCATTER[0] * k) - R) * np.minimum(blur[sy0:sy1, sx0:sx1], 1)[..., None]
    _over(Q, a, np.minimum(col, 0.95), _grain(grain[0], _CLOUD, grain[1], y1 - y0, x1 - x0), code)


def _push_frames(out: np.ndarray, blasts: list[_Blast], mask: np.ndarray | None, tl: Timeline, seed: int,
                 code: float, say) -> None:
    """Draws pushes of air (``blasts``, in the order they start) over ``out`` (uint8 frames) in place: on each frame
    each push's bubble bends what is behind it, its ring of dusty air bends it too and is laid on as a faint haze lit
    by the room's light, the dust it throws and the room's dust (_Room) are drawn through the camera's motion since;
    then each push knocks the phone, and those that come at the lens swell the picture (buffet)."""
    from .bending import camera_between

    n, h, w = out.shape[:3]
    cam = tl.camera
    k = min(h, w) / 1080.0
    sc = 4  # the softer dust is drawn this many times smaller
    hq = (-(-h // CLOUD_Q), -(-w // CLOUD_Q))
    room = _Room(blasts[0], seed)
    first, last = blasts[0].start, min(n - 1, max(b.last_frame() for b in blasts))

    def hand_at(t: int) -> np.ndarray | None:
        return None if mask is None else mask[min(t, len(mask) - 1)].astype(np.float32) / 255.0

    light = _light_dir(out[first], hand_at(first), blasts[0].o)
    buffets = []
    tone_s = None
    for t in range(first, last + 1):
        started = [b for b in blasts if b.start <= t]
        live = [b for b in started if t <= b.last_frame()]
        if not live:
            continue
        P = lin(out[t])
        tone = _tone(out[t])
        tone_s = tone if tone_s is None else 0.85 * tone_s + 0.15 * tone
        hand = hand_at(t)
        lit, drk = _speck_bufs(h, w, sc)
        air_ = _Haze.empty(hq)
        lump = 2.0
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
                room.throw(b, kf, t, camera_between(cam, room.start, b.start))
            haze, fore, puffs = (np.zeros(hq, np.float32) for _ in range(3))
            depth, shows, wide = b.dust(kf, H, Hp, lit, drk, sc, haze, fore, puffs, hand)
            if haze.any():  # the dusty air: diffusing as it goes and out of focus near the lens (CLOUD_SOFT_MAX at
                # the most), as thick as its parcels are close together
                near = zoom / (depth * CLOUD_Q)
                soft = min(((max(CLOUD_SOFT[0], BURST_SOFT if b.burst else 0.0) + CLOUD_SOFT[1] * kf)
                            + CLOUD_DEFOCUS * max(0.0, 1 - depth)) * near, CLOUD_SOFT_MAX / CLOUD_Q) * k
                per = np.float32((min(h, w) / CLOUD_Q) ** 2)
                d_ = cv2.GaussianBlur(haze, (0, 0), soft) * per
                air_.dens += d_
                air_.shown += d_ * np.float32(shows)
                if wide > 0:
                    air_.broad += d_ * np.float32(wide)
                if b.whole > 0:
                    air_.whole += d_ * np.float32(b.whole)
                if b.burst:
                    air_.burst += d_
                air_.front += cv2.GaussianBlur(fore, (0, 0), soft) * per
                if puffs.any():
                    air_.puff += cv2.GaussianBlur(puffs, (0, 0), soft) * per
                core = b.core0 * (1 + 0.6 * (1 - math.exp(-kf / RING_SLOW))) * near
                lump = max(lump, math.hypot(core, soft))  # (its core as blurred)
            if kf == b.knock():
                buffets.append((t, o, b.gain * (1 - b.q), KNOCK_PX * k * b.gain, u))
        b = started[-1]
        room.dust(t, camera_between(cam, room.start, t), camera_between(cam, room.start, max(room.start, t - 1)), lit,
                  drk, sc, hand, b, camera_between(cam, room.start, b.start))
        if air_.dens.any():
            _cloud_bend(P, air_, lump, hand)
        P0 = None if hand is None else P.copy()  # (bent, before what the air carries is laid on: _reedge)
        if air_.dens.any():
            _lay_cloud(P, air_, tone_s, light, lump, hand, (seed, t), code)
        _lay_specks(P, lit, drk, sc, tone_s, lambda y0, y1, x0, x1: _grain(seed, _BLAST, t, y1 - y0, x1 - x0), code,
                    k)
        if hand is not None:
            _reedge(P, P0, hand, k)
        out[t] = gam(P)
        if (t - first) % 8 == 0:
            say("air", (t - first) / max(1, last + 1 - first), f"Pushing the air out: frame {t - first + 1} of "
                                                                 f"{last + 1 - first}")
    for t, o, swell, knock, u in buffets:
        buffet(out, t, o, swell, knock, u)


def _blast_shot(plate: np.ndarray, out: np.ndarray, track: dict, mask: np.ndarray | None, tl: Timeline, seed: int,
                code: float, say) -> None:
    n, h, w = plate.shape[:3]
    xy, _, size = fx.track_arrays(track, n, w, h)
    normal = palm_normals(track, n, w, h)
    opened = openness(track, n, w, h)
    hits = [f for f in tl.hits if 0 <= f < n] or ([tl.tap_frame] if 0 <= tl.tap_frame < n else [])
    sizes = [tl.sizes[i] for i, f in enumerate(tl.hits) if 0 <= f < n] if tl.hits else [1.0] * len(hits)
    blasts = []
    for j, (f, sz) in enumerate(zip(hits, sizes, strict=True)):
        g = _open_from(opened, f)  # where and which way the push goes: from the open hand
        u, q = push_direction(track, g, normal[g], tl.aim, tl.tap_frame, n, w, h)
        gain = 0.55 + 0.45 * float(np.clip(sz, 0, 1))  # a "pf" pushes three quarters as hard as the loudest hit
        blasts.append(_Blast(xy[g], u, q, normal[g], size[g], w, h, f, gain, seed, j))
    if blasts:
        _push_frames(out, blasts, mask, tl, seed, code, say)


def buffet(frames_u8: np.ndarray, at: int, centre, swell: float, knock: float, along=(0.0, -1.0)) -> None:
    """The push reaching the phone, in place, from frame ``at``: the phone knocked ``knock`` px along the push
    (``along``, its direction in the picture) and back within three frames (KNOCK_SHAPE), and the picture swelling
    round the push and settling over six as much as ``swell`` (0..1: how much of a full push came at the lens); each
    frame smeared over the shutter's half frame of its move (a shake and a zoom blur). It never dims the picture."""
    if swell <= 0.05 and knock < 0.5:
        return
    n, h, w = frames_u8.shape[:3]
    ax, ay = np.asarray(along, np.float64) / max(float(np.hypot(*along)), 1e-9)
    grow = (0.0, 0.4, 1.0, 0.65, 0.3, 0.1, 0.0)  # from the frame before the gust on
    shove = (0.0, *KNOCK_SHAPE) + (0.0,) * (6 - len(KNOCK_SHAPE))
    swell = swell if swell > 0.05 else 0.0
    for d in range(6):
        f = at + d
        if not 0 <= f < n:
            continue
        s, s_was = (1 + BUFFET * swell * grow[d + i] for i in (1, 0))
        m, m_was = (knock * shove[d + i] for i in (1, 0))
        if s == s_was == 1.0 and m == m_was == 0.0:
            continue
        taps = 1 + round(6 * abs(grow[d + 1] - grow[d]) * swell / 0.6 + abs(m - m_was) / 3.0)  # (the faster it moves,
        # the longer the smear)
        acc = np.zeros((h, w, 3), np.float32)
        for j in range(taps):
            back = 0.5 * (1 - j / max(taps - 1, 1))
            sj, mj = s + (s_was - s) * back, m + (m_was - m) * back
            M = np.array([[sj, 0, centre[0] * (1 - sj) + mj * ax], [0, sj, centre[1] * (1 - sj) + mj * ay]],
                         np.float32)
            acc += cv2.warpAffine(frames_u8[f], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frames_u8[f] = np.clip(acc * np.float32(1 / taps) + 0.5, 0, 255).astype(np.uint8)


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
