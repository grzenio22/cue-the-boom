"""Air, made rather than rendered (cuesync.elements): a spinning ball of air over the open palm (``ball``), or a push
of air out of it (``blast``).

Air is invisible. What shows is what it carries (dust: specks of many sizes lighter and darker than what is behind
them, in its hue, smeared by what they move while the shutter is open, and hazes of dust too fine to see a speck of),
what it bends (the plate seen through it, shifted where the air is denser or swirls, so a straight edge behind it
wobbles, never steeper than BEND_SLOPE, so it never steps) and the light it gathers and spreads (a shadowgraph: denser
air focuses the room's light into brighter ripples with darker ones beside them, as the shadow of a candle's hot air
does on a wall). All of it is drawn from the plate itself, the hand's track and its matte, in linear light, with numpy
and OpenCV, every random part from ``seed``: the same inputs give the same bytes. Air throws no light of its own, so
nothing here lights or darkens the hand or the room as a whole: its dust keeps the hue of what is behind it, about half
lint lighter than it and half grit darker, with the plate's own grain. Nothing it paints is milky: the dusty air of a
push covers what is behind it a third at the most, and the shadowgraph moves it a few grey levels either way.

- ``ball``: a sphere of air standing on the palm, held by the voice, nearer the lens than the palm and so drawn bigger,
  never wider than about one and a half spans of the hand. What carries it is the dust it whirls round in its shell:
  a couple of thousand specks, most of them fine, a third medium and a few soft flakes, more of them towards the
  shell's outside and strongest on its outer third, where the shell is seen edge on, so its round edge reads at a
  glance; most strung in trains along arcs round one spin axis, each speck at its own pace, so each smears its own
  length (mostly under 12 px at 1080, a tenth or so longer); the near half's bigger, softer and fainter, the far
  half's small and sharp and behind the hand's matte, the near half's moving dust in front of the fingers, laid there
  as its own colour; and each train carries a haze of fine dust, a band of fine lines lighter and darker than the wall
  in its hue, line by line, densest at the limb, so the swirl reads at phone size as grey streamers of dust winding
  round it. Its bend: what is behind it is magnified a little (a weak lens, left off the hand), twisted round its spin
  axis by lobes that turn with it, and bent 6-12 px at 1080 wide by the swirling air inside it (its shell's streaky
  density, a puff of swirling air through its middle and the streaks it carries, each a filament of denser air), all
  turning with the spin, gone by a third of a radius past its rim; past that, the room's air it drags round with it
  pushes what is seen through it out and in by a few px in lobes turning with it, out to nearly three radii, so a
  switch plate near it wobbles on every frame; over the hand only what turns with the spin bends it, its mean over the
  hand taken out, so the fingers seen through it ripple a few px about where they are; the same density's fine
  shadowgraph ripples over the plate, one faint bright band broken into arcs runs round its wobbling outline, and its
  streaks, fragments of arcs no longer than 60 degrees on orbits leaning their own ways, show as a sheen as dark round
  each as it is light along it. As it is born the room's air is drawn into it, a ring converging on it from two radii
  that kinks what it passes towards the ball; it gathers in from wide as the voice starts,
  grows and spins faster the longer and louder the voice holds (twice as fast for a few frames after each of its hits),
  rides out a breath, pops when the voice comes back after a dip (its streaks widen, its bend flashes and it swells on
  the hit, the whole picture lifting little more than before, throwing a push of air whose dust is there at once and
  whose front crosses the room, and it gathers tight again within about eight frames), and when the voice lets go
  bursts into a push of air, as the blast's. The hand's edge is redrawn as the plate's mix of skin and wall, each
  taking the air's change on its side.
- ``blast``: a push of air out of the palm on each of the voice's hits. A pressure front leads, out from round the
  palm on the push's frame and across the room at about 1500 px a second at 1080 wide, slowing (what is seen through
  it is bent outward 6-10 px in a band 40-60 px wide at its front, all the way across the room, so a switch plate's
  straight edges kink as it passes; the band is a shadowgraph, with squeezed air darker behind it and thinned air
  lighter behind that; over the hand it bends nothing), a turbulent wake behind it smears what it passes, and after
  it, at about half its pace: a body of dust thrown off the fingers and the hand's outline in a cone about 40 degrees
  wide along the push (towards the lens it grows and goes soft; all round for a true push at the lens), fine specks, a
  third of them medium and a few tumbling flakes, with a haze of finer dust riding with it a dozen grey levels lighter
  than the wall in its hue, some of it drifting on slowly, and a thin vortex ring of dusty air rolling out of the
  palm, bending what is seen through it and showing
  only as a faint haze in the colour of what it covers (brighter on its side towards the room's light, less so on the
  other, under a third opaque, far less over the hand; each push's ring a little different) trailing a wake; the
  room's own dust near the push is shoved outward as the front passes it, and what lies on its ledges is kicked up
  a frame or two after as specks and puffs of dust lit by the room, in the hue of what they cover, the puffs rising
  and spreading as they fade within about twelve frames (one pool for the shot, which the next push throws again),
  and all of it then slows, swirls, drifts down and settles over twenty to thirty frames; what goes behind
  the palm's plane is hidden by the hand, what is in front of it is drawn over the hand, the dust faintly, the ring in
  soft faint puffs near the palm's plane and with its body further in front or when the whole ring crosses the hand.
  A ring swelling past the frame shows only its lumps and body, never a haze over the whole frame. A true push at the
  lens sends a doughnut of dusty air at the lens, starting off the palm with its middle above it so it rings the
  hand, its lower arc in front of the hand, drifting towards the frame's middle and swelling as it comes. Every push
  sweeps a few motes of dust past the lens, big soft discs out of focus lit in the room's hue that grow as they leave
  the frame, one or two within three frames, and a push out of a palm veils the lens for the frame its gust reaches it
  (the picture a few per cent lighter). What was
  thrown stays in the room: on a handheld shot it moves with the room, not with the screen. Every push knocks the
  phone 10-15 px at 1080 wide, smearing the picture out from the push as it is hit, which swings back past where it
  was and settles as a damped spring; one at the lens also swells and smears the picture outward as the gust reaches
  it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import fx_light as fx
from .elements import FORMS, Timeline
from .transfer import gam, lin

_BALL, _BLAST, _ROOM, _GRAIN, _CLOUD, _SWIRL, _LENS, _DUST, _DRAFT = 11, 12, 13, 14, 15, 16, 17, 18, 19  # mixed
# into each seed, so one seed gives each part its own randomness (_SWIRL, _LENS: a push's second and third streams, for
# its ring's eddies and the dust it sweeps past the lens; _DUST, _DRAFT a ball's second and third, for its dust and the
# air it drags round it: what the first stream draws does not hang on them)

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
BALL_IDLE = 0.8  # between two sounds the ball thins to this much of itself (at 0.6, its dust thinned as much, the
# ball between two hits was a faint outline at phone size)
BALL_MIN = 22  # frames it lives at least: a short "pf" still shows a ball forming, spinning and bursting
BALL_SPIN = 0.9  # revolutions a second once formed on a quiet voice ...
BALL_SPIN_VOICE, BALL_SPIN_HELD = 1.0, 0.6  # ... this much faster at the voice's loudest, and this much after
# BALL_GROW_S seconds held (about 2.5 a second at the most: faster smeared every streak into a ring) ...
HIT_SPIN = (1.0, 8, 4)  # ... and this much faster again (twice) for this many frames after each of the voice's hits,
# easing back over this many more (each hit whips the dust round: the voice drives what is seen)
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
BALL_HAND = (0.4, 0.5)  # over the hand the swirling air's bend is cut by this share and the shimmer by this (bent in
# full, a palm under the ball melted into a disc and the fingers' outlines broke) ...
HAND_STEADY = 0.35  # radii: ... and only what turns with the spin bends it there, the density's build broader than
# this and the twist's steady part left out, as the steady lens is ...
HAND_BEND = 3.5  # ... and what is left never bends the hand more than this many px at 1080: the fingers seen through
# the ball ripple with its spin about where they are (with the lens on them, they were bent a steady 5-7 px and looked
# fatter than in the plate; with the twist's steady part, a steady 1.5-2 px one way)
BALL_REACH = (1.02, 1.32)  # radii: the ball's bend (its drag on the air round it included) eases off from the first
# and is gone by the second, well inside the frame's crop round it (cut off at the crop's edge 1.4 radii out, a
# switch plate's corner there took an 8-10 px step)
BEND_SLOPE = 0.25  # no bend shifts what is behind it by more than this many px per px of picture: a straight edge
# behind the air wobbles smoothly and never steps (a ball's pulse bent a switch plate's divider by 8-10 px over a few)
BALL_CA = 0.03  # colour fringe of the bend, none over the hand and just past its edge (0.22 drew red and cyan
# outlines round the fingers; over the matte, a thin cyan line ran along the thumb)
BALL_WOB = (0.035, 0.02)  # radii: its outline wobbles this much in three lobes and in five, turning with the spin and
# slowly changing (round to 3.5 % and ringed by a crisp band, it was a crystal ball; at 9 % and 6 % with its rim's
# bright and dark pair, a lumpy jelly; at 4.5 % and 3 %, with that pair, 2-3 nested lumpy contours: a water drop)
BALL_SHADOW = (0.07, 0.6, 0.3)  # the shadowgraph: the rim focuses the room's light into one band this much
# brighter (about 4 grey levels), shown on this share of the round, where the rim's arc noise is highest, easing in
# over this much of its spread: arcs turning with the spin (at 0.12 its band drew a hard round outline; at 0.07 alone
# with the streaks still painted, a grey ring; with a darker band inside it, a bright and dark pair that drew nested
# contours round the ball: a ripple, a water drop)
RIPPLE_FINE = 0.12  # radii: the shadowgraph's ripples come from the density's detail finer than this, so the band is
# the one contour round the ball (its broad build, the shell's limb and the puff's dome, drew a second and a third
# ring inside it: a water drop)
BALL_BAND = 0.05  # radii: the band's softness (at 0.035 it was a drawn line)
MIST_CORE = 0.55  # the shell's density: this share of it through the ball's depth, densest through its middle as a
# solid ball of air is (all of it densest at the limb, the ball bent as a hollow jar)
BALL_BACK = 0.45  # the far side shows this much (seen through the near side), and not through the fingers
BALL_CAUSTIC = 0.03  # the palm under the ball takes its light focused into moving patches, this much brighter / darker
BALL_STREAKS = 90  # streaks of denser air orbiting the ball, each a fragment of an arc ...
FRAGMENT = (0.45, 1.0)  # ... this many rad long (out to under 60 degrees: with arcs of up to 170 degrees on shared
# bands of latitude they overlapped into closed wound ellipses, a Slinky) ...
FRAG_TILT = 0.55  # ... each on its own orbit, leaning off the spin axis by up to this many rad its own way (stacked
# square to the axis they read as coils; leaning every which way, as a ball of yarn with no axis)
STREAK_FINE = 0.25  # share of the streaks that are fine, the rest broad wisps
CORE_PUFF = (0.6, 2.5)  # inside this many radii the ball's middle is a puff of swirling air (in place of the inner
# shells' streaks), its swirl wound this many rad from its edge to its middle
BALL_AXIS_Z = 0.7  # the spin axis leans at least this far off the lens's axis (the largest z it keeps), so its streaks
# read as arcs round an axis (a palm turned to the lens stood the axis at the lens: rings round a whirlpool)
BALL_NEAR = 0.35  # radii: the streaks brighten from the far side's BALL_BACK to full over this much either side of
# the limb (cut at the limb, every streak broke there)
STREAK_ALPHA = 0.75  # a streak of brightness 1 is this dense a filament of air: it bends what is behind it
# (BALL_DENS) and shows as its sheen (SHEEN) ...
STREAK_PEAK = 1.3  # ... and no streak is brighter than this many times the mean of the brighter half (one crisp
# bright arc read as wire)
SHEEN = (0.16, 18.0, 6.0)  # the streaks' sheen is a shadowgraph of them, neither lighter nor darker on the whole: what
# is behind a streak this much brighter for a density of 1 along it, as much darker round it out to this many px at
# 1080, and the sheen with the rest of the ball's shadowgraph never more than this many 8-bit grey levels lighter or
# darker than what is behind it (lit, its streaks traced lighter wound ellipses on the wall)
STREAK_SKIN = (0.06, 0.3)  # over the hand the streaks are drawn only as broad wisps, at this density at the most, and
# show this much of themselves, their far half not at all (a fine bright arc across the palm read as a white glare
# smear on the skin; at 0.25, every streak sat in front of the hand)
NEAR_LIT = 0.6  # its lint in front of the hand is this much of the way from the wall's colour, lighter, to the
# room's lit tone: dust catching the light (as the wall's colour alone, over a lit palm it read as grey smears) ...
NEAR_SKIN = (0.0, 0.8, 3.0, 6.0, 0.5, 0.4)  # ... and the ball's near half is in front of the fingers: over the hand, laid as
# its own colour (_in_front), its specks show this share of themselves as they barely move, and this share once
# smeared from the third to the fourth px at 1080 or longer (moving dust, not dirt on the skin), and its haze's lines
# this share; its grit this share of that (dark dots on the skin read as dirt); the far half's none (its dust cut to a
# third over the skin and its haze to nothing, the swirl sat behind the fingertips like a decal on the wall)
HAND_SOFT = 6.0  # px at 1080: the matte is softened this much wherever the ball eases off over the hand (cut at its
# edge, a fingertip got a jagged bite)
STREAK_PX = (1.3, 9.0)  # px at 1080: a fine streak's softness, and a broad wisp's, at its head (a wisp under 3 px
# was under a pixel at phone size, and the ball a ghost; at 6.5 the bands were thin lines) ...
STREAK_TAPER = (0.4, 0.55)  # ... and past this share of its length it is drawn this much as wide (as wide all along,
# 46 streaks read as wire wound round a globe)
STREAK_STEP = 0.7  # px between the points a streak is drawn from (further apart a soft streak turned to beads)
STREAK_END = (3.0, 0.3)  # where a streak turns towards the lens it is brighter by its foreshortening (up to the first)
# to this power
SHELL_DUST, ROOM_DUST = 2400, 24  # dust whirled round in the ball's shell; and the room's dust drawn in (45 specks
# inside it, a third of an opacity of 0.35 lit a little, changed the wall by 9-10 grey levels at phone size: the ball
# nearly vanished; 1100 in its shell read at phone size as a faint whorl) ...
DUST_MIX = (0.6, 0.34)  # ... this share of it fine specks, this share medium ones, the rest soft flakes (all alike,
# 4 px across and smeared 25-30 px, every speck was the same hard-ended rod: grass clippings, iron filings, gnats)
SHELL_R = ((1.2, 2.0), (2.8, 3.8), (5.0, 8.0))  # px at 1080: the core of a fine speck, a medium one and a flake ...
SHELL_SEEN = ((12.0, 20.0), (30.0, 48.0), (10.0, 15.0))  # ... and how many 8-bit grey levels lighter or darker than
# what is behind it each makes at full strength (the medium ones, what reads of the specks themselves at phone size)
SHELL_LEVELS = 48.0  # the most of those (each speck's strength is its share of it)
SHELL, SHELL_TRAINS = (0.6, 1.0), (30, 0.3, 1.25)  # radii: the shell the dust turns in; most of it strung in this many
# trains along arcs of it, this share loose, a train up to this many rad long: the swirl reads as streamers of dust
# winding round it (spread evenly, the specks read as scribbles at phone size) ...
SHELL_OUT = 1.5  # ... more of it towards the shell's outside: its depth in the shell is a uniform share to this power
# (spread evenly through it, the ball had no edge at phone size, a whorl, not a ball)
LIMB = (0.8, 0.6, 0.25)  # the specks seen past this share of its radius (the second) are up to this much stronger
# (the first), fully this much further out: on the outer third the shell is seen edge on and its trains pile up, so
# its medium specks there make 35-50 grey levels and the ball's round edge reads at a glance
DUST_SPIN = (0.6, 0.3)  # the dust turns this many times as fast as the ball's spin, and up to this much more (each
# train its own pace) ...
DUST_SURGE = (0.12, (1.5, 3.0))  # ... each speck surging ahead of its train and falling back by up to this many rad,
# this many times a turn of the spin, so neighbours move at their own pace and smear their own lengths (a train moving
# as one drew a run of identical rods)
DUST_LOUD = (0.85, 0.15)  # ... and shows this much on a quiet voice, and this much more at its loudest
DUST_BACK = 0.6  # the far side's dust shows this much (seen through the near side's air), and not through the hand
DUST_NEAR = (0.45, 2.0, 0.1, 0.25)  # the near half's dust is drawn this much bigger, softened this many px at 1080
# (nearer the lens than the hand the lens is focused on) and this much fainter; the far half's this much smaller and
# sharp: the swirl has depth
LONG = (15.0, 0.11, 0.6)  # px at 1080 and shares: no more than this share of the ball's specks are smeared longer than
# this at once, the rest shortened to between this share of it and all of it, each speck its own (cut to it alone, on
# a hit's fast spin most specks were 15 px rods again)
HAZE = (20.0, 4.0, 0.35, 8, (0.035, 0.065))  # the trains carry a haze of fine dust, too fine to see a speck of, each
# speck smeared into a thin line by its motion: so each train's haze is this many fine lines of it (the fourth), this
# many 8-bit grey levels darker than what is behind them at full strength, in its hue, this soft (px at 1080), spread
# across a band this many rad of the shell wide either side of the train (15-35 px across), broken along their length,
# their middle across the ball showing this much of what their limb does (the eye grazes the most of a shell at its
# limb): at phone size the swirl reads by these striated streamers, not by single specks (as smooth bands 16-28 px
# soft, the haze was a dark smudge round the ball, a ghost)
HAZE_MOST = 1.6  # ... and where its lines pile up (at the limb, crossing) it is never more than this many times one
# line's full strength, thicker haze adding less and less (at 2.2 and 15 levels, piled up 25-30 levels darker, the
# swirl read as charcoal scribbles on the wall)
HAZE_LIT = (0.55, 1.0)  # this share of each train's haze lines carries dust lit by the room: this much as many grey
# levels lighter than what is behind it, not darker (all of it darker, the swirl read as grime on the wall; lit or dark
# a train at a time, the lit trains read as chalk smears and the dark as charcoal: mixed, each is a grainy grey band)
BALL_PAD = 1.4  # radii: the ball's air (its bend, the dust it draws in) reaches this far from its middle ...
DRAFT = (6.0, (0.9, 1.25), (1.6, 2.8), 0.5)  # ... and the room's air it drags round it, out past its rim: what is
# seen through it is pushed out and in by about this many px at 1080 (their spread; up to about twice it), in lobes
# turning with the spin, eased in over the first pair of radii and out over the second, and dragged round with it this
# share as far: a switch plate or a door's edge near the ball wobbles on every frame (with its bend gone by a third of
# a radius past its rim, a plate two radii off it bent only as a hit swelled the ball)
DRAFT_LOBES = ((3, 4, 5, 7), (0.7, 1.0, 0.8, 0.5), (0.25, 0.45))  # its lobes: this many round, this strong each,
# turning this many to this many times as fast as the ball (the room's air lags it, each lobe at its own pace, so the
# wobble changes as it travels)
GATHER = (5, (2.1, 1.0), 4.0, 30.0)  # as a ball is born the room's air is drawn into it: over its first this many
# frames a ring converges on it from the first to the second of its radii (as formed), pulling what is seen through it
# this many px at 1080 towards it, in a band this many px at 1080 across (sigma): the plate's edge kinks towards it as
# the air comes in (with nothing, the voice's first hit bent nothing)
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
FRONT_FROM = 0.06  # share of the short side: the front's radius as it leaves the palm (a ball's leaves its shell) ...
FRONT_SPEED = (110.0, 0.5, 5.0, 0.25)  # ... running out at this many px a frame at 1080 as it leaves, slowing by this
# share of that over about this many frames (about 1500 px a second over its run, as a blast's front slows), and this
# share faster on screen for a push at the lens (out to the frame's edge within eight frames, a blink: it was gone
# before the eye found it)
FRONT_HAND = (10.0, 8.0)  # px at 1080: the front bends nothing over the hand's matte grown this much and softened this
# much more, so it leads from the palm on the push's frame without bending the fingers (held off until it was a hand
# and a half out, it trailed the ring and the dust it throws)
BUBBLE_PX = 9.0  # px at 1080: how far the pressure front pushes what is seen through it outward, at its strongest
# (30, held under a slope of 0.2, spread the front over a band 200 px wide: a bulge, not a front; 18 needed a slope
# of 0.45, and a switch plate it crossed bent by over a px in 3) ...
BUBBLE_HOLD = 0.75  # ... all of it until this share of its run, so it crosses the room's straight edges whole (a switch
# plate, a door's edge, a skirting board: fading from the palm on, it had 3-5 px left as it crossed the plate, whose
# edges stayed straight), then fading out by its end
FRONT_LEAST = 0.65  # ... and on a soft push (a ball's pulse) at least this share of it, so its front still kinks the
# straight edges it crosses (at the pulse's own 0.3-0.6, a switch plate under it moved 2-4 px: a soft wave, unseen)
BUBBLE_INNER = 0.2  # ... most of it a thin band at the front, and this share of it a push growing out to the front
# from inside (all of it growing to the front, the whole room bulged like a funhouse mirror)
FRONT_W = (40.0, 60.0)  # px at 1080: the front's band as it leaves the palm, and at the end of its run, and at least
# as wide as keeps its bend under BEND_SLOPE
SHADOWGRAPH = (0.17, 0.1)  # the front gathers and spreads the light behind it: a bright band, dark either side, this
# deep (at 0.06 the front was invisible on a plain wall), the band trailing it only this share as dark (as dark, it
# sank the inner edge of the ring of dusty air following the front under the wall)
COMPRESS = (-0.09, 0.06, 45.0, 40.0)  # behind its crest the air is squeezed, then thinned: what is seen through it
# this share darker in a band, then this share lighter in the next, the first's middle this many px at 1080 behind the
# crest, each this many across, in the colour of what is behind it (with only a 1-2 px bright line on it, at phone size
# the front read as a ripple, not as a push)
WAKE_PX = 3.5  # px at 1080: the turbulent wake's radial smear behind the front ...
WAKE_ROUND = 8.0  # ... its eddies this many half-degrees across round the front (at 0.8 they were a few px across
# where the front crossed the switch plate, and waved its straight edge like a glitch; at 2.5 they still bent what was
# behind them by over a px per px, folding it)
BLAST_CA = 0.02  # (0.12 split an orange fringe off the fingertips, 0.05 a cyan one off the switch plate)
CLOUD_N = 4000  # parcels of dusty air the push sets rolling: a vortex ring, what a palm pushed through air makes (a
# cloud of clumps thrown off the palm slowed and faded where it was: a glow stuck on the hand, not air going somewhere)
RING_R = (0.5, 1.6)  # hand sizes: the ring's radius as it leaves the palm, and how many times wider it ends ...
RING_FORM = 0.5  # ... rolling up from this share of that over the frames its dusty air comes in over (CLOUD_IN), so
# it follows the front out of the palm (there whole at once, it was out past the front on the push's first frame)
RING_CORE = 0.075  # ring radii (as it leaves): the radius of its core, which the dusty air rolls round (it fattens as
# it slows; at 0.16 the ring was a thick scalloped torus, a smoke ring stamped on the room; at a quarter, a ring seen at
# a slant closed its hole and read as a cauliflower)
RING_BEND = (10.0, 22.0)  # px at 1080: the ring's dusty air bends what is seen through it this far where it thickens
# fastest, its thickness smoothed over this many px first (its body is a refractive band with a faint haze, not paint;
# smoothed over its lumps alone, a few px, the bend would be steeper than a px per px and fold what is behind it)
RING_SPEED = 0.045  # share of the short side a frame: how fast the ring leaves the palm on the loudest hit, about
# half as fast on screen as the front, which leads it (at 0.06, rushed at the lens, the ring outran the front) ...
RING_SLOW = 12.0  # ... slowing over this many frames, as the front does (a ring holds together and carries on; slowed
# over 9, it fell to a fifth of the front's pace)
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
LENS_RUSH = (1.5, 0.0)  # ... a push at the lens sends its ring this much faster, straight at the lens, swelling as it
# comes but slower on screen than the front ahead of it (rushed so a true one swelled past the frame's edges in about
# ten frames, it outran the front: the puff led and the shock trailed) ...
LENS_WIDE = 0.2  # ... this much wider as it leaves, so it rings the hand (dust in front of a lit palm hardly shows:
# a ring on the palm read as puffs between the fingers and a smudge by the wrist; 0.8 wider, it was out past the
# front) ...
LENS_THICK = (4.5, 2.0)  # ... and draws it this much thicker: seen face on, a ring's air is spread round a wide
# annulus rather than piled up along a slanted band, and at the sideways thickness it was a faint veil (a true push at
# 3.5 round a fat core sat at full opacity, flat translucent petals; a ball's release ring at 3.5, its shadow side and
# its inner edge now kept off the wall's tone, was a little fainter than before) ...
LENS_CORE = (-0.4, 0.5)  # ... round a core this much fatter: a true push's a doughnut with a body (blurred with its
# spread, a fat ring closed its hole and read as a blob; kept sharp (CLOUD_SOFT_MAX), a thin one read as a rope), a
# push partly at the lens thinner (it goes soft as it nears the lens, and fat it closed its hole) ...
LENS_LIFT = (0.5, 0.25)  # ... a true push's starts this many hand sizes off the palm along its normal, its middle this
# many up the picture from the palm, so its lower arc crosses the hand (centred on a palm low in the frame, only its
# top arch was on screen, and read as smoke rising behind the hand; 0.6 up, its top arch was out past the front) ...
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
CLOUD_IN = 4  # ... and it comes in over the push's first this many frames, behind the front (the bend and the shock
# ring own them; there at once, it sat at the base of the fingers as a grey halo: dirt on the hand), but at once out of
# a ball's burst (there it is the burst)
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
DUST_N, FLAKES = 600, 30  # a body of dust and grit thrown off the hand behind the front; bigger flakes that tumble
# (300 fanned out all round, the wall between the hand and the front was clean at phone size once the front passed: a
# stock ripple, not air carrying something; 450 specks too faint to see covered 0.4 % of the frame at phone size by
# frame 50; 1400 read as sensor snow; 50 soft round flakes, a sixth of the burst, read as snow out of focus)
DUST_R = (3.0, 5.0)  # px at 1080 at the palm's depth: a speck of the room's dust's core (1.1-2.4, and 2-4 at 20 grey
# levels, too fine to read at phone size) ...
DUST_SEEN = (0.43, 0.52)  # ... and it shows this much of itself (SPECK_LEVELS: 22-26 grey levels lighter or darker
# than the wall), more of it as it moves fast, and 0.75 of that as it hangs
DEBRIS = (0.64, 0.85)  # of the burst's specks (its flakes aside) this share fine, the rest medium; and this share of
# all of it thrown off the hand's outline, most from its side the push goes out of, the rest out of the palm (all out
# of the palm, alike, it fanned out of two or three gaps between the fingers as needles: a hedgehog)
DEBRIS_R = ((1.6, 2.4), (3.0, 4.0), (5.0, 8.0), (2.4, 3.4))  # px at 1080: the core of a fine speck, a medium one, a
# flake and one drifting (DRIFT) ...
DEBRIS_SEEN = ((0.29, 0.4), (0.75, 1.0), (0.52, 0.72), (0.43, 0.54))  # ... and how much of SPECK_LEVELS each shows
# (fine 14-20 grey levels, medium 35-50: what reads of the body at phone size) ...
DEBRIS_HANG = 0.9  # ... all of it moving fast (12 px a frame at 1080), and this much of it as it slows and hangs (at
# 0.75, the body slowing behind the front faded at phone size)
BODY = (0.2, 0.35, (0.4, 0.58))  # out of a palm, the body of dust goes out round the push's way in the picture, spread
# this many rad (sigma) and cut at this many either side of it (a cone 40 degrees wide; all round for a true push at
# the lens), each speck leaving at this share of the front's pace, so its leading edge rides at about half the front's
# (fanned all round at up to 0.6, the dust spread thin over the whole room)
BODY_HAZE = (14.0, 1.35, 16.0, 0.3)  # ... and a haze of dust too fine to see a speck of rides with it: this many grey
# levels lighter than what is behind it, in its hue, where the body is as thick as as it leaves, piling up to this many
# times that at the most, this soft (px at 1080), each speck carrying this much of it (with specks alone, at phone size
# the push was a ripple ring on a clean wall)
DRIFT = (0.25, 0.35, 10)  # this share of the burst's specks drift: leaving at this share of the others' pace and
# hanging this many frames longer, still in the air once the rest has settled (by frame 65 the room was clean)
MOTES = (4, 6)  # dust swept past the lens by a push, this many motes on its softest and on its hardest: what says the
# air reached the camera, with the knock (every speck flew across the wall; 8-15 small faint ones read as smudges on
# the lens, and none at phone size) ...
MOTE_FROM = 0.12  # short sides: ... coming from round the palm this far ...
MOTE_D = (60.0, 140.0)  # ... each this many px at 1080 across (out of focus so near the lens: a soft disc, its softness
# a third of that) ...
MOTE_GROW = 1.5  # ... growing this many times as wide as it comes ...
MOTE_RUN = (1, 3, 4.0, 6.5)  # ... showing from this many frames after the push to this many, and out past the frame's
# edge this many to this many frames later ...
MOTE_FAST = (2, 2.0, 3.0)  # ... but the first this many out past it within this many to this many frames: the gust
# sweeps them off the lens
MOTE_LEVELS = (18.0, 28.0)  # ... this many 8-bit grey levels lighter than what is behind it, the room's light they
# scatter, in its lit hue
GUST = (2, 0.04, 0.03)  # frames after the push its gust reaches the lens, and on that frame alone the picture is lifted
# this share, with the room's lit tone laid over it this share of the picture's mean: a veil of dust on the lens (with
# motes alone, nothing said the air hit the camera)
BURST_N = (300, 24)  # a ball's burst flings this many specks and flakes of its shell's dust all round, with no body of
# it nor its haze (the push's 600 and their haze after each of a ball's pops hung a light veil over the room) ...
DUST_FOLLOW = 0.6  # ... at up to this share of the front's speed, so it follows the front ...
DUST_DRAG = 10.0  # ... slowing over this many frames, out into the room behind it (out at a third of the short side a
# frame, it outran the front; slowed over 6 at half its speed, it hung over the hand)
DUST_AIM = (0.7, 0.2, 0.4)  # ... flying out round the push's own way, this much along it in the picture (along the
# push itself, a push at the lens sent it at the lens and it barely spread on screen), from a patch of the palm this
# many hand sizes across (from 0.4 it speckled the whole hand at once), with only this share of its speed towards or
# away from the lens: it spreads over the room, not into the lens
DUST_STIR = 0.012  # share of the short side a frame: how hard the eddies push the dust about once it has slowed ...
DUST_CALM = 14.0  # ... calming over this many frames (stirred as hard for its whole life, the hanging dust zipped
# about in streaks: film scratches)
DUST_FALL = 0.0004  # share of the short side: thrown dust gains this much speed down a frame, so once slowed (by
# DUST_DRAG) it drifts down at about 65 px a second at 1080
ROOM_N = 320  # the room's own dust, one pool for the shot: still and unseen until a front shoves it ...
ROOM_NEAR = 0.6  # short sides: ... most of it near the first push and ahead of it, thinning out this far (1500 specks
# over the whole frame read as sensor snow)
ROOM_PUSH = 0.045  # share of the short side a frame: how fast the front throws the room's dust it passes (at 0.06,
# as fast as the front, the dust it threw ran along with it)
ROOM_SEEN = 0.75  # thrown, it shows at least this much as it slows, fading out with its life (seen only while fast, it
# flashed past and was gone before the eye found it; held, it hung through the next push)
LEDGE = (60, 8, (1, 3), 1.15, 0.6, 0.5)  # the room's dust lying on its ledges, where the plate has a level edge (a
# skirting board's top, a switch plate's, the floor's line): this many specks, each with a puff of finer dust in this
# many parcels, kicked up this many frames after the front reaches it, lying this many palm depths from the lens (on
# the wall, behind the hand), thrown at this share of the floating dust's pace and lifted off the ledge by this share
# of it: the front kicks dust off the surfaces it passes (with only dust in the air, the room never reacted to it)
LEDGE_PUFF = (0.3, 1.5, 9.0, 4.5, 0.3, 4.0, 12, 3.0)  # a puff: fine dust lit by the room's light as it falls there
# (the scene round it, blurred), covering what is behind it this much at the most, as bright as that light times the
# second and in the hue of what it covers, so it lifts what is behind it in its own hue, never more than the last times
# (in the room's lit hue, on an orange skirting it lifted the blue alone: violet stains; lit as the scene round it,
# where the wall's grey met the skirting it still lifted its blue most); its parcels this soft (px at 1080), drifting
# apart at up to this many px a frame, carried at this share of its speck's pace, rising this many px a frame (fine
# dust floats), and gone this many frames after it is kicked up (hanging where it was kicked up for 10-15 frames, it
# read as a blotch)
DUST_LIFE = (18, 12)  # frames: thrown dust hangs this long after the push, then settles out over the next this many,
# drifting down (gone by 20 frames after the push, the room was clean again before the eye found the dust)
FOCAL = 1.0  # focal length in short sides (a phone's main camera: about 53 degrees across the short side)
BUFFET = 0.006  # a push straight at the lens swells the picture this much when the gust reaches the phone ...
BUFFET_SHAPE = (0.0, 0.6, 1.0, 0.7, 0.3, 0.08, 0.0)  # ... this much of it from the frame before the gust on, with
# the knock (lagging it, it hid the knock's swing back)
# (with the knock, the room moves 12-15 px at its most: at 0.035 it threw the room 18 px and snapped back by 10 in a
# frame; it never dims the picture: on one frame the dip was an exposure flicker)
KNOCK_PX = 11.0  # px at 1080: every push knocks the phone this far along it at its loudest (3 px, and only for a push
# at the lens, was no knock at all) ...
KNOCK_SHAPE = (0.67, 1.0, 0.69, 0.21, -0.13, -0.23, -0.18, -0.06)  # ... this much of it from the knock's frame on, a
# damped spring: out in two frames, back by under 6 px a frame and past where it was by about 3 px, settled in five
# more (off on three frames and back at once, it snapped back by 10 px in one)
ZOOM_BLUR = (4.0, (1.0, 0.6, 0.25))  # px at 1080 at the frame's furthest corner: the knock smears the picture out
# from the push by this much at its loudest, this share of it on the knock's first frames (the knock alone moved the
# picture and nothing said the air had hit the camera)
OUTLINE_REACH = 1.35  # hand sizes: the dust a push throws off the hand's outline comes off it this far from the palm
# at the most (the fingers', not the arm's)
KNOCK_AT = (0, 2)  # frames after the push: a push across the picture knocks the phone from the push's frame on (the
# hand's shove), one straight at the lens from this many, as its gust arrives (waiting for the front to pass the
# frame's corners, the knock came 0.4 s after the voice, with nothing arriving at the lens)
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


def _steep(f: np.ndarray, axis: int) -> np.ndarray:
    """|change| of ``f`` (h, w) to its neighbours along ``axis``, the larger of the two at each point."""
    d = np.abs(np.diff(f, axis=axis))
    out = np.zeros_like(f)
    if axis == 0:
        out[:-1] = d
        out[1:] = np.maximum(out[1:], d)
    else:
        out[:, :-1] = d
        out[:, 1:] = np.maximum(out[:, 1:], d)
    return out


def _tame(fx_: np.ndarray, fy_: np.ndarray, cell: float, slope: float = BEND_SLOPE) -> tuple[np.ndarray, np.ndarray]:
    """A bend (``fx_``, ``fy_`` (h, w): px each point of a grid ``cell`` px apart reads the plate from, off itself)
    smoothed wherever it is steeper than ``slope`` px per px of picture, each time wider, until nowhere is over: so a
    straight edge behind it wobbles and never steps (scaled down where steep instead, a bend of 20 px took the scale's
    own slope and stepped more)."""
    lim = slope * cell
    fx_, fy_ = fx_.astype(np.float32), fy_.astype(np.float32)
    if min(fx_.shape) < 3:
        return fx_, fy_
    for i in range(8):
        J = np.maximum.reduce([_steep(f, ax) for f in (fx_, fy_) for ax in (0, 1)])
        if float(J.max()) <= 1.02 * lim:
            break
        sig = 2.0 * 1.5 ** i
        r = int(sig) + 1
        over = cv2.dilate(np.clip(J / lim - 1, 0, 1).astype(np.float32), np.ones((2 * r + 1,) * 2))
        w = np.minimum(1.0, 2 * cv2.GaussianBlur(over, (0, 0), sig))
        fx_ = fx_ + (cv2.GaussianBlur(fx_, (0, 0), sig) - fx_) * w
        fy_ = fy_ + (cv2.GaussianBlur(fy_, (0, 0), sig) - fy_) * w
    return fx_, fy_


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


def _at(small: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """A map SCENE_Q times smaller than the frame (_behind's grid, (h, w) or (h, w, channels)) read at frame px
    (``ys``, ``xs``), as _up draws it (bilinearly between its cells' middles, its edge held)."""
    q = SCENE_Q
    fy = np.clip((ys + 0.5) / q - 0.5, 0, small.shape[0] - 1)
    fx_ = np.clip((xs + 0.5) / q - 0.5, 0, small.shape[1] - 1)
    iy, ix = fy.astype(np.int64), fx_.astype(np.int64)
    ty, tx = (fy - iy).astype(np.float32), (fx_ - ix).astype(np.float32)
    if small.ndim == 3:
        ty, tx = ty[:, None], tx[:, None]
    iy1, ix1 = np.minimum(iy + 1, small.shape[0] - 1), np.minimum(ix + 1, small.shape[1] - 1)
    return (small[iy, ix] * (1 - ty) + small[iy1, ix] * ty) * (1 - tx) + (small[iy, ix1] * (1 - ty)
                                                                        + small[iy1, ix1] * ty) * tx


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


def hit_spin(t: int, hits: list[int]) -> float:
    """How many times as fast the ball spins on frame ``t`` for the voice's ``hits``: 1 + HIT_SPIN's first for its
    second's frames from a hit, easing back over its third."""
    return 1 + HIT_SPIN[0] * max([float(_smooth((HIT_SPIN[1] + HIT_SPIN[2] - (t - f)) / HIT_SPIN[2]))
                                  for f in hits if 0 <= t - f] or [0.0])


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


def _turn_sense(B: np.ndarray) -> float:
    """Which way (+1 or -1, as the picture's angles grow) the near side of a ball with axes ``B`` (rows e1, e2, the
    spin axis) crosses the picture as it spins."""
    ph = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    pt = np.cos(ph)[:, None] * B[0] + np.sin(ph)[:, None] * B[1]
    vel = -np.sin(ph)[:, None] * B[0] + np.cos(ph)[:, None] * B[1]
    near = pt[:, 2] > 0
    if not near.any():
        return 1.0
    return float(np.sign(np.mean(pt[near, 0] * vel[near, 1] - pt[near, 1] * vel[near, 0])) or 1.0)


class _Ball:
    """One ball of air: its seeded streaks, dust and textures, and the state it carries from frame to frame (how far
    it has spun, its axes)."""

    def __init__(self, seed: int, salt: int):
        rng = fx._rng(seed, _BALL, salt)
        N = BALL_STREAKS
        tilt = FRAG_TILT * rng.uniform(0, 1, N) ** 0.7
        az = rng.uniform(0, 2 * np.pi, N)
        ct, st, ca, sa = np.cos(tilt), np.sin(tilt), np.cos(az), np.sin(az)
        # each streak's own orbit: (u, v) span its plane, w is its axis, leaning off the ball's by its tilt
        self.u = np.column_stack([ct * ca, ct * sa, -st])
        self.v = np.column_stack([-sa, ca, np.zeros(N)])
        self.w = np.column_stack([st * ca, st * sa, ct])
        self.lat = np.arcsin(rng.uniform(-0.9, 0.9, N))  # spread over the sphere, thinning towards its poles
        self.rho = rng.uniform(0.82, 1.03, N)  # on its outer shells (CORE_PUFF fills its middle)
        self.phase = rng.uniform(0, 2 * np.pi, N)
        self.span = rng.uniform(*FRAGMENT, N)
        self.speed = rng.uniform(0.9, 1.15, N)
        self.helix = rng.uniform(-0.15, 0.15, N)
        self.bright = rng.uniform(0.12, 1.0, N) ** 2  # most faint, a few bright (as bright as each other, the streaks
        # read as wire)
        self.glow = rng.uniform(0, 2 * np.pi, N)  # ... and each brightens and fades as it turns, so a different few
        self.glow_f = rng.uniform(0.08, 0.2, N)  # read at a time, at this many times the spin
        self.soft = rng.random(N) >= STREAK_FINE  # broad soft wisps; the rest fine streaks
        self.wob = rng.uniform(0, 2 * np.pi, N)
        self.gaps = rng.uniform(0.4, 1.0, N)  # each streak comes and goes along its length this many times
        self.gph = rng.uniform(0, 2 * np.pi, N)
        dr = fx._rng(seed, _BALL, salt, _DUST)  # (its own stream: the textures do not hang on how much dust it has)
        m, nt = SHELL_DUST, SHELL_TRAINS[0]  # dust whirled round the axis in the shell, most of it in trains strung
        # along arcs (SHELL_TRAINS), each train at its own pace and swaying in latitude as one, the rest loose
        tr = dr.integers(0, nt, m)
        loose = dr.random(m) < SHELL_TRAINS[1]
        def deep(k_):  # depths in the shell, more of them towards its outside (SHELL_OUT)
            return SHELL[1] - (SHELL[1] - SHELL[0]) * dr.random(k_) ** SHELL_OUT

        self.t_lat = np.arcsin(dr.uniform(-0.8, 0.8, nt))  # (round its middle: specks near its poles barely move)
        self.t_rho, self.t_phase = deep(nt), dr.uniform(0, 2 * np.pi, nt)
        self.t_len = dr.uniform(0.3, 1.0, nt) * SHELL_TRAINS[2]
        self.t_turn = DUST_SPIN[0] + DUST_SPIN[1] * dr.random(nt)
        self.t_sway = dr.uniform(0, 2 * np.pi, nt)
        nl = HAZE[3]  # its haze's fine lines (HAZE): off the train across its band, how strong, their gaps, and which
        # are lit by the room (HAZE_LIT)
        self.h_lat = dr.uniform(-1, 1, (nt, nl)) * dr.uniform(*HAZE[4], nt)[:, None]
        self.h_rho = dr.normal(0, 0.015, (nt, nl))
        self.h_w = dr.uniform(0.35, 1.0, (nt, nl))
        self.h_gap = dr.uniform(1.0, 3.5, (nt, nl))
        self.h_gph = dr.uniform(0, 2 * np.pi, (nt, nl))
        self.h_lit = dr.random((nt, nl)) < HAZE_LIT[0]
        self.d_lat = np.where(loose, np.arcsin(dr.uniform(-0.8, 0.8, m)), self.t_lat[tr] + dr.normal(0, 0.05, m))
        self.d_rho = np.where(loose, deep(m), np.clip(self.t_rho[tr] + dr.normal(0, 0.03, m), *SHELL))
        self.d_phase = np.where(loose, dr.uniform(0, 2 * np.pi, m),
                                self.t_phase[tr] + dr.uniform(0, 1, m) * self.t_len[tr])
        self.d_turn = np.where(loose, DUST_SPIN[0] + DUST_SPIN[1] * dr.random(m), self.t_turn[tr])
        self.d_sway = np.where(loose, dr.uniform(0, 2 * np.pi, m), self.t_sway[tr])
        self.d_surge = dr.uniform(*DUST_SURGE[1], m)  # (DUST_SURGE)
        self.d_surge_ph = dr.uniform(0, 2 * np.pi, m)
        self.d_glint = dr.uniform(0, 2 * np.pi, m)
        cls = np.searchsorted(np.cumsum(DUST_MIX), dr.random(m))  # fine, medium, flake (DUST_MIX)
        self.d_r0 = np.choose(cls, [dr.uniform(*r, m) for r in SHELL_R])
        self.d_bright = np.choose(cls, [dr.uniform(*s, m) for s in SHELL_SEEN]) / SHELL_LEVELS
        self.d_dark = dr.random(m) < GRIT  # grit darker than what is behind it; the rest lint, lighter
        q = ROOM_DUST  # the room's dust, drawn in and round as the ball feeds on the air, from just outside it (from
        # 2.7 radii it swept past the frame's edges): medium specks
        self.r_rho = dr.uniform(1.05, 1.2, q)
        self.r_phase = dr.uniform(0, 2 * np.pi, q)
        self.r_lat = dr.normal(0, 0.2, q)
        self.r_bright = dr.uniform(0.6, 1.0, q) * SHELL_SEEN[1][1] / SHELL_LEVELS
        self.r_r0 = dr.uniform(*SHELL_R[1], q)
        self.r_dark = dr.random(q) < GRIT
        self.cut = dr.uniform(LONG[2], 1.0, m + q)  # (LONG)
        self.noise = _tile_noise(rng, 256, 256, 6.0, 6.0, 3)  # the co-rotating shimmer, and the caustics
        self.mist = _tile_noise(rng, 160, 640, 16.0, 1.8)  # lat x lon: long along the flow, thin across it
        self.arcs = _tile_noise(rng, 1, 512, 18.0, 0.5).reshape(1, -1)  # where round the rim its band is strong ...
        self.arc_cut = float(np.quantile(self.arcs, 1 - BALL_SHADOW[1]))  # ... over this (BALL_SHADOW's share of it)
        gr = fx._rng(seed, _BALL, salt, _DRAFT)  # the air it drags round it (DRAFT_LOBES): each lobe's phase and pace,
        # and the drag's own lobes
        nlb = len(DRAFT_LOBES[0])
        self.l_ph, self.l_pace = gr.uniform(0, 2 * np.pi, (2, nlb)), gr.uniform(*DRAFT_LOBES[2], (2, nlb))
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
        """The shell's density on the near (sign +1) or far (-1) side, fixed in the spinning air, long along the flow
        and thin across it; ``sx``, ``sy`` in radii from the middle."""
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
        return dens * polar * limb

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

    def points(self, c, R, B, form, rel, vis, omega, k):
        """The streaks the ball carries, as points: (x, y, depth px towards the lens, weight, kind) with kind 0 a fine
        streak, 1 a broad wisp, 2 and 3 the tails of a fine streak and of a broad wisp (drawn thinner:
        STREAK_TAPER). No streak is brighter than STREAK_PEAK times the brighter half's mean; each streak is
        brightest on the near side (BALL_NEAR)."""
        spread = 1 + 1.4 * (1 - form) ** 2  # forming: it gathers in from wide
        u, v, w_ = self.u, self.v, self.w
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
        d2 = np.concatenate([d2, d2[:, -1:]], 1)  # ... and covers in the picture: where a streak turns towards the
        # lens its points bunch up, and weighed by their length in the room they piled into a white crescent; a
        # little of that is kept (STREAK_END), as a filament seen end on does gather more light
        g = self.bright * (0.5 + 0.5 * np.sin(self.glow + self.glow_f * self.spin) ** 2)
        g = np.minimum(g, STREAK_PEAK * float(g[g >= np.median(g)].mean()))
        wt = g[:, None] * tail * vis * d2 * np.clip(ds / np.maximum(d2, 1e-3), 1.0, STREAK_END[0]) ** STREAK_END[1]
        kind = (self.soft[:, None].astype(np.int8) + 2 * (s > STREAK_TAPER[0])).astype(np.int8)
        x, y, z, wgt, kd_ = pts[..., 0].ravel(), pts[..., 1].ravel(), pts[..., 2].ravel(), wt.ravel(), kind.ravel()
        wgt = wgt * (BALL_BACK + (1 - BALL_BACK) * _smooth((z / R + BALL_NEAR) / (2 * BALL_NEAR)))
        return c[0] + x, c[1] + y, z, wgt, kd_

    def specks(self, R, B, form, rel, vis, omega, t, loud):
        """The dust the ball carries, as specks for _specks: (where each is as the shutter opens and now, (n, 2) px off
        the ball's middle; depth px towards the lens; weight; size px at 1080; dark): dust whirled round in its shell
        (SHELL), each speck at its own pace (DUST_SPIN, DUST_SURGE) and swaying in latitude, catching the light a little
        as it turns, showing more with the voice (``loud``: DUST_LOUD) and flung out as the ball lets go (``rel``); and
        the room's dust drawn in, spiralling slowly in round the equator."""
        spread = 1 + 1.4 * (1 - form) ** 2
        prog = (self.r_phase * 0.37 + 0.02 * self.spin) % 1.0
        rr2 = R * (1.0 + (self.r_rho - 1.0) * (1 - prog))
        rr = self.d_rho * R * spread * (1 + 0.8 * rel)

        def at(spin):
            la = self.d_lat + 0.12 * np.sin(0.35 * self.d_turn * spin + self.d_sway)
            ph = self.d_phase + self.d_turn * spin + DUST_SURGE[0] * np.sin(self.d_surge * spin + self.d_surge_ph)
            ph2 = self.r_phase + 0.55 * spin * R / rr2
            return np.concatenate([
                (rr * np.cos(la))[:, None] * (np.cos(ph)[:, None] * B[0] + np.sin(ph)[:, None] * B[1])
                + (rr * np.sin(la))[:, None] * B[2],
                rr2[:, None] * (np.cos(ph2)[:, None] * B[0] + np.sin(ph2)[:, None] * B[1])
                + (self.r_lat * R)[:, None] * B[2]])

        was, now = at(self.spin - SHUTTER * omega), at(self.spin)
        glint = 0.9 + 0.1 * np.cos(self.d_glint + 0.3 * t) ** 2
        wgt = np.concatenate([self.d_bright * glint * (DUST_LOUD[0] + DUST_LOUD[1] * loud),
                              self.r_bright * np.sin(np.pi * prog) * (1 - rel)]) * vis
        r0 = np.concatenate([self.d_r0, self.r_r0])
        dark = np.concatenate([self.d_dark, self.r_dark])
        return was[:, :2], now[:, :2], now[:, 2], wgt, r0, dark

    def haze(self, R, B, form, rel, omega, step):
        """The haze of fine dust each train carries (HAZE), as points ``step`` px apart along its fine lines, each
        smeared over the shutter's share of ``omega``: (x, y px off the ball's middle; depth px towards the lens;
        weight, a line's own, broken along it, fading in and out at its ends and thinner where the eye looks through
        the shell than at its limb, per px of it; whether its line's dust is lit, HAZE_LIT), where its specks
        are."""
        spread = 1 + 1.4 * (1 - form) ** 2
        rr = (self.t_rho[:, None] + self.h_rho) * R * spread * (1 + 0.8 * rel)  # (train, line)
        span = self.t_len + SHUTTER * omega * self.t_turn  # rad: the train and its motion while the shutter is open
        S = int(np.clip(np.ceil((span[:, None] * rr).max() / step), 4, 600))
        s = np.linspace(0, 1, S)
        ph = (self.t_phase + self.t_turn * self.spin + self.t_len - span)[:, None, None] + s * span[:, None, None]
        la = (self.t_lat + 0.12 * np.sin(0.35 * self.t_turn * self.spin + self.t_sway))[:, None, None] + \
            self.h_lat[..., None]
        p = rr[..., None, None] * (np.cos(la)[..., None] * (np.cos(ph)[..., None] * B[0]
                                                            + np.sin(ph)[..., None] * B[1])
                                   + np.sin(la)[..., None] * B[2])  # (train, line, point, xyz)
        ds = (span[:, None] * rr / (S - 1))[..., None]  # (px of it each point stands for)
        limb = HAZE[2] + (1 - HAZE[2]) * (1 - np.abs(p[..., 2]) / rr[..., None]) ** 2
        broken = np.clip(0.45 + 0.75 * np.sin(2 * np.pi * self.h_gap[..., None] * s + self.h_gph[..., None]), 0, 1)
        wt = self.h_w[..., None] * np.sin(np.pi * s) ** 0.6 * broken * limb * ds
        lit = np.broadcast_to(self.h_lit[..., None], wt.shape)
        return p[..., 0].ravel(), p[..., 1].ravel(), p[..., 2].ravel(), wt.ravel(), lit.ravel()

    def field(self, dx: np.ndarray, dy: np.ndarray, R: float, vis: float, t: int, turn: float,
              keep: np.ndarray | None, k: float) -> tuple[np.ndarray, ...]:
        """The ball's bend at points ``dx`` (1, m), ``dy`` (n, 1) px off its middle, on a grid FIELD_Q px apart:
        radius ``R`` px, ``vis`` how much of itself it is, ``turn`` the sense its near side crosses the picture in,
        ``keep`` how much of each point is over the hand (None: no hand; over it the steady lens is left out, and the
        twist and shimmer turning with the spin are cut, BALL_HAND). Returns (the px each point reads the plate from,
        x and y, off itself; r, its distance from the middle in radii of the wobbling outline; phi, its angle; wob;
        the co-rotating noise, 3 channels). The twist and the squeeze are continuous, the rim included, and eased over
        BALL_EASE (the shell's chord rises steeply at its rim and round its core)."""
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
        if keep is not None:  # over the hand only the lobes turning with the spin: the steady twist dragged the
            # fingers seen through the ball a steady 1.5-2 px one way
            lobes = lobes - 0.6 * keep
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
        ct, st = np.cos(-theta), np.sin(-theta)  # read from behind the flow: the plate is dragged with the spin
        tx, ty = dx * ct - dy * st - dx, dx * st + dy * ct - dy
        cs, sn = math.cos(-turn * self.spin), math.sin(-turn * self.spin)
        u = (dx * cs - dy * sn) / (0.5 * R) * 20 + 128
        v = (dx * sn + dy * cs) / (0.5 * R) * 20 + 128
        nz = cv2.remap(self.noise, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_WRAP)
        tb = BALL_TURB * k * cv2.GaussianBlur(sw.astype(np.float32), (0, 0), ease)
        lens, swirl = 1.0, 1.0
        if keep is not None:
            lens, swirl, tb = 1 - keep, 1 - BALL_HAND[0] * keep, tb * (1 - BALL_HAND[1] * keep)
        return (-m * dx * lens + tx * swirl + nz[..., 0] * tb, -m * dy * lens + ty * swirl + nz[..., 1] * tb, r, phi,
                wob, nz)

    def stir(self, frame: np.ndarray, c: np.ndarray, R: float, vis: float, gather: tuple[float, float] | None,
             hand: np.ndarray | None, k: float) -> None:
        """In place on ``frame`` (uint8 RGB, the whole frame as displayed: this is a bend alone): what is seen through
        the room's air round the ball (middle ``c``, radius ``R`` px, ``vis`` how much of it there is), which it drags
        round with it (DRAFT): pushed out and in by lobes turning with its spin (DRAFT_LOBES) and dragged round the
        way its near side goes, from its rim out to DRAFT's last radii; and as it is born the room's air drawn into it
        (GATHER; ``gather``: the converging ring's radius and pull, px, or None). None of it over the hand's matte
        ``hand`` (uint8 (h, w), or None), grown and softened as the blast's front leaves it (FRONT_HAND), so nothing
        next to the hand is read from the hand."""
        h, w = frame.shape[:2]
        if vis <= 1e-3 and gather is None:
            return
        cx, cy = float(c[0]), float(c[1])
        reach = DRAFT[2][1] * R if gather is None else max(DRAFT[2][1] * R, gather[0] + 3 * GATHER[3] * k)
        box = _clip_box(cx, cy, reach, reach, w, h)
        if box is None:
            return
        y0, y1, x0, x1 = box
        q = 4  # (it is smooth: worked out on a grid this many px apart)
        gx, gy = _coarse_grid(box, q)
        dx, dy = gx - cx, gy - cy
        dist = np.sqrt(dx * dx + dy * dy) + np.float32(1e-3)
        nx, ny = dx / dist, dy / dist
        phi, r = np.arctan2(dy, dx), dist / np.float32(R)
        turn = 1.0 if self.axes is None else _turn_sense(self.axes)
        m_, a_ = DRAFT_LOBES[0], DRAFT_LOBES[1]
        rms = math.sqrt(sum(a * a for a in a_) / 2)
        lobes = [sum(a * np.sin(m * (phi - turn * self.spin * self.l_pace[j, i]) + self.l_ph[j, i])
                     for i, (m, a) in enumerate(zip(m_, a_, strict=True))) / rms for j in range(2)]  # (unit spread)
        env = _smooth((r - DRAFT[1][0]) / (DRAFT[1][1] - DRAFT[1][0])) * _smooth(
            (DRAFT[2][1] - r) / (DRAFT[2][1] - DRAFT[2][0]))
        amp = np.float32(DRAFT[0] * k * vis) * env
        radial = amp * lobes[0]
        across = amp * np.float32(DRAFT[3] * turn) * (0.5 + 0.5 * lobes[1])
        if gather is not None:  # (pulled towards the ball: read from further out)
            radial = radial - np.float32(gather[1]) * np.exp(-0.5 * ((dist - gather[0]) / (GATHER[3] * k)) ** 2)
        if hand is not None:
            hc = cv2.resize(hand[y0:y1, x0:x1].astype(np.float32) * np.float32(1 / 255), (gx.shape[1], gy.shape[0]),
                            interpolation=cv2.INTER_AREA)
            g_ = max(1, round(FRONT_HAND[0] * k / q))
            hc = cv2.dilate(hc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g_ + 1, 2 * g_ + 1)))
            off = 1 - np.clip(cv2.GaussianBlur(hc, (0, 0), max(0.5, FRONT_HAND[1] * k / q)), 0, 1)
            radial, across = radial * off, across * off
        fx_, fy_ = _tame(-nx * radial + ny * across, -ny * radial - nx * across, q)
        if float(np.abs(fx_).max() + np.abs(fy_).max()) < 0.05:
            return
        bh, bw = y1 - y0, x1 - x0
        mx = np.arange(x0, x1, dtype=np.float32)[None, :] + _upsample(fx_.astype(np.float32), bh, bw)
        my = np.arange(y0, y1, dtype=np.float32)[:, None] + _upsample(fy_.astype(np.float32), bh, bw)
        frame[y0:y1, x0:x1] = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def draw(self, P: np.ndarray, off: tuple[int, int], c: np.ndarray, R: float, a: np.ndarray, form: float,
             rel: float, pulse: float, flash: float, idle: float, omega: float, loud: float, t: int,
             hand: np.ndarray | None, noise: np.ndarray | None, code: float, k: float,
             tone: np.ndarray | None = None) -> np.ndarray | None:
        """Draws the ball on ``P`` (a linear crop of the frame whose top-left is ``off`` = (x, y) in frame px) in
        place: centre ``c`` (frame px), radius ``R`` (px), spin axis ``a``; ``form`` 0..1 how gathered, ``rel`` 0..1
        how far it has let go, ``pulse`` how far a comeback hit has burst it (PULSE_BURST at the most) and ``flash``
        0..1 how much its shell flashes with it (PULSE_FLASH), ``idle`` how much of itself it is between two sounds,
        ``omega`` its spin this frame (rad), ``loud`` the voice's level, ``hand`` the matte over the crop (0..1;
        softened by HAND_SOFT, and bent as the hand seen through the ball is), ``tone`` the room's lit tone (linear, or
        None: the light its lint in front of the hand catches, NEAR_LIT); the hand's edge redrawn last (_reedge).
        First its bend: the lens and twist (field) and the swirling air's density (BALL_DENS: its shell, its middle's
        puff and its streaks, smoothed) bending what is behind towards where it is denser, over the hand HAND_BEND at
        the most, eased off by BALL_REACH and never steeper than BEND_SLOPE; then the light it gathers and spreads:
        that density's fine shadowgraph, a band round its rim and the sheen of its streaks, as much darker as lighter
        (SHEEN); then the dust it whirls round: its trains' haze (HAZE), and specks lighter and darker than what is
        behind them (_specks), the near half's in front of the hand (_in_front). Returns the crop as bent, before its
        dust was laid on it (None when it drew nothing)."""
        self.spin += omega
        B = self.turn_axes(a)
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
        turn = _turn_sense(B)  # (the plate behind is dragged the way its near side's air crosses the picture)

        # ---- the streaks it carries, splatted: filaments of denser air in its bend and its sheen; the far side's
        # behind the fingers, and over the skin only broad and barely there
        x, y, z, wgt, kind = self.points(c, R, B, form, spread, vis, omega, k)
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

        # ---- the bend and the light it gathers: on a coarse grid round the ball
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
            far = self._mist(sx, sy, -1, B)
            if hc is not None:  # the far side of the shell passes behind the fingers
                far = far * (1 - 0.9 * hc)
            dens = (BALL_DENS[0] * self._mist(sx, sy, 1, B) + BALL_DENS[1] * far) * edge \
                + BALL_DENS[2] * self._core(sx, sy, B) + BALL_DENS[3] * cv2.resize(
                    lit[y0:y1, x0:x1], (gx.shape[1], gy.shape[0]), interpolation=cv2.INTER_AREA)
            sig = BALL_REFRACT[1] * k / FIELD_Q
            dens = cv2.GaussianBlur(dens.astype(np.float32), (0, 0), sig) * np.float32(
                vis * (1 + PULSE_FLASH[3] * flash))
            ddy, ddx = np.gradient(dens)
            if keep is not None:  # over the hand only what turns with the spin bends it: the density's broad build
                # (the shell's limb, the puff's dome, steady as it spins) left out, so the fingers ripple about where
                # they are
                by, bx = np.gradient(cv2.GaussianBlur(dens, (0, 0), HAND_STEADY * R / FIELD_Q))
                ddx, ddy = ddx - keep * bx, ddy - keep * by
            pull = np.float32(BALL_REFRACT[0] * k * sig * math.sqrt(math.e)) * bend  # (a bump of density 1 as wide
            # as the smoothing bends what is behind it BALL_REFRACT's first at its steepest)
            fx_, fy_ = fx_ + pull * ddx, fy_ + pull * ddy  # (read from up the density: a bump of it magnifies what is
            # behind it, as the lens does)
            if keep is not None:  # over the hand, a ripple of HAND_BEND at the most, about where the fingers are: its
                # mean over the hand inside the ball taken out each frame (left in, the fingers sat a steady 1-2 px one
                # way)
                cap = HAND_BEND * k / np.maximum(keep, 1e-3)
                on = keep * keep * (r < 1.0)
                for capped in (False, True):  # (the cap moves the mean a little: taken out again after it)
                    if capped:
                        fx_, fy_ = (f * np.minimum(1, cap / np.maximum(np.hypot(fx_, fy_), 1e-6)) for f in (fx_, fy_))
                    if float(on.sum()) > 1.0:
                        fx_, fy_ = (f - keep * np.float32(float((f * on).sum()) / float(on.sum())) for f in (fx_, fy_))
            reach = _smooth((BALL_REACH[1] - np.hypot(dx, dy) / R) / (BALL_REACH[1] - BALL_REACH[0]))
            fx_, fy_ = _tame(fx_ * reach, fy_ * reach, FIELD_Q)
            ca = BALL_CA if keep is None else _upsample((BALL_CA * (1 - keep)).astype(np.float32), bh, bw)
            DX, DY = _upsample(fx_.astype(np.float32), bh, bw), _upsample(fy_.astype(np.float32), bh, bw)
            _remap_rgb(P, DX, DY, box, ca)
            if matte is not None:  # the matte bent with the hand seen through the ball
                matte = matte.copy()
                matte[y0:y1, x0:x1] = cv2.remap(matte[y0:y1, x0:x1], np.arange(bw, dtype=np.float32)[None, :] + DX,
                                                np.arange(bh, dtype=np.float32)[:, None] + DY, cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_REPLICATE)
                hand = cv2.GaussianBlur(matte, (0, 0), HAND_SOFT * k)
            # the shadowgraph: ripples where the density's detail curves (light gathered where it is denser,
            # RIPPLE_FINE), and one soft band on the rim, broken into arcs that turn with the spin and wander in and
            # out with them; over the hand eased off as the bend is (its bright arc across a lit palm read as glare on
            # the skin)
            ripple = np.clip(-cv2.Laplacian(dens - cv2.GaussianBlur(dens, (0, 0), RIPPLE_FINE * R / FIELD_Q),
                                            cv2.CV_32F) * np.float32(sig * sig / 2), -1.5, 1.5)
            arc = cv2.remap(self.arcs, (((phi - turn * self.spin * 0.6) % (2 * np.pi)) * (512 / (2 * np.pi)))
                            .astype(np.float32), np.zeros_like(phi, np.float32), cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_WRAP)
            band = np.exp(-((r - 1.0 - 0.04 * arc) / BALL_BAND) ** 2) * _smooth(
                (arc - self.arc_cut) / BALL_SHADOW[2] + 0.5)
            gain_ = (BALL_SHADOW[0] * vis * band + BALL_RIPPLE * ripple) * bend
            if hc is not None:  # the palm under it takes the light the spinning air focuses, in moving patches
                gain_ = gain_ + BALL_CAUSTIC * vis * hc * (np.clip(nz[..., 2], 0, None) ** 2 - 0.4) * \
                    _smooth((1.15 - r) / 0.3)
            # the streaks' sheen: brighter along each, as much darker round it; over the skin barely there
            sl = lit[y0:y1, x0:x1] if hand is None else np.minimum(
                lit[y0:y1, x0:x1], 1 - (1 - STREAK_SKIN[0]) * hand[y0:y1, x0:x1])
            g = _upsample(gain_.astype(np.float32), bh, bw) + np.float32(SHEEN[0]) * (
                sl - cv2.GaussianBlur(sl, (0, 0), SHEEN[1] * k))
            Q = P[y0:y1, x0:x1]
            cap = np.float32(SHEEN[2] / 255) * _STEP_LUT[(np.sqrt(np.clip(Q @ fx._LUMA, 0.0, 1.0))
                                                          * np.float32(1023)).astype(np.int32)]  # (the table's steps
            # are 255ths: read as whole ones, the cap never held)
            Q *= 1 + np.clip(g, -cap, cap)[..., None]
        bent = P.copy()  # (bent and lit, before the dust it carries is laid on)
        behind = _behind(P)
        before_hand = None if hand is None else (_unhanded(behind, hand), hand, tone)  # (its near half in front of
        # the hand)
        sc = 4

        # ---- the haze of fine dust its trains carry: lighter or darker than what is behind it, in its hue; the far
        # side's behind the hand, the near side's in front of it, fainter (NEAR_SKIN)
        hq = 2  # (its fine lines drawn at half size)
        hx, hy, hz, hw, hl = self.haze(R, B, form, spread, omega, 0.5 * hq)
        hx, hy = hx + cx, hy + cy
        hw = hw * vis * (DUST_LOUD[0] + DUST_LOUD[1] * loud) * np.where(hz < 0, DUST_BACK, 1.0)
        if hand is not None:
            hw = hw * (1 - hand[np.clip(hy.astype(np.int64), 0, h - 1), np.clip(hx.astype(np.int64), 0, w - 1)] * (
                1 - NEAR_SKIN[4] * _smooth(hz / (0.3 * R))))
        for lit_, levels in ((False, -HAZE[0]), (True, HAZE_LIT[1] * HAZE[0])):
            hb = np.zeros((-(-h // hq), -(-w // hq)), np.float32)
            _splat(hb, (hx[hl == lit_] + 0.5) / hq - 0.5, (hy[hl == lit_] + 0.5) / hq - 0.5, hw[hl == lit_] / hq)
            _lay_haze(P, _line_blur(hb, HAZE[1] * k / hq), hq, behind, levels, front=before_hand)

        # ---- the dust it whirls round: the near half's bigger, softer and fainter, the far half's smaller and sharp,
        # the outer third's stronger (LIMB), and no more than LONG's second smeared longer than its first; over the
        # hand the near half's moving specks in front of the fingers (NEAR_SKIN), the far half's behind them
        was, now, zp, wgt, r0, dark = self.specks(R, B, form, spread, vis, omega, t, loud)
        near, far = _smooth(zp / (0.6 * R)), _smooth(-zp / (0.6 * R))
        r0 = r0 * (1 + DUST_NEAR[0] * near - DUST_NEAR[3] * far)
        wgt = wgt * (DUST_BACK + (1 - DUST_BACK) * _smooth((zp / R + BALL_NEAR) / (2 * BALL_NEAR))) * (
            1 - DUST_NEAR[2] * near) * (1 + LIMB[0] * _smooth((np.hypot(*now.T) / R - LIMB[1]) / LIMB[2]))
        was, now = was + (cx, cy), now + (cx, cy)
        smear = np.hypot(*(now - was).T)
        long_ = np.argsort(-smear)[int(LONG[1] * len(smear)):]
        most = LONG[0] * k * self.cut[long_]
        cut, most = long_[smear[long_] > most], most[smear[long_] > most]
        was[cut] = now[cut] + (was[cut] - now[cut]) * (most / smear[cut])[:, None]
        if hand is not None:
            xi, yi = np.clip(now[:, 0].astype(np.int64), 0, w - 1), np.clip(now[:, 1].astype(np.int64), 0, h - 1)
            moving = _smooth((np.hypot(*(now - was).T) / k - NEAR_SKIN[2]) / (NEAR_SKIN[3] - NEAR_SKIN[2]))
            front = _smooth(zp / (0.3 * R)) * (NEAR_SKIN[0] + (NEAR_SKIN[1] - NEAR_SKIN[0]) * moving) * np.where(
                dark, NEAR_SKIN[5], 1.0)
            wgt = wgt * (1 - hand[yi, xi] * (1 - front))
        lit_s, drk_s = _speck_bufs(h, w, sc)
        _specks(was, now, np.clip(1 - zp / (FOCAL * 1080 * k), 0.5, 1.5), wgt, r0, dark, lit_s, drk_s, sc, k,
                soft=DUST_NEAR[1] * near)
        _lay_specks(P, lit_s, drk_s, sc, None if noise is None else (lambda y0, y1, x0, x1: noise[y0:y1, x0:x1]),
                    code, k, behind=behind, levels=SHELL_LEVELS, front=before_hand)
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
        held, idle, shift_before = 0, 1.0, 0.0
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
            omega = 2 * np.pi * BALL_SPIN * (0.5 + 0.5 * form + BALL_SPIN_VOICE * e + BALL_SPIN_HELD * grow) / fx.FPS \
                * hit_spin(t, tl.hits)
            if t == t_rel or t in pulses:  # letting go, or a pulse: the ball bursts into a push of air, its ring as
                # wide as the ball
                u, q = push_direction(track, t, normal[t], tl.aim, tl.tap_frame, n, w, h)
                gain = RELEASE_GAIN if t == t_rel else PULSE_GAIN
                blasts.append(_Blast(c, u, q, normal[t], size[t], w, h, t, gain[0] + gain[1] * e, seed,
                                     j if t == t_rel else 1000 + 64 * j + pulses.index(t), ring=R0, palm=xy[t],
                                     cloud_in=CLOUD_IN if t == t_rel else 0))
            pad = BALL_PAD * R * (1 + rel) + 8 * k
            box = _clip_box(c[0], c[1], pad, pad, w, h)
            done += 1
            if box is None:
                ball.spin += omega
                continue
            matte = None if mask is None else mask[min(t, len(mask) - 1)]
            vis = BALL_IDLE + (1 - BALL_IDLE) * idle
            kf = t - t0
            ball.stir(out[t], c, R, float(form * (1 - rel) ** 1.5 * vis * (1 - burst)), None if kf >= GATHER[0] else (
                R0 * (GATHER[1][0] + (GATHER[1][1] - GATHER[1][0]) * kf / max(1, GATHER[0] - 1)),
                GATHER[2] * k * (1 - 0.4 * kf / max(1, GATHER[0] - 1))), matte, k)  # (the room's air round it; burst,
            # its air is flung out thin, and the push it throws bends the room)
            y0, y1, x0, x1 = box
            P = lin(out[t, y0:y1, x0:x1])  # (over the ball before it, still bursting)
            hand = None if matte is None else matte[y0:y1, x0:x1].astype(np.float32) / 255.0
            noise = _grain(seed, _BALL, t, y1 - y0, x1 - x0)
            bent = ball.draw(P, (x0, y0), c, R, spin_axis(normal[t]), form, rel, burst, flash, vis, omega, e, t, hand,
                             noise, code, k, None if hand is None else _tone(out[t]))
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
        _push_frames(plate, out, blasts, mask, tl, seed, code,
                     lambda stage, share, msg: say(stage, 0.5 + 0.5 * share, msg))


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
    whose plane hides what is behind it, its dusty air coming in over its first ``cloud_in`` frames; out of a palm, a
    share of its dust thrown off the hand's ``outline`` (points (n, 2) frame px just outside it, or None). All of it
    lives in the room as the start frame saw it: each frame is drawn through the camera's motion since. Points in it
    are 3D: x, y in px at the palm's depth, z in palm depths (the lens at 0)."""

    def __init__(self, origin, u, q: float, normal, size: float, w: int, h: int, start: int, gain: float, seed: int,
                 salt: int, ring: float | None = None, palm=None, cloud_in: int = CLOUD_IN, outline=None):
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
        self.run = FRONT_SPEED[0] * self.k * (1 + FRONT_SPEED[3] * (1 - self.q))  # px a frame the front leaves at ...
        self.r_from = side * FRONT_FROM if ring is None else max(side * FRONT_FROM, float(ring))  # ... from this far
        # out (a ball's from its shell: from its middle, it trailed the ring the ball burst into)
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
            """Points leaving the palm's face (a hand ``spread`` across, just off its skin; a ball's shell, ``ring``
            px round its middle) along unit directions ``d`` (x, y and towards the lens) at ``sp`` px a frame: their
            positions and velocities."""
            a, b = rng.standard_normal((2, len(d))) * spread * size
            off = a[:, None] * e1 + b[:, None] * e2 + 0.08 * size * self.nrm if ring is None else float(ring) * d
            p_ = np.column_stack([self.o[0] + off[:, 0], self.o[1] + off[:, 1], 1 - off[:, 2] / f])
            return p_, np.column_stack([d[:, 0] * sp, d[:, 1] * sp, -d[:, 2] * sp / f])  # (the lens: z falls)

        def ahead(d: np.ndarray) -> np.ndarray:
            """Unit directions spread round the push's own from random vectors ``d``, none of them back into it,
            flattened towards the picture (DUST_AIM)."""
            d = d / np.linalg.norm(d, axis=-1, keepdims=True)
            d = np.array([self.u[0], self.u[1], 0.0]) * DUST_AIM[0] + n3 * 0.3 + d
            d[np.sum(d * n3, -1) < 0] *= -1
            d[:, 2] *= DUST_AIM[2]
            return d / np.linalg.norm(d, axis=-1, keepdims=True)

        # dust and flakes following the front: fine specks, medium ones and tumbling flakes (DEBRIS), a share of it
        # fine dust drifting slowly and hanging on (DRIFT). Out of a palm, a body of it thrown off the hand's outline
        # (most from its side the push goes out of; out of the palm without one) in a cone round the push's way (BODY,
        # all round for a true push at the lens); out of a ball's burst, flung all round out of its shell (BURST_N)
        specks, flakes = (DUST_N, FLAKES) if ring is None else BURST_N
        N = specks + flakes
        delay = rng.uniform(0, 2.5, N).astype(int)
        flake = np.arange(N) >= specks
        kind = np.where(flake, 2, (rng.random(N) >= DEBRIS[0]).astype(int))  # fine, medium, flake
        drift = ~flake & (rng.random(N) < DRIFT[0])
        kind = np.where(drift, 3, kind)
        r0 = np.choose(kind, [rng.uniform(*r, N) for r in DEBRIS_R])
        bright = np.choose(kind, [rng.uniform(*s, N) for s in DEBRIS_SEEN])
        dark = rng.random(N) < GRIT
        hang = np.where(drift, DUST_LIFE[0] + DRIFT[2], DUST_LIFE[0])  # frames each hangs before settling out
        if ring is None:
            sp = rng.uniform(*BODY[2], N) * self.run * gain  # px a frame at the palm's depth
            half = BODY[1] + (np.pi - BODY[1]) * float(_smooth((LENS_Q[0] - self.q) / LENS_Q[0]))
            ang = math.atan2(self.u[1], self.u[0]) + np.clip(rng.normal(0, BODY[0] * half / BODY[1], N), -half, half)
            d = np.column_stack([np.cos(ang), np.sin(ang), DUST_AIM[2] * n3[2] * rng.uniform(-0.3, 1.0, N)])
            p, v = thrown(d / np.linalg.norm(d, axis=1, keepdims=True), sp, DUST_AIM[1])
            if outline is not None and len(outline):
                rim = rng.random(N) < DEBRIS[1]
                rel = outline - self.o
                pick = np.clip((rel / np.maximum(np.hypot(*rel.T), 1e-6)[:, None]) @ self.u, 0, 1) ** 2 + 0.05
                at_ = rng.choice(len(outline), int(rim.sum()), p=pick / pick.sum())
                p[rim] = np.column_stack([outline[at_], np.ones(len(at_))])
        else:
            sp = rng.uniform(0.35, 1.0, N) ** 1.3 * DUST_FOLLOW * self.run * gain
            p, v = thrown(ahead(rng.standard_normal((N, 3))), sp, DUST_AIM[1])
        v[drift] *= DRIFT[1]
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
        self.hang = np.concatenate([hang, np.zeros(C)])
        self.cloud = np.arange(N + C) >= N
        self.tum = rng.uniform(0, 2 * np.pi, N + C)
        self.tumf = rng.uniform(0.25, 0.8, N + C)
        self.alive = np.ones(N + C, bool)
        self.room_hit: np.ndarray | None = None  # which of the room's dust this push has thrown (_Room.throw)
        # dust swept past the lens (MOTES), from round the palm out past the frame's edge, its own stream of the seed
        mr = fx._rng(seed, _BLAST, salt, _LENS)
        nm = int(round(MOTES[0] + (MOTES[1] - MOTES[0]) * min(1.0, gain)))
        off = mr.standard_normal((nm, 2)) * MOTE_FROM * side
        d = off / np.maximum(np.hypot(*off.T), 1e-6)[:, None] + self.q * self.u
        d /= np.maximum(np.hypot(*d.T), 1e-6)[:, None]
        at0 = self.o + off
        with np.errstate(divide="ignore", invalid="ignore"):  # how far each has to go to leave the frame
            reach = np.min(np.where(d != 0, np.maximum((np.where(d > 0, [w, h], 0.0) - at0) / d, 0), np.inf), axis=1)
        self.m_d = mr.uniform(*MOTE_D, nm) * self.k
        self.m_t0 = mr.integers(MOTE_RUN[0], MOTE_RUN[1] + 1, nm)
        run = mr.uniform(MOTE_RUN[2], MOTE_RUN[3], nm)
        run[:MOTE_FAST[0]] = mr.uniform(MOTE_FAST[1], MOTE_FAST[2], min(nm, MOTE_FAST[0]))
        self.m_xy, self.m_v = at0, d * ((reach + 0.6 * self.m_d) / run)[:, None]
        self.m_run = run
        self.m_lev = mr.uniform(*MOTE_LEVELS, nm) * min(1.0, 0.5 + 0.5 * gain)

    def ring(self, kf: float) -> np.ndarray:
        """The dusty air's parcels on the push's ``kf``-th frame, as self.p holds them: on a torus round the ring's
        axis that travels out along it, widens and slows, its air rolling round its core; the wake's parcels lag back
        towards the palm, narrower and looser; the room's eddies push it about as it slows; out of a palm it rolls up
        from RING_FORM of its width as its air comes in (CLOUD_IN). Coming at the lens it swells as it nears it
        (kept to a share of its travel there, it sat on a palm turned to the lens and fogged it), drifting to the
        frame's middle as it comes (LENS_MID)."""
        go = 1 - math.exp(-kf / RING_SLOW)
        travel = self.speed * self.side * self.gain * RING_SLOW * go
        rolled = RING_FORM + (1 - RING_FORM) * float(_smooth(kf / self.cloud_in)) if self.cloud_in > 0 else 1.0
        radius = self.ring0 * (1 + (RING_R[1] - 1) * go) * rolled
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
        """The front on its ``kf``-th frame: (radius px, bulge px, share of its run). It leaves the palm (a ball's
        shell) at FRONT_SPEED and slows as a blast's front does, and fades as it goes."""
        uu = min(1.0, kf / BLAST_FRAMES)
        slow, tau = FRONT_SPEED[1], FRONT_SPEED[2]
        radius = self.r_from + self.run * ((1 - slow) * kf + slow * tau * (1 - math.exp(-kf / tau)))
        strength = BUBBLE_PX * self.k * max(self.gain, FRONT_LEAST) * min(1.0, (kf + 1) / 2) * (
            1 - float(_smooth((uu - BUBBLE_HOLD) / (1 - BUBBLE_HOLD))))
        return radius, strength, uu

    def knock(self) -> int:
        """Frames after the push it knocks the phone: KNOCK_AT's first for a push across the picture, its second for
        one straight at the lens, between them as it leans."""
        return round(KNOCK_AT[0] + (KNOCK_AT[1] - KNOCK_AT[0]) * (1 - self.q))

    def last_frame(self) -> int:
        """The last frame anything it throws shows on (the room's dust its front reaches as it ends included)."""
        return self.start + BLAST_FRAMES + DUST_LIFE[0] + DUST_LIFE[1]

    def motes(self, kf: int) -> np.ndarray:
        """The dust it sweeps past the lens on its ``kf``-th frame (MOTES), those on screen: rows of (x, y as the
        shutter opens, x, y now: frame px; softness px; grey levels lighter than what is behind it), each growing as
        it nears the lens."""
        age = kf - self.m_t0
        live = (age >= 0) & (age <= self.m_run + 1)
        if not live.any():
            return np.zeros((0, 6))
        a = age[live].astype(np.float64)
        now = self.m_xy[live] + self.m_v[live] * a[:, None]
        was = now - self.m_v[live] * SHUTTER
        soft = self.m_d[live] / 3 * (1 + (MOTE_GROW - 1) * np.minimum(a / self.m_run[live], 1.2))
        return np.column_stack([was, now, soft, self.m_lev[live] * np.minimum(1.0, (a + 1) / 1.5)])

    def behind(self, p: np.ndarray) -> np.ndarray:
        """Which points are behind the plane of the palm (so behind the hand, where its matte is)."""
        return (p[:, 0] - self.palm[0]) * self.nrm[0] + (p[:, 1] - self.palm[1]) * self.nrm[1] - (
            p[:, 2] - 1.0) * self.f * self.nrm[2] < 0

    def bend(self, P: np.ndarray, kf: int, o: np.ndarray, u: np.ndarray, zoom: float,
             hand: np.ndarray | None = None) -> None:
        """The pressure bubble on frame ``kf`` of the push (``o``, ``u``, ``zoom``: its origin, aim and scale as the
        camera sees them now): what is seen through it pushed outward, most in a band of bend at its front sweeping the
        room, which is a shadowgraph, the air squeezed darker just behind it and thinned lighter behind that
        (COMPRESS), and behind it a turbulent wake, in place; none of it over ``hand`` (0..1: the hand's matte grown and
        softened, FRONT_HAND), so it runs out from round the palm at once."""
        h, w = P.shape[:2]
        radius, strength, uu = self.front(kf)
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
                    0.61 * strength / BEND_SLOPE)  # (a Gaussian's sigma: its band at half height is 2.355 of it, and
        # its steepest slope 0.61 of its strength over it)
        zz = (dist - rr) / width
        bulge = strength * cone * (BUBBLE_INNER * np.clip(x, 0, 1) ** 3 * np.exp(-0.5 * zz * zz * (zz > 0))
                                   + np.exp(-0.5 * zz * zz))  # (inside it barely moves; a band one front wide sweeps
        # across; outside it all falls off within one front's width)
        behind = np.clip((rr - dist) / (0.45 * radius + 1), 0, 1)
        wake_amp = WAKE_PX * self.k * zoom * self.gain * (1 - uu) ** 0.6 * np.exp(-3 * behind) * _smooth(
            -zz / 1.5) * cone  # (eased in behind the front's band: cut at its crest, the wake stepped what it smeared)
        wake_amp *= _smooth((dist - 0.6 * self.size * zoom) / (1.2 * self.size * zoom))  # (its fine grain near the
        # palm sawed the fingers' edges)
        mu = ((ang + np.pi) * (720 / (2 * np.pi))).astype(np.float32)
        mv = (dist / (0.06 * self.side * zoom) - 0.9 * kf).astype(np.float32) % 64
        wn = cv2.remap(self.wake, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        radial = bulge + wake_amp * wn[..., 0]
        across = 0.35 * wake_amp * wn[..., 1]
        bh, bw = y1 - y0, x1 - x0
        off = 1.0 if hand is None else 1 - cv2.resize(hand[y0:y1, x0:x1], (gx.shape[1], gy.shape[0]),
                                                      interpolation=cv2.INTER_AREA)
        fx_, fy_ = _tame((-nx * radial - ny * across) * off, (-ny * radial + nx * across) * off, FIELD_Q)
        _remap_rgb(P, _upsample(fx_, bh, bw), _upsample(fy_, bh, bw), box, BLAST_CA)
        arc = cv2.remap(self.arcs, mu, np.zeros_like(mu), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        sg = (1 - zz * zz) * np.exp(0.5 * (1 - zz * zz))  # bright on the front, dark either side of it ...
        sg = np.where((zz < 0) & (sg < 0), SHADOWGRAPH[1] * sg, sg) * np.clip(0.8 + 0.4 * arc, 0.3, 1.5)  # ... less
        # behind it, stronger and weaker round it
        b = (rr - dist) / (self.k * zoom) - COMPRESS[2]  # px at 1080 behind the middle of its compression band
        squeeze = COMPRESS[0] * np.exp(-0.5 * (b / (COMPRESS[3] / 2.355)) ** 2) + COMPRESS[1] * np.exp(
            -0.5 * ((b - COMPRESS[3]) / (COMPRESS[3] / 2.355)) ** 2)
        rel = strength / (BUBBLE_PX * self.k * zoom * self.gain + 1e-6)
        P[y0:y1, x0:x1] *= 1 + _upsample((self.gain * rel * cone * off * (SHADOWGRAPH[0] * sg + squeeze))
                                         .astype(np.float32), bh, bw)[..., None]

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
             fore: np.ndarray, puffs: np.ndarray, hand: np.ndarray | None,
             body: np.ndarray | None = None) -> tuple[float, float, float]:
        """Moves the dust and the dusty air on a frame (``kf``: frames since the push; ``H``: the start frame's px ->
        this frame's, ``H_prev`` the frame before's) and splats them: the dust into ``lit`` and ``drk`` (_specks) and,
        out of a palm, the fine dust riding with it into ``body`` (``scale`` times smaller, BODY_HAZE; or None), the
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
        self.v[go] *= math.exp(-1 / DUST_DRAG)  # drag: out behind the front, slowing
        cx, cy = self.curl(self.p[go, 0], self.p[go, 1], kf)
        turb = DUST_STIR * side * self.gain * float(_eddies(kf))
        self.v[go, 0] += cx * turb
        self.v[go, 1] += cy * turb + DUST_FALL * side
        self.p[go] += self.v[go]
        self.p[self.cloud] = self.ring(kf)
        self.alive &= self.p[:, 2] > 0.07
        after = _project(self.p, self.pp)
        before, after = _cam(H_prev, before), _cam(H, after)
        zz = np.clip(self.p[:, 2], 0.07, None)
        a0 = after + (before - after) * 0.5  # the dusty air smeared over half a frame of motion (smooth), the dust
        # over the shutter's
        disp = np.hypot(*(after - a0).T)
        over = max(float(_smooth((kf - CLOUD_OVER[0]) / CLOUD_OVER[1])), self.off_palm)
        seen = self.seen(self.p, after, hand, np.where(self.cloud, over, DUST_SKIN))
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
        life = np.clip(1 - (kf - self.hang) / DUST_LIFE[1], 0, 1) * float(_smooth(kf / 1.5))
        sel = self.alive & go & (life > 0)
        if not sel.any():
            return depth, life_c, broad
        bright = self.bright * np.clip(DEBRIS_HANG + disp / (12 * k), DEBRIS_HANG, 1.0)  # dust shows most as it moves
        bright = np.where(self.flake, bright * (0.4 + 0.6 * np.abs(np.cos(self.tum + self.tumf * kf)) ** 3), bright)
        _specks(after[sel] + (before[sel] - after[sel]) * SHUTTER, after[sel], zz[sel], (bright * life * seen)[sel],
                self.r0[sel], self.dark[sel], lit, drk, scale, k)
        if body is not None and not self.burst:  # (BURST_N)
            fine = sel & ~self.flake
            _splat(body, (after[fine, 0] + 0.5) / scale - 0.5, (after[fine, 1] + 0.5) / scale - 0.5,
                   (life * seen)[fine] * BODY_HAZE[3])
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


SPECK_LEVELS = 50.0  # a speck at full strength makes what is behind it this many 8-bit grey levels lighter (lint) or
# darker (grit) in its own hue, which reads at phone size (lit 1.45 times in linear light at an opacity of 0.35, specks
# changed a wall by about 8 levels and vanished; in the room's lit tone they were white dots; laid as a colour at half
# opacity, lint lifted the wall half as far again as grit could sink it)
GRIT = 0.45  # share of the dust that is grit, darker than what is behind it; the rest lint, lighter
DUST_SKIN = 0.35  # over the hand, dust in front of it shows this much of itself (in full, specks on the skin read as
# hairs)
SHUTTER = 0.12  # share of a frame the shutter is open: a speck is smeared over this much of its motion (over half a
# frame, with the cap, every speck in the ball was a 25-30 px rod and every speck thrown a needle)
SPECK_TAIL = 0.6  # a speck smeared by its motion is drawn this strong at its tail, full at its head, its ends fading
SMEAR_MAX = 30.0  # px at 1080: never smeared longer than this
SMEAR_FADE = 0.2  # a smeared speck is fainter by its length over its size to this power (its light spread along it:
# drawn as strong however long, a fast speck was a hard rod and a slow one a dot)
DUST_SIGMA = (0.9, 1.6, 3.2, 7.0)  # px at 1080: how soft each class of speck is drawn (a 2-4 px core ... out of
# focus) ...
DUST_CLASS = (1.25, 2.4, 5.0)  # ... a speck needing a core this soft or softer drawn in the next class


def _specks(a0: np.ndarray, after: np.ndarray, zz: np.ndarray, weight: np.ndarray, r0: np.ndarray, dark: np.ndarray,
            lit: list, drk: list, scale: float, k: float, soft: np.ndarray | None = None) -> None:
    """Splats specks of dust into ``lit`` (lint) and ``drk`` (grit: ``dark``), one buffer per softness (DUST_SIGMA;
    the softer two ``scale`` times smaller than the frame), for _lay_specks: each a core ``r0`` px across at the palm's
    depth (at 1080), ``zz`` palm depths from the lens (bigger nearer it, and out of focus well in front of it, its
    light spread out), softened ``soft`` px at 1080 more (or none), smeared from ``a0`` to ``after`` (frame px: its
    motion while the shutter is open, SMEAR_MAX at the most, and fainter the longer, SMEAR_FADE) with its head
    strongest and its ends fading (SPECK_TAIL), as strong as ``weight`` (0..1) along it."""
    rad = r0 * k / zz
    core = np.maximum(rad / 2.355, 0.8 * k)
    if soft is not None:
        core = np.hypot(core, soft * k)
    sig = core + 4.0 * k * np.clip((0.4 - zz) / 0.33, 0, 1) ** 1.5
    weight = weight * (core / sig) ** 2
    d = after - a0
    L = np.hypot(d[:, 0], d[:, 1])
    cut = np.minimum(1.0, SMEAR_MAX * k / np.maximum(L, 1e-6))
    a0, L = after - d * cut[:, None], L * cut
    weight = weight * (2.355 * sig / np.maximum(L, 2.355 * sig)) ** SMEAR_FADE
    for j, (lo, hi) in enumerate(zip((0.0, *DUST_CLASS), (*DUST_CLASS, 1e9), strict=True)):
        sel = (sig >= lo * k) & (sig < hi * k) & (weight > 1e-3)
        if not sel.any():
            continue
        q = 1.0 if j < 2 else scale  # (a pixel's middle lands on the middle of the coarse pixel holding it)
        sc = DUST_SIGMA[j] * k
        Ls = L[sel]
        S = int(np.clip(np.ceil(Ls.max() / (0.7 * q)) + 2, 2, 64))
        s = np.linspace(0, 1, S)[None, :]
        X = (a0[sel, 0:1] + (after[sel, 0:1] - a0[sel, 0:1]) * s + 0.5) / q - 0.5
        Y = (a0[sel, 1:2] + (after[sel, 1:2] - a0[sel, 1:2]) * s + 0.5) / q - 0.5
        prof = (SPECK_TAIL + (1 - SPECK_TAIL) * s) * _smooth(s / 0.3) * _smooth((1 - s) / 0.15)  # (fading at both
        # ends: with only its tail dimmed, every smear was a hard-ended capsule)
        taper = 1 - (1 - prof) * np.minimum(1.0, Ls / (3 * sc))[:, None]  # (a dot even all round)
        wv = (weight[sel] * np.maximum(Ls, math.sqrt(2 * math.pi) * sc) / (q * S))[:, None] * taper  # (as strong
        # along it however long, and a dot as strong at its middle)
        dk = dark[sel]
        _splat(lit[j], X[~dk].ravel(), Y[~dk].ravel(), wv[~dk].ravel())
        _splat(drk[j], X[dk].ravel(), Y[dk].ravel(), wv[dk].ravel())


def _speck_bufs(h: int, w: int, scale: int) -> tuple[list, list]:
    """Empty buffers for _specks over an (h, w) picture: (lint, grit), each one per class of DUST_SIGMA, the softer
    two ``scale`` times smaller."""
    lit = [np.zeros((h, w), np.float32) for _ in range(2)] + [
        np.zeros((-(-h // scale), -(-w // scale)), np.float32) for _ in range(2)]
    return lit, [np.zeros_like(b) for b in lit]


def _speck_gain(behind: np.ndarray, levels: float) -> np.ndarray:
    """(h, w): how many times as bright a speck at full strength makes what is behind it (linear (h, w, 3); the same
    for each channel, so it keeps its hue): ``levels`` 8-bit grey levels lighter (lint, up to 250), or darker (grit:
    ``levels`` below 0)."""
    g = np.clip(behind @ fx._LUMA, 0.0, 1.0) ** (1 / 2.2) * np.float32(255)
    gain = (np.clip(g + np.float32(levels), 0, 250) / np.maximum(g, 1.0)) ** np.float32(2.2)
    return np.minimum(gain, 60.0).astype(np.float32)


def _unhanded(behind: np.ndarray, hand: np.ndarray) -> np.ndarray:
    """_behind's map ``behind`` with what the hand's matte ``hand`` (0..1, the size of the frame it was made from)
    covers filled in from the room round it: the colour dust in front of the hand has, lit as the room is (where the
    hand covers it all, ``behind`` itself)."""
    room = 1 - cv2.resize(hand.astype(np.float32), behind.shape[1::-1], interpolation=cv2.INTER_AREA)
    filled = behind
    for sig in (36.0, 12.0, 4.0):  # cells: from the room far round a wide palm, then nearer where there is some
        n = cv2.GaussianBlur(room, (0, 0), sig)
        fill = cv2.GaussianBlur(behind * room[..., None], (0, 0), sig) / np.maximum(n, 1e-6)[..., None]
        filled = filled + (fill - filled) * _smooth((n - 0.02) / 0.1)[..., None]
    return (behind + (filled - behind) * (1 - room[..., None])).astype(np.float32)


def _in_front(q_: np.ndarray, q0: np.ndarray, s: np.ndarray, ys: np.ndarray, xs: np.ndarray, levels: float,
              front: tuple[np.ndarray, np.ndarray, np.ndarray | None]) -> None:
    """In place on ``q_`` (pixels at frame px ``ys``, ``xs``: dust laid over ``q0``, what they were, ``s`` its strength
    0..1): where ``front`` = (room, cover, tone) says the dust is in front of what is there (``cover``, 0..1 (h, w),
    the hand's matte), it is laid as its own colour, what it is against the room (``room``: _unhanded, ``levels`` grey
    levels lighter or darker than it), lint lighter, NEAR_LIT of the way to the room's lit ``tone`` (linear, or None),
    covering what is there by its strength: dust in front of a lit palm reads as dust, not as a lighter or darker patch
    of skin (laid in the skin's own hue, over a bright palm it barely showed)."""
    room, cover, tone = front
    c_ = cover[ys, xs]
    on = c_ > 1e-3
    if not on.any():
        return
    yo, xo = ys[on], xs[on]
    own = _at(room, yo, xo) * _at(_speck_gain(room, levels), yo, xo)[:, None]
    if levels > 0 and tone is not None:
        own = own + (tone.astype(np.float32) - own) * np.float32(NEAR_LIT)
    q_[on] += c_[on, None] * (q0[on] + s[on, None] * (own - q0[on]) - q_[on])


def _lay_specks(P: np.ndarray, lit: list, drk: list, scale: int, grain, code: float, k: float,
                behind: np.ndarray | None = None, levels: float = SPECK_LEVELS,
                front: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None) -> None:
    """In place: the specks splatted into ``lit`` and ``drk`` (_specks) laid over ``P`` (linear, (h, w, 3)), each
    class blurred to its softness, the grit making what is behind it darker and the lint lighter, by ``levels`` at
    full strength (_speck_gain of ``behind``, _behind of ``P``, worked out here when None), and in front of the hand
    as their own colour (``front``: _in_front's, or None), with the plate's grain (``grain(y0, y1, x0, x1)``: unit
    grain for that box, or None). Only the pixels they touch are touched."""
    h, w = P.shape[:2]
    for bufs, sign, gr in ((drk, -1.0, None), (lit, 1.0, grain)):
        a = np.zeros((h, w), np.float32)
        for j, sig_ in enumerate(DUST_SIGMA):
            buf = bufs[j]
            box = _bbox(buf > 0)
            if box is None:
                continue
            q = 1 if j < 2 else scale
            s = max(0.5, sig_ * k / q)
            m = int(3 * s) + 2
            bh, bw = buf.shape
            y0, y1, x0, x1 = max(0, box[0] - m), min(bh, box[1] + m), max(0, box[2] - m), min(bw, box[3] + m)
            b_ = cv2.GaussianBlur(buf[y0:y1, x0:x1], (0, 0), s) * np.float32(math.sqrt(2 * math.pi) * s)  # (a line
            # of weight 1 a px reads 1 along its middle)
            if q == 1:
                a[y0:y1, x0:x1] += b_
            else:
                Y0, Y1, X0, X1 = y0 * q, min(h, y1 * q), x0 * q, min(w, x1 * q)
                a[Y0:Y1, X0:X1] += cv2.resize(b_, ((x1 - x0) * q, (y1 - y0) * q),
                                              interpolation=cv2.INTER_LINEAR)[:Y1 - Y0, :X1 - X0]
        box = _bbox(a > 2e-3)
        if box is None:
            continue
        if behind is None:
            behind = _behind(P)
        y0, y1, x0, x1 = box
        s = np.minimum(a[y0:y1, x0:x1], 1.0)
        ys, xs = np.nonzero(s > 2e-3)
        s = s[ys, xs]
        f = _at(_speck_gain(behind, sign * levels), ys + y0, xs + x0)
        Q = P[y0:y1, x0:x1]
        q0 = Q[ys, xs]
        q_ = q0 * (1 + (f - 1) * s)[:, None]
        if front is not None:
            _in_front(q_, q0, s, ys + y0, xs + x0, sign * levels, front)
        if gr is not None and code > 0:  # (the footage's grain in what it changed)
            i = (np.sqrt(np.clip(q_ @ fx._LUMA, 0.0, 1.0)) * np.float32(1023)).astype(np.int32)
            q_ *= 1 + (s * gr(y0, y1, x0, x1)[ys, xs] * (np.float32(code) * _STEP_LUT[i]))[:, None]
        Q[ys, xs] = q_


def _lay_haze(P: np.ndarray, haze: np.ndarray, scale: int, behind: np.ndarray, levels: float,
              most: float = HAZE_MOST,
              front: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None) -> None:
    """In place: a haze of fine dust (``haze``: how thick, 1 where one train's is thickest, ``scale`` times coarser
    than ``P``) laid over ``P`` (linear (h, w, 3)) as what is behind it made about ``levels`` grey levels lighter
    (darker, below 0) in its own hue where it is 1, thicker haze adding less and less up to ``most`` times that, and
    in front of the hand as its own colour (``front``: _in_front's, or None); ``behind``: _behind of ``P``. Only the
    pixels it touches are touched."""
    box = _bbox(haze > 1e-3)
    if box is None:
        return
    h, w = P.shape[:2]
    y0, y1, x0, x1 = box
    Y0, Y1, X0, X1 = y0 * scale, min(h, y1 * scale), x0 * scale, min(w, x1 * scale)
    a = cv2.resize(haze[y0:y1, x0:x1], ((x1 - x0) * scale, (y1 - y0) * scale),
                   interpolation=cv2.INTER_LINEAR)[:Y1 - Y0, :X1 - X0]
    a = np.float32(most) * (1 - np.exp(-a / np.float32(most)))
    ys, xs = np.nonzero(a > 2e-3)
    f = _at(_speck_gain(behind, levels), ys + Y0, xs + X0)
    Q = P[Y0:Y1, X0:X1]
    q0 = Q[ys, xs]
    q_ = q0 * (1 + (f - 1) * a[ys, xs])[:, None]
    if front is not None:
        _in_front(q_, q0, np.minimum(a[ys, xs], 1.0), ys + Y0, xs + X0, levels, front)
    Q[ys, xs] = q_


def _lay_puffs(P: np.ndarray, puffs: np.ndarray, scale: int, behind: np.ndarray, k: float) -> None:
    """In place: the puffs of fine dust kicked off the room's ledges (``puffs``: their parcels splatted ``scale`` times
    coarser than ``P``, each as much as it shows) laid over ``P`` (linear (h, w, 3)) as soft dust lit by the room's
    light as it falls there (LEDGE_PUFF; ``behind``: _behind of ``P``, the scene round each pixel): over what is behind
    it, as bright as the scene round it times the lift and in the hue of what it covers, never darker than it, so it
    lifts what is behind it in its own hue and reads as dust on a wall, a skirting board, a white plate or a shadow
    alike. Only the pixels it touches are touched."""
    s = LEDGE_PUFF[2] * k / scale
    a = cv2.GaussianBlur(puffs, (0, 0), s) * np.float32(2 * math.pi * s * s)  # (a lone parcel peaks at its share)
    box = _bbox(a > 1e-3)
    if box is None:
        return
    h, w = P.shape[:2]
    y0, y1, x0, x1 = box
    Y0, Y1, X0, X1 = y0 * scale, min(h, y1 * scale), x0 * scale, min(w, x1 * scale)
    a = cv2.resize(a[y0:y1, x0:x1], ((x1 - x0) * scale, (y1 - y0) * scale),
                   interpolation=cv2.INTER_LINEAR)[:Y1 - Y0, :X1 - X0]
    a = np.float32(LEDGE_PUFF[0]) * (1 - np.exp(-1.5 * a))
    ys, xs = np.nonzero(a > 2e-3)
    Q = P[Y0:Y1, X0:X1]
    q = Q[ys, xs]
    lift = (_at(behind, ys + Y0, xs + X0) @ fx._LUMA) * np.float32(LEDGE_PUFF[1]) / np.maximum(q @ fx._LUMA, 1e-4)
    col = np.minimum(q * np.clip(lift, 1.0, LEDGE_PUFF[7])[:, None], 0.95)  # (the same for each channel: its hue)
    Q[ys, xs] = q + a[ys, xs, None] * (col - q)


def _lay_motes(P: np.ndarray, motes: np.ndarray, behind: np.ndarray, tone: np.ndarray) -> None:
    """In place: dust passing the lens (rows of _Blast.motes) laid over ``P`` (linear (h, w, 3), the whole frame;
    ``behind``: _behind of it) as soft discs out of focus, smeared along their motion, the room's light they scatter
    added over what is behind them in the hue of its lit ``tone``, as many grey levels lighter as each row says. Only
    the pixels they touch are touched."""
    h, w = P.shape[:2]
    hue = tone.astype(np.float32) / max(float(tone @ fx._LUMA), 1e-4)
    for x0_, y0_, x1_, y1_, s, lev in motes:
        pad = 3 * s
        box = _clip_box(0.5 * (x0_ + x1_), 0.5 * (y0_ + y1_), 0.5 * abs(x1_ - x0_) + pad, 0.5 * abs(y1_ - y0_) + pad,
                        w, h)
        if box is None:
            continue
        y0, y1, x0, x1 = box
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = x1_ - x0_, y1_ - y0_
        along = np.clip(((xx - x0_) * dx + (yy - y0_) * dy) / max(dx * dx + dy * dy, 1e-6), 0, 1)
        d2 = (xx - x0_ - along * dx) ** 2 + (yy - y0_ - along * dy) ** 2
        a = np.exp(-0.5 * d2 / np.float32(s * s))
        ys, xs = np.nonzero(a > 2e-3)
        f = _at(_speck_gain(behind, float(lev)), ys + y0, xs + x0)
        Q = P[y0:y1, x0:x1]
        q = Q[ys, xs]
        Q[ys, xs] = q + ((f - 1) * a[ys, xs] * (q @ fx._LUMA))[:, None] * hue


def _ledges(frame_u8: np.ndarray, hand: np.ndarray | None, first: _Blast, rng: np.random.Generator) -> np.ndarray:
    """Up to LEDGE's first points (n, 2: frame px) on the room's ledges in ``frame_u8``: its level edges most, steep
    ones a little, off the hand's matte ``hand`` (0..1, or None) and the frame's edge, where the push ``first``'s front
    reaches, and more of them ahead of it."""
    h, w = frame_u8.shape[:2]
    q = 8
    g = cv2.resize(frame_u8, (max(1, w // q), max(1, h // q)), interpolation=cv2.INTER_AREA).astype(np.float32) \
        @ fx._LUMA
    edge = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)) + 0.35 * np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0))
    gh, gw = g.shape
    ok = np.zeros((gh, gw), bool)
    ok[2:-2, 2:-2] = True
    if hand is not None:
        ok &= cv2.dilate((cv2.resize(hand, (gw, gh), interpolation=cv2.INTER_AREA) > 0.05).astype(np.uint8),
                         np.ones((7, 7), np.uint8)) == 0
    ys, xs = np.nonzero(ok)
    at = np.column_stack([(xs + 0.5) * w / gw, (ys + 0.5) * h / gh])
    rel = at - first.o
    dd = np.hypot(*rel.T) + 1e-6
    wt = edge[ys, xs] * first.cone((rel @ first.u) / dd) * (dd < 0.92 * first.front(BLAST_FRAMES)[0]) * (
        dd > first.size)
    if not (wt > 0).any():
        return np.zeros((0, 2))
    pick = rng.choice(len(at), min(LEDGE[0], int((wt > 0).sum())), replace=False, p=wt / wt.sum())
    return at[pick] + rng.uniform(-0.5, 0.5, (len(pick), 2)) * (w / gw, h / gh)


def _outline(matte_u8: np.ndarray, palm, reach: float) -> np.ndarray | None:
    """Points (n, 2: frame px) just outside the hand's outline in ``matte_u8`` (its matte, 0..255) within ``reach`` px
    of ``palm`` and off the frame's edge, or None when there are too few."""
    h, w = matte_u8.shape[:2]
    cs, _ = cv2.findContours((matte_u8 > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    pts = np.concatenate([c[:, 0, :] for c in cs]).astype(np.float64)
    rel = pts - np.asarray(palm, np.float64)
    r = np.hypot(*rel.T)
    keep = (r < reach) & (r > 1) & (pts[:, 0] > 2) & (pts[:, 1] > 2) & (pts[:, 0] < w - 3) & (pts[:, 1] < h - 3)
    if keep.sum() < 20:
        return None
    return pts[keep] + 3 * rel[keep] / r[keep, None]


class _Room:
    """The room's own dust, one pool for all of a shot's pushes: still and unseen until a front reaches it, then
    thrown outward from the palm, slowing, swirling and fading out as it settles, and thrown again by the next push
    (each push brought its own and it built up: the dust of one still hung through the next). Most of it is near the
    first push and ahead of it, thinning out with distance (spread evenly over the frame, it read as sensor snow); and,
    given the start frame, some of it lies on the room's ledges (LEDGE), kicked up as puffs a frame or two after the
    front passes. It lives in the room as the first push's start frame saw it."""

    def __init__(self, first: _Blast, seed: int, frame: np.ndarray | None = None, hand: np.ndarray | None = None):
        rr = fx._rng(seed, _ROOM)
        w, h, side = first.w, first.h, first.side
        m = 4 * ROOM_N  # where it may be: anywhere in front of the wall
        zr = rr.uniform(0.6, 1.6, m)
        scr = np.column_stack([rr.uniform(-0.1, 1.1, m) * w, rr.uniform(-0.1, 1.1, m) * h])
        r0, bright, dark = rr.uniform(*DUST_R, m), rr.uniform(*DUST_SEEN, m), rr.random(m) < GRIT
        rel = scr - first.o
        dd = np.hypot(rel[:, 0], rel[:, 1]) + 1e-6
        near = first.cone((rel @ first.u) / dd) * np.exp(-dd / (ROOM_NEAR * side))
        keep = np.sort(rr.choice(m, ROOM_N, replace=False, p=near / near.sum()))
        self.start, self.side, self.k, self.f, self.pp = first.start, side, first.k, first.f, first.pp
        p = np.column_stack([first.pp + (scr[keep] - first.pp) * zr[keep, None], zr[keep]])
        r0, bright, dark = r0[keep], bright[keep], dark[keep]
        kind, lag, puff = np.zeros(ROOM_N, int), np.zeros(ROOM_N, int), np.zeros((ROOM_N, 2))
        at = np.zeros((0, 2)) if frame is None else _ledges(frame, hand, first, rr)
        if len(at):  # on the ledges: a speck, and the puff of finer dust kicked up with it
            nl, per = len(at), 1 + LEDGE[1]
            kind_l = np.tile(np.r_[1, np.full(LEDGE[1], 2)], nl)
            n_ = nl * per
            ang = rr.uniform(0, 2 * np.pi, n_)
            p = np.concatenate([p, np.column_stack([first.pp + (np.repeat(at, per, 0) - first.pp) * LEDGE[3],
                                                    np.full(n_, LEDGE[3])])])
            r0 = np.concatenate([r0, rr.uniform(*DEBRIS_R[1], n_)])
            bright = np.concatenate([bright, np.where(kind_l == 1, rr.uniform(*DEBRIS_SEEN[1], n_),
                                                      rr.uniform(0.5, 1.0, n_))])
            dark = np.concatenate([dark, rr.random(n_) < GRIT])
            kind, lag = np.concatenate([kind, kind_l]), np.concatenate(
                [lag, np.repeat(rr.integers(LEDGE[2][0], LEDGE[2][1] + 1, nl), per)])
            puff = np.concatenate([puff, (np.column_stack([np.cos(ang), np.sin(ang)]) * (
                LEDGE_PUFF[3] * rr.uniform(0.2, 1.0, n_))[:, None] - (0.0, LEDGE_PUFF[5])) * (
                self.k * (kind_l == 2))[:, None]])
        n = len(p)
        self.p, self.v = p, np.zeros((n, 3))
        self.r0, self.bright, self.dark = r0, bright, dark
        self.kind, self.lag, self.puff = kind, lag, puff  # 0 the room's floating dust, 1 a speck on a ledge, 2 its
        # puff's parcels; the frames each is kicked up after the front reaches it; a puff's parcels' drift apart and
        # up
        self.kick = np.full(n, -1)  # the frame each speck was last thrown on (-1: not yet)
        self.gain = np.zeros(n)  # ... and how hard that push was
        self.curl = _Curl(rr, 0.1 * side)

    def throw(self, b: _Blast, kf: int, t: int, Hb: np.ndarray) -> None:
        """Throws the dust that push ``b``'s front reaches on its ``kf``-th frame (frame ``t``; ``Hb``: the room's px
        -> ``b``'s start frame's) outward from its palm, harder ahead of the push and nearer it; what lies on a ledge
        is kicked up off it a little after, slower, its puff spreading."""
        if b.room_hit is None:
            b.room_hit = np.zeros(len(self.p), bool)
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
        kd = self.kind[hit]
        sp = sp * np.choose(kd, [1.0, LEDGE[4], LEDGE[4] * LEDGE_PUFF[4]])
        lift = np.where(kd > 0, LEDGE[5] * sp, 0.0)  # (off the ledge, up)
        self.v[hit] = np.column_stack([D[:, 0] * sp + self.puff[hit, 0], D[:, 1] * sp - lift + self.puff[hit, 1],
                                       D[:, 2] * sp / self.f])
        self.kick[hit], self.gain[hit] = t + self.lag[hit], b.gain
        b.room_hit |= hit

    def dust(self, t: int, H: np.ndarray, H_prev: np.ndarray, lit: list, drk: list, scale: int,
             hand: np.ndarray | None, b: _Blast, Hb: np.ndarray, puffs: np.ndarray) -> None:
        """Moves the thrown dust on frame ``t`` and splats it into ``lit`` and ``drk`` (_specks; ``H``: the room's px
        -> this frame's, ``H_prev`` the frame before's), and the puffs off the ledges into ``puffs`` (``scale`` times
        coarser than the frame, for _lay_puffs). What is behind the palm of the latest push ``b`` (``Hb``: the room's
        px -> its start frame's) is hidden by the hand's matte ``hand``."""
        thrown = (self.kick >= 0) & (t >= self.kick)
        if not thrown.any():
            return
        age = t - self.kick
        before = _project(self.p, self.pp)
        self.v[thrown] *= math.exp(-1 / DUST_DRAG)
        cx, cy = self.curl(self.p[thrown, 0], self.p[thrown, 1], t)
        turb = 0.5 * DUST_STIR * self.side * self.gain[thrown] * _eddies(age[thrown])  # (the room's air is calmer)
        self.v[thrown, 0] += cx * turb
        self.v[thrown, 1] += cy * turb + DUST_FALL * self.side * (self.kind[thrown] < 2)  # (a puff's fine dust floats)
        self.p[thrown] += self.v[thrown]
        after = _project(self.p, self.pp)
        before, after = _cam(H_prev, before), _cam(H, after)
        zz = np.clip(self.p[:, 2], 0.07, None)
        a0 = after + (before - after) * SHUTTER
        disp = np.hypot(*(after - before).T) * 0.5
        seen = b.seen(np.column_stack([_cam(Hb, self.p[:, :2]), self.p[:, 2]]), after, hand, DUST_SKIN)
        life = np.where(self.kind == 2, np.clip(1 - age / LEDGE_PUFF[6], 0, 1), np.clip(
            1 - (age - DUST_LIFE[0]) / DUST_LIFE[1], 0, 1)) * _smooth(age / 1.5)  # (a puff spreads and is gone soon)
        fade = np.clip(1 - age / (DUST_LIFE[0] + DUST_LIFE[1]), 0, 1)
        weight = self.bright * np.clip(ROOM_SEEN * fade + disp / (2.5 * self.k), 0, 1) * life * seen * thrown
        sel = (weight > 0) & (self.kind < 2)
        if sel.any():
            _specks(a0[sel], after[sel], zz[sel], weight[sel], self.r0[sel], self.dark[sel], lit, drk, scale, self.k)
        sel = (weight > 0) & (self.kind == 2)
        if sel.any():
            _splat(puffs, (after[sel, 0] + 0.5) / scale - 0.5, (after[sel, 1] + 0.5) / scale - 0.5,
                   (self.bright * life * seen)[sel])


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
    dx, dy = _tame(pull * ddx, pull * ddy, CLOUD_Q)
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


def _push_frames(plate: np.ndarray, out: np.ndarray, blasts: list[_Blast], mask: np.ndarray | None, tl: Timeline,
                 seed: int, code: float, say) -> None:
    """Draws pushes of air (``blasts``, in the order they start) over ``out`` (uint8 frames; ``plate``: the shot's
    own, where the room's ledges are found) in place: on each frame each push's bubble bends what is behind it, its
    ring of dusty air bends it too and is laid on as a faint haze lit by the room's light, the dust it throws and the
    room's dust (_Room) are drawn through the camera's motion since, and the dust it sweeps past the lens over all of
    it; then each push knocks the phone, smearing the picture out from it, and those that come at the lens swell the
    picture (buffet)."""
    from .bending import camera_between

    n, h, w = out.shape[:3]
    cam = tl.camera
    k = min(h, w) / 1080.0
    sc = 4  # the softer dust is drawn this many times smaller
    hq = (-(-h // CLOUD_Q), -(-w // CLOUD_Q))
    first, last = blasts[0].start, min(n - 1, max(b.last_frame() for b in blasts))

    def hand_at(t: int) -> np.ndarray | None:
        return None if mask is None else mask[min(t, len(mask) - 1)].astype(np.float32) / 255.0

    room = _Room(blasts[0], seed, plate[first], hand_at(first))
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
        held = None if hand is None else cv2.GaussianBlur(cv2.dilate(
            hand, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * round(FRONT_HAND[0] * k) + 1,) * 2)), (0, 0),
            FRONT_HAND[1] * k)  # (what the front leaves unbent)
        lit, drk = _speck_bufs(h, w, sc)
        body = np.zeros((-(-h // sc), -(-w // sc)), np.float32)
        air_ = _Haze.empty(hq)
        lump = 2.0
        gust = 0.0
        motes = np.concatenate([b.motes(t - b.start) for b in live])
        for b in live:
            kf = t - b.start
            H = camera_between(cam, b.start, t)
            Hp = camera_between(cam, b.start, max(b.start, t - 1))
            o = H[:2, :2] @ b.o + H[:2, 2]
            zoom = math.sqrt(abs(np.linalg.det(H[:2, :2])))
            u = H[:2, :2] @ b.u
            u = u / max(float(np.hypot(*u)), 1e-9)
            if kf <= BLAST_FRAMES:
                b.bend(P, kf, o, u, zoom, held)
                room.throw(b, kf, t, camera_between(cam, room.start, b.start))
            haze, fore, puffs = (np.zeros(hq, np.float32) for _ in range(3))
            depth, shows, wide = b.dust(kf, H, Hp, lit, drk, sc, haze, fore, puffs, hand, body)
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
                buffets.append((t, o, b.gain * (1 - b.q), KNOCK_PX * k * b.gain, u, ZOOM_BLUR[0] * k * b.gain))
            if kf == GUST[0] and not b.burst:  # (a ball's pop keeps its punch round the ball: PULSE_RISE)
                gust = max(gust, min(1.0, b.gain))
        b = started[-1]
        puffs = np.zeros((-(-h // sc), -(-w // sc)), np.float32)
        room.dust(t, camera_between(cam, room.start, t), camera_between(cam, room.start, max(room.start, t - 1)), lit,
                  drk, sc, hand, b, camera_between(cam, room.start, b.start), puffs)
        if air_.dens.any():
            _cloud_bend(P, air_, lump, hand)
        P0 = None if hand is None else P.copy()  # (bent, before what the air carries is laid on: _reedge)
        if air_.dens.any():
            _lay_cloud(P, air_, tone_s, light, lump, hand, (seed, t), code)
        behind = _behind(P)
        if puffs.any():
            _lay_puffs(P, puffs, sc, behind, k)
        if body.any():  # the fine dust riding with the thrown body (BODY_HAZE: its specks' weight within its blur)
            s_ = BODY_HAZE[2] * k / sc
            _lay_haze(P, cv2.GaussianBlur(body, (0, 0), s_) * np.float32(2 * math.pi * s_ * s_), sc, behind,
                      BODY_HAZE[0], BODY_HAZE[1])
        _lay_specks(P, lit, drk, sc, lambda y0, y1, x0, x1: _grain(seed, _BLAST, t, y1 - y0, x1 - x0), code, k,
                    behind=behind)
        if len(motes):
            _lay_motes(P, motes, behind, tone_s)
        if gust > 0:  # the gust at the lens: a veil of dust on it, lit by the room, on this frame alone (GUST)
            hue = tone_s.astype(np.float32) / max(float(tone_s @ fx._LUMA), 1e-4)
            veil = np.float32(GUST[2] * gust * float((P[::8, ::8] @ fx._LUMA).mean())) * hue
            P *= np.float32(1 + GUST[1] * gust)
            P += veil
        if hand is not None:
            _reedge(P, P0, hand, k)
        out[t] = gam(P)
        if (t - first) % 8 == 0:
            say("air", (t - first) / max(1, last + 1 - first), f"Pushing the air out: frame {t - first + 1} of "
                                                                 f"{last + 1 - first}")
    for t, o, swell, knock, u, blur in buffets:
        buffet(out, t, o, swell, knock, u, blur)


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
        rim = None if mask is None else _outline(mask[min(g, len(mask) - 1)], xy[g], OUTLINE_REACH * size[g])
        blasts.append(_Blast(xy[g], u, q, normal[g], size[g], w, h, f, gain, seed, j, outline=rim))
    if blasts:
        _push_frames(plate, out, blasts, mask, tl, seed, code, say)


def buffet(frames_u8: np.ndarray, at: int, centre, swell: float, knock: float, along=(0.0, -1.0),
           blur: float = 0.0) -> None:
    """The push reaching the phone, in place, from frame ``at``: the phone knocked ``knock`` px along the push
    (``along``, its direction in the picture) and back as a damped spring, past where it was and settling within
    about five frames (KNOCK_SHAPE), and the picture swelling round the push and settling over about seven as much as
    ``swell`` (0..1: how much of a full push came at the lens; BUFFET_SHAPE); each frame smeared over the shutter's
    half frame of its move (a shake and a zoom blur), and the knock's first frames smeared out from ``centre`` by
    ``blur`` px at the frame's furthest corner (ZOOM_BLUR's shares). It never dims the picture."""
    if swell <= 0.05 and knock < 0.5 and blur < 0.5:
        return
    n, h, w = frames_u8.shape[:3]
    ax, ay = np.asarray(along, np.float64) / max(float(np.hypot(*along)), 1e-9)
    m_ = max(len(BUFFET_SHAPE), len(KNOCK_SHAPE) + 2)
    grow = BUFFET_SHAPE + (0.0,) * (m_ - len(BUFFET_SHAPE))  # from the frame before the gust on
    shove = (0.0, *KNOCK_SHAPE) + (0.0,) * (m_ - 1 - len(KNOCK_SHAPE))
    swell = swell if swell > 0.05 else 0.0
    corner = max(math.hypot(x - centre[0], y - centre[1]) for x in (0, w) for y in (0, h))
    for d in range(m_ - 1):
        f = at + d
        if not 0 <= f < n:
            continue
        s, s_was = (1 + BUFFET * swell * grow[d + i] for i in (1, 0))
        m, m_was = (knock * shove[d + i] for i in (1, 0))
        zb = blur * ZOOM_BLUR[1][d] if d < len(ZOOM_BLUR[1]) else 0.0
        if s == s_was == 1.0 and m == m_was == 0.0 and zb < 0.5:
            continue
        taps = 1 + round(6 * abs(grow[d + 1] - grow[d]) * swell / 0.6 + abs(m - m_was) / 3.0 + zb)  # (the faster
        # it moves, the longer the smear)
        acc = np.zeros((h, w, 3), np.float32)
        for j in range(taps):
            back = 0.5 * (1 - j / max(taps - 1, 1))
            sj, mj = s + (s_was - s) * back, m + (m_was - m) * back
            if taps > 1:
                sj *= 1 + zb / corner * (j / (taps - 1) - 0.5)
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
