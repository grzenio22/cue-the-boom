"""Lightning made from the hand (cuesync.elements): arcs struck from the fingertips, or a crackling ball of them
hovering over an open palm, drawn frame by frame from the hand's track and lit onto the plate.

- **Channels.** A bolt is a jagged line (recursive midpoint displacement: each midpoint pushed across its segment by a
  share of that segment's length, gently at the coarse levels so the channel runs on, sharply at the fine ones so it
  kinks) with forks that leave it more often towards its far end, each much dimmer and thinner, as a streamer branches
  when it runs out. One main channel leaves each striking fingertip (two or three streams from one finger read as
  parallel cables): it comes out along the finger (its first stretch on the finger's aim, the rest never far off it)
  and thins from where it leaves to a fifth of that at its far end; a fork fades and thins along its length, a wedge
  rather than a wire. No channel crosses another or itself, no fork runs back to the channel it leaves, and none runs
  alongside another (each is drawn again until it keeps off those struck before it, or dropped): a closed loop is the
  most obviously fake shape lightning can take, two wires side by side the next. Two fingers held together each
  strike a main channel, turned well apart so the two separate, their forks on their outer sides; the second strikes
  with the first, kept well clear of the first's, turned further out each time it would not be (from fingers one
  behind the other along their aim, it leaves past the first's tip, bending out there rather than turning at an
  elbow); where either strikes nothing, the two end well round from each other, or the second is not struck. A tap
  on the palm strikes from the index or the middle fingertip, never from the middle of the hand.
- **Where it strikes.** A bolt goes somewhere: it strikes an edge of the plate in its way (a floor's line, a door's
  jamb, a tub's rim: a long, clear line in the picture, off the hand and well inside the frame) within a cone round
  where its finger points, about as far out as the voice is loud, one of the best few, and each time it is struck
  again somewhere new, well away from where it struck last, within a wider cone (so a burst's strikes go back and
  forth between places rather than hold one tether); now and then it ends in two prongs landing on two points; with a
  contact flash where it touches: a white-hot point, a pool of its light along that surface (brightest as it strikes)
  and a splash of forks along and off it on its first frames. With nothing to strike it runs off the frame, or ends in
  the air in a brush of dim forks, with no end node; a weak stroke has none either.
- **Strikes.** A channel is struck and then re-struck down the same path for a frame at most (real lightning's
  return strokes), flashing or going dark, before a new path is found: a new one every frame or two (held for 3-4,
  the channels hung in the air). While it lives, its root rides the fingertip and its far end stays where it struck
  (carried by the camera on a handheld shot): a path drawn anew every frame jumped about like static on a screen, and
  one drawn once and carried whole with the hand swung its far end through the room. A new main channel leaves near
  its last aim, which sways slowly, so the strikes sweep rather than jump. A finger's aim is its own direction in the
  picture where it is long and clear there, its vanishing point where it is short. A second finger strikes only on
  the frames the picture shows it out beside the first (its tip past its middle joint, away from the palm, level with
  the first's and on the silhouette's outline), never from a finger curled into the hand whatever the tracker's depth
  says of it. The bolts go into the room behind the hand: a channel is drawn again rather than lie across the hand's
  matte, and whatever of one, or of the crackle, still passes over the hand past the fingertip's own spot is hidden
  by it: only light falls on the skin.
- **The ball.** A light held over the palm, a ball of plasma hovering well off it (up the picture, or along the palm's
  normal as it shows) and wandering a little there: a volume of light, white-hot only in its middle (a cool
  near-white far over white, PLASMA_RGB), a clear translucent cyan-blue round that, never blown out to white, falling
  off smoothly with no outline of its own, then a faint blue haze of its light round it, breathing with the voice,
  the same over skin and wall. Its body is moving plasma: filaments of brighter and denser light that boil and swirl
  through it at about a turn and a half a second (PLASMA), not an airbrushed marble. It rolls off to white with its
  hue kept as far as its own light outshines what is behind it, never clipped channel by channel to a flat pale aqua.
  It hides what is behind it as much as it is dense, most of it behind its middle and little at its edge, where the
  palm's lines show through, brightened; what shows through it is scattered in it, so the wall in a gap between two
  fingers is no window through a glass marble.
  An irregular shell of jagged arcs crawls over its skin, held to it (in 3D, turning slowly: brightest where they run
  along its outline, which they draw, dimmer across its face and behind it; a few on a sphere inside it), and two or
  three a side jump from it to the fingertips, struck anew every frame or two, each bowing through the air off the
  hand and coming down onto its finger past the tip (on the skin only where it lands), white-hot in a cyan-blue glow,
  a white-hot point in a blue glow where it lands. They thin out as it shrinks and go before it does, and its shell
  goes with its body.
- **The voice.** The louder the voice, the further a channel reaches (to most of the frame at the loudest), the
  thicker it leaves the finger and the more it forks (two to four forks a main channel, each a clear, dimmer line);
  a hit strikes every channel afresh at once, a little further. Between strikes, and for a moment after the
  voice stops, a crackle of short arcs leaves a steady burning fingertip into the room. The ball grows from a point
  over about 5 frames while the voice holds, a hit flaring its middle, and shrinks when it stops, dense and bright at
  any size; barely charged it has no body yet, only its shell's crackle reaching past where its skin will be.
- **The look.** The channel itself is far over white (a phone clips it to white with a narrow blue rim), the air round
  it glows blue, then violet further out, faintly; only the brightest channels clip (CORE_GAMMA), so a fork stays a
  dimmer line beside its channel. Over a bright wall a channel leaves the finger wider, each new stroke burns at full
  strength for two frames, and the wall round it is darker and bluer, so it stands off it (BRIGHT_*). The channels are
  as soft as the plate round the hand (its focus and the phone's
  smoothing, measured on each frame), and smeared with the finger where they leave it, as the plate's finger is,
  though never so far that a channel comes away from its fingertip. A small bolt glows as hot as a strike's. The glow
  of many channels piled up is held under GLOW_CAP. The fingertip burns white-blue; the ball's middle is a white spark.
- **The light it throws** (cuesync.fx_light's physics): every lit pixel of the channels lights the hand close in front
  of it, the wall just behind it and the room further behind, shaded on the round hand, times each surface's own
  colour, so skin keeps its texture under it: the arcs' blue-violet on the hand, a cooler blue-white on the room,
  divided by the room's own tint so it lands as the same blue on a tinted wall (two fingers light it as one does).
  And the lightning's own image, blurred wide (KEY), lights the plate and, through its matte, the hand, a cool white
  with blue, held under a luminance (KEY_CAP): in a dark room the brightest thing in the frame lights the hand and
  what is round it, flickering with the strikes. Where a bolt strikes, its light pools along the surface it hits.
  On the hand the arcs' light is a cool white: skin they lift a lot goes towards a cool white of its brightness, and
  red never rises much more than green (SKIN_RED, SKIN_COOL), so a fingertip blows to white, not hot pink, while the
  rest of the lit skin keeps most of its colour (SKIN_KEEP), never a grey mauve; the hand's light and the room's are
  blended across its outline (LIGHT_FEATHER). The ball is a light, not a tint: from its middle, as from a ball of its
  radius, its light falls off as the inverse square (softened by how far it is over the palm, or in front of the
  wall) and is gone by three radii, nothing further off changing colour (BALL_LIGHT): on the hand a cool white, so
  pink skin by it goes a cool white rather than lavender, on the room a blue-white; it pools softly on the palm under
  it, apart from the ball, and the edges of the fingers facing it catch a rim of it; skin it lights rolls off with its
  hue kept (BRIGHT_TONE), never washed chalky. The arcs' large-area light does not strobe with the channels (a
  photosensitivity limit: at most about 3 big changes a second): its strength holds the strokes' peak while a channel
  is on screen (and through a gap of a frame or two between strokes), with a share of their flicker, and goes with
  the last of them (a tenth of it, a fifth as shown, for one frame), and its shape is pooled over the last few frames'
  channels. On the hand and the room it is capped (HAND_CAP, ROOM_CAP), at the peak of the burst so the cap does not
  hold it up as it dies. The camera exposes down while the lightning shines (as a phone's does for a bright light in
  the picture: a flash for the arcs, never so far that the room they light goes darker than the plate; an eighth of a
  stop for the ball, settling once for its phrases, held through the gaps between them, and coming back once it has
  gone), so skin already at the top of red has room to take the light, and the plate's own highlights are never
  dimmed by the roll-off to white. Over a bright plate the plate round a channel is a little darker (CONTRAST), as a
  phone's sharpening rings a bright line.

Everything is numpy and OpenCV; the random parts come from ``seed`` (the strikes, in one pass over the frames, then
each frame's own draws from the seed and its index), so the same inputs give the same bytes on any number of threads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np

from . import fx_light as fx
from .elements import FORMS, Timeline
from .hand_track import REACH_OUT, _pointing, finger_reach
from .transfer import gam, lin

_STRIKES, _CRACKLE, _FRAME, _SPARK, _WANDER, _PLASMA, _JUMPS = 11, 12, 13, 14, 16, 17, 18  # mixed into each seed:
# each random part its own stream

# ---- the voice
GATE = (0.15, 0.25)  # energy: under the first a sound is a breath or a click, not the voice (it lit stray sparks on a
# silent hand); over the second it plays in full, faded in between
CRACKLE_LEFT = 0.72  # the crackle at the fingertip outlasts the voice, dying by this share a frame (about 0.3 s)
CRACKLE_MIN = 0.06  # below this the crackle is gone
CRACKLE_WIDTH, CRACKLE_INTEN = 1.6, 0.8  # a crackle arc's width (times the hand's width scale) and intensity: a few
# bright arcs (four 1 px hairs at 0.45 could not be seen at phone size, and the tip glowed with no source)

# ---- arcs from the fingertips
ARC_REACH = (0.15, 0.6)  # the picture's diagonals: how far a main channel reaches, from the quietest voice to the
# loudest, and so how far out what it strikes may be (TARGET_*). In the frame's terms, as a phone shows it (in hand
# sizes, a near hand's bolts went to the frame's edge and a far one's hung short of everything)
ARC_WIDTH = (2.2, 4.8)  # px for a hand HAND_PX across: a main channel's width where it leaves the finger, from the
# quietest voice to the loudest (from 1.3, a fading voice's bolt was a hair at phone size)
MAIN_TAPER = (0.22, 0.7)  # it thins to this share of that width at its far end, as this power of how far along it is
# (thinned only to 0.47 of it, a loud channel 5 px wide all the way to the wall read as neon tubing)
HAND_PX = 220.0  # widths are drawn for a hand this many px across (the dev takes') and scale with it, within WIDTH_K
WIDTH_K = (0.6, 1.2)  # (up to 1.8, a near hand's loud bolt left the finger 7 px wide: a cable, not a 3-5 px core)
ARC_WANDER, ARC_WANDER_HZ = 5.0, (0.6, 1.7)  # degrees and Hz: the main channel's aim sways slowly (a new path anywhere
# in the fan read as random flashes, not one storm)
ARC_JITTER = 4.0  # degrees: each new main channel's own turn off its aim
ARC_OFF = 12.0  # degrees: a main channel's heading is never further off its finger's aim than this (its sway and
# jitter added up), two fingers' each turned PAIR_SPLIT / 2 further, away from the other
ARC_LEAD = 0.2  # of its reach: a main channel's first stretch runs along its finger's aim before it turns to where it
# strikes, so it comes out of the finger however far its end strays
LEAD_TURN = 30.0  # degrees: where that stretch turns more than this, the turn is spread over the lead's second half and
# as far again past it, at most this much a piece (a second finger's channel led out past the first's tip turned at a
# right angle there: an elbow)
LIFE, LIFE_P = (1, 2), (0.45, 0.55)  # frames a path is kept, and how often each: struck anew every frame or two (kept
# 1-3 frames, and 4 over a loud voice from two fingers, the same channels hung in the air between strokes)
RESTROKE = (0.45, 0.5)  # a re-stroke flashes with this chance, plus this times the level; else the channel is dark
HIT_REACH = 0.3  # a hit's strike reaches this much further at full size
ROUGH, FORK = 0.2, (0.08, 0.14)  # a channel's jaggedness (share of a segment), its forking (at the quietest, more when
# louder: a loud strike is one channel forking more, never more channels from one finger)
ROOM_PAD, ROOM_SKIP, REDRAW = 1.5, 0.1, 4  # px at 1080, share of its length, times: a fork may not come within this of
# another channel (its width and theirs apart) past this share of its own length from where it leaves, drawn again up to
# this many times before it is dropped, as is a second finger's main channel; the first finger's is drawn again as
# often, then kept (a fork crossing another channel, or rejoining its own, closed a loop: the most obviously fake shape
# lightning can take; two fingers in line along their aim struck two channels side by side)
ALONG = (26.0, 3.0, 0.25)  # px at 1080, widths, share: nor may it lie within the larger of the first two (its width
# and theirs apart) of another channel for more than this share of its length (a fork ran 30 px beside its channel for
# 150 px, two wires side by side; held a hand's 0.12 off, a near hand's 40 px, every fork of a bolt was dropped)
BRANCH, TAPER = (0.65, 0.85), (0.7, 1.5)  # a channel at most the first share as bright as the brightest drawn with it
# is a branch (a fork), from the second on not, faded between; a branch fades along its length to 1 - 0.7 of itself at
# its end, as the 1.5th power of how far along it is (at one brightness end to end, a thin one read as wire)
BRANCH_W = (1.0, 0.3)  # and thins from this many times its width where it leaves to this at its end: a wedge, not a
# wire (from 1.5 times, with FORK_WIDTH, a fork left a loud channel nearly as thick as it: a second tube)
FORK_INTEN, FORK_WIDTH = (0.45, 0.65), 0.55  # a fork's intensity, and its width against the channel's where it
# leaves it: a clear line under its channel's white, 1-2 px, fading along it (BRANCH) (at 0.35-0.6 and 0.6, unfaded,
# the forks clipped to the same white as their channel, a bundle of equal wires; at 0.15-0.35 and 0.3 they could not be
# seen at phone size, a bolt an unbranched wire)
FORK_MIN, FORK_LEN = 2, (60.0, 200.0)  # a main channel forks at least this many times, each fork this long (px at
# 1080, and at most two thirds of what is left of its channel)
FORK_TURN = ((30.0, 60.0), (15.0, 50.0))  # degrees a fork turns off a main channel's course there, and off a fork's
# (turned 10-35 degrees off its jagged heading, nearly all ran beside their channel and were dropped)
FORK_MOST = 70.0  # degrees: and turned at most this far off its channel's own way out
FORK_PULL = 0.15  # a main channel's fork is pulled this far back towards the channel's own course (turned 50 degrees
# on top of the fan, the wander and the jitter, one crossed the shoes, half off the finger's aim)
FORK_OUT = 0.1  # a fork ends at least this share of its length from the channel it leaves, and no nearer it than its
# middle is, or it is drawn again: it runs out, never back (one curled back to end a few px from its channel, a closed
# loop to the eye)
TARGET_CONE = (25.0, 40.0, 70.0)  # degrees either side of its finger's aim (two fingers' each turned PAIR_SPLIT / 2
# out): a burst's first main channel strikes an edge of the plate within the first, else within the second (a tub's
# rim just past where a finger points), else nothing; a re-strike anywhere within the third, never where the last one
# struck (TARGET_APART): every strike from f30 to f65 of a 1.5 s burst ended in one 140 x 60 px patch of the rim
TARGET_AT = (0.22, 0.4)  # the picture's diagonals: how far out it strikes, from the quietest voice to the loudest
# (0.25-0.45 of a portrait frame's height), preferring what is nearest that (striking as far as it could, a burst's
# every stroke was the same length whatever the voice did)
TARGET_BAND = ((0.5, 1.5), (0.15, 0.5))  # shares of that: how far out what it strikes may be; two fingers' each, with
# none there, within the second (from two fingers pointing along a floor's line, both ran off the frame side by side:
# two stretched cables; now they strike the floor and the wall's foot beside them)
TARGET_BORDER = 0.12  # share of the picture's short side: never this close to its edge (struck as far as it could get
# inside a 4 % margin when nothing was in its way, the bolts all ended pinned to the frame's edge)
TARGET_TOP, TARGET_APART = 3, 150.0  # it strikes one of the best this many edges, each this far from the others (px at
# 1080) and from where it struck last: each new strike lands somewhere new, and a burst's strikes go back and forth
# between two places at least (40 px apart, they were one place)
PRONG = (0.25, (90.0, 240.0), (0.72, 0.85), 0.8)  # (chance, px at 1080, share, against its channel): one main channel
# in four that strikes an edge ends in two prongs, landing on two points this far apart, the second leaving it this far
# along it, as bright and thick against it there
TARGET_EXIT, EXIT_PAD = 1.25, 0.05  # with nothing to strike, it runs off the frame (to this share of the short side
# past its edge) when its heading leaves the frame within this many reaches; else it ends in the air (BRUSH)
EDGE_Q, EDGE_BLUR, EDGE_OFF = 4, 2.0, 0.05  # the plate's edges are read at 1 / EDGE_Q of its size, blurred EDGE_BLUR of
# those px, and none within EDGE_OFF of its short side of the hand (its own outline is no surface to strike)
EDGE_MIN, EDGE_LEN = 22.0, 20  # the gradient (code values a px, Sobel, at that size) a line in the plate must reach,
# and how long it must run (px at that size), to be struck: a floor's line, a door's jamb, a tub's rim (a blank wall's
# grain and a speck are no surface)
IMPACT = (1.4, 3.0, 0.6)  # (sigma px at 1080, amplitude at a stroke's brightness, the least stroke): the white-hot
# point where a channel touches what it strikes, 3-5 px of clipped white, on a stroke at least this bright (a round
# 40 px bead on every end, a dim re-stroke's or a short tail's in the middle of a blank wall, read as a pin)
POOL = (150.0, 40.0, 3.0, 2.5, 0.05, 2.0, 0.8)  # (length px at 1080, core px at 1080, squeeze across the surface,
# gain on the surface's own colour, gain in the air, more on its first frame, at most): and its light pools on that
# surface, falling off as 1 / r from that core along the surface (squeeze times faster across it) out to about its
# length, brightest as it strikes, held softly under the last (at a 16 px core and 110 px it barely lit the floor;
# unheld, two strikes side by side blew a baseboard to a white tube)
POOL_GLOW = (24.0, 7.0, 0.35)  # (px at 1080 along and across the surface, amplitude): the flash in the air round it
SPLASH = ((3, 6), (30.0, 80.0), (0.55, 0.8), 0.45, 2)  # (count, length px at 1080, intensity against its stroke, width
# against its root's, frames): forks splashing out where it strikes, along the surface and back off it, on its first 2
# frames (at 0.25-0.5 of its intensity and 0.25 of its width they could not be seen at any size)
BRUSH = ((2, 3), (0.06, 0.14), (0.3, 0.5), (15.0, 40.0))  # (count, length in its own, intensity, degrees turned): a
# channel that strikes nothing ends in the air in a brush of short dim forks fanning on from its end, as a streamer
# does
FINGER_NEAR = 0.6  # hand sizes: the tap names the fingertip within this of the emitter; a second finger beside it
# (index and middle) may arc too when it is out straight there (REACH_OUT) and its tip within FINGER_PAIR of the
# first's. A tap further from every tip (the palm's) strikes from the index or the middle, whichever reaches further
# along the aim: the arcs leave a fingertip, never the middle of the hand (from the palm, a bolt ran up the fingers and
# a fork down the palm, drawn on the skin)
FINGER_PAIR = 0.8
TIP_OUT = 0.12  # hand sizes: from a fingertip the tap is not on, a channel leaves where the matte ends past its tip
# landmark (which sits a little inside the finger's end), at most this far out along the finger
PAIR_LONG, PAIR_BENT = 0.6, 0.5  # and it strikes on a frame only while the picture shows it out: its middle joint to
# its tip at least this share of the first finger's, bent there by at most acos(this) (60 degrees), its tip further from
# the palm's middle than that joint. MediaPipe's depth called a finger curled into a fist out straight (reach 1.5-1.8,
# 'Victory') with its tip landmark on its knuckle, and a second bolt erupted from the middle of the hand
PAIR_BEHIND = 0.15  # hand sizes: its tip at most this far behind the first's along the first finger (two fingers held
# out together end side by side; that knuckle tip was 0.3-0.7 behind)
PAIR_EDGE = 0.06  # hand sizes: with a matte, its tip on it and the background no further past it, along the finger,
# than past the first's tip plus this: on the silhouette's outline, not inside the hand (a phone's matte runs a little
# past every fingertip, 0.1-0.16 past the index's on IMG_6653, so the first finger's is the measure)
PAIR_HOLD = 3  # frames: the second finger starts or stops striking only once this many frames in a row say so (the
# landmarks jitter about the bounds)
ON_HAND = (0.2, 0.03)  # hand sizes, share: a channel lying on the hand's matte past this far from where it leaves for
# more than this share of its length is drawn again (a fork or a second finger's channel then dropped if it still does;
# the first finger's main channel kept): the bolts are struck into the room behind the hand, never across it (the
# matte runs past the fingertip it leaves, so its first stretch may lie on it)
HIDE_NEAR = (0.08, 0.16)  # hand sizes from a burning fingertip: and further out than this (faded in from the first),
# whatever of the channels or the crackle passes over the hand is hidden by it, as much as the matte covers it; only
# the tip's own spot and the channels' roots stay on top (left on top out to half a hand, a channel crossed the fingers
# and ran down the palm like a drawing on the skin)
PAIR_SPLIT = 28.0  # degrees: two fingers' main channels leave at least this far apart, each turned away from the
# other (at 14, within ARC_OFF, two channels ran side by side from the two tips to the frame's edge: parallel cables)
PAIR_APART = 15.0  # degrees: and where either strikes nothing, root to end at least this far round from the first's,
# or it is not struck (two fingers pointing along a floor ran off the frame side by side: two stretched cables)
PAIR_CLEAR = 0.22  # hand sizes: the second finger's main channel keeps this much further off the first's channels, or
# is not struck (from fingers one behind the other along their aim, it ran beside the first's: a rope of two; at 0.12
# it ran 30-50 px beside it and past its tip)
PAIR_RETRY = (0.0, 4.0, 7.0, 11.0, 15.0, 23.0)  # degrees: it is tried turned this much further outwards each time,
# twice each, before it is not struck (dropped on its first failure, fingers one behind the other struck one channel on
# most frames, the second finger only a jumper into the first's tip, as if feeding its bolt)
PAIR_LEAD = 0.1  # hand sizes: its first stretch (ARC_LEAD) ends this far to its outer side of the aim, and leaves
# turned out as far as it must to pass the first's tip PAIR_CLEAR off it, from a tip behind that one along the aim
# (along the aim, it ran into the first's tip), or as far as it is tried turned, at most PAIR_PAST degrees
PAIR_PAST = 45.0
PAIR_LINED = 0.1  # hand sizes: tips closer than this to one line along the aim are one behind the other: which side
# each strikes on is kept from the last strike (it flipped with the landmarks' jitter)
JUMP_P = 0.25  # an arc jumps between the two tips with this chance a frame, only while both main channels strike (at
# 0.6, with the second channel gone, the jumper read as the second finger feeding the first's bolt)
PAIR_PEAK, PAIR_LIFE = 0.5, 2  # over this level two fingers' paths are kept this many frames and re-struck on each,
# never left dark (new channels on and off every frame over a dark wall flashed the room 3 times a second, the
# photosensitivity limit; kept 4 frames, they hung in the air)
PAIR_SECOND = 0.8  # the second finger's main channel strikes this bright against the first's: one still leads (two
# equal channels at once doubled the white-hot core)
POINT_LONG = (0.15, 0.3)  # hand sizes, the finger's middle joint to its tip in the picture: a finger shorter than the
# first points at its vanishing point (cuesync.hand_track._pointing, in 3D), one longer than the second its own way in
# the picture, blended between (pointing away but long and clear, it was bent towards the picture's middle: arcs
# straight up from a finger pointing up and left)
CRACKLE_FAN = 70.0  # degrees: the crackle's arcs leave the burning tip at most this far off the finger's aim, into the
# room (at 110, some ran back over the finger)

# ---- the ball over the palm
BALL_R = (0.12, 0.5)  # hand sizes: its radius, from a point as the voice begins to one held long and loud, growing
# as the square root of its charge (its area as the charge), so it is soon a ball, and shrinking with it when the voice
# drops (from 0.1 to 0.42, growing over 10 frames, three short sounds held it at a third of a hand across: a flare
# stamped on the palm; at 0.55 a held one covered the palm, pasted over it)
BALL_RISE = 0.2  # its growth (the square root of its charge, 0-1) rises at most this much a frame, its radius eased
# along it (smoothstep): from a crackle to full size over about 5 frames (sized from the charge itself, which jumps at
# the voice's onset, it was 60-65 % of its full size on its first frame: it popped rather than grew)
FIL_REACH = 0.7  # radii: a young ball's shell arcs crackle this much further out than its body, drawn in to its skin as
# it charges (a forming ball is a spark with a crackle round it, not a disc)
BALL_LIFT, LIFT_PX = 0.5, (140.0, 180.0)  # hand sizes, px at 1080: it hovers this far off the palm along the palm's
# normal, and its middle sits between the two px off the palm's point in the picture, up the picture where the normal
# shows no way across it (on the palm's point, square on, it was a sticker on the skin)
WANDER = (15.0, (3.1, 4.6))  # px at 1080, Hz: its middle wanders about this far round where it hovers, at these rates
# (held dead still on its point, it was pasted on)
BALL_ON, BALL_BRIDGE = (0.03, 0.015), 3  # it shows once the voice (its loudest over the last BALL_BRIDGE frames) or
# its charge passes the first, and goes only when both are under the second: as a voice began softly, a frame's dip in
# it, before the charge had built up, put the ball out and on again
BALL_FADE = 3  # frames it fades in over
BALL_GROW, BALL_SHRINK = 6.0, 3.0  # frames: it grows towards the voice's level this slowly, and shrinks this fast
# (at 5 a ball hung on, flickering, through the second's pause between two sounds)
SHELL = (10, 18)  # arcs crawling over its skin, small to full grown: an irregular shell of electrical filaments, which
# is what draws its outline (filaments from its middle out to its skin, and streaks of glow along them, were a radial
# starburst: a lens flare)
SHELL_OUTLINE, SHELL_ROUGH = 0.5, 0.35  # how often a shell arc runs along its outline, and how jagged it is (half of
# them on the outline, whole and smooth, drew a circle round it)
SHELL_SPAN = (50.0, 130.0)  # degrees of its great circle a shell arc runs over (longer, those on its outline joined
# into a ring)
SHELL_INNER = (0.3, (0.5, 0.85))  # share, radii: of them, this many crawl over a sphere this far in, seen through its
# body, so its inside has a structure rather than a blank glow
SHELL_INNER_INTEN, SHELL_BACK = 0.55, 0.4  # an inner arc's brightness, and that of a stretch behind its middle, against
# one on its skin before it
SHELL_FACE = 0.35  # and a stretch facing the camera against one along its outline, where the shell is seen edge on
# and shows densest: the outline is what resolves it as a ball (all as bright, the arcs were scribbles over the palm)
SHELL_FORK = 0.4  # a shell arc forks once with this chance
SHELL_GONE = (0.45, 0.3)  # shares of the largest it grew to: shrinking, its shell fades out between these, and with
# its body once it has been a ball (its crackle, reaching out round a ball all but gone, crawled over the fingers for
# 4 frames with no source)
SHELL_SKIN = 0.05  # radii: a shell arc is held this close to its sphere, however jagged (thrown 30 % past its skin,
# its arcs lay flat on the palm round it like glowing veins)
LAND = (1.6, 2.5, 6.0, 0.3)  # (sigma px at 1080, amplitude; sigma px at 1080, amplitude): where a jumping arc lands
# on a finger, a white-hot point 4-6 px across in a blue glow 12-20 px across (a 14 px white spot over red skin read
# as a pink dot)
KNOT_TURN, KNOT_STRAIGHT = 200.0, 0.45  # a shell arc is drawn again when, cut into 12 equal pieces as seen, it turns
# more than this many degrees one way over 5 of them, or its ends are under this share of its length apart: seen edge
# on, a stretch of great circle folds back on itself into a knot of wire, not a spark
BALL_TURN = 0.05  # radians a frame: it turns slowly (its arcs' far ends move, but never jump)
BALL_PULSE = 0.12  # a hit swells it by this share, gone over 3 frames
BALL_FLARE = 1.0  # and on the frame it lands, its middle's spark flares by this share of the hit's size, its white-hot
# glow by at most INNER_FLARE: the sound's sync, now that the ball grows rather than pops into being
BALL_CORE, BALL_SPARK = 0.09, 2.0  # radii, and amplitude at the loudest: the spark at its middle, flickering and
# flaring on a hit (at 0.05 and 1.2 the ball had no source; at 0.11 it was a flat white disc)
PLASMA_RGB = np.array([0.75, 0.85, 1.0], np.float32)  # linear: the colour of its white-hot middle, a cool near-white
# (lightning and an electric arc glow at about 7000-10000 K); the light it throws is LIGHT_RGB's
BALL_HOT = (2.4, 0.13)  # (gain, sigma in radii): its middle, where the plasma is hottest and densest, glows this far
# over white in PLASMA_RGB, falling off as a Gaussian of that sigma: white-hot out to about a fifth of its radius, a
# core about 60 px across at full size on a phone (out to a third of it, at 3 and 0.2, the ball was blown out to white)
BODY = (0.75, 0.97)  # (gain, radii): the rest of it glows in BODY_RGB, as exp(-(r / that)^4) through it: nearly as
# bright out to two thirds of its radius, then falling off smoothly over its last third and a little past it, so it
# has no outline of its own (its shell's arcs draw it). Under white, so it stays a cyan-blue with the palm showing
# through it and its arcs show white over it (at 0.75 it rolled off to a pale white-cyan: blown out)
BODY_RGB = np.array([0.12, 0.5, 1.0], np.float32)  # a cyan-blue. Green in it matters: added to pink skin, a blue with
# little green made lavender and mauve
HALO = (0.22, 0.8)  # (gain, sigma in radii): and round it, in HALO_RGB, the faint haze of its light scattered in the
# air and the lens, a Gaussian of that sigma, the same light added over skin and wall alike (in the picture as shown, a
# band of saturated blue over its limb read as a drawn ring; at 0.3, with a blue haze, it veiled the palm by the ball)
HALO_VOICE = (0.4, 0.6)  # the haze glows this share of HALO, plus this share times the voice's envelope: it breathes
# with the voice
HALO_RGB = np.array([0.08, 0.35, 1.0], np.float32)  # a blue (in a blue-violet, over skin round a ball this big, it went
# magenta; in a cyan-blue, it washed the palm round it chalky)
BALL_TAU = (3.5, 0.47)  # (optical depth through its middle, sigma in radii): it hides what is behind it as much as it
# is dense, as 1 - exp(-tau), tau falling off as a Gaussian of that sigma: 86 % behind its middle, 40 % at 0.7 of its
# radius, 15 % at its radius; the palm shows through it, scattered (at 4 its middle was an opaque disc)
PLASMA = (0.45, 1.5, (4.0, 9.0, 18.0), (0.5, 0.3, 0.2), (1.0, -0.75, 1.35), 0.15)  # (contrast, turns a second,
# cycles round it, weight and rate of each octave, contrast in its white-hot middle): its body is moving plasma, not an
# airbrushed marble: its glow and density each a share brighter or darker by three octaves of ridged noise on the sphere
# (filaments), the front and, seen through it, the back, each octave turning about its own axis at its own rate, about
# 1.5 turns a second, so it boils and swirls; faded out over its limb, where its haze is smooth
PLASMA_MAP = (128, 256)  # px (latitude, longitude) of the noise on the sphere
SCATTER = 0.35  # radii: what shows through its body is blurred this much, as much as the body hides it (seen sharp
# through it, the wall in the gap between two fingers was a dark window across its top: a glass marble)
BALL_FORMED = 0.08  # its charge (and level, as its body's glow is) at which it is as dense and bright as it gets:
# almost at once, so a young ball is small and bright, not big and faint (at 0.25 a forming ball was a pale
# pink-lavender disc, a soap bubble); under half of it there is no body yet, only its shell's crackle (_dense)
GROUND = (4, 6)  # arcs jumping from its skin to the fingertips, from a quiet voice to the loudest at its full size
# (BALL_R's second), each struck anew every frame or two (LIFE): two or three a side (at 6-10, each ending in a dot, the
# ball read at phone size as a spider standing on white legs), as many fewer as it is smaller, and none once it has
# shrunk under GROUND_GONE of the largest it grew to (they crawled over the fingers for 4 frames after it had gone)
GROUND_GONE = 0.3
TIPS = (4, 8, 12, 16, 20)  # the landmarks a jumping arc goes to: the fingertips, those clear of the ball, spread over
# them (sent to the fingers' joints and bases too, they ran down the skin to a dot mid-finger)
GROUND_BOW = ((30.0, 60.0), (60.0, 110.0))  # px at 1080: each bows through the air, its middle this far off the
# straight line either way, the first tried first and the second once those all lie on the hand (drawn straight,
# they lay on the palm as thin whiskers)
GROUND_LEAVE = 60.0  # degrees: it leaves the ball's skin at most this far round from the way to its fingertip
GROUND_FREE, GROUND_ON = 1.2, 0.04  # radii, share: out to the first from the ball's middle an arc may pass over the
# hand (the ball and the air in front of the palm round it); past it at most the second share of it may lie on the
# hand's matte (its soft edge) but for where it lands (LAND_IN), or it is drawn again, GROUND_TRIES times, then not
# struck: it bows through the air off the skin and touches a finger only at its end (bowed up the picture whatever
# lay there, many lay on the skin like drawn veins)
GROUND_TRIES = 20  # (each of the five fingertips, the fewest arcs first, twice with a short bow, twice with a long one)
LAND_IN, LAND_NEAR = 12.0, 25.0  # px at 1080: it comes down onto the finger from the air past its tip, landing this far
# inside the finger's outline, at most LAND_NEAR from the fingertip's landmark (or it is drawn again)
GROUND_ARC = (2.2, 1.0)  # (width px for a hand HAND_PX across, intensity): a white-hot core in a glow 6-10 px wide
INNER_FLARE = 0.5  # a hit's flare (BALL_FLARE) brightens its white-hot middle by at most this share (more, and the
# light it throws washed the palm round it white)
WRAP_PX, WRAP_GAIN, WRAP_REACH = 12.0, 2.0, 3.0  # px at 1080, gain at the loudest, radii: the fingers' edges facing
# the ball, within about twice the first inside the hand's outline, catch its light (its colour on the hand), falling
# off as the inverse square of their distance, within the last, where all its light is gone (BALL_LIGHT): the inner
# faces of the fingers round it lit (at 5 px, 4 and 3 radii, the fingers next to it went 145 -> 144)
WRAP_BEHIND = 0.15  # radii: the edges behind its body take none of it, coming up over this much past its outline (a
# finger's lit edge seen through the body was a bright seam across it)
SPILL = (0.7, 0.2)  # (sigma in its radii, gain at the loudest): its light pools softly on the palm under it, on the
# palm's point, apart from the ball hovering over it (lit only round the ball itself, the light read as the ball's own
# edge on the skin: a sticker)

# ---- the look (linear light)
CORE_RGB = np.array([0.92, 0.96, 1.0], np.float32)  # the channel itself: white, the faintest blue
CORE_GAIN = 3.0  # the channel is far over white; the shoulder takes it to a clipped white with a blue rim
CORE_GAMMA = 1.6  # the core's intensity is raised to this before CORE_GAIN, so only the brightest channels clip
# (with the gain alone everything over a third went the same white: forks as bright as their channel)
GLOWS = (  # (sigma in hand sizes, peak over a 1 px line at core 1, linear rgb): the air round the channel glows blue,
    (0.008, 0.55, (0.25, 0.50, 1.00)),  # then violet further out (saturated: a paler blue over skin read pink-white);
    (0.025, 0.15, (0.12, 0.30, 1.00)),  # the narrow blue rim kept, the broad halo round it a third under what it was
    (0.08, 0.055, (0.12, 0.22, 1.00)),  # (at 0.22, 0.08 and 0.03, the white core sat in a 48 px blue tube: neon)
    (0.25, 0.02, (0.25, 0.15, 1.00)),
)
GLOW_CAP = {"arc": 0.45, "ball": 1.6}  # linear luminance: the glow is held under this, softly. Arcs over a wall held
# to 0.15 read as white paint, at 0.6 as neon; a ball held to 0.3 had no bloom and was dimmer than the palm round it
BRIGHT_TONE = (0.85, 2.0)  # the ball's own light rolls off to white with each pixel's hue kept: its channels scaled
# by the roll-off of its largest from the first on, and as far over white as that is, by the second's multiples, it
# goes towards white (a share 1 - exp(-over / this)): white-hot in its middle, pale cyan-blue round that (each channel
# rolled off on its own, its body clipped to a flat pale aqua). It does so as much as its light outshines the picture
# under it; the lit skin round it rolls off channel by channel, as skin does, to a pale pink and then white (the two
# ways split at a fixed radius drew a ring round it, where lit skin changed from one to the other)
DEFOCUS = (0.7, 2.0)  # px at 1080: the channels are blurred as much as the plate is soft round the hand, within this
# (drawn crisp over a soft phone picture, they were the cut-out's tell)
SOFT_AREA, SOFT_R, SOFT_AA = 1.5, 1.5, 0.92  # the plate's softness is read within this many hand sizes of the emitter,
# by blurring its strongest edges SOFT_R px more; a crisp anti-aliased edge reads SOFT_AA px, which the drawing already
# has
SMEAR_NEAR = (0.35, 0.7)  # hand sizes: round the fingertip the channels are smeared along its motion as the plate's
# finger is (a 180 degree shutter), in full this close and fading out by the second; the strokes further out are
# instant and stay sharp (crisp arcs on a blurred finger were the cut-out's clearest tell)
SHUTTER = 0.5  # share of a frame the shutter is open
SMEAR_MAX = 12.0  # px at 1080: the smear is never longer than this (as long as a fast hand's 40 px, it spread a thin
# channel across the finger into a faint band, and the bolt began a finger's width off the tip)
WELD, WELD_KEEP = (0.06, 0.1), 0.6  # hand sizes, and share: this close to the fingertip (fading out by the second) a
# smeared channel is at least this much of its sharp self, so a stub always joins the bolt to the tip
CRAWL_STILL = 15.0  # px at 1080: on a frame the fingertip moves further than this no arc crackles out of its tip or
# jumps between two tips (smeared with it, those at the tip were broken hooks and combed dashes)
GLOW_FLOOR, GLOW_HOT = 0.6, (0.4, 0.6)  # an arc's glow is drawn from a channel at least this share of a loud main
# channel's width once its intensity passes the second (faded in from the first): a small bolt on a quiet voice is
# short but as hot as a strike's (with its glow from its own thin, dim line, it read as a drawn line)
SPARK_RGB = np.array([0.86, 0.94, 1.0], np.float32)  # the burning fingertip and the ball's middle: white, a little cyan
BALL_GLOW_RGB = np.array([0.4, 0.82, 1.0], np.float32)  # the glow round a ball's arcs: a cyan-blue,
# blue over its body and the wall, a cool white over skin (in the arcs' blue-violet, added to skin already at the top
# of red, they were magenta scratches on the fingers)
HAZE_RGB = np.array([0.05, 0.35, 1.0], np.float32)  # the halo round a spark: cyan-blue (the violet-blue it was read
# as lavender over skin)
CORONA, CORONA_MIN = 0.07, 0.35  # hand sizes, and its least brightness for the crackle's level: the burning
# fingertip's spot burns while the voice holds (at 0.04 it could not be seen)
CONTRAST = (0.1, 9.0, (0.24, 0.32, 0.85, 0.95))  # (share, sigma px at 1080, linear luminance): over a bright plate,
# the plate round a channel is a share darker within about twice that of it (as a phone's sharpening rings a bright
# line), faded in over the first two luminances (code values 135-155) and out over the last two (a highlight blown out
# stays white), so a thin bolt keeps its contrast without more glow (over a lit wall at 150-200, the channels went
# faint)
BRIGHT_WALL = (125.0, 150.0)  # code values: a plate whose room (off the hand, its median luma) is brighter than the
# first is a bright one, all of it from the second: there the bolts are drawn to stand off it (BRIGHT_*): over a beige
# wall at 155 a channel was a faint 1-2 px scribble that all but vanished at phone size
BRIGHT_WIDTH = 0.4  # it leaves the finger this share wider (a 3-4 px root)
BRIGHT_BAND = (0.08, 5.0, 0.35)  # (share, sigma px at 1080, blue's share of it): the band round it is this much more
# darker (15-20 % in all) and wider (20-30 px), whatever the plate's brightness there, its blue darkened only this
# share as much, so it reads as blue air against the wall
BRIGHT_GLOW = (6.0, 0.4, (0.04, 0.18, 1.0))  # (sigma px at 1080, peak over a 1 px line at core 1, linear rgb): and a
# saturated blue glow 12-16 px across round it
BRIGHT_HOLD = 2  # frames: each new stroke burns at full strength this long before it is re-struck or goes dark

# ---- the light it throws
LIGHT_RGB = {form: tuple(np.array(rgb, np.float32) / float(np.array(rgb, np.float32) @ fx._LUMA) for rgb in pair)
             for form, pair in (
    # its colour, at luminance 1, (on the hand, on the room):
    ("arc", ((0.45, 0.33, 1.0),  # on the hand blue-violet (on skin at the top of red, a greener blue added only green
             (0.30, 0.42, 1.0))),  # and blue: the hand went cyan); on the room a cooler blue-white (in the hand's
    # violet the olive wall went mauve and the white wall and the floor violet)
    ("ball", ((0.7, 1.0, 0.9),  # the ball's on the hand a cool white, as the plasma's light is, so pink skin goes a
              (0.4, 0.66, 1.0))))}  # cool white where it is lit (in a deep blue, the fingers went lavender, their
# saturation 108 -> 63); on the room a blue-white
LIGHT = (0.2, 2.5, 5.0, 1.5)  # the arcs': (how far in front of the hand the channels are, in hand sizes; gain on the
# hand; gain on the room; gain on the wall right behind them, ROOM_NEAR away). A thin channel has no radius to stand off
# by
BALL_LIGHT = (1.9, 0.3, 40.0, 1.0, 3.0)  # the ball's: (gain on the hand, on the room; px at 1080 its middle is over
# the palm, radii the wall is behind it; radii it reaches): a light at its middle, falling on what is round it as from
# a ball of its radius, as R^2 / (d^2 + h^2) at most 1 (d across the picture from its middle, h how far it is over the
# surface), gone smoothly by the last, so nothing further off changes (spread from its image over the whole frame, its
# light shifted the far hallway blue, a colour grade, while the skin 150-250 px from it took +2-6)
BALL_SHADE, BALL_SHADE_R = 0.0, 0.9  # what is seen through its dense middle (out to about the second, radii) takes
# this share of its light, hidden in its glow (lit in full, the palm behind it washed its body a milky white), its
# translucent edge all of it (shaded out to its limb, the lit palm round it outshone its edge: a dark outline)
BALL_LIGHT_VOICE = 0.5  # its light is this share of its brightest at a whisper, the rest with the voice
KEY = (200.0, 110.0, 0.35)  # the arcs': (gain on the hand, on the room, flicker): and the light of the lightning's own
# image, its luminance blurred KEY_SIGMA wide, falls on the plate and through the hand's matte (times each surface's own
# colour, half shaded on the round hand), keeping this share of the strokes' flicker: in a dark room the brightest
# thing in the frame lights the hand and what is round it, flickering with the strikes (with the light close round the
# channels alone, held still, a palm in a dark bathroom went 94 -> 97 and the tub's rim 37 -> 35 under a white bolt)
KEY_SIGMA = 0.28  # share of the picture's short side: how wide the image is blurred (300 px at 1080)
KEY_RGB = tuple(np.array(rgb, np.float32) / float(np.array(rgb, np.float32) @ fx._LUMA)
                for rgb in ((0.62, 0.74, 1.0), (0.5, 0.66, 1.0)))  # its colour at luminance 1, (on the hand, on the
# room): a cool white with blue, the hand's blue rising most
KEY_CAP = (0.32, 0.12)  # linear luminance: it brightens the hand by at most the first and the room by the second,
# softly (held only under twice its own colour, the walls of a lit hallway went white round two bolts)
ROOM_NEAR = 1.0  # hand sizes: the wall the arcs strike is this far behind them (with the room's far light alone, a
# lone bolt lit nothing next to it; at 0.5 the light on it jumped about with each new path)
ROOM_CAST = 1.5  # the room's light is divided by the room's own colour (room_cast), each channel of which is held
# within this factor of grey (on a wall already lavender in the plate, its blue-white graded the room violet)
ROOM_CAP = 2.0 ** 0.5 - 1.0  # the light on the room brightens it by at most this (+0.5 stops), softly, over the
# plate's white (on a dark plate the loudest strikes lit the whole room +0.75 stops)
SKIN_RED = 10.0  # code values: on the hand, the arcs lift red at most this much more than green (lit by a cool light,
# skin already red takes far more of it in red: a fingertip went hot pink)
SKIN_COOL = (140.0, 0.6, (0.94, 0.98, 1.06))  # (code values, share, rgb): and skin lifted this much by them goes up to
# this share of the way to a cool white of its own brightness (the light swamping its colour, as a phone shows a hand
# held into a flash), so a fingertip blows to a cool white rather than lavender and magenta
SKIN_KEEP, SKIN_KEEP_UNDER = 0.6, 235.0  # (share, code value): and lit skin keeps at least this share of its
# saturation, short of where it blows to white (under the arcs' violet blue a pointing finger went a grey mauve, its
# saturation 0.43 of the plate's)
BALL_COOL = 0.25  # the ball's light takes skin this share of the way there (its own blue haze cools it further: at the
# arcs' 0.6 the palm round it went a grey blue)
LIGHT_FEATHER = 3.5  # px at 1080: the light's matte of the hand is feathered this much either way across its outline,
# the hand's light and the room's blended over it (cut at the outline, the two met in a dark seam with a magenta band)
HAND_CAP = 1.0  # the light may brighten the hand by at most this (+1 stop), softly: at fx_light's 4 a palm under the
# ball went white and lost its lines
LIGHT_HOLD, LIGHT_TAIL = 4, 0.1  # the light's strength holds the strokes' peak over this many frames while a channel
# is on screen (following each stroke, the room strobed at 12 Hz), and once none is, keeps this share of itself for a
# frame (about a fifth of the lift as shown, over a dark room), then only what still burns there (let go by 0.55 a
# frame, the room stayed lit with no bolt in it for 4 frames between strikes and after the last)
LIGHT_BRIDGE = 2  # frames: a gap this short between strokes (a "ka-ka-ka", the crackle still burning at the tip) holds
# the light through it (following each stroke there, a bright wall flashed 6 times a second: the photosensitivity limit)
EXPOSURE = {"arc": (0.15, LIGHT_HOLD, 1.5, 2.0, LIGHT_HOLD), "ball": (0.12, 36, 4.0, 8.0, 8)}  # (stops, frames held,
# frames rising, frames falling, frames gone): the camera exposes down this much for the lightning at its brightest (the
# voice's level for the arcs, the ball's brightness), following the brightest it has been over the frames held, so skin
# at the top of red has room to take the light; once the lightning has been gone the last frames, and is not back
# within the frames held (the render is made offline: it can look ahead), it lets go and comes back. The arcs are a
# flash; the ball settles once for a held phrase and stays while it shows (following the ball itself by 0.9 stops, the
# whole room breathed darker and lighter with each sound, three times in 4 s) and through the gaps between phrases (let
# go in a gap, it rose by 0.2 stops and fell again), coming back over about 24 frames once it has gone (held 36 frames,
# the room stayed dark a second after it). An eighth of a stop for the ball: a light a twentieth of the frame across
# (at a third of a stop the whole room went 8-19 levels darker, and the hand under it looked unlit)
ARC_DIP_ROOM = 0.6  # the camera exposes down for the arcs by at most this share of the stops their light lifts the room
# away from them (ROOM_FAR), so the room never goes darker than the plate while they strike (at the loudest, two
# fingers over a lavender wall took it 0.12-0.17 stops darker: the lightning dimmed the room it lit)
ROOM_FAR = 2  # percentile of the room's light over the frame: the room away from the channels (at 10, with the
# light pooled round where a bolt strikes, the corner furthest from it went a hair darker than the plate)
EXPOSURE_OUT = 0.05  # the lightning this faint is out, for the camera (a breath between two phrases put a ball a few
# px across on for three frames, and the room held dark through the gap)
DIP_MIN = 0.005  # stops: less than this and the camera is as it was
LIGHT_FLICKER = 0.06  # share of the strokes' own flicker kept in the light (under the 10 % a flash is counted from)
LIGHT_POOL = (1.0, 0.7, 0.45, 0.25)  # the light's shape: this frame's channels and the three before, weighted (a
# stream swinging across the fan swung the room's light with it)
FIELD_Q = 6  # the light is worked out on a copy this many times smaller (it is smooth)
LOST_FADE = 4  # frames: a hand the tracker lost lets its lightning go over these
CHUNK = 8  # frames made at a time (their layers are kept until lit)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _rot(u: np.ndarray, deg: float) -> np.ndarray:
    """``u`` turned ``deg`` degrees clockwise on screen (y down)."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([u[0] * c - u[1] * s, u[0] * s + u[1] * c])


# ---- channels ------------------------------------------------------------------------------------------------------

def displace(a, b, rng: np.random.Generator, rough: float = ROUGH, min_seg: float = 3.0,
             curve: float = 0.0) -> np.ndarray:
    """A jagged channel from ``a`` to ``b`` (2D or 3D): recursive midpoint displacement down to segments of about
    ``min_seg``, each midpoint pushed across its segment by a Gaussian of ``rough`` times the segment's length (and a
    little along it: kinks, not only sideways). ``curve`` first bows the whole path by that share of its length (2D),
    so a long arc swings rather than runs straight."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return np.stack([a, b])
    pts = np.stack([a, b])
    if curve and a.size == 2:
        n = np.array([-d[1], d[0]]) / length
        pts = np.stack([a, (a + b) / 2 + n * curve * length, b])
    levels = int(np.clip(math.ceil(math.log2(max(length / (len(pts) - 1) / min_seg, 1.0))), 0, 10))
    for lev in range(levels):  # every segment halved at each level: a fractal, no long straight pieces left
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        r = rough * (0.45 + 0.55 * min(1.0, lev / 3.0))  # its course gentle (a channel runs on), its kinks sharp
        dirs = np.diff(pts, axis=0) / np.maximum(seg, 1e-9)[:, None]
        if pts.shape[1] == 2:
            off = np.stack([-dirs[:, 1], dirs[:, 0]], 1) * (rng.normal(0, r, len(seg)) * seg)[:, None]
            off += dirs * (rng.normal(0, 0.2 * r, len(seg)) * seg)[:, None]
        else:  # 3D: any direction across the segment
            off = rng.normal(0, 1, (len(seg), 3))
            off -= (off * dirs).sum(1, keepdims=True) * dirs
            off *= (r * seg)[:, None]
        out = np.empty((2 * len(pts) - 1, pts.shape[1]))
        out[0::2], out[1::2] = pts, (pts[:-1] + pts[1:]) / 2 + off
        pts = out
    return pts


def _loops(pts: np.ndarray, step: float = 4.0) -> bool:
    """Whether a channel (its first two coordinates, as seen) crosses itself: cut into pieces about ``step`` px long,
    any two of them not side by side meet (a closed loop: no kink of a real channel makes one)."""
    p = pts[:, :2]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    if cum[-1] < 3 * step:
        return False
    at = np.linspace(0.0, cum[-1], int(cum[-1] / step) + 1)
    q = np.stack([np.interp(at, cum, p[:, 0]), np.interp(at, cum, p[:, 1])], 1)
    a, d = q[:-1], np.diff(q, axis=0)
    rel = a[None] - a[:, None]  # (i, j): from piece i's start to piece j's
    den = d[:, None, 0] * d[None, :, 1] - d[:, None, 1] * d[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ti = (rel[..., 0] * d[None, :, 1] - rel[..., 1] * d[None, :, 0]) / den
        tj = (rel[..., 0] * d[:, None, 1] - rel[..., 1] * d[:, None, 0]) / den
    meet = (den != 0) & (ti >= 0) & (ti <= 1) & (tj >= 0) & (tj <= 1)
    return bool(np.triu(meet, 2).any())


class _Room:
    """Where the channels struck on a frame lie (a mask of the frame, each drawn ROOM_PAD wider than it is), and how
    far round them is beside them (ALONG: another mask, each drawn that much wider), so a new fork or splash can be
    kept off them and from running alongside them. ``along``: ALONG's distance in px for this frame; ``hand``:
    this frame's matte (uint8) or None, and ``off_hand``: how far from where it leaves a channel may lie on it
    (ON_HAND)."""

    def __init__(self, w: int, h: int, pad: float):
        self.mask = np.zeros((h, w), np.uint8)
        self.beside = np.zeros((h, w), np.uint8)
        self.pad, self.along = pad, 0.0
        self.hand: np.ndarray | None = None
        self.off_hand = 0.0

    def clear(self) -> None:
        self.mask[:] = 0
        self.beside[:] = 0

    def add(self, pts: np.ndarray, width: float) -> None:
        line = [np.round(pts[:, :2] * 4).astype(np.int32).reshape(-1, 1, 2)]
        cv2.polylines(self.mask, line, False, 1, max(2, int(round(width + 2 * self.pad))), cv2.LINE_8, shift=2)  # (a
        # hair 1 px wide could be crossed between its pixels)
        cv2.polylines(self.beside, line, False, 1, int(round(width + 2 * max(self.along, ALONG[1] * width))),
                      cv2.LINE_8, shift=2)

    def free(self, pts: np.ndarray, width: float, near: float, skip: float = ROOM_SKIP, beside: bool = True,
             fan: float = 0.0) -> bool:
        """Whether a channel ``width`` px wide keeps off every channel already here once it is ``skip`` of its length
        and ``near`` px from where it leaves (until then it may lie on the one it leaves), with ``beside`` lies beside
        them (``along``) once it is ``fan`` px out for at most ALONG's share of its length (a fork fanning out of its
        channel lies beside it at first; a splash or a brush, short and at a channel's end, may), (``hand``) lies on
        the hand past ``off_hand`` px from where it leaves for at most ON_HAND's share of it, and never crosses itself
        (_loops)."""
        p = pts[:, :2]
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if _loops(p):
            return False
        h, w = self.mask.shape
        xi, yi = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
        seen = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if self.hand is not None and cum[-1] > 0:
            on = np.zeros(len(p), bool)
            on[seen] = self.hand[yi[seen], xi[seen]] >= 128
            on &= np.linalg.norm(p - p[0], axis=1) > self.off_hand
            if float(seg[on[1:] & on[:-1]].sum()) > ON_HAND[1] * cum[-1]:
                return False
        away = (cum > skip * cum[-1]) & (np.linalg.norm(p - p[0], axis=1) > near)
        if not away.any():
            return True
        if beside and cum[-1] > 0:
            by = np.zeros(len(p), bool)
            by[seen] = self.beside[yi[seen], xi[seen]] > 0
            by &= away & (np.linalg.norm(p - p[0], axis=1) > fan)
            if float(seg[by[1:] & by[:-1]].sum()) > ALONG[2] * cum[-1]:
                return False
        p = p[int(np.argmax(away)):]
        th = max(1, int(round(width)))
        x0, y0 = np.maximum(np.floor(p.min(0)).astype(int) - th, 0)
        x1, y1 = np.minimum(np.ceil(p.max(0)).astype(int) + th + 1, (w, h))
        if x1 <= x0 or y1 <= y0:
            return True
        mine = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.polylines(mine, [np.round((p - (x0, y0)) * 4).astype(np.int32).reshape(-1, 1, 2)], False, 1, th,
                      cv2.LINE_4, shift=2)  # (4-connected over the room's 8-connected: no crossing slips between)
        return not (mine & self.mask[y0:y1, x0:x1]).any()


def _course(p0: np.ndarray, k: np.ndarray, end: np.ndarray, rng: np.random.Generator, rough: float,
            min_seg: float) -> np.ndarray:
    """A jagged channel from ``p0`` through ``k`` (its lead) to ``end``. Where it turns more than LEAD_TURN at ``k``,
    the turn is spread along a curve from halfway along the lead to as far past ``k`` (at most half the rest), in
    pieces that each turn at most LEAD_TURN, so it bends rather than elbows."""
    a, b = k - p0, end - k
    la, lb = float(np.hypot(*a)), float(np.hypot(*b))
    turn = math.degrees(abs(math.atan2(a[0] * b[1] - a[1] * b[0], float(a @ b)))) if la > 1e-9 and lb > 1e-9 else 0.0
    if turn <= LEAD_TURN:
        return np.concatenate([displace(p0, k, rng, rough, min_seg)[:-1],
                               displace(k, end, rng, rough, min_seg, curve=rng.normal(0, 0.07))])
    k1, k2 = k - 0.5 * a, k + b * (min(0.5 * la, 0.5 * lb) / lb)
    m = int(math.ceil(turn / LEAD_TURN))
    tt = np.linspace(0.0, 1.0, m + 1)[:, None]
    base = [p0, *((1 - tt) ** 2 * k1 + 2 * tt * (1 - tt) * k + tt ** 2 * k2), end]
    pieces = [displace(base[i], base[i + 1], rng, rough, min_seg)[:-1] for i in range(len(base) - 2)]
    return np.concatenate(pieces + [displace(base[-2], end, rng, rough, min_seg, curve=rng.normal(0, 0.07))])


def _taper(u: float | np.ndarray) -> float | np.ndarray:
    """A main channel's width ``u`` of the way along it, against its width where it leaves the finger (MAIN_TAPER)."""
    return 1.0 - (1.0 - MAIN_TAPER[0]) * np.asarray(u, np.float64) ** MAIN_TAPER[1]


def bolt_tree(p0, u, length: float, rng: np.random.Generator, width: float, rough: float = ROUGH,
              fork: float = FORK[0], min_seg: float = 3.0, inten: float = 1.0, depth: int = 0, max_depth: int = 2,
              out: list | None = None, room: _Room | None = None, drop: bool = True, lead=None,
              near: float = 0.0, side: int = 0, clear: float = 0.0, tries: int = 1 + REDRAW,
              lead_off=(0.0, 0.0), ok=None, scale: float = 1.0, way=None) -> list[tuple[np.ndarray, float, float]]:
    """A bolt from ``p0`` heading ``u`` for ``length`` px with its forks: [(points, width px, intensity)], the main
    channel first. Forks leave at its points, more often towards its far end (a main channel at least FORK_MIN of
    them, each FORK_LEN long at ``scale`` px a px at 1080; a fork's own a quarter to two thirds of what is left of it),
    FORK_TURN degrees off its local heading (a main channel's pulled back towards its course), never more than
    FORK_MOST off the bolt's own ``way`` (its main channel's, ``u`` for the bolt itself), dimmer and thinner; they fork
    once more themselves. ``side`` (1 clockwise, -1 the other way, 0 either): the side its own forks leave
    on. ``lead``: the way its first ARC_LEAD runs before it turns to its end (that stretch ending ``lead_off`` px
    further; _course). With a ``room`` (_Room), each channel keeps off those already in it past ``near`` px from where
    it leaves, from running beside them, from the hand and from crossing itself (a fork always; the bolt itself, by
    ``clear`` px more and passing ``ok(points)`` if given, drawn ``tries`` times, then dropped if ``drop``, else kept),
    and goes into it."""
    out = [] if out is None else out
    p0, u = np.asarray(p0, np.float64), np.asarray(u, np.float64)
    way = u if way is None else way
    for _ in range(tries if room is not None else 1):
        if lead is None:
            pts = displace(p0, p0 + u * length, rng, rough, min_seg, curve=rng.normal(0, 0.07))
        else:
            k = p0 + np.asarray(lead, np.float64) * ARC_LEAD * length + np.asarray(lead_off, np.float64)
            pts = _course(p0, k, p0 + u * length, rng, rough, min_seg)
        if room is None or room.free(pts, width + clear, near, skip=0.0 if clear else ROOM_SKIP,
                                     fan=1.5 * room.along + 2.0 * width if depth else 0.0) and \
                (ok is None or ok(pts)):  # (one kept clear of the others leaves none of them: it is clear of them all
            # but its own root)
            break
    else:
        if drop:
            return out
    out.append((pts, width, inten))
    if room is not None:
        room.add(pts, width)
    if depth >= max_depth:
        return out
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    alongs = [a for a in np.linspace(0.1, 0.9, 16)  # sixteen places a fork may leave from
              if rng.random() < fork * (0.3 + 1.4 * a) * (0.5 if depth else 1.0)]
    kept, more = 0, 2 * FORK_MIN if depth == 0 else 0  # (a main channel: at least FORK_MIN, towards its far end,
    while alongs or (kept < FORK_MIN and more):  # looked for again where one could not keep clear)
        if alongs:
            along = alongs.pop(0)
        else:
            along, more = float(rng.uniform(0.3, 0.9)), more - 1
        i = int(np.searchsorted(cum, along * cum[-1]))
        j = min(i + max(2, len(pts) // (12 if depth == 0 else 40)), len(pts) - 1)  # (a main channel's course, not
        # its last kink)
        here = _unit(pts[j] - pts[i])
        turn = rng.choice([-1, 1])
        if depth == 0:  # (out of its channel's bend: on the side the rest of it does not swing to)
            ahead = pts[min(i + len(pts) // 4, len(pts) - 1)] - pts[i]
            bend = float(here[0] * ahead[1] - here[1] * ahead[0]) / max(float(np.hypot(*ahead)), 1e-9)
            turn = -1 if bend > 0.05 else 1 if bend < -0.05 else turn
        v = _rot(here, (side or turn) * rng.uniform(*FORK_TURN[min(depth, 1)]))
        left = cum[-1] - cum[i]
        if depth == 0:
            v = _unit((1.0 - FORK_PULL) * v + FORK_PULL * u)
        back = math.degrees(math.atan2(way[0] * v[1] - way[1] * v[0], float(way @ v)))
        if abs(back) > FORK_MOST:  # (never back towards the hand)
            v = _rot(way, math.copysign(FORK_MOST, back))
        if depth == 0:
            ln = min(rng.uniform(*FORK_LEN) * scale, 0.65 * left)
        else:
            ln = left * rng.uniform(0.25, 0.65)
        w_fork = width * FORK_WIDTH * (_taper(along) if depth == 0 else 1.0)  # (as thick as its channel is there)
        had = len(out)
        bolt_tree(pts[i], v, ln, rng, w_fork, rough, fork, min_seg, inten * rng.uniform(*FORK_INTEN), depth + 1,
                  max_depth, out, room, near=2.0 * (width + w_fork + 2.0 * room.pad) if room is not None else 0.0,
                  ok=_leaves(pts), scale=scale, way=way)
        kept += len(out) > had
    return out


def _leaves(parent: np.ndarray):
    """A test that a fork runs out from the ``parent`` channel (points) it leaves, never back (FORK_OUT)."""
    q = parent[:, :2]

    def gap(x: np.ndarray) -> float:
        return float(np.min(np.hypot(q[:, 0] - x[0], q[:, 1] - x[1])))

    def ok(pts: np.ndarray) -> bool:
        p = pts[:, :2]
        end = gap(p[-1])
        return end >= gap(p[len(p) // 2]) and end >= FORK_OUT * float(np.hypot(*(p[-1] - p[0])))

    return ok


def _between(rng: np.random.Generator, width: float, inten: float, rough: float, min_seg: float, bow: float = 0.25):
    """A short arc from (0, 0) to (1, 0), to be carried onto its two ends (_carry): finger to finger, ball to
    fingertip, a crackle at the tip."""
    return [(displace((0.0, 0.0), (1.0, 0.0), rng, rough, min_seg, curve=rng.normal(0, bow)), width, inten)]


def _carry(tree: list, a0: np.ndarray, b0: np.ndarray, a1: np.ndarray, b1: np.ndarray) -> list:
    """The tree with the similarity that takes (a0, b0) onto (a1, b1) applied to its points: its root moved to where
    the hand is now, its far end to where it struck, the shape between kept."""
    d0, d1 = b0 - a0, b1 - a1
    l0 = float(np.hypot(*d0))
    if l0 < 1e-9:
        return [(p - a0 + a1, w, i) for p, w, i in tree]
    k = float(np.hypot(*d1)) / l0
    turn = math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0])
    c, s = math.cos(turn) * k, math.sin(turn) * k
    M = np.array([[c, s], [-s, c]])
    return [((p - a0) @ M + a1, w, i) for p, w, i in tree]


def _bowed(tree: list, a: np.ndarray, b: np.ndarray, lift: np.ndarray) -> list:
    """The tree (drawn from (0, 0) to (1, 0)) carried onto a bow from ``a`` to ``b`` whose middle is ``lift`` (px)
    off the straight line: along a quadratic curve, each point's second coordinate across it, as long as the bow's
    chord."""
    c = (a + b) / 2.0 + 2.0 * lift  # (the curve's control point: its middle is lift off the chord)
    chord = float(np.hypot(*(b - a)))
    out = []
    for p, w, i in tree:
        u, v = p[:, :1], p[:, 1:2]
        at = (1 - u) ** 2 * a + 2 * u * (1 - u) * c + u ** 2 * b
        tan = 2 * (1 - u) * (c - a) + 2 * u * (b - c)
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
        out.append((at + v * chord * np.c_[-tan[:, 1], tan[:, 0]], w, i))
    return out


def _splash(main: np.ndarray, along: np.ndarray, rng: np.random.Generator, width: float, k: float) -> list:
    """The forks splashing out where a main channel (points) strikes a surface running ``along`` (unit) (SPLASH):
    [(points, width px, intensity)], two of every three along the surface, either way, the third back off it the way
    the channel came in, none on into it; ``k``: px a px at 1080."""
    count, length, inten, wide, _ = SPLASH
    p = main[:, :2]
    come = _unit(p[-1] - p[max(0, len(p) - 1 - max(2, len(p) // 10))])
    way = float(rng.choice([-1.0, 1.0]))
    out = []
    for i in range(int(rng.integers(count[0], count[1] + 1))):
        if i % 3 == 2:
            v = _rot(-come, rng.uniform(-50.0, 50.0))
        else:
            way = -way
            v = _rot(along * way, rng.uniform(-30.0, 30.0))
            if v @ come > 0.5:  # (a surface struck at a glance: that way runs on into it)
                v = _unit(v - 2.0 * (v @ come) * come)
        ln = k * rng.uniform(*length)
        out.append((displace(p[-1], p[-1] + v * ln, rng, 1.3 * ROUGH, max(1.5, 1.5 * k), curve=rng.normal(0, 0.1)),
                    width * wide, rng.uniform(*inten)))
    return out


def _prong(tree: list, two: tuple, rng: np.random.Generator, width: float, s: float, room: _Room) -> tuple | None:
    """A second prong for a bolt (``tree``: its main channel first) striking a surface, to ``two`` ((x, y), the way the
    surface runs there): a channel from PRONG's share along its main channel to that point, PRONG's as bright and thick
    as the channel there, kept off the channels in the ``room`` as a fork is (drawn again REDRAW times, then dropped),
    put into the tree and the room; ``two``, or None when it was dropped. ``s``: the hand's size."""
    p = tree[0][0][:, :2]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    for _ in range(1 + REDRAW):
        u = float(rng.uniform(*PRONG[2]))
        i = int(np.searchsorted(cum, u * cum[-1]))
        wd = width * float(_taper(u)) * PRONG[3]
        q = displace(p[i], two[0], rng, ROUGH, max(2.0, 0.008 * s), curve=rng.normal(0, 0.07))
        if room.free(q, wd, 2.0 * (width + wd + 2.0 * room.pad), fan=1.5 * room.along + 2.0 * wd):
            tree.append((q, wd, PRONG[3]))
            room.add(q, wd)
            return two
    return None


def _brush(main: np.ndarray, rng: np.random.Generator, width: float, k: float, room: _Room) -> list:
    """The short dim forks a main channel (points) that strikes nothing ends in (BRUSH): [(points, width px,
    intensity)], fanning on from its last stretch, each kept off the channels in the ``room`` as a fork is (drawn
    again REDRAW times, then dropped) and put into it."""
    count, length, inten, turn = BRUSH
    p = main[:, :2]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    wd = width * MAIN_TAPER[0] * 1.5
    out = []
    for _ in range(int(rng.integers(count[0], count[1] + 1))):
        for _ in range(1 + REDRAW):
            i = min(int(np.searchsorted(cum, rng.uniform(0.82, 0.96) * cum[-1])), len(p) - 2)
            j = min(i + max(2, len(p) // 30), len(p) - 1)
            v = _rot(_unit(p[j] - p[i]), float(rng.choice([-1.0, 1.0])) * rng.uniform(*turn))
            q = displace(p[i], p[i] + v * cum[-1] * rng.uniform(*length), rng, 1.3 * ROUGH, max(1.5, 1.5 * k),
                         curve=rng.normal(0, 0.1))
            if room.free(q, wd, 2.0 * (width + wd + 2.0 * room.pad), fan=1.5 * room.along + 2.0 * wd):
                out.append((q, wd, rng.uniform(*inten)))
                room.add(q, wd)
                break
    return out


@dataclass
class _Path:
    """A channel kept for its life: struck on ``born`` with its root at ``a`` and its far end at ``b``; ``to``: the
    landmark a jumping arc goes to, ``lift``: how far it bows through the air (px at 1080, to its chord's left as it
    runs, or its right under 0) and ``leave``: how far round the ball's skin from the way to its landmark it leaves
    (radians); ``splash``: the forks splashing out of where it strikes (_splash); ``along``: the surface it strikes
    there (unit), or None (it strikes nothing); ``prong``: where its second prong lands and the way the surface runs
    there, or None."""

    born: int
    life: int
    a: np.ndarray
    b: np.ndarray
    tree: list
    to: int = -1
    splash: list = field(default_factory=list)
    along: np.ndarray | None = None
    lift: float = 0.0
    leave: float = 0.0
    prong: tuple | None = None


# ---- the hand ------------------------------------------------------------------------------------------------------

@dataclass
class _Hand:
    """The track in the plate's pixels: the emitter (n, 2), its direction (n, 2), the hand's size (n,), the landmarks
    (n, 21, 3: MediaPipe's z in px; a frame without them holds the last ones seen) or None, how much of the element
    shows on each frame (n,) for where the hand is, and whether the tracker only held the hand there (n,: it had not
    seen it yet, or had lost it)."""

    xy: np.ndarray
    u: np.ndarray
    s: np.ndarray
    lm3: np.ndarray | None
    presence: np.ndarray
    size: tuple[int, int]  # the plate's (w, h)
    held: np.ndarray | None = None

    @property
    def lm(self) -> np.ndarray | None:
        return None if self.lm3 is None else self.lm3[..., :2]


def _hand(plate: np.ndarray, track: dict) -> _Hand:
    """The track read onto the plate (_Hand). The lightning shows where the tracker saw the hand: not before the hand
    came into the picture (the tracker holds its first sight back to the clip's start), and let go over LOST_FADE
    frames when the tracker lost a hand that moved on (a still one it lost keeps it)."""
    from .bending import still_there

    n, h, w = plate.shape[:3]
    xy, u, s = fx.track_arrays(track, n, w, h)
    kx, ky = w / float(track.get("W", w)), h / float(track.get("H", h))
    lms = (track.get("landmarks") or [])[:n]
    lm3 = None
    if any(v is not None for v in lms):  # (cuesync.bending.filled_landmarks, with the depth kept)
        rows = [None if v is None else np.asarray(v, np.float64).reshape(21, -1) for v in lms]
        rows = [None if r is None else np.c_[r[:, :2], r[:, 2] if r.shape[1] > 2 else np.zeros(21)] for r in rows]
        last = next(r for r in rows if r is not None)
        lm3 = np.zeros((n, 21, 3))
        for t in range(n):
            last = rows[t] if t < len(rows) and rows[t] is not None else last
            lm3[t] = last * (kx, ky, math.sqrt(kx * ky))
    here = track | {"xy": xy.tolist(), "dir": u.tolist(), "scale": s.tolist()}
    src = list(track.get("src") or [])[:n] + ["mp"] * n
    presence = np.ones(n)
    first = next((t for t in range(n) if src[t] not in ("hold", "lost")), None)
    if first is not None:  # the tracker holds its first sight back to the clip's start: a hand that came in later
        for t in range(first):  # wasn't there yet
            if not still_there(plate, here, first, t):
                presence[t] = 0.0
        run, seen = 0, first
        for t in range(first, n):  # lost: let it go rather than freeze it (unless nothing moved where it was)
            gone = src[t] == "lost" and not still_there(plate, here, seen, t)
            run = run + 1 if gone else 0
            seen = t if src[t] not in ("lost", "hold") else seen
            presence[t] = min(presence[t], max(0.0, 1.0 - run / LOST_FADE))
    return _Hand(xy, u, s, lm3, presence, (w, h), np.array([v in ("hold", "lost") for v in src[:n]]))


def _smooth(v: np.ndarray, k: int = 2) -> np.ndarray:
    """Zero-phase smoothing over +-k frames (triangular weights) along axis 0."""
    wts = np.concatenate([np.arange(1, k + 2), np.arange(k, 0, -1)]).astype(np.float64)
    p = np.pad(v, [(k, k)] + [(0, 0)] * (v.ndim - 1), mode="edge")
    return sum(wt * p[i: i + len(v)] for i, wt in enumerate(wts)) / wts.sum()


def _fingers(hand: _Hand, tap: int, aim: float | None = None) -> list[int]:
    """The fingertip landmarks the arcs leave from: the one the emitter was carried onto at the tap (the nearest
    within FINGER_NEAR hands of it) or, with the tap further from every tip (on the palm), the index or the middle,
    whichever reaches further along the aim (``aim``, degrees clockwise from up; the track's own direction without
    one); and the index or middle finger beside it when that one is out straight next to it (two fingers together).
    [] with no landmarks: the emitter alone."""
    if hand.lm is None:
        return []
    k = int(np.clip(tap, 0, len(hand.xy) - 1))
    lm, s = hand.lm[k], hand.s[k]
    tips = (4, 8, 12, 16, 20)
    d = {f: float(np.linalg.norm(lm[f] - hand.xy[k])) for f in tips}
    first = min(d, key=d.get)
    if d[first] > FINGER_NEAR * s:
        u = hand.u[k] if aim is None else np.array([math.sin(math.radians(aim)), -math.cos(math.radians(aim))])
        first = max((8, 12), key=lambda f: float((lm[f] - hand.xy[k]) @ u))
    pair = {8: 12, 12: 8}.get(first)
    if pair is not None and finger_reach(hand.lm3[k])[(pair - 8) // 4] > REACH_OUT and \
            np.linalg.norm(lm[pair] - lm[first]) < FINGER_PAIR * s:
        return [first, pair]
    return [first]


def _past_tip(matte: np.ndarray, tip: np.ndarray, u: np.ndarray, s: float) -> float | None:
    """Hand sizes from a fingertip along its finger ``u`` (unit) to where the matte (uint8) gives way to the background
    (under a quarter), searched from a little behind the tip to well past it; None when the tip is not on the hand
    (nothing over half there or just behind it) or the matte runs on past the search."""
    h, w = matte.shape
    inside = False
    for k in np.arange(-0.05, 0.8, 0.01):
        x, y = (int(round(v)) for v in tip + u * (k * s))
        a = int(matte[y, x]) if 0 <= x < w and 0 <= y < h else 0
        if not inside:
            if a >= 128:
                inside = True
            elif k > 0.0:
                return None
        elif a < 64:
            return float(k)
    return None


def _pair_out(hand: _Hand, fingers: list[int], matte: np.ndarray | None = None) -> np.ndarray:
    """(n,) bool: on which frames the second of two ``fingers`` (_fingers) strikes, as the picture shows it: out
    (PAIR_LONG, PAIR_BENT, its tip further from the palm's middle than its middle joint), its tip beside the first's
    (PAIR_BEHIND) and, with a ``matte`` (n, h, w uint8), on the silhouette's outline as the first's is (PAIR_EDGE),
    changing only once PAIR_HOLD frames in a row agree. A curled finger strikes nothing, whatever MediaPipe's depth
    says of it."""
    n = len(hand.xy)
    if len(fingers) < 2:
        return np.zeros(n, bool)
    first, pair = fingers
    lm = hand.lm
    raw = np.zeros(n, bool)
    for t in range(n):
        s = float(hand.s[t])
        v1, v2, b2 = lm[t, first] - lm[t, first - 2], lm[t, pair] - lm[t, pair - 2], lm[t, pair - 2] - lm[t, pair - 3]
        l1, l2, lb = (float(np.hypot(*v)) for v in (v1, v2, b2))
        if min(l1, l2, lb) < 1e-6:
            continue
        palm = lm[t, [0, 5, 9, 13, 17]].mean(0)
        ok = l2 >= PAIR_LONG * l1 and float(v2 @ b2) >= PAIR_BENT * l2 * lb and \
            np.hypot(*(lm[t, pair] - palm)) > np.hypot(*(lm[t, pair - 2] - palm)) and \
            float((lm[t, first] - lm[t, pair]) @ v1) / l1 <= PAIR_BEHIND * s
        if ok and matte is not None:
            mine, theirs = _past_tip(matte[t], lm[t, pair], v2 / l2, s), _past_tip(matte[t], lm[t, first], v1 / l1, s)
            ok = mine is not None and (theirs is None or mine <= theirs + PAIR_EDGE)
        raw[t] = ok
    out = np.zeros(n, bool)
    state, run = bool(raw[:PAIR_HOLD].sum() * 2 > min(n, PAIR_HOLD)), 0
    for t in range(n):
        run = run + 1 if raw[t] != state else 0
        if run >= PAIR_HOLD:
            state, run = bool(raw[t]), 0
        out[t] = state
    return out


def _tips(hand: _Hand, fingers: list[int], tap: int, matte: np.ndarray | None = None) -> np.ndarray | None:
    """(len(fingers), n, 2): where each finger's channel leaves it on every frame. With the tap on the first one's
    tip, the emitter (carried with the hand from where the performer tapped) and the other tip as far from it as the
    landmarks are; else each fingertip landmark carried out along its last joint to where the ``matte`` (n, h, w uint8,
    or None) ends past it (TIP_OUT at most), smoothed over a few frames. None with no fingers: the emitter alone."""
    if not fingers:
        return None
    n = len(hand.xy)
    k = int(np.clip(tap, 0, n - 1))
    lm = hand.lm
    if np.linalg.norm(lm[k, fingers[0]] - hand.xy[k]) <= FINGER_NEAR * hand.s[k]:
        return np.stack([hand.xy + (lm[:, f] - lm[:, fingers[0]]) for f in fingers])
    out = np.stack([lm[:, f] for f in fingers]).astype(np.float64)
    if matte is not None:
        for i, f in enumerate(fingers):
            for t in range(n):
                v = lm[t, f] - lm[t, f - 1]
                ln = float(np.hypot(*v))
                if ln > 1e-6:
                    past = _past_tip(matte[t], lm[t, f], v / ln, float(hand.s[t]))
                    out[i, t] += v / ln * float(np.clip(past or 0.0, 0.0, TIP_OUT)) * hand.s[t]
    return np.stack([_smooth(o, 1) for o in out])


def _aims(hand: _Hand, timeline: Timeline, finger: int | None, frame: tuple[int, int]) -> np.ndarray:
    """(n, 2) unit: which way the arcs leave on each frame. The studio's aim carried with the hand (the track's
    direction turned so it points the aim on the tap frame, which it already does when cuesync.bending.anchored turned
    it); without one, the finger's own, from its middle joint to its tip, smoothed over a few frames (the landmarks
    jitter): as it shows in the picture where it is long and clear there, in 3D where it is short (cuesync.hand_track's:
    a finger pointing into the scene points at its vanishing point), blended over POINT_LONG. From the knuckle, a
    finger crooked as in a finger gun aimed 20 degrees off its last joint, which is what the eye follows. Where the
    finger points at the camera and is too short in the picture to read, the track's own direction."""
    n = len(hand.xy)
    if timeline.aim is not None or finger is None:
        u = hand.u
        if timeline.aim is not None:
            k = int(np.clip(timeline.tap_frame, 0, n - 1))
            now = math.degrees(math.atan2(u[k, 0], -u[k, 1]))
            u = np.stack([_rot(v, (timeline.aim - now + 180.0) % 360.0 - 180.0) for v in u])
        return _unit(u)
    seen = hand.lm[:, finger] - hand.lm[:, finger - 2]
    long = (np.linalg.norm(seen, axis=-1) / hand.s)[:, None]
    deep = np.clip((POINT_LONG[1] - long) / (POINT_LONG[1] - POINT_LONG[0]), 0, 1)
    d = _unit(deep * _pointing(hand.lm3[:, finger], hand.lm3[:, finger - 2], frame) + (1 - deep) * _unit(seen))
    sure = np.clip((long - 0.1) / 0.15, 0, 1)
    return _unit(_smooth(d * sure + hand.u * (1 - sure)))


# ---- the voice -----------------------------------------------------------------------------------------------------

def levels(timeline: Timeline) -> np.ndarray:
    """(n,) the voice as the lightning plays it: the energy, faded out under GATE (a breath is not the voice)."""
    e = timeline.energy.astype(np.float64)
    return e * np.clip((e - GATE[0]) / (GATE[1] - GATE[0]), 0.0, 1.0)


def _follow(x: np.ndarray, up: float, down: float) -> np.ndarray:
    """``x`` followed with a time constant of ``up`` frames rising, ``down`` falling."""
    out, c = np.zeros_like(x), 0.0
    for t, v in enumerate(x):
        c += (v - c) * (1.0 - math.exp(-1.0 / (up if v > c else down)))
        out[t] = c
    return out


# ---- the strikes ---------------------------------------------------------------------------------------------------

def _width_k(s: float) -> float:
    return float(np.clip(s / HAND_PX, *WIDTH_K))


def _bright(plate: np.ndarray, matte: np.ndarray | None = None) -> float:
    """How much the ``plate`` ((n, h, w, 3) uint8) is a bright one, 0-1 (BRIGHT_WALL): the median luma of its room (off
    the hand's ``matte``, (n, h, w) uint8, or None) over a few frames at a quarter of its size."""
    n = len(plate)
    ys = []
    for t in np.unique(np.linspace(0, n - 1, min(n, 6)).round().astype(int)):
        g = cv2.cvtColor(np.ascontiguousarray(plate[t, ::4, ::4]), cv2.COLOR_RGB2GRAY)
        ys.append(g[matte[t, ::4, ::4] < 128] if matte is not None else g.ravel())
    y = float(np.median(np.concatenate(ys))) if ys and sum(len(v) for v in ys) else 0.0
    return float(np.clip((y - BRIGHT_WALL[0]) / (BRIGHT_WALL[1] - BRIGHT_WALL[0]), 0.0, 1.0))


def surfaces(plate: np.ndarray, matte: np.ndarray | None = None) -> list[np.ndarray]:
    """Per frame, the edges of the plate a bolt may strike, (k, 5) float32 rows (x, y, strength, ax, ay) in the
    plate's px: the lines along which its surfaces meet (a floor's line, a door's jamb, a tub's rim), found on its
    luminance at 1 / EDGE_Q of its size blurred EDGE_BLUR (Canny's thin lines), each point's gradient there at least
    EDGE_MIN and its line at least EDGE_LEN long, none within EDGE_OFF of the hand (``matte``, (n, h, w) uint8, or
    None), whose own outline is no surface; (ax, ay): which way the line runs there (unit)."""
    n, h, w = plate.shape[:3]
    hq, wq = max(1, h // EDGE_Q), max(1, w // EDGE_Q)
    off = max(1, int(round(EDGE_OFF * min(h, w) / EDGE_Q)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * off + 1, 2 * off + 1))
    out = []
    for t in range(n):
        g = cv2.resize(cv2.cvtColor(np.ascontiguousarray(plate[t]), cv2.COLOR_RGB2GRAY), (wq, hq),
                       interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), EDGE_BLUR)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        mag = np.hypot(gx, gy)
        line = cv2.Canny(np.clip(np.round(g), 0, 255).astype(np.uint8), 0.5 * EDGE_MIN, EDGE_MIN, L2gradient=True)
        line[mag < 0.5 * EDGE_MIN] = 0
        if matte is not None:
            m = cv2.resize(matte[t], (wq, hq), interpolation=cv2.INTER_AREA) >= 64
            line[cv2.dilate(m.astype(np.uint8), ker) > 0] = 0
        count, label, stats, _ = cv2.connectedComponentsWithStats((line > 0).astype(np.uint8), connectivity=8)
        long = np.zeros(count, bool)
        long[1:] = stats[1:, cv2.CC_STAT_AREA] >= EDGE_LEN
        ys, xs = np.nonzero(long[label] & (mag >= EDGE_MIN))
        a = np.stack([-gy[ys, xs], gx[ys, xs]], 1) / np.maximum(mag[ys, xs], 1e-6)[:, None]
        out.append(np.c_[(xs + 0.5) * (w / wq) - 0.5, (ys + 0.5) * (h / hq) - 0.5, mag[ys, xs], a].astype(np.float32))
    return out


def _exit(root: np.ndarray, v: np.ndarray, size: tuple[int, int]) -> float:
    """px from ``root`` along ``v`` (unit) to where it leaves the frame (``size``: w, h); 0 from outside it."""
    w, h = size
    span = [(((w, h)[k] - 1 if v[k] > 0 else 0.0) - root[k]) / v[k] for k in (0, 1) if abs(v[k]) > 1e-9]
    return max(0.0, min(span, default=0.0))


def _target(root: np.ndarray, aim: np.ndarray, turn: float, reach: float, want: float, size: tuple[int, int],
            edges: np.ndarray | None, rng: np.random.Generator, centre: float = 0.0, again: bool = False,
            clear_of: tuple = (), near: bool = False, prong: bool = False) \
        -> tuple[np.ndarray, np.ndarray | None, tuple | None]:
    """Where a main channel from ``root`` struck ``turn`` degrees off its finger's ``aim`` ends, the surface it
    strikes there ((x, y), (ax, ay) the way it runs, or None: nothing) and, with ``prong``, a second point it lands on
    too ((x, y), (ax, ay)), or None. Of the plate's ``edges`` (this frame's, from surfaces) TARGET_CONE degrees either
    side of ``centre`` degrees off the aim (a burst's first strike within its first cone, else its second; struck
    ``again``, within its third), TARGET_APART from the points ``clear_of`` (where it struck last, where another
    finger strikes now), TARGET_BAND's first of ``want`` px
    out (with ``near``, its second where none is) and TARGET_BORDER inside the frame (``size``: w, h): the strongest,
    nearest ``want`` px out and nearest its heading, with a little chance in it; then one of the best TARGET_TOP of
    those, TARGET_APART from each other. The second point: the best of those PRONG's px from the first. With none, it
    runs on along its heading off the frame (EXIT_PAD past its edge) where it leaves the frame within TARGET_EXIT
    ``reach``-es, else for its reach."""
    w, h = size
    k = min(w, h) / 1080.0
    if edges is not None and len(edges):
        border = TARGET_BORDER * min(w, h)
        apart = TARGET_APART * k
        x, y = edges[:, 0], edges[:, 1]
        d = edges[:, :2] - root
        dist = np.hypot(d[:, 0], d[:, 1])
        off = np.degrees(np.arctan2(aim[0] * d[:, 1] - aim[1] * d[:, 0], d @ aim)) - centre  # (clockwise, as _rot)
        ok = (x >= border) & (x <= w - 1 - border) & (y >= border) & (y <= h - 1 - border)
        for q in clear_of:
            ok &= np.hypot(x - q[0], y - q[1]) >= apart
        here = np.zeros(len(edges), bool)
        for band in TARGET_BAND[:2 if near else 1]:
            for cone in TARGET_CONE[2:] if again else TARGET_CONE[:2]:
                here = ok & (np.abs(off) <= cone) & (dist >= band[0] * want) & (dist <= band[1] * want)
                if here.any():
                    break
            if here.any():
                break
        if here.any():
            i = np.flatnonzero(here)
            score = (np.minimum(edges[i, 2] / (2.0 * EDGE_MIN), 1.0) - np.abs(dist[i] - want) / want
                     - 0.5 * np.abs(off[i] - (turn - centre)) / cone + 0.3 * rng.random(len(i)))
            order = i[np.argsort(-score)]
            best: list[int] = []
            for j in order:
                if any(np.hypot(*(edges[j, :2] - edges[b, :2])) < apart for b in best):
                    continue
                best.append(int(j))
                if len(best) == TARGET_TOP:
                    break
            j = best[int(rng.integers(len(best)))]
            two = None
            if prong:
                gap = np.hypot(*(edges[order, :2] - edges[j, :2]).T)
                fit = order[(gap >= PRONG[1][0] * k) & (gap <= PRONG[1][1] * k)]
                if fit.size:
                    two = edges[fit[0], :2].astype(np.float64), edges[fit[0], 3:5].astype(np.float64)
            return edges[j, :2].astype(np.float64), edges[j, 3:5].astype(np.float64), two
    v = _rot(aim, turn)
    out = _exit(root, v, size)
    return root + v * (out + EXIT_PAD * min(w, h) if 0 < out <= TARGET_EXIT * reach else reach), None, None


def arc_frames(hand: _Hand, timeline: Timeline, seed: int = 0, matte: np.ndarray | None = None,
               plate: np.ndarray | None = None) -> tuple[list[list], list[list], list[list]]:
    """Per frame, the arcs from the fingertips: ([(points, width px, intensity)], [(x, y, sigma, amplitude)] the
    burning fingertips the arcs leave from, [(x, y, ax, ay, gain, whether it struck there on this frame)] where they
    strike a surface running (ax, ay), for the contact flash). One main channel leaves each striking fingertip, struck,
    re-struck and let go as the module docstring says, at an edge of what the ``plate`` ((n, h, w, 3) uint8, or None:
    the frame alone) shows in its way (surfaces, _target: as far out as the voice is loud, a new place each time it is
    struck, now and then in two prongs, _prong), splashing there for its first frames, or off the frame, or into the
    air ending in a brush; a
    second finger strikes only on the frames the picture shows it out (_pair_out), its tip burning and crackling only
    while it strikes. ``Timeline.camera`` carries a far end with the room on a handheld shot; the hand's ``matte`` ((n,
    h, w) uint8, or None) keeps the channels off the hand (ON_HAND), places a fingertip's root and judges the second
    finger."""
    from .bending import camera_between

    n = timeline.n
    L = levels(timeline)
    crackle = np.zeros(n)
    for t in range(n):
        crackle[t] = max(L[t], crackle[t - 1] * CRACKLE_LEFT if t else 0.0)
    fingers = _fingers(hand, timeline.tap_frame, timeline.aim)
    both = _pair_out(hand, fingers, matte)
    tips = _tips(hand, fingers, timeline.tap_frame, matte)
    aims = _aims(hand, timeline, fingers[0] if fingers else None, hand.size)
    edges = None if plate is None else surfaces(plate, matte)
    bright = 0.0 if plate is None else _bright(plate, matte)
    hold = 1 + int(round((BRIGHT_HOLD - 1) * bright))  # (frames a new stroke burns at full strength)
    hits = dict(zip(timeline.hits, timeline.sizes))
    step = np.hypot(*np.diff(hand.xy, axis=0).T) if n > 1 else np.zeros(0)  # px a frame, either side of each frame
    speed = np.maximum(np.r_[0.0, step], np.r_[step, 0.0]) if n > 1 else np.zeros(n)
    if hand.held is not None:  # (a hand the tracker only held may be moving any way)
        speed[hand.held] = np.inf
    k = min(hand.size) / 1080.0
    crawl_still = CRAWL_STILL * k
    diag = float(np.hypot(*hand.size))
    rng = np.random.default_rng([seed & 0xFFFFFFFF, _STRIKES])
    sway = rng.uniform(0, 2 * np.pi, (max(1, len(fingers)), 2))
    paths: dict[int, _Path] = {}  # each finger's main channel
    hit_at: dict[int, np.ndarray] = {}  # where each finger last struck a surface in this burst
    room = _Room(*hand.size, ROOM_PAD * min(hand.size) / 1080.0)
    bolts, spots, struck = [], [], []
    outer = 1  # two fingers: the side of the aim the second's tip is on, its channel turned that way, the first's
    # the other
    for t in range(n):
        s, lv, pres = float(hand.s[t]), float(L[t]), float(hand.presence[t])
        if pres <= 0 or (lv <= 0 and crackle[t] < CRACKLE_MIN):
            paths.clear()
            hit_at.clear()
            bolts.append([])
            spots.append([])
            struck.append([])
            continue
        kw = _width_k(s)
        out_now = fingers if both[t] else fingers[:1]
        roots = [hand.xy[t]] if tips is None else [tips[i, t] for i in range(len(out_now))]
        pair = len(roots) > 1
        if not pair:  # (a second finger curled: its channel goes)
            paths.pop(1, None)
        if lv <= 0:  # (the voice has stopped: the crackle at the tip alone)
            paths.clear()
            hit_at.clear()
        peak = pair and lv > PAIR_PEAK
        hit = hits.get(t)
        lit, out, at = 0.0, [], []
        across = _rot(aims[t], 90)
        if pair:  # (one tip behind the other along the aim: each keeps the side it struck on)
            lat = float((roots[1] - roots[0]) @ across)
            outer = (1 if lat > 0 else -1) if abs(lat) >= PAIR_LINED * s else outer
        room.along = ALONG[0] * k
        room.hand, room.off_hand = (None if matte is None else matte[t]), ON_HAND[0] * s
        here = None if edges is None else edges[t]
        # which channels are struck anew: a finger's when its life is over, on a hit, or the second's when the first's
        # is. Then, the first finger's first, each keeps off those before it (_Room): a new one as it is struck, a kept
        # one where it is now (carried with the hand, it can come to cross the other), or it goes; the first stays
        anew: dict[int, bool] = {}
        for fi in range(len(roots) if lv > 0 else 0):
            p = paths.get(fi)
            anew[fi] = p is None or t - p.born >= p.life or hit is not None or bool(fi and anew[0])
        room.clear()
        flashes = [0.0] * len(roots)
        for fi in sorted(anew):
            root = roots[fi]
            away = (outer if fi else -outer) if pair else 0  # (which side of two fingers this one is on)
            if anew[fi]:
                prng = np.random.default_rng(int(rng.integers(1 << 62)))
                width = kw * (ARC_WIDTH[0] + (ARC_WIDTH[1] - ARC_WIDTH[0]) * lv) * (1.0 + BRIGHT_WIDTH * bright)
                fork = FORK[0] + FORK[1] * lv
                reach = diag * (ARC_REACH[0] + (ARC_REACH[1] - ARC_REACH[0]) * lv) * prng.uniform(0.8, 1.05)
                reach *= 1.0 + HIT_REACH * (hit or 0.0)
                want = diag * (TARGET_AT[0] + (TARGET_AT[1] - TARGET_AT[0]) * lv) * prng.uniform(0.9, 1.1) * \
                    (1.0 + HIT_REACH * (hit or 0.0))
                sec = 2 * math.pi * t / fx.FPS
                wander = ARC_WANDER * (0.6 * math.sin(ARC_WANDER_HZ[0] * sec + sway[fi, 0])
                                       + 0.4 * math.sin(ARC_WANDER_HZ[1] * sec + sway[fi, 1]))
                split = 0.5 * PAIR_SPLIT * away
                turn = float(np.clip(wander + prng.normal(0, ARC_JITTER), -ARC_OFF, ARC_OFF)) + split
                again = fi in hit_at
                clear_of = tuple(q for q in (hit_at.get(fi), None if fi == 0 or paths.get(0) is None or
                                             paths[0].along is None else paths[0].b) if q is not None)
                two = None
                if fi == 0:  # along the swaying aim, out of the finger along it, to what it strikes
                    end, along, two = _target(root, aims[t], turn, reach, want, hand.size, here, prng, split, again,
                                              clear_of, pair, prong=prng.random() < PRONG[0])
                    tree = bolt_tree(root, _unit(end - root), float(np.hypot(*(end - root))), prng, width, ROUGH, fork,
                                     max(2.0, 0.008 * s), room=room, drop=False, lead=aims[t], near=0.15 * s,
                                     side=away, scale=k)  # (two fingers' forks leave on their outer sides)
                    two = None if two is None else _prong(tree, two, prng, width, s, room)
                else:  # the second finger's: turned further out each time it would come near the first's, its first
                    # stretch led out to its side, past the first's tip when it is behind that along the aim
                    behind = float((roots[0] - root) @ aims[t])
                    short = PAIR_CLEAR * s + 2.0 * width - float((root - roots[0]) @ across) * away
                    past = math.degrees(math.atan2(max(short, 0.0), behind)) if behind > 0 else 0.0
                    first = None if 0 not in paths else _unit(paths[0].b - roots[0])
                    for more in PAIR_RETRY:
                        end, along, _ = _target(root, aims[t], turn + away * more, reach, want, hand.size, here,
                                                prng, split + away * more, again, clear_of, near=True)
                        tree = bolt_tree(root, _unit(end - root), float(np.hypot(*(end - root))), prng, width, ROUGH,
                                         fork, max(2.0, 0.008 * s), PAIR_SECOND, room=room, near=0.15 * s, side=away,
                                         lead=_rot(aims[t], away * min(PAIR_PAST, max(past, more))),
                                         clear=PAIR_CLEAR * s, tries=2, lead_off=across * away * PAIR_LEAD * s, scale=k)
                        if tree and first is not None and (along is None or paths[0].along is None) and \
                                float(_unit(end - root) @ first) > math.cos(math.radians(PAIR_APART)):
                            tree = []  # (one runs off into the air, the other its way: two cables side by side)
                        if tree:
                            break
                if not tree:  # (the second finger's channel could not keep off the first's)
                    paths.pop(fi, None)
                    continue
                splash = []
                if along is not None:  # (it strikes a surface: forks splash out there, each kept off the others)
                    for q, wd, it in _splash(tree[0][0], along, prng, width, k):
                        if room.free(q, wd, 2.0 * (width + wd + 2.0 * room.pad), beside=False):
                            splash.append((q, wd, it))
                            room.add(q, wd)
                elif _exit(end, _unit(end - root), hand.size) > 0:  # (it ends in the air: a brush)
                    tree += _brush(tree[0][0], prng, width, k, room)
                life = max(PAIR_LIFE if peak else int(rng.choice(LIFE, p=LIFE_P)), hold)
                p = paths[fi] = _Path(t, life, root.copy(), tree[0][0][-1].copy(), tree, splash=splash, along=along,
                                      prong=two)
                if along is not None:
                    hit_at[fi] = p.b
                drawn, end, flash = tree, p.b, 1.0  # a new path: its first, brightest stroke
            else:
                p = paths[fi]
                end = p.b
                if timeline.camera is not None and t > p.born:  # the far end struck the room: it moves with it
                    end = (camera_between(timeline.camera, p.born, t) @ np.r_[p.b, 1.0])[:2]
                drawn = _carry(p.tree, p.a, p.b, root, end)
                if fi and not all(room.free(q, wd + (0.0 if i else PAIR_CLEAR * s), 0.0 if i else 0.15 * s,
                                            skip=ROOM_SKIP if i else 0.0) for i, (q, wd, _) in enumerate(drawn)):
                    paths.pop(fi)  # (the second finger's, carried with the hand, came to cross the first's)
                    continue
                for q, wd, _ in drawn:
                    room.add(q, wd)
                on = peak or rng.random() < RESTROKE[0] + RESTROKE[1] * lv
                flash = float(rng.uniform(0.55, 0.9)) if on else float(rng.uniform(0.0, 0.1))
                if t - p.born < hold:  # (over a bright wall, a stroke burns on at full strength)
                    flash = 1.0
            if t - p.born < SPLASH[-1] and p.splash:  # (it splashes where it strikes on its first frames)
                drawn = drawn + _carry(p.splash, p.a, p.b, root, end)
            flashes[fi] = flash
            gain = flash * (0.55 + 0.45 * lv) * pres
            if gain > 0:  # (a channel gone dark still glows faintly)
                out += [(q, wd, it * gain) for q, wd, it in drawn]
                ends = [] if p.along is None else [(end, p.along, gain)]
                if p.prong is not None:  # (its second prong's end, carried as its own is)
                    ends.append((_carry([(p.prong[0][None], 0.0, 0.0)], p.a, p.b, root, end)[0][0][0], p.prong[1],
                                 gain * PRONG[3]))
                for e_, a_, g_ in ends:
                    xi, yi = int(round(e_[0])), int(round(e_[1]))
                    if g_ >= IMPACT[2] and 0 <= xi < hand.size[0] and 0 <= yi < hand.size[1] and \
                            (matte is None or matte[t, yi, xi] < 128):  # (the hand come in front of it hides it)
                        at.append((float(e_[0]), float(e_[1]), float(a_[0]), float(a_[1]), g_, t == p.born))
            lit = max(lit, gain)
        # the crackle at the fingertips: small arcs out of the tip into the room, and between two tips while both
        # strike, struck every frame, while the finger is steady enough not to smear them into broken hooks (smeared
        # with the finger when drawn: layer); a second finger's tip burns only while its channel strikes (left burning
        # when it could not keep clear, it was a white blob with no channel), and with the voice stopped a tip burns
        # only while the finger is steady (on a hand swung fast the tracker lags it: a white dot hung in the air)
        c = float(crackle[t])
        crng = np.random.default_rng([seed & 0xFFFFFFFF, _CRACKLE, t])
        sp = []
        for fi, root in enumerate(roots):
            if fi and flashes[fi] < 0.3:
                continue
            if speed[t] <= crawl_still:
                for _ in range(1 + int(round(c))):
                    tip = root + _rot(aims[t], crng.uniform(-CRACKLE_FAN, CRACKLE_FAN)) * s * crng.uniform(0.08, 0.28)
                    out += _carry(_between(crng, CRACKLE_WIDTH * kw, CRACKLE_INTEN * (0.4 + 0.6 * c) *
                                           crng.uniform(0.6, 1.0) * pres, 0.35, 0.04),
                                  np.zeros(2), np.array([1.0, 0.0]), root, tip)
                if fi == 1 and min(flashes) >= 0.5 and crng.random() < JUMP_P:  # two fingers: an arc jumps between
                    out += _carry(_between(crng, kw, 0.55 * (0.4 + lit) * pres, 0.4, 0.03), np.zeros(2),  # their tips
                                  np.array([1.0, 0.0]), roots[0], root)
            if lv > 0 or speed[t] <= crawl_still:
                sp.append((root[0], root[1], CORONA * s, max(CORONA_MIN * c, 0.15 * c + 0.4 * lit * (0.5 + lv)) * pres))
        bolts.append(out)
        spots.append(sp)
        struck.append(at)
    return bolts, spots, struck


def knotted(pts: np.ndarray) -> bool:
    """Whether a channel (2D or 3D points: as seen, their first two coordinates) is knotted: cut into 12 pieces of
    equal length as seen, it turns more than KNOT_TURN degrees one way over any 5 of them, or its ends are less than
    KNOT_STRAIGHT of the 12 pieces' length apart (its finest kinks are not counted)."""
    p2 = pts[:, :2]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p2, axis=0), axis=1))])
    if cum[-1] < 1e-9:
        return True  # (seen end on: a point)
    at = np.linspace(0.0, cum[-1], 13)
    q = np.stack([np.interp(at, cum, p2[:, 0]), np.interp(at, cum, p2[:, 1])], 1)
    piece = np.diff(q, axis=0)
    if np.linalg.norm(q[-1] - q[0]) < KNOT_STRAIGHT * np.linalg.norm(piece, axis=1).sum():
        return True
    a = np.arctan2(piece[:, 1], piece[:, 0])
    turn = np.degrees((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)
    return bool(np.abs(np.convolve(turn, np.ones(5), "valid")).max() > KNOT_TURN)


def _onto(pts: np.ndarray, r: float, near: float) -> np.ndarray:
    """3D points held within ``near`` of the sphere of radius ``r`` (each moved along its radius)."""
    n = np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-9)
    return pts * (np.clip(n, r - near, r + near) / n)


def _shell(rng: np.random.Generator) -> list:
    """An arc crawling over the ball's skin (3D, the unit ball, as seen: z towards the camera): a jagged stretch of a
    great circle SHELL_SPAN degrees long, SHELL_OUTLINE of them lying across the picture along its outline (which is
    what draws its edge) and SHELL_INNER's share on a sphere inside it, kept within SHELL_SKIN of its sphere, forking
    once now and then (SHELL_FORK); drawn again (a few times at most) while it is knotted as seen."""
    outline = rng.random() < SHELL_OUTLINE
    inner = rng.random() < SHELL_INNER[0]
    r = rng.uniform(*SHELL_INNER[1]) if inner else 1.0
    for _ in range(6):
        nrm = _unit(np.array([0.0, 0.0, 1.0]) + rng.normal(0, 0.25, 3)) if outline else _unit(rng.normal(0, 1, 3))
        e1 = _unit(np.cross(nrm, rng.normal(0, 1, 3)))
        e2 = np.cross(nrm, e1)
        phi = rng.uniform(0, 2 * np.pi) + np.radians(rng.uniform(*SHELL_SPAN)) * np.linspace(0, 1, 9)[:, None]
        arc = (np.cos(phi) * e1 + np.sin(phi) * e2) * r
        pts = _onto(np.concatenate([displace(arc[j], arc[j + 1], rng, SHELL_ROUGH, 0.02)[:-1] for j in range(8)]
                                   + [arc[-1:]]), r, SHELL_SKIN)
        if not knotted(pts):
            break
    tree = [(pts, 1.0, SHELL_INNER_INTEN if inner else 1.0)]
    if rng.random() < SHELL_FORK:  # (a fork off it, along the skin)
        a = pts[int(rng.integers(len(pts) // 4, 3 * len(pts) // 4))]
        b = _unit(a + rng.normal(0, 0.45, 3)) * r
        tree.append((_onto(displace(a, b, rng, SHELL_ROUGH, 0.02), r, SHELL_SKIN), 0.6, 0.5 * tree[0][2]))
    return tree


def _palm_lift(hand: _Hand) -> np.ndarray:
    """(n, 2) px: where the ball's middle sits off the palm's point in the picture: BALL_LIFT hands along the palm's
    normal as it projects (the normal of the wrist and the index and little knuckles in MediaPipe's depth, the side
    facing the camera), held between LIFT_PX's two lengths (px at 1080), and up the picture as far as the normal shows
    no way across it (a palm seen square on, or no landmarks): a ball held over a palm seen from above sits further up
    the picture than the palm, and one held before a palm facing the camera still clear of it."""
    n = len(hand.xy)
    up = np.array([0.0, -1.0])
    k = min(hand.size) / 1080.0
    if hand.lm3 is None:
        return np.tile(up * LIFT_PX[0] * k, (n, 1))
    lm = hand.lm3
    nrm = _unit(np.cross(lm[:, 5] - lm[:, 0], lm[:, 17] - lm[:, 0]))
    nrm *= np.where(nrm[:, 2:] > 0, -1.0, 1.0)  # towards the camera (MediaPipe's depth grows away from it)
    across = _smooth(nrm[:, :2])
    lean = np.hypot(across[:, 0], across[:, 1])
    sure = np.clip(lean / 0.3, 0.0, 1.0)[:, None]  # (a normal hardly across the picture shows no way)
    way = _unit(sure * _unit(across) + (1.0 - sure) * up)
    return way * np.clip(BALL_LIFT * lean * hand.s, LIFT_PX[0] * k, LIFT_PX[1] * k)[:, None]


class Ball(NamedTuple):
    """A frame's ball: its middle (x, y) and radius R, px; ``level``: the voice's level now (or its charge, if
    higher), how bright it is; ``charge``: what the voice has built up, which grows and shrinks smoothly and makes its
    body dense; ``flicker``: how bright its middle burns, flaring on a hit; ``glow``: the voice's envelope, which its
    haze breathes with; (px, py): the point of the palm under it, where its light pools (None: nowhere)."""

    x: float
    y: float
    R: float
    level: float
    charge: float
    flicker: float
    glow: float = 1.0
    px: float | None = None
    py: float | None = None


def _jump(tree: list, c: np.ndarray, R: float, tip: np.ndarray, leave: float, lift: float, k: float,
          matte: np.ndarray | None) -> list | None:
    """An arc jumping from the ball (middle ``c``, radius ``R``) to a fingertip landmark ``tip``: the ``tree`` (drawn
    from (0, 0) to (1, 0)) carried from its skin, ``leave`` radians round from the way to the tip, onto a bow ``lift``
    px at 1080 (``k`` px a px) through the air, coming down onto the finger past its tip LAND_IN inside its outline
    (the ``matte``, uint8, or None: on the landmark); or None where it cannot (no outline within LAND_NEAR of the tip
    the way it comes in, or, past GROUND_FREE of the ball, more than GROUND_ON of it on the hand, or running back over
    the ball)."""
    phi = math.atan2(tip[1] - c[1], tip[0] - c[0]) + leave
    a = c + R * np.array([math.cos(phi), math.sin(phi)])
    chord = tip - a
    span = float(np.hypot(*chord))
    if span < 1e-6:
        return None
    side = np.array([chord[1], -chord[0]]) / span * lift * k  # (to the chord's left as it runs, on screen)
    land = tip
    if matte is not None:
        h, w = matte.shape
        come = _unit(tip - ((a + tip) / 2.0 + 2.0 * side))  # (its way in at its end)
        edge = None
        for e in np.arange(0.0, (LAND_NEAR + LAND_IN) * k, 1.0):  # (back along it, to where the finger ends)
            x, y = (int(round(v)) for v in tip - come * e)
            if not (0 <= x < w and 0 <= y < h) or matte[y, x] < 128:
                edge = e
                break
        if edge is None:
            return None
        land = tip - come * max(0.0, edge - LAND_IN * k)
    drawn = _bowed(tree, a, land, side)
    p = drawn[0][0]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    r = np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])
    if (r[cum > 0.15 * cum[-1]] < 0.95 * R).any():  # (back over the ball)
        return None
    if matte is not None:
        h, w = matte.shape
        xi, yi = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        on = np.zeros(len(p), bool)
        on[inside] = matte[yi[inside], xi[inside]] >= 128
        free = (r > GROUND_FREE * R) & (cum < cum[-1] - (LAND_IN + 3.0) * k)
        held = free[1:] & free[:-1]
        if float(seg[held & on[1:] & on[:-1]].sum()) > GROUND_ON * max(float(seg[held].sum()), 1e-6):
            return None
    return drawn


def ball_frames(hand: _Hand, timeline: Timeline, seed: int = 0,
                matte: np.ndarray | None = None) -> tuple[list[list], list[list], list[Ball | None]]:
    """Per frame, the ball over the palm: ([(points, width px, intensity)], [(x, y, sigma, amplitude[, glow])]
    sparks, its Ball or None). It hovers off the palm (_palm_lift), wandering a little round there (WANDER), and grows
    from a crackle to its size over about 5 frames (BALL_RISE). An irregular shell of arcs crawls over its skin and
    through it (in 3D, turning slowly, projected: those behind its middle dimmer, seen through it), and once it has a
    body arcs jump from it to the fingertips, as many as it is large (GROUND, GROUND_GONE), each bowing through the air
    off the hand's ``matte`` ((n, h, w) uint8, or None) and coming down onto the finger past its tip (_jump), white-hot
    where it lands in a blue glow (LAND): it is held, not pasted on. Each is struck and re-struck as a channel is, the
    shell's in the ball's own frame, so it rides the palm. It shows and goes on one gate (BALL_ON), fading in over
    BALL_FADE frames."""
    n = timeline.n
    L = levels(timeline)
    charge = _follow(L, BALL_GROW, BALL_SHRINK)  # it grows while the voice holds, and shrinks when it stops
    envelope = _follow(L, 1.0, 2.5)  # (the voice as heard: its haze breathes with it)
    lift = _palm_lift(hand)
    drift = WANDER[0] * min(hand.size) / 1080.0
    phase = np.random.default_rng([seed & 0xFFFFFFFF, _WANDER]).uniform(0, 2 * np.pi, 4)
    hits = dict(zip(timeline.hits, timeline.sizes))
    rng = np.random.default_rng([seed & 0xFFFFFFFF, _STRIKES])
    grng = np.random.default_rng([seed & 0xFFFFFFFF, _JUMPS])  # (the arcs to the fingers draw their own: the shell is
    # the same whatever the hand's matte makes of them)
    paths: dict[tuple[str, int], _Path] = {}
    pulse = grown = most = top = last_r = 0.0  # (how grown it has been at most, its largest radius, its last)
    shown, born = 0, False  # frames it has shown for, and whether its body has formed
    bolts, sparks, balls = [], [], []
    k = min(hand.size) / 1080.0

    def restroke(lv: float, r: np.random.Generator) -> float:
        """How bright a kept path is re-struck: a flash, or dark."""
        on = r.random() < RESTROKE[0] + RESTROKE[1] * lv
        return float(r.uniform(0.55, 0.9)) if on else float(r.uniform(0.0, 0.1))

    def strike(key: tuple[str, int], make, t: int, lv: float, hit: float | None) -> tuple[_Path, float]:
        """The path in ``key``'s slot on frame t, struck anew when its life (LIFE's) is over or on a hit, and how
        bright."""
        p = paths.get(key)
        if p is None or t - p.born >= p.life or hit is not None:
            prng = np.random.default_rng(int(rng.integers(1 << 62)))
            p = paths[key] = _Path(t, int(rng.choice(LIFE, p=LIFE_P)), np.zeros(2), np.array([1.0, 0.0]), make(prng))
            return p, 1.0
        return p, restroke(lv, rng)

    for t in range(n):
        s, lv, pres = float(hand.s[t]), max(float(L[t]), float(charge[t])), float(hand.presence[t])
        hit = hits.get(t)
        pulse = BALL_PULSE * hit if hit is not None else pulse * 0.6
        if pres <= 0 or max(float(L[max(0, t - BALL_BRIDGE + 1): t + 1].max()), lv) < BALL_ON[1 if shown else 0]:
            shown, born, grown, most, top, last_r = 0, False, 0.0, 0.0, 0.0, 0.0
            paths.clear()
            bolts.append([])
            sparks.append([])
            balls.append(None)
            continue
        shown += 1
        pres *= min(1.0, shown / BALL_FADE)
        kw = _width_k(s)
        sec = 2 * math.pi * t / fx.FPS
        (f0, f1), ph = WANDER[1], phase
        sway = (0.6 * math.sin(f0 * sec + ph[0]) + 0.4 * math.sin(f1 * sec + ph[1]),
                0.6 * math.sin(f1 * sec + ph[2]) + 0.4 * math.sin(f0 * sec + ph[3]))
        c = hand.xy[t] + lift[t] + drift * np.array(sway)
        if born:  # (its body grows from its least size on the frame it forms, and shrinks with its charge, as fast)
            grown = min(math.sqrt(float(charge[t])), grown + BALL_RISE)
        born = born or _dense(lv * pres, float(charge[t]) * pres) > 0
        R = s * (BALL_R[0] + (BALL_R[1] - BALL_R[0]) * grown * grown * (3.0 - 2.0 * grown)) * (1.0 + pulse)
        if R > last_r and R < SHELL_GONE[0] * top:  # (grown again from all but nothing: a new ball)
            most, top = 0.0, 0.0
        most, top, last_r = max(most, grown), max(top, R), R
        reach = 1.0 + FIL_REACH * (1.0 - most) ** 2  # (its shell's arcs, in radii: a young ball's crackle reaches
        # out, a dying one's goes in with it)
        dying = float(np.clip((R / top - SHELL_GONE[1]) / (SHELL_GONE[0] - SHELL_GONE[1]), 0.0, 1.0))
        if most >= 0.5:  # (and, once it has been a ball, with its body)
            dying *= _dense(lv * pres, float(charge[t]) * pres)
        ang, tilt = BALL_TURN * t, 0.35
        turn = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]]) @ \
            np.array([[math.cos(ang), 0, math.sin(ang)], [0, 1, 0], [-math.sin(ang), 0, math.cos(ang)]])
        out, sp = [], []

        def arc(prng: np.random.Generator) -> list:
            """A shell arc struck as seen now, kept in the ball's own frame (it turns with it)."""
            return [(q @ turn, w_, it) for q, w_, it in _shell(prng)]

        wd = (0.6 + 0.5 * lv) * kw
        for i in range(int(round(SHELL[0] + (SHELL[1] - SHELL[0]) * charge[t]))):
            p, flash = strike(("shell", i), arc, t, lv, hit)
            gain = flash * (0.4 + 0.6 * lv) * pres * dying
            if gain <= 0.02:
                continue
            for q, w_, it in p.tree:
                q3 = q @ turn.T
                q2 = c + R * reach * q3[:, :2]
                front = q3[:, 2] >= -0.1  # (its stretches behind its middle are seen through it: dimmer)
                limb = 1.0 - np.abs(q3[:, 2]) / np.maximum(np.linalg.norm(q3, axis=1), 1e-9)  # (1 on its outline)
                for run in np.split(np.arange(len(q2)), np.flatnonzero(np.diff(front.astype(np.int8))) + 1):
                    idx = np.arange(run[0], min(run[-1] + 2, len(q2)))  # (each run on to where the next begins)
                    if len(idx) >= 2:
                        k_ = (SHELL_FACE + (1.0 - SHELL_FACE) * float(limb[idx].mean())) * \
                            (1.0 if front[run[0]] else SHELL_BACK)
                        out.append((q2[idx], w_ * wd, it * gain * k_))
        # the arcs to the fingertips, once it has a body: as many as it is large, none once it has shrunk under
        # GROUND_GONE of the largest it grew to; each kept while it lives and still bows clear of the hand, else
        # struck anew to the fingertip with the fewest (tried round the next ones), or not struck
        size = min(1.0, R / (s * BALL_R[1])) if R >= GROUND_GONE * top else 0.0
        count = int(round((GROUND[0] + (GROUND[1] - GROUND[0]) * lv) * size)) \
            if hand.lm is not None and _dense(lv * pres, float(charge[t]) * pres) > 0 else 0
        m = None if matte is None else matte[t]
        free = [] if hand.lm is None else [f for f in TIPS if np.linalg.norm(hand.lm[t, f] - c) >= 1.15 * R]
        used: dict[int, int] = {}
        for i in range(count):
            key = ("ground", i)
            p = paths.get(key)
            drawn = None
            if p is not None and t - p.born < p.life and hit is None and p.to in free:
                drawn = _jump(p.tree, c, R, hand.lm[t, p.to], p.leave, p.lift, k, m)
                flash = restroke(lv, grng) if drawn is not None else 1.0
            if drawn is None:
                paths.pop(key, None)
                if not free:
                    continue
                prng = np.random.default_rng(int(grng.integers(1 << 62)))
                life = int(grng.choice(LIFE, p=LIFE_P))
                order = sorted(free, key=lambda f: (used.get(f, 0), float(prng.random())))
                for j in range(GROUND_TRIES):
                    to = order[(j // 4) % len(order)]
                    tree = _between(prng, *GROUND_ARC, 0.2, 0.01, 0.0)
                    leave = math.radians(float(prng.uniform(-GROUND_LEAVE, GROUND_LEAVE)))
                    bow = float(prng.choice([-1.0, 1.0])) * float(prng.uniform(*GROUND_BOW[0 if j % 4 < 2 else 1]))
                    drawn = _jump(tree, c, R, hand.lm[t, to], leave, bow, k, m)
                    if drawn is not None:
                        p = paths[key] = _Path(t, life, np.zeros(2), np.array([1.0, 0.0]), tree, to, lift=bow,
                                               leave=leave)
                        flash = 1.0
                        break
                if drawn is None:
                    continue
            used[p.to] = used.get(p.to, 0) + 1
            if flash <= 0.05:
                continue
            out += [(q, w_ * kw * (0.7 + 0.5 * lv), it * flash * pres) for q, w_, it in drawn]
            end = drawn[0][0][-1]
            sp.append((float(end[0]), float(end[1]), LAND[0] * k, LAND[1] * flash * pres, LAND[3] * flash * pres))
        bolts.append(out)
        sparks.append(sp)
        flicker = float(np.random.default_rng([seed & 0xFFFFFFFF, _SPARK, t]).uniform(0.7, 1.0))
        flicker *= 1.0 + BALL_FLARE * (hit or 0.0)
        balls.append(Ball(float(c[0]), float(c[1]), float(R), lv * pres, float(charge[t]) * pres, flicker,
                          float(envelope[t]) * pres, float(hand.xy[t, 0]), float(hand.xy[t, 1])))
    return bolts, sparks, balls


# ---- drawing -------------------------------------------------------------------------------------------------------

def _box(bolts: list, spots: list, s: float, h: int, w: int) -> tuple[int, int, int, int] | None:
    """(y0, y1, x0, x1): where the channels and spots are, with room for their widest glow."""
    xs, ys = [], []
    for p, _, it in bolts:
        if it > 0 and len(p) >= 2:
            xs += [p[:, 0].min(), p[:, 0].max()]
            ys += [p[:, 1].min(), p[:, 1].max()]
    for x, y, r, a, *_ in spots:
        if a > 0:
            xs += [x - 3 * r, x + 3 * r]
            ys += [y - 3 * r, y + 3 * r]
    if not xs:
        return None
    pad = 3 * GLOWS[-1][0] * s + 8
    y0, y1 = max(0, int(min(ys) - pad)), min(h, int(max(ys) + pad) + 1)
    x0, x1 = max(0, int(min(xs) - pad)), min(w, int(max(xs) + pad) + 1)
    return None if y1 <= y0 or x1 <= x0 else (y0, y1, x0, x1)


def draw(bolts: list, box: tuple[int, int, int, int], seed: int = 0, frame: int = 0,
         floor: float = 0.0) -> np.ndarray:
    """The channels' core in ``box`` ((y0, y1, x0, x1) of the frame), float32 linear: each polyline drawn anti-aliased
    at sub-pixel precision, in brighter and dimmer stretches, a main channel thinning towards its end (MAIN_TAPER), a
    branch (BRANCH) fading (TAPER) and thinning (BRANCH_W) along it; where they cross, the brighter shows. (bh, bw,
    2): the core, and what the glow round it is drawn from, the same lines with a hot channel (GLOW_HOT) as if at
    least ``floor`` px wide (GLOW_FLOOR)."""
    y0, y1, x0, x1 = box
    core = np.zeros((y1 - y0, x1 - x0, 2), np.float32)
    tmp = np.zeros(core.shape[:2], np.uint8)
    rng = np.random.default_rng([seed & 0xFFFFFFFF, _FRAME, frame])
    top = max((it for _, _, it in bolts), default=0.0)
    for pts, width, inten in bolts:
        if len(pts) < 2 or inten <= 0:
            continue
        branch = float(np.clip((BRANCH[1] - inten / top) / (BRANCH[1] - BRANCH[0]), 0.0, 1.0))
        fade = TAPER[0] * branch
        p = pts[:, :2] - (x0, y0)
        a0 = np.floor(p.min(0)).astype(int) - int(BRANCH_W[0] * width) - 3
        a1 = np.ceil(p.max(0)).astype(int) + int(BRANCH_W[0] * width) + 3
        bx0, by0 = max(0, a0[0]), max(0, a0[1])
        bx1, by1 = min(core.shape[1], a1[0]), min(core.shape[0], a1[1])
        if bx1 <= bx0 or by1 <= by0:
            continue
        sub = tmp[by0:by1, bx0:bx1]
        sub[:] = 0
        chunks = 10
        bead = rng.uniform(0.6, 1.0, chunks)
        for c in range(chunks):
            a, b = c * (len(p) - 1) // chunks, (c + 1) * (len(p) - 1) // chunks + 1
            if b - a < 2:
                continue
            u = c / (chunks - 1)
            wd = width * (_taper((c + 0.5) / chunks) * (1.0 - branch)
                          + (BRANCH_W[0] + (BRANCH_W[1] - BRANCH_W[0]) * u) * branch)
            th = max(1, int(round(wd)))
            lv = int(np.clip(255 * bead[c] * min(1.0, wd / th) * (1.0 - fade * ((c + 0.5) / chunks) ** TAPER[1]), 30,
                             255))
            q = np.round((p[a:b] - (bx0, by0)) * 4).astype(np.int32)  # (2 fractional bits: sub-pixel)
            cv2.polylines(sub, [q.reshape(-1, 1, 2)], False, lv, th, cv2.LINE_AA, shift=2)
        hot = np.clip((inten - GLOW_HOT[0]) / (GLOW_HOT[1] - GLOW_HOT[0]), 0.0, 1.0)
        boost = 1.0 + (max(1.0, floor / max(width, 1e-6)) - 1.0) * hot
        line = sub.astype(np.float32) * np.float32(inten / 255.0)
        reg = core[by0:by1, bx0:bx1]
        np.maximum(reg[..., 0], line, out=reg[..., 0])
        np.maximum(reg[..., 1], line * np.float32(boost), out=reg[..., 1])
    return core


def _spot(out: np.ndarray, box: tuple, x: float, y: float, sigma: float, amp: float) -> None:
    """Adds a Gaussian spot to ``out`` (a layer over ``box``)."""
    y0, _, x0, _ = box
    r = int(4 * sigma) + 2
    cx, cy = x - x0, y - y0
    a0, a1 = max(0, int(cx) - r), min(out.shape[1], int(cx) + r)
    b0, b1 = max(0, int(cy) - r), min(out.shape[0], int(cy) + r)
    if a1 <= a0 or b1 <= b0 or amp <= 0:
        return
    xx = (np.arange(a0, a1, dtype=np.float32) - cx) ** 2
    yy = (np.arange(b0, b1, dtype=np.float32)[:, None] - cy) ** 2
    out[b0:b1, a0:a1] += np.float32(amp) * np.exp(-(xx + yy) / np.float32(2 * sigma * sigma))


def _streak(out: np.ndarray, box: tuple, x: float, y: float, ax: float, ay: float, along: float, across: float,
            amp: float) -> None:
    """Adds to ``out`` (a layer over ``box``) a Gaussian spot ``along`` px (sigma) the way (ax, ay) and ``across`` px
    the other way."""
    y0, _, x0, _ = box
    r = int(4 * max(along, across)) + 2
    cx, cy = x - x0, y - y0
    a0, a1 = max(0, int(cx) - r), min(out.shape[1], int(cx) + r)
    b0, b1 = max(0, int(cy) - r), min(out.shape[0], int(cy) + r)
    if a1 <= a0 or b1 <= b0 or amp <= 0:
        return
    dx = np.arange(a0, a1, dtype=np.float32) - np.float32(cx)
    dy = np.arange(b0, b1, dtype=np.float32)[:, None] - np.float32(cy)
    u, v = dx * np.float32(ax) + dy * np.float32(ay), dy * np.float32(ax) - dx * np.float32(ay)
    out[b0:b1, a0:a1] += np.float32(amp) * np.exp(-0.5 * ((u / np.float32(along)) ** 2 + (v / np.float32(across)) ** 2))


def pool_light(x: float, y: float, ax: float, ay: float, gain: float, side: int, h: int, w: int) \
        -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """The pool of a strike's light on the surface it strikes at (x, y), running (ax, ay) (POOL): (box, (bh, bw) light
    over it, to multiply its colour by), 1 / r from its core along the surface and ``squeeze`` times faster across
    it, out to about its length, times ``gain``; or None (off the frame). ``side``: the picture's short side."""
    k = side / 1080.0
    length, core, squeeze = POOL[0] * k, POOL[1] * k, POOL[2]
    box = _around(x, y, 1.6 * length, h, w)
    if box is None or gain <= 0:
        return None
    y0, y1, x0, x1 = box
    dx = np.arange(x0, x1, dtype=np.float32) - np.float32(x)
    dy = np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(y)
    u, v = dx * np.float32(ax) + dy * np.float32(ay), (dy * np.float32(ax) - dx * np.float32(ay)) * np.float32(squeeze)
    r = np.sqrt(u * u + v * v)
    return box, np.float32(gain * core) / (np.float32(core) + r) * np.exp(-(r / np.float32(length)) ** 2)


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur, done small when it is wide (it is smooth)."""
    h, w = x.shape[:2]
    q = int(np.clip(sigma / 3, 1, 16))
    if q == 1 or min(h, w) < 4 * q:
        return cv2.GaussianBlur(x, (0, 0), max(sigma, 0.3))
    small = cv2.resize(x, (max(1, w // q), max(1, h // q)), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / q), (w, h), interpolation=cv2.INTER_LINEAR)


def softness(frame: np.ndarray, xy, s: float) -> float:
    """px: how much softer than a crisp drawn line the plate is round the hand on this frame (its focus, its motion,
    the phone's own smoothing). Its strongest edges blurred SOFT_R px more keep sqrt(sigma^2 + SOFT_R^2) / sigma of
    their gradient, for an edge blurred by sigma; a crisp anti-aliased edge reads SOFT_AA, taken off."""
    h, w = frame.shape[:2]
    r, x, y = int(SOFT_AREA * s) + 8, int(xy[0]), int(xy[1])
    sub = frame[max(0, y - r): min(h, y + r), max(0, x - r): min(w, x + r)]
    if min(sub.shape[:2]) < 16:
        return 0.0
    g = cv2.cvtColor(np.ascontiguousarray(sub), cv2.COLOR_RGB2GRAY).astype(np.float32)

    def grad(im: np.ndarray) -> np.ndarray:
        return np.hypot(cv2.Sobel(im, cv2.CV_32F, 1, 0), cv2.Sobel(im, cv2.CV_32F, 0, 1))

    m = grad(g)
    edge = m >= np.percentile(m, 98)
    ratio = float(np.median(m[edge] / np.maximum(grad(cv2.GaussianBlur(g, (0, 0), SOFT_R))[edge], 1e-6)))
    sigma = SOFT_R / math.sqrt(max(ratio * ratio - 1.0, 1e-6))
    return math.sqrt(max(sigma * sigma - SOFT_AA * SOFT_AA, 0.0))


def _smear(core: np.ndarray, box: tuple, move: np.ndarray, around: list, s: float, most: float) -> np.ndarray:
    """``core`` (over ``box``, (bh, bw, c)) blurred along ``move`` (px: how far the hand went while the shutter was
    open; at most ``most`` px, SMEAR_MAX) within SMEAR_NEAR of the points ``around`` ([(x, y)] in the frame), in
    place; within WELD of them it keeps WELD_KEEP of its sharp self, so a channel stays joined to its fingertip."""
    length = float(np.hypot(*move))
    if length > most:
        move, length = np.asarray(move, np.float64) * (most / length), most
    if length < 1.0 or not around:
        return core
    k = 2 * int(math.ceil(length / 2)) + 1
    ker = np.zeros((k, k), np.uint8)
    c, d = (k - 1) / 2, np.asarray(move, np.float64) / 2
    cv2.line(ker, (round((c - d[0]) * 16), round((c - d[1]) * 16)), (round((c + d[0]) * 16), round((c + d[1]) * 16)),
             255, 1, cv2.LINE_AA, shift=4)
    ker = ker.astype(np.float32) / float(ker.sum())
    near, far = SMEAR_NEAR[0] * s, SMEAR_NEAR[1] * s
    pts = np.asarray(around, np.float64) - (box[2], box[0])
    a0, b0 = np.maximum(0, np.floor(pts.min(0) - far - k)).astype(int)
    a1, b1 = np.minimum((core.shape[1], core.shape[0]), np.ceil(pts.max(0) + far + k)).astype(int)
    if a1 <= a0 or b1 <= b0:
        return core
    sub = core[b0:b1, a0:a1]
    sharp = sub.copy()
    xx, yy = np.arange(a0, a1, dtype=np.float32), np.arange(b0, b1, dtype=np.float32)[:, None]
    gap = np.min([np.hypot(xx - px, yy - py) for px, py in pts], axis=0)[..., None]
    sub += (cv2.filter2D(sub, -1, ker, borderType=cv2.BORDER_CONSTANT).reshape(sub.shape) - sub) * \
        np.clip((far - gap) / (far - near), 0, 1)
    weld = np.clip((WELD[1] * s - gap) / ((WELD[1] - WELD[0]) * s), 0, 1) * np.float32(WELD_KEEP)
    np.maximum(sub, sharp * weld, out=sub)
    return core


_PLASMA_MAPS: dict[int, np.ndarray] = {}


def _plasma_map(seed: int) -> np.ndarray:
    """(octaves, *PLASMA_MAP) float32: for ``seed``, each of PLASMA's octaves of ridged noise (white noise low-passed
    to that many cycles round the sphere, its zero crossings bright: filaments) over latitude and longitude, wrapping
    round it, scaled to mean 0 and spread 1. Made once a seed."""
    if seed not in _PLASMA_MAPS:
        rng = np.random.default_rng([seed & 0xFFFFFFFF, _PLASMA])
        th, tw = PLASMA_MAP
        ky, kx = np.fft.fftfreq(th)[:, None] * th, np.fft.rfftfreq(tw)[None, :] * tw / 2.0  # (cycles round it)
        maps = []
        for cycles in PLASMA[2]:
            n = np.fft.irfft2(np.fft.rfft2(rng.normal(0.0, 1.0, (th, tw))) * np.exp(-(np.hypot(kx, ky) / cycles) ** 2),
                              (th, tw))
            ridge = (1.0 - np.abs(np.tanh(n / n.std()))) ** 2
            maps.append((ridge - ridge.mean()) / ridge.std())
        _PLASMA_MAPS[seed] = np.stack(maps).astype(np.float32)
    return _PLASMA_MAPS[seed]


def _turned(axis, angle: float) -> np.ndarray:
    """The 3x3 rotation by ``angle`` radians about ``axis`` (Rodrigues)."""
    a = _unit(np.asarray(axis, np.float64))
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


_PLASMA_AXES = ((0.2, 1.0, 0.3), (-0.5, 1.0, 0.2), (0.3, 1.0, -0.6))  # each octave turns about its own axis


def plasma(dx: np.ndarray, dy: np.ndarray, seed: int, frame: int) -> np.ndarray:
    """The ball's plasma on frame ``frame``, over (dx, dy) (its radii, from its middle): PLASMA's octaves on the
    sphere, each turned about its own axis at its own rate, front and back (seen through it, 0.6 as much), summed;
    scaled to [-1, 1] at twice its spread."""
    maps = _plasma_map(seed)
    th, tw = PLASMA_MAP
    r2 = dx * dx + dy * dy
    k = np.where(r2 < 1.0, 1.0, 1.0 / np.sqrt(np.maximum(r2, 1e-9))).astype(np.float32)  # (past its skin: on it)
    z = np.sqrt(np.maximum(1.0 - r2, 0.0)).astype(np.float32)
    total = np.zeros(r2.shape, np.float32)
    sec = frame / fx.FPS
    for o, (weight, rate) in enumerate(zip(PLASMA[3], PLASMA[4])):
        rot = _turned(_PLASMA_AXES[o], 2.0 * math.pi * PLASMA[1] * rate * sec).astype(np.float32)
        for side, share in ((1.0, 1.0), (-1.0, 0.6)):
            px, py, pz = (rot[i, 0] * dx * k + rot[i, 1] * dy * k + rot[i, 2] * side * z for i in range(3))
            mx = (np.arctan2(px, pz) / np.float32(2 * math.pi) + 0.5) * tw
            my = (np.arcsin(np.clip(py, -1.0, 1.0)) / np.float32(math.pi) + 0.5) * th
            total += np.float32(weight * share) * cv2.remap(maps[o], mx.astype(np.float32), my.astype(np.float32),
                                                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    spread = math.sqrt(sum(w * w for w in PLASMA[3]) * (1.0 + 0.36))
    return np.clip(total / np.float32(2.0 * spread), -1.0, 1.0)


def _sphere(out: np.ndarray, veil: np.ndarray, box: tuple, ball: Ball, seed: int = 0, frame: int = 0) -> None:
    """Adds the ball's body (over ``box``) to ``out``, in place: a volume of light, white-hot in PLASMA_RGB in its
    middle (BALL_HOT), a translucent cyan-blue glow out to where its density runs out (BODY), moving plasma in both
    (PLASMA, on frame ``frame`` for ``seed``), and the cyan-blue haze of its light round it (HALO), breathing with
    the voice (HALO_VOICE), with no outline of its own (its shell's arcs draw one); and to ``veil`` how much of what
    is behind it it hides, only where it is dense (BALL_TAU). A young or fading ball (_dense) has as much less of a
    body; a hit's flare brightens its middle by at most INNER_FLARE."""
    dense = _dense(ball.level, ball.charge)
    if dense <= 0:
        return
    x, y, R = ball.x, ball.y, ball.R
    y0, _, x0, _ = box
    r = int(R * max(BODY[1], 3.5 * HALO[1])) + 2
    a0, a1 = max(0, int(x - x0) - r), min(veil.shape[1], int(x - x0) + r)
    b0, b1 = max(0, int(y - y0) - r), min(veil.shape[0], int(y - y0) + r)
    if a1 <= a0 or b1 <= b0:
        return
    dx = np.arange(a0, a1, dtype=np.float32) - np.float32(x - x0)
    dy = np.arange(b0, b1, dtype=np.float32)[:, None] - np.float32(y - y0)
    d2 = (dx * dx + dy * dy) / np.float32(R * R)
    swirl = plasma(dx / np.float32(R), dy / np.float32(R), seed, frame) * \
        np.clip((1.02 - np.sqrt(d2)) / 0.12, 0.0, 1.0).astype(np.float32)  # (faded out over its limb)
    hot = np.float32(BALL_HOT[0] * min(max(ball.flicker, 1.0), 1.0 + INNER_FLARE)) * \
        np.exp(d2 * np.float32(-0.5 / BALL_HOT[1] ** 2)) * (1.0 + np.float32(PLASMA[5]) * swirl)
    body = np.exp(-(d2 / np.float32(BODY[1] ** 2)) ** 2) * np.float32(BODY[0]) * (1.0 + np.float32(PLASMA[0]) * swirl)
    halo = np.float32(HALO[0] * (HALO_VOICE[0] + HALO_VOICE[1] * ball.glow)) * \
        np.exp(d2 * np.float32(-0.5 / HALO[1] ** 2))
    out[b0:b1, a0:a1] += np.float32(dense) * (hot[..., None] * PLASMA_RGB + body[..., None] * BODY_RGB +
                                              halo[..., None] * HALO_RGB)
    tau = np.float32(BALL_TAU[0]) * np.exp(d2 * np.float32(-0.5 / BALL_TAU[1] ** 2)) * \
        (1.0 + np.float32(PLASMA[0]) * swirl)
    veil[b0:b1, a0:a1] = np.float32(dense) * (1.0 - np.exp(-tau))


def _dense(lv: float, full: float) -> float:
    """How much of a body a ball has, in [0, 1], for the voice's level now and its charge: none until it passes half
    of BALL_FORMED, all of it from BALL_FORMED on."""
    return float(np.clip(2.0 * (0.3 * lv + 0.7 * full) / BALL_FORMED - 1.0, 0.0, 1.0))


def layer(bolts: list, spots: list, ball: Ball | None, s: float, box: tuple, cap: float, soft: float,
          move: np.ndarray, around: list, seed: int = 0, frame: int = 0, side: int = 1080,
          floor: float = 0.0, hide: np.ndarray | None = None, impacts: list = (),
          bright: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Over ``box``: linear RGB (bh, bw, 3), the light added to the plate: the channels' white-hot core and the
    blue-violet glow round them (a ball's arcs white-hot as an arc's, over skin and over its body alike, in
    BALL_GLOW_RGB's glow), the white-hot ``spots`` ([(x, y, sigma, amplitude[, a blue glow's amplitude])]: white, a
    blue halo), at each of the ``impacts`` ([(x, y, ax, ay, gain, ...)]: where a channel strikes a surface running (ax,
    ay)) its white-hot point (IMPACT) and the flash in the air
    along the surface (POOL_GLOW), and the ball's middle and body (_sphere); and (bh, bw) how much of the plate it
    hides: what is behind the ball's dense middle, or round the arcs the share of a bright plate darkened (CONTRAST).
    The channels are smeared along ``move`` round ``around`` (_smear), hidden as much as ``hide`` (bh, bw: what of
    them passes behind the hand, HIDE_NEAR) says, and blurred ``soft`` px (the plate's softness); only the brightest
    clip (CORE_GAMMA), and their glow, from the channels as drawn (a hot one as if ``floor`` px wide: draw), is held
    under ``cap`` (GLOW_CAP). Over a ``bright`` plate (0-1, _bright) the band is darker and wider (BRIGHT_BAND) and a
    saturated blue glow rings them (BRIGHT_GLOW). ``side``: the picture's short side, which the px constants are given
    at 1080 of."""
    k = side / 1080.0
    lines = _smear(draw(bolts, box, seed, frame, floor), box, move, around, s, SMEAR_MAX * k)
    if hide is not None:
        lines *= (1.0 - hide)[..., None]
    if soft > 0:
        lines = cv2.GaussianBlur(lines, (0, 0), soft)
    core, hot = np.maximum(lines[..., 0], 0), np.maximum(lines[..., 1], 0)  # (a wide smear is filtered by FFT, a hair
    # under 0 here and there)
    out = (core ** np.float32(CORE_GAMMA))[..., None] * (CORE_RGB * np.float32(CORE_GAIN))
    glow = np.zeros_like(out)
    for sig_h, g, rgb in GLOWS:  # (a blur spreads a line's energy: scaled so its peak over a 1 px line is g)
        sig = sig_h * s
        rgb = rgb if ball is None else BALL_GLOW_RGB
        glow += _blur(hot, sig)[..., None] * (np.float32(g * sig * math.sqrt(2 * math.pi)) * np.float32(rgb))
    if bright > 0 and ball is None:
        sig, g, rgb = BRIGHT_GLOW[0] * k, BRIGHT_GLOW[1] * bright, BRIGHT_GLOW[2]
        glow += _blur(hot, sig)[..., None] * (np.float32(g * sig * math.sqrt(2 * math.pi)) * np.float32(rgb))
    spark, haze = np.zeros(core.shape, np.float32), np.zeros(core.shape, np.float32)
    for x, y, r, a, *g in spots:  # (where a ball's arc lands, a blue glow round its point too: LAND)
        _spot(spark, box, x, y, r, a)
        if g:
            _spot(haze, box, x, y, LAND[2] * k, g[0])
    glow += haze[..., None] * HAZE_RGB
    flash = np.zeros(core.shape, np.float32)
    for x, y, ax, ay, g, *_ in impacts:
        _spot(spark, box, x, y, IMPACT[0] * k, IMPACT[1] * g)
        _streak(flash, box, x, y, ax, ay, POOL_GLOW[0] * k, POOL_GLOW[1] * k, POOL_GLOW[2] * g)
    glow += flash[..., None] * np.float32(GLOWS[0][2])
    veil = np.zeros(core.shape, np.float32)
    if ball is not None:
        _spot(spark, box, ball.x, ball.y, BALL_CORE * ball.R, BALL_SPARK * ball.level * ball.flicker)
        _sphere(out, veil, box, ball, seed, frame)
    elif core.any():  # (the plate round the channels, a share darker where it is bright: render)
        veil = np.float32(CONTRAST[0] + BRIGHT_BAND[0] * bright) * \
            np.minimum(1.0, 4.0 * _blur((core > 0.2).astype(np.float32), (CONTRAST[1] + BRIGHT_BAND[1] * bright) * k))
    if spark.any():
        out += spark[..., None] * SPARK_RGB
        glow += (0.6 * _blur(spark, 0.12 * s))[..., None] * HAZE_RGB
    y = fx._lum(glow)
    glow *= (cap * np.tanh(y / cap) / np.maximum(y, 1e-6))[..., None]
    return out + glow, veil


# ---- the light -----------------------------------------------------------------------------------------------------

def light_power(power: np.ndarray, on: np.ndarray | None = None,
                flicker: float = LIGHT_FLICKER) -> tuple[np.ndarray, np.ndarray]:
    """(n,) the strength of the light the channels throw on the hand and the room, from their own power (luminance
    summed): while a channel is ``on`` screen ((n,) bool: struck, re-struck or gone dark between strokes; None: while
    there is power), held at its peak over the last LIGHT_HOLD frames it has been, with ``flicker`` of the strokes'
    own flicker kept; once none is, LIGHT_TAIL of the last frame's for one frame, or what still burns (the crackle at
    a tip), if more. And (n,) its peak, the strongest it has been since it was last out. So it comes up on the first
    stroke and stays while they strike, flickering with them rather than strobing, and goes with the last."""
    on = power > 0 if on is None else on
    out, top = np.zeros_like(power), np.zeros_like(power)
    peak, run = 0.0, 0
    for t in range(len(power)):
        if on[t]:
            run += 1
            held = float(power[t - min(run, LIGHT_HOLD) + 1: t + 1].max())
            out[t] = held * (1.0 - flicker * (1.0 - power[t] / held)) if held > 0 else 0.0
        else:
            out[t] = max(float(power[t]), LIGHT_TAIL * out[t - 1] if run else 0.0)
            run = 0
        peak = max(peak, out[t]) if out[t] > 0 else 0.0
        top[t] = peak
    return out, top


def exposure_dip(bright: np.ndarray, stops: float, held: int, rise: float, fall: float, gone: int) -> np.ndarray:
    """(n,) stops the camera exposes down on each frame for lightning ``bright`` (n,) in [0, 1] (EXPOSURE): the
    brightest it has been over the last ``held`` frames, but none of what came before it was last out (under
    EXPOSURE_OUT) for ``gone`` frames in a row and not back within ``held`` frames (a stuttering voice, or the gaps in
    a phrase and between phrases, don't pump the exposure; once the light has gone the camera comes back), followed
    with time constants of ``rise`` and ``fall`` frames."""
    on = np.flatnonzero(bright > EXPOSURE_OUT)
    top = np.zeros(len(bright))
    since, dark = 0, 0  # the first frame it may still remember, and how many frames it has been out
    for t, b in enumerate(bright):
        dark = dark + 1 if b <= EXPOSURE_OUT else 0
        back = on[np.searchsorted(on, t)] - t if on.size and on[-1] > t else None  # (frames until it is on again)
        if dark >= gone and (back is None or back > held):
            since = t + 1
        top[t] = bright[max(since, t - held + 1): t + 1].max(initial=0.0)
    return stops * _follow(top, rise, fall)


def room_cast(some: np.ndarray, white: float, tint: np.ndarray, matte: np.ndarray | None) -> np.ndarray:
    """(3,) the room's own colour as the light multiplies it, at luminance 1: the mean of its surfaces' albedo
    (cuesync.fx_light.albedo) over a few linear frames ``some`` (k, h, w, 3), off the hand where ``matte`` (k, h, w)
    says, each channel held within ROOM_CAST of grey. The room's light is divided by it, so it lands on a tinted wall
    as the same cool blue as on a grey one."""
    alb = fx.albedo(some, white, tint)
    room = alb.reshape(-1, 3) if matte is None else alb[matte < 128]
    c = room.mean(0) if len(room) else np.ones(3, np.float32)
    c = c / max(float(c @ fx._LUMA), 1e-6)
    return np.clip(c, 1.0 / ROOM_CAST, ROOM_CAST).astype(np.float32)


def _soft_cap(x: np.ndarray, cap: float) -> np.ndarray:
    return np.float32(cap) * (1.0 - np.exp(-x / np.float32(cap)))


def room_lift(lit: np.ndarray, plate: np.ndarray, matte: np.ndarray | None) -> float:
    """Stops the light lifts the room away from the channels: the ROOM_FAR percentile, over the frame off the hand
    (``matte``, 0-1, or None), of how much brighter the lit frame ``lit`` is than ``plate`` (both linear RGB)."""
    q = 8
    ratio = fx._lum(lit[::q, ::q]) / np.maximum(fx._lum(plate[::q, ::q]), np.float32(1e-4))
    room = ratio if matte is None else ratio[matte[::q, ::q] < 0.5]
    return max(0.0, math.log2(max(float(np.percentile(room, ROOM_FAR)), 1e-6))) if room.size else 0.0


def hue_shoulder(P: np.ndarray) -> np.ndarray:
    """Linear RGB (..., 3) rolled off to white with each pixel's hue kept (BRIGHT_TONE): its channels scaled by the
    roll-off (tanh) of its largest past the knee, and as far as that largest is over white, that much towards white.
    A blue over-exposed a little stays blue, a lot goes cyan-white and then white, never a flat clipped pale aqua."""
    knee, over = BRIGHT_TONE
    m = P.max(-1)
    span = np.float32(1.0 - knee)
    top = np.where(m > knee, np.float32(knee) + span * np.tanh((m - knee) / span), m)
    out = P * (top / np.maximum(m, np.float32(1e-6)))[..., None]
    white = 1.0 - np.exp(-np.maximum(m - 1.0, 0.0) / np.float32(over))
    out += (top[..., None] - out) * white[..., None]
    return out


def _around(x: float, y: float, r: float, h: int, w: int) -> tuple[int, int, int, int] | None:
    """(y0, y1, x0, x1): the frame within ``r`` px of (x, y), or None off it."""
    y0, y1, x0, x1 = max(0, int(y - r)), min(h, int(y + r) + 1), max(0, int(x - r)), min(w, int(x + r) + 1)
    return None if y1 <= y0 or x1 <= x0 else (y0, y1, x0, x1)


def ball_light(ball: Ball, side: int, h: int, w: int) \
        -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    """The ball's light round it (BALL_LIGHT): (box, (bh, bw) light on the hand, (bh, bw) light on the room), to
    multiply each one's colour by: from its middle, as from a ball of its radius R, R^2 / (d^2 + h^2) at most 1 (d
    across the picture, h how far it is over the palm, or in front of the wall behind it), as bright as it is and as
    dense, gone smoothly by BALL_LIGHT's reach, and on what is seen through its body only BALL_SHADE's share of that;
    or None (no body yet, or off the frame). ``side``: the picture's short side; (h, w) its size."""
    hand_gain, room_gain, hover, behind, reach = BALL_LIGHT
    lit = (BALL_LIGHT_VOICE + (1.0 - BALL_LIGHT_VOICE) * ball.level) * _dense(ball.level, ball.charge)
    box = _around(ball.x, ball.y, reach * ball.R, h, w)
    if box is None or lit <= 0:
        return None
    y0, y1, x0, x1 = box
    R2 = np.float32(ball.R * ball.R)
    d2 = (np.arange(x0, x1, dtype=np.float32) - np.float32(ball.x))[None, :] ** 2 + \
        (np.arange(y0, y1, dtype=np.float32) - np.float32(ball.y))[:, None] ** 2
    gone = np.clip(1.0 - d2 / (R2 * np.float32(reach * reach)), 0.0, 1.0) ** 2
    gone *= 1.0 - np.float32(1.0 - BALL_SHADE) * np.exp(-(d2 / (R2 * np.float32(BALL_SHADE_R ** 2))) ** 4)
    hand = np.minimum(1.0, R2 / (d2 + np.float32((hover * side / 1080.0) ** 2))) * gone
    room = np.minimum(1.0, R2 / (d2 + R2 * np.float32(behind * behind))) * gone
    return box, hand * np.float32(hand_gain * lit), room * np.float32(room_gain * lit)


def wrap_light(matte: np.ndarray, ball: Ball, side: int) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """The ball's light on the edges of the hand that face it: (box, (bh, bw) light over it, to multiply its light's
    colour by) within WRAP_REACH radii of its middle and outside its body (WRAP_BEHIND), inside the matte (0-1 float)
    within about 2 WRAP_PX of its outline where the outline faces the ball, falling off as the inverse square of the
    distance; or None."""
    x, y, R = ball.x, ball.y, ball.R
    box = _around(x, y, WRAP_REACH * R, *matte.shape)
    if box is None or ball.charge <= 0:
        return None
    y0, y1, x0, x1 = box
    m = matte[y0:y1, x0:x1]
    if not m.any():
        return None
    e = cv2.GaussianBlur(m, (0, 0), WRAP_PX * side / 1080.0)  # (under 1 inside the hand near its outline)
    gx, gy = cv2.Sobel(e, cv2.CV_32F, 1, 0), cv2.Sobel(e, cv2.CV_32F, 0, 1)  # (pointing into the hand)
    bx = np.float32(x) - np.arange(x0, x1, dtype=np.float32)
    by = np.float32(y) - np.arange(y0, y1, dtype=np.float32)[:, None]
    dist = np.hypot(bx, by) + np.float32(1e-3)
    facing = np.clip(-(gx * bx + gy * by) / ((np.hypot(gx, gy) + np.float32(1e-6)) * dist), 0.0, 1.0)
    band = m * np.clip(2.0 * (1.0 - e), 0.0, 1.0)
    behind = np.clip((dist / np.float32(R) - 1.0) / np.float32(WRAP_BEHIND), 0.0, 1.0)
    return box, np.float32(WRAP_GAIN * ball.charge) * band * facing * behind * \
        np.minimum(1.0, (np.float32(R) / dist) ** 2)


def spill_light(matte: np.ndarray, ball: Ball) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """The ball's light pooled on the palm under it (SPILL): (box, (bh, bw) light over it, to multiply its light's
    colour by), a soft round of it SPILL's sigma of its radius about the palm's point, as bright as the ball is, on the
    hand's matte (0-1 float) only; or None (no point of the palm under it)."""
    if ball.px is None or ball.level <= 0:
        return None
    sig = SPILL[0] * ball.R
    box = _around(ball.px, ball.py, 3.0 * sig, *matte.shape)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    gx = np.exp(-0.5 * ((np.arange(x0, x1, dtype=np.float32) - np.float32(ball.px)) / np.float32(sig)) ** 2)
    gy = np.exp(-0.5 * ((np.arange(y0, y1, dtype=np.float32) - np.float32(ball.py)) / np.float32(sig)) ** 2)
    return box, np.float32(SPILL[1] * ball.level) * matte[y0:y1, x0:x1] * (gy[:, None] * gx[None, :])


# ---- the shot ------------------------------------------------------------------------------------------------------

def _cool(o: np.ndarray, q: np.ndarray, mm: np.ndarray, share: float = SKIN_COOL[1]) -> np.ndarray:
    """Code values (..., 3) of skin ``o`` lit by the lightning over what it was, ``q``, ``mm`` (...) how much of each
    is skin, in place: a cool white light on it, as far as it is lifted taken up to ``share`` of the way towards a cool
    white of its own brightness (SKIN_COOL); then, short of white, back towards its own colour at its new brightness as
    far as it takes to keep SKIN_KEEP of its saturation; and its red lifted at most SKIN_RED more than its green."""
    y = o @ fx._LUMA
    cool = np.minimum(y[..., None] * np.float32(SKIN_COOL[2]), 255.0)
    o += (cool - o) * (np.clip((y - q @ fx._LUMA) / np.float32(SKIN_COOL[0]), 0, 1) * np.float32(share) * mm)[
        ..., None]

    def sat(x: np.ndarray) -> np.ndarray:
        top = x.max(-1)
        return (top - x.min(-1)) / np.maximum(top, np.float32(1.0))

    own = np.minimum(q * ((o @ fx._LUMA) / np.maximum(q @ fx._LUMA, np.float32(1.0)))[..., None], 255.0)
    now = sat(o)
    back = np.clip((np.float32(SKIN_KEEP) * sat(q) - now) / np.maximum(sat(own) - now, np.float32(1e-3)), 0, 1)
    o += (own - o) * (back * mm * np.clip((np.float32(SKIN_KEEP_UNDER) - o.max(-1)) / 30.0, 0, 1))[..., None]
    o[..., 0] -= np.maximum((o[..., 0] - q[..., 0]) - (o[..., 1] - q[..., 1]) - np.float32(SKIN_RED), 0) * mm
    return o


def _keyed(alb: np.ndarray, light: np.ndarray, rgb: np.ndarray, cap: float) -> np.ndarray:
    """(h, w, 3) linear: a ``light`` (h, w) of colour ``rgb`` on surfaces of colour ``alb`` (h, w, 3), its luminance
    held under ``cap``, softly, each pixel's colour kept."""
    lit = alb * light[..., None] * rgb
    y = fx._lum(lit)
    return lit * (_soft_cap(y, cap) / np.maximum(y, np.float32(1e-9)))[..., None]


def _wide(x: np.ndarray, sigma: float) -> np.ndarray:
    """``x`` (h, w) blurred by a Gaussian ``sigma`` px wide, done small (it is smooth), its light let go past the
    frame's edge (none comes back in off it)."""
    h, w = x.shape
    q = max(1, int(sigma / 8))
    small = cv2.resize(x, (max(1, w // q), max(1, h // q)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / q, borderType=cv2.BORDER_CONSTANT)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def render(plate: np.ndarray, track: dict, mask: np.ndarray | None, timeline: Timeline, form: str, seed: int = 0,
           progress=None) -> np.ndarray:
    """The shot with lightning from the hand: uint8 RGB [T, H, W, 3] (cuesync.elements' contract). ``form``: "arc"
    (bolts from the fingertip the emitter was carried onto, or two fingers together, along the aim, into what the plate
    shows there) or "ball" (a crackling ball hovering over the palm where the emitter is). ``progress(stage, frac,
    msg)``, as cuesync.bending's, hears how far it is. Frames with no lightning, no light on them and the camera's
    exposure as it was are the plate's own bytes."""
    from .bending import shoulder

    if form not in FORMS["lightning"]:
        raise ValueError(f"unknown lightning form {form!r}; one of {FORMS['lightning']}")
    n, h, w = plate.shape[:3]
    if timeline.n != n:
        raise ValueError(f"the timeline has {timeline.n} frames, the plate {n}")
    if mask is not None and mask.shape[:3] != (n, h, w):
        raise ValueError(f"the mask is {mask.shape[:3]}, the plate {(n, h, w)}")
    say = progress or (lambda stage, frac, msg: None)
    say("lightning", 0.0, "Striking the lightning")
    hand = _hand(plate, track)
    if form == "arc":
        bolts, spots, struck = arc_frames(hand, timeline, seed, mask, plate)
        balls = [None] * n
        bright = levels(timeline) * hand.presence
    else:
        bolts, spots, balls = ball_frames(hand, timeline, seed, mask)
        struck = [[] for _ in range(n)]
        bright = np.array([0.0 if b is None else b.level for b in balls])
    on = bright > 0  # (a channel on screen, struck or dark between strokes, and a short gap between them)
    for t in np.flatnonzero(on[:-1] & ~on[1:]) + 1:
        back = np.flatnonzero(on[t:t + LIGHT_BRIDGE + 1])
        if back.size:
            on[t:t + back[0]] = True
    dip = exposure_dip(bright, *EXPOSURE[form])  # the camera exposing down for the light, in stops
    depth, hand_gain, room_gain, near_gain = LIGHT
    hand_rgb, room_rgb = LIGHT_RGB[form]
    key_hand_gain, key_room_gain, key_flicker = KEY
    key_hand, key_room = KEY_RGB
    side = min(h, w)
    floor = GLOW_FLOOR * ARC_WIDTH[1] if form == "arc" else 0.0  # (times the hand's width scale)
    soft_px = (DEFOCUS[0] * side / 1080, DEFOCUS[1] * side / 1080)
    move = np.gradient(hand.xy, axis=0) * SHUTTER if n > 1 else np.zeros_like(hand.xy)  # the hand, shutter open
    picks = np.linspace(0, n - 1, 8).astype(int)
    some = np.stack([lin(plate[i][::2, ::2]) for i in picks])
    white, tint = fx.plate_white(some), fx.plate_tint(some)
    cast = room_cast(some, white, tint, None if mask is None else mask[picks, ::2, ::2])
    room_rgb, key_room = ((c / cast / float((c / cast) @ fx._LUMA)).astype(np.float32) for c in (room_rgb, key_room))
    room_cap = ROOM_CAP * white  # (a surface reads its albedo times the plate's white: brightened this much more)
    del some
    noise = fx.shadow_noise(plate)
    bright = _bright(plate, mask) if form == "arc" else 0.0
    hf, wf = max(1, h // FIELD_Q), max(1, w // FIELD_Q)
    out = np.empty_like(plate)

    def behind(t: int, box: tuple, s: float, tips: list) -> np.ndarray | None:
        """(bh, bw) over ``box``: how much of the arcs passes behind the hand on frame t (HIDE_NEAR from the burning
        ``tips``), or None (a ball, no matte, or none of the hand there)."""
        if form != "arc" or mask is None or not tips:
            return None
        y0, y1, x0, x1 = box
        m = mask[t, y0:y1, x0:x1]
        if not m.any():
            return None
        xx, yy = np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32)[:, None]
        near = np.min([np.hypot(xx - np.float32(x), yy - np.float32(y)) for x, y in tips], axis=0)
        a, b = HIDE_NEAR[0] * s, HIDE_NEAR[1] * s
        return m.astype(np.float32) * np.float32(1 / 255) * np.clip((near - np.float32(a)) / np.float32(b - a), 0, 1)

    def made(t: int):
        """Frame t's lightning: (box, linear RGB layer over it, its power in plate px, what it hides of the plate), or
        None."""
        ball = balls[t]
        flashes = [(x, y, POOL_GLOW[0] * side / 1080.0, 1.0) for x, y, *_ in struck[t]]  # (where they strike)
        box = _box(bolts[t], spots[t] + flashes, hand.s[t], h, w)
        if ball is not None:
            bb = _box([], [(ball.x, ball.y, ball.R / 3.0, 1.0)], hand.s[t], h, w)  # (its body, out to R, and its glow)
            box = bb if box is None else \
                (min(box[0], bb[0]), max(box[1], bb[1]), min(box[2], bb[2]), max(box[3], bb[3]))
        if box is None:
            return None
        s = float(hand.s[t])
        soft = float(np.clip(softness(plate[t], hand.xy[t], s), *soft_px))
        tips = [(x, y) for x, y, _, amp in spots[t] if amp > 0] if form == "arc" else []  # (a ball's strokes are
        # instant: only the crackle at a fingertip glows on while the shutter is open)
        E, veil = layer(bolts[t], spots[t], ball, s, box, GLOW_CAP[form], soft, move[t], tips, seed, t, side,
                        floor * _width_k(s), behind(t, box, s, [(x, y) for x, y, *_ in spots[t]]), struck[t], bright)
        return box, E, float(fx._lum(E).sum(dtype=np.float64)), veil

    layers: dict[int, tuple | None] = {}
    smalls: list[np.ndarray | None] = []  # each frame's channels, small, at unit power: the light's shape
    power = np.zeros(n)
    fields: dict[int, tuple] = {}
    last = None
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        got: list = [None] * (b - a)
        fx._each(b - a, lambda i: got.__setitem__(i, made(a + i)))
        for t in range(a, b):
            layers[t] = got[t - a]
            small = None
            if layers[t] is not None and layers[t][2] > 0 and form == "arc":
                (y0, y1, x0, x1), E, power[t], _ = layers[t]
                power[t] /= max(1, len(spots[t]))  # (two fingers together light the room as one does: a tip each)
                small = np.zeros((hf, wf), np.float32)
                sy0, sx0 = y0 // FIELD_Q, x0 // FIELD_Q
                sy1, sx1 = max(sy0 + 1, min(hf, y1 // FIELD_Q)), max(sx0 + 1, min(wf, x1 // FIELD_Q))
                small[sy0:sy1, sx0:sx1] = cv2.resize(fx._lum(E), (sx1 - sx0, sy1 - sy0), interpolation=cv2.INTER_AREA)
                small /= max(float(small.sum(dtype=np.float64)), 1e-12)
                last = small
            smalls.append(small)
        strength, peak = light_power(power[:b], on[:b])  # (causal: frames already made don't change)
        keyed, _ = light_power(power[:b], on[:b], key_flicker)
        for t in range(a, b):
            pool = [(wt, smalls[t - k]) for k, wt in enumerate(LIGHT_POOL) if t - k >= 0 and smalls[t - k] is not None]
            if strength[t] <= 0 or form == "ball":  # (the ball's light is its own, close round it: ball_light)
                fields[t] = None
                continue
            src = sum(wt * sm for wt, sm in pool) / sum(wt for wt, _ in pool) if pool else last
            if src is None:
                fields[t] = None
                continue
            s_f = hand.s[t] / FIELD_Q
            near, wall, far = fx.spread(src * np.float32(1 / (FIELD_Q * FIELD_Q)),  # (per small px: the mean of the
                                        [max(depth * s_f, 1.0), ROOM_NEAR * s_f, fx.FAR * s_f])  # px it covers)
            # the hand's light and the room's at the burst's peak (where HAND_CAP and ROOM_CAP hold them), scaled down
            # with the burst after the cap, so the cap does not hold them up as it dies (capped as it was, the room kept
            # 45 % of its light a frame after the last channel); and the key, the lightning's own image blurred wide,
            # at the burst's peak (KEY_CAP), scaled after the cap to as strong as it is now with its strokes' flicker
            room = (far * np.float32(room_gain) + wall * np.float32(near_gain)) * np.float32(peak[t])
            key = _wide(src, KEY_SIGMA * side / FIELD_Q) * np.float32(peak[t] / (FIELD_Q * FIELD_Q))
            now = np.float32(strength[t] / peak[t])
            fields[t] = (near * np.float32(hand_gain * peak[t]), np.float32(room_cap) * np.tanh(room / room_cap) * now,
                         src, now, key, np.float32(keyed[t] / peak[t]))

        def finish(i: int) -> None:
            t = a + i
            stops = float(dip[t]) if form == "ball" or fields[t] is not None else 0.0  # (no light, no dip for arcs)
            if layers[t] is None and fields[t] is None and stops < DIP_MIN:
                out[t] = plate[t]
                return
            P0 = lin(plate[t])
            P = P0.copy()
            m = None
            if fields[t] is not None or balls[t] is not None:  # (the hand's matte, feathered: LIGHT_FEATHER)
                m = None if mask is None else mask[t]
                if m is not None and not m[::4, ::4].any():
                    m = None
                if m is None and mask is None:
                    m = fx.arm_matte(tuple(hand.xy[t]), tuple(hand.u[t]), float(hand.s[t]), w, h)
                elif m is not None:
                    m = cv2.GaussianBlur(m.astype(np.float32) * np.float32(1 / 255), (0, 0),
                                         max(0.5, LIGHT_FEATHER * side / 1080.0))
            if fields[t] is not None:
                near, light, src, k, key, k_key = fields[t]
                light = cv2.resize(light, (w, h), interpolation=cv2.INTER_LINEAR)  # (the room's)
                key = cv2.resize(key, (w, h), interpolation=cv2.INTER_LINEAR)
                alb = fx.albedo(P0, white, tint)
                lit = light[..., None] * room_rgb
                if m is None:
                    P += _keyed(alb, key * np.float32(key_room_gain), key_room, KEY_CAP[1]) * k_key
                else:
                    shade = fx.hand_shading(m, src, float(hand.s[t]))
                    on_hand = cv2.resize(near, (w, h), interpolation=cv2.INTER_LINEAR) * shade
                    on_hand = m * (_soft_cap(on_hand, HAND_CAP) * k + light)  # (and the room's light, where it
                    lit *= (1.0 - m)[..., None]  # falls on the hand, in the hand's colour)
                    lit += on_hand[..., None] * hand_rgb
                    P += (_keyed(alb, key * np.float32(key_room_gain), key_room, KEY_CAP[1]) * (1.0 - m)[..., None]
                          + _keyed(alb, key * (0.5 + 0.5 * shade) * np.float32(key_hand_gain), key_hand, KEY_CAP[0])
                          * m[..., None]) * k_key  # (each capped whole, then blended: capped in parts, the two
                    # added up to a bright rim along the hand's outline)
                for x, y, ax, ay, g, fresh in struck[t]:  # (the pool of each strike's light on what it strikes,
                    # brightest on its first frame: POOL)
                    got_ = pool_light(x, y, ax, ay, g * (POOL[5] if fresh else 1.0), side, h, w)
                    if got_ is not None:
                        (y0, y1, x0, x1), pl = got_
                        lit[y0:y1, x0:x1] += (np.float32(POOL[6]) * np.tanh(pl * np.float32(POOL[3] / POOL[6])))[
                            ..., None] * room_rgb
                        P[y0:y1, x0:x1] += pl[..., None] * (np.float32(POOL[4]) * room_rgb)
                P += alb * lit
                stops = min(stops, ARC_DIP_ROOM * room_lift(P, P0, m))  # (never so far down that the room away from
                # the arcs goes darker than the plate)
            lit_box = None
            if balls[t] is not None:  # the ball's light round it on the hand and the room, and on the edges of the
                # fingers facing it and the palm under it, in the hand's colour
                got_ = ball_light(balls[t], side, h, w)
                lights = [] if m is None else [g_ for g_ in (wrap_light(m, balls[t], side), spill_light(m, balls[t]))
                                               if g_ is not None]
                if got_ is not None:
                    lit_box, on_hand, on_room = got_
                    y0, y1, x0, x1 = lit_box
                    mm = np.zeros((y1 - y0, x1 - x0, 1), np.float32) if m is None else m[y0:y1, x0:x1, None]
                    P[y0:y1, x0:x1] += fx.albedo(P0[y0:y1, x0:x1], white, tint) * \
                        (on_hand[..., None] * hand_rgb * mm + on_room[..., None] * room_rgb * (1.0 - mm))
                for (y0, y1, x0, x1), wl in lights:
                    P[y0:y1, x0:x1] += fx.albedo(P0[y0:y1, x0:x1], white, tint) * (wl[..., None] * hand_rgb)
            if stops >= DIP_MIN:  # (the lightning itself is drawn as the camera shows it, exposed down)
                e = np.float32(2.0 ** -stops)
                P *= e
                P0 *= e
            if lit_box is not None and m is not None:  # (on the hand, its light a cool white: _cool)
                y0, y1, x0, x1 = lit_box
                mm = m[y0:y1, x0:x1]
                sub = P[y0:y1, x0:x1]
                was = np.float32(255.0) * np.clip(sub, 0.0, 1.0) ** np.float32(1 / 2.2)
                o = _cool(was.copy(), np.float32(255.0) * np.clip(P0[y0:y1, x0:x1], 0.0, 1.0) ** np.float32(1 / 2.2),
                          mm, BALL_COOL)
                sub *= np.where(mm[..., None] > 0, ((o + 0.5) / (was + 0.5)) ** np.float32(2.2), 1.0)  # (as a share of
                # itself: skin it lifts over white keeps what is over, for the roll-off to white)
            share = None
            skin = None
            if form == "ball" and m is not None:  # (lit skin keeps its colour as it rolls off: BRIGHT_TONE, as the
                ys, xs = np.nonzero(m[::4, ::4] > 0.02)  # ball's own light does)
                if ys.size:
                    skin = (4 * ys.min(), 4 * ys.max() + 4, 4 * xs.min(), 4 * xs.max() + 4)
            if layers[t] is not None:
                (y0, y1, x0, x1), E, _, veil = layers[t]
                if veil.any() and balls[t] is not None:  # (what shows through the ball's dense middle is
                    for img in (P, P0):  # scattered in it, and hidden)
                        sub = img[y0:y1, x0:x1]
                        sub += (_blur(sub, SCATTER * balls[t].R) - sub) * veil[..., None]
                        sub *= 1.0 - veil[..., None]
                elif veil.any():  # (round the channels, a bright plate a share darker: CONTRAST, BRIGHT_BAND)
                    c0, c1, c2, c3 = CONTRAST[2]
                    y_ = fx._lum(P0[y0:y1, x0:x1])
                    dim = veil * np.maximum(np.clip((y_ - c0) / np.float32(c1 - c0), 0.0, 1.0), np.float32(bright)) * \
                        np.clip((c3 - y_) / np.float32(c3 - c2), 0.0, 1.0)
                    dim = 1.0 - dim[..., None] * np.array([1.0, 1.0, 1.0 - (1.0 - BRIGHT_BAND[2]) * bright], np.float32)
                    for img in (P, P0):
                        img[y0:y1, x0:x1] *= dim
                E = fx.grain(E[None], noise, seed=seed, frame0=t)[0]
                if balls[t] is not None:  # (the ball's own light keeps its hue as it rolls off to white, as much
                    # as it outshines what it lies over, and all of what is behind its dense middle, skin or wall
                    # alike: BRIGHT_TONE; rolled off as the skin or the wall there, a gap between two fingers showed
                    # through it as a sharp-edged window)
                    own = fx._lum(E)
                    share = np.maximum(own / (own + fx._lum(P[y0:y1, x0:x1]) + np.float32(1e-6)),
                                       np.minimum(np.float32(2.0) * veil, np.float32(1.0)))
                P[y0:y1, x0:x1] += E
                if share is not None:
                    toned = hue_shoulder(P[y0:y1, x0:x1])
            if skin is not None:
                kept = hue_shoulder(P[skin[0]:skin[1], skin[2]:skin[3]])
            shoulder(P)  # (the channels' cores clip to white softly)
            if skin is not None:
                sub = P[skin[0]:skin[1], skin[2]:skin[3]]
                sub += (kept - sub) * m[skin[0]:skin[1], skin[2]:skin[3], None]
            if share is not None:
                sub = P[y0:y1, x0:x1]
                sub += (toned - sub) * share[..., None]
            np.maximum(P, P0, out=P)  # (but the plate's own highlights stay as bright as they were)
            out[t] = gam(P)
            if form == "arc" and m is not None:  # (on the hand, a cool white light: SKIN_RED, SKIN_COOL)
                ys, xs = np.nonzero(m[::4, ::4] > 0.02)
                if ys.size:
                    y0, y1, x0, x1 = 4 * ys.min(), 4 * ys.max() + 4, 4 * xs.min(), 4 * xs.max() + 4
                    o = _cool(out[t][y0:y1, x0:x1].astype(np.float32), gam(P0[y0:y1, x0:x1]).astype(np.float32),
                              m[y0:y1, x0:x1])
                    out[t][y0:y1, x0:x1] = np.clip(o + 0.5, 0, 255).astype(np.uint8)

        fx._each(b - a, finish)
        for t in range(a, b):
            layers.pop(t, None)
            fields.pop(t, None)
        say("lightning", b / n, f"Striking the lightning: frame {b} of {n}")
    return out
