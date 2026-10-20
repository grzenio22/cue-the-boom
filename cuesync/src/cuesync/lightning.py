"""Lightning made from the hand (cuesync.elements): arcs struck from the fingertips, or a crackling ball of them over
an open palm, drawn frame by frame from the hand's track and lit onto the plate.

- **Channels.** A bolt is a jagged line (recursive midpoint displacement: each midpoint pushed across its segment by a
  share of that segment's length, gently at the coarse levels so the channel runs on, sharply at the fine ones so it
  kinks) with forks that leave it more often towards its far end, each much dimmer and thinner, as a streamer branches
  when it runs out. A finger's first stream is its main channel; the others beside it are dimmer, thinner and shorter
  streamers that leave well off its heading on either side, so a loud voice is one channel with branches, not a rope
  of equal wires. Two fingers held together each strike a main channel, turned apart so the two separate; the
  streamers are the first finger's.
- **Strikes.** A channel is struck and then re-struck down the same path for 1-3 frames (real lightning's return
  strokes), flashing or going dark from frame to frame, before a new path is found. While it lives, its root rides the
  fingertip (or the palm) and its far end stays where it struck (carried by the camera on a handheld shot): a path
  drawn anew every frame jumped about like static on a screen, and one drawn once and carried whole with the hand
  swung its far end through the room. A new main channel leaves near its last aim, which sways slowly, so the
  streams sweep rather than jump; its streamers are struck anew with it.
- **The ball.** A sphere of plasma held over the palm: a round body of saturated blue, as dense nearly to its outline
  as at its middle and churning, a broken ring of light at its limb, its white-hot middle, filaments from that out to
  its skin (in 3D, turning slowly, mostly across the picture: one pointing at the camera is a short scribble, a knotted
  one is drawn again), broken arcs crawling over its skin and arcs jumping from it to a fingertip. Dense enough to hide
  most of the palm behind it, and only inside its outline, it reads blue over skin already at the top of red while
  the palm round it keeps its lines. It hovers off the palm along the palm's normal, and lights a pool on the palm under
  it, the fingers round it and the edges of them facing it in its own cyan-blue. The lens's glare round it is laid
  over the picture as shown, so it is the same light over skin and over the wall.
- **The voice.** The louder the voice, the more streams leave each finger, the further they reach and the thicker
  they are; a hit strikes every stream afresh at once, a little further. Between strikes, and for a moment after the
  voice stops, a crackle stays at the burning fingertip, and arcs crawl up a steady finger. The ball grows while the
  voice holds and shrinks when it stops.
- **The look.** The channel itself is far over white (a phone clips it to white with a blue rim), the air round it
  glows blue, then violet further out; only the brightest channels clip (CORE_GAMMA), so a fork stays a dimmer line
  beside its channel. The channels are as soft as the plate round the hand (its focus and the phone's smoothing,
  measured on each frame), and smeared with the finger where they leave it, as the plate's finger is, though never
  so far that a channel comes away from its fingertip. A small bolt glows as hot as a strike's. The glow of many
  channels piled up is held under GLOW_CAP. The fingertip burns white-blue; the ball's middle is a white spark.
- **The light it throws** (cuesync.fx_light's physics): every lit pixel of the channels lights the hand close in front
  of it, the wall just behind it and the room further behind, shaded on the round hand, times each surface's own
  colour, so skin keeps its texture under it: the arcs' blue-violet on the hand, a cooler blue-white on the room (two
  fingers light it as one does). The large-area light does not strobe with the channels (a
  photosensitivity limit: at most about 3 big changes a second): its strength holds the strokes' peak while they
  strike and dies within a few frames of the last, and its shape is pooled over the last few frames' channels; only
  the channels and their glow flicker frame to frame. On the hand it is capped (HAND_CAP), at the peak of the burst
  so the cap does not hold it up as it dies. The camera exposes down while the lightning shines (as a phone's does
  for a bright light in the picture: a flash for the arcs, settling once for the ball's whole phrase), so skin already
  at the top of red has room to take the blue light as blue, and the plate's own highlights are never dimmed by the
  roll-off to white.

Everything is numpy and OpenCV; the random parts come from ``seed`` (the strikes, in one pass over the frames, then
each frame's own draws from the seed and its index), so the same inputs give the same bytes on any number of threads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import fx_light as fx
from .elements import FORMS, Timeline
from .hand_track import REACH_OUT, _pointing, finger_reach
from .transfer import gam, lin

_STRIKES, _CRACKLE, _FRAME, _SPARK, _LIMB, _PLASMA = 11, 12, 13, 14, 15, 16  # mixed into each seed: each random part
# its own stream

# ---- the voice
GATE = (0.15, 0.25)  # energy: under the first a sound is a breath or a click, not the voice (it lit stray sparks on a
# silent hand); over the second it plays in full, faded in between
CRACKLE_LEFT = 0.72  # the crackle at the fingertip outlasts the voice, dying by this share a frame (about 0.3 s)
CRACKLE_MIN = 0.06  # below this the crackle is gone
CRACKLE_WIDTH, CRACKLE_INTEN = 1.6, 0.8  # a crackle arc's width (times the hand's width scale) and intensity: a few
# bright arcs (four 1 px hairs at 0.45 could not be seen at phone size, and the tip glowed with no source)

# ---- arcs from the fingertips
ARC_REACH = (1.4, 5.0)  # hand sizes: how far a stream reaches, from the quietest voice to the loudest
ARC_WIDTH = (1.3, 4.0)  # px for a hand HAND_PX across: the main channel's width from the quietest voice to the loudest
HAND_PX = 220.0  # widths are drawn for a hand this many px across (the dev takes') and scale with it, within WIDTH_K
WIDTH_K = (0.6, 1.8)
ARC_STREAMS = 3  # at most this many streams per finger (one for a quiet voice): the first is the main channel
SIDE_ON = (0.3, 0.85)  # the voice's level over which a finger's second, and third, stream strikes: three only at the
# voice's peak (from 0.75 on, two streamers bunched round the main channel's root on every loud frame)
SIDE_INTEN, SIDE_WIDTH, SIDE_REACH = 0.5, 0.5, (0.45, 0.8)  # a streamer beside the main channel, against it: dimmer,
# thinner and shorter (as bright, as thick and as long, three streams braided into a rope with no main channel)
SIDE_LIFE = 2  # frames a streamer's path is kept at most (struck anew with its main channel too: held as long, all
# the streams froze together for three frames)
SIDE_GAP = (18.0, 26.0)  # degrees a streamer leaves off its main channel's heading, the two on opposite sides (aimed
# in a fan round the finger's aim, each swaying on its own, they came within a few degrees of the main channel and ran
# beside it for half its reach)
ARC_WANDER, ARC_WANDER_HZ = 5.0, (0.6, 1.7)  # degrees and Hz: the main channel's aim sways slowly (a new path anywhere
# in the fan read as random flashes, not one storm)
ARC_JITTER = 4.0  # degrees: each new main channel's own turn off its aim
LIFE, LIFE_P = (1, 2, 3), (0.25, 0.45, 0.3)  # frames a path is kept, and how often each
RESTROKE = (0.45, 0.5)  # a re-stroke flashes with this chance, plus this times the level; else the channel is dark
HIT_REACH = 0.3  # a hit's strike reaches this much further at full size
ROUGH, FORK = 0.2, (0.08, 0.16)  # a stream's jaggedness (share of a segment), its forking (at the quietest, more
# when louder: dim, its forks branch a loud strike out without braiding it)
FORK_INTEN, FORK_WIDTH = (0.15, 0.35), 0.45  # a fork's intensity and width against the channel it leaves (at 0.35-0.6
# and 0.6 the forks clipped to the same white as their channel: a bundle of equal wires)
FORK_TURN = ((10.0, 35.0), (15.0, 50.0))  # degrees a fork turns off a main channel's heading, and off a fork's
FORK_PULL = 0.3  # a main channel's fork is pulled this far back towards the channel's own course (turned 50 degrees
# on top of the fan, the wander and the jitter, one crossed the shoes, half off the finger's aim)
FINGER_NEAR = 0.6  # hand sizes: the tap names the fingertip within this of the emitter; a second finger beside it
# (index and middle) arcs too when it is out straight (REACH_OUT) and its tip within FINGER_PAIR of the first's
FINGER_PAIR = 0.8
PAIR_SPLIT = 14.0  # degrees: two fingers' main channels leave this far apart, each turned away from the other (along
# one aim from tips side by side, they ran together as one thick rope); the second finger strikes its main channel
# only, its streamers left to the first (with three streams each, two fingers were a thicket of equal wires)

# ---- the ball over the palm
BALL_R = (0.25, 0.55)  # hand sizes: its radius, from a voice just begun to one held long and loud (from 0.15 it
# mostly played small: the charge seldom nears 1)
BALL_LIFT = 0.3  # hand sizes: it hovers this far off the palm, along the palm's normal (at the palm's depth it sat
# flat on it, a decal)
BALL_ON, BALL_BRIDGE = (0.03, 0.015), 3  # it shows once the voice (its loudest over the last BALL_BRIDGE frames) or
# its charge passes the first, and goes only when both are under the second: as a voice began softly, a frame's dip in
# it, before the charge had built up, put the ball out and on again
BALL_FADE = 3  # frames it fades in over
BALL_GROW, BALL_SHRINK = 10.0, 3.0  # frames: it grows towards the voice's level this slowly, and shrinks this fast
# (at 5 a ball hung on, flickering, through the second's pause between two sounds)
BALL_FILAMENTS = (5, 14)  # filaments from its middle to its skin, small to full grown
BALL_FLAT = 0.5  # radii: a filament that reaches less far across the picture than this is not drawn (one pointing
# at the camera, or away, is a short jagged curl in the middle)
BALL_DEPTH = 0.45  # a filament leaves its middle at a random angle, its depth (towards or away from the camera)
# scaled by this: mostly across the picture
BALL_SKIN, BALL_GROUND = 4, 2  # arcs crawling over its skin, and arcs jumping from it to a fingertip or knuckle
SKIN_OUTLINE, SKIN_ROUGH = 0.25, 0.4  # how often a skin arc runs along its outline, and how jagged it is; each draws
# 3-5 of the 8 stretches of its great circle (half of them on the outline, whole and smooth, drew a circle round it)
GROUND_P = (0.25, 0.45)  # a jumping arc strikes with this chance, plus this times the level (more often, and thicker,
# they outshone the ball)
BALL_ROUGH = 0.22  # its filaments' jaggedness (at 0.3 they tangled and sprawled over the palm: no longer a ball)
KNOT_TURN, KNOT_STRAIGHT = 200.0, 0.45  # a filament drawn again when, cut into 12 equal pieces as seen, it turns more
# than this many degrees one way over 5 of them, or its ends are under this share of its length apart: a tight squiggle
# reads as a knot of wire, not a spark
BALL_TURN = 0.05  # radians a frame: it turns slowly (its filaments' far ends move, but never jump)
BALL_PULSE = 0.12  # a hit swells it by this share, gone over 3 frames
BALL_CORE, BALL_SPARK = 0.09, 2.4  # radii, and amplitude at the loudest: its middle, clipped to white-cyan (at 0.05
# and 1.2 the ball had no source; at 0.11 it was a flat white disc)
BALL_BODY = 1.0  # the glowing air inside it, at its middle, in BODY_RGB: a round body of saturated blue, not a white
# dot in a haze (at 1.3 it was a flat sky-blue disc, at 0.8 a dark blue marble)
BODY_RGB = np.array([0.06, 0.35, 1.0], np.float32)
BODY_FALL = 0.3  # its glow falls off this much from its middle to its edge (as deep as the sphere is there): round
# (at 0.45 its outer part was mauve)
BODY_INNER = (1.0, 0.35)  # and it glows this much more round its middle (a Gaussian of that sigma, in radii), paler
# there for its brightness: hottest inside (evenly lit, it was a blue marble)
BODY_PLASMA, PLASMA_WAVES = 0.45, 5  # its glow is this much brighter along ridges where this many waves through it
# cross, drifting: churning plasma, not a flat fill
BALL_VEIL = 0.85  # and so dense it hides this much of the palm behind its middle, inside its outline only: added over
# a palm at the top of red, even a bright cyan-blue made it pink-white (under 0.75 it read lavender); spread past its
# edge (a wide Gaussian), the palm round it went milky and lost its lines
BALL_THICK = 4.0  # its optical depth through the middle: its body glows and hides the palm as 1 - exp(-this x the
# depth of the sphere along the line of sight), nearly as much near its edge as at its middle (as the depth alone, the
# outer half of it was a pale lavender rim; at 2.5 a mauve one)
BALL_FORMED = 0.25  # its charge (and level, as its body's glow is) at which it is as dense as it gets; it grows bigger
# after (as dense as its charge, a ball half charged showed the palm through it pink)
BALL_EDGE = 2.5  # px at 1080: its outline is as soft as this (the plate's own edges are about as soft)
LIMB_AT, LIMB_W, LIMB = 0.93, 0.035, 0.45  # radii, radii, and share of its middle's brightness: a thin ring (a
# Gaussian of that sigma) just inside its outline glows, where the line of sight grazes its shell (a sphere of plasma
# brightens towards its limb); as a shell seen through, its faces in front of the middle lit wedges across it, and from
# 0.65 radii out it was a thick band with square ends
LIMB_RGB = np.array([0.3, 0.72, 1.0], np.float32)  # (paler, at 0.55 of its middle, its pieces were white dashes)
LIMB_WAVES, LIMB_SPIN, LIMB_CUT, LIMB_FADE = (3, 5, 9, 14), 0.25, (0.1, 0.15), 0.5  # the limb is broken by waves this
# many times round, each drifting up to LIMB_SPIN radians a frame; where their sum is under the first (give or take the
# second, frame to frame) it is dark, coming up over the last: arcs of light round it, not a drawn circle (coming up
# over 0.3, their ends were cut square)
CONTACT = 0.025  # hand sizes: the spark where a filament meets the ball's skin, facing us
TARGETS = (4, 8, 12, 16, 20, 5, 9, 13)  # the landmarks a grounding arc may jump to: fingertips and knuckles
POOL_R, POOL_GAIN = 0.6, 0.9  # its radii, and gain: the pool of light it throws on the palm right under it, at the
# loudest (without it the ball didn't light what it hovers over)
GLARE = ((0.45, 0.8), (0.35, 0.2), (0.14, 1.4))  # the lens's glare round it, in the picture as shown: (strength at
# the loudest, size in radii) of a halo round its middle (a Gaussian of that sigma), which flickers with it; of its body
# bloomed past its outline (falling off over that much either side of it); and of a wide veil (a Gaussian), the last two
# following its charge. Screened over the shown picture, it is the same light over skin and over the wall: added in
# linear light under the tone curve, it was cyan over the wall and pink over the palm, and the fingers' outlines showed
# through it
GLARE_RGB = np.array([0.3, 0.65, 1.0], np.float32)  # (as shown, not linear; a paler cyan whitened the body)
WRAP_PX, WRAP_GAIN, WRAP_REACH = 5.0, 1.5, 3.0  # px at 1080, gain at the loudest, radii: the fingers' edges facing the
# ball, within about twice the first inside the hand's outline, catch its light, falling off as the inverse square of
# their distance, within the last (lit only through the hand's broad light, the edges next to it stayed pink)
WRAP_RGB = np.array([0.15, 0.55, 1.0], np.float32) / np.float32(0.497)  # (at luminance 1)

# ---- the look (linear light)
CORE_RGB = np.array([0.92, 0.96, 1.0], np.float32)  # the channel itself: white, the faintest blue
CORE_GAIN = 3.0  # the channel is far over white; the shoulder takes it to a clipped white with a blue rim
CORE_GAMMA = 1.6  # the core's intensity is raised to this before CORE_GAIN, so only the brightest channels clip
# (with the gain alone everything over a third went the same white: forks as bright as their channel)
GLOWS = (  # (sigma in hand sizes, peak over a 1 px line at core 1, linear rgb): the air round the channel glows blue,
    (0.008, 0.55, (0.25, 0.50, 1.00)),  # then violet further out (saturated: a paler blue over skin read pink-white)
    (0.025, 0.22, (0.12, 0.30, 1.00)),
    (0.08, 0.08, (0.12, 0.22, 1.00)),
    (0.25, 0.03, (0.25, 0.15, 1.00)),
)
GLOW_CAP = {"arc": 0.6, "ball": 1.6}  # linear luminance: the glow is held under this, softly. Arcs over a wall held
# to 0.15 read as white paint; a ball held to 0.3 had no bloom and was dimmer than the palm round it
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
CRAWL_STILL = 15.0  # px at 1080: on a frame the fingertip moves further than this no arc crawls up the finger (smeared
# with it, the arc was a pale band across the finger)
GLOW_FLOOR, GLOW_HOT = 0.6, (0.4, 0.6)  # an arc's glow is drawn from a channel at least this share of a loud main
# channel's width once its intensity passes the second (faded in from the first): a small bolt on a quiet voice is
# short but as hot as a strike's (with its glow from its own thin, dim line, it read as a drawn line)
SPARK_RGB = np.array([0.86, 0.94, 1.0], np.float32)  # the burning fingertip and the ball's middle: white, a little cyan
HAZE_RGB = np.array([0.05, 0.35, 1.0], np.float32)  # the halo round a spark: cyan-blue (the violet-blue it was read
# as lavender over skin)
CORONA, CORONA_MIN = 0.07, 0.35  # hand sizes, and its least brightness for the crackle's level: the burning
# fingertip's spot burns while the voice holds (at 0.04 it could not be seen)

# ---- the light it throws
LIGHT_RGB = {form: tuple(np.array(rgb, np.float32) / float(np.array(rgb, np.float32) @ fx._LUMA) for rgb in pair)
             for form, pair in (
    # its colour, at luminance 1, (on the hand, on the room):
    ("arc", ((0.45, 0.33, 1.0),  # on the hand blue-violet (on skin at the top of red, a greener blue added only green
             (0.30, 0.42, 1.0))),  # and blue: the hand went cyan); on the room a cooler blue-white (in the hand's
    # violet the olive wall went mauve and the white wall and the floor violet)
    ("ball", ((0.15, 0.4, 1.0),) * 2))}  # cyan-blue, as the ball's body is (in the arcs' violet-blue the palm round it
# went pink)
LIGHT = {"arc": (0.2, 2.5, 5.0, 1.5), "ball": (0.45, 1.4, 5.0, 0.0)}  # (how far in front of the hand the channels
# are, in hand sizes; gain on the hand; gain on the room; gain on the wall right behind them, ROOM_NEAR away). A thin
# channel has no radius to stand off by; at fx_light's own HAND_DEPTH of it the palm under a ball went white. A ball
# lit at 0.4 left the fingers round it as they were (its saturated blue body has little luminance to light by: at 0.8
# and 4 it hardly lit the palm round it)
ROOM_NEAR = 1.0  # hand sizes: the wall the arcs strike is this far behind them (with the room's far light alone, a
# lone bolt lit nothing next to it; at 0.5 the light on it jumped about with each new path)
HAND_CAP = 1.0  # the light may brighten the hand by at most this (+1 stop), softly: at fx_light's 4 a palm under the
# ball went white and lost its lines
LIGHT_HOLD, LIGHT_LETGO = 4, 0.55  # the light's strength holds the strokes' peak over this many frames (following
# each stroke, the room strobed at 12 Hz; held over 2 or 3, a "ka-ka-ka" still strobed it at 6), then lets go by this
# share a frame (by 0.85, the hand and the room stayed lit half a second after the last channel: a glow with no
# source)
EXPOSURE = {"arc": (0.3, LIGHT_HOLD, 1.5, 2.0), "ball": (0.35, 36, 4.0, 40.0)}  # (stops, frames held, frames rising,
# frames falling): the camera exposes down this much for the lightning at its brightest (the voice's level for the
# arcs, the ball's brightness), following the brightest it has been over the frames held, so skin at the top of red has
# room to take the blue light. The arcs are a flash; the ball settles once for a held phrase and stays (following the
# ball itself by 0.9 stops, the whole room breathed darker and lighter with each sound, three times in 4 s)
DIP_MIN = 0.005  # stops: less than this and the camera is as it was
LIGHT_GONE = 0.02  # under this share of its burst's peak the light is gone
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


def bolt_tree(p0, u, length: float, rng: np.random.Generator, width: float, rough: float = ROUGH,
              fork: float = FORK[0], min_seg: float = 3.0, inten: float = 1.0, depth: int = 0, max_depth: int = 2,
              out: list | None = None) -> list[tuple[np.ndarray, float, float]]:
    """A bolt from ``p0`` heading ``u`` for ``length`` px with its forks: [(points, width px, intensity)], the main
    channel first. Forks leave at its points, more often towards its far end, FORK_TURN degrees off its local heading
    (a main channel's pulled back towards its course), a quarter to two thirds of what is left of it long, much dimmer
    and thinner; they fork once more themselves."""
    out = [] if out is None else out
    p0, u = np.asarray(p0, np.float64), np.asarray(u, np.float64)
    pts = displace(p0, p0 + u * length, rng, rough, min_seg, curve=rng.normal(0, 0.07))
    out.append((pts, width, inten))
    if depth >= max_depth:
        return out
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    for along in np.linspace(0.1, 0.9, 16):  # sixteen places a fork may leave from
        if rng.random() >= fork * (0.3 + 1.4 * along) * (0.5 if depth else 1.0):
            continue
        i = int(np.searchsorted(cum, along * cum[-1]))
        j = min(i + max(2, len(pts) // 40), len(pts) - 1)
        v = _rot(_unit(pts[j] - pts[i]), rng.choice([-1, 1]) * rng.uniform(*FORK_TURN[min(depth, 1)]))
        if depth == 0:
            v = _unit((1.0 - FORK_PULL) * v + FORK_PULL * u)
        bolt_tree(pts[i], v, (cum[-1] - cum[i]) * rng.uniform(0.25, 0.65), rng, width * FORK_WIDTH, rough, fork,
                  min_seg, inten * rng.uniform(*FORK_INTEN), depth + 1, max_depth, out)
    return out


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


@dataclass
class _Path:
    """A channel kept for its life: struck on ``born`` with its root at ``a`` and its far end at ``b``; ``to``: the
    landmark a grounding arc jumps to (-1: it didn't strike this time)."""

    born: int
    life: int
    a: np.ndarray
    b: np.ndarray
    tree: list
    to: int = -1


# ---- the hand ------------------------------------------------------------------------------------------------------

@dataclass
class _Hand:
    """The track in the plate's pixels: the emitter (n, 2), its direction (n, 2), the hand's size (n,), the landmarks
    (n, 21, 3: MediaPipe's z in px; a frame without them holds the last ones seen) or None, and how much of the
    element shows on each frame (n,) for where the hand is."""

    xy: np.ndarray
    u: np.ndarray
    s: np.ndarray
    lm3: np.ndarray | None
    presence: np.ndarray
    size: tuple[int, int]  # the plate's (w, h)

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
    return _Hand(xy, u, s, lm3, presence, (w, h))


def _smooth(v: np.ndarray, k: int = 2) -> np.ndarray:
    """Zero-phase smoothing over +-k frames (triangular weights) along axis 0."""
    wts = np.concatenate([np.arange(1, k + 2), np.arange(k, 0, -1)]).astype(np.float64)
    p = np.pad(v, [(k, k)] + [(0, 0)] * (v.ndim - 1), mode="edge")
    return sum(wt * p[i: i + len(v)] for i, wt in enumerate(wts)) / wts.sum()


def _fingers(hand: _Hand, tap: int) -> list[int]:
    """The fingertip landmarks the arcs leave from: the one the emitter was carried onto at the tap (the nearest
    within FINGER_NEAR hands of it), and the index or middle finger beside it when that one is out straight next to
    it (two fingers together). [] with no landmarks, or a tap off the fingertips: the emitter alone."""
    if hand.lm is None:
        return []
    k = int(np.clip(tap, 0, len(hand.xy) - 1))
    lm, s = hand.lm[k], hand.s[k]
    tips = (4, 8, 12, 16, 20)
    d = {f: float(np.linalg.norm(lm[f] - hand.xy[k])) for f in tips}
    first = min(d, key=d.get)
    if d[first] > FINGER_NEAR * s:
        return []
    pair = {8: 12, 12: 8}.get(first)
    if pair is not None and finger_reach(hand.lm3[k])[(pair - 8) // 4] > REACH_OUT and \
            np.linalg.norm(lm[pair] - lm[first]) < FINGER_PAIR * s:
        return [first, pair]
    return [first]


def _aims(hand: _Hand, timeline: Timeline, finger: int | None, frame: tuple[int, int]) -> np.ndarray:
    """(n, 2) unit: which way the arcs leave on each frame. The studio's aim carried with the hand (the track's
    direction turned so it points the aim on the tap frame, which it already does when cuesync.bending.anchored turned
    it); without one, the finger's own, from its middle joint to its tip, in 3D (cuesync.hand_track's: a finger
    pointing into the scene points at its vanishing point), smoothed over a few frames (the landmarks jitter). From the
    knuckle, a finger crooked as in a finger gun aimed 20 degrees off its last joint, which is what the eye follows.
    Where the finger points at the camera and is too short in the picture to read, the track's own direction."""
    n = len(hand.xy)
    if timeline.aim is not None or finger is None:
        u = hand.u
        if timeline.aim is not None:
            k = int(np.clip(timeline.tap_frame, 0, n - 1))
            now = math.degrees(math.atan2(u[k, 0], -u[k, 1]))
            u = np.stack([_rot(v, (timeline.aim - now + 180.0) % 360.0 - 180.0) for v in u])
        return _unit(u)
    d = _pointing(hand.lm3[:, finger], hand.lm3[:, finger - 2], frame)
    short = np.linalg.norm(hand.lm[:, finger] - hand.lm[:, finger - 2], axis=-1) / hand.s
    sure = np.clip((short - 0.1) / 0.15, 0, 1)[:, None]
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


def arc_frames(hand: _Hand, timeline: Timeline, seed: int = 0) -> tuple[list[list], list[list]]:
    """Per frame, the arcs from the fingertips: ([(points, width px, intensity)], [(x, y, sigma, amplitude)] the
    burning fingertips the arcs leave from). Each finger's streams are struck, re-struck and let go as the module
    docstring says; ``Timeline.camera`` carries a far end with the room on a handheld shot."""
    from .bending import camera_between

    n = timeline.n
    L = levels(timeline)
    crackle = np.zeros(n)
    for t in range(n):
        crackle[t] = max(L[t], crackle[t - 1] * CRACKLE_LEFT if t else 0.0)
    fingers = _fingers(hand, timeline.tap_frame)
    aims = _aims(hand, timeline, fingers[0] if fingers else None, hand.size)
    hits = dict(zip(timeline.hits, timeline.sizes))
    speed = np.hypot(*np.gradient(hand.xy, axis=0).T) if n > 1 else np.zeros(n)  # px a frame
    crawl_still = CRAWL_STILL * min(hand.size) / 1080.0
    rng = np.random.default_rng([seed & 0xFFFFFFFF, _STRIKES])
    sway = rng.uniform(0, 2 * np.pi, (max(1, len(fingers)), 2))
    paths: dict[tuple[int, int], _Path] = {}
    bolts, spots = [], []
    for t in range(n):
        s, lv, pres = float(hand.s[t]), float(L[t]), float(hand.presence[t])
        if pres <= 0 or (lv <= 0 and crackle[t] < CRACKLE_MIN):
            paths.clear()
            bolts.append([])
            spots.append([])
            continue
        kw = _width_k(s)
        roots = [hand.xy[t]] if not fingers else \
            [hand.xy[t] + (hand.lm[t, f] - hand.lm[t, fingers[0]]) for f in fingers]  # (the emitter rides the tip)
        streams = 0 if lv <= 0 else 1 + sum(lv > x for x in SIDE_ON)
        hit = hits.get(t)
        lit, out = 0.0, []
        mid = np.mean(roots, axis=0)
        for fi, root in enumerate(roots):
            heading, fresh = aims[t], False  # the main channel's, root to far end, and whether it is struck anew
            split = 0.0 if len(roots) < 2 else 0.5 * PAIR_SPLIT * (1 if (root - mid) @ _rot(aims[t], 90) > 0 else -1)
            for j in range(ARC_STREAMS):
                key = (fi, j)
                if j >= (streams if fi == 0 else min(streams, 1)):
                    paths.pop(key, None)
                    continue
                p = paths.get(key)
                if p is None or t - p.born >= p.life or hit is not None or (j and fresh):  # (its streamers with it)
                    prng = np.random.default_rng(int(rng.integers(1 << 62)))
                    fresh = fresh or j == 0
                    reach = s * (ARC_REACH[0] + (ARC_REACH[1] - ARC_REACH[0]) * lv) * prng.uniform(0.8, 1.05)
                    width = kw * (ARC_WIDTH[0] + (ARC_WIDTH[1] - ARC_WIDTH[0]) * lv)
                    inten = 1.0
                    if j == 0:  # the main channel, along the swaying aim
                        sec = 2 * math.pi * t / fx.FPS
                        wander = ARC_WANDER * (0.6 * math.sin(ARC_WANDER_HZ[0] * sec + sway[fi, 0])
                                               + 0.4 * math.sin(ARC_WANDER_HZ[1] * sec + sway[fi, 1]))
                        v = _rot(aims[t], split + wander + prng.normal(0, ARC_JITTER))
                    else:  # a streamer beside it, off its heading, the two on opposite sides
                        v = _rot(heading, (1 if j % 2 else -1) * prng.uniform(*SIDE_GAP))
                        reach, width, inten = reach * prng.uniform(*SIDE_REACH), width * SIDE_WIDTH, SIDE_INTEN
                    if hit is not None:
                        reach *= 1.0 + HIT_REACH * hit
                    tree = bolt_tree(root, v, reach, prng, width, ROUGH, FORK[0] + FORK[1] * lv, max(2.0, 0.008 * s),
                                     inten)
                    life = int(rng.choice(LIFE, p=LIFE_P))
                    p = paths[key] = _Path(t, min(life, SIDE_LIFE) if j else life, root.copy(), tree[0][0][-1].copy(),
                                           tree)
                    flash = 1.0  # a new path: its first, brightest stroke
                else:
                    on = rng.random() < RESTROKE[0] + RESTROKE[1] * lv
                    flash = float(rng.uniform(0.55, 0.9)) if on else float(rng.uniform(0.0, 0.1))
                end = p.b
                if timeline.camera is not None and t > p.born:  # the far end struck the room: it moves with it
                    end = (camera_between(timeline.camera, p.born, t) @ np.r_[p.b, 1.0])[:2]
                if j == 0:
                    heading = _unit(end - root)
                gain = flash * (0.55 + 0.45 * lv) * pres
                if gain > 0.02:
                    out += [(q, wd, it * gain) for q, wd, it in _carry(p.tree, p.a, p.b, root, end)]
                lit = max(lit, gain)
        # the crackle at the fingertips: small arcs out of the tip and crawling up the finger (while it is steady
        # enough not to smear them into a band), struck every frame (smeared with the finger when drawn: layer)
        c = float(crackle[t])
        crng = np.random.default_rng([seed & 0xFFFFFFFF, _CRACKLE, t])
        sp = []
        for fi, root in enumerate(roots):
            back = hand.lm[t, fingers[fi] - 2] if fingers else root - aims[t] * 0.5 * s  # the finger's middle joint
            for _ in range(1 + int(round(c))):
                tip = root + _rot(aims[t], crng.uniform(-110, 110)) * s * crng.uniform(0.08, 0.28)
                out += _carry(_between(crng, CRACKLE_WIDTH * kw, CRACKLE_INTEN * (0.4 + 0.6 * c) *
                                       crng.uniform(0.6, 1.0) * pres, 0.35, 0.04),
                              np.zeros(2), np.array([1.0, 0.0]), root, tip)
            if crng.random() < 0.5 + 0.4 * c and speed[t] <= crawl_still:
                a = back + crng.uniform(-0.12, 0.12, 2) * s
                out += _carry(_between(crng, 0.9 * kw, 0.4 * (0.4 + lit) * pres, 0.35, 0.03), np.zeros(2),
                              np.array([1.0, 0.0]), a, root)
            if fi == 1 and crng.random() < 0.6:  # two fingers: arcs jump between their tips
                out += _carry(_between(crng, kw, 0.55 * (0.4 + lit) * pres, 0.4, 0.03), np.zeros(2),
                              np.array([1.0, 0.0]), roots[0], root)
            sp.append((root[0], root[1], CORONA * s, max(CORONA_MIN * c, 0.15 * c + 0.4 * lit * (0.5 + lv)) * pres))
        bolts.append(out)
        spots.append(sp)
    return bolts, spots


def knotted(pts: np.ndarray) -> bool:
    """Whether a channel (2D or 3D points: as seen, their first two coordinates) is knotted: cut into 12 pieces of
    equal length as seen, it turns more than KNOT_TURN degrees one way over any 5 of them, or its ends are less than
    KNOT_STRAIGHT of the 12 pieces' length apart (its finest kinks are not counted)."""
    p2 = pts[:, :2]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p2, axis=0), axis=1))])
    if cum[-1] < 1e-9:
        return False  # (seen end on: not drawn, BALL_FLAT)
    at = np.linspace(0.0, cum[-1], 13)
    q = np.stack([np.interp(at, cum, p2[:, 0]), np.interp(at, cum, p2[:, 1])], 1)
    piece = np.diff(q, axis=0)
    if np.linalg.norm(q[-1] - q[0]) < KNOT_STRAIGHT * np.linalg.norm(piece, axis=1).sum():
        return True
    a = np.arctan2(piece[:, 1], piece[:, 0])
    turn = np.degrees((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)
    return bool(np.abs(np.convolve(turn, np.ones(5), "valid")).max() > KNOT_TURN)


def _filament(rng: np.random.Generator) -> list:
    """A ball's filament in the unit ball (3D, as seen: z towards the camera): from near its middle out to its skin,
    mostly across the picture (BALL_DEPTH), forking there; each drawn again (a few times at most) while knotted."""
    d = _unit(rng.normal(0, 1, 3) * (1.0, 1.0, BALL_DEPTH))
    b = d * rng.uniform(0.92, 1.05)
    a = rng.normal(0, 0.05, 3)

    def untangled(start, end) -> np.ndarray:
        for _ in range(6):
            pts = displace(start, end, rng, BALL_ROUGH, 0.02)
            if not knotted(pts):
                break
        return pts

    tree = [(untangled(a, b), 1.0, 1.0)]
    for _ in range(rng.integers(1, 3) if rng.random() < 0.7 else 0):
        tree.append((untangled(b, _unit(d + rng.normal(0, 0.35, 3))), 0.6, 0.5))
    return tree


def _skin(rng: np.random.Generator) -> list:
    """An arc crawling over a ball's skin (3D, the unit ball, as seen: z towards the camera): a jagged, broken stretch
    of great circle, 3-5 of its 8 pieces, a quarter of them along its outline, which is what draws its edge."""
    a3, b3 = rng.normal(0, 1, (2, 3)) * ((1.0, 1.0, 0.25) if rng.random() < SKIN_OUTLINE else 1.0)
    a3, b3 = _unit(a3), _unit(b3)
    arc = _unit(np.stack([a3 * (1 - x) + b3 * x for x in np.linspace(0, 1, 9)])) * 1.02
    k = int(rng.integers(3, 6))
    j0 = int(rng.integers(0, 9 - k))
    return [(np.concatenate([displace(arc[j], arc[j + 1], rng, SKIN_ROUGH, 0.02)[:-1] for j in range(j0, j0 + k)]
                            + [arc[j0 + k: j0 + k + 1]]), 0.9, 0.55)]


def _palm_lift(hand: _Hand) -> np.ndarray:
    """(n, 2) per hand size: which way, and how far, the palm's normal leans across the picture (the normal of the
    wrist and the index and little knuckles in MediaPipe's depth, the side facing the camera, as it projects): a ball
    held over a palm seen from above sits further up the picture than the palm. Nothing for a palm seen square on, or
    with no landmarks."""
    if hand.lm3 is None:
        return np.zeros((len(hand.xy), 2))
    lm = hand.lm3
    nrm = _unit(np.cross(lm[:, 5] - lm[:, 0], lm[:, 17] - lm[:, 0]))
    nrm *= np.where(nrm[:, 2:] > 0, -1.0, 1.0)  # towards the camera (MediaPipe's depth grows away from it)
    return _smooth(nrm[:, :2])


def ball_frames(hand: _Hand, timeline: Timeline, seed: int = 0) -> tuple[list[list], list[list], list[tuple | None]]:
    """Per frame, the ball over the palm: ([(points, width px, intensity)], [(x, y, sigma, amplitude)] sparks,
    (x, y, radius, level, charge, flicker) or None: its middle, BALL_LIFT off the palm; the voice's level now, and the
    charge it has built up, which grows and shrinks smoothly and sizes the ball and its body). Filaments from its
    middle out to its skin (in 3D, turning slowly, projected: those at the back dimmer, brightest where they meet the
    skin, with a spark there), arcs crawling over its skin, and arcs jumping from it to a fingertip or knuckle (it is
    held, not pasted on). Each is struck and re-struck for 1-3 frames as a channel is, in the ball's own frame, so it
    rides the palm. It shows and goes on one gate (BALL_ON), fading in over BALL_FADE frames."""
    n = timeline.n
    L = levels(timeline)
    charge = _follow(L, BALL_GROW, BALL_SHRINK)  # it grows while the voice holds, and shrinks when it stops
    lift = _palm_lift(hand)
    hits = dict(zip(timeline.hits, timeline.sizes))
    rng = np.random.default_rng([seed & 0xFFFFFFFF, _STRIKES])
    paths: dict[tuple[str, int], _Path] = {}
    pulse = 0.0
    shown = 0  # frames it has shown for
    bolts, sparks, balls = [], [], []

    def strike(key: tuple[str, int], make, t: int, lv: float, hit: float | None) -> tuple[_Path, float]:
        """The path in ``key``'s slot on frame t, struck anew when its life is over (or on a hit), and how bright."""
        p = paths.get(key)
        if p is None or t - p.born >= p.life or hit is not None:
            prng = np.random.default_rng(int(rng.integers(1 << 62)))
            to = int(prng.integers(len(TARGETS))) if prng.random() < GROUND_P[0] + GROUND_P[1] * lv else -1
            p = paths[key] = _Path(t, int(rng.choice(LIFE, p=LIFE_P)), np.zeros(2), np.array([1.0, 0.0]),
                                   make(prng), to)
            return p, 1.0
        on = rng.random() < RESTROKE[0] + RESTROKE[1] * lv
        return p, float(rng.uniform(0.55, 0.9)) if on else float(rng.uniform(0.0, 0.1))

    for t in range(n):
        s, lv, pres = float(hand.s[t]), max(float(L[t]), float(charge[t])), float(hand.presence[t])
        hit = hits.get(t)
        pulse = BALL_PULSE * hit if hit is not None else pulse * 0.6
        if pres <= 0 or max(float(L[max(0, t - BALL_BRIDGE + 1): t + 1].max()), lv) < BALL_ON[1 if shown else 0]:
            shown = 0
            paths.clear()
            bolts.append([])
            sparks.append([])
            balls.append(None)
            continue
        shown += 1
        pres *= min(1.0, shown / BALL_FADE)
        kw = _width_k(s)
        c = hand.xy[t] + lift[t] * BALL_LIFT * s
        R = s * (BALL_R[0] + (BALL_R[1] - BALL_R[0]) * charge[t]) * (1.0 + pulse)
        ang, tilt = BALL_TURN * t, 0.35
        turn = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]]) @ \
            np.array([[math.cos(ang), 0, math.sin(ang)], [0, 1, 0], [-math.sin(ang), 0, math.cos(ang)]])
        out, sp = [], []
        K = int(round(BALL_FILAMENTS[0] + (BALL_FILAMENTS[1] - BALL_FILAMENTS[0]) * charge[t]))

        def filament(prng: np.random.Generator) -> list:
            """A filament struck as seen now, kept in the ball's own frame (it turns with it)."""
            return [(q @ turn, w_, it) for q, w_, it in _filament(prng)]

        for key, count, make, wd in (("fil", K, filament, 0.8 + 0.6 * lv), ("skin", BALL_SKIN, _skin, 1.0)):
            for i in range(count):
                p, flash = strike((key, i), make, t, lv, hit)
                gain = flash * (0.4 + 0.6 * lv) * pres
                if gain <= 0.02 or key == "fil" and np.hypot(*(p.tree[0][0][-1] @ turn.T)[:2]) < BALL_FLAT:
                    continue
                for q, w_, it in p.tree:
                    q3 = q @ turn.T if key == "fil" else q  # (an arc on its skin lives too briefly to turn)
                    front = 0.55 + 0.45 * float(np.clip(q3[-1, 2], -1, 1))  # those at the back dimmer
                    q2 = c + R * q3[:, :2]
                    if key == "fil" and w_ == 1.0:  # dim inside (the middle is the spark's), brightest at the skin:
                        k_ = int(0.45 * len(q2))  # a ball, not a white star of glows piled up in the middle
                        out += [(q2[: k_ + 1], wd * kw * 0.7, 0.4 * gain * front), (q2[k_:], wd * kw, gain * front)]
                        if q3[-1, 2] > 0.2:
                            sp.append((q2[-1, 0], q2[-1, 1], CONTACT * s, 0.3 * gain * front))
                    else:
                        out.append((q2, w_ * wd * kw, it * gain * front))
        for i in range(BALL_GROUND if hand.lm is not None else 0):  # grounded on the hand
            p, flash = strike(("ground", i), lambda prng: _between(prng, 1.2, 0.75, 0.2, 0.01, 0.1), t, lv, hit)
            tgt = None if p.to < 0 else hand.lm[t, TARGETS[p.to]]
            if tgt is None or flash <= 0.05 or np.linalg.norm(tgt - c) < 1.2 * R:  # (not to a knuckle under the ball)
                continue
            out += [(q, w_ * kw * (0.7 + 0.5 * lv), it * flash * pres) for q, w_, it in
                    _carry(p.tree, p.a, p.b, c + _unit(tgt - c) * R, tgt)]
        bolts.append(out)
        sparks.append(sp)
        flicker = float(np.random.default_rng([seed & 0xFFFFFFFF, _SPARK, t]).uniform(0.7, 1.0))
        balls.append((float(c[0]), float(c[1]), float(R), lv * pres, float(charge[t]) * pres, flicker))
    return bolts, sparks, balls


# ---- drawing -------------------------------------------------------------------------------------------------------

def _box(bolts: list, spots: list, s: float, h: int, w: int) -> tuple[int, int, int, int] | None:
    """(y0, y1, x0, x1): where the channels and spots are, with room for their widest glow."""
    xs, ys = [], []
    for p, _, it in bolts:
        if it > 0 and len(p) >= 2:
            xs += [p[:, 0].min(), p[:, 0].max()]
            ys += [p[:, 1].min(), p[:, 1].max()]
    for x, y, r, a in spots:
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
    at sub-pixel precision, thinning towards its end, in brighter and dimmer stretches; where they cross, the brighter
    shows. (bh, bw, 2): the core, and what the glow round it is drawn from, the same lines with a hot channel (GLOW_HOT)
    as if at least ``floor`` px wide (GLOW_FLOOR)."""
    y0, y1, x0, x1 = box
    core = np.zeros((y1 - y0, x1 - x0, 2), np.float32)
    tmp = np.zeros(core.shape[:2], np.uint8)
    rng = np.random.default_rng([seed & 0xFFFFFFFF, _FRAME, frame])
    for pts, width, inten in bolts:
        if len(pts) < 2 or inten <= 0:
            continue
        p = pts[:, :2] - (x0, y0)
        a0 = np.floor(p.min(0)).astype(int) - int(width) - 3
        a1 = np.ceil(p.max(0)).astype(int) + int(width) + 3
        bx0, by0 = max(0, a0[0]), max(0, a0[1])
        bx1, by1 = min(core.shape[1], a1[0]), min(core.shape[0], a1[1])
        if bx1 <= bx0 or by1 <= by0:
            continue
        sub = tmp[by0:by1, bx0:bx1]
        sub[:] = 0
        chunks = 8
        bead = rng.uniform(0.6, 1.0, chunks)
        for c in range(chunks):
            a, b = c * (len(p) - 1) // chunks, (c + 1) * (len(p) - 1) // chunks + 1
            if b - a < 2:
                continue
            wd = width * (1.0 - 0.6 * c / chunks)
            th = max(1, int(round(wd)))
            lv = int(np.clip(255 * bead[c] * min(1.0, wd / th), 30, 255))
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


def _limb(theta: np.ndarray, seed: int, frame: int) -> np.ndarray:
    """How much of the ball's limb is lit at each angle ``theta`` (radians) on ``frame``, in [0, 1]: LIMB_WAVES
    drifting round it, dark where their sum is under LIMB_CUT (moved a little each frame, so pieces come and go)."""
    rng = np.random.default_rng([seed & 0xFFFFFFFF, _LIMB])
    phase, spin = rng.uniform(0, 2 * np.pi, len(LIMB_WAVES)), rng.uniform(-LIMB_SPIN, LIMB_SPIN, len(LIMB_WAVES))
    wave = sum(np.sin(k * theta + np.float32(p + sp * frame)) * np.float32(1.0 / k ** 0.5)
               for k, p, sp in zip(LIMB_WAVES, phase, spin)) / np.float32(sum(k ** -0.5 for k in LIMB_WAVES))
    cut = LIMB_CUT[0] + LIMB_CUT[1] * np.random.default_rng([seed & 0xFFFFFFFF, _LIMB, frame]).uniform(-1, 1)
    return np.clip((wave - np.float32(cut)) / np.float32(LIMB_FADE), 0.0, 1.0) ** 2


def _plasma(u: np.ndarray, v: np.ndarray, seed: int, frame: int) -> np.ndarray:
    """In [0, 1] at (u, v) (radii from the ball's middle) on ``frame``: bright along the ridges where PLASMA_WAVES
    plane waves of about 1.5-3 cycles across the ball cross, each drifting its own way."""
    rng = np.random.default_rng([seed & 0xFFFFFFFF, _PLASMA])
    turn = rng.uniform(0, np.pi, PLASMA_WAVES)
    k = rng.uniform(1.5, 3.0, PLASMA_WAVES) * np.pi  # (radians a radius: 1.5-3 cycles across the ball)
    phase, drift = rng.uniform(0, 2 * np.pi, PLASMA_WAVES), rng.uniform(0.1, 0.3, PLASMA_WAVES) * rng.choice([-1, 1],
                                                                                                           PLASMA_WAVES)
    wave = sum(np.sin(np.float32(kk * math.cos(a)) * u + np.float32(kk * math.sin(a)) * v + np.float32(p + dr * frame))
               for kk, a, p, dr in zip(k, turn, phase, drift)) / np.float32(math.sqrt(PLASMA_WAVES))
    return (1.0 - np.minimum(np.abs(wave), 1.0)) ** 3


def _sphere(glow: np.ndarray, out: np.ndarray, veil: np.ndarray, box: tuple, ball: tuple, edge: float, seed: int,
            frame: int) -> None:
    """Adds the ball's round body (over ``box``), in place: inside its outline (soft over ``edge`` px), BODY_RGB air
    glowing as much of it as the line of sight goes through (BALL_THICK), brighter at its middle (BODY_FALL,
    BODY_INNER) and along drifting ridges (_plasma), to ``glow``; a ring glowing at its limb, broken (_limb), to ``out``; and the palm
    behind it hidden by BALL_VEIL as much as there is of it, to ``veil``."""
    x, y, R, lv, full, _ = ball
    y0, _, x0, _ = box
    r = int(R + 2 * edge) + 2
    a0, a1 = max(0, int(x - x0) - r), min(veil.shape[1], int(x - x0) + r)
    b0, b1 = max(0, int(y - y0) - r), min(veil.shape[0], int(y - y0) + r)
    if a1 <= a0 or b1 <= b0:
        return
    dx = np.arange(a0, a1, dtype=np.float32) - np.float32(x - x0)
    dy = np.arange(b0, b1, dtype=np.float32)[:, None] - np.float32(y - y0)
    dist = np.hypot(dx, dy)
    d2 = (dist / np.float32(R)) ** 2
    inside = np.clip((np.float32(R) - dist) / np.float32(edge) + 0.5, 0.0, 1.0)
    depth = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0))  # (the sphere's, along the line of sight, 1 through its middle)
    deep = (1.0 - np.exp(np.float32(-BALL_THICK) * depth)) * inside
    deep *= np.float32(1.0 / (1.0 - math.exp(-BALL_THICK)))  # (how much of it there is along the line of sight)
    ring = np.exp(((dist / np.float32(R) - np.float32(LIMB_AT)) / np.float32(LIMB_W)) ** 2 * np.float32(-0.5))
    lit = _limb(np.arctan2(dy, dx) + np.float32(BALL_TURN * frame), seed, frame)
    formed = min(1.0, (0.3 * lv + 0.7 * full) / BALL_FORMED)
    body = deep * (np.float32(1.0 - BODY_FALL) + np.float32(BODY_FALL) * depth
                   + np.float32(BODY_INNER[0]) * np.exp(d2 * np.float32(-0.5 / BODY_INNER[1] ** 2))) * \
        (np.float32(1.0 - BODY_PLASMA) + np.float32(BODY_PLASMA) * _plasma(dx / np.float32(R), dy / np.float32(R),
                                                                          seed, frame))
    glow[b0:b1, a0:a1] += (np.float32(BALL_BODY * formed) * body)[..., None] * BODY_RGB
    out[b0:b1, a0:a1] += (np.float32(LIMB * BALL_SPARK * lv) * ring * lit)[..., None] * LIMB_RGB
    veil[b0:b1, a0:a1] = np.float32(BALL_VEIL * min(1.0, full / BALL_FORMED)) * deep


def layer(bolts: list, spots: list, ball: tuple | None, s: float, box: tuple, cap: float, soft: float,
          move: np.ndarray, around: list, seed: int = 0, frame: int = 0, side: int = 1080,
          floor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Over ``box``: linear RGB (bh, bw, 3), the channels' white-hot core and the blue-violet glow round them, the
    burning spots (white, a blue halo) and the ball's middle and round body (_sphere); and (bh, bw) how much of the
    plate behind the ball's body it hides (BALL_VEIL). The channels are smeared along ``move`` round ``around``
    (_smear) and blurred ``soft`` px (the plate's softness); only the brightest clip (CORE_GAMMA), and their glow, from
    the channels as drawn (a hot one as if ``floor`` px wide: draw), is held under ``cap`` (GLOW_CAP). ``side``: the
    picture's short side, which the px constants are given at 1080 of."""
    k = side / 1080.0
    lines = _smear(draw(bolts, box, seed, frame, floor), box, move, around, s, SMEAR_MAX * k)
    if soft > 0:
        lines = cv2.GaussianBlur(lines, (0, 0), soft)
    core, hot = np.maximum(lines[..., 0], 0), np.maximum(lines[..., 1], 0)  # (a wide smear is filtered by FFT, a hair
    # under 0 here and there)
    out = (core ** np.float32(CORE_GAMMA))[..., None] * (CORE_RGB * np.float32(CORE_GAIN))
    glow = np.zeros_like(out)
    for sig_h, g, rgb in GLOWS:  # (a blur spreads a line's energy: scaled so its peak over a 1 px line is g)
        sig = sig_h * s
        glow += _blur(hot, sig)[..., None] * (np.float32(g * sig * math.sqrt(2 * math.pi)) * np.float32(rgb))
    spark = np.zeros(core.shape, np.float32)
    for x, y, r, a in spots:
        _spot(spark, box, x, y, r, a)
    veil = np.zeros(core.shape, np.float32)
    if ball is not None:
        x, y, R, lv, _, flick = ball
        _spot(spark, box, x, y, BALL_CORE * R, BALL_SPARK * lv * flick)
        _sphere(glow, out, veil, box, ball, BALL_EDGE * k, seed, frame)
    if spark.any():
        out += spark[..., None] * SPARK_RGB
        glow += (0.6 * _blur(spark, 0.12 * s))[..., None] * HAZE_RGB
    y = fx._lum(glow)
    glow *= (cap * np.tanh(y / cap) / np.maximum(y, 1e-6))[..., None]
    return out + glow, veil


# ---- the light -----------------------------------------------------------------------------------------------------

def light_power(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n,) the strength of the light the channels throw on the hand and the room: their own power (luminance summed)
    held at its peak over LIGHT_HOLD frames and let go by LIGHT_LETGO a frame once they stop, gone under LIGHT_GONE of
    its burst's peak, with LIGHT_FLICKER of the strokes' own flicker kept; and (n,) that peak, the strongest it has
    been since it was last gone. So it comes up on the first stroke and stays while they strike, instead of strobing
    with them, and dies within a few frames of the last."""
    out, top = np.zeros_like(power), np.zeros_like(power)
    held = peak = 0.0
    for t in range(len(power)):
        now = float(power[max(0, t - LIGHT_HOLD + 1): t + 1].max())
        held = max(now, held * LIGHT_LETGO)
        if now <= 0 and held <= LIGHT_GONE * peak:
            held = 0.0
        peak = max(peak, held) if held > 0 else 0.0
        out[t] = held * (1.0 - LIGHT_FLICKER * (1.0 - power[t] / held)) if held > 0 else 0.0
        top[t] = peak
    return out, top


def _soft_cap(x: np.ndarray, cap: float) -> np.ndarray:
    return np.float32(cap) * (1.0 - np.exp(-x / np.float32(cap)))


def _around(x: float, y: float, r: float, h: int, w: int) -> tuple[int, int, int, int] | None:
    """(y0, y1, x0, x1): the frame within ``r`` px of (x, y), or None off it."""
    y0, y1, x0, x1 = max(0, int(y - r)), min(h, int(y + r) + 1), max(0, int(x - r)), min(w, int(x + r) + 1)
    return None if y1 <= y0 or x1 <= x0 else (y0, y1, x0, x1)


def wrap_light(matte: np.ndarray, ball: tuple, side: int) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """The ball's light on the edges of the hand that face it: (box, (bh, bw) light over it, at WRAP_RGB) within
    WRAP_REACH radii of its middle, inside the matte (0-1 float) within about 2 WRAP_PX of its outline where the
    outline faces the ball, falling off as the inverse square of the distance; or None."""
    x, y, R, _, full, _ = ball
    box = _around(x, y, WRAP_REACH * R, *matte.shape)
    if box is None or full <= 0:
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
    return box, np.float32(WRAP_GAIN * full) * band * facing * np.minimum(1.0, (np.float32(R) / dist) ** 2)


def glare(frame: np.ndarray, ball: tuple) -> None:
    """The lens's glare round the ball, screened over ``frame`` (uint8 RGB, as shown) in place: GLARE's halo, bloom and
    veil in GLARE_RGB, the same light over whatever is behind it."""
    x, y, R, lv, full, flick = ball
    (g_halo, s_halo), (g_bloom, s_bloom), (g_veil, s_veil) = GLARE
    box = _around(x, y, 3.0 * max(s_halo, s_veil) * R, *frame.shape[:2])
    if box is None or lv <= 0:
        return
    y0, y1, x0, x1 = box
    d2 = ((np.arange(x0, x1, dtype=np.float32) - np.float32(x)) ** 2
          + (np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(y)) ** 2) / np.float32(R * R)
    g = np.float32(g_halo * lv * flick) * np.exp(d2 * np.float32(-0.5 / s_halo ** 2)) + \
        np.float32(g_veil * full) * np.exp(d2 * np.float32(-0.5 / s_veil ** 2)) + \
        np.float32(g_bloom * min(1.0, full / BALL_FORMED)) / (1.0 + np.exp((np.sqrt(d2) - 1.0) / np.float32(s_bloom)))
    shown = frame[y0:y1, x0:x1].astype(np.float32) * np.float32(1 / 255)
    shown = 1.0 - (1.0 - shown) * (1.0 - g[..., None] * GLARE_RGB)
    frame[y0:y1, x0:x1] = (shown * 255.0 + 0.5).astype(np.uint8)


# ---- the shot ------------------------------------------------------------------------------------------------------

def render(plate: np.ndarray, track: dict, mask: np.ndarray | None, timeline: Timeline, form: str, seed: int = 0,
           progress=None) -> np.ndarray:
    """The shot with lightning from the hand: uint8 RGB [T, H, W, 3] (cuesync.elements' contract). ``form``: "arc"
    (bolts from the fingertip the emitter was carried onto, or two fingers together, along the aim) or "ball" (a
    crackling ball over the palm where the emitter is). ``progress(stage, frac, msg)``, as cuesync.bending's, hears
    how far it is. Frames with no lightning, no light on them and the camera's exposure as it was are the plate's own
    bytes."""
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
        bolts, spots = arc_frames(hand, timeline, seed)
        balls = [None] * n
        bright = levels(timeline) * hand.presence
    else:
        bolts, spots, balls = ball_frames(hand, timeline, seed)
        bright = np.array([0.0 if b is None else b[3] for b in balls])
    stops, held, rise, fall = EXPOSURE[form]
    bright = np.array([bright[max(0, t - held + 1): t + 1].max() for t in range(n)])  # (a stuttering voice, or the
    # gaps in a phrase, don't pump the exposure)
    dip = stops * _follow(bright, rise, fall)  # the camera exposing down for the light, in stops
    depth, hand_gain, room_gain, near_gain = LIGHT[form]
    hand_rgb, room_rgb = LIGHT_RGB[form]
    two_colours = not np.array_equal(hand_rgb, room_rgb)
    side = min(h, w)
    floor = GLOW_FLOOR * ARC_WIDTH[1] if form == "arc" else 0.0  # (times the hand's width scale)
    soft_px = (DEFOCUS[0] * side / 1080, DEFOCUS[1] * side / 1080)
    move = np.gradient(hand.xy, axis=0) * SHUTTER if n > 1 else np.zeros_like(hand.xy)  # the hand, shutter open
    some = np.stack([lin(plate[i][::2, ::2]) for i in np.linspace(0, n - 1, 8).astype(int)])
    white, tint = fx.plate_white(some), fx.plate_tint(some)
    del some
    noise = fx.shadow_noise(plate)
    hf, wf = max(1, h // FIELD_Q), max(1, w // FIELD_Q)
    out = np.empty_like(plate)

    def made(t: int):
        """Frame t's lightning: (box, linear RGB layer over it, its power in plate px, what it hides of the plate), or
        None."""
        ball = balls[t]
        box = _box(bolts[t], spots[t], hand.s[t], h, w)
        if ball is not None:
            x, y, R = ball[:3]
            bb = _box([], [(x, y, R / 3.0, 1.0)], hand.s[t], h, w)  # (its body, out to R, and its glow)
            box = bb if box is None else \
                (min(box[0], bb[0]), max(box[1], bb[1]), min(box[2], bb[2]), max(box[3], bb[3]))
        if box is None:
            return None
        s = float(hand.s[t])
        soft = float(np.clip(softness(plate[t], hand.xy[t], s), *soft_px))
        tips = [(x, y) for x, y, _, amp in spots[t] if amp > 0] if form == "arc" else []  # (a ball's strokes are
        # instant: only the crackle at a fingertip glows on while the shutter is open)
        E, veil = layer(bolts[t], spots[t], ball, s, box, GLOW_CAP[form], soft, move[t], tips, seed, t, side,
                        floor * _width_k(s))
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
            if layers[t] is not None and layers[t][2] > 0:
                (y0, y1, x0, x1), E, power[t], _ = layers[t]
                if form == "arc":  # (two fingers together light the room as one does: one burning tip each)
                    power[t] /= max(1, len(spots[t]))
                small = np.zeros((hf, wf), np.float32)
                sy0, sx0 = y0 // FIELD_Q, x0 // FIELD_Q
                sy1, sx1 = max(sy0 + 1, min(hf, y1 // FIELD_Q)), max(sx0 + 1, min(wf, x1 // FIELD_Q))
                small[sy0:sy1, sx0:sx1] = cv2.resize(fx._lum(E), (sx1 - sx0, sy1 - sy0), interpolation=cv2.INTER_AREA)
                small /= max(float(small.sum(dtype=np.float64)), 1e-12)
                last = small
            smalls.append(small)
        strength, peak = light_power(power[:b])  # (causal: frames already made don't change)
        for t in range(a, b):
            pool = [(wt, smalls[t - k]) for k, wt in enumerate(LIGHT_POOL) if t - k >= 0 and smalls[t - k] is not None]
            if strength[t] <= 0:
                fields[t] = None
                continue
            src = sum(wt * sm for wt, sm in pool) / sum(wt for wt, _ in pool) if pool else last
            if src is None:
                fields[t] = None
                continue
            s_f = hand.s[t] / FIELD_Q
            near, wall, far = fx.spread(src * np.float32(1 / (FIELD_Q * FIELD_Q)),  # (per small px: the mean of the
                                        [max(depth * s_f, 1.0), ROOM_NEAR * s_f, fx.FAR * s_f])  # px it covers)
            # the hand's light at the burst's peak (where HAND_CAP holds it), scaled down with the burst after the cap,
            # so the cap does not hold it up as the burst dies; the room's as strong as the light is now
            fields[t] = (near * np.float32(hand_gain * peak[t]),
                         (far * np.float32(room_gain) + wall * np.float32(near_gain)) * np.float32(strength[t]), src,
                         np.float32(strength[t] / peak[t]))

        def finish(i: int) -> None:
            t = a + i
            if layers[t] is None and fields[t] is None and dip[t] < DIP_MIN:
                out[t] = plate[t]
                return
            P0 = lin(plate[t])
            P = P0.copy()
            if fields[t] is not None:
                near, light, src, k = fields[t]
                light = cv2.resize(light, (w, h), interpolation=cv2.INTER_LINEAR)  # (the room's)
                m = None if mask is None else mask[t]
                if m is not None and not m[::4, ::4].any():
                    m = None
                if m is None and mask is None:
                    m = fx.arm_matte(tuple(hand.xy[t]), tuple(hand.u[t]), float(hand.s[t]), w, h)
                elif m is not None:
                    m = m.astype(np.float32) * np.float32(1 / 255)
                    m = np.minimum(m, cv2.GaussianBlur(m, (0, 0), max(0.5, fx.MATTE_SOFT * side)))
                alb = fx.albedo(P0, white, tint)
                lit = light[..., None] * room_rgb
                if m is not None:
                    on_hand = cv2.resize(near, (w, h), interpolation=cv2.INTER_LINEAR) * \
                        fx.hand_shading(m, src, float(hand.s[t]))
                    if balls[t] is not None:  # the pool the ball throws on the palm right under it
                        R, full = balls[t][2], balls[t][4]
                        _spot(on_hand, (0, h, 0, w), hand.xy[t, 0], hand.xy[t, 1], POOL_R * R, POOL_GAIN * full)
                    on_hand = m * (_soft_cap(on_hand, HAND_CAP) * k)
                    if two_colours:  # (the room's light, where it falls on the hand, in the hand's colour)
                        on_hand += m * light
                        lit *= (1.0 - m)[..., None]
                    lit += on_hand[..., None] * hand_rgb
                    if balls[t] is not None:
                        wrap = wrap_light(m, balls[t], side)
                        if wrap is not None:
                            (y0, y1, x0, x1), wl = wrap
                            lit[y0:y1, x0:x1] += wl[..., None] * WRAP_RGB
                P += alb * lit
            if dip[t] >= DIP_MIN:  # (the lightning itself is drawn as the camera shows it, exposed down)
                e = np.float32(2.0 ** -dip[t])
                P *= e
                P0 *= e
            if layers[t] is not None:
                (y0, y1, x0, x1), E, _, veil = layers[t]
                if veil.any():
                    P[y0:y1, x0:x1] *= 1.0 - veil[..., None]
                    P0[y0:y1, x0:x1] *= 1.0 - veil[..., None]
                P[y0:y1, x0:x1] += fx.grain(E[None], noise, seed=seed, frame0=t)[0]
            shoulder(P)  # (the channels' cores clip to white softly)
            np.maximum(P, P0, out=P)  # (but the plate's own highlights stay as bright as they were)
            out[t] = gam(P)
            if balls[t] is not None:  # (the lens's, over the picture as shown)
                glare(out[t], balls[t])

        fx._each(b - a, finish)
        for t in range(a, b):
            layers.pop(t, None)
            fields.pop(t, None)
        say("lightning", b / n, f"Striking the lightning: frame {b} of {n}")
    return out
